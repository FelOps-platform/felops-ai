"""Unit tests for `adapters/llm/litellm.py`. No network: `litellm.acompletion` is patched."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from typing import Any

import litellm as vendor  # the vendor SDK (our module is felops.adapters.llm.litellm)
import pytest

from felops.adapters.llm import LiteLLMProvider
from felops.adapters.llm import litellm as mod
from felops.ports.llm import (
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMToolSpec,
    ProviderCaps,
)

CAPS = ProviderCaps(tools=True, json_mode=True, max_ctx=131_072)


def _provider(**kw: Any) -> LiteLLMProvider:
    kw.setdefault("capabilities", CAPS)
    return LiteLLMProvider(provider="groq", default_model="llama-3.3-70b-versatile", **kw)


def _req(**kw: Any) -> LLMRequest:
    return LLMRequest(messages=(LLMMessage("system", "be brief"), LLMMessage("user", "hi")), **kw)


def _resp(content: Any = "hello", *, finish: Any = "stop", tool_calls: Any = None, **kw: Any) -> NS:
    return NS(
        model=kw.get("model", "llama-3.3-70b-versatile"),
        choices=[NS(finish_reason=finish, message=NS(content=content, tool_calls=tool_calls))],
        usage=kw.get("usage", NS(prompt_tokens=7, completion_tokens=3)),
    )


def _tc(id: str = "c1", name: str = "lookup_order", args: Any = '{"order_id": "1042"}') -> NS:
    return NS(id=id, function=NS(name=name, arguments=args))


def _patch(monkeypatch: pytest.MonkeyPatch, result: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> Any:
        seen.update(kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mod.litellm, "acompletion", fake_acompletion)
    return seen


def _complete(p: LiteLLMProvider, req: LLMRequest) -> Any:
    return asyncio.run(p.complete(req))


# -- construction ---------------------------------------------------------


def test_satisfies_port() -> None:
    assert isinstance(_provider(), LLMProvider)


@pytest.mark.parametrize(
    "kw",
    [
        {"provider": " "},
        {"default_model": "  "},
        {"capabilities": ProviderCaps(tools=True, json_mode=True, max_ctx=0)},
        {"timeout": 0},
    ],
)
def test_invalid_construction_is_rejected(kw: dict[str, Any]) -> None:
    base: dict[str, Any] = {"provider": "groq", "capabilities": CAPS}
    with pytest.raises(ValueError):
        LiteLLMProvider(**{**base, **kw})


# -- request mapping ------------------------------------------------------


def test_complete_maps_request_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _resp("hello"))
    tool = LLMToolSpec("lookup_order", "Look up an order", {"type": "object"})
    out = _complete(
        _provider(api_key="k", api_base="http://x", timeout=12.5),
        _req(
            temperature=0.0,
            max_tokens=50,
            json_mode=True,
            tools=(tool,),
            stop=("END",),
            metadata={"workspace_id": "w1"},
        ),
    )
    assert seen["model"] == "groq/llama-3.3-70b-versatile"
    assert seen["messages"][0] == {"role": "system", "content": "be brief"}
    assert seen["temperature"] == 0.0
    assert seen["timeout"] == 12.5
    assert seen["stream"] is False
    assert seen["num_retries"] == 0 and seen["max_retries"] == 0  # retry is the gateway's job
    assert seen["max_tokens"] == 50
    assert seen["response_format"] == {"type": "json_object"}
    assert seen["tools"][0] == {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up an order",
            "parameters": {"type": "object"},
        },
    }
    assert seen["stop"] == ["END"]
    assert seen["metadata"] == {"workspace_id": "w1"}
    assert seen["_skip_mcp_handler"] is True
    assert (seen["api_key"], seen["api_base"]) == ("k", "http://x")
    assert (out.text, out.finish_reason, out.usage.total_tokens) == ("hello", "stop", 10)
    assert out.model == "llama-3.3-70b-versatile"


def test_optional_fields_are_omitted_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _resp())
    req = _req()
    _complete(_provider(), req)
    assert seen["temperature"] == req.temperature
    assert seen["timeout"] == 60.0
    for key in (
        "max_tokens",
        "response_format",
        "tools",
        "stop",
        "metadata",
        "api_key",
        "api_base",
        "_skip_mcp_handler",
    ):
        assert key not in seen


def test_tool_role_message_keeps_tool_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _resp())
    req = LLMRequest(messages=(LLMMessage("tool", "shipped", tool_call_id="c1"),))
    _complete(_provider(), req)
    assert seen["messages"][0] == {"role": "tool", "content": "shipped", "tool_call_id": "c1"}


def test_request_model_overrides_default_without_double_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch(monkeypatch, _resp())
    _complete(_provider(), _req(model="llama-3.1-8b-instant"))
    assert seen["model"] == "groq/llama-3.1-8b-instant"
    _complete(_provider(), _req(model="groq/llama-3.1-8b-instant"))
    assert seen["model"] == "groq/llama-3.1-8b-instant"


def test_no_model_is_a_permanent_llm_error() -> None:
    p = LiteLLMProvider(provider="groq", capabilities=CAPS)
    with pytest.raises(LLMError) as ei:
        _complete(p, _req())
    assert ei.value.retryable is False


def test_capability_check_happens_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _resp())
    p = _provider(capabilities=ProviderCaps(tools=False, json_mode=False, max_ctx=1000))
    with pytest.raises(LLMError) as e1:
        _complete(p, _req(tools=(LLMToolSpec("t", "d", {}),)))
    with pytest.raises(LLMError):
        _complete(p, _req(json_mode=True))
    assert e1.value.retryable is False
    assert seen == {}


# -- response parsing -----------------------------------------------------


def test_tool_calls_are_parsed_never_executed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _resp(None, finish="tool_calls", tool_calls=[_tc()]))
    out = _complete(_provider(), _req())
    assert out.text == "" and out.finish_reason == "tool_calls"
    assert (out.tool_calls[0].id, out.tool_calls[0].name) == ("c1", "lookup_order")
    assert out.tool_calls[0].arguments == {"order_id": "1042"}


def test_tool_arguments_are_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _resp(None, finish="tool_calls", tool_calls=[_tc()]))
    args = _complete(_provider(), _req()).tool_calls[0].arguments
    with pytest.raises(TypeError):
        args["order_id"] = "x"  # type: ignore[index]


@pytest.mark.parametrize("raw_args", ["", None])
def test_empty_tool_arguments_become_empty_mapping(
    monkeypatch: pytest.MonkeyPatch, raw_args: Any
) -> None:
    _patch(monkeypatch, _resp(None, tool_calls=[_tc(args=raw_args)]))
    assert _complete(_provider(), _req()).tool_calls[0].arguments == {}


def test_mapping_tool_arguments_become_read_only_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        _resp(None, finish="tool_calls", tool_calls=[_tc(args={"order_id": "1042"})]),
    )
    args = _complete(_provider(), _req()).tool_calls[0].arguments
    assert args == {"order_id": "1042"}
    with pytest.raises(TypeError):
        args["order_id"] = "x"  # type: ignore[index]


def test_nested_tool_arguments_are_deeply_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = '{"filters": {"status": "open"}, "items": [{"sku": "A"}]}'
    _patch(monkeypatch, _resp(None, finish="tool_calls", tool_calls=[_tc(args=raw)]))
    args = _complete(_provider(), _req()).tool_calls[0].arguments
    assert args["filters"]["status"] == "open"
    assert args["items"] == ({"sku": "A"},)
    with pytest.raises(TypeError):
        args["filters"]["status"] = "closed"  # type: ignore[index]
    with pytest.raises(TypeError):
        args["items"][0] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        args["items"][0]["sku"] = "B"  # type: ignore[index]


def test_mapping_tool_arguments_do_not_alias_caller_nested_containers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_args: dict[str, Any] = {"filters": {"status": "open"}, "items": [{"sku": "A"}]}
    _patch(monkeypatch, _resp(None, finish="tool_calls", tool_calls=[_tc(args=raw_args)]))
    call = _complete(_provider(), _req()).tool_calls[0]
    raw_args["filters"]["status"] = "closed"
    raw_args["items"][0]["sku"] = "B"
    raw_args["items"].append({"sku": "C"})
    assert call.arguments["filters"]["status"] == "open"
    assert call.arguments["items"] == ({"sku": "A"},)
    with pytest.raises(TypeError):
        call.arguments["filters"]["status"] = "closed"  # type: ignore[index]


def test_streamed_tool_arguments_are_deeply_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            _chunk(
                None, tool_calls=[_delta(id="c1", name="lookup", args='{"filters": {"status":')]
            ),
            _chunk(None, tool_calls=[_delta(args=' "open"}, "items": [{"sku": "A"}]}')]),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    args = _stream()[-1].tool_calls[0].arguments
    assert args["filters"]["status"] == "open"
    assert args["items"] == ({"sku": "A"},)
    with pytest.raises(TypeError):
        args["filters"]["status"] = "closed"  # type: ignore[index]
    with pytest.raises(TypeError):
        args["items"][0]["sku"] = "B"  # type: ignore[index]


@pytest.mark.parametrize("bad_args", ["{not json", "[1, 2]", '"text"'])
def test_malformed_tool_arguments_are_retryable(
    monkeypatch: pytest.MonkeyPatch, bad_args: str
) -> None:
    _patch(monkeypatch, _resp(None, tool_calls=[_tc(args=bad_args)]))
    with pytest.raises(LLMError) as ei:
        _complete(_provider(), _req())
    assert ei.value.retryable is True


@pytest.mark.parametrize("call", [_tc(id=""), _tc(name=""), NS(id="c1", function=None)])
def test_incomplete_tool_call_is_retryable(monkeypatch: pytest.MonkeyPatch, call: NS) -> None:
    _patch(monkeypatch, _resp(None, tool_calls=[call]))
    with pytest.raises(LLMError) as ei:
        _complete(_provider(), _req())
    assert ei.value.retryable is True


@pytest.mark.parametrize(
    ("raw", "has_calls", "expected"),
    [
        ("stop", False, "stop"),
        ("length", False, "length"),
        ("tool_calls", True, "tool_calls"),
        ("function_call", True, "tool_calls"),
        ("stop", True, "tool_calls"),  # some providers report stop with tool calls
        (None, True, "tool_calls"),
        ("length", True, "length"),  # never override a truncation
        ("content_filter", False, "error"),
        ("something_new", False, "error"),
        (123, False, "error"),
        (None, False, "stop"),  # complete(): a missing reason means a normal end
    ],
)
def test_finish_reason_normalisation(
    monkeypatch: pytest.MonkeyPatch, raw: Any, has_calls: bool, expected: str
) -> None:
    _patch(monkeypatch, _resp("x", finish=raw, tool_calls=[_tc()] if has_calls else None))
    assert _complete(_provider(), _req()).finish_reason == expected


def test_no_choices_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, NS(model="m", choices=[], usage=None))
    with pytest.raises(LLMError) as ei:
        _complete(_provider(), _req())
    assert ei.value.retryable is True


def test_non_text_content_is_a_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _resp(["part"]))
    with pytest.raises(LLMError) as ei:
        _complete(_provider(), _req())
    assert ei.value.retryable is False


@pytest.mark.parametrize(
    "usage",
    [
        None,
        NS(prompt_tokens="x", completion_tokens=-1),
        NS(prompt_tokens=True, completion_tokens=None),
    ],
)
def test_bad_usage_never_kills_a_good_response(monkeypatch: pytest.MonkeyPatch, usage: Any) -> None:
    _patch(monkeypatch, _resp("ok", usage=usage))
    out = _complete(_provider(), _req())
    assert out.text == "ok" and out.usage.total_tokens == 0


def test_model_falls_back_to_routed_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _resp("ok", model=None))
    assert _complete(_provider(), _req()).model == "groq/llama-3.3-70b-versatile"


def test_mapping_response_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        {
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "checking",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "lookup_order",
                                    "arguments": {"order_id": "1042"},
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6},
        },
    )
    out = _complete(_provider(), _req())
    assert out.text == "checking"
    assert out.finish_reason == "tool_calls"
    assert out.usage.total_tokens == 10
    assert out.tool_calls[0].arguments == {"order_id": "1042"}


# -- error mapping --------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (vendor.exceptions.RateLimitError("slow down", "groq", "m"), True),
        (vendor.exceptions.ServiceUnavailableError("down", "groq", "m"), True),
        (vendor.exceptions.APIConnectionError("reset", "groq", "m"), True),
        (vendor.exceptions.Timeout("timed out", "m", "groq"), True),
        (vendor.exceptions.InternalServerError("oops", "m", "groq"), True),
        (vendor.exceptions.AuthenticationError("bad key", "groq", "m"), False),
        (vendor.exceptions.BadRequestError("bad", "m", "groq"), False),
        (ValueError("anything else"), False),
        (type("E503", (Exception,), {"status_code": 503})("x"), True),
        (type("E429", (Exception,), {"status_code": 429})("x"), True),
        (type("E408", (Exception,), {"status_code": 408})("x"), True),
        (type("E409", (Exception,), {"status_code": 409})("x"), True),
        (type("E400", (Exception,), {"status_code": 400})("x"), False),
    ],
)
def test_vendor_errors_become_llm_error(
    monkeypatch: pytest.MonkeyPatch, exc: Exception, retryable: bool
) -> None:
    _patch(monkeypatch, exc)
    with pytest.raises(LLMError) as ei:
        _complete(_provider(), _req())
    assert ei.value.retryable is retryable
    assert ei.value.__cause__ is None


def test_error_message_does_not_leak_vendor_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, vendor.exceptions.BadRequestError("SECRET PROMPT gsk_key", "m", "groq"))
    with pytest.raises(LLMError) as ei:
        _complete(_provider(), _req())
    assert "SECRET" not in str(ei.value) and "gsk_key" not in str(ei.value)
    assert "BadRequestError" in str(ei.value) and "status=400" in str(ei.value)


def test_colliding_metadata_keys_do_not_break_failure_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, RuntimeError("boom"))
    with pytest.raises(LLMError):  # a KeyError from logging would surface here instead
        _complete(_provider(), _req(metadata={"name": "x", "message": "y", "args": "z"}))


# -- streaming ------------------------------------------------------------


def _chunk(
    content: Any = None, *, finish: str | None = None, tool_calls: Any = None, usage: Any = None
) -> NS:
    return NS(
        choices=[NS(finish_reason=finish, delta=NS(content=content, tool_calls=tool_calls))],
        usage=usage,
    )


def _delta(index: Any = 0, *, id: str | None = None, name: str | None = None, args: str = "") -> NS:
    return NS(index=index, id=id, function=NS(name=name, arguments=args))


class _Stream:
    def __init__(
        self, chunks: list[Any], *, fail_with: Exception | None = None, close_raises: bool = False
    ) -> None:
        self._chunks, self._fail, self._close_raises = chunks, fail_with, close_raises
        self.closed = False

    def __aiter__(self) -> _Stream:
        self._it = iter(self._chunks)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            if self._fail:
                raise self._fail from None
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.closed = True
        if self._close_raises:
            raise RuntimeError("close failed")


async def _drain(p: LiteLLMProvider, req: LLMRequest) -> list[LLMChunk]:
    return [c async for c in p.stream(req)]


def _stream(p: LiteLLMProvider | None = None, req: LLMRequest | None = None) -> list[LLMChunk]:
    return asyncio.run(_drain(p or _provider(), req or _req()))


def test_stream_sets_skip_mcp_handler_when_tools_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _patch(monkeypatch, _Stream([_chunk(None, finish="stop")]))
    _stream(_provider(), _req(tools=(LLMToolSpec("t", "d", {}),)))
    assert seen["_skip_mcp_handler"] is True


def test_stream_text_usage_and_single_final_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            _chunk("Hel"),
            _chunk("lo"),
            _chunk(None, finish="stop"),
            NS(choices=[], usage=NS(prompt_tokens=5, completion_tokens=2)),  # usage-only chunk
        ]
    )
    seen = _patch(monkeypatch, s)
    chunks = _stream()
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert "".join(c.delta for c in chunks) == "Hello"
    assert [c.finish_reason for c in chunks if c.finish_reason] == ["stop"]
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens == 7
    assert s.closed


def test_stream_openai_style_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            _chunk(None, tool_calls=[_delta(id="c1", name="lookup_order")]),
            _chunk(None, tool_calls=[_delta(args='{"order_')]),
            _chunk(None, tool_calls=[_delta(args='id": "1042"}')]),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    last = _stream()[-1]
    assert last.finish_reason == "tool_calls"
    assert (last.tool_calls[0].id, last.tool_calls[0].name) == ("c1", "lookup_order")
    assert last.tool_calls[0].arguments == {"order_id": "1042"}


def test_stream_repeated_id_and_name_do_not_get_concatenated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _Stream(
        [
            _chunk(None, tool_calls=[_delta(id="c1", name="lookup_order", args='{"order_')]),
            _chunk(None, tool_calls=[_delta(id="c1", name="lookup_order", args='id": "1042"}')]),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    call = _stream()[-1].tool_calls[0]
    assert (call.id, call.name, dict(call.arguments)) == (
        "c1",
        "lookup_order",
        {"order_id": "1042"},
    )


def test_stream_parallel_tool_calls_keep_order(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            _chunk(
                None,
                tool_calls=[
                    _delta(0, id="a", name="one", args="{}"),
                    _delta(1, id="b", name="two"),
                ],
            ),
            _chunk(None, tool_calls=[_delta(1, args='{"x": 1}')]),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    calls = _stream()[-1].tool_calls
    assert [(c.id, c.name) for c in calls] == [("a", "one"), ("b", "two")]
    assert dict(calls[1].arguments) == {"x": 1}


def test_stream_reused_index_with_new_id_is_a_new_call(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            _chunk(None, tool_calls=[_delta(0, id="a", name="one", args="{}")]),
            _chunk(None, tool_calls=[_delta(0, id="b", name="two", args="{}")]),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    assert [c.id for c in _stream()[-1].tool_calls] == ["a", "b"]


def test_stream_cut_short_reports_error_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _Stream([_chunk("partial")]))
    assert _stream()[-1].finish_reason == "error"


def test_stream_midway_failure_becomes_llm_error_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exc = vendor.exceptions.APIConnectionError("reset", "groq", "m")
    s = _Stream([_chunk("a")], fail_with=exc)
    _patch(monkeypatch, s)
    with pytest.raises(LLMError) as ei:
        _stream()
    assert ei.value.retryable is True and s.closed


def test_stream_setup_failure_becomes_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, vendor.exceptions.AuthenticationError("bad key", "groq", "m"))
    with pytest.raises(LLMError) as ei:
        _stream()
    assert ei.value.retryable is False


def test_stream_capability_check_happens_before_any_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _patch(monkeypatch, _Stream([]))
    p = _provider(capabilities=ProviderCaps(tools=False, json_mode=True, max_ctx=1000))
    with pytest.raises(LLMError):
        _stream(p, _req(tools=(LLMToolSpec("t", "d", {}),)))
    assert seen == {}


def test_stream_non_text_delta_is_a_permanent_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _Stream([_chunk(["x"])]))
    with pytest.raises(LLMError) as ei:
        _stream()
    assert ei.value.retryable is False


def test_stream_early_exit_releases_the_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream([_chunk("a"), _chunk("b"), _chunk(None, finish="stop")])
    _patch(monkeypatch, s)

    async def run() -> None:
        agen = _provider().stream(_req())
        await agen.__anext__()
        await agen.aclose()  # consumer walks away mid-stream

    asyncio.run(run())
    assert s.closed


def test_failing_aclose_never_masks_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _Stream([_chunk("a"), _chunk(None, finish="stop")], close_raises=True))
    assert _stream()[-1].finish_reason == "stop"


def test_stream_mapping_chunks_and_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}], "usage": None},
            {
                "choices": [
                    {
                        "delta": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        ]
    )
    _patch(monkeypatch, s)
    chunks = _stream()
    assert "".join(c.delta for c in chunks) == "Hi"
    last = chunks[-1]
    assert last.finish_reason == "tool_calls"
    assert last.usage is not None and last.usage.total_tokens == 5
    assert (last.tool_calls[0].id, last.tool_calls[0].name) == ("c1", "lookup")


def test_stream_ignores_none_delta_and_none_function(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            NS(choices=[NS(finish_reason=None, delta=None)], usage=None),
            _chunk(None, tool_calls=[NS(index=0, id="c1", function=None)]),
            _chunk(None, tool_calls=[_delta(0, name="lookup", args="{}")]),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    last = _stream()[-1]
    assert (last.tool_calls[0].id, last.tool_calls[0].name, dict(last.tool_calls[0].arguments)) == (
        "c1",
        "lookup",
        {},
    )


def test_stream_sparse_and_non_int_tool_call_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Stream(
        [
            _chunk(
                None,
                tool_calls=[
                    _delta(2, id="a", name="one", args='{"x":'),
                    _delta(5, id="b", name="two", args="{}"),
                ],
            ),
            _chunk(None, tool_calls=[_delta(2, args=" 1}")]),
            _chunk(
                None,
                tool_calls=[
                    _delta("bad", id="c", name="three", args="{}"),
                    _delta(True, id="d", name="four", args="{}"),
                ],
            ),
            _chunk(None, finish="tool_calls"),
        ]
    )
    _patch(monkeypatch, s)
    calls = _stream()[-1].tool_calls
    assert [(c.id, c.name, dict(c.arguments)) for c in calls] == [
        ("a", "one", {"x": 1}),
        ("b", "two", {}),
        ("c", "three", {}),
        ("d", "four", {}),
    ]
