"""LiteLLM implementation of the `LLMProvider` port (§3, §5.3 Adapter).

One adapter covers openai | anthropic | gemini through LiteLLM. Routing policy
(tiers, fallback, retry, budget) is NOT here: it belongs to the Model Gateway.
This class does exactly one call to one provider and normalises the result.

The vendor SDK is imported only in this module (§5: vendor SDKs live in
`adapters/`). Vendor exceptions never leave it: everything is raised as
`LLMError`, with `retryable` set so the gateway can choose retry vs fallback.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import litellm

from felops.adapters.llm._common import ensure_supported
from felops.ports.llm import (
    FinishReason,
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMToolCall,
    LLMToolSpec,
    ProviderCaps,
    TokenUsage,
)

logger = logging.getLogger(__name__)

# Transient failures: the gateway may retry these.
_RETRYABLE_ERRORS: tuple[type[BaseException], ...] = (
    litellm.exceptions.RateLimitError,
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.ServiceUnavailableError,
    litellm.exceptions.InternalServerError,
)
_RETRYABLE_STATUS = frozenset({408, 409, 429})

_FINISH_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "error",
}


# -- small helpers ---------------------------------------------------------


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read `key` from a LiteLLM response object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_llm_error(exc: Exception) -> LLMError:
    """Map any vendor exception to `LLMError`.

    The message carries only the exception type and status: LiteLLM exception
    text often embeds raw provider payloads (sometimes prompt content) and can
    be huge.
    """
    if isinstance(exc, LLMError):
        return exc
    status = getattr(exc, "status_code", None)
    retryable = _is_retryable_exception(exc, status)
    suffix = f" (status={status})" if isinstance(status, int) else ""
    return LLMError(f"{type(exc).__name__}{suffix}", retryable=retryable)


def _is_retryable_exception(exc: Exception, status: Any) -> bool:
    if isinstance(exc, _RETRYABLE_ERRORS):
        return True
    if not isinstance(status, int):
        return False
    return status >= 500 or status in _RETRYABLE_STATUS


def _non_negative_int(usage: Any, key: str) -> int:
    value = _get(usage, key, 0)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    if value < 0:
        return 0
    return value


def _to_usage(usage: Any) -> TokenUsage:
    """Tolerant on purpose: a billing field must never kill a good response."""
    return TokenUsage(
        prompt_tokens=_non_negative_int(usage, "prompt_tokens"),
        completion_tokens=_non_negative_int(usage, "completion_tokens"),
    )


def _normalise_finish_raw(raw: Any) -> FinishReason | None:
    if not raw:
        return None
    if not isinstance(raw, str):
        return "error"
    return _FINISH_MAP.get(raw, "error")


def _resolve_finish(
    raw: Any,
    *,
    has_tool_calls: bool,
    missing: FinishReason,
) -> FinishReason:
    """Normalise a vendor finish reason.

    - missing/empty  -> `missing` (caller decides: "stop" for complete(),
      "error" for a stream that was cut short)
    - unknown value  -> "error"
    - tool calls present but reason is stop/missing (some providers, notably
      Gemini via LiteLLM) -> "tool_calls". Error/length are never overridden.
    """
    finish = _normalise_finish_raw(raw)
    if has_tool_calls and finish in (None, "stop"):
        return "tool_calls"
    return finish if finish is not None else missing


def _build_tool_call(call_id: str, name: str, raw_args: Any) -> LLMToolCall:
    """Single place where a tool call is validated. Used by complete() and stream()."""
    if not call_id or not name:
        # A call without id/name cannot be answered with a role="tool" message.
        raise LLMError("incomplete tool call from provider", retryable=True)
    arguments = _parse_tool_arguments(raw_args)
    return LLMToolCall(id=call_id, name=name, arguments=MappingProxyType(arguments))


def _parse_tool_arguments(raw_args: Any) -> dict[str, Any]:
    if raw_args is None or raw_args == "":
        return {}
    if isinstance(raw_args, Mapping):
        return dict(raw_args)
    return _decode_json_object(raw_args)


def _decode_json_object(raw_args: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_args)
    except (TypeError, ValueError):
        # Broken JSON from the model is non-deterministic: a retry may succeed.
        raise LLMError("malformed tool arguments", retryable=True) from None
    if not isinstance(parsed, dict):
        raise LLMError("tool arguments must be an object", retryable=True)
    return parsed


def _delta_index(delta: Any, position: int) -> int:
    index = _get(delta, "index")
    return index if isinstance(index, int) and not isinstance(index, bool) else position


def _validate_non_blank(value: str, message: str) -> None:
    if not value.strip():
        raise ValueError(message)


def _validate_provider_config(
    provider: str, capabilities: ProviderCaps, default_model: str | None, timeout: float
) -> None:
    _validate_non_blank(provider, "provider must not be empty")
    if default_model is not None:
        _validate_non_blank(default_model, "default_model must not be blank")
    if capabilities.max_ctx <= 0:
        raise ValueError("capabilities.max_ctx must be positive")
    if timeout <= 0:
        raise ValueError("timeout must be positive")


def _request_option_kwargs(req: LLMRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    kwargs.update(_optional_collection_kwargs(req))
    return kwargs


def _optional_collection_kwargs(req: LLMRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if req.tools:
        kwargs["tools"] = [_tool_to_dict(t) for t in req.tools]
    if req.stop:
        kwargs["stop"] = list(req.stop)
    if req.metadata:
        kwargs["metadata"] = dict(req.metadata)
    return kwargs


def _message_to_dict(m: LLMMessage) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_call_id is not None:
        out["tool_call_id"] = m.tool_call_id
    return out


def _tool_to_dict(t: LLMToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": dict(t.parameters),
        },
    }


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if not isinstance(content, str):
        raise LLMError("provider returned non-text message content", retryable=False)
    return content


# -- streamed tool-call assembly -------------------------------------------


@dataclass(slots=True)
class _Slot:
    id: str = ""
    name: str = ""
    args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _StreamState:
    calls: _ToolCallAccumulator = field(default_factory=lambda: _ToolCallAccumulator())
    finish_raw: Any = None
    usage: TokenUsage | None = None


def _apply_choice_delta(choice: Any, state: _StreamState) -> str | None:
    delta = _get(choice, "delta")
    state.calls.add(_get(delta, "tool_calls"))
    text = _text_of(_get(delta, "content"))
    reason = _get(choice, "finish_reason")
    if reason:
        state.finish_raw = reason
    return text or None


def _process_chunk(chunk: Any, state: _StreamState) -> str | None:
    chunk_usage = _get(chunk, "usage")
    if chunk_usage is not None:
        state.usage = _to_usage(chunk_usage)
    choices = _get(chunk, "choices") or []
    if not choices:
        return None
    return _apply_choice_delta(choices[0], state)


def _terminal_chunk(state: _StreamState) -> LLMChunk:
    tool_calls = state.calls.build()
    return LLMChunk(
        finish_reason=_resolve_finish(
            state.finish_raw, has_tool_calls=bool(tool_calls), missing="error"
        ),
        tool_calls=tool_calls,
        usage=state.usage,
    )


async def _close_stream(upstream: Any) -> None:
    aclose = getattr(upstream, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()


class _ToolCallAccumulator:
    """Rebuilds tool calls from streamed deltas.

    Tolerates both delta styles:
    - OpenAI: `id` and `name` arrive once (first delta), arguments in pieces.
    - Some compatible/local servers: `id` and `name` repeated in every delta.
    So `id` and `name` are set once (never appended); only arguments are
    appended. If an index is reused with a *different* id, it is a new call.
    """

    def __init__(self) -> None:
        self._slots: list[_Slot] = []
        self._by_index: dict[int, _Slot] = {}

    def add(self, deltas: Any) -> None:
        if not deltas:
            return
        for position, delta in enumerate(deltas):
            index = _delta_index(delta, position)
            call_id = _get(delta, "id") or ""
            self._apply_delta(self._get_or_create_slot(index, call_id), delta, call_id)

    def _get_or_create_slot(self, index: int, call_id: str) -> _Slot:
        slot = self._by_index.get(index)
        if _is_new_call(slot, call_id):
            slot = _Slot()
            self._slots.append(slot)
            self._by_index[index] = slot
        return slot

    @staticmethod
    def _apply_delta(slot: _Slot, delta: Any, call_id: str) -> None:
        function = _get(delta, "function")
        slot.id = slot.id or call_id
        slot.name = slot.name or (_get(function, "name") or "")
        slot.args.append(_get(function, "arguments") or "")

    def build(self) -> tuple[LLMToolCall, ...]:
        return tuple(_build_tool_call(s.id, s.name, "".join(s.args)) for s in self._slots)


def _is_new_call(slot: _Slot | None, call_id: str) -> bool:
    if slot is None:
        return True
    if not call_id or not slot.id:
        return False
    return call_id != slot.id


# -- the adapter -----------------------------------------------------------


class LiteLLMProvider:
    """`LLMProvider` backed by LiteLLM.

    Args:
        provider: LiteLLM provider prefix, i.e. the `llm.provider` config value
            (`anthropic`, `openai`, `gemini`, ...).
        capabilities: what the configured model(s) can do. Required: the
            gateway routes on this, so it must be stated, not guessed.
        default_model: used when `LLMRequest.model` is None. The gateway normally
            sends a concrete model id (it maps tier -> model).
        api_key / api_base: optional. When None LiteLLM reads the provider's
            standard environment variable (e.g. `ANTHROPIC_API_KEY`).
        timeout: per-request timeout in seconds.
    """

    def __init__(
        self,
        *,
        provider: str,
        capabilities: ProviderCaps,
        **options: Any,
    ) -> None:
        unknown = options.keys() - {"default_model", "api_key", "api_base", "timeout"}
        if unknown:
            name = sorted(unknown)[0]
            raise TypeError(f"unexpected keyword argument: {name!r}")
        default_model = options.get("default_model")
        api_key = options.get("api_key")
        api_base = options.get("api_base")
        timeout = options.get("timeout", 60.0)
        _validate_provider_config(provider, capabilities, default_model, timeout)
        self.provider = provider.strip()
        self.capabilities = capabilities
        self._default_model = default_model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout = timeout

    # -- request building --------------------------------------------------

    def _route(self, model: str | None) -> str:
        concrete = model or self._default_model
        if not concrete:
            raise LLMError("no model given and no default_model configured", retryable=False)
        prefix = f"{self.provider}/"
        return concrete if concrete.startswith(prefix) else prefix + concrete

    def _kwargs(self, req: LLMRequest, *, stream: bool) -> dict[str, Any]:
        ensure_supported(self.capabilities, req)
        kwargs: dict[str, Any] = {
            "model": self._route(req.model),
            "messages": [_message_to_dict(m) for m in req.messages],
            "temperature": req.temperature,
            "timeout": self._timeout,
            # Retry is the Model Gateway's job. Never retry twice: disable both
            # LiteLLM's own retries and the underlying SDK's retries.
            "num_retries": 0,
            "max_retries": 0,
            "stream": stream,
        }
        kwargs.update(_request_option_kwargs(req))
        kwargs.update(self._provider_option_kwargs(stream=stream))
        return kwargs

    def _provider_option_kwargs(self, *, stream: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if stream:
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    @staticmethod
    def _log_failure(what: str, kwargs: Mapping[str, Any], req: LLMRequest) -> None:
        # Namespaced keys: arbitrary metadata keys must never collide with
        # LogRecord attributes (which would raise KeyError inside logging).
        logger.warning(
            what,
            extra={"llm_model": kwargs["model"], "llm_meta": dict(req.metadata)},
        )

    # -- LLMProvider -------------------------------------------------------

    async def complete(self, req: LLMRequest) -> LLMResponse:
        kwargs = self._kwargs(req, stream=False)
        try:
            resp: Any = await litellm.acompletion(**kwargs)
            return self._parse_response(resp, kwargs["model"])
        except LLMError:
            raise
        except Exception as exc:
            self._log_failure("llm call failed", kwargs, req)
            raise _to_llm_error(exc) from None

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        kwargs = self._kwargs(req, stream=True)
        state = _StreamState()
        upstream: Any = None
        try:
            upstream = await litellm.acompletion(**kwargs)
            async for chunk in upstream:
                text = _process_chunk(chunk, state)
                if text:
                    yield LLMChunk(delta=text)
            # Exactly one terminal chunk. A stream that ended without a finish
            # reason was cut short, so it is reported as "error", not "stop".
            yield _terminal_chunk(state)
        except LLMError:
            raise
        except Exception as exc:
            self._log_failure("llm stream failed", kwargs, req)
            raise _to_llm_error(exc) from None
        finally:
            # Runs on normal end, error and consumer cancellation/early break,
            # so the HTTP connection is released. Must never mask the original
            # exception.
            await _close_stream(upstream)

    # -- response parsing --------------------------------------------------

    @staticmethod
    def _parse_response(resp: Any, routed_model: str) -> LLMResponse:
        choices = _get(resp, "choices") or []
        if not choices:
            raise LLMError("provider returned no choices", retryable=True)
        choice = choices[0]
        message = _get(choice, "message")
        tool_calls = _parse_tool_calls(message)
        return LLMResponse(
            text=_text_of(_get(message, "content")),
            model=_pick_model(resp, routed_model),
            finish_reason=_resolve_finish(
                _get(choice, "finish_reason"),
                has_tool_calls=bool(tool_calls),
                missing="stop",
            ),
            tool_calls=tool_calls,
            usage=_to_usage(_get(resp, "usage")),
        )


def _parse_tool_calls(message: Any) -> tuple[LLMToolCall, ...]:
    return tuple(
            _build_tool_call(
                _get(tc, "id") or "",
                _get(_get(tc, "function"), "name") or "",
                _get(_get(tc, "function"), "arguments"),
            )
            for tc in _get(message, "tool_calls") or []
        )


def _pick_model(resp: Any, routed_model: str) -> str:
    model = _get(resp, "model")
    return model if isinstance(model, str) and model else routed_model
