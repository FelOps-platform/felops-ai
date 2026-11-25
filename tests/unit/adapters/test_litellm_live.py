"""Real round-trips through `LiteLLMProvider`.

Skipped unless API_KEY is set (shell or repo-root `.env`).

    pytest tests/unit/adapters/test_litellm_live.py -v

Optional overrides:
    FELOPS_LIVE_LLM_MODEL   model to use (default: openai/gpt-oss-20b,
                            which supports tools and JSON mode on Groq)

Free-tier rate limits are not a code failure: a retryable rate-limit error
skips the test instead of failing it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

import pytest

from felops.adapters.llm import LiteLLMProvider
from felops.adapters.prompts import FilePromptStore
from felops.ports.llm import (
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMRequest,
    LLMToolSpec,
    ProviderCaps,
)
from felops.ports.prompts import PromptContext

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _unquote_env_value(val: str) -> str | None:
    if len(val) < 2 or val[0] not in ('"', "'"):
        return None
    end = val.find(val[0], 1)
    if end == -1:
        return None
    trailing = val[end + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        return None
    return val[1:end]


def _parse_env_value(raw: str) -> str:
    val = raw.strip()
    unquoted = _unquote_env_value(val)
    if unquoted is not None:
        return unquoted
    return val.split("#", 1)[0].strip()


def _strip_export_prefix(line: str) -> str:
    if line.startswith("export") and len(line) > 6 and line[6].isspace():
        return line[6:].lstrip()
    return line


def _parse_env_line(raw: str) -> tuple[str, str] | None:
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = _strip_export_prefix(line).partition("=")
    key, value = key.strip(), _parse_env_value(value)
    if not key or not value:
        return None
    return key, value


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw)
        if parsed is not None:
            os.environ.setdefault(*parsed)


_load_env_file(_ENV_FILE)

_API_KEY = os.getenv("API_KEY", "").strip()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not _API_KEY or _API_KEY == "your_api_key_here",
        reason="API_KEY not set",
    ),
]

MODEL = os.getenv("FELOPS_LIVE_LLM_MODEL", "openai/gpt-oss-20b")
CAPS = ProviderCaps(tools=True, json_mode=True, max_ctx=131_072)
REPO_PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
ARABIC = re.compile(r"[\u0600-\u06FF]")

ORDER_TOOL = LLMToolSpec(
    name="get_order_status",
    description="Look up the status of a customer's order by its order id.",
    parameters={
        "type": "object",
        "properties": {"order_id": {"type": "string", "description": "The order number"}},
        "required": ["order_id"],
    },
)

T = TypeVar("T")


def _provider(**kw: Any) -> LiteLLMProvider:
    # LiteLLM's Groq backend reads GROQ_API_KEY, not API_KEY, so pass the key
    # explicitly. A caller-supplied api_key (the bad-key tests) still wins.
    kw.setdefault("api_key", os.getenv("API_KEY"))
    return LiteLLMProvider(provider="groq", capabilities=CAPS, default_model=MODEL, **kw)


def _run(coro: Awaitable[T]) -> T:
    """Run a call; turn a free-tier rate limit into a skip, not a failure."""

    async def go() -> T:
        return await coro

    try:
        return asyncio.run(go())
    except LLMError as exc:
        if exc.retryable and ("RateLimitError" in str(exc) or "status=429" in str(exc)):
            pytest.skip("Groq rate limit hit (free tier)")
        raise


def _order_req(**kw: Any) -> LLMRequest:
    return LLMRequest(
        messages=(
            LLMMessage("system", "You are a support agent. Use the tools you are given."),
            LLMMessage("user", "Where is my order #1042?"),
        ),
        tools=(ORDER_TOOL,),
        max_tokens=200,
        **kw,
    )


async def _collect(req: LLMRequest, provider: LiteLLMProvider | None = None) -> list[LLMChunk]:
    return [c async for c in (provider or _provider()).stream(req)]


# -- complete() -----------------------------------------------------------


def test_completion_works() -> None:
    req = LLMRequest(
        messages=(LLMMessage("user", "Reply with the single word: pong"),),
        max_tokens=300,  # reasoning model: thinking tokens count against this
        metadata={"workspace_id": "w1", "node": "live_test"},
    )
    out = _run(_provider().complete(req))
    assert "pong" in out.text.lower()
    assert out.finish_reason == "stop"
    assert out.model
    assert out.usage.prompt_tokens > 0 and out.usage.completion_tokens > 0


def test_request_model_overrides_default() -> None:
    req = LLMRequest(
        messages=(LLMMessage("user", "Say hi."),), model="openai/gpt-oss-20b", max_tokens=300
    )
    out = _run(_provider().complete(req))
    assert "20b" in out.model.lower()


def test_json_mode_returns_parseable_json() -> None:
    req = LLMRequest(
        messages=(
            LLMMessage(
                "user", 'Return a JSON object with the keys "city" and "country" for Paris.'
            ),
        ),
        json_mode=True,
        max_tokens=400,
    )
    data = json.loads(_run(_provider().complete(req)).text)
    assert "paris" in str(data.get("city", "")).lower()


def test_tool_call_is_returned_not_executed() -> None:
    out = _run(_provider().complete(_order_req()))
    assert out.finish_reason == "tool_calls"
    call = out.tool_calls[0]
    assert call.name == "get_order_status" and call.id
    assert "1042" in str(call.arguments["order_id"])


def test_arabic_in_arabic_out() -> None:
    req = LLMRequest(
        messages=(LLMMessage("user", "ما هي عاصمة مصر؟ أجب بجملة قصيرة."),), max_tokens=400
    )
    assert ARABIC.search(_run(_provider().complete(req)).text)


def test_length_limit_is_reported_as_length() -> None:
    req = LLMRequest(
        messages=(
            LLMMessage(
                "user",
                "Write a detailed 500-word essay about the history of public libraries.",
            ),
        ),
        max_tokens=50,
    )
    out = _run(_provider().complete(req))
    assert out.finish_reason == "length"


# -- stream() -------------------------------------------------------------


def test_streaming_works() -> None:
    req = LLMRequest(
        messages=(LLMMessage("user", "Write two short sentences about the sea."),), max_tokens=400
    )
    chunks = _run(_collect(req))
    deltas = [c for c in chunks if c.delta]
    assert len(deltas) >= 1, "expected at least one content delta"
    assert "".join(c.delta for c in chunks).strip()
    finals = [c for c in chunks if c.finish_reason]
    assert len(finals) == 1 and chunks[-1] is finals[0]  # exactly one, and it is last
    assert chunks[-1].finish_reason in ("stop", "length")
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens > 0


def test_streamed_tool_call_is_assembled() -> None:
    chunks = _run(_collect(_order_req()))
    last = chunks[-1]
    assert last.finish_reason == "tool_calls"
    call = last.tool_calls[0]
    assert call.name == "get_order_status" and call.id
    assert "1042" in str(call.arguments["order_id"])


# -- failure mapping ------------------------------------------------------


def test_bad_api_key_is_permanent_and_not_leaked() -> None:
    req = LLMRequest(messages=(LLMMessage("user", "hi"),), max_tokens=5)
    with pytest.raises(LLMError) as ei:
        _run(_provider(api_key="gsk_this_key_is_invalid").complete(req))
    assert ei.value.retryable is False
    assert "gsk_this_key_is_invalid" not in str(ei.value)


def test_unknown_model_is_permanent() -> None:
    req = LLMRequest(messages=(LLMMessage("user", "hi"),), model="no-such-model-xyz", max_tokens=5)
    with pytest.raises(LLMError) as ei:
        _run(_provider().complete(req))
    assert ei.value.retryable is False


def test_timeout_is_retryable() -> None:
    req = LLMRequest(messages=(LLMMessage("user", "hi"),), max_tokens=5)
    with pytest.raises(LLMError) as ei:
        _run(_provider(timeout=0.001).complete(req))
    assert ei.value.retryable is True


def test_stream_failure_is_mapped_too() -> None:
    req = LLMRequest(messages=(LLMMessage("user", "hi"),), max_tokens=5)
    with pytest.raises(LLMError) as ei:
        _run(_collect(req, _provider(api_key="gsk_this_key_is_invalid")))
    assert ei.value.retryable is False


# -- with the file PromptStore (the other half of task 0.7) ----------------


def test_platform_prompts_drive_a_real_reply_in_the_customer_language() -> None:
    ctx = PromptContext(workspace_id="w1", agent_id="a1", sub_agent_id="customer_support")
    store = FilePromptStore(REPO_PROMPTS)
    system = (
        store.resolve("default", ctx).template + "\n\n" + store.resolve("responder", ctx).template
    )
    req = LLMRequest(
        messages=(
            LLMMessage("system", system),
            LLMMessage("user", "مرحبا، ما هي ساعات العمل عندكم؟"),
        ),
        max_tokens=500,
    )
    out = _run(_provider().complete(req))
    assert ARABIC.search(out.text), "platform default says: reply in the customer's language"
