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
    be huge. The full text is available at DEBUG level.
    """
    if isinstance(exc, LLMError):
        return exc
    status = getattr(exc, "status_code", None)
    retryable = isinstance(exc, _RETRYABLE_ERRORS) or (
        isinstance(status, int) and (status >= 500 or status in _RETRYABLE_STATUS)
    )
    logger.debug("llm vendor error detail: %s", exc)
    suffix = f" (status={status})" if isinstance(status, int) else ""
    err = LLMError(f"{type(exc).__name__}{suffix}", retryable=retryable)
    err.__cause__ = exc
    return err


def _to_usage(usage: Any) -> TokenUsage:
    """Tolerant on purpose: a billing field must never kill a good response."""

    def _count(key: str) -> int:
        value = _get(usage, key, 0)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return 0

    return TokenUsage(
        prompt_tokens=_count("prompt_tokens"),
        completion_tokens=_count("completion_tokens"),
    )


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
    if not raw:
        finish: FinishReason | None = None
    else:
        finish = _FINISH_MAP.get(raw, "error") if isinstance(raw, str) else "error"
    if has_tool_calls and finish in (None, "stop"):
        return "tool_calls"
    return finish if finish is not None else missing


def _build_tool_call(call_id: str, name: str, raw_args: Any) -> LLMToolCall:
    """Single place where a tool call is validated. Used by complete() and stream()."""
    if not call_id or not name:
        # A call without id/name cannot be answered with a role="tool" message.
        raise LLMError("incomplete tool call from provider", retryable=True)
    if raw_args is None or raw_args == "":
        arguments: dict[str, Any] = {}
    elif isinstance(raw_args, Mapping):
        arguments = dict(raw_args)
    else:
        try:
            parsed = json.loads(raw_args)
        except (TypeError, ValueError) as exc:
            # Broken JSON from the model is non-deterministic: a retry may succeed.
            raise LLMError(f"malformed arguments for tool {name!r}", retryable=True) from exc
        if not isinstance(parsed, dict):
            raise LLMError(f"arguments for tool {name!r} are not an object", retryable=True)
        arguments = parsed
    return LLMToolCall(id=call_id, name=name, arguments=MappingProxyType(arguments))


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
            index = _get(delta, "index")
            if not isinstance(index, int) or isinstance(index, bool):
                index = position
            call_id = _get(delta, "id") or ""
            function = _get(delta, "function")
            slot = self._by_index.get(index)
            if slot is None or (call_id and slot.id and call_id != slot.id):
                slot = _Slot()
                self._slots.append(slot)
                self._by_index[index] = slot
            slot.id = slot.id or call_id
            slot.name = slot.name or (_get(function, "name") or "")
            slot.args.append(_get(function, "arguments") or "")

    def build(self) -> tuple[LLMToolCall, ...]:
        return tuple(_build_tool_call(s.id, s.name, "".join(s.args)) for s in self._slots)


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
        default_model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not provider or not provider.strip():
            raise ValueError("provider must not be empty")
        if default_model is not None and not default_model.strip():
            raise ValueError("default_model must not be blank")
        if capabilities.max_ctx <= 0:
            raise ValueError("capabilities.max_ctx must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
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
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if req.tools:
            kwargs["tools"] = [_tool_to_dict(t) for t in req.tools]
        if req.stop:
            kwargs["stop"] = list(req.stop)
        if req.metadata:
            # Trace baggage only: forwarded to LiteLLM callbacks, never used
            # for behaviour.
            kwargs["metadata"] = dict(req.metadata)
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
            raise _to_llm_error(exc) from exc

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        kwargs = self._kwargs(req, stream=True)
        finish_raw: Any = None
        usage: TokenUsage | None = None
        calls = _ToolCallAccumulator()
        upstream: Any = None
        try:
            upstream = await litellm.acompletion(**kwargs)
            async for chunk in upstream:
                chunk_usage = _get(chunk, "usage")
                if chunk_usage is not None:
                    usage = _to_usage(chunk_usage)
                choices = _get(chunk, "choices") or []
                if not choices:
                    continue  # e.g. the usage-only final chunk
                choice = choices[0]
                delta = _get(choice, "delta")
                calls.add(_get(delta, "tool_calls"))
                text = _text_of(_get(delta, "content"))
                if text:
                    yield LLMChunk(delta=text)
                reason = _get(choice, "finish_reason")
                if reason:
                    finish_raw = reason
            tool_calls = calls.build()
            # Exactly one terminal chunk. A stream that ended without a finish
            # reason was cut short, so it is reported as "error", not "stop".
            yield LLMChunk(
                finish_reason=_resolve_finish(
                    finish_raw, has_tool_calls=bool(tool_calls), missing="error"
                ),
                tool_calls=tool_calls,
                usage=usage,
            )
        except LLMError:
            raise
        except Exception as exc:
            self._log_failure("llm stream failed", kwargs, req)
            raise _to_llm_error(exc) from exc
        finally:
            # Runs on normal end, error and consumer cancellation/early break,
            # so the HTTP connection is released. Must never mask the original
            # exception.
            aclose = getattr(upstream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    # -- response parsing --------------------------------------------------

    @staticmethod
    def _parse_response(resp: Any, routed_model: str) -> LLMResponse:
        choices = _get(resp, "choices") or []
        if not choices:
            raise LLMError("provider returned no choices", retryable=True)
        choice = choices[0]
        message = _get(choice, "message")
        tool_calls = tuple(
            _build_tool_call(
                _get(tc, "id") or "",
                _get(_get(tc, "function"), "name") or "",
                _get(_get(tc, "function"), "arguments"),
            )
            for tc in _get(message, "tool_calls") or []
        )
        model = _get(resp, "model")
        return LLMResponse(
            text=_text_of(_get(message, "content")),
            model=model if isinstance(model, str) and model else routed_model,
            finish_reason=_resolve_finish(
                _get(choice, "finish_reason"),
                has_tool_calls=bool(tool_calls),
                missing="stop",
            ),
            tool_calls=tool_calls,
            usage=_to_usage(_get(resp, "usage")),
        )
