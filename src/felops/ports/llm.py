from __future__ import annotations
 
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable
 
Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "error"]
 
 
class LLMError(Exception):
    """Raised by every adapter instead of vendor exceptions.
 
    `retryable` lets the Model Gateway decide between retry (transient) and
    fallback (permanent) without knowing the vendor.
    """
 
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable
 
 
@dataclass(frozen=True, slots=True)
class ProviderCaps:
    """What a provider can do. The gateway checks this before routing."""
 
    tools: bool
    json_mode: bool
    max_ctx: int
 
 
@dataclass(frozen=True, slots=True)
class LLMMessage:
    role: Role
    content: str
    tool_call_id: str | None = None  # set only on role="tool"
 
 
@dataclass(frozen=True, slots=True)
class LLMToolSpec:
    """A tool the model may request. `parameters` is a JSON schema."""
 
    name: str
    description: str
    parameters: Mapping[str, Any]
 
 
@dataclass(frozen=True, slots=True)
class LLMToolCall:
    """A tool the model asked for. Never executed here: it goes through
    tool_selector -> policy_check -> executor (§6.3)."""
 
    id: str
    name: str
    arguments: Mapping[str, Any]
 
 
@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
 
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
 
 
@dataclass(frozen=True, slots=True)
class LLMRequest:
    messages: tuple[LLMMessage, ...]
    model: str | None = None  # concrete model id; the gateway maps tier -> model
    temperature: float = 0.0
    max_tokens: int | None = None
    json_mode: bool = False
    tools: tuple[LLMToolSpec, ...] = ()
    stop: tuple[str, ...] = ()
    # Trace baggage (workspace_id, agent_id, sub_agent_id, node). Never used
    # for behaviour by adapters, only forwarded to logs and usage records.
    metadata: Mapping[str, str] = field(default_factory=dict)
 
    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("LLMRequest.messages must not be empty")
 
 
@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    finish_reason: FinishReason = "stop"
    tool_calls: tuple[LLMToolCall, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)
 
 
@dataclass(frozen=True, slots=True)
class LLMChunk:
    """One streamed piece. The last chunk carries `finish_reason` (and
    `usage` when the provider reports it)."""
 
    delta: str = ""
    finish_reason: FinishReason | None = None
    tool_calls: tuple[LLMToolCall, ...] = ()
    usage: TokenUsage | None = None
 
 
@runtime_checkable
class LLMProvider(Protocol):  # openai | anthropic | gemini | local
    capabilities: ProviderCaps
 
    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Single-shot completion.
 
        Raises `LLMError` for any provider failure, and also when the request
        needs a capability the provider lacks (tools, json_mode).
        """
        ...
 
    def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Streamed completion. Implemented as an async generator (a plain
        `def` returning an async iterator, so callers `async for` it directly).
        Ends with exactly one chunk whose `finish_reason` is set."""
        ...