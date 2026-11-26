from __future__ import annotations

from collections.abc import AsyncIterator

from felops.ports.llm import (
    LLMChunk,
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderCaps,
    TokenUsage,
)


class StubLLMProvider:
    def __init__(self, capabilities: ProviderCaps | None = None) -> None:
        self.capabilities = capabilities or ProviderCaps(tools=True, json_mode=True, max_ctx=8192)

    def _check(self, req: LLMRequest) -> None:
        if req.tools and not self.capabilities.tools:
            raise LLMError("tools not supported")
        if req.json_mode and not self.capabilities.json_mode:
            raise LLMError("json_mode not supported")

    @staticmethod
    def _reply(req: LLMRequest) -> str:
        last_user = next((m.content for m in reversed(req.messages) if m.role == "user"), "")
        return f"stub: {last_user}"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self._check(req)
        text = self._reply(req)
        return LLMResponse(
            text=text,
            model=req.model or "stub-model",
            usage=TokenUsage(prompt_tokens=len(req.messages), completion_tokens=len(text.split())),
        )

    async def stream(self, req: LLMRequest) -> AsyncIterator[LLMChunk]:
        self._check(req)
        words = self._reply(req).split(" ")
        for i, word in enumerate(words):
            yield LLMChunk(delta=word if i == 0 else f" {word}")
        yield LLMChunk(finish_reason="stop", usage=TokenUsage(len(req.messages), len(words)))
