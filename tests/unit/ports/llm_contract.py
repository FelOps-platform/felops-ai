import asyncio
 
import pytest
 
from felops.ports.llm import (
    LLMChunk,
    LLMError,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMToolSpec,
    ProviderCaps,
)
 
 
def _req(text: str = "hello", **kw) -> LLMRequest:
    return LLMRequest(messages=(LLMMessage("user", text),), **kw)
 
 
class LLMProviderContract:
    def make_provider(self) -> LLMProvider:  # override in adapter tests
        raise NotImplementedError
 
    @pytest.fixture
    def provider(self) -> LLMProvider:
        return self.make_provider()
 
    def test_satisfies_protocol(self, provider):
        assert isinstance(provider, LLMProvider)
 
    def test_capabilities_declared(self, provider):
        caps = provider.capabilities
        assert isinstance(caps, ProviderCaps)
        assert caps.max_ctx > 0
 
    def test_complete_returns_response(self, provider):
        resp = asyncio.run(provider.complete(_req()))
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.text, str) and resp.text
        assert resp.model
        assert resp.usage.total_tokens >= 0
 
    def test_stream_yields_chunks_and_one_final(self, provider):
        async def collect() -> list[LLMChunk]:
            return [c async for c in provider.stream(_req())]
 
        chunks = asyncio.run(collect())
        assert chunks and all(isinstance(c, LLMChunk) for c in chunks)
        finals = [c for c in chunks if c.finish_reason is not None]
        assert len(finals) == 1 and chunks[-1] is finals[0]
        assert "".join(c.delta for c in chunks)
 
    def test_unsupported_json_mode_raises_llm_error(self, provider):
        if provider.capabilities.json_mode:
            pytest.skip("provider supports json_mode")
        with pytest.raises(LLMError):
            asyncio.run(provider.complete(_req(json_mode=True)))
 
    def test_unsupported_tools_raise_llm_error(self, provider):
        if provider.capabilities.tools:
            pytest.skip("provider supports tools")
        tool = LLMToolSpec("lookup_order", "Look up an order", {"type": "object"})
        with pytest.raises(LLMError):
            asyncio.run(provider.complete(_req(tools=(tool,))))