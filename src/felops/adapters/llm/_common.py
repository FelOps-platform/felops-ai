"""Helpers shared by the LLM adapters. Private to `adapters/llm/`."""

from __future__ import annotations

from felops.ports.llm import LLMError, LLMRequest, ProviderCaps


def ensure_supported(caps: ProviderCaps, req: LLMRequest) -> None:
    """Fail fast, before any network call, when the request needs a capability
    the provider lacks. The port requires an `LLMError` here, and it is
    permanent (`retryable=False`): retrying cannot add a capability."""
    if req.tools and not caps.tools:
        raise LLMError("provider does not support tools", retryable=False)
    if req.json_mode and not caps.json_mode:
        raise LLMError("provider does not support json_mode", retryable=False)
