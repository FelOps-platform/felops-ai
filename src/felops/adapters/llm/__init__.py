"""LLM adapters. Selected by `llm.provider` in config; wired only by `app/container.py`."""

from felops.adapters.llm.litellm import LiteLLMProvider

__all__ = ["LiteLLMProvider"]
