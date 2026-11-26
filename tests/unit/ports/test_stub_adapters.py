"""Runs the contract suites against the in-memory stubs (task 0.4: 'Contract
test stub'). Real adapters (0.7 LiteLLM + file store) add their own subclasses."""

from __future__ import annotations

from felops.ports.llm import ProviderCaps
from tests.fakes.llm import StubLLMProvider
from tests.fakes.prompt import StubPromptStore
from tests.unit.ports.llm_contract import LLMProviderContract
from tests.unit.ports.prompts_contract import PromptSeed, PromptStoreContract


class TestStubLLMProvider(LLMProviderContract):
    def make_provider(self):
        return StubLLMProvider()


class TestStubLLMProviderLimitedCaps(LLMProviderContract):
    def make_provider(self):
        return StubLLMProvider(ProviderCaps(tools=False, json_mode=False, max_ctx=4096))


class TestStubPromptStore(PromptStoreContract):
    def make_store(self, seeds: list[PromptSeed]):
        store = StubPromptStore()
        for s in seeds:
            store.add(
                s.scope,
                s.name,
                s.template,
                workspace_id=s.workspace_id,
                agent_id=s.agent_id,
                sub_agent_id=s.sub_agent_id,
            )
        return store
