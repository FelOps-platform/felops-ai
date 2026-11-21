

from __future__ import annotations

from dataclasses import dataclass

import pytest

from felops.ports.prompts import (
    PromptContext,
    PromptNotFound,
    PromptScope,
    PromptStore,
    PromptVersion,
)

S = PromptScope


@dataclass(frozen=True)
class PromptSeed:
    scope: PromptScope
    name: str
    template: str
    workspace_id: str | None = None
    agent_id: str | None = None
    sub_agent_id: str | None = None


SEEDS = [
    PromptSeed(S.PLATFORM, "persona", "platform-default"),
    PromptSeed(S.PLATFORM_SUB_AGENT, "persona", "platform-cs", sub_agent_id="customer_support"),
    PromptSeed(S.WORKSPACE_SUB_AGENT, "persona", "acme-cs",
               workspace_id="acme", sub_agent_id="customer_support"),
    PromptSeed(S.WORKSPACE_AGENT_SUB_AGENT, "persona", "acme-agent1-cs",
               workspace_id="acme", agent_id="agent_1", sub_agent_id="customer_support"),
]


def ctx(ws="acme", agent="agent_1", sub="customer_support", **kw) -> PromptContext:
    return PromptContext(workspace_id=ws, agent_id=agent, sub_agent_id=sub, **kw)


class PromptStoreContract:
    def make_store(self, seeds: list[PromptSeed]) -> PromptStore:  # override
        raise NotImplementedError

    @pytest.fixture
    def store(self) -> PromptStore:
        return self.make_store(SEEDS)

    def test_satisfies_protocol(self, store):
        assert isinstance(store, PromptStore)

    def test_returns_prompt_version(self, store):
        pv = store.resolve("persona", ctx())
        assert isinstance(pv, PromptVersion)
        assert pv.name == "persona" and pv.version and pv.template

    def test_most_specific_wins(self, store):
        pv = store.resolve("persona", ctx())
        assert (pv.template, pv.scope) == ("acme-agent1-cs", S.WORKSPACE_AGENT_SUB_AGENT)

    def test_falls_back_to_workspace_sub_agent(self, store):
        pv = store.resolve("persona", ctx(agent="agent_2"))
        assert (pv.template, pv.scope) == ("acme-cs", S.WORKSPACE_SUB_AGENT)

    def test_falls_back_to_platform_sub_agent(self, store):
        pv = store.resolve("persona", ctx(ws="other"))
        assert (pv.template, pv.scope) == ("platform-cs", S.PLATFORM_SUB_AGENT)

    def test_falls_back_to_platform_default(self, store):
        pv = store.resolve("persona", ctx(ws="other", sub="billing"))
        assert (pv.template, pv.scope) == ("platform-default", S.PLATFORM)

    def test_other_workspace_never_sees_override(self, store):
        for sub in ("customer_support", "billing"):
            pv = store.resolve("persona", ctx(ws="other", sub=sub))
            assert not pv.template.startswith("acme")

    def test_unknown_name_raises(self, store):
        with pytest.raises(PromptNotFound):
            store.resolve("does_not_exist", ctx())

    def test_deterministic(self, store):
        assert store.resolve("persona", ctx()) == store.resolve("persona", ctx())

    def test_accepts_run_id_pin(self, store):
        assert isinstance(store.resolve("persona", ctx(run_id="run-123")), PromptVersion)

    def test_node_is_context_not_a_resolution_key(self, store):
        assert store.resolve("persona", ctx()) == store.resolve(
            "persona", ctx(node="triage")
        )