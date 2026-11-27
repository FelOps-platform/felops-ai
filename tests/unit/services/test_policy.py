"""Tests for Policy Engine v0 (task 2.3, §4 Policy).

The allowlist is the active sub-agent's effective tool set, so this suite
covers each way a call can fall outside it: deselected, unknown to the
catalog, deprecated, wrong sub-agent or wrong audience.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from felops.domain.agent import AgentRef, SubAgentSpec
from felops.domain.policy import PolicyDecision
from felops.domain.state import AgentConfig, AgentToolConfig, ToolCall
from felops.domain.tools import CatalogSnapshot
from felops.services.policy import PolicyEngine
from felops.services.tool_registry import ToolRegistry, load_subagent_specs

_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_PATH = _ROOT / "fixtures" / "catalog" / "catalog.json"
_MANIFEST_DIR = _ROOT / "config" / "subagents"
_TOOL_IDS = (
    "lookup_order",
    "refund_order",
    "send_invoice",
    "get_invoice",
    "create_case",
    "get_case",
)


def _catalog_payload() -> dict[str, Any]:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _registry() -> ToolRegistry:
    return ToolRegistry(
        CatalogSnapshot.model_validate(_catalog_payload()),
        load_subagent_specs(_MANIFEST_DIR),
    )


def _config(*tool_ids: str, approval: str | tuple[str, ...] = ()) -> AgentConfig:
    approved = {approval} if isinstance(approval, str) else set(approval)
    unknown = approved - set(tool_ids)
    if unknown:
        raise ValueError(f"approval ids must also be selected: {sorted(unknown)}")
    return AgentConfig(
        tools=[
            AgentToolConfig(tool_id=tool_id, requires_approval=tool_id in approved)
            for tool_id in tool_ids
        ]
    )


def _call(tool_id: str) -> ToolCall:
    return ToolCall(tool_id=tool_id, idem_key="idem-policy")


_AGENT_REF = AgentRef(agent_id="agent-1")


def test_call_inside_the_effective_set_is_allowed() -> None:
    verdict = PolicyEngine(_registry()).evaluate(
        _call("lookup_order"), _AGENT_REF, _config(*_TOOL_IDS)
    )

    assert verdict.decision is PolicyDecision.ALLOW
    assert verdict.reason == "in_effective_set"


def test_deselected_tool_is_denied() -> None:
    verdict = PolicyEngine(_registry()).evaluate(
        _call("refund_order"), _AGENT_REF, _config("lookup_order", "get_invoice")
    )

    assert verdict.decision is PolicyDecision.DENY
    assert verdict.reason == "out_of_effective_set"


def test_unknown_tool_is_denied() -> None:
    verdict = PolicyEngine(_registry()).evaluate(
        _call("delete_everything"), _AGENT_REF, _config(*_TOOL_IDS)
    )

    assert verdict.decision is PolicyDecision.DENY


def test_tool_outside_the_manifest_is_denied() -> None:
    agent_ref = AgentRef(agent_id="agent-1", sub_agent_id="billing")

    verdict = PolicyEngine(_registry()).evaluate(
        _call("refund_order"), agent_ref, _config(*_TOOL_IDS)
    )

    assert verdict.decision is PolicyDecision.DENY


def test_deprecated_tool_is_denied() -> None:
    payload = _catalog_payload()
    for tool in payload["tools"]:
        if tool["tool_id"] == "refund_order":
            tool["deprecated"] = True
    registry = ToolRegistry(
        CatalogSnapshot.model_validate(payload), load_subagent_specs(_MANIFEST_DIR)
    )

    verdict = PolicyEngine(registry).evaluate(
        _call("refund_order"), _AGENT_REF, _config(*_TOOL_IDS)
    )

    assert verdict.decision is PolicyDecision.DENY


def test_audience_mismatch_is_denied() -> None:
    agent_ref = AgentRef(agent_id="agent-1", audience="internal")

    verdict = PolicyEngine(_registry()).evaluate(
        _call("lookup_order"), agent_ref, _config(*_TOOL_IDS)
    )

    assert verdict.decision is PolicyDecision.DENY


def test_every_selected_tool_is_allowed() -> None:
    engine = PolicyEngine(_registry())

    decisions = {
        tool_id: engine.evaluate(_call(tool_id), _AGENT_REF, _config(*_TOOL_IDS)).decision
        for tool_id in _TOOL_IDS
    }

    assert set(decisions.values()) == {PolicyDecision.ALLOW}


# ---------------------------------------------------------------------------
# requires_approval (task 3.3)
# ---------------------------------------------------------------------------


def test_requires_approval_returns_require_approval() -> None:
    verdict = PolicyEngine(_registry()).evaluate(
        _call("refund_order"),
        _AGENT_REF,
        _config(*_TOOL_IDS, approval="refund_order"),
    )

    assert verdict.decision is PolicyDecision.REQUIRE_APPROVAL
    assert verdict.reason == "requires_approval"


def test_requires_approval_does_not_affect_other_tools() -> None:
    engine = PolicyEngine(_registry())
    config = _config(*_TOOL_IDS, approval="refund_order")

    lookup = engine.evaluate(_call("lookup_order"), _AGENT_REF, config)
    refund = engine.evaluate(_call("refund_order"), _AGENT_REF, config)

    assert lookup.decision is PolicyDecision.ALLOW
    assert refund.decision is PolicyDecision.REQUIRE_APPROVAL


def test_removing_the_tool_revokes_its_approval() -> None:
    """Removing a tool from S removes it from E, so a pending approval is void."""
    engine = PolicyEngine(_registry())
    approved = _config(*_TOOL_IDS, approval="refund_order")
    revoked = _config("lookup_order")

    assert (
        engine.evaluate(_call("refund_order"), _AGENT_REF, approved).decision
        is PolicyDecision.REQUIRE_APPROVAL
    )
    verdict = engine.evaluate(_call("refund_order"), _AGENT_REF, revoked)
    assert verdict.decision is PolicyDecision.DENY
    assert verdict.reason == "out_of_effective_set"


def test_allowlist_wins_over_approval_for_a_deprecated_tool() -> None:
    payload = _catalog_payload()
    for tool in payload["tools"]:
        if tool["tool_id"] == "refund_order":
            tool["deprecated"] = True
    registry = ToolRegistry(
        CatalogSnapshot.model_validate(payload), load_subagent_specs(_MANIFEST_DIR)
    )

    verdict = PolicyEngine(registry).evaluate(
        _call("refund_order"),
        _AGENT_REF,
        _config(*_TOOL_IDS, approval="refund_order"),
    )

    assert verdict.decision is PolicyDecision.DENY
    assert verdict.reason == "out_of_effective_set"


def test_approval_applies_to_every_sub_agent_holding_the_tool() -> None:
    """requires_approval is per (agent, tool), not per sub-agent (§4.1 inv. 5)."""
    subagents = load_subagent_specs(_MANIFEST_DIR)
    subagents["billing"] = SubAgentSpec(
        sub_agent_id="billing",
        predefined_tool_ids=("lookup_order", "refund_order"),
    )
    engine = PolicyEngine(
        ToolRegistry(CatalogSnapshot.model_validate(_catalog_payload()), subagents)
    )
    config = _config(*_TOOL_IDS, approval="refund_order")

    for sub_agent_id in ("customer_support", "billing"):
        verdict = engine.evaluate(
            _call("refund_order"),
            AgentRef(agent_id="agent-1", sub_agent_id=sub_agent_id),
            config,
        )
        assert verdict.decision is PolicyDecision.REQUIRE_APPROVAL, sub_agent_id
        assert verdict.reason == "requires_approval"


def test_requires_approval_is_ignored_when_the_sub_agent_lacks_the_tool() -> None:
    subagents = load_subagent_specs(_MANIFEST_DIR)
    subagents["customer_support"] = subagents["customer_support"].model_copy(
        update={"predefined_tool_ids": ("lookup_order",)}
    )
    registry = ToolRegistry(
        CatalogSnapshot.model_validate(_catalog_payload()), subagents
    )

    verdict = PolicyEngine(registry).evaluate(
        _call("refund_order"),
        _AGENT_REF,
        _config(*_TOOL_IDS, approval="refund_order"),
    )

    assert verdict.decision is PolicyDecision.DENY
