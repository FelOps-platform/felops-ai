"""Tests for Tool Registry v0 (task 2.1, §4.1).

The registry owns the catalog copy and the sub-agent manifests, derives
``E_s = P_s ∩ S`` through ``effective_tools``, and validates call arguments
against each tool's ``parameters`` JSON Schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from felops.domain.agent import AgentRef, SubAgentSpec
from felops.domain.state import AgentConfig, AgentToolConfig, ToolCall
from felops.domain.tools import CatalogSnapshot
from felops.services.tool_registry import ToolRegistry, load_subagent_specs

_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_PATH = _ROOT / "fixtures" / "catalog" / "catalog.json"
_MANIFEST_DIR = _ROOT / "config" / "subagents"
_SUB_AGENT = "customer_support"
_TOOL_IDS = {
    "lookup_order",
    "refund_order",
    "send_invoice",
    "get_invoice",
    "create_case",
    "get_case",
}


def _catalog_payload() -> dict[str, Any]:
    return json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot.model_validate(_catalog_payload())


def _manifest() -> dict[str, SubAgentSpec]:
    return load_subagent_specs(_MANIFEST_DIR)


def _registry() -> ToolRegistry:
    return ToolRegistry(_catalog(), _manifest())


def _config(*tool_ids: str) -> AgentConfig:
    return AgentConfig(tools=[AgentToolConfig(tool_id=tool_id) for tool_id in tool_ids])


def _call(tool_id: str, **arguments: Any) -> ToolCall:
    return ToolCall(tool_id=tool_id, arguments=arguments, idem_key="idem-test")


# ---------------------------------------------------------------------------
# Manifest and catalog loading
# ---------------------------------------------------------------------------


def test_manifest_defines_the_six_predefined_tools() -> None:
    spec = _manifest()[_SUB_AGENT]

    assert spec.audience == "external"
    assert set(spec.predefined_tool_ids) == _TOOL_IDS


def test_every_predefined_tool_exists_in_the_catalog() -> None:
    """A manifest id must be callable, so the catalog must contain it (§4.1)."""
    catalog = _catalog()

    for spec in _manifest().values():
        for tool_id in spec.predefined_tool_ids:
            assert catalog.get(tool_id) is not None, (spec.sub_agent_id, tool_id)


def test_registry_exposes_the_six_catalog_specs() -> None:
    registry = _registry()

    assert {tool.tool_id for tool in registry.catalog.tools} == _TOOL_IDS
    lookup = registry.get("lookup_order")
    assert lookup is not None
    assert lookup.connector_id == "shopify"
    assert registry.get("delete_everything") is None


def test_from_files_loads_the_real_config() -> None:
    registry = ToolRegistry.from_files(_CATALOG_PATH, _MANIFEST_DIR)

    assert registry.get("refund_order") is not None
    assert registry.effective_for(AgentRef(agent_id="agent-1"), _config(*_TOOL_IDS)) == _TOOL_IDS


# ---------------------------------------------------------------------------
# effective_for: E_s = P_s ∩ S
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        (_TOOL_IDS, _TOOL_IDS),
        ({"lookup_order", "refund_order"}, {"lookup_order", "refund_order"}),
        ({"lookup_order", "not_a_tool"}, {"lookup_order"}),
        (set(), set()),
    ],
)
def test_effective_set_is_derived_from_the_selection(
    selected: set[str], expected: set[str]
) -> None:
    registry = _registry()

    assert registry.effective_for(AgentRef(agent_id="agent-1"), _config(*selected)) == expected


def test_unknown_sub_agent_has_no_tools() -> None:
    registry = _registry()
    agent_ref = AgentRef(agent_id="agent-1", sub_agent_id="billing")

    assert registry.effective_for(agent_ref, _config(*_TOOL_IDS)) == frozenset()


def test_audience_mismatch_has_no_tools() -> None:
    """An external agent_ref must not receive an internal sub-agent's tools."""
    registry = _registry()
    agent_ref = AgentRef(agent_id="agent-1", audience="internal")

    assert registry.effective_for(agent_ref, _config(*_TOOL_IDS)) == frozenset()


def test_effective_set_is_intersected_with_the_manifest() -> None:
    """Selection outside P is dropped even when the catalog carries the tool."""
    subagents = {
        _SUB_AGENT: SubAgentSpec(
            sub_agent_id=_SUB_AGENT,
            predefined_tool_ids=("lookup_order", "get_case"),
        )
    }
    registry = ToolRegistry(_catalog(), subagents)
    config = _config("lookup_order", "get_case", "refund_order", "send_invoice")

    assert registry.effective_for(AgentRef(agent_id="agent-1"), config) == {
        "lookup_order",
        "get_case",
    }


def test_deprecated_tool_is_dropped_from_the_effective_set() -> None:
    payload = _catalog_payload()
    for tool in payload["tools"]:
        if tool["tool_id"] == "refund_order":
            tool["deprecated"] = True
    registry = ToolRegistry(CatalogSnapshot.model_validate(payload), _manifest())

    effective = registry.effective_for(AgentRef(agent_id="agent-1"), _config(*_TOOL_IDS))

    assert "refund_order" not in effective
    assert effective == _TOOL_IDS - {"refund_order"}


def test_tool_missing_from_the_catalog_is_dropped() -> None:
    payload = _catalog_payload()
    payload["tools"] = [
        tool for tool in payload["tools"] if tool["tool_id"] != "send_invoice"
    ]
    registry = ToolRegistry(CatalogSnapshot.model_validate(payload), _manifest())

    effective = registry.effective_for(AgentRef(agent_id="agent-1"), _config(*_TOOL_IDS))

    assert "send_invoice" not in effective
    assert len(effective) == 5


def test_apply_catalog_replaces_the_snapshot_atomically() -> None:
    registry = _registry()
    payload = _catalog_payload()
    payload["tools"] = [
        tool for tool in payload["tools"] if tool["tool_id"] != "refund_order"
    ]

    registry.apply_catalog(payload)

    assert registry.get("refund_order") is None
    agent_ref = AgentRef(agent_id="agent-1")
    assert registry.effective_for(agent_ref, _config(*_TOOL_IDS)) == _TOOL_IDS - {
        "refund_order"
    }

    with pytest.raises(ValidationError):
        registry.apply_catalog({})
    assert registry.get("lookup_order") is not None


def test_invalid_json_schema_is_rejected_at_load() -> None:
    payload = _catalog_payload()
    payload["tools"][0]["parameters"] = {"type": "not-a-json-schema-type"}

    with pytest.raises(ValueError, match="invalid JSON Schema"):
        ToolRegistry(CatalogSnapshot.model_validate(payload), _manifest())


# ---------------------------------------------------------------------------
# validate_args
# ---------------------------------------------------------------------------


def test_validate_args_accepts_every_catalog_example() -> None:
    registry = _registry()
    examples = _catalog_payload()["examples"]

    for tool_id, example in examples.items():
        assert registry.validate_args(_call(tool_id, **example["arguments"])) == [], tool_id


def test_validate_args_reports_missing_required_field() -> None:
    errors = _registry().validate_args(_call("lookup_order"))

    assert len(errors) == 1
    assert "order_number" in errors[0]


def test_validate_args_rejects_additional_properties() -> None:
    errors = _registry().validate_args(
        _call("lookup_order", order_number="#1042", surprise="nope")
    )

    assert any("surprise" in error for error in errors)


def test_validate_args_rejects_invalid_pattern_and_enum() -> None:
    registry = _registry()

    pattern_errors = registry.validate_args(_call("get_invoice", invoice_number="nope"))
    enum_errors = registry.validate_args(
        _call("refund_order", order_number="#1042", amount=10.0, reason="because")
    )

    assert any("invoice_number" in error or "pattern" in error for error in pattern_errors)
    assert any("because" in error for error in enum_errors)


def test_validate_args_rejects_invalid_email_formats() -> None:
    """``format: email`` is enforced, not just documented (§4.3 parameters)."""
    registry = _registry()

    send_errors = registry.validate_args(
        _call(
            "send_invoice",
            invoice_number="INV-2210",
            amount=349.0,
            recipient_email="not-an-email",
        )
    )
    case_errors = registry.validate_args(
        _call(
            "create_case",
            subject="Damaged lamp",
            description="Cracked base",
            contact_email="not-an-email",
            priority="high",
        )
    )

    assert any("recipient_email" in error for error in send_errors)
    assert any("contact_email" in error for error in case_errors)


def test_validate_args_rejects_a_deprecated_tool() -> None:
    payload = _catalog_payload()
    for tool in payload["tools"]:
        if tool["tool_id"] == "lookup_order":
            tool["deprecated"] = True
    registry = ToolRegistry(CatalogSnapshot.model_validate(payload), _manifest())

    errors = registry.validate_args(_call("lookup_order", order_number="#1042"))

    assert errors == ["deprecated tool: lookup_order"]


def test_validate_args_rejects_an_unknown_tool() -> None:
    assert _registry().validate_args(_call("delete_everything")) == [
        "unknown tool: delete_everything"
    ]


# ---------------------------------------------------------------------------
# Manifest invariants
# ---------------------------------------------------------------------------


def test_manifest_rejects_duplicate_predefined_tool_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate predefined tool_id"):
        SubAgentSpec(sub_agent_id="customer_support", predefined_tool_ids=("a", "a"))


def test_manifest_requires_predefined_tool_ids() -> None:
    with pytest.raises(ValidationError):
        SubAgentSpec(sub_agent_id="customer_support")


def test_manifest_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        SubAgentSpec(
            sub_agent_id="customer_support",
            predefined_tool_ids=("lookup_order",),
            predefined_tools=["lookup_order"],
        )


def test_missing_manifest_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="manifest directory not found"):
        load_subagent_specs(tmp_path / "missing")


def test_empty_manifest_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no sub-agent manifests"):
        load_subagent_specs(tmp_path)


def test_manifest_path_must_be_a_directory(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_subagent_specs(path)


def test_duplicate_sub_agent_manifests_are_rejected(tmp_path: Path) -> None:
    manifest = {"sub_agent_id": "customer_support", "predefined_tool_ids": ["lookup_order"]}
    (tmp_path / "a.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate sub_agent_id"):
        load_subagent_specs(tmp_path)
