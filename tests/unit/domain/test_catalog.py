"""Tests for the catalog snapshot loader (task 1.12, §4.1/§4.3).

The backend pushes one full snapshot, ``{"tools": [...]}``, and the AI service
keeps a read-only copy.  The fixture also carries ``examples`` and
``failure_example`` for the demo/contract tests; those are not part of the
pushed payload and must be ignored by the loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from felops.domain.tools import CatalogSnapshot, ToolSpec

_CATALOG_FIXTURE = (
    Path(__file__).resolve().parents[3] / "fixtures" / "catalog" / "catalog.json"
)
_TOOL_IDS = {
    "lookup_order",
    "refund_order",
    "send_invoice",
    "get_invoice",
    "create_case",
    "get_case",
}
_READ_ONLY = {"lookup_order", "get_invoice", "get_case"}
_SIDE_EFFECTING = {"refund_order", "send_invoice", "create_case"}


def _fixture_payload() -> dict[str, Any]:
    return json.loads(_CATALOG_FIXTURE.read_text(encoding="utf-8"))


def _pushed_payload() -> dict[str, Any]:
    """Return the exact payload the backend pushes (fixture keys dropped)."""
    return {"tools": _fixture_payload()["tools"]}


def _entry(tool_id: str = "lookup_order", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "tool_id": tool_id,
        "connector_id": "shopify",
        "description": "Read an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_number": {"type": "string"}},
            "required": ["order_number"],
            "additionalProperties": False,
        },
        "result_summary": "Order {order_number} · {status}",
        "effect_summary": None,
        "version": 1,
        "deprecated": False,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The six specs load
# ---------------------------------------------------------------------------


def test_six_specs_load_from_the_pushed_snapshot() -> None:
    snapshot = CatalogSnapshot.model_validate(_pushed_payload())

    assert len(snapshot.tools) == 6
    assert {tool.tool_id for tool in snapshot.tools} == _TOOL_IDS
    assert snapshot.get("lookup_order") is not None
    assert snapshot.get("delete_everything") is None


def test_fixture_only_keys_are_ignored() -> None:
    """``examples``/``failure_example`` never reach the loader's contract."""
    snapshot = CatalogSnapshot.model_validate(_fixture_payload())

    assert len(snapshot.tools) == 6
    assert set(snapshot.model_dump()) == {"tools"}
    assert snapshot == CatalogSnapshot.model_validate(_pushed_payload())


def test_each_entry_carries_an_input_json_schema() -> None:
    snapshot = CatalogSnapshot.model_validate(_pushed_payload())

    for tool in snapshot.tools:
        assert tool.parameters["type"] == "object", tool.tool_id
        assert tool.parameters["additionalProperties"] is False, tool.tool_id
        required = tool.parameters["required"]
        assert required, tool.tool_id
        assert set(required) <= set(tool.parameters["properties"]), tool.tool_id


# ---------------------------------------------------------------------------
# Side-effect class and the backend-owned verifier
# ---------------------------------------------------------------------------


def test_side_effect_class_is_derived_from_effect_summary() -> None:
    snapshot = CatalogSnapshot.model_validate(_pushed_payload())

    by_id = {tool.tool_id: tool for tool in snapshot.tools}
    for tool_id in _READ_ONLY:
        assert by_id[tool_id].effect_summary is None
        assert by_id[tool_id].is_side_effecting is False
    for tool_id in _SIDE_EFFECTING:
        assert by_id[tool_id].effect_summary
        assert by_id[tool_id].is_side_effecting is True


def test_catalog_carries_neither_a_verifier_nor_an_approval_flag() -> None:
    """Verification is backend-owned and approval is per (agent, tool)."""
    assert "verifier" not in ToolSpec.model_fields
    assert "requires_approval" not in ToolSpec.model_fields

    entry = _entry(verifier={"tool_id": "lookup_order"}, requires_approval=True)
    snapshot = CatalogSnapshot.model_validate({"tools": [entry]})
    tool = snapshot.get("lookup_order")

    assert tool is not None
    assert "verifier" not in tool.model_dump()
    assert "requires_approval" not in tool.model_dump()


# ---------------------------------------------------------------------------
# Snapshot invariants
# ---------------------------------------------------------------------------


def test_missing_tools_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CatalogSnapshot.model_validate({})
    with pytest.raises(ValidationError):
        CatalogSnapshot.model_validate({"tools": None})


def test_empty_catalog_is_valid() -> None:
    assert CatalogSnapshot.model_validate({"tools": []}).tools == ()


def test_duplicate_tool_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate tool_id"):
        CatalogSnapshot.model_validate({"tools": [_entry(), _entry()]})


def test_malformed_entry_is_rejected() -> None:
    with pytest.raises(ValidationError):
        CatalogSnapshot.model_validate({"tools": [_entry(connector_id="")]})
    with pytest.raises(ValidationError):
        CatalogSnapshot.model_validate({"tools": [{"tool_id": "x"}]})


def test_partial_entries_are_rejected() -> None:
    """A truncated or summary-mode push must fail closed, not default read-only."""
    summary_mode = {
        "tool_id": "lookup_order",
        "connector_id": "shopify",
        "version": 1,
        "deprecated": False,
    }
    missing_effect = {key: value for key, value in _entry().items() if key != "effect_summary"}
    missing_parameters = {key: value for key, value in _entry().items() if key != "parameters"}

    with pytest.raises(ValidationError, match="missing required field"):
        CatalogSnapshot.model_validate({"tools": [summary_mode]})
    with pytest.raises(ValidationError, match="effect_summary"):
        CatalogSnapshot.model_validate({"tools": [missing_effect]})
    with pytest.raises(ValidationError, match="parameters"):
        CatalogSnapshot.model_validate({"tools": [missing_parameters]})


def test_deprecated_entries_are_preserved() -> None:
    """The backend deprecates instead of deleting; ids are never reused."""
    snapshot = CatalogSnapshot.model_validate(
        {"tools": [_entry("refund_order", deprecated=True)]}
    )

    tool = snapshot.get("refund_order")
    assert tool is not None
    assert tool.deprecated is True


def test_unknown_tool_fields_are_ignored() -> None:
    entry = _entry(mcp_tool="lookup_order", future_field={"anything": True})

    snapshot = CatalogSnapshot.model_validate({"tools": [entry]})
    tool = snapshot.get("lookup_order")

    assert tool is not None
    assert "mcp_tool" not in tool.model_dump()
    assert "future_field" not in tool.model_dump()
