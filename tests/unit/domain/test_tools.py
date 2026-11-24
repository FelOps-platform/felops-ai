"""Unit tests for the tool value objects (task 0.6).

The Connector port (§5.1) returns ``ToolSpec`` and ``ToolResult``, so these
types ship with the port.  Validation covers the §4.3 catalog shape and the
§4.1–§4.2 structured execution envelope.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from felops.domain.tools import ToolError, ToolResult, ToolSpec

_CATALOG_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "catalog" / "catalog.json"


def _spec(**overrides: object) -> ToolSpec:
    base: dict[str, object] = {
        "tool_id": "lookup_order",
        "connector_id": "shopify",
        "description": "Read an order by order number.",
        "parameters": {"type": "object"},
        "result_summary": "Order {order_number} · {status}",
        "effect_summary": None,
        "version": 1,
        "deprecated": False,
    }
    base.update(overrides)
    return ToolSpec(**base)  # type: ignore[arg-type]


def test_tool_spec_rejects_empty_tool_id() -> None:
    """tool_id is the single stable identifier across catalog and runtime."""
    with pytest.raises(ValidationError):
        _spec(tool_id="")


def test_tool_spec_rejects_empty_connector_id() -> None:
    """Every catalog tool belongs to a backend connector."""
    with pytest.raises(ValidationError):
        _spec(connector_id="")


def test_tool_spec_rejects_empty_effect_summary() -> None:
    """Read-only tools use None; empty string must not masquerade as a template."""
    with pytest.raises(ValidationError):
        _spec(effect_summary="")


def test_tool_spec_rejects_non_positive_version() -> None:
    with pytest.raises(ValidationError):
        _spec(version=0)


def test_tool_spec_infers_side_effecting_from_effect_summary() -> None:
    read_spec = _spec(effect_summary=None)
    write_spec = _spec(
        tool_id="send_invoice",
        connector_id="quickbooks",
        effect_summary="Send invoice {invoice_number} for AED {amount} to {recipient_email}",
    )

    assert read_spec.is_side_effecting is False
    assert write_spec.is_side_effecting is True


def test_tool_spec_and_result_validate_catalog_fixture() -> None:
    """All 6 catalog entries and example results match the domain schema."""
    catalog = json.loads(_CATALOG_FIXTURE.read_text(encoding="utf-8"))

    specs = [ToolSpec.model_validate(entry) for entry in catalog["tools"]]
    assert len(specs) == 6
    assert {spec.tool_id for spec in specs} == {
        "lookup_order",
        "refund_order",
        "send_invoice",
        "get_invoice",
        "create_case",
        "get_case",
    }

    for example in catalog["examples"].values():
        result = ToolResult.model_validate(example["result"])
        assert result.status == "succeeded"
        assert result.error is None
        assert result.data is not None

    failure = ToolResult.model_validate(catalog["failure_example"]["result"])
    assert failure.status == "failed"
    assert failure.data is None
    assert failure.error is not None
    assert failure.error.code == "provider_error"
    assert failure.error.retryable is False


def test_tool_result_succeeded_defaults() -> None:
    result = ToolResult(status="succeeded", data={"order_number": "#1042"})

    assert result.status == "succeeded"
    assert result.data == {"order_number": "#1042"}
    assert result.error is None


def test_tool_result_succeeded_rejects_error() -> None:
    with pytest.raises(ValidationError):
        ToolResult(
            status="succeeded",
            data={},
            error=ToolError(code="provider_error", message="boom"),
        )


def test_tool_result_failed_requires_structured_error() -> None:
    with pytest.raises(ValidationError):
        ToolResult(status="failed")


def test_tool_result_failed_preserves_error_fields() -> None:
    err = ToolError(
        code="connector_not_connected",
        message="QuickBooks connector is not connected for this workspace.",
        retryable=False,
    )
    result = ToolResult(status="failed", error=err)

    assert result.status == "failed"
    assert result.data is None
    assert result.error == err


def test_tool_error_rejects_empty_code_or_message() -> None:
    with pytest.raises(ValidationError):
        ToolError(code="", message="something failed")
    with pytest.raises(ValidationError):
        ToolError(code="provider_error", message="")
