"""Unit tests for the tool value objects (task 0.6).

The Connector port (§5.1) returns ``ToolSpec`` and ``ToolResult``, so these
types ship with the port.  They carry no I/O: validation only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from felops.domain.state import SideEffectClass
from felops.domain.tools import ToolResult, ToolSpec


def _spec(**overrides: object) -> ToolSpec:
    base: dict[str, object] = {
        "action_id": "act-1",
        "name": "shopify_get_order",
        "description": "Look up an order by id",
        "input_schema": {"type": "object"},
        "side_effect_class": SideEffectClass.READ_ONLY,
    }
    base.update(overrides)
    return ToolSpec(**base)  # type: ignore[arg-type]


def test_tool_spec_rejects_empty_action_id() -> None:
    """A tool without an action id cannot be selected or approved."""
    with pytest.raises(ValidationError):
        _spec(action_id="")


def test_tool_spec_rejects_empty_name() -> None:
    """The connector is called by tool name, so it must not be blank."""
    with pytest.raises(ValidationError):
        _spec(name="")


def test_tool_spec_requires_a_side_effect_class() -> None:
    """Defaulting to read-only would skip the idempotency ledger on a write."""
    with pytest.raises(ValidationError):
        ToolSpec(action_id="act-1", name="quickbooks_send_invoice")


def test_tool_result_defaults_to_no_data_and_no_error() -> None:
    result = ToolResult(ok=True)

    assert result.data == {}
    assert result.error is None
