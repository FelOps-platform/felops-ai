"""Tests for the in-memory idempotency ledger (§4 Executor, §4.3)."""

from __future__ import annotations

import pytest

from felops.adapters.ledger.memory import InMemoryLedger
from felops.domain.tools import ToolError, ToolResult


def _success(data: dict[str, object]) -> ToolResult:
    return ToolResult(status="succeeded", data=data, error=None)


@pytest.mark.asyncio
async def test_put_then_get_round_trips() -> None:
    ledger = InMemoryLedger()
    result = _success({"refund_id": "RFD-3081"})

    assert await ledger.put("ws-1", "idem-1", result) is True
    assert await ledger.get("ws-1", "idem-1") == result


@pytest.mark.asyncio
async def test_unknown_key_returns_none() -> None:
    assert await InMemoryLedger().get("ws-1", "missing") is None


@pytest.mark.asyncio
async def test_first_writer_wins() -> None:
    ledger = InMemoryLedger()
    first = _success({"refund_id": "RFD-1"})
    second = _success({"refund_id": "RFD-2"})

    assert await ledger.put("ws-1", "idem-1", first) is True
    assert await ledger.put("ws-1", "idem-1", second) is False
    assert await ledger.get("ws-1", "idem-1") == first


@pytest.mark.asyncio
async def test_entries_are_isolated_by_workspace() -> None:
    ledger = InMemoryLedger()
    result = _success({"refund_id": "RFD-1"})

    await ledger.put("ws-1", "idem-1", result)

    assert await ledger.get("ws-2", "idem-1") is None


@pytest.mark.asyncio
async def test_failed_results_are_stored_too() -> None:
    ledger = InMemoryLedger()
    failure = ToolResult(
        status="failed",
        data=None,
        error=ToolError(code="provider_error", message="boom", retryable=False),
    )

    await ledger.put("ws-1", "idem-1", failure)

    assert await ledger.get("ws-1", "idem-1") == failure
