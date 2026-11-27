"""Live Postgres ledger tests (§4 Executor, §4.3).

Skipped unless ``DATABASE_URL`` points at a Postgres instance (the compose
stack provides one).  Run with::

    $env:DATABASE_URL = "postgresql://felops:felops_dev@localhost:5432/felops"

Each test uses a unique workspace/key so runs never interfere.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from felops.adapters.ledger.postgres import PostgresLedger
from felops.domain.tools import ToolError, ToolResult

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"), reason="DATABASE_URL is not set"
)


@pytest_asyncio.fixture()
async def ledger() -> AsyncIterator[PostgresLedger]:
    ledger = await PostgresLedger.connect(os.environ["DATABASE_URL"])
    try:
        yield ledger
    finally:
        await ledger.aclose()


def _success(refund_id: str) -> ToolResult:
    return ToolResult(
        status="succeeded", data={"refund_id": refund_id, "status": "completed"}
    )


@pytest.mark.asyncio
async def test_round_trip_and_first_writer_wins(ledger: PostgresLedger) -> None:
    workspace = f"ws-test-{uuid.uuid4().hex}"
    key = uuid.uuid4().hex
    first = _success("RFD-1")
    second = _success("RFD-2")

    assert await ledger.get(workspace, key) is None
    assert await ledger.put(workspace, key, first) is True
    assert await ledger.put(workspace, key, second) is False

    assert await ledger.get(workspace, key) == first


@pytest.mark.asyncio
async def test_entries_are_isolated_by_workspace(ledger: PostgresLedger) -> None:
    key = uuid.uuid4().hex
    await ledger.put(f"ws-a-{uuid.uuid4().hex}", key, _success("RFD-1"))

    assert await ledger.get(f"ws-b-{uuid.uuid4().hex}", key) is None


@pytest.mark.asyncio
async def test_failed_envelope_round_trips(ledger: PostgresLedger) -> None:
    workspace = f"ws-test-{uuid.uuid4().hex}"
    key = uuid.uuid4().hex
    failure = ToolResult(
        status="failed",
        data=None,
        error=ToolError(code="provider_error", message="boom", retryable=True),
    )

    await ledger.put(workspace, key, failure)

    assert await ledger.get(workspace, key) == failure
