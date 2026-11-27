"""Offline tests for the Postgres ledger adapter (task 3.6, §4.3).

No database is needed: a fake pool pins SQL parameter order, first-writer-wins
parsing, pool ownership and the bounded-timeout behaviour.  The live round trip
lives in ``test_postgres_ledger_live.py`` and runs when ``DATABASE_URL`` is set.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import felops.adapters.ledger.postgres as postgres_module
from felops.adapters.ledger.postgres import PostgresLedger
from felops.domain.tools import ToolError, ToolResult


class FakePool:
    def __init__(
        self,
        *,
        row: dict[str, Any] | None = None,
        status: str = "INSERT 0 1",
        delay: float = 0.0,
    ) -> None:
        self.row = row
        self.status = status
        self.delay = delay
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        await asyncio.sleep(self.delay)
        self.calls.append((query, args))
        return self.row

    async def execute(self, query: str, *args: Any) -> str:
        await asyncio.sleep(self.delay)
        self.calls.append((query, args))
        return self.status

    async def close(self) -> None:
        self.closed = True


def _success(refund_id: str = "RFD-3081") -> ToolResult:
    return ToolResult(status="succeeded", data={"refund_id": refund_id})


# ---------------------------------------------------------------------------
# Read and write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_returns_none_for_a_missing_row() -> None:
    ledger = PostgresLedger(FakePool(row=None))

    assert await ledger.get("ws-1", "idem-1") is None


@pytest.mark.asyncio
async def test_get_parses_the_stored_envelope() -> None:
    result = _success()
    ledger = PostgresLedger(FakePool(row={"result": result.model_dump_json()}))

    assert await ledger.get("ws-1", "idem-1") == result


@pytest.mark.asyncio
async def test_put_forwards_workspace_key_and_json() -> None:
    pool = FakePool(status="INSERT 0 1")
    ledger = PostgresLedger(pool)

    stored = await ledger.put("ws-1", "idem-1", _success())

    assert stored is True
    query, args = pool.calls[0]
    assert "ON CONFLICT" in query
    assert args[0] == "ws-1"
    assert args[1] == "idem-1"
    assert json.loads(args[2])["data"]["refund_id"] == "RFD-3081"


@pytest.mark.asyncio
async def test_put_is_first_writer_wins() -> None:
    ledger = PostgresLedger(FakePool(status="INSERT 0 0"))

    assert await ledger.put("ws-1", "idem-1", _success()) is False


@pytest.mark.asyncio
async def test_failed_envelope_round_trips() -> None:
    failure = ToolResult(
        status="failed",
        data=None,
        error=ToolError(code="provider_error", message="boom", retryable=False),
    )
    ledger = PostgresLedger(FakePool(row={"result": failure.model_dump_json()}))

    assert await ledger.get("ws-1", "idem-1") == failure


# ---------------------------------------------------------------------------
# Schema, ownership and timeouts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_schema_creates_the_table() -> None:
    pool = FakePool()
    ledger = PostgresLedger(pool)

    await ledger.ensure_schema()

    assert "CREATE TABLE IF NOT EXISTS tool_execution_ledger" in pool.calls[0][0]


@pytest.mark.asyncio
async def test_aclose_only_closes_an_owned_pool() -> None:
    injected = FakePool()
    owned = FakePool()

    await PostgresLedger(injected).aclose()
    await PostgresLedger(owned, owns_pool=True).aclose()

    assert injected.closed is False
    assert owned.closed is True


@pytest.mark.asyncio
async def test_operations_time_out() -> None:
    ledger = PostgresLedger(FakePool(delay=0.1), timeout=0.01)

    with pytest.raises(TimeoutError):
        await ledger.get("ws-1", "idem-1")


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError):
        PostgresLedger(FakePool(), timeout=0)


@pytest.mark.asyncio
async def test_connect_closes_the_pool_when_schema_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool()

    async def fake_create_pool(dsn: str, **kwargs: Any) -> FakePool:
        return pool

    async def failing_schema(self: PostgresLedger) -> None:
        raise RuntimeError("schema failed")

    monkeypatch.setattr(postgres_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(PostgresLedger, "ensure_schema", failing_schema)

    with pytest.raises(RuntimeError, match="schema failed"):
        await PostgresLedger.connect("postgresql://test")

    assert pool.closed is True


@pytest.mark.asyncio
async def test_connect_returns_an_owned_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    pool = FakePool()
    seen: dict[str, Any] = {}

    async def fake_create_pool(dsn: str, **kwargs: Any) -> FakePool:
        seen.update(kwargs)
        return pool

    async def noop_schema(self: PostgresLedger) -> None:
        return None

    monkeypatch.setattr(postgres_module.asyncpg, "create_pool", fake_create_pool)
    monkeypatch.setattr(PostgresLedger, "ensure_schema", noop_schema)

    ledger = await PostgresLedger.connect("postgresql://test", timeout=2.5)

    assert seen["timeout"] == 2.5
    assert seen["command_timeout"] == 2.5
    await ledger.aclose()
    assert pool.closed is True
