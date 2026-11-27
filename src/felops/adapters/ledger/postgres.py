"""Postgres idempotency ledger (§4 Executor, §4.3).

One row per ``(workspace_id, idem_key)``; the primary key makes the insert
first-writer-wins and safe across processes.  Results are stored as ``jsonb``
so the terminal envelope can be replayed exactly.

The executor only consults the ledger for side-effecting calls; reads run live.
Every pool acquisition and query is bounded by ``timeout`` so a stalled
database hits the executor's fail-open path instead of blocking a run.
"""

from __future__ import annotations

import asyncio

import asyncpg

from felops.domain.tools import ToolResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_execution_ledger (
    workspace_id text        NOT NULL,
    idem_key     text        NOT NULL,
    result       jsonb       NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, idem_key)
)
"""

_GET = """
SELECT result::text AS result
FROM tool_execution_ledger
WHERE workspace_id = $1 AND idem_key = $2
"""

_PUT = """
INSERT INTO tool_execution_ledger (workspace_id, idem_key, result)
VALUES ($1, $2, $3::jsonb)
ON CONFLICT (workspace_id, idem_key) DO NOTHING
"""


class PostgresLedger:
    """Durable first-writer-wins ledger backed by an asyncpg pool."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        timeout: float = 5.0,
        owns_pool: bool = False,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._pool = pool
        self._timeout = timeout
        self._owns_pool = owns_pool

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 5,
        timeout: float = 5.0,
    ) -> PostgresLedger:
        """Open a pool and ensure the ledger table exists."""
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            command_timeout=timeout,
        )
        ledger = cls(pool, timeout=timeout, owns_pool=True)
        try:
            await ledger.ensure_schema()
        except BaseException:
            await pool.close()
            raise
        return ledger

    async def ensure_schema(self) -> None:
        async with asyncio.timeout(self._timeout):
            await self._pool.execute(_SCHEMA)

    async def get(self, workspace_id: str, idem_key: str) -> ToolResult | None:
        async with asyncio.timeout(self._timeout):
            row = await self._pool.fetchrow(_GET, workspace_id, idem_key)
        if row is None:
            return None
        return ToolResult.model_validate_json(row["result"])

    async def put(
        self, workspace_id: str, idem_key: str, result: ToolResult
    ) -> bool:
        async with asyncio.timeout(self._timeout):
            status = await self._pool.execute(
                _PUT, workspace_id, idem_key, result.model_dump_json()
            )
        return status == "INSERT 0 1"

    async def aclose(self) -> None:
        """Close the pool only when this ledger created it."""
        if self._owns_pool:
            await self._pool.close()


__all__ = ["PostgresLedger"]
