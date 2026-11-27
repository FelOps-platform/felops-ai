"""In-memory idempotency ledger (§4 Executor, §4.3).

Process-local and non-durable: intended for tests, the demo and single-process
runs.  Postgres is the durable implementation.
"""

from __future__ import annotations

import asyncio

from felops.domain.tools import ToolResult


class InMemoryLedger:
    """First-writer-wins ledger keyed by (workspace_id, idem_key)."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], ToolResult] = {}
        self._lock = asyncio.Lock()

    async def get(self, workspace_id: str, idem_key: str) -> ToolResult | None:
        async with self._lock:
            return self._entries.get((workspace_id, idem_key))

    async def put(
        self, workspace_id: str, idem_key: str, result: ToolResult
    ) -> bool:
        async with self._lock:
            key = (workspace_id, idem_key)
            if key in self._entries:
                return False
            self._entries[key] = result
            return True


__all__ = ["InMemoryLedger"]
