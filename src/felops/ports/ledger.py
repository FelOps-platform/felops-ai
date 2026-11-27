"""Idempotency ledger port (§4 Executor, §4.3).

The executor uses the ledger to avoid re-executing a **side-effecting** call
when the same ``idem_key`` is delivered again (retry, replay after a restart,
duplicate job delivery).  Reads are keyed too, but run live every time and are
never read from this ledger, so a cached result can never serve stale data.

Implementations must be first-writer-wins: ``put`` returns ``False`` when a
terminal result already exists and must not overwrite it.
"""

from __future__ import annotations

from typing import Protocol

from felops.domain.tools import ToolResult


class IdempotencyLedger(Protocol):
    async def get(self, workspace_id: str, idem_key: str) -> ToolResult | None:
        """Return the stored terminal result, or ``None`` when unseen."""
        ...

    async def put(
        self, workspace_id: str, idem_key: str, result: ToolResult
    ) -> bool:
        """Store a terminal result; return ``False`` when one already existed."""
        ...
