"""MemoryStore protocol — §5.1.

The single swap point for conversation and customer memory.  Adapters:
Postgres (P4), Redis, vector-backed.

Workspace and thread scoping (§2.3, §4, task 7.9):

* Workspace and thread boundaries are strict: reads never cross workspaces and
  never mix long-term memory (``thread_id=None``) with thread-scoped memory.
  Callers that need both compose two reads.
* Long-term memory is workspace-scoped: items written with ``agent_id=None``
  are readable from any agent in that workspace (as well as workspace-wide
  reads with ``agent_id=None``).  Items written with a specific ``agent_id``
  remain isolated to that agent.

``write`` is an upsert by ``MemoryItem.item_id`` so retried writes of the same
item cannot create duplicate records.
"""

from __future__ import annotations

from typing import Protocol

from felops.domain.memory import MemoryItem, MemoryScope


class MemoryStore(Protocol):
    """Async memory port.

    Methods
    -------
    `read`
        Return items visible under *scope*, narrowed or ranked by *q* when it
        is given.  ``q=None`` returns all items visible under *scope*.  An
        adapter may rank by similarity or recency, and never crosses workspace
        or thread boundaries.  When ``scope.agent_id`` is set, workspace-wide
        items (written with ``agent_id=None`` for the same workspace and
        ``thread_id``) are also visible.
    `write`
        Store *items* under *scope*, replacing any existing item in that exact
        scope with the same ``item_id``.
    """

    async def read(self, scope: MemoryScope, q: str | None) -> list[MemoryItem]: ...

    async def write(self, scope: MemoryScope, items: list[MemoryItem]) -> None: ...
