"""MemoryStore protocol — §5.1.

The single swap point for conversation and customer memory.  Adapters:
Postgres (P4), Redis, vector-backed.

Workspace and thread scoping (§2.3, §4, task 7.9):

* Workspace and thread boundaries are strict: reads never cross workspaces and
  never mix long-term memory (``thread_id=None``) with thread-scoped memory.
  Callers that need both compose two reads.
* Long-term memory is workspace-scoped: items written with ``agent_id=None``
  are readable from any agent in that workspace (as well as workspace-wide
  reads with ``agent_id=None``).  Workspace-wide reads (``agent_id=None``)
  return **only** items written with ``agent_id=None`` (for that
  ``workspace_id`` and ``thread_id``), never agent-scoped items.  Items
  written with a specific ``agent_id`` remain isolated to that agent.

``write`` is an upsert by the composite key ``(scope, item.item_id)`` (i.e.
``(workspace_id, agent_id, thread_id, item_id)``) so retried writes of the
same item cannot create duplicate records, and identical ``item_id``s
(derived or explicit) in different scopes never overwrite each other.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from felops.domain.memory import MemoryItem, MemoryScope


@runtime_checkable
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
        ``thread_id``) are also visible.  When ``scope.agent_id`` is ``None``,
        ``read`` returns **only** workspace-wide items (written with
        ``agent_id=None`` for the same ``workspace_id`` and ``thread_id``);
        agent-scoped items are not included.
    `write`
        Store *items* under *scope*, upserting by the composite key
        ``(scope, item_id)`` (replacing any existing item in that exact scope
        with the same ``item_id`` without affecting other scopes).
    """

    async def read(self, scope: MemoryScope, q: str | None) -> list[MemoryItem]: ...

    async def write(self, scope: MemoryScope, items: list[MemoryItem]) -> None: ...
