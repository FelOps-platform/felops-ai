"""MemoryStore protocol — §5.1.

The single swap point for conversation and customer memory.  Adapters:
Postgres (P4), Redis, vector-backed.

The scope is the entire partition key (§2.3): items written under one
workspace, agent or thread are never returned under another.  ``read`` never
widens the scope to be helpful; callers that need long-term and thread memory
issue two reads with two scopes.

``write`` is an upsert by ``MemoryItem.item_id`` so a retried write cannot
duplicate a fact.
"""

from __future__ import annotations

from typing import Protocol

from felops.domain.memory import MemoryItem, MemoryScope


class MemoryStore(Protocol):
    """Async memory port.

    Methods
    -------
    `read`
        Return the items stored under exactly *scope*, narrowed by *q* when
        it is given.  ``q=None`` means "everything in scope"; an adapter may
        rank by similarity or recency, but never crosses the scope.
    `write`
        Store *items* under *scope*, replacing any existing item with the
        same ``item_id``.
    """

    async def read(self, scope: MemoryScope, q: str | None) -> list[MemoryItem]: ...

    async def write(self, scope: MemoryScope, items: list[MemoryItem]) -> None: ...
