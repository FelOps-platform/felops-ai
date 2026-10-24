"""VectorStore protocol — §5.1.

The single swap point for vector storage.  Adapters: Qdrant (P1),
pgvector (P10), Weaviate, Pinecone.

Workspace scoping (§2.3): every ``search`` and ``delete`` is scoped by the
``NameSpace`` argument.  ``NameSpace.workspace_id`` is the single source of
truth; ``Filter`` carries payload conditions only, and a ``workspace_id``
key inside a filter is not part of the contract.
"""

from __future__ import annotations

from typing import Protocol

from felops.domain.retrieval_types import Filter, Hit, HybridQuery, VectorItem
from felops.domain.workspace import NameSpace


class VectorStore(Protocol):
    """Async vector-store port.

    Methods
    -------
    `upsert`
        Store (or replace) items under the given namespace.
    `search`
        Hybrid search scoped to ``ns``, with payload filtering.
    `delete`
        Remove items matching *flt* under ``ns``; return how many were
        removed.  The count is scoped by both ``ns`` and ``flt`` and is
        ``0`` when nothing matches, so repeated deletes are idempotent.
        It is best-effort under concurrent writes; stores that cannot count
        synchronously may run a follow-up query, and callers must not use the
        value for exact accounting.
    """

    async def upsert(self, ns: NameSpace, items: list[VectorItem]) -> None: ...

    async def search(self, ns: NameSpace, q: HybridQuery, k: int) -> list[Hit]: ...

    async def delete(self, ns: NameSpace, flt: Filter) -> int: ...
