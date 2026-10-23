"""VectorStore protocol — §5.1.

The single swap point for vector storage.  Adapters: Qdrant (P1),
pgvector (P10), Weaviate, Pinecone.

Invariant (§2.3): the adapter **must** reject any ``search`` or ``delete``
call whose ``flt`` does not include a ``workspace_id`` condition.
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
        Hybrid search with mandatory workspace filter.
    `delete`
        Remove items matching *flt* under the namespace; return count.
    """

    async def upsert(self, ns: NameSpace, items: list[VectorItem]) -> None: ...

    async def search(self, ns: NameSpace, q: HybridQuery, k: int) -> list[Hit]: ...

    async def delete(self, ns: NameSpace, flt: Filter) -> int: ...