"""Contract test stub for VectorStore (§5.1, task 0.5).

A *contract test* proves that a concrete adapter satisfies the VectorStore
protocol.  This module provides:

1. ``FakeVectorStore`` — an in-memory stub that is itself a valid adapter.
2. Tests that exercise every method in the protocol against the stub.

When a real adapter lands (Qdrant in P1, pgvector in P10), it will run
the same suite via parametrize/fixture so regressions are caught at CI.
"""

from __future__ import annotations

import pytest

from felops.domain.retrieval_types import Filter, Hit, HybridQuery, Vector, VectorItem
from felops.domain.workspace import NameSpace
from felops.ports.vector import VectorStore

# ---------------------------------------------------------------------------
# In-memory stub — the simplest valid VectorStore
# ---------------------------------------------------------------------------


class FakeVectorStore:
    """In-memory VectorStore for tests.

    Stores items in a plain dict keyed by (namespace-tuple, item-id).
    Search returns all items in the namespace (no real similarity math).
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str], dict[str, VectorItem]] = {}

    def _ns_key(self, ns: NameSpace) -> tuple[str, str, str]:
        return (ns.workspace_id, ns.agent_id, ns.source_id)

    async def upsert(self, ns: NameSpace, items: list[VectorItem]) -> None:
        key = self._ns_key(ns)
        bucket = self._data.setdefault(key, {})
        for item in items:
            bucket[item.id] = item

    async def search(self, ns: NameSpace, q: HybridQuery, k: int) -> list[Hit]:
        key = self._ns_key(ns)
        bucket = self._data.get(key, {})
        hits: list[Hit] = []
        for item in list(bucket.values())[:k]:
            hits.append(Hit(id=item.id, score=1.0, payload=item.payload))
        return hits

    async def delete(self, ns: NameSpace, flt: Filter) -> int:
        key = self._ns_key(ns)
        bucket = self._data.pop(key, {})
        return len(bucket)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NS = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")


def _make_vector() -> Vector:
    return Vector(dense=[0.1, 0.2, 0.3])


def _make_item(item_id: str = "item-1", **payload_extra: object) -> VectorItem:
    payload: dict[str, object] = {"workspace_id": "ws-1", "text": "hello world"}
    payload.update(payload_extra)
    return VectorItem(id=item_id, vector=_make_vector(), payload=payload)


# ---------------------------------------------------------------------------
# Protocol structural conformance
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    """Verify FakeVectorStore is structurally compatible with VectorStore."""

    def test_is_structurally_compatible(self) -> None:
        """The stub must satisfy the Protocol at type-check time.

        We do a runtime isinstance check via Protocol's __subclasshook__
        indirectly: we assign to a typed variable.
        """
        store: VectorStore = FakeVectorStore()
        assert store is not None


# ---------------------------------------------------------------------------
# Contract tests — every VectorStore adapter must pass these
# ---------------------------------------------------------------------------


class TestVectorStoreContract:
    """Contract tests exercising VectorStore semantics."""

    @pytest.fixture()
    def store(self) -> FakeVectorStore:
        return FakeVectorStore()

    @pytest.mark.asyncio
    async def test_upsert_and_search(self, store: FakeVectorStore) -> None:
        item = _make_item()
        await store.upsert(_NS, [item])

        q = HybridQuery(
            text="hello",
            vector=_make_vector(),
            filter=Filter(must=[{"key": "workspace_id", "match": {"value": "ws-1"}}]),
        )
        hits = await store.search(_NS, q, k=10)
        assert len(hits) == 1
        assert hits[0].id == "item-1"

    @pytest.mark.asyncio
    async def test_upsert_replaces_existing(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item("a", version=1)])
        await store.upsert(_NS, [_make_item("a", version=2)])

        q = HybridQuery(text="x", vector=_make_vector())
        hits = await store.search(_NS, q, k=10)
        assert len(hits) == 1
        assert hits[0].payload["version"] == 2

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty(self, store: FakeVectorStore) -> None:
        q = HybridQuery(text="nothing", vector=_make_vector())
        hits = await store.search(_NS, q, k=10)
        assert hits == []

    @pytest.mark.asyncio
    async def test_search_respects_k(self, store: FakeVectorStore) -> None:
        items = [_make_item(f"item-{i}") for i in range(5)]
        await store.upsert(_NS, items)

        q = HybridQuery(text="hello", vector=_make_vector())
        hits = await store.search(_NS, q, k=3)
        assert len(hits) == 3

    @pytest.mark.asyncio
    async def test_delete_returns_count(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item("a"), _make_item("b")])
        deleted = await store.delete(_NS, Filter())
        assert deleted == 2

    @pytest.mark.asyncio
    async def test_delete_on_empty_returns_zero(self, store: FakeVectorStore) -> None:
        deleted = await store.delete(_NS, Filter())
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_search_after_delete_returns_empty(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item()])
        await store.delete(_NS, Filter())

        q = HybridQuery(text="hello", vector=_make_vector())
        hits = await store.search(_NS, q, k=10)
        assert hits == []

    @pytest.mark.asyncio
    async def test_namespaces_are_isolated(self, store: FakeVectorStore) -> None:
        ns_a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-a")
        ns_b = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-b")

        await store.upsert(ns_a, [_make_item("a")])
        await store.upsert(ns_b, [_make_item("b")])

        q = HybridQuery(text="hello", vector=_make_vector())
        hits_a = await store.search(ns_a, q, k=10)
        hits_b = await store.search(ns_b, q, k=10)

        assert [h.id for h in hits_a] == ["a"]
        assert [h.id for h in hits_b] == ["b"]
