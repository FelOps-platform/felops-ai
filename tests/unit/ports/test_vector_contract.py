"""Contract test stub for VectorStore (§5.1, task 0.5).

A *contract test* proves that a concrete adapter satisfies the VectorStore
protocol.  This module provides:

1. ``FakeVectorStore`` — an in-memory stub that is itself a valid adapter.
2. Tests that exercise every method in the protocol against the stub.

Workspace scoping (§2.3) is derived solely from ``NameSpace.workspace_id``;
``Filter`` carries payload conditions only.  The contract suite therefore
checks two things: payload filters narrow results, and data in one workspace
is unreachable from another.

When a real adapter lands (Qdrant in P1, pgvector in P10), it will run the
same suite via parametrize/fixture so regressions are caught at CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from felops.domain.retrieval_types import Filter, Hit, HybridQuery, Vector, VectorItem
from felops.domain.workspace import NameSpace
from felops.ports.vector import VectorStore

# ---------------------------------------------------------------------------
# In-memory stub — the simplest valid VectorStore
# ---------------------------------------------------------------------------


def _matches(payload: dict[str, Any], flt: Filter) -> bool:
    """Return whether *payload* satisfies every condition in *flt*.

    Only the Qdrant-style ``{"key": k, "match": {"value": v}}`` shape is
    supported.  A ``workspace_id`` condition is ignored on purpose: workspace
    scoping comes from ``NameSpace`` alone and is never a filter concern.
    """
    for condition in flt.must:
        key = condition.get("key")
        if key == "workspace_id":
            continue
        match = condition.get("match")
        if not isinstance(match, dict) or "value" not in match:
            raise ValueError(f"unsupported filter condition: {condition!r}")
        if payload.get(key) != match["value"]:
            return False
    return True


class FakeVectorStore:
    """In-memory VectorStore for tests.

    Stores items in a plain dict keyed by (namespace-tuple, item-id).
    Search returns all items in the namespace that match the filter (no real
    similarity math).  Filter conditions use the Qdrant ``match.value`` shape.
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
        for item in bucket.values():
            if not _matches(item.payload, q.filter):
                continue
            hits.append(Hit(id=item.id, score=1.0, payload=item.payload))
            if len(hits) == k:
                break
        return hits

    async def delete(self, ns: NameSpace, flt: Filter) -> int:
        key = self._ns_key(ns)
        bucket = self._data.get(key)
        if not bucket:
            return 0

        kept = {
            item_id: item for item_id, item in bucket.items() if not _matches(item.payload, flt)
        }
        deleted = len(bucket) - len(kept)
        if kept:
            self._data[key] = kept
        else:
            self._data.pop(key, None)
        return deleted


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


def _filter(**payload: object) -> Filter:
    """Build a payload filter using the Qdrant ``match.value`` shape."""
    return Filter(must=[{"key": key, "match": {"value": value}} for key, value in payload.items()])


def _query(flt: Filter | None = None) -> HybridQuery:
    return HybridQuery(text="hello", vector=_make_vector(), filter=flt or Filter())


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

        hits = await store.search(_NS, _query(_filter(text="hello world")), k=10)
        assert len(hits) == 1
        assert hits[0].id == "item-1"

    @pytest.mark.asyncio
    async def test_upsert_replaces_existing(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item("a", version=1)])
        await store.upsert(_NS, [_make_item("a", version=2)])

        hits = await store.search(_NS, _query(), k=10)
        assert len(hits) == 1
        assert hits[0].payload["version"] == 2

    @pytest.mark.asyncio
    async def test_search_empty_returns_empty(self, store: FakeVectorStore) -> None:
        hits = await store.search(_NS, _query(), k=10)
        assert hits == []

    @pytest.mark.asyncio
    async def test_search_respects_k(self, store: FakeVectorStore) -> None:
        items = [_make_item(f"item-{i}") for i in range(5)]
        await store.upsert(_NS, items)

        hits = await store.search(_NS, _query(), k=3)
        assert len(hits) == 3

    @pytest.mark.asyncio
    async def test_search_applies_payload_filter(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item("en", lang="en"), _make_item("fr", lang="fr")])

        hits = await store.search(_NS, _query(_filter(lang="en")), k=10)
        assert [h.id for h in hits] == ["en"]

    @pytest.mark.asyncio
    async def test_search_ignores_workspace_id_in_filter(self, store: FakeVectorStore) -> None:
        """Workspace scoping comes from ``ns``; a filter key is not authoritative."""
        await store.upsert(_NS, [_make_item()])

        hits = await store.search(_NS, _query(_filter(workspace_id="ws-other")), k=10)
        assert [h.id for h in hits] == ["item-1"]

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
    async def test_delete_only_removes_matching_items(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item("a", version=1), _make_item("b", version=2)])

        deleted = await store.delete(_NS, _filter(version=1))
        assert deleted == 1

        hits = await store.search(_NS, _query(), k=10)
        assert [h.id for h in hits] == ["b"]

    @pytest.mark.asyncio
    async def test_search_after_delete_returns_empty(self, store: FakeVectorStore) -> None:
        await store.upsert(_NS, [_make_item()])
        await store.delete(_NS, Filter())

        hits = await store.search(_NS, _query(), k=10)
        assert hits == []

    @pytest.mark.asyncio
    async def test_namespaces_are_isolated(self, store: FakeVectorStore) -> None:
        ns_a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-a")
        ns_b = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-b")

        await store.upsert(ns_a, [_make_item("a")])
        await store.upsert(ns_b, [_make_item("b")])

        hits_a = await store.search(ns_a, _query(), k=10)
        hits_b = await store.search(ns_b, _query(), k=10)

        assert [h.id for h in hits_a] == ["a"]
        assert [h.id for h in hits_b] == ["b"]

    @pytest.mark.asyncio
    async def test_workspaces_are_isolated(self, store: FakeVectorStore) -> None:
        """The same agent/source in another workspace must be unreachable (§2.3)."""
        ns_a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        ns_b = NameSpace(workspace_id="ws-2", agent_id="ag-1", source_id="src-1")

        await store.upsert(ns_a, [_make_item("a", workspace_id="ws-1")])
        await store.upsert(ns_b, [_make_item("b", workspace_id="ws-2")])

        assert [h.id for h in await store.search(ns_a, _query(), k=10)] == ["a"]
        assert [h.id for h in await store.search(ns_b, _query(), k=10)] == ["b"]

        assert await store.delete(ns_a, Filter()) == 1
        assert await store.search(ns_a, _query(), k=10) == []
        assert [h.id for h in await store.search(ns_b, _query(), k=10)] == ["b"]
