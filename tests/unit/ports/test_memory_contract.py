"""Contract test stub for MemoryStore (§5.1, task 0.6).

A *contract test* proves that a concrete adapter satisfies the MemoryStore
protocol.  This module provides:

1. ``FakeMemoryStore`` — an in-memory stub that is itself a valid adapter.
2. Tests that exercise the documented contract against the stub.

The contract in one line: **the scope is the whole partition key**.  Items
written under one workspace, agent or thread are never returned under
another (§2.3).  ``thread_id=None`` addresses long-term, agent-scoped memory;
callers that need long-term and thread memory compose two reads.

When the Postgres adapter lands (P4, task 4.8), it runs the same suite via a
fixture, so a scope leak fails CI.
"""

from __future__ import annotations

import pytest

from felops.domain.memory import MemoryItem, MemoryScope
from felops.ports.memory import MemoryStore

_THREAD = MemoryScope(workspace_id="ws-1", agent_id="ag-1", thread_id="thread-1")
_OTHER_THREAD = MemoryScope(workspace_id="ws-1", agent_id="ag-1", thread_id="thread-2")
_OTHER_WORKSPACE = MemoryScope(workspace_id="ws-2", agent_id="ag-1", thread_id="thread-1")


class FakeMemoryStore:
    """In-memory MemoryStore keyed by the full scope tuple.

    ``write`` is an upsert by ``item_id``.  ``read`` returns items in
    insertion order and narrows by a case-insensitive substring match when
    ``q`` is given.  Real adapters may rank by similarity instead; the query
    is a hint, the scope is the contract.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str, str | None], dict[str, MemoryItem]] = {}

    @staticmethod
    def _key(scope: MemoryScope) -> tuple[str, str, str | None]:
        return (scope.workspace_id, scope.agent_id, scope.thread_id)

    async def read(self, scope: MemoryScope, q: str | None) -> list[MemoryItem]:
        items = list(self._data.get(self._key(scope), {}).values())
        if q is None:
            return items
        needle = q.casefold()
        return [item for item in items if needle in item.content.casefold()]

    async def write(self, scope: MemoryScope, items: list[MemoryItem]) -> None:
        bucket = self._data.setdefault(self._key(scope), {})
        for item in items:
            bucket[item.item_id] = item


def _item(item_id: str, content: str) -> MemoryItem:
    return MemoryItem(item_id=item_id, content=content)


# ---------------------------------------------------------------------------
# Protocol structural conformance
# ---------------------------------------------------------------------------


class TestMemoryStoreProtocol:
    def test_is_structurally_compatible(self) -> None:
        store: MemoryStore = FakeMemoryStore()
        assert store is not None


# ---------------------------------------------------------------------------
# Contract tests — every MemoryStore adapter must pass these
# ---------------------------------------------------------------------------


class TestMemoryStoreContract:
    @pytest.fixture()
    def store(self) -> FakeMemoryStore:
        return FakeMemoryStore()

    @pytest.mark.asyncio
    async def test_write_then_read_round_trip(self, store: FakeMemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "prefers Arabic"), _item("m-2", "VIP customer")])

        items = await store.read(_THREAD, None)

        assert [(item.item_id, item.content) for item in items] == [
            ("m-1", "prefers Arabic"),
            ("m-2", "VIP customer"),
        ]

    @pytest.mark.asyncio
    async def test_read_of_empty_scope_returns_empty(self, store: FakeMemoryStore) -> None:
        assert await store.read(_THREAD, None) == []

    @pytest.mark.asyncio
    async def test_workspace_scope_isolates_items(self, store: FakeMemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "secret from ws-1")])

        assert await store.read(_OTHER_WORKSPACE, None) == []

    @pytest.mark.asyncio
    async def test_thread_scope_isolates_items(self, store: FakeMemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "thread one context")])

        assert await store.read(_OTHER_THREAD, None) == []

    @pytest.mark.asyncio
    async def test_query_narrows_without_crossing_scope(self, store: FakeMemoryStore) -> None:
        await store.write(
            _THREAD,
            [_item("m-1", "refund policy is 14 days"), _item("m-2", "order 1042 shipped")],
        )

        items = await store.read(_THREAD, "refund")

        assert [item.item_id for item in items] == ["m-1"]

    @pytest.mark.asyncio
    async def test_write_is_an_upsert_by_item_id(self, store: FakeMemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "old fact")])
        await store.write(_THREAD, [_item("m-1", "new fact")])

        items = await store.read(_THREAD, None)

        assert [item.content for item in items] == ["new fact"]
