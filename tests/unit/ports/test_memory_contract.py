"""Contract test stub for MemoryStore (§5.1, task 0.6).

A *contract test* proves that a concrete adapter satisfies the MemoryStore
protocol.  This module provides:

1. ``FakeMemoryStore`` — an in-memory stub that is itself a valid adapter.
2. ``TestMemoryStoreContract`` — protocol-typed tests that any real adapter
   module can subclass by overriding the ``store`` fixture.

Contract summary (§2.3, §4, task 7.9):

* Workspace and thread boundaries are strict: items written in one workspace or
  thread are never returned from another, and long-term memory
  (``thread_id=None``) never bleeds into thread-scoped reads.
* Long-term memory is workspace-scoped: items written with ``agent_id=None``
  are readable from any agent in that workspace, while items written with a
  specific ``agent_id`` stay isolated to that agent.
* ``write`` is an upsert by ``MemoryItem.item_id``; omitting ``item_id``
  derives a deterministic id from normalized content and metadata so retried
  identical writes collapse into one record.

When the Postgres adapter lands (P4, task 4.8), it subclasses
``TestMemoryStoreContract`` and overrides the fixture so a scope leak fails CI.
"""

from __future__ import annotations

import pytest

from felops.domain.memory import MemoryItem, MemoryScope
from felops.ports.memory import MemoryStore

_WS_LONG_TERM = MemoryScope(workspace_id="ws-1")
_AG1_LONG_TERM = MemoryScope(workspace_id="ws-1", agent_id="ag-1")
_AG2_LONG_TERM = MemoryScope(workspace_id="ws-1", agent_id="ag-2")
_OTHER_WS_LONG_TERM = MemoryScope(workspace_id="ws-2")
_OTHER_WS_AG1 = MemoryScope(workspace_id="ws-2", agent_id="ag-1")

_THREAD = MemoryScope(workspace_id="ws-1", agent_id="ag-1", thread_id="thread-1")
_OTHER_THREAD = MemoryScope(workspace_id="ws-1", agent_id="ag-1", thread_id="thread-2")
_OTHER_WORKSPACE_THREAD = MemoryScope(workspace_id="ws-2", agent_id="ag-1", thread_id="thread-1")


class FakeMemoryStore:
    """In-memory MemoryStore enforcing workspace, agent, and thread scoping.

    ``write`` upserts by ``item_id`` inside the exact ``(workspace_id, agent_id,
    thread_id)`` bucket.  ``read`` returns items in that bucket plus any
    workspace-wide items (``agent_id=None``) for the same workspace and
    ``thread_id`` when ``scope.agent_id`` is set.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str | None, str | None], dict[str, MemoryItem]] = {}

    @staticmethod
    def _key(scope: MemoryScope) -> tuple[str, str | None, str | None]:
        return (scope.workspace_id, scope.agent_id, scope.thread_id)

    async def read(self, scope: MemoryScope, q: str | None) -> list[MemoryItem]:
        combined: dict[str, MemoryItem] = {}
        if scope.agent_id is not None:
            ws_wide_key = (scope.workspace_id, None, scope.thread_id)
            combined.update(self._data.get(ws_wide_key, {}))
        combined.update(self._data.get(self._key(scope), {}))

        items = list(combined.values())
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
    def store(self) -> MemoryStore:
        return FakeMemoryStore()

    @pytest.mark.asyncio
    async def test_write_then_read_round_trip(self, store: MemoryStore) -> None:
        """All written items are returned without assuming a specific ordering."""
        await store.write(_THREAD, [_item("m-1", "prefers Arabic"), _item("m-2", "VIP customer")])

        items = await store.read(_THREAD, None)

        assert {(item.item_id, item.content) for item in items} == {
            ("m-1", "prefers Arabic"),
            ("m-2", "VIP customer"),
        }

    @pytest.mark.asyncio
    async def test_read_of_empty_scope_returns_empty(self, store: MemoryStore) -> None:
        assert await store.read(_THREAD, None) == []

    @pytest.mark.asyncio
    async def test_workspace_scope_isolates_items(self, store: MemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "secret from ws-1")])

        assert await store.read(_OTHER_WORKSPACE_THREAD, None) == []

    @pytest.mark.asyncio
    async def test_thread_scope_isolates_items(self, store: MemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "thread one context")])

        assert await store.read(_OTHER_THREAD, None) == []

    @pytest.mark.asyncio
    async def test_workspace_scoped_item_is_readable_by_any_agent(
        self, store: MemoryStore
    ) -> None:
        """An item written with agent_id=None is readable by any agent in that workspace."""
        await store.write(_WS_LONG_TERM, [_item("fact-1", "customer prefers Arabic")])

        for scope in (_WS_LONG_TERM, _AG1_LONG_TERM, _AG2_LONG_TERM):
            items = await store.read(scope, None)
            assert {item.item_id for item in items} == {"fact-1"}

        assert await store.read(_OTHER_WS_LONG_TERM, None) == []
        assert await store.read(_OTHER_WS_AG1, None) == []
        assert await store.read(_THREAD, None) == []

    @pytest.mark.asyncio
    async def test_agent_scoped_items_are_isolated_between_agents(
        self, store: MemoryStore
    ) -> None:
        await store.write(_AG1_LONG_TERM, [_item("ag1-only", "agent-1 private note")])

        assert {item.item_id for item in await store.read(_AG1_LONG_TERM, None)} == {"ag1-only"}
        assert await store.read(_AG2_LONG_TERM, None) == []
        assert await store.read(_WS_LONG_TERM, None) == []

    @pytest.mark.asyncio
    async def test_query_returns_relevant_item_without_crossing_scope(
        self, store: MemoryStore
    ) -> None:
        """Querying returns relevant in-scope items and never leaks out-of-scope items."""
        await store.write(
            _THREAD,
            [_item("m-1", "refund policy is 14 days"), _item("m-2", "order 1042 shipped")],
        )
        await store.write(_OTHER_THREAD, [_item("m-other-thread", "refund policy other thread")])
        await store.write(
            _OTHER_WORKSPACE_THREAD,
            [_item("m-other-ws", "refund policy other workspace")],
        )

        items = await store.read(_THREAD, "refund")
        returned_ids = {item.item_id for item in items}

        assert "m-1" in returned_ids
        assert "m-other-thread" not in returned_ids
        assert "m-other-ws" not in returned_ids

    @pytest.mark.asyncio
    async def test_write_is_an_upsert_by_item_id(self, store: MemoryStore) -> None:
        await store.write(_THREAD, [_item("m-1", "old fact")])
        await store.write(_THREAD, [_item("m-1", "new fact")])

        items = await store.read(_THREAD, None)

        assert [item.content for item in items] == ["new fact"]

    @pytest.mark.asyncio
    async def test_retried_write_without_explicit_id_deduplicates(
        self, store: MemoryStore
    ) -> None:
        """Two independently constructed identical facts collapse into one stored item."""
        await store.write(_WS_LONG_TERM, [MemoryItem(content="Prefers  Arabic", metadata={"a": 1})])
        await store.write(_WS_LONG_TERM, [MemoryItem(content="prefers arabic", metadata={"a": 1})])

        items = await store.read(_WS_LONG_TERM, None)

        assert len(items) == 1
