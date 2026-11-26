"""Unit tests for `adapters/vector/qdrant.py` (task 1.1).

The contract suite from `tests/unit/ports/test_vector_contract.py` runs against
the real adapter backed by Qdrant's in-memory local mode, so namespace
isolation and filter semantics are exercised without a container.  The
adapter-specific tests cover the extra guarantees the issue asks for:
workspace scoping comes from `NameSpace`, an unscoped call is rejected, and a
payload key cannot spoof another namespace.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from tests.unit.ports.test_vector_contract import (
    TestVectorStoreContract as _VectorStoreContract,
)

from felops.adapters.vector import QdrantConfig, QdrantVectorStore
from felops.domain.retrieval_types import Filter, HybridQuery, Vector, VectorItem
from felops.domain.workspace import NameSpace
from felops.ports.vector import VectorStore

# Qdrant's local mode cannot create payload indexes; the adapter still asks,
# because a real server needs them. Silence the vendor's warning in unit tests.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect in the local Qdrant:UserWarning"
)

_NS = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")


def _vector(dense: list[float] | None = None, sparse: dict[int, float] | None = None) -> Vector:
    return Vector(dense=dense or [0.1, 0.2, 0.3], sparse=sparse)


def _item(
    item_id: str = "item-1",
    *,
    payload: dict[str, object] | None = None,
    dense: list[float] | None = None,
    sparse: dict[int, float] | None = None,
) -> VectorItem:
    return VectorItem(
        id=item_id,
        vector=_vector(dense=dense, sparse=sparse),
        payload=payload if payload is not None else {"text": "hello world"},
    )


def _query(flt: Filter | None = None, sparse: dict[int, float] | None = None) -> HybridQuery:
    return HybridQuery(
        text="hello",
        vector=_vector(sparse=sparse),
        filter=flt or Filter(),
    )


def _unscoped(workspace_id: str = "") -> NameSpace:
    """Build a namespace that bypasses pydantic validation, like a bad caller."""
    return NameSpace.model_construct(workspace_id=workspace_id, agent_id="ag-1", source_id="src-1")


@pytest_asyncio.fixture
async def store() -> AsyncIterator[QdrantVectorStore]:
    client = AsyncQdrantClient(location=":memory:")
    adapter = QdrantVectorStore(QdrantConfig(collection_name="test"), client=client)
    yield adapter
    await adapter.close()


# ---------------------------------------------------------------------------
# The shared VectorStore contract
# ---------------------------------------------------------------------------


class TestQdrantVectorStoreContract(_VectorStoreContract):
    """Run the protocol contract against the real adapter, not the stub."""

    @pytest_asyncio.fixture
    async def store(self) -> AsyncIterator[QdrantVectorStore]:
        client = AsyncQdrantClient(location=":memory:")
        adapter = QdrantVectorStore(QdrantConfig(collection_name="contract"), client=client)
        yield adapter
        await adapter.close()


# ---------------------------------------------------------------------------
# Construction and port conformance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_satisfies_port(store: QdrantVectorStore) -> None:
    adapter: VectorStore = store
    assert adapter is not None


def test_requires_url_when_no_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    with pytest.raises(ValueError, match="QDRANT_URL"):
        QdrantVectorStore()


def test_reads_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("felops.adapters.vector.qdrant.AsyncQdrantClient", FakeClient)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    adapter = QdrantVectorStore()

    assert adapter.collection_name == "felops_knowledge"
    assert seen["url"] == "http://qdrant:6333"


def test_passes_client_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)

    monkeypatch.setattr("felops.adapters.vector.qdrant.AsyncQdrantClient", FakeClient)

    QdrantVectorStore(QdrantConfig(url="http://qdrant:6333", api_key="secret", timeout=12))

    assert seen == {"url": "http://qdrant:6333", "api_key": "secret", "timeout": 12}


def test_config_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")

    config = QdrantConfig.from_env()

    assert config.url == "http://qdrant:6333"
    assert config.collection_name == "felops_knowledge"
    assert config.timeout == 60


@pytest.mark.parametrize("collection_name", ["", "   "])
def test_rejects_blank_collection_name(collection_name: str) -> None:
    with pytest.raises(ValueError, match="collection_name"):
        # The client is never used: validation fails first.
        QdrantVectorStore(QdrantConfig(collection_name=collection_name), client=object())


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout"):
        QdrantVectorStore(QdrantConfig(timeout=0), client=object())


# ---------------------------------------------------------------------------
# Workspace scoping comes from NameSpace, never from Filter
# ---------------------------------------------------------------------------


class TestUnscopedCallsAreRejected:
    """The issue's requirement: reject an operation with no workspace scope."""

    @pytest.mark.asyncio
    async def test_upsert_rejects_blank_workspace(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="workspace_id"):
            await store.upsert(_unscoped(), [_item()])

    @pytest.mark.asyncio
    async def test_search_rejects_blank_workspace(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="workspace_id"):
            await store.search(_unscoped(), _query(), k=10)

    @pytest.mark.asyncio
    async def test_delete_rejects_blank_workspace(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="workspace_id"):
            await store.delete(_unscoped(), Filter())

    @pytest.mark.asyncio
    async def test_rejects_missing_agent(self, store: QdrantVectorStore) -> None:
        ns = NameSpace.model_construct(workspace_id="ws-1", agent_id="", source_id="src-1")
        with pytest.raises(ValueError, match="agent_id"):
            await store.search(ns, _query(), k=10)

    @pytest.mark.asyncio
    async def test_rejects_missing_source(self, store: QdrantVectorStore) -> None:
        ns = NameSpace.model_construct(workspace_id="ws-1", agent_id="ag-1", source_id=" ")
        with pytest.raises(ValueError, match="source_id"):
            await store.delete(ns, Filter())


class TestPayloadIsolation:
    @pytest.mark.asyncio
    async def test_workspace_id_in_filter_is_ignored(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [_item()])

        hits = await store.search(
            _NS,
            _query(Filter(must=[{"key": "workspace_id", "match": {"value": "ws-other"}}])),
            k=10,
        )

        assert [hit.id for hit in hits] == ["item-1"]

    @pytest.mark.asyncio
    async def test_payload_cannot_spoof_workspace_id(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [_item(payload={"text": "secret", "workspace_id": "ws-evil"})])

        own = await store.search(_NS, _query(), k=10)
        other_ns = NameSpace(workspace_id="ws-evil", agent_id="ag-1", source_id="src-1")
        other = await store.search(other_ns, _query(), k=10)

        assert [hit.id for hit in own] == ["item-1"]
        assert own[0].payload["workspace_id"] == "ws-1"
        assert other == []

    @pytest.mark.asyncio
    async def test_same_item_id_in_two_namespaces_does_not_collide(
        self, store: QdrantVectorStore
    ) -> None:
        ns_a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-a")
        ns_b = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-b")

        await store.upsert(ns_a, [_item("same", payload={"text": "a"})])
        await store.upsert(ns_b, [_item("same", payload={"text": "b"})])

        hits_a = await store.search(ns_a, _query(), k=10)
        hits_b = await store.search(ns_b, _query(), k=10)

        assert [hit.payload["text"] for hit in hits_a] == ["a"]
        assert [hit.payload["text"] for hit in hits_b] == ["b"]

    @pytest.mark.asyncio
    async def test_same_item_id_in_two_workspaces_does_not_collide(
        self, store: QdrantVectorStore
    ) -> None:
        ns_a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        ns_b = NameSpace(workspace_id="ws-2", agent_id="ag-1", source_id="src-1")

        await store.upsert(ns_a, [_item("same", payload={"text": "ws-1"})])
        await store.upsert(ns_b, [_item("same", payload={"text": "ws-2"})])

        assert [hit.payload["text"] for hit in await store.search(ns_a, _query(), k=10)] == ["ws-1"]
        assert [hit.payload["text"] for hit in await store.search(ns_b, _query(), k=10)] == ["ws-2"]

        assert await store.delete(ns_a, Filter()) == 1
        assert await store.search(ns_a, _query(), k=10) == []
        assert [hit.payload["text"] for hit in await store.search(ns_b, _query(), k=10)] == ["ws-2"]


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------


class TestCollectionLifecycle:
    @pytest.mark.asyncio
    async def test_second_adapter_reuses_an_existing_collection(self) -> None:
        client = AsyncQdrantClient(location=":memory:")
        first = QdrantVectorStore(QdrantConfig(collection_name="shared"), client=client)
        second = QdrantVectorStore(QdrantConfig(collection_name="shared"), client=client)
        try:
            await first.upsert(_NS, [_item("item-1")])
            await second.upsert(_NS, [_item("item-2")])

            hits = await second.search(_NS, _query(), k=10)
            assert {hit.id for hit in hits} == {"item-1", "item-2"}
        finally:
            await first.close()


# ---------------------------------------------------------------------------
# Filter and vector validation
# ---------------------------------------------------------------------------


class TestFilterValidation:
    @pytest.mark.asyncio
    async def test_rejects_unsupported_match_shape(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [_item()])

        with pytest.raises(ValueError, match="unsupported filter condition"):
            await store.search(
                _NS,
                _query(Filter(must=[{"key": "lang", "match": {"any": ["en", "ar"]}}])),
                k=10,
            )

    @pytest.mark.asyncio
    async def test_rejects_nested_filter_condition(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [_item()])

        with pytest.raises(ValueError, match="unsupported filter condition"):
            await store.delete(_NS, Filter(must=[{"must": []}]))

    @pytest.mark.asyncio
    async def test_invalid_filter_raises_even_when_collection_is_missing(
        self, store: QdrantVectorStore
    ) -> None:
        with pytest.raises(ValueError, match="unsupported filter condition"):
            await store.search(
                _NS,
                _query(Filter(must=[{"key": "lang", "match": {"any": ["en"]}}])),
                k=10,
            )

    @pytest.mark.asyncio
    async def test_k_zero_returns_empty(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [_item()])

        assert await store.search(_NS, _query(), k=0) == []


class TestVectorValidation:
    @pytest.mark.asyncio
    async def test_sparse_upsert_is_rejected(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="sparse vectors"):
            await store.upsert(_NS, [_item(sparse={1: 0.5})])

    @pytest.mark.asyncio
    async def test_sparse_query_is_rejected(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [_item()])

        with pytest.raises(ValueError, match="sparse vectors"):
            await store.search(_NS, _query(sparse={1: 0.5}), k=10)

    @pytest.mark.asyncio
    async def test_mixed_dimensions_are_rejected(self, store: QdrantVectorStore) -> None:
        with pytest.raises(ValueError, match="same dimension"):
            await store.upsert(
                _NS,
                [_item("a", dense=[0.1, 0.2]), _item("b", dense=[0.1, 0.2, 0.3])],
            )

    @pytest.mark.asyncio
    async def test_empty_upsert_is_a_no_op(self, store: QdrantVectorStore) -> None:
        await store.upsert(_NS, [])

        assert await store.search(_NS, _query(), k=10) == []
