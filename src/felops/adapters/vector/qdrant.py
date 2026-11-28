"""Qdrant implementation of the `VectorStore` port (§2.3, §5.1, task 1.1).

P1 scope
--------
One shared collection with dense vectors.  The collection is created lazily
on the first `upsert` using the embedding dimension of that batch.  Sparse
vectors and hybrid search arrive in P5 (task 5.1); passing a sparse vector to
this adapter raises instead of silently dropping it.

Isolation (§2.3)
----------------
Every point stores the three namespace fields as payload:

    workspace_id, agent_id, source_id

Every `search` and `delete` ANDs those fields into the Qdrant filter, built
from the `NameSpace` argument.  `Filter` is payload-only: a ``workspace_id``
condition inside it is ignored, never trusted.  The adapter refuses a
namespace whose workspace (or agent/source) id is missing or blank, so an
unscoped read or delete cannot be issued by accident.

Point ids
---------
Qdrant accepts only unsigned integers or UUIDs as point ids.  `VectorItem.id`
is an arbitrary string, so the adapter derives a deterministic UUIDv5 from
``(workspace_id, agent_id, source_id, item.id)`` and stores the original id in
the payload.  Deriving from the full namespace means the same content id in
two workspaces or sources maps to different points and cannot overwrite
another namespace's data.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from felops.domain.retrieval_types import Filter, Hit, HybridQuery, Vector, VectorItem
from felops.domain.workspace import NameSpace

# Namespace fields are written after the caller payload, so a payload key with
# the same name can never spoof another workspace.
_NAMESPACE_FIELDS = ("workspace_id", "agent_id", "source_id")

# Original `VectorItem.id` is kept here because Qdrant needs a UUID/int id.
_ITEM_ID_KEY = "_felops_item_id"

# Stable namespace for UUIDv5 point ids; changing it re-keys every point.
_POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "felops/vector-store/point")


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    """Connection and collection settings for `QdrantVectorStore`.

    Constructed centrally by `felops.app.settings.Settings.to_qdrant_config()`
    (or directly in tests).

    Args:
        collection_name: Shared collection name.
        url: Qdrant HTTP URL (for example ``http://qdrant:6333``).  Ignored
            when a client is injected into the adapter.
        api_key: Optional Qdrant API key.  Ignored when a client is injected.
        timeout: Qdrant client timeout in seconds.  Ignored when a client is
            injected.
        distance: Vector distance used when the collection is created.
    """

    collection_name: str
    url: str | None = None
    api_key: str | None = None
    timeout: int = 60
    distance: models.Distance = models.Distance.COSINE

    def __post_init__(self) -> None:
        _validate_name(self.collection_name, "collection_name")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


def _validate_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _validate_namespace(ns: NameSpace) -> None:
    """Refuse an unscoped operation.

    ``NameSpace`` already validates non-empty fields on construction, but
    callers can build one with ``model_construct`` or pass a duck-typed object.
    This check is the last line of defence before a read or delete.
    """
    for field in _NAMESPACE_FIELDS:
        value = getattr(ns, field, None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"namespace {field} is required; workspace scoping is derived from NameSpace"
            )


def _dense_values(vector: Vector) -> list[float]:
    """Return the dense vector, rejecting P5-only sparse input."""
    if vector.sparse is not None:
        raise ValueError(
            "sparse vectors are not supported by the P1 Qdrant adapter; "
            "hybrid retrieval arrives in P5 (task 5.1)"
        )
    return list(vector.dense)


def _point_id(ns: NameSpace, item_id: str) -> str:
    """Deterministic Qdrant point id, unique per namespace and item id."""
    key = json.dumps(
        [ns.workspace_id, ns.agent_id, ns.source_id, item_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, key))


def _extract_filter_key(condition: Any) -> str | None:
    if not isinstance(condition, dict):
        raise ValueError(f"unsupported filter condition: {condition!r}")
    key = condition.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"unsupported filter condition: {condition!r}")
    return None if key == "workspace_id" else key


def _extract_match_value(condition: dict[str, Any]) -> Any:
    match = condition.get("match")
    if not isinstance(match, dict) or "value" not in match:
        raise ValueError(f"unsupported filter condition: {condition!r}")
    return match["value"]


def _parse_filter_condition(condition: Any) -> dict[str, Any] | None:
    """Validate and normalize a single payload filter condition.

    Returns None for workspace_id (which is scoped from NameSpace, not Filter).
    """
    key = _extract_filter_key(condition)
    if key is None:
        return None
    return {"key": key, "match": {"value": _extract_match_value(condition)}}


def _payload_filter(ns: NameSpace, flt: Filter) -> models.Filter:
    """Build a Qdrant filter from the namespace plus caller payload conditions.

    Only the documented ``{"key": ..., "match": {"value": ...}}`` shape is
    accepted.  Rebuilding each condition (instead of passing it through) keeps
    nested filters and other DSL features from bypassing namespace scoping.
    """
    conditions: list[dict[str, Any]] = []
    for condition in flt.must:
        parsed = _parse_filter_condition(condition)
        if parsed is not None:
            conditions.append(parsed)

    namespace_conditions = [
        {"key": field, "match": {"value": getattr(ns, field)}} for field in _NAMESPACE_FIELDS
    ]
    try:
        return models.Filter(must=[*namespace_conditions, *conditions])
    except ValidationError as exc:
        raise ValueError(f"unsupported filter condition: {exc}") from None


def _to_hit(point: models.ScoredPoint) -> Hit:
    payload = dict(point.payload or {})
    item_id = payload.pop(_ITEM_ID_KEY, None)
    if not isinstance(item_id, str) or not item_id:
        # Points written outside this adapter have no original id; fall back to
        # the Qdrant id so the result is still usable.
        item_id = str(point.id)
    score = float(point.score) if point.score is not None else 0.0
    return Hit(id=item_id, score=score, payload=payload)


class QdrantVectorStore:
    """`VectorStore` backed by Qdrant.

    Args:
        config: Connection and collection settings.
        client: Pre-built async client, mainly for tests.  When supplied the
            config's connection fields are ignored.  The adapter takes
            ownership and closes it in `close()`.
    """

    def __init__(
        self,
        config: QdrantConfig,
        *,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        if client is None:
            if config.url is None or not config.url.strip():
                raise ValueError("url must be a non-empty string when no client is supplied")
            client = AsyncQdrantClient(
                url=config.url,
                api_key=config.api_key,
                timeout=config.timeout,
            )
        self._client = client
        self.collection_name = config.collection_name
        self._distance = config.distance
        self._lock = asyncio.Lock()
        self._exists: bool | None = None

    async def __aenter__(self) -> QdrantVectorStore:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Release the underlying client's connections."""
        await self._client.close()

    # -- VectorStore -------------------------------------------------------

    async def upsert(self, ns: NameSpace, items: list[VectorItem]) -> None:
        _validate_namespace(ns)
        if not items:
            return

        dim = self._validated_dim(items)
        await self._ensure_collection(dim)
        await self._client.upsert(
            collection_name=self.collection_name,
            points=[self._point(ns, item) for item in items],
            wait=True,
        )

    async def search(self, ns: NameSpace, q: HybridQuery, k: int) -> list[Hit]:
        _validate_namespace(ns)
        if k <= 0:
            return []
        dense = _dense_values(q.vector)
        query_filter = _payload_filter(ns, q.filter)
        if not await self._has_collection():
            return []

        response = await self._client.query_points(
            collection_name=self.collection_name,
            query=dense,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
            with_vectors=False,
        )
        return [_to_hit(point) for point in response.points]

    async def delete(self, ns: NameSpace, flt: Filter) -> int:
        _validate_namespace(ns)
        query_filter = _payload_filter(ns, flt)
        if not await self._has_collection():
            return 0

        count = await self._client.count(
            collection_name=self.collection_name,
            count_filter=query_filter,
            exact=True,
        )
        removed = int(count.count)
        if removed == 0:
            return 0

        # Qdrant does not return the number of deleted points, so count first.
        # Concurrent writes can make this a best-effort number, as the port
        # allows.
        await self._client.delete(
            collection_name=self.collection_name,
            points_selector=query_filter,
            wait=True,
        )
        return removed

    # -- collection lifecycle ---------------------------------------------

    def _validated_dim(self, items: list[VectorItem]) -> int:
        dims = {len(_dense_values(item.vector)) for item in items}
        if len(dims) != 1:
            raise ValueError("all vectors in one upsert must have the same dimension")
        dim = dims.pop()
        if dim <= 0:
            raise ValueError("vector dimension must be positive")
        return dim

    def _point(self, ns: NameSpace, item: VectorItem) -> models.PointStruct:
        payload = dict(item.payload)
        payload["workspace_id"] = ns.workspace_id
        payload["agent_id"] = ns.agent_id
        payload["source_id"] = ns.source_id
        payload[_ITEM_ID_KEY] = item.id
        return models.PointStruct(
            id=_point_id(ns, item.id),
            vector=list(item.vector.dense),
            payload=payload,
        )

    async def _has_collection(self) -> bool:
        if self._exists:
            return True
        self._exists = await self._client.collection_exists(self.collection_name)
        return self._exists

    async def _ensure_collection(self, dim: int) -> None:
        if self._exists:
            return
        async with self._lock:
            if self._exists:
                return
            exists = await self._client.collection_exists(self.collection_name)
            if not exists:
                try:
                    await self._client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=dim,
                            distance=self._distance,
                        ),
                    )
                except (ValueError, UnexpectedResponse):
                    # Another process may have created the collection between
                    # the existence check and this call.
                    if not await self._client.collection_exists(self.collection_name):
                        raise
            # Create the indexes on every fresh process, not only when this
            # process created the collection.  Qdrant ignores an index that
            # already exists, so this is idempotent.
            await self._create_namespace_indexes()
            self._exists = True

    async def _create_namespace_indexes(self) -> None:
        for field in _NAMESPACE_FIELDS:
            await self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
