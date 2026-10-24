"""Value objects for the retrieval subsystem.

These types are used in the VectorStore and EmbeddingModel protocol signatures
(§5.1).  They live in ``domain/`` because they are pure data with no I/O —
anyone may import them.

Types
-----
Vector          Dense (and optional sparse) embedding representation.
VectorItem      A document chunk prepared for upsert into a vector store.
HybridQuery     A query carrying both dense and optional sparse vectors.
Filter          Metadata filter applied during vector search.
Hit             A single search result returned from the vector store.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Vector(BaseModel, frozen=True):
    """A dense embedding vector with an optional sparse counterpart.

    dense   Required.  The float embedding produced by the EmbeddingModel.
    sparse  Optional.  Non-zero (index, value) pairs for the BM25 / sparse
            leg of hybrid retrieval (Qdrant sparse vectors, §3).
            ``None`` when the embedding model does not produce sparse output
            or when only dense retrieval is used.
    """

    dense: list[float] = Field(min_length=1)
    sparse: dict[int, float] | None = None


class VectorItem(BaseModel, frozen=True):
    """A chunk ready to be upserted into the vector store.

    Carries the embedding and arbitrary metadata (source_id, headings,
    language, …).  The original text lives inside *payload* as ``payload["text"]``.
    ``payload`` is untyped, so that key is not validated here.  
    The ``id`` must be deterministic so that re-ingesting the same content is idempotent.
    """

    id: str = Field(min_length=1)
    vector: Vector
    payload: dict[str, Any] = Field(default_factory=dict)


class Filter(BaseModel, frozen=True):
    """Metadata filter for vector search and delete operations.

    ``must`` conditions are ANDed.  Each condition is a dict like
    ``{"key": "language", "match": {"value": "en"}}``.
    The exact shape follows the Qdrant filter DSL so the adapter can
    pass it through, but other adapters (pgvector, Weaviate) translate it.

    ``must``-only is a deliberate P1 decision: ``should`` (OR) and
    ``must_not`` (NOT) are deferred until a feature needs them.  The dict
    shape is also deliberate while P1 is Qdrant-first; new adapters translate
    it into their own DSL.  Condition keys are schema-free payload field
    names, so only the shape is fixed — key names cannot be validated here.

    Workspace scoping is **not** a filter concern: it is derived solely from
    ``NameSpace.workspace_id`` (§2.3 isolation rule).  Adapters must not
    require, and callers must not rely on, a ``workspace_id`` condition here.
    """

    must: list[dict[str, Any]] = Field(default_factory=list)


class HybridQuery(BaseModel, frozen=True):
    """A query for hybrid (dense + optional sparse) vector search.

    text    The raw query string (kept for logging / debugging).
    vector  The dense (and optionally sparse) embedding of the query,
            produced by ``EmbeddingModel.embed_query()``.
    filter  Payload metadata filter.  Workspace scoping comes from the
            ``NameSpace`` passed alongside the query, never from here.
    """

    text: str
    vector: Vector
    filter: Filter = Field(default_factory=Filter)


class Hit(BaseModel, frozen=True):
    """A single search result returned by ``VectorStore.search()``.

    ``id`` is the vector-item id (same as VectorItem.id).
    ``score`` is the relevance score (higher = better, scale depends on the
    store and the fusion method, e.g. RRF).
    ``payload`` echoes the stored metadata so the Retriever can build ``Chunk``
    objects without a second lookup.
    """

    id: str
    score: float
    payload: dict[str, Any] = Field(default_factory=dict)
