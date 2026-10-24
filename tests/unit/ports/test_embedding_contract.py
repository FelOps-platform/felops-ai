"""Contract test stub for EmbeddingModel (§5.1, task 0.5).

A *contract test* proves that a concrete adapter satisfies the EmbeddingModel
protocol.  This module provides:

1. ``FakeEmbeddingModel`` — a deterministic stub that returns fixed-dimension
   zero vectors.  Valid adapter, no real inference.
2. Tests that exercise every method and attribute in the protocol.

When the BGE-M3 adapter lands (P1) and the OpenAI-compatible adapter (P10),
each will be parametrized into this suite so regressions are caught at CI.
"""

from __future__ import annotations

import pytest

from felops.domain.retrieval_types import Vector
from felops.ports.embedding import EmbeddingModel

# ---------------------------------------------------------------------------
# Deterministic stub — the simplest valid EmbeddingModel
# ---------------------------------------------------------------------------


class FakeEmbeddingModel:
    """Deterministic EmbeddingModel for tests.

    Returns zero-vectors of dimension ``dim``.  No real inference.
    """

    def __init__(self, model_id: str = "fake-embed", dim: int = 3) -> None:
        self.model_id = model_id
        self.dim = dim

    def _zero_vector(self) -> Vector:
        return Vector(dense=[0.0] * self.dim)

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._zero_vector() for _ in texts]

    async def embed_query(self, text: str) -> Vector:
        return self._zero_vector()


# ---------------------------------------------------------------------------
# Protocol structural conformance
# ---------------------------------------------------------------------------


class TestEmbeddingModelProtocol:
    """Verify FakeEmbeddingModel is structurally compatible with EmbeddingModel."""

    def test_is_structurally_compatible(self) -> None:
        model: EmbeddingModel = FakeEmbeddingModel()
        assert model is not None

    def test_has_model_id_attribute(self) -> None:
        model = FakeEmbeddingModel(model_id="bge-m3")
        assert model.model_id == "bge-m3"

    def test_has_dim_attribute(self) -> None:
        model = FakeEmbeddingModel(dim=1024)
        assert model.dim == 1024


# ---------------------------------------------------------------------------
# Contract tests — every EmbeddingModel adapter must pass these
# ---------------------------------------------------------------------------


class TestEmbeddingModelContract:
    """Contract tests exercising EmbeddingModel semantics."""

    @pytest.fixture()
    def model(self) -> FakeEmbeddingModel:
        return FakeEmbeddingModel(dim=4)

    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(self, model: FakeEmbeddingModel) -> None:
        vec = await model.embed_query("hello world")
        assert isinstance(vec, Vector)
        assert len(vec.dense) == model.dim

    @pytest.mark.asyncio
    async def test_embed_documents_returns_list_of_vectors(
        self, model: FakeEmbeddingModel
    ) -> None:
        texts = ["doc one", "doc two", "doc three"]
        vecs = await model.embed_documents(texts)
        assert len(vecs) == len(texts)
        for v in vecs:
            assert isinstance(v, Vector)
            assert len(v.dense) == model.dim

    @pytest.mark.asyncio
    async def test_embed_documents_empty_list(self, model: FakeEmbeddingModel) -> None:
        vecs = await model.embed_documents([])
        assert vecs == []

    @pytest.mark.asyncio
    async def test_embed_query_dimension_matches_dim(
        self, model: FakeEmbeddingModel
    ) -> None:
        vec = await model.embed_query("test")
        assert len(vec.dense) == model.dim

    @pytest.mark.asyncio
    async def test_embed_documents_dimension_matches_dim(
        self, model: FakeEmbeddingModel
    ) -> None:
        vecs = await model.embed_documents(["a", "b"])
        for v in vecs:
            assert len(v.dense) == model.dim
