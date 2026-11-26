"""Unit tests for felops.domain.retrieval_types — Vector, VectorItem, HybridQuery, Filter, Hit."""

import pytest
from pydantic import ValidationError

from felops.domain.retrieval_types import Filter, Hit, HybridQuery, Vector, VectorItem

# ---------------------------------------------------------------------------
# Vector
# ---------------------------------------------------------------------------


class TestVector:
    def test_creation_dense_only(self):
        v = Vector(dense=[0.1, 0.2, 0.3])
        assert v.dense == [0.1, 0.2, 0.3]
        assert v.sparse is None

    def test_creation_with_sparse(self):
        v = Vector(dense=[0.1, 0.2], sparse={0: 0.5, 42: 1.2})
        assert v.sparse == {0: 0.5, 42: 1.2}

    def test_empty_dense_rejected(self):
        with pytest.raises(ValidationError):
            Vector(dense=[])

    def test_frozen_rejects_mutation(self):
        v = Vector(dense=[1.0])
        with pytest.raises(ValidationError):
            v.dense = [2.0]  # type: ignore[misc]

    def test_equality(self):
        a = Vector(dense=[1.0, 2.0])
        b = Vector(dense=[1.0, 2.0])
        assert a == b

    def test_inequality(self):
        a = Vector(dense=[1.0])
        b = Vector(dense=[2.0])
        assert a != b


# ---------------------------------------------------------------------------
# VectorItem
# ---------------------------------------------------------------------------


class TestVectorItem:
    def test_creation(self):
        v = Vector(dense=[0.1])
        item = VectorItem(id="chunk-1", vector=v, payload={"text": "hello"})
        assert item.id == "chunk-1"
        assert item.vector == v
        assert item.payload["text"] == "hello"

    def test_empty_id_rejected(self):
        with pytest.raises(ValidationError):
            VectorItem(id="", vector=Vector(dense=[0.1]))

    def test_default_payload_is_empty_dict(self):
        item = VectorItem(id="x", vector=Vector(dense=[0.1]))
        assert item.payload == {}

    def test_frozen_rejects_mutation(self):
        item = VectorItem(id="x", vector=Vector(dense=[0.1]))
        with pytest.raises(ValidationError):
            item.id = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


class TestFilter:
    def test_default_is_empty_must(self):
        f = Filter()
        assert f.must == []

    def test_creation_with_conditions(self):
        conds = [{"key": "language", "match": {"value": "en"}}]
        f = Filter(must=conds)
        assert len(f.must) == 1
        assert f.must[0]["key"] == "language"

    def test_frozen_rejects_mutation(self):
        f = Filter()
        with pytest.raises(ValidationError):
            f.must = [{"key": "x"}]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# HybridQuery
# ---------------------------------------------------------------------------


class TestHybridQuery:
    def test_creation(self):
        v = Vector(dense=[0.1, 0.2])
        q = HybridQuery(text="hello", vector=v)
        assert q.text == "hello"
        assert q.vector == v
        assert q.filter.must == []

    def test_creation_with_filter(self):
        v = Vector(dense=[0.1])
        flt = Filter(must=[{"key": "language", "match": {"value": "en"}}])
        q = HybridQuery(text="hello", vector=v, filter=flt)
        assert len(q.filter.must) == 1

    def test_frozen_rejects_mutation(self):
        q = HybridQuery(text="hello", vector=Vector(dense=[0.1]))
        with pytest.raises(ValidationError):
            q.text = "world"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hit
# ---------------------------------------------------------------------------


class TestHit:
    def test_creation(self):
        h = Hit(id="chunk-1", score=0.95, payload={"text": "hello"})
        assert h.id == "chunk-1"
        assert h.score == 0.95
        assert h.payload["text"] == "hello"

    def test_default_payload_is_empty_dict(self):
        h = Hit(id="x", score=0.5)
        assert h.payload == {}

    def test_frozen_rejects_mutation(self):
        h = Hit(id="x", score=0.5)
        with pytest.raises(ValidationError):
            h.score = 0.9  # type: ignore[misc]

    def test_equality(self):
        a = Hit(id="x", score=0.5, payload={"k": "v"})
        b = Hit(id="x", score=0.5, payload={"k": "v"})
        assert a == b
