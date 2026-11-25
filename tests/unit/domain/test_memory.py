"""Unit tests for the memory value objects (task 0.6).

The MemoryStore port (§5.1) is scoped by ``MemoryScope`` and moves
``MemoryItem`` values, so these types ship with the port.  They carry no I/O.
"""

from __future__ import annotations

import copy
import json
import warnings

import pytest
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import ValidationError

from felops.domain.memory import MemoryItem, MemoryScope


def test_memory_scope_rejects_empty_workspace_id() -> None:
    """Every memory read is workspace-scoped (§2.3)."""
    with pytest.raises(ValidationError):
        MemoryScope(workspace_id="")


def test_memory_scope_rejects_empty_agent_id() -> None:
    with pytest.raises(ValidationError):
        MemoryScope(workspace_id="ws-1", agent_id="")


def test_memory_scope_rejects_empty_thread_id() -> None:
    with pytest.raises(ValidationError):
        MemoryScope(workspace_id="ws-1", thread_id="")


def test_memory_scope_defaults_to_workspace_wide_long_term() -> None:
    """Omitting agent_id and thread_id addresses long-term, workspace-scoped memory."""
    scope = MemoryScope(workspace_id="ws-1")

    assert scope.agent_id is None
    assert scope.thread_id is None


def test_memory_item_rejects_empty_or_blank_content() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(content="")
    with pytest.raises(ValidationError):
        MemoryItem(content="   \n\t  ")


def test_memory_item_rejects_empty_explicit_item_id() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(item_id="", content="prefers Arabic")


def test_memory_item_derives_deterministic_id_for_identical_items() -> None:
    """Two independently constructed items with the same content and metadata share an id."""
    first = MemoryItem(content="prefers Arabic", metadata={"channel": "whatsapp", "tier": "vip"})
    second = MemoryItem(content="prefers Arabic", metadata={"channel": "whatsapp", "tier": "vip"})

    assert first.item_id == second.item_id


def test_memory_item_normalizes_whitespace_casing_and_metadata_order() -> None:
    first = MemoryItem(
        content="  Prefers   Arabic\n",
        metadata={"b": 2, "a": 1},
    )
    second = MemoryItem(
        content="prefers arabic",
        metadata={"a": 1, "b": 2},
    )

    assert first.item_id == second.item_id


def test_memory_item_excludes_score_from_derived_id() -> None:
    """score is a read-time value, not stored identity."""
    low = MemoryItem(content="prefers Arabic", metadata={"source": "chat"}, score=0.1)
    high = MemoryItem(content="prefers Arabic", metadata={"source": "chat"}, score=0.95)

    assert low.item_id == high.item_id


def test_memory_item_changes_derived_id_when_content_changes() -> None:
    arabic = MemoryItem(content="prefers Arabic")
    english = MemoryItem(content="prefers English")

    assert arabic.item_id != english.item_id


def test_memory_item_explicit_id_wins_over_derived_id() -> None:
    explicit = MemoryItem(item_id="customer:123:language", content="prefers Arabic")
    derived = MemoryItem(content="prefers Arabic")

    assert explicit.item_id == "customer:123:language"
    assert explicit.item_id != derived.item_id


def test_memory_item_model_json_schema_does_not_require_item_id() -> None:
    schema = MemoryItem.model_json_schema()

    assert "item_id" not in schema.get("required", [])
    assert "content" in schema["required"]
    assert MemoryItem.model_fields["item_id"].is_required() is False


def test_memory_item_validates_from_dict_and_json_without_item_id() -> None:
    from_dict = MemoryItem.model_validate({"content": "prefers Arabic"})
    from_json = MemoryItem.model_validate_json('{"content": "prefers Arabic"}')
    direct = MemoryItem(content="prefers Arabic")

    assert from_dict.item_id == from_json.item_id == direct.item_id
    assert (
        MemoryItem.model_validate(
            MemoryItem(item_id="explicit:1", content="prefers Arabic")
        ).item_id
        == "explicit:1"
    )
    with pytest.raises(ValidationError):
        MemoryItem.model_validate({"item_id": None, "content": "prefers Arabic"})


def test_memory_item_deep_copies_caller_metadata_and_resists_caller_mutation() -> None:
    raw: dict[str, object] = {"nested": {"k": "v"}, "tags": ["a", {"b": 1}]}
    item = MemoryItem(content="prefers Arabic", metadata=raw)

    raw["extra"] = 1
    assert isinstance(raw["nested"], dict)
    raw["nested"]["k"] = "mutated"
    assert isinstance(raw["tags"], list)
    raw["tags"].append("mutated")
    assert isinstance(raw["tags"][1], dict)
    raw["tags"][1]["b"] = 99

    expected = {"nested": {"k": "v"}, "tags": ["a", {"b": 1}]}
    assert item.metadata == expected
    assert item.item_id == MemoryItem(content="prefers Arabic", metadata=expected).item_id


def test_memory_item_metadata_is_immutable_after_construction() -> None:
    item = MemoryItem(
        content="prefers Arabic",
        metadata={"nested": {"k": "v"}, "tags": ["a", {"b": 1}]},
    )
    default_item = MemoryItem(content="prefers Arabic")

    with pytest.raises(TypeError):
        default_item.metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        item.metadata["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        item.metadata.update({"x": 1})  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        item.metadata.pop("nested")  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        del item.metadata["nested"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        item.metadata["nested"]["k"] = "mutated"
    with pytest.raises(TypeError):
        item.metadata["tags"].append("b")
    with pytest.raises(TypeError):
        item.metadata["tags"][0] = "b"
    with pytest.raises(TypeError):
        item.metadata["tags"][1]["b"] = 99


def test_memory_item_serialization_and_deepcopy_with_frozen_metadata() -> None:
    item = MemoryItem(
        content="prefers Arabic",
        metadata={"b": 2, "a": {"k": "v"}, "tags": ["x", "y"]},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dumped = item.model_dump()
    assert caught == []
    assert type(dumped["metadata"]) is dict
    assert type(dumped["metadata"]["a"]) is dict
    assert type(dumped["metadata"]["tags"]) is list

    assert MemoryItem.model_validate_json(item.model_dump_json()) == item

    reordered = {"a": {"k": "v"}, "tags": ["x", "y"], "b": 2}
    assert json.dumps(item.metadata, sort_keys=True, default=str) == json.dumps(
        reordered, sort_keys=True, default=str
    )

    deep_copied = copy.deepcopy(item)
    model_copied = item.model_copy(deep=True)
    assert deep_copied == item
    assert model_copied == item
    with pytest.raises(TypeError):
        deep_copied.metadata["b"] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        model_copied.metadata["a"]["k"] = "mutated"

    serde = JsonPlusSerializer(allowed_msgpack_modules=[MemoryItem])
    assert serde.loads_typed(serde.dumps_typed(item)) == item
