"""Unit tests for the memory value objects (task 0.6).

The MemoryStore port (§5.1) is scoped by ``MemoryScope`` and moves
``MemoryItem`` values, so these types ship with the port.  They carry no I/O.
"""

from __future__ import annotations

import pytest
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
