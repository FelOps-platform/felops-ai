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
        MemoryScope(workspace_id="", agent_id="ag-1")


def test_memory_scope_rejects_empty_agent_id() -> None:
    with pytest.raises(ValidationError):
        MemoryScope(workspace_id="ws-1", agent_id="")


def test_memory_scope_defaults_to_agent_level() -> None:
    """A missing thread_id means long-term, agent-scoped memory."""
    scope = MemoryScope(workspace_id="ws-1", agent_id="ag-1")

    assert scope.thread_id is None


def test_memory_item_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        MemoryItem(content="")
