"""Memory value objects and scoping (§5.1).

Memory is workspace-scoped (§2.3).  A ``MemoryScope`` is the full partition
key: workspace, agent and an optional thread.

``thread_id=None`` addresses long-term, agent-scoped memory (customer facts,
P7).  A set ``thread_id`` addresses short-term thread memory (P4).  The scope
is exact: callers that need long-term and thread memory compose two reads.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class MemoryScope(BaseModel, frozen=True):
    """The partition key for one memory read or write.

    Attributes
    ----------
    `workspace_id` : str
        Isolation boundary; never omitted (§2.3).
    `agent_id` : str
        Knowledge and memory are agent-level, shared by its sub-agents.
    `thread_id` : str | None
        ``None`` for long-term, agent-scoped memory; a thread id for
        short-term conversation memory.
    """

    workspace_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    thread_id: str | None = None


class MemoryItem(BaseModel, frozen=True):
    """One stored memory.

    ``item_id`` doubles as the upsert key: writing an item with an existing id
    replaces it, so a retried write cannot duplicate a fact.
    """

    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
