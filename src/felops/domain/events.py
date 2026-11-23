"""Event envelope and type vocabulary

Every event emitted by the AI service is wrapped in an EventEnvelope.
The envelope carries workspace, agent, sub-agent, conversation and job
identity so downstream consumers (SSE clients, the backend webhook sink,
the /trace endpoint) can project events without additional lookups.

seq is monotonic per conversation and backs Last-Event-ID replay.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from felops.domain.agent import DEFAULT_SUB_AGENT_ID


class EventType(StrEnum):
    """All event types emitted by the AI service.

    String values match the wire format exactly (dot-separated, lower-case).
    """

    PLAN_CREATED = "plan.created"
    RETRIEVAL_COMPLETED = "retrieval.completed"
    CONFIG_WARNING = "config.warning"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    APPROVAL_REQUIRED = "approval.required"
    APPROVAL_RESOLVED = "approval.resolved"
    HANDOFF_REQUESTED = "handoff.requested"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_FINAL = "message.final"
    USAGE_RECORDED = "usage.recorded"
    ERROR = "error"


class EventEnvelope(BaseModel, frozen=True):
    """Immutable envelope wrapping every event.

    Fields mirror the wire contract:
      {event_id, seq, ts, type, workspace_id, agent_id,
       sub_agent_id, conversation_id, job_id, data}

    seq is monotonic per conversation; the events table and SSE replay
    depend on its ordering guarantee.
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    seq: int = Field(ge=0)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    type: EventType
    workspace_id: str
    agent_id: str
    sub_agent_id: str = DEFAULT_SUB_AGENT_ID
    conversation_id: str = ""
    job_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
