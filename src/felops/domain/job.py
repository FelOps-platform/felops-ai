"""Job value objects for the async pipeline (§2.2).

A Job tracks a single graph run. Its status lifecycle:
    queued → running → done | failed | waiting_approval

V1 (task 1.7): in-memory dict of jobs.
P3 (task 3.5): Postgres jobs table.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    """Job lifecycle states (§2.2)."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    DONE = "done"
    FAILED = "failed"


class JobType(StrEnum):
    """Job types (§2.2).

    interactive: live chat turn.
    resume:      post-approval resume (task 3.4).
    ingest:      parse, chunk, embed (task 1.9).
    """

    INTERACTIVE = "interactive"
    RESUME = "resume"
    INGEST = "ingest"


class JobSender(BaseModel, frozen=True):
    """Message sender snapshot stored on Job.

    Stored early in P1 just to be ready for P4 / downstream attribution
    (e.g. approval requester context in §8.3); not read by P1 graph nodes.
    """

    id: str = Field(min_length=1)
    name: str = ""


class Job(BaseModel):
    """A tracked graph run.

    Attributes:
        job_id:          Unique job identifier returned in the 202 response.
        job_type:        interactive | resume | ingest.
        status:          Current lifecycle state.
        workspace_id:    Workspace isolation key.
        agent_id:        Agent that owns this run.
        conversation_id: Conversation thread this job belongs to.
        idempotency_key: The Idempotency-Key header value (workspace-scoped).
        locale:          BCP-47 locale from the request; stored on Job now just
                         to be ready for P4 (e.g. Arabic normalization); unused in P1.
        sender:          Sender metadata from the request; stored on Job now just
                         to be ready for P4 / downstream attribution; unused in P1.
        created_at:      When the job was created.
        finished_at:     When the job reached a terminal state (done/failed).
        error:           Error message if status is 'failed'.
    """

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_type: JobType = JobType.INTERACTIVE
    status: JobStatus = JobStatus.QUEUED
    workspace_id: str
    agent_id: str
    conversation_id: str
    idempotency_key: str
    # Stored on Job in P1 just to be ready for P4 (Arabic normalization); unused in P1.
    locale: str = "en"
    # Stored on Job in P1 just to be ready for P4 / downstream attribution; unused in P1.
    sender: JobSender | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None
    error: str | None = None
