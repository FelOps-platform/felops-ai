"""Unit tests for felops.domain.job (Job, JobStatus, JobType)."""

from __future__ import annotations

import uuid

from felops.domain.job import Job, JobSender, JobStatus, JobType


def test_job_defaults() -> None:
    job = Job(
        workspace_id="ws-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        idempotency_key="key-1",
    )
    assert job.status == JobStatus.QUEUED
    assert job.job_type == JobType.INTERACTIVE
    assert job.locale == "en"
    assert job.sender is None
    assert job.finished_at is None
    assert job.error is None
    assert job.created_at is not None


def test_job_stores_locale_and_sender() -> None:
    sender = JobSender(id="cus_1", name="Sara Ali")
    job = Job(
        workspace_id="ws-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        idempotency_key="key-1",
        locale="ar",
        sender=sender,
    )
    assert job.locale == "ar"
    assert job.sender == sender


def test_job_id_is_uuid() -> None:
    job = Job(
        workspace_id="ws-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        idempotency_key="key-1",
    )
    parsed = uuid.UUID(job.job_id)
    assert str(parsed) == job.job_id


def test_job_status_values() -> None:
    assert {s.value for s in JobStatus} == {
        "queued",
        "running",
        "waiting_approval",
        "done",
        "failed",
    }


def test_job_type_values() -> None:
    assert {t.value for t in JobType} == {
        "interactive",
        "resume",
        "ingest",
    }


def test_job_status_transition() -> None:
    job = Job(
        workspace_id="ws-1",
        agent_id="agent-1",
        conversation_id="conv-1",
        idempotency_key="key-1",
    )
    job.status = JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING
