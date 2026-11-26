"""Unit tests for felops.services.message_service.MessageService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage

from felops.domain.agent import DEFAULT_AUDIENCE, DEFAULT_SUB_AGENT_ID
from felops.domain.job import JobSender, JobStatus
from felops.domain.state import AgentConfig, AgentInstructions, AgentState
from felops.domain.workspace import WorkspaceContext, make_thread_id
from felops.services.message_service import (
    ConversationBusy,
    DuplicateIdempotencyKey,
    MessageService,
)


def _make_graph(ainvoke_impl=None) -> MagicMock:
    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=ainvoke_impl)
    return graph


def _make_config() -> AgentConfig:
    return AgentConfig(
        name="Customer Experience",
        instructions=AgentInstructions(role="Help customers."),
    )


async def _drain_tasks(service: MessageService) -> None:
    if service._tasks:
        await asyncio.gather(*list(service._tasks), return_exceptions=True)


@pytest.mark.asyncio
async def test_submit_returns_job_with_queued_status() -> None:
    service = MessageService()
    gate = asyncio.Event()

    async def slow_graph(*args, **kwargs):
        await gate.wait()
        return {}

    graph = _make_graph(slow_graph)
    job = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    assert job.status == JobStatus.QUEUED
    gate.set()
    await _drain_tasks(service)


@pytest.mark.asyncio
async def test_submit_returns_job_id_in_response() -> None:
    service = MessageService()
    graph = _make_graph()
    job = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    assert isinstance(job.job_id, str)
    assert len(job.job_id) > 0
    await _drain_tasks(service)


@pytest.mark.asyncio
async def test_submit_stores_job_retrievable_by_id() -> None:
    service = MessageService()
    graph = _make_graph()
    job = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    assert service.get_job(job.job_id) is job
    await _drain_tasks(service)


@pytest.mark.asyncio
async def test_duplicate_idempotency_key_raises() -> None:
    service = MessageService()
    graph = _make_graph()
    first = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-dup",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    with pytest.raises(DuplicateIdempotencyKey) as exc_info:
        await service.submit_message(
            workspace=WorkspaceContext(workspace_id="ws-1"),
            conversation_id="conv-1",
            idempotency_key="key-dup",
            agent_id="agent-1",
            channel="widget",
            content="Changed content",
            locale="en",
            agent_config=_make_config(),
            graph=graph,
        )
    assert exc_info.value.existing_job is first
    await _drain_tasks(service)
    assert graph.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_duplicate_key_different_workspace_ok() -> None:
    service = MessageService()
    graph = _make_graph()
    job1 = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="shared-key",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    job2 = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-2"),
        conversation_id="conv-1",
        idempotency_key="shared-key",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    assert job1.job_id != job2.job_id
    await _drain_tasks(service)


@pytest.mark.asyncio
async def test_conversation_busy_raises() -> None:
    service = MessageService()
    gate = asyncio.Event()

    async def slow_graph(*args, **kwargs):
        await gate.wait()
        return {}

    graph = _make_graph(slow_graph)
    first = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="First",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    with pytest.raises(ConversationBusy) as exc_info:
        await service.submit_message(
            workspace=WorkspaceContext(workspace_id="ws-1"),
            conversation_id="conv-1",
            idempotency_key="key-2",
            agent_id="agent-1",
            channel="widget",
            content="Second",
            locale="en",
            agent_config=_make_config(),
            graph=graph,
        )
    assert exc_info.value.existing_job is first
    gate.set()
    await _drain_tasks(service)


@pytest.mark.asyncio
async def test_conversation_accepts_new_job_after_completion() -> None:
    service = MessageService()
    graph = _make_graph()
    job1 = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="First",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    assert job1.status == JobStatus.DONE

    job2 = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-2",
        agent_id="agent-1",
        channel="widget",
        content="Second",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    assert job2.job_id != job1.job_id
    assert job2.status == JobStatus.DONE


@pytest.mark.asyncio
async def test_graph_runs_in_background() -> None:
    service = MessageService()
    graph = _make_graph()
    await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    assert graph.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_job_status_transitions_to_done() -> None:
    service = MessageService()
    graph = _make_graph()
    job = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    assert job.status == JobStatus.DONE
    assert job.finished_at is not None
    assert job.error is None


@pytest.mark.asyncio
async def test_job_status_transitions_to_failed_on_error() -> None:
    service = MessageService()
    graph = _make_graph(RuntimeError("boom"))
    job = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-1",
        agent_id="agent-1",
        channel="widget",
        content="Hello",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    assert job.status == JobStatus.FAILED
    assert job.finished_at is not None
    assert job.error == "boom"


@pytest.mark.asyncio
async def test_initial_state_has_correct_fields() -> None:
    service = MessageService()
    graph = _make_graph()
    cfg = _make_config()
    await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-42"),
        conversation_id="conv-99",
        idempotency_key="key-1",
        agent_id="agent-7",
        channel="whatsapp",
        content="Where is my order?",
        locale="en",
        agent_config=cfg,
        graph=graph,
    )
    await _drain_tasks(service)
    state: AgentState = graph.ainvoke.call_args.args[0]
    assert isinstance(state, AgentState)
    assert state.workspace_id == "ws-42"
    assert state.agent_id == "agent-7"
    assert state.sub_agent_id == DEFAULT_SUB_AGENT_ID
    assert state.audience == DEFAULT_AUDIENCE
    assert state.channel == "whatsapp"
    assert state.conversation_id == "conv-99"
    assert state.agent_config == cfg
    assert len(state.messages) == 1
    assert isinstance(state.messages[0], HumanMessage)
    assert state.messages[0].content == "Where is my order?"


@pytest.mark.asyncio
async def test_thread_config_uses_canonical_thread_id() -> None:
    service = MessageService()
    graph = _make_graph()
    await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-42"),
        conversation_id="conv-99",
        idempotency_key="key-1",
        agent_id="agent-7",
        channel="widget",
        content="Hi",
        locale="en",
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    thread_cfg = graph.ainvoke.call_args.kwargs["config"]
    assert thread_cfg == {"configurable": {"thread_id": make_thread_id("ws-42", "conv-99")}}


@pytest.mark.asyncio
async def test_submit_stores_locale_and_sender_on_job() -> None:
    service = MessageService()
    graph = _make_graph()
    sender = JobSender(id="cus_123", name="Sara Ali")
    job = await service.submit_message(
        workspace=WorkspaceContext(workspace_id="ws-1"),
        conversation_id="conv-1",
        idempotency_key="key-locale-sender",
        agent_id="agent-1",
        channel="widget",
        content="مرحبا",
        locale="ar",
        sender=sender,
        agent_config=_make_config(),
        graph=graph,
    )
    await _drain_tasks(service)
    assert job.locale == "ar"
    assert job.sender == sender
