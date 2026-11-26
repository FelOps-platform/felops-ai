"""POST /workspaces/{workspace_id}/conversations/{id}/messages route.

Task 1.7 — §2.2, §7.1, §3.1 of contract.

This module contains ONLY the HTTP handler. All business logic lives in
felops.services.message_service.MessageService.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Path, status
from langgraph.graph.state import CompiledStateGraph

from felops.api.deps import get_graph, get_message_service, get_workspace_context
from felops.api.schemas.messages import SubmitMessageRequest, SubmitMessageResponse
from felops.domain.job import JobSender
from felops.domain.workspace import WorkspaceContext
from felops.services.message_service import (
    ConversationBusy,
    DuplicateIdempotencyKey,
    MessageService,
    SubmitMessageCommand,
)

router = APIRouter()


async def _build_submit_command(
    body: SubmitMessageRequest,
    workspace: Annotated[WorkspaceContext, Depends(get_workspace_context)],
    conversation_id: Annotated[str, Path(description="Conversation identifier", min_length=1)],
    idempotency_key: Annotated[
        str,
        Header(alias="Idempotency-Key", min_length=1),
    ],
) -> SubmitMessageCommand:
    if not idempotency_key.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Idempotency-Key header cannot be empty.",
        )
    return SubmitMessageCommand(
        workspace=workspace,
        conversation_id=conversation_id,
        idempotency_key=idempotency_key,
        agent_id=body.agent_id,
        channel=body.channel,
        content=body.content,
        locale=body.locale,
        sender=JobSender(id=body.sender.id, name=body.sender.name),
        agent_config=body.agent_config,
    )


@router.post(
    "/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=SubmitMessageResponse,
    summary="Submit a message to a conversation",
    description=(
        "Accepts a customer message and starts an async graph run. "
        "Returns 202 with a job_id. Duplicate Idempotency-Key returns "
        "the same job_id."
    ),
)
async def submit_message(
    command: Annotated[SubmitMessageCommand, Depends(_build_submit_command)],
    graph: Annotated[CompiledStateGraph, Depends(get_graph)],
    message_service: Annotated[MessageService, Depends(get_message_service)],
) -> SubmitMessageResponse:
    """Handle POST /workspaces/{workspace_id}/conversations/{conversation_id}/messages."""
    try:
        job = await message_service.submit_message(command, graph=graph)
    except DuplicateIdempotencyKey as exc:
        # Return the same job_id — not an error (§7.1 contract).
        return SubmitMessageResponse(job_id=exc.existing_job.job_id)
    except ConversationBusy as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(f"A job is already running on this conversation: {exc.existing_job.job_id}"),
        ) from exc

    return SubmitMessageResponse(job_id=job.job_id)
