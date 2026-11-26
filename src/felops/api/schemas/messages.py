"""Request and response schemas for POST /messages (§3.1 of contract)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from felops.domain.state import AgentConfig


class Sender(BaseModel):
    """Message sender identity."""

    id: str = Field(min_length=1)
    name: str = ""


class SubmitMessageRequest(BaseModel):
    """Body of POST /workspaces/{workspace_id}/conversations/{id}/messages.

    Mirrors §3.1 of ai-service-contract.md.
    """

    agent_id: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    sender: Sender
    content: str = Field(min_length=1)
    locale: str = "en"
    agent_config: AgentConfig


class SubmitMessageResponse(BaseModel):
    """202 response body."""

    job_id: str
