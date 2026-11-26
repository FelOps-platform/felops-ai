"""Workspace isolation types.

WorkspaceContext is set once at ingress (from the /workspaces/{workspace_id}/ path parameter) and
propagated into graph state, background-task context, MCP calls, log and trace
baggage.  Repositories require it as a non-optional argument.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field


class WorkspaceContext(BaseModel, frozen=True):
    """Immutable workspace identity, propagated everywhere"""

    workspace_id: str = Field(min_length=1)


def make_thread_id(workspace_id: str, conversation_id: str) -> str:
    """Return the canonical domain thread identifier."""
    return f"{workspace_id}:{conversation_id}"


class NameSpace(BaseModel, frozen=True):
    """Knowledge namespace: workspace + agent + source .

    Knowledge is agent-level and shared by all of the agent's sub-agents.
    Default: shared collection per embedding model with indexed workspace_id
    payload.  Enterprise tier: dedicated collection.
    """

    workspace_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)


def coerce_domain_id(value: Any) -> str:
    """Normalize wire str or integer IDs at API ingress to canonical strings."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("identifier must be a non-empty string or integer")
    normalized = str(value)
    if not normalized.strip():
        raise ValueError("identifier must not be blank")
    return normalized


IngressId = Annotated[str, BeforeValidator(coerce_domain_id)]
