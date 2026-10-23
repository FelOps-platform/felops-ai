"""Workspace isolation types.

WorkspaceContext is set once at ingress (from the X-Workspace-Id header) and
propagated into graph state, background-task context, MCP calls, log and trace
baggage.  Repositories require it as a non-optional argument.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WorkspaceContext(BaseModel, frozen=True):
    """Immutable workspace identity, propagated everywhere """

    workspace_id: str = Field(min_length=1)


class NameSpace(BaseModel, frozen=True):
    """Knowledge namespace: workspace + agent + source .

    Knowledge is agent-level and shared by all of the agent's sub-agents.
    Default: shared collection per embedding model with indexed workspace_id
    payload.  Enterprise tier: dedicated collection.
    """

    workspace_id: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
