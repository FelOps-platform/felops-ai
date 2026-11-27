"""Agent identity and handoff types.

AgentRef is resolved at ingress by the submit-message use case and pinned in
graph state, checkpoint, events and usage records for the lifetime of the run.

V1 constants:
  sub_agent_id = "customer_support"  (the only sub-agent in V1)
  audience     = "external"          (all channels are external in V1)
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# V1 constants

DEFAULT_SUB_AGENT_ID: str = "customer_support"
DEFAULT_AUDIENCE: Literal["external"] = "external"


# Domain types


class AgentRef(BaseModel, frozen=True):
    """Resolved agent + sub-agent identity.

    Frozen and hashable so it can be used as a dict key (e.g. in the
    Tool Registry effective-set cache keyed by AgentRef).

    V1: sub_agent_id is always 'customer_support', audience 'external'.
    The type is future-proof for multi-sub-agent without a schema change.
    """

    agent_id: str = Field(min_length=1)
    sub_agent_id: str = Field(default=DEFAULT_SUB_AGENT_ID, min_length=1)
    audience: Literal["external", "internal"] = Field(default=DEFAULT_AUDIENCE)


class SubAgentSpec(BaseModel):
    """A sub-agent manifest (§4.2).

    V1 ships one manifest, ``customer_support``.  ``predefined_tool_ids`` is
    the platform-owned set ``P_s`` that the Tool Registry intersects with the
    user's agent-level selection ``S``; it is versioned with the code and never
    leaves the AI repo (§4.1 inv. 3).

    Manifests are repo-owned, so unknown keys and a missing tool list are
    rejected instead of silently producing an empty predefined set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sub_agent_id: str = Field(min_length=1)
    audience: Literal["external", "internal"] = Field(default=DEFAULT_AUDIENCE)
    predefined_tool_ids: tuple[str, ...]

    @model_validator(mode="after")
    def _check_predefined_tool_ids(self) -> SubAgentSpec:
        seen: set[str] = set()
        for tool_id in self.predefined_tool_ids:
            if not tool_id:
                raise ValueError("predefined_tool_ids must not contain empty ids")
            if tool_id in seen:
                raise ValueError(f"duplicate predefined tool_id: {tool_id}")
            seen.add(tool_id)
        return self


class HandoffReason(StrEnum):
    """Reasons the agent hands off to a human.

    LOW_CONFIDENCE          confidence < τ or retrieval miss on a policy-sensitive topic.
    TOOL_FAILED             tool execution failure or exhausted retries on Backend.
    CONNECTOR_NOT_CONNECTED connector is not connected or authorization is invalid.
    ASSISTANCE_REQUESTED    explicit agent request for human assistance.
    """

    LOW_CONFIDENCE = "low_confidence"
    TOOL_FAILED = "tool_failed"
    CONNECTOR_NOT_CONNECTED = "connector_not_connected"
    ASSISTANCE_REQUESTED = "assistance_requested"
