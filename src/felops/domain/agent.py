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

from pydantic import BaseModel, Field

# V1 constants

DEFAULT_SUB_AGENT_ID: str = "customer_support"
DEFAULT_AUDIENCE: Literal["external"] = "external"


# Domain types


class AgentRef(BaseModel, frozen=True):
    """Resolved agent + sub-agent identity.

    Frozen and hashable so it can be used as a dict key (e.g. in the
    Tool Registry effective-set cache keyed by AgentRef + config_version).

    V1: sub_agent_id is always 'customer_support', audience 'external'.
    The type is future-proof for multi-sub-agent without a schema change.
    """

    agent_id: str = Field(min_length=1)
    sub_agent_id: str = Field(default=DEFAULT_SUB_AGENT_ID, min_length=1)
    audience: Literal["external", "internal"] = Field(default=DEFAULT_AUDIENCE)


class HandoffReason(StrEnum):
    """Reasons the agent hands off to a human.

    LOW_CONFIDENCE       confidence < τ or retrieval miss on a policy-sensitive topic.
    VERIFY_FAILED        verifier detected a mismatch; no re-verify, no re-execute.
    ASSISTANCE_REQUESTED  explicit agent request for human assistance.
    """

    LOW_CONFIDENCE = "low_confidence"
    VERIFY_FAILED = "verify_failed"
    ASSISTANCE_REQUESTED = "assistance_requested"
