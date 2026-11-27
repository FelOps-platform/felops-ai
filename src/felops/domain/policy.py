"""Policy decisions (§4 Policy).

The Policy Engine returns one of three outcomes per proposed call:

* ``ALLOW`` — the call is inside the sub-agent's effective tool set.
* ``REQUIRE_APPROVAL`` — the user marked the tool ``requires_approval`` in
  ``agent_config`` (added in task 3.3).
* ``DENY`` — the call is outside the effective set; it can never be planned,
  approved or executed (§4.1).

The AI repo adds no risk classification of its own: the allowlist is the
effective set and approval is the user's per-(agent, tool) setting.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class PolicyDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class Decision(BaseModel, frozen=True):
    """The policy outcome for one proposed call, with an audit reason."""

    decision: PolicyDecision
    reason: str = Field(min_length=1)


__all__ = ["Decision", "PolicyDecision"]
