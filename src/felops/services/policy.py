"""Policy Engine v0 (task 2.3, §4 Policy).

The allowlist is the active sub-agent's effective tool set.  Every consumer
obtains it through ``ToolRegistry.effective_for`` (§4.1 inv. 6), so a tool the
user deselected, a tool the sub-agent does not hold, and a tool missing from
the catalog or deprecated are all denied by the same check.

``requires_approval`` is the next rule (3.3); v0 only allows or denies.
"""

from __future__ import annotations

from felops.domain.agent import AgentRef
from felops.domain.policy import Decision, PolicyDecision
from felops.domain.state import AgentConfig, ToolCall
from felops.services.tool_registry import ToolRegistry


class PolicyEngine:
    """Evaluates calls against the sub-agent's effective tool set."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def evaluate(
        self, call: ToolCall, agent_ref: AgentRef, agent_config: AgentConfig
    ) -> Decision:
        """Return ``ALLOW`` only when *call* is inside the effective set."""
        effective = self._registry.effective_for(agent_ref, agent_config)
        if call.tool_id not in effective:
            return Decision(
                decision=PolicyDecision.DENY,
                reason="out_of_effective_set",
            )
        return Decision(
            decision=PolicyDecision.ALLOW,
            reason="in_effective_set",
        )


__all__ = ["PolicyEngine"]
