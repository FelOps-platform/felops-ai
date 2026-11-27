"""Policy Engine v1 (tasks 2.3 and 3.3, §4 Policy).

Rules are evaluated in order:

1. **Allowlist** — the tool must be in the active sub-agent's effective set.
   Every consumer obtains it through ``ToolRegistry.effective_for`` (§4.1
   inv. 6), so a tool the user deselected, a tool the sub-agent does not hold,
   and a tool missing from the catalog or deprecated are all denied.
2. **Approval** — ``requires_approval`` is the user's per-(agent, tool) flag,
   delivered in ``agent_config.tools``.  It applies to every sub-agent that
   holds the tool and is checked *after* the allowlist, so removing a tool
   also revokes its pending approvals (§4.1 inv. 2 and 5).

The AI repo adds no risk classification of its own.
"""

from __future__ import annotations

from felops.domain.agent import AgentRef
from felops.domain.policy import Decision, PolicyDecision
from felops.domain.state import AgentConfig, ToolCall
from felops.services.tool_registry import ToolRegistry


class PolicyEngine:
    """Evaluates calls against the effective set and the approval setting."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def evaluate(
        self, call: ToolCall, agent_ref: AgentRef, agent_config: AgentConfig
    ) -> Decision:
        """Return ``ALLOW``, ``REQUIRE_APPROVAL`` or ``DENY`` for *call*."""
        effective = self._registry.effective_for(agent_ref, agent_config)
        if call.tool_id not in effective:
            return Decision(
                decision=PolicyDecision.DENY,
                reason="out_of_effective_set",
            )
        if self._requires_approval(call.tool_id, agent_config):
            return Decision(
                decision=PolicyDecision.REQUIRE_APPROVAL,
                reason="requires_approval",
            )
        return Decision(
            decision=PolicyDecision.ALLOW,
            reason="in_effective_set",
        )

    @staticmethod
    def _requires_approval(tool_id: str, agent_config: AgentConfig) -> bool:
        return any(
            tool.tool_id == tool_id and tool.requires_approval
            for tool in agent_config.tools
        )


__all__ = ["PolicyEngine"]
