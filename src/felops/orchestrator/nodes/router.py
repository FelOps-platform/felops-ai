"""Router node — P0 stub (§6.2, §6.3).

Routes every turn to the responder in P0. In the full graph (§6.3),
router conditional edges select retriever, planner, or responder after
guardrails_in; subsequent conditional edges select the reasoner and
downstream execution path. The new architecture has no verifier node.
In P1 (task 1.5) the intent detection is wired here.
In P4 (task 4.7) the small-LLM intent router replaces this stub.

Import rule: only felops.domain and felops.ports are imported here.
"""

from __future__ import annotations

from typing import Any

from felops.domain.state import AgentState, TraceEvent


def router(state: AgentState) -> dict[str, Any]:
    """P0 stub: always route to responder.

    Returns a partial state dict. Only writes to execution_trace
    (appended by operator.add reducer). Messages are left to the responder.

    Future: read state.messages[-1] to classify intent (question /
    task / greeting) and set a route flag that the conditional edge reads.
    """
    trace = TraceEvent(
        event_type="router.routed",
        data={"intent": "stub", "route": "responder"},
    )
    return {"execution_trace": [trace]}
