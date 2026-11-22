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

from felops.domain.state import AgentState


def router(state: AgentState) -> dict[str, Any]:
    """P0 stub: always route to responder.

    P0 writes nothing to state; trace is responder-only. The old route
    breadcrumb was emitted as a router route event, which has been dropped
    entirely.

    Future: read state.messages[-1] to classify intent (question /
    task / greeting) and set a route flag that the conditional edge reads.

    TODO(task-1.5): add a typed ``route`` field to AgentState and conditional
    edges; the old route breadcrumb lived in the dropped router route event.
    """
    return {}
