"""Walking skeleton graph — P0, task 0.9 (§6.3).

Implements the minimal graph:
    START → router → responder → END

This is the structural foundation every future node plugs into.
The checkpointer is injected so tests use MemorySaver and production
uses the Postgres checkpointer (task 3.1).

thread_id convention (§2.3): "{workspace_id}:{conversation_id}"
Use make_thread_config() everywhere — do not construct thread_id ad-hoc.

Future nodes (see §6.3 for the full graph):
  P1: guardrails_in, retriever, reasoner, guardrails_out, memory_writer
  P2: tool_selector, policy_check, executor
  P3: planner, approval_gate
  P10 seam: sub_agent_selector on the guardrails_in → router edge (§6.6)
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from felops.domain.state import AgentState
from felops.orchestrator.nodes.responder import responder
from felops.orchestrator.nodes.router import router


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile and return the agent state graph.

    Args:
        checkpointer: Checkpoint saver to use. Defaults to MemorySaver
            (in-memory, suitable for tests and the P0 walking skeleton).
            Pass a Postgres checkpointer in production (task 3.1).

    Returns:
        A compiled LangGraph ready to invoke or stream.

    P0 graph topology:
        START → router → responder → END

    Full topology (§6.3 / §6.4 of new architecture):

        START → guardrails_in
                  →(pass)    router
                               →(intent=question)              retriever → reasoner
                               →(intent=task)                  planner
                               →(greeting/needs clarification) responder
                               →(human_control)                END
                  →(blocked) responder

        planner → retriever (step needs knowledge)
                → tool_selector (step is tool call)
        retriever → reasoner (always)
        reasoner  → tool_selector (needs tool)
                  → responder    (answer ready; handoff=low_confidence if confidence < τ)
        tool_selector → policy_check (always)
        policy_check  → executor     (allow)
                      → approval_gate → executor / responder (approved / rejected|expired)
                      → responder    (deny)
        executor → planner   (succeeded, steps remain)
                 → responder (succeeded, plan done)
                 → responder (failed, handoff=tool_failed)
        responder → guardrails_out → memory_writer → END

    Note: the verifier node is eliminated in the new architecture (§6.3).
    The backend owns tool verification and returns a final succeeded/failed
    execution status (§4.2 of ai-service-contract.md). executor routes
    directly to planner (steps remain), responder (plan done), or responder
    with handoff=tool_failed (execution failed).

    §6.6 seam (P10): insert sub_agent_selector on the
        guardrails_in → router edge. No other code changes.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("router", router)
    builder.add_node("responder", responder)

    # Edges
    # P0: unconditional path through the two nodes.
    # P1: replace with conditional edges on router output (intent classification).
    builder.add_edge(START, "router")
    builder.add_edge("router", "responder")
    builder.add_edge("responder", END)

    # Compile
    return builder.compile(checkpointer=checkpointer)


def make_thread_config(workspace_id: str, conversation_id: str) -> dict[str, Any]:
    """Return a LangGraph config dict with the canonical thread_id.

    thread_id format: "{workspace_id}:{conversation_id}" (§2.3).
    Always use this function — never construct thread_id inline.

    Example:
        config = make_thread_config("acme", "conv-42")
        graph.invoke(initial_input, config=config)
    """
    thread_id = f"{workspace_id}:{conversation_id}"
    return {"configurable": {"thread_id": thread_id}}
