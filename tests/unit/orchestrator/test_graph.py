"""Unit tests for the P0 walking skeleton graph (task 0.9).

Done-when criterion (execution plan §3):
    "Canned reply via graph" — POST a message, get an AIMessage reply
    back through the graph without a live LLM.

These tests verify:
  1. The graph compiles without errors.
  2. A canned AIMessage is appended to state.messages.
  3. Identity fields (workspace_id, agent_id, sub_agent_id) are preserved.
  4. A trace event is recorded for each node.
  5. State is checkpointed between invocations on the same thread.
  6. Two different threads are fully isolated.
  7. make_thread_config() produces the correct format.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from felops.domain.state import AgentState
from felops.orchestrator.graph import build_graph, make_thread_config

# Fixtures


@pytest.fixture
def graph():
    """Fresh compiled graph with an in-memory checkpointer for each test."""
    return build_graph(checkpointer=MemorySaver())


def _initial_input(
    *,
    workspace_id: str = "ws-test",
    agent_id: str = "agent-test",
    conversation_id: str = "conv-001",
    text: str = "Hello",
) -> tuple[AgentState, dict[str, Any]]:
    """Returns (state_input, thread_config)."""
    state = AgentState(
        workspace_id=workspace_id,
        agent_id=agent_id,
        conversation_id=conversation_id,
        messages=[HumanMessage(content=text)],
    )
    config = make_thread_config(workspace_id, conversation_id)
    return state, config


# Tests


class TestGraphCompiles:
    def test_build_graph_returns_compiled_graph(self):
        """build_graph() must return without raising."""
        g = build_graph()
        assert g is not None

    def test_build_graph_with_explicit_checkpointer(self):
        """Explicit MemorySaver injection must work."""
        g = build_graph(checkpointer=MemorySaver())
        assert g is not None


class TestCannedReply:
    def test_graph_appends_ai_message(self, graph):
        """The graph must add at least one AIMessage to state.messages."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert ai_messages, "Expected at least one AIMessage in messages"

    def test_graph_accepts_dict_input(self, graph):
        """The graph must accept a raw dictionary input and return a canned reply."""
        dict_input = {
            "workspace_id": "ws-dict",
            "agent_id": "agent-dict",
            "messages": [HumanMessage(content="Hello via dict")],
        }
        config = make_thread_config("ws-dict", "conv-dict")
        result = graph.invoke(dict_input, config=config)
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert ai_messages, "Expected canned AIMessage when invoked with dict"
        expected_canned = "I received your message. (stub reply — LLM not wired yet)"
        assert result["messages"][-1].content == expected_canned

    def test_canned_reply_is_non_empty(self, graph):
        """The canned AIMessage must have non-empty content."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert all(m.content for m in ai_msgs), "AIMessage content must not be empty"

    def test_human_message_is_preserved(self, graph):
        """The original HumanMessage must still be in messages after the run."""
        state, config = _initial_input(text="Hello from the user")
        result = graph.invoke(state, config=config)
        human_msgs = [m for m in result["messages"] if isinstance(m, HumanMessage)]
        assert any("Hello from the user" in m.content for m in human_msgs)


class TestStateFieldPropagation:
    def test_workspace_id_preserved(self, graph):
        """workspace_id set at ingress must survive the full graph run."""
        state, config = _initial_input(workspace_id="acme")
        result = graph.invoke(state, config=config)
        assert result["workspace_id"] == "acme"

    def test_agent_id_preserved(self, graph):
        """agent_id set at ingress must survive the full graph run."""
        state, config = _initial_input(agent_id="agent-42")
        result = graph.invoke(state, config=config)
        assert result["agent_id"] == "agent-42"

    def test_sub_agent_id_defaults_to_customer_support(self, graph):
        """sub_agent_id must default to 'customer_support' (V1 constant, §2.3)."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        assert result["sub_agent_id"] == "customer_support"


class TestExecutionTrace:
    def test_trace_events_are_recorded(self, graph):
        """Both router and responder must append TraceEvents."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        assert len(result["execution_trace"]) >= 2, (
            "Expected at least one trace event per node (router + responder)"
        )

    def test_trace_contains_router_event(self, graph):
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        event_types = [e.event_type for e in result["execution_trace"]]
        assert "router.routed" in event_types

    def test_trace_contains_responder_event(self, graph):
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        event_types = [e.event_type for e in result["execution_trace"]]
        assert "message.final" in event_types


class TestCheckpointing:
    def test_state_persists_between_invocations_on_same_thread(self, graph):
        """A second invocation on the same thread_id must load the prior state."""
        state, config = _initial_input(conversation_id="conv-persist")
        graph.invoke(state, config=config)

        # Second turn: only send a new message; prior state loads from checkpoint.
        second_turn = {"messages": [HumanMessage(content="Second message")]}
        result = graph.invoke(second_turn, config=config)

        # Both turns' messages must be in state now.
        all_content = " ".join(m.content for m in result["messages"])
        assert "Hello" in all_content
        assert "Second message" in all_content

    def test_different_threads_are_isolated(self, graph):
        """Two different conversation threads must not share state."""
        state_a, config_a = _initial_input(conversation_id="conv-A", text="Message A")
        state_b, config_b = _initial_input(conversation_id="conv-B", text="Message B")

        graph.invoke(state_a, config=config_a)
        result_b = graph.invoke(state_b, config=config_b)

        # Thread B must not contain Thread A's message.
        all_content_b = " ".join(m.content for m in result_b["messages"])
        assert "Message A" not in all_content_b


class TestThreadConfig:
    def test_make_thread_config_format(self):
        """thread_id must be '{workspace_id}:{conversation_id}' (§2.3)."""
        cfg = make_thread_config("acme", "conv-42")
        assert cfg == {"configurable": {"thread_id": "acme:conv-42"}}

    def test_make_thread_config_different_workspaces(self):
        """Same conversation_id in two workspaces must produce different thread_ids."""
        cfg_a = make_thread_config("workspace-a", "conv-1")
        cfg_b = make_thread_config("workspace-b", "conv-1")
        assert cfg_a["configurable"]["thread_id"] != cfg_b["configurable"]["thread_id"]
