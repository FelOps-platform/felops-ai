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
  8. agent_config (AgentConfig model and nested dict) is preserved through
     graph execution and across checkpointed turns on the same thread.
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

    def test_build_graph_no_args_is_invocable(self):
        """build_graph() with no args (default MemorySaver) must produce a graph
        that can be invoked end-to-end without error."""
        g = build_graph()
        state, config = _initial_input()
        result = g.invoke(state, config=config)
        assert result is not None
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert ai_msgs, "Expected at least one AIMessage from default-checkpointer graph"


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
        assert ai_msgs, "Expected at least one AIMessage in messages"
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

    def test_sub_agent_id_is_preserved(self, graph):
        """sub_agent_id set at AgentState construction must survive the full graph run."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        assert result["sub_agent_id"] == "customer_support"


class TestAgentConfigPropagation:
    """agent_config is received from backend per-request (§3.1 of contract).
    The graph must preserve it in state and in checkpoints."""

    def test_agent_config_model_is_preserved_through_graph(self, graph):
        """AgentConfig typed model passed in state must survive the full graph run."""
        from felops.domain.state import AgentConfig, AgentInstructions

        config_obj = AgentConfig(
            name="Customer Experience",
            instructions=AgentInstructions(
                role="Help customers resolve order questions.",
                language_constraints="Do not issue refunds without approval.",
                response_format="Concise and friendly.",
            ),
        )
        state = AgentState(
            workspace_id="ws-cfg-test",
            agent_id="agent-cfg-test",
            conversation_id="conv-cfg-001",
            messages=[HumanMessage(content="Hello")],
            agent_config=config_obj,
        )
        config = make_thread_config("ws-cfg-test", "conv-cfg-001")
        result = graph.invoke(state, config=config)

        result_cfg = result["agent_config"]
        assert result_cfg is not None, "agent_config must not be None after graph run"
        if isinstance(result_cfg, dict):
            assert result_cfg["name"] == "Customer Experience"
            assert result_cfg["instructions"]["role"] == "Help customers resolve order questions."
        else:
            assert result_cfg.name == "Customer Experience"
            assert result_cfg.instructions.role == "Help customers resolve order questions."

    def test_agent_config_as_nested_dict_is_preserved(self, graph):
        """agent_config passed as a raw nested dict (as LangGraph deserializes
        checkpoints) must be preserved through graph execution."""
        dict_config = {
            "name": "Support Bot",
            "instructions": {
                "role": "Support role.",
                "language_constraints": "",
                "response_format": "",
            },
            "tools": [{"tool_id": "vondera__get_product", "requires_approval": False}],
            "sub_agents": ["customer_support"],
        }
        input_dict = {
            "workspace_id": "ws-dict-cfg",
            "agent_id": "agent-dict-cfg",
            "conversation_id": "conv-dict-cfg",
            "messages": [HumanMessage(content="Hello from dict")],
            "agent_config": dict_config,
        }
        config = make_thread_config("ws-dict-cfg", "conv-dict-cfg")
        result = graph.invoke(input_dict, config=config)

        result_cfg = result["agent_config"]
        assert result_cfg is not None, "agent_config must survive a dict input run"
        if isinstance(result_cfg, dict):
            assert result_cfg["name"] == "Support Bot"
        else:
            assert result_cfg.name == "Support Bot"

    def test_agent_config_defaults_to_none_when_not_provided(self, graph):
        """When no agent_config is passed, state.agent_config must be None."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        assert result.get("agent_config") is None

    def test_agent_config_preserved_across_checkpointed_turns(self, graph):
        """agent_config passed in turn 1 must be restored from the checkpoint
        and present in turn 2's result state on the same thread."""
        from felops.domain.state import AgentConfig, AgentInstructions

        config_obj = AgentConfig(
            name="Multi-Turn Agent",
            instructions=AgentInstructions(role="Multi-turn role."),
        )
        state = AgentState(
            workspace_id="ws-chk",
            agent_id="agent-chk",
            conversation_id="conv-chk-persist",
            messages=[HumanMessage(content="Turn 1")],
            agent_config=config_obj,
        )
        thread_config = make_thread_config("ws-chk", "conv-chk-persist")

        graph.invoke(state, config=thread_config)
        second_turn = {"messages": [HumanMessage(content="Turn 2")]}
        result = graph.invoke(second_turn, config=thread_config)

        all_content = " ".join(m.content for m in result["messages"])
        assert "Turn 1" in all_content
        assert "Turn 2" in all_content

        result_cfg = result["agent_config"]
        assert result_cfg is not None, "agent_config must be restored from checkpoint on turn 2"
        if isinstance(result_cfg, dict):
            assert result_cfg["name"] == "Multi-Turn Agent"
        else:
            assert result_cfg.name == "Multi-Turn Agent"

    def test_agent_config_with_tools_list_preserved(self, graph):
        """AgentConfig with a non-empty tools list must be preserved correctly."""
        from felops.domain.state import AgentConfig, AgentToolConfig

        config_obj = AgentConfig(
            name="Tool Agent",
            tools=[
                AgentToolConfig(tool_id="vondera__get_product", requires_approval=False),
                AgentToolConfig(tool_id="vondera__create_invoice", requires_approval=True),
            ],
        )
        state = AgentState(
            workspace_id="ws-tools",
            agent_id="agent-tools",
            conversation_id="conv-tools-001",
            messages=[HumanMessage(content="Check tools")],
            agent_config=config_obj,
        )
        config = make_thread_config("ws-tools", "conv-tools-001")
        result = graph.invoke(state, config=config)

        result_cfg = result["agent_config"]
        assert result_cfg is not None
        tools = result_cfg.get("tools", []) if isinstance(result_cfg, dict) else result_cfg.tools
        assert len(tools) == 2, "Both tools must be preserved in agent_config"
        tool_ids = [t["tool_id"] if isinstance(t, dict) else t.tool_id for t in tools]
        assert tool_ids == ["vondera__get_product", "vondera__create_invoice"]
        approval_flags = [
            t["requires_approval"] if isinstance(t, dict) else t.requires_approval for t in tools
        ]
        assert approval_flags == [False, True]

    def test_agent_config_updated_on_subsequent_turn(self, graph):
        """Under contract §3.1, a subsequent request may provide an updated agent_config,
        which updates the run's config state."""
        from felops.domain.state import AgentConfig, AgentInstructions

        cfg_turn1 = AgentConfig(
            name="V1 Bot",
            instructions=AgentInstructions(role="Initial role"),
        )
        state_turn1 = AgentState(
            workspace_id="ws-update",
            agent_id="agent-update",
            conversation_id="conv-update-001",
            messages=[HumanMessage(content="Turn 1")],
            agent_config=cfg_turn1,
        )
        thread_config = make_thread_config("ws-update", "conv-update-001")
        res1 = graph.invoke(state_turn1, config=thread_config)
        cfg1 = res1["agent_config"]
        assert (cfg1["name"] if isinstance(cfg1, dict) else cfg1.name) == "V1 Bot"

        # Turn 2 sends an updated config
        cfg_turn2 = AgentConfig(
            name="V2 Bot",
            instructions=AgentInstructions(role="Updated role"),
        )
        input_turn2 = {
            "messages": [HumanMessage(content="Turn 2")],
            "agent_config": cfg_turn2,
        }
        res2 = graph.invoke(input_turn2, config=thread_config)
        cfg2 = res2["agent_config"]
        assert (cfg2["name"] if isinstance(cfg2, dict) else cfg2.name) == "V2 Bot"
        role2 = cfg2["instructions"]["role"] if isinstance(cfg2, dict) else cfg2.instructions.role
        assert role2 == "Updated role", "instructions.role must reflect the updated config"


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

    def test_message_final_data_has_contract_keys(self, graph):
        """message.final TraceEvent.data must match the wire contract (§7.2, §9):
        keys: exact set {message_id, text, citations}.
        message_id must match the appended AIMessage id and be unique across turns."""
        state, config = _initial_input()
        result = graph.invoke(state, config=config)
        final_events = [e for e in result["execution_trace"] if e.event_type == "message.final"]
        assert final_events, "Expected a message.final trace event"
        data = final_events[-1].data

        # Exact key set assertion (§7.2, §9)
        assert set(data.keys()) == {"message_id", "text", "citations"}
        assert isinstance(data["message_id"], str) and len(data["message_id"]) > 0
        assert data["text"] == "I received your message. (stub reply — LLM not wired yet)"
        assert data["citations"] == []

        # message_id matches the AIMessage id
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert ai_msgs[-1].id == data["message_id"]

        # Envelope fields must NOT be in data
        assert "workspace_id" not in data
        assert "agent_id" not in data
        assert "sub_agent_id" not in data

        # Across multiple turns, message_ids must be unique
        second_turn = {"messages": [HumanMessage(content="Another turn")]}
        result2 = graph.invoke(second_turn, config=config)
        final_events2 = [e for e in result2["execution_trace"] if e.event_type == "message.final"]
        assert len(final_events2) == 2
        assert final_events2[0].data["message_id"] != final_events2[1].data["message_id"]


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

    def test_invoke_without_thread_id_raises_value_error(self):
        """graph.invoke() with a checkpointer but no thread_id config must raise ValueError."""
        g = build_graph()
        state, _ = _initial_input()
        with pytest.raises(ValueError, match="thread_id"):
            g.invoke(state)
