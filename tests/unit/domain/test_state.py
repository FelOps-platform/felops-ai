"""Unit tests for felops.domain.state  AgentState and all value objects."""

import operator

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from felops.domain.agent import HandoffReason
from felops.domain.state import (
    AgentConfig,
    AgentInstructions,
    AgentState,
    AgentToolConfig,
    Approval,
    ApprovalStatus,
    Chunk,
    Plan,
    PlanStep,
    SideEffectClass,
    ToolCall,
    TraceEvent,
)

# PlanStep


class TestPlanStep:
    def test_creation_with_required_field(self):
        step = PlanStep(description="Look up order")
        assert step.description == "Look up order"
        assert step.tool_id is None
        assert step.depends_on == []
        assert step.status == "pending"

    def test_step_id_auto_generated(self):
        a = PlanStep(description="a")
        b = PlanStep(description="b")
        assert a.step_id != b.step_id

    def test_explicit_tool_id(self):
        step = PlanStep(description="lookup", tool_id="shopify_lookup")
        assert step.tool_id == "shopify_lookup"

    def test_valid_statuses(self):
        for status in ("pending", "running", "done", "failed"):
            step = PlanStep(description="x", status=status)  # type: ignore[arg-type]
            assert step.status == status

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            PlanStep(description="x", status="cancelled")  # type: ignore[arg-type]

    def test_frozen(self):
        step = PlanStep(description="x")
        with pytest.raises(ValidationError):
            step.status = "done"  # type: ignore[misc]


# Plan


class TestPlan:
    def test_defaults(self):
        plan = Plan()
        assert plan.steps == []
        assert plan.cursor == 0

    def test_mutable_cursor(self):
        plan = Plan()
        plan.cursor = 1
        assert plan.cursor == 1

    def test_with_steps(self):
        s1 = PlanStep(description="step 1")
        s2 = PlanStep(description="step 2")
        plan = Plan(steps=[s1, s2])
        assert len(plan.steps) == 2


# Chunk


class TestChunk:
    def test_creation(self):
        c = Chunk(chunk_id="c-1", source_id="src-1", content="hello")
        assert c.chunk_id == "c-1"
        assert c.source_id == "src-1"
        assert c.content == "hello"
        assert c.score == 0.0
        assert c.metadata == {}

    def test_score_and_metadata(self):
        c = Chunk(chunk_id="c-1", source_id="s", content="x", score=0.9, metadata={"page": 1})
        assert c.score == 0.9
        assert c.metadata["page"] == 1

    def test_frozen(self):
        c = Chunk(chunk_id="c-1", source_id="s", content="x")
        with pytest.raises(ValidationError):
            c.content = "y"  # type: ignore[misc]


# SideEffectClass


class TestSideEffectClass:
    def test_values(self):
        assert SideEffectClass.READ_ONLY == "read_only"
        assert SideEffectClass.SIDE_EFFECTING == "side_effecting"

    def test_is_str(self):
        assert isinstance(SideEffectClass.READ_ONLY, str)


# ToolCall


class TestToolCall:
    def test_defaults(self):
        tc = ToolCall(tool_id="tool-1", tool_name="shopify_lookup")
        assert tc.tool_id == "tool-1"
        assert tc.tool_name == "shopify_lookup"
        assert tc.arguments == {}
        assert tc.params_hash is None
        assert tc.side_effect_class == SideEffectClass.READ_ONLY
        assert tc.idem_key is None
        assert tc.outcome is None

    def test_call_id_auto_generated(self):
        a = ToolCall(tool_id="a", tool_name="t")
        b = ToolCall(tool_id="a", tool_name="t")
        assert a.call_id != b.call_id

    def test_side_effecting_with_idem_key(self):
        tc = ToolCall(
            tool_id="tool-2",
            tool_name="quickbooks_send",
            side_effect_class=SideEffectClass.SIDE_EFFECTING,
            idem_key="idem-abc123",
        )
        assert tc.side_effect_class == SideEffectClass.SIDE_EFFECTING
        assert tc.idem_key == "idem-abc123"

    def test_read_only_with_idem_key_allowed(self):
        """V2 contract requires/allows idempotency key on read-only tool calls."""
        tc = ToolCall(
            tool_id="tool-1",
            tool_name="shopify_lookup",
            side_effect_class=SideEffectClass.READ_ONLY,
            idem_key="idem-read-123",
        )
        assert tc.side_effect_class == SideEffectClass.READ_ONLY
        assert tc.idem_key == "idem-read-123"

    def test_tool_call_supports_tool_id_and_connector_id(self):
        tc = ToolCall(
            tool_id="vondera__get_product",
            connector_id="vondera",
            arguments={"productId": "prod_123"},
            idem_key="idem-123",
        )
        assert tc.tool_id == "vondera__get_product"
        assert tc.connector_id == "vondera"
        assert tc.arguments == {"productId": "prod_123"}

    def test_mutable(self):
        tc = ToolCall(tool_id="a", tool_name="t")
        tc.outcome = {"status": "ok"}
        assert tc.outcome == {"status": "ok"}


# ApprovalStatus


class TestApprovalStatus:
    def test_all_values(self):
        assert ApprovalStatus.PENDING == "pending"
        assert ApprovalStatus.APPROVED == "approved"
        assert ApprovalStatus.REJECTED == "rejected"
        assert ApprovalStatus.EXPIRED == "expired"

    def test_member_count(self):
        assert len(ApprovalStatus) == 4


# Approval


class TestApproval:
    def test_defaults(self):
        ap = Approval(call_id="c-1", tool_id="tool-1", params_hash="hash-abc")
        assert ap.status == ApprovalStatus.PENDING
        assert ap.tool_id == "tool-1"
        assert ap.action == ""
        assert ap.integration == ""
        assert ap.params == {}
        assert ap.effect_summary == ""
        assert ap.expires_at is None
        assert ap.approver_id is None

    def test_params_snapshot_stored(self):
        """Params snapshot is persisted for UI and approval.required event."""
        snapshot = {"invoice_id": "INV-101", "amount": 250.0}
        ap = Approval(
            call_id="c-1",
            tool_id="tool-1",
            action="Send invoice",
            integration="QuickBooks",
            params=snapshot,
            params_hash="hash-abc",
            effect_summary="Send invoice INV-101 for $250.00",
        )
        assert ap.params == snapshot
        assert ap.action == "Send invoice"
        assert ap.integration == "QuickBooks"
        assert ap.effect_summary == "Send invoice INV-101 for $250.00"

    def test_approval_id_auto_generated(self):
        a = Approval(call_id="c-1", tool_id="a", params_hash="h")
        b = Approval(call_id="c-1", tool_id="a", params_hash="h")
        assert a.approval_id != b.approval_id

    def test_mutable_status(self):
        ap = Approval(call_id="c-1", tool_id="a", params_hash="h")
        ap.status = ApprovalStatus.APPROVED
        ap.approver_id = "user-42"
        assert ap.status == ApprovalStatus.APPROVED
        assert ap.approver_id == "user-42"


# TraceEvent


class TestTraceEvent:
    def test_creation(self):
        te = TraceEvent(event_type="tool.started", data={"call_id": "c-1"})
        assert te.event_type == "tool.started"
        assert te.data["call_id"] == "c-1"

    def test_timestamp_auto_set(self):
        te = TraceEvent(event_type="x")
        assert te.timestamp is not None

    def test_frozen(self):
        te = TraceEvent(event_type="x")
        with pytest.raises(ValidationError):
            te.event_type = "y"  # type: ignore[misc]

    def test_operator_add_reducer(self):
        """execution_trace uses operator.add  plain list concatenation."""
        existing = [TraceEvent(event_type="a")]
        new = [TraceEvent(event_type="b")]
        result = operator.add(existing, new)
        assert len(result) == 2
        assert result[0].event_type == "a"
        assert result[1].event_type == "b"


# AgentConfig


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.name == ""
        assert cfg.description is None
        assert cfg.instructions.role == ""
        assert cfg.tools == []
        assert cfg.sub_agents == ["customer_support"]

    def test_custom_values(self):
        cfg = AgentConfig(
            name="Customer Experience",
            description="Support agent",
            instructions=AgentInstructions(
                role="Help customers",
                language_constraints="No refunds without approval",
                response_format="Concise",
            ),
            tools=[
                AgentToolConfig(tool_id="get_order", requires_approval=False),
                AgentToolConfig(tool_id="refund_order", requires_approval=True),
            ],
            sub_agents=["customer_support"],
        )
        assert cfg.name == "Customer Experience"
        assert len(cfg.tools) == 2
        assert cfg.tools[1].requires_approval is True
        assert cfg.instructions.role == "Help customers"


# AgentState


class TestAgentState:
    def test_minimal_creation(self):
        state = AgentState(workspace_id="ws-1", agent_id="ag-1")
        assert state.workspace_id == "ws-1"
        assert state.agent_id == "ag-1"

    def test_v1_defaults(self):
        state = AgentState(workspace_id="ws-1", agent_id="ag-1")
        assert state.sub_agent_id == "customer_support"
        assert state.audience == "external"
        assert state.channel == ""
        assert state.conversation_id == ""
        assert state.agent_config is None

    def test_empty_collections_by_default(self):
        state = AgentState(workspace_id="ws-1", agent_id="ag-1")
        assert state.messages == []
        assert state.retrieved_context == []
        assert state.tool_calls == []
        assert state.pending_approvals == []
        assert state.execution_trace == []

    def test_none_fields_by_default(self):
        state = AgentState(workspace_id="ws-1", agent_id="ag-1")
        assert state.plan is None
        assert state.handoff is None

    def test_confidence_default(self):
        state = AgentState(workspace_id="ws-1", agent_id="ag-1")
        assert state.confidence == 0.0

    def test_messages_accepts_base_messages(self):
        state = AgentState(workspace_id="ws-1", agent_id="ag-1")
        state.messages = [HumanMessage(content="hello")]
        assert len(state.messages) == 1
        assert state.messages[0].content == "hello"

    def test_add_messages_reducer_merges_and_deduplicates(self):
        """LangGraph add_messages appends distinct messages and updates same-id messages."""
        m1 = HumanMessage(content="first", id="msg-1")
        m2 = AIMessage(content="second", id="msg-2")
        merged = add_messages([m1], [m2])
        assert len(merged) == 2
        assert merged[0].content == "first"
        assert merged[1].content == "second"

        # Updating msg-1 in place
        m1_updated = HumanMessage(content="updated first", id="msg-1")
        updated = add_messages(merged, [m1_updated])
        assert len(updated) == 2
        assert updated[0].content == "updated first"
        assert updated[1].content == "second"

    def test_messages_with_human_and_ai(self):
        human = HumanMessage(content="Where is my order?")
        ai = AIMessage(content="Let me check.")
        state = AgentState(
            workspace_id="ws-1",
            agent_id="ag-1",
            messages=[human, ai],
        )
        assert len(state.messages) == 2
        assert state.messages[0].content == "Where is my order?"
        assert state.messages[1].content == "Let me check."

    def test_handoff_reason_accepted(self):
        state = AgentState(
            workspace_id="ws-1",
            agent_id="ag-1",
            handoff=HandoffReason.LOW_CONFIDENCE,
        )
        assert state.handoff == HandoffReason.LOW_CONFIDENCE

    def test_invalid_audience_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(workspace_id="ws-1", agent_id="ag-1", audience="public")  # type: ignore[arg-type]

    def test_full_state_serialization(self):
        state = AgentState(
            workspace_id="ws-1",
            agent_id="ag-1",
            conversation_id="conv-1",
            messages=[HumanMessage(content="hi")],
        )
        d = state.model_dump()
        assert d["workspace_id"] == "ws-1"
        assert d["agent_id"] == "ag-1"
        assert d["sub_agent_id"] == "customer_support"
        assert isinstance(d["messages"], list)

    def test_plan_can_be_set(self):
        plan = Plan(steps=[PlanStep(description="step 1")])
        state = AgentState(workspace_id="ws-1", agent_id="ag-1", plan=plan)
        assert state.plan is not None
        assert len(state.plan.steps) == 1

    def test_agent_config_can_be_set(self):
        cfg = AgentConfig(name="Support")
        state = AgentState(workspace_id="ws-1", agent_id="ag-1", agent_config=cfg)
        assert state.agent_config is not None
        assert state.agent_config.name == "Support"
