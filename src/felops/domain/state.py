"""LangGraph state schema and supporting value objects.

AgentState is the single object that flows through every graph node.
It is a Pydantic v2 BaseModel.

Message handling uses LangGraph's canonical add_messages reducer together with
langchain_core.messages.BaseMessage, so task 0.9 (walking skeleton graph) can
wire nodes without any state-layer changes.
"""

from __future__ import annotations

import operator
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from felops.domain.agent import (
    DEFAULT_AUDIENCE,
    DEFAULT_SUB_AGENT_ID,
    HandoffReason,
)

# Supporting value objects


class PlanStep(BaseModel, frozen=True):
    """A single step in a plan.

    status transitions: pending → running → done | failed
    """

    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    tool_id: str | None = None
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed"] = "pending"


class Plan(BaseModel):
    """Bounded plan over the active sub-agent's effective tools.

    cursor tracks which step is currently executing.
    """

    steps: list[PlanStep] = Field(default_factory=list)
    cursor: int = 0


class Chunk(BaseModel, frozen=True):
    """A retrieved knowledge chunk with citation """

    chunk_id: str
    source_id: str
    content: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentInstructions(BaseModel):
    """Agent instruction slots populated from wizard step 1."""

    role: str = ""
    language_constraints: str = ""
    response_format: str = ""


class AgentToolConfig(BaseModel):
    """Tool selection S with per-tool user approval toggle."""

    tool_id: str
    requires_approval: bool = False


class AgentConfig(BaseModel):
    """Run-scoped agent configuration delivered per-request by the Backend."""

    name: str = ""
    description: str | None = None
    instructions: AgentInstructions = Field(default_factory=AgentInstructions)
    tools: list[AgentToolConfig] = Field(default_factory=list)
    sub_agents: list[str] = Field(default_factory=lambda: ["customer_support"])


class SideEffectClass(StrEnum):
    """Whether a tool call is read-only or side-effecting."""

    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"


class ToolCall(BaseModel):
    """A proposed or executed tool call.

    Under the V2 contract, idempotencyKey is generated for every tool call (reads and writes).
    tool_id is the primary identifier matching the catalog and backend contract.
    """

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    connector_id: str | None = None
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    params_hash: str | None = None
    side_effect_class: SideEffectClass = SideEffectClass.READ_ONLY
    idem_key: str | None = None
    outcome: dict[str, Any] | None = None


class ApprovalStatus(StrEnum):
    """Approval lifecycle states."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class Approval(BaseModel):
    """A pending or resolved approval.

    Holds the immutable params snapshot alongside params_hash so the approver
    card in the UI and the approval.required event can render
    actual call parameters.
    effect_summary is rendered from a deterministic template (never free LLM
    text) so the approver sees a stable, auditable description.
    """

    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    call_id: str
    tool_id: str
    action: str = ""
    integration: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    params_hash: str
    effect_summary: str = ""
    status: ApprovalStatus = ApprovalStatus.PENDING
    expires_at: datetime | None = None
    approver_id: str | None = None


class TraceEvent(BaseModel, frozen=True):
    """A single event appended to the execution trace ."""

    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: dict[str, Any] = Field(default_factory=dict)


# LangGraph state


class AgentState(BaseModel):
    """LangGraph state schema  flows through every graph node.

    Pydantic v2 BaseModel is natively supported by LangGraph 1.x StateGraph:
      StateGraph(AgentState)
    Nodes access fields via attribute notation (state.messages, state.plan).

    Annotated reducers:
      messages         add_messages (LangGraph): append + deduplicate by .id
      execution_trace  operator.add: plain append (immutable TraceEvent items)
    """

    # Identity  pinned at ingress by the submit-message use case 
    workspace_id: str
    agent_id: str
    sub_agent_id: str = DEFAULT_SUB_AGENT_ID
    audience: Literal["external", "internal"] = DEFAULT_AUDIENCE
    channel: str = ""
    conversation_id: str = ""
    agent_config: AgentConfig | None = None

    # Conversation  LangGraph's add_messages deduplicates by message .id
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Planning (Planner)
    plan: Plan | None = None

    # Retrieval (Retriever)
    retrieved_context: list[Chunk] = Field(default_factory=list)

    # Tool execution (Executor)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    pending_approvals: list[Approval] = Field(default_factory=list)

    # Execution trace  operator.add appends TraceEvent items
    execution_trace: Annotated[list[TraceEvent], operator.add] = Field(default_factory=list)

    # Confidence & handoff
    confidence: float = 0.0
    handoff: HandoffReason | None = None
