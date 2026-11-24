"""Unit tests for felops.domain.agent  AgentRef, HandoffReason."""

import pytest
from pydantic import ValidationError

from felops.domain.agent import (
    DEFAULT_AUDIENCE,
    DEFAULT_SUB_AGENT_ID,
    AgentRef,
    HandoffReason,
)

# Constants


class TestConstants:
    def test_default_sub_agent_id(self):
        assert DEFAULT_SUB_AGENT_ID == "customer_support"

    def test_default_audience(self):
        assert DEFAULT_AUDIENCE == "external"


# AgentRef


class TestAgentRef:
    def test_minimal_creation_uses_v1_defaults(self):
        ref = AgentRef(agent_id="ag-1")
        assert ref.agent_id == "ag-1"
        assert ref.sub_agent_id == "customer_support"
        assert ref.audience == "external"

    def test_explicit_values_accepted(self):
        ref = AgentRef(agent_id="ag-1", sub_agent_id="billing", audience="internal")
        assert ref.sub_agent_id == "billing"
        assert ref.audience == "internal"

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ValidationError):
            AgentRef(agent_id="")

    def test_empty_sub_agent_id_rejected(self):
        with pytest.raises(ValidationError):
            AgentRef(agent_id="ag-1", sub_agent_id="")

    def test_invalid_audience_rejected(self):
        with pytest.raises(ValidationError):
            AgentRef(agent_id="ag-1", audience="unknown")  # type: ignore[arg-type]

    def test_frozen_rejects_mutation(self):
        ref = AgentRef(agent_id="ag-1")
        with pytest.raises(ValidationError):
            ref.agent_id = "ag-2"  # type: ignore[misc]

    def test_hashable(self):
        ref = AgentRef(agent_id="ag-1")
        d = {ref: "ok"}
        assert d[ref] == "ok"

    def test_equality(self):
        a = AgentRef(agent_id="ag-1")
        b = AgentRef(agent_id="ag-1")
        assert a == b

    def test_inequality_on_sub_agent(self):
        a = AgentRef(agent_id="ag-1", sub_agent_id="customer_support")
        b = AgentRef(agent_id="ag-1", sub_agent_id="billing")
        assert a != b

    def test_usable_as_dict_key(self):
        ref1 = AgentRef(agent_id="ag-1")
        ref2 = AgentRef(agent_id="ag-1")
        cache: dict[AgentRef, str] = {}
        cache[ref1] = "effective-tools"
        assert cache[ref2] == "effective-tools"  # same hash, same value


# HandoffReason


class TestHandoffReason:
    def test_all_members_exist(self):
        assert HandoffReason.LOW_CONFIDENCE
        assert HandoffReason.TOOL_FAILED
        assert HandoffReason.CONNECTOR_NOT_CONNECTED
        assert HandoffReason.ASSISTANCE_REQUESTED

    def test_wire_values(self):
        assert HandoffReason.LOW_CONFIDENCE == "low_confidence"
        assert HandoffReason.TOOL_FAILED == "tool_failed"
        assert HandoffReason.CONNECTOR_NOT_CONNECTED == "connector_not_connected"
        assert HandoffReason.ASSISTANCE_REQUESTED == "assistance_requested"

    def test_is_str(self):
        # StrEnum members are plain strings  safe for JSON serialisation
        assert isinstance(HandoffReason.LOW_CONFIDENCE, str)
        assert isinstance(HandoffReason.TOOL_FAILED, str)
        assert isinstance(HandoffReason.CONNECTOR_NOT_CONNECTED, str)
        assert isinstance(HandoffReason.ASSISTANCE_REQUESTED, str)

    def test_member_count(self):
        assert len(HandoffReason) == 4
