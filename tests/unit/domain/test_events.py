"""Unit tests for felops.domain.events  EventType, EventEnvelope."""

import pytest
from pydantic import ValidationError

from felops.domain.events import EventEnvelope, EventType

# EventType


class TestEventType:
    def test_all_13_members_exist(self):
        expected = {
            "plan.created",
            "retrieval.completed",
            "tool.started",
            "tool.completed",
            "approval.required",
            "approval.resolved",
            "verifier.passed",
            "verifier.failed",
            "handoff.requested",
            "message.delta",
            "message.final",
            "usage.recorded",
            "error",
        }
        actual = {e.value for e in EventType}
        assert actual == expected

    def test_member_count(self):
        assert len(EventType) == 13

    def test_wire_values_match_spec(self):
        """Each member's value must match exactly."""
        assert EventType.PLAN_CREATED == "plan.created"
        assert EventType.RETRIEVAL_COMPLETED == "retrieval.completed"
        assert EventType.TOOL_STARTED == "tool.started"
        assert EventType.TOOL_COMPLETED == "tool.completed"
        assert EventType.APPROVAL_REQUIRED == "approval.required"
        assert EventType.APPROVAL_RESOLVED == "approval.resolved"
        assert EventType.VERIFIER_PASSED == "verifier.passed"
        assert EventType.VERIFIER_FAILED == "verifier.failed"
        assert EventType.HANDOFF_REQUESTED == "handoff.requested"
        assert EventType.MESSAGE_DELTA == "message.delta"
        assert EventType.MESSAGE_FINAL == "message.final"
        assert EventType.USAGE_RECORDED == "usage.recorded"
        assert EventType.ERROR == "error"

    def test_is_str(self):
        """StrEnum members are plain strings  safe for JSON serialisation."""
        for member in EventType:
            assert isinstance(member, str), f"{member} is not a str"


# EventEnvelope


class TestEventEnvelope:
    def _make(self, **kwargs) -> EventEnvelope:
        defaults = dict(
            seq=0,
            type=EventType.MESSAGE_FINAL,
            workspace_id="ws-1",
            agent_id="ag-1",
        )
        defaults.update(kwargs)
        return EventEnvelope(**defaults)

    def test_creation_with_required_fields(self):
        env = self._make()
        assert env.workspace_id == "ws-1"
        assert env.agent_id == "ag-1"
        assert env.type == EventType.MESSAGE_FINAL
        assert env.seq == 0

    def test_event_id_auto_generated(self):
        a = self._make()
        b = self._make()
        assert a.event_id != b.event_id

    def test_ts_auto_generated(self):
        env = self._make()
        assert env.ts is not None

    def test_v1_defaults(self):
        env = self._make()
        assert env.sub_agent_id == "customer_support"
        assert env.conversation_id == ""
        assert env.job_id == ""
        assert env.data == {}

    def test_seq_zero_allowed(self):
        env = self._make(seq=0)
        assert env.seq == 0

    def test_seq_positive_allowed(self):
        env = self._make(seq=42)
        assert env.seq == 42

    def test_seq_negative_rejected(self):
        with pytest.raises(ValidationError):
            self._make(seq=-1)

    def test_frozen_rejects_mutation(self):
        env = self._make()
        with pytest.raises(ValidationError):
            env.seq = 1  # type: ignore[misc]

    def test_data_payload(self):
        env = self._make(
            type=EventType.TOOL_COMPLETED,
            data={"call_id": "c-1", "status": "ok"},
        )
        assert env.data["call_id"] == "c-1"

    def test_all_event_types_accepted(self):
        """Every EventType value must be a valid envelope type."""
        for et in EventType:
            env = self._make(type=et)
            assert env.type == et

    def test_serialization_round_trip(self):
        env = self._make(seq=5, data={"text": "hello"})
        d = env.model_dump()
        assert d["seq"] == 5
        assert d["workspace_id"] == "ws-1"
        assert d["agent_id"] == "ag-1"
        assert d["data"] == {"text": "hello"}
        assert d["sub_agent_id"] == "customer_support"

    def test_model_validate_round_trip(self):
        env = self._make(seq=1)
        restored = EventEnvelope.model_validate(env.model_dump())
        assert restored.event_id == env.event_id
        assert restored.seq == env.seq
        assert restored.type == env.type
