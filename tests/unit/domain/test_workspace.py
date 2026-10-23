"""Unit tests for felops.domain.workspace  WorkspaceContext, NameSpace."""

import pytest
from pydantic import ValidationError

from felops.domain.workspace import NameSpace, WorkspaceContext

# WorkspaceContext


class TestWorkspaceContext:
    def test_creation(self):
        ctx = WorkspaceContext(workspace_id="ws-1")
        assert ctx.workspace_id == "ws-1"

    def test_empty_workspace_id_rejected(self):
        with pytest.raises(ValidationError):
            WorkspaceContext(workspace_id="")

    def test_frozen_rejects_mutation(self):
        ctx = WorkspaceContext(workspace_id="ws-1")
        with pytest.raises(ValidationError):
            ctx.workspace_id = "ws-2"  # type: ignore[misc]

    def test_hashable(self):
        ctx = WorkspaceContext(workspace_id="ws-1")
        d = {ctx: "value"}
        assert d[ctx] == "value"

    def test_equality(self):
        a = WorkspaceContext(workspace_id="ws-1")
        b = WorkspaceContext(workspace_id="ws-1")
        assert a == b

    def test_inequality(self):
        a = WorkspaceContext(workspace_id="ws-1")
        b = WorkspaceContext(workspace_id="ws-2")
        assert a != b

    def test_usable_in_set(self):
        a = WorkspaceContext(workspace_id="ws-1")
        b = WorkspaceContext(workspace_id="ws-1")
        assert len({a, b}) == 1


# NameSpace


class TestNamespace:
    def test_creation(self):
        ns = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        assert ns.workspace_id == "ws-1"
        assert ns.agent_id == "ag-1"
        assert ns.source_id == "src-1"

    def test_empty_workspace_id_rejected(self):
        with pytest.raises(ValidationError):
            NameSpace(workspace_id="", agent_id="ag-1", source_id="src-1")

    def test_empty_agent_id_rejected(self):
        with pytest.raises(ValidationError):
            NameSpace(workspace_id="ws-1", agent_id="", source_id="src-1")

    def test_empty_source_id_rejected(self):
        with pytest.raises(ValidationError):
            NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="")

    def test_frozen_rejects_mutation(self):
        ns = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        with pytest.raises(ValidationError):
            ns.workspace_id = "ws-2"  # type: ignore[misc]

    def test_hashable(self):
        ns = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        d = {ns: "ok"}
        assert d[ns] == "ok"

    def test_equality(self):
        a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        b = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        assert a == b

    def test_inequality_on_source(self):
        a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        b = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-2")
        assert a != b

    def test_usable_in_set(self):
        a = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        b = NameSpace(workspace_id="ws-1", agent_id="ag-1", source_id="src-1")
        assert len({a, b}) == 1
