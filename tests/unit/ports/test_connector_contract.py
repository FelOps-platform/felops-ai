"""Contract test stub for Connector (§5.1, task 0.6).

A *contract test* proves that a concrete adapter satisfies the Connector
protocol.  This module provides:

1. ``FakeConnector`` — an in-memory stub that is itself a valid adapter.
2. Tests that exercise the documented contract against the stub.

The contract has two parts beyond the signatures:

* **Idempotency** (§4.1, §2.2): the caller passes ``idem_key`` iff the call
  is side-effecting.  A read-only call passes ``None`` and always runs live.
  The connector forwards the key verbatim; the executor's ledger, not the
  connector, suppresses duplicates.
* **Errors**: a tool-level failure is returned as ``ToolResult(ok=False)``.
  Transport failures (auth, timeout) may raise.

When the MCP adapter lands (task 1.10), it runs the same suite via a fixture,
so a regression in key forwarding or error mapping fails CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from felops.domain.state import SideEffectClass
from felops.domain.tools import ToolResult, ToolSpec
from felops.domain.workspace import WorkspaceContext
from felops.ports.connector import Connector

_READ = ToolSpec(
    action_id="act-1",
    name="shopify_get_order",
    description="Look up an order by id",
    input_schema={"type": "object", "properties": {"order_id": {"type": "string"}}},
    side_effect_class=SideEffectClass.READ_ONLY,
)

_WRITE = ToolSpec(
    action_id="act-2",
    name="quickbooks_send_invoice",
    description="Send an invoice",
    input_schema={"type": "object"},
    side_effect_class=SideEffectClass.SIDE_EFFECTING,
)

_CTX = WorkspaceContext(workspace_id="ws-1")


class FakeConnector:
    """In-memory Connector that records every call it receives.

    Tools are declared at construction; ``call`` echoes the arguments back so
    the test can assert what was forwarded.  The special tool name ``broken``
    returns a tool-level failure instead of raising.
    """

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = tools
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self, ctx: WorkspaceContext) -> list[ToolSpec]:
        return list(self._tools)

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str | None = None,
    ) -> ToolResult:
        self.calls.append({"tool": tool, "args": args, "ctx": ctx, "idem_key": idem_key})
        if tool == "broken":
            return ToolResult(ok=False, error="upstream rejected the call")
        return ToolResult(ok=True, data={"tool": tool, "args": args})


# ---------------------------------------------------------------------------
# Protocol structural conformance
# ---------------------------------------------------------------------------


class TestConnectorProtocol:
    def test_is_structurally_compatible(self) -> None:
        connector: Connector = FakeConnector([_READ])
        assert connector is not None


# ---------------------------------------------------------------------------
# Contract tests — every Connector adapter must pass these
# ---------------------------------------------------------------------------


class TestConnectorContract:
    @pytest.fixture()
    def connector(self) -> FakeConnector:
        return FakeConnector([_READ, _WRITE])

    @pytest.mark.asyncio
    async def test_list_tools_returns_the_exposed_specs(self, connector: FakeConnector) -> None:
        tools = await connector.list_tools(_CTX)

        assert [tool.name for tool in tools] == ["shopify_get_order", "quickbooks_send_invoice"]

    @pytest.mark.asyncio
    async def test_call_forwards_arguments_unchanged(self, connector: FakeConnector) -> None:
        args = {"order_id": "1042", "include_lines": True}

        result = await connector.call("shopify_get_order", args, _CTX)

        assert result.ok is True
        assert result.data["args"] == args

    @pytest.mark.asyncio
    async def test_call_forwards_the_workspace_context(self, connector: FakeConnector) -> None:
        await connector.call("shopify_get_order", {}, _CTX)

        assert connector.calls[0]["ctx"] == _CTX

    @pytest.mark.asyncio
    async def test_read_only_call_carries_no_idempotency_key(
        self, connector: FakeConnector
    ) -> None:
        await connector.call("shopify_get_order", {"order_id": "1042"}, _CTX)

        assert connector.calls[0]["idem_key"] is None

    @pytest.mark.asyncio
    async def test_side_effecting_call_forwards_the_idempotency_key(
        self, connector: FakeConnector
    ) -> None:
        await connector.call("quickbooks_send_invoice", {"invoice": 7}, _CTX, idem_key="idem-42")

        assert connector.calls[0]["idem_key"] == "idem-42"

    @pytest.mark.asyncio
    async def test_tool_error_is_returned_not_raised(self, connector: FakeConnector) -> None:
        result = await connector.call("broken", {}, _CTX)

        assert result.ok is False
        assert result.error
