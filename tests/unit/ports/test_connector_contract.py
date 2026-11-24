"""Contract test stub for Connector (§5.1, task 0.6).

A *contract test* proves that a concrete adapter satisfies the Connector
protocol.  This module provides:

1. ``FakeConnector`` — an in-memory stub that is itself a valid adapter, keyed
   by ``WorkspaceContext.workspace_id``.
2. ``TestConnectorContract`` — protocol-typed tests that any real adapter
   module can subclass by overriding the fixtures.

The contract has three parts beyond the signatures:

* **Idempotency** (§4.3): **every** call (reads and writes) carries a required
  ``idem_key``.  The connector forwards the key verbatim and never invents one.
* **Isolation** (§2.3): tools and data exposed under one workspace are never
  visible or callable from another workspace.
* **Errors** (§4.2): execution failures and disconnected/unauthorized
  connectors (``connector_not_connected``) are returned as
  ``ToolResult(status="failed", error=ToolError(...))`` rather than raised.

When the MCP adapter lands (task 1.10), it subclasses ``TestConnectorContract``
and overrides the fixtures so a regression in isolation, key forwarding, or
error mapping fails CI.
"""

from __future__ import annotations

from typing import Any

import pytest

from felops.domain.tools import ToolError, ToolResult, ToolSpec
from felops.domain.workspace import WorkspaceContext
from felops.ports.connector import Connector

_READ = ToolSpec(
    tool_id="lookup_order",
    connector_id="shopify",
    description="Read an order with its fulfillment, payment and refund status.",
    parameters={
        "type": "object",
        "properties": {"order_number": {"type": "string"}},
        "required": ["order_number"],
        "additionalProperties": False,
    },
    result_summary="Order {order_number} · {status}",
    effect_summary=None,
    version=1,
    deprecated=False,
)

_WRITE = ToolSpec(
    tool_id="send_invoice",
    connector_id="quickbooks",
    description="Email an existing tax invoice to the customer.",
    parameters={
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "amount": {"type": "number"},
            "recipient_email": {"type": "string"},
        },
        "required": ["invoice_number", "amount", "recipient_email"],
        "additionalProperties": False,
    },
    result_summary="Invoice {invoice_number} sent to {recipient_email}",
    effect_summary="Send invoice {invoice_number} for AED {amount} to {recipient_email}",
    version=1,
    deprecated=False,
)

_WS2_TOOL = ToolSpec(
    tool_id="get_case",
    connector_id="salesforce",
    description="Read a support case with its status and subject.",
    parameters={
        "type": "object",
        "properties": {"case_id": {"type": "string"}},
        "required": ["case_id"],
        "additionalProperties": False,
    },
    result_summary="Case {case_id} · {status} · {priority}",
    effect_summary=None,
    version=1,
    deprecated=False,
)

_WS1 = WorkspaceContext(workspace_id="ws-1")
_WS2 = WorkspaceContext(workspace_id="ws-2")


class FakeConnector:
    """In-memory Connector keyed by ``workspace_id``.

    Each workspace receives only its own registered ``ToolSpec`` list.  Calls
    record the exact forwarded ``tool``, ``args``, ``ctx``, and ``idem_key``.
    Configured ``failing_tools`` return structured ``ToolResult(status="failed")``
    envelopes without relying on magic sentinel tool names.
    """

    def __init__(
        self,
        tools_by_workspace: dict[str, list[ToolSpec]],
        failing_tools: dict[str, ToolError] | None = None,
    ) -> None:
        self._tools_by_workspace = {
            ws_id: list(specs) for ws_id, specs in tools_by_workspace.items()
        }
        self._failing_tools = dict(failing_tools or {})
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self, ctx: WorkspaceContext) -> list[ToolSpec]:
        return list(self._tools_by_workspace.get(ctx.workspace_id, ()))

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str,
    ) -> ToolResult:
        self.calls.append({"tool": tool, "args": args, "ctx": ctx, "idem_key": idem_key})

        if tool in self._failing_tools:
            return ToolResult(status="failed", data=None, error=self._failing_tools[tool])

        workspace_tool_ids = {
            spec.tool_id for spec in self._tools_by_workspace.get(ctx.workspace_id, ())
        }
        if tool not in workspace_tool_ids:
            return ToolResult(
                status="failed",
                data=None,
                error=ToolError(
                    code="connector_not_connected",
                    message=f"tool '{tool}' is not available in workspace '{ctx.workspace_id}'",
                    retryable=False,
                ),
            )

        return ToolResult(
            status="succeeded",
            data={"workspace_id": ctx.workspace_id, "tool_id": tool, "args": args},
            error=None,
        )


# ---------------------------------------------------------------------------
# Protocol structural conformance
# ---------------------------------------------------------------------------


class TestConnectorProtocol:
    def test_is_structurally_compatible(self) -> None:
        connector: Connector = FakeConnector({_WS1.workspace_id: [_READ]})
        assert connector is not None


# ---------------------------------------------------------------------------
# Contract tests — every Connector adapter must pass these
# ---------------------------------------------------------------------------


class TestConnectorContract:
    @pytest.fixture()
    def connector(self) -> Connector:
        return FakeConnector(
            {
                _WS1.workspace_id: [_READ, _WRITE],
                _WS2.workspace_id: [_WS2_TOOL],
            }
        )

    @pytest.fixture()
    def failing_connector(self) -> tuple[Connector, str]:
        """Return ``(connector, tool_id)`` whose call yields a structured failure.

        Real adapter subclasses override this fixture to point at a failing or
        disconnected tool on their test double.
        """
        err = ToolError(
            code="connector_not_connected",
            message="QuickBooks connector is not connected.",
            retryable=False,
        )
        return (
            FakeConnector(
                {_WS1.workspace_id: [_WRITE]},
                failing_tools={_WRITE.tool_id: err},
            ),
            _WRITE.tool_id,
        )

    @pytest.mark.asyncio
    async def test_list_tools_returns_the_exposed_specs(self, connector: Connector) -> None:
        tools = await connector.list_tools(_WS1)

        assert {tool.tool_id for tool in tools} == {"lookup_order", "send_invoice"}

    @pytest.mark.asyncio
    async def test_read_only_call_accepts_required_idem_key(self, connector: Connector) -> None:
        args = {"order_number": "#1042"}

        result = await connector.call("lookup_order", args, _WS1, idem_key="idem-read-1")

        assert result.status == "succeeded"
        assert result.error is None
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_side_effecting_call_accepts_required_idem_key(
        self, connector: Connector
    ) -> None:
        args = {
            "invoice_number": "INV-2210",
            "amount": 349.0,
            "recipient_email": "sara.ali@example.com",
        }

        result = await connector.call("send_invoice", args, _WS1, idem_key="idem-write-1")

        assert result.status == "succeeded"
        assert result.error is None
        assert result.data is not None

    @pytest.mark.asyncio
    async def test_workspaces_are_isolated(self, connector: Connector) -> None:
        """Tools and executions from one workspace are never visible from another (§2.3)."""
        ws1_ids = {tool.tool_id for tool in await connector.list_tools(_WS1)}
        ws2_ids = {tool.tool_id for tool in await connector.list_tools(_WS2)}

        assert ws1_ids == {"lookup_order", "send_invoice"}
        assert ws2_ids == {"get_case"}

        cross_ws = await connector.call(
            "lookup_order",
            {"order_number": "#1042"},
            _WS2,
            idem_key="idem-cross-ws",
        )
        assert cross_ws.status == "failed"
        assert cross_ws.error is not None
        assert cross_ws.error.code == "connector_not_connected"

    @pytest.mark.asyncio
    async def test_tool_error_is_returned_not_raised(
        self, failing_connector: tuple[Connector, str]
    ) -> None:
        connector, tool_id = failing_connector

        result = await connector.call(tool_id, {}, _WS1, idem_key="idem-fail-1")

        assert result.status == "failed"
        assert result.data is None
        assert result.error is not None
        assert result.error.code
        assert result.error.message
        assert isinstance(result.error.retryable, bool)


# ---------------------------------------------------------------------------
# FakeConnector forwarding checks (specific to the in-memory test double)
# ---------------------------------------------------------------------------


class TestFakeConnectorForwarding:
    @pytest.mark.asyncio
    async def test_forwards_arguments_context_and_idem_key_verbatim(self) -> None:
        fake = FakeConnector({_WS1.workspace_id: [_READ, _WRITE]})
        read_args = {"order_number": "#1042"}

        read_result = await fake.call("lookup_order", read_args, _WS1, idem_key="idem-read-42")
        await fake.call("send_invoice", {"invoice_number": "INV-2210"}, _WS1, idem_key="idem-99")

        assert read_result.data is not None
        assert read_result.data["args"] == read_args
        assert fake.calls[0] == {
            "tool": "lookup_order",
            "args": read_args,
            "ctx": _WS1,
            "idem_key": "idem-read-42",
        }
        assert fake.calls[1]["idem_key"] == "idem-99"
        assert fake.calls[1]["ctx"] == _WS1
