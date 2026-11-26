"""Contract and behaviour tests for the MCP Connector adapter (task 1.10).

``TestMcpConnectorContract`` reuses the shared Connector contract suite
(``tests/unit/ports/test_connector_contract.py``) against a real HTTP round
trip through the mock MCP app, so isolation, idempotency-key forwarding and
error mapping cannot regress silently.  The remaining tests cover the parts
specific to this adapter: catalog metadata projection, session reuse and the
transport failure paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio
from mocks.mock_mcp.server import MockMcpConfig, create_app
from tests.unit.ports.test_connector_contract import (
    TestConnectorContract as _ConnectorContract,
)

from felops.adapters.connector.mcp import McpConnector
from felops.domain.workspace import WorkspaceContext

_WS1 = WorkspaceContext(workspace_id="ws-1")
_WS2 = WorkspaceContext(workspace_id="ws-2")
_WORKSPACE_TOOLS = {"ws-1": ("lookup_order", "send_invoice"), "ws-2": ("get_case",)}
_FAILING_TOOL = "send_invoice"


def _app(failures: dict[str, dict[str, Any]] | None = None) -> Any:
    return create_app(MockMcpConfig(workspace_tools=_WORKSPACE_TOOLS, failures=failures))


async def _open_connector(app: Any) -> tuple[McpConnector, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://mock-mcp",
    )
    return McpConnector("http://mock-mcp", client=client), client


# ---------------------------------------------------------------------------
# Shared Connector contract
# ---------------------------------------------------------------------------


class TestMcpConnectorContract(_ConnectorContract):
    """Run the shared Connector contract against the MCP adapter."""

    @pytest_asyncio.fixture()
    async def connector(self) -> AsyncIterator[McpConnector]:
        adapter, client = await _open_connector(_app())
        try:
            yield adapter
        finally:
            await client.aclose()

    @pytest_asyncio.fixture()
    async def failing_connector(self) -> AsyncIterator[tuple[McpConnector, str]]:
        failures = {
            _FAILING_TOOL: {
                "code": "provider_error",
                "message": "QuickBooks did not respond after retries.",
                "retryable": True,
            }
        }
        adapter, client = await _open_connector(_app(failures))
        try:
            yield adapter, _FAILING_TOOL
        finally:
            await client.aclose()


# ---------------------------------------------------------------------------
# Adapter-specific behaviour
# ---------------------------------------------------------------------------


@pytest.fixture()
def mcp_app() -> Any:
    return _app()


@pytest_asyncio.fixture()
async def connector(mcp_app: Any) -> AsyncIterator[McpConnector]:
    adapter, client = await _open_connector(mcp_app)
    try:
        yield adapter
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_list_tools_rebuilds_full_tool_specs(connector: McpConnector) -> None:
    specs = {spec.tool_id: spec for spec in await connector.list_tools(_WS1)}

    lookup = specs["lookup_order"]
    assert lookup.connector_id == "shopify"
    assert lookup.result_summary == "Order {order_number} · {status}"
    assert lookup.effect_summary is None
    assert lookup.is_side_effecting is False
    assert lookup.version == 1
    assert lookup.deprecated is False
    assert lookup.parameters["required"] == ["order_number"]
    assert specs["send_invoice"].is_side_effecting is True


@pytest.mark.asyncio
async def test_call_forwards_idem_key_and_workspace(
    connector: McpConnector, mcp_app: Any
) -> None:
    await connector.call("lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-42")

    call = mcp_app.state.mock_mcp.calls[-1]
    assert call["idem_key"] == "idem-42"
    assert call["workspace_id"] == "ws-1"
    assert call["tool_id"] == "lookup_order"


@pytest.mark.asyncio
async def test_session_is_initialized_once(connector: McpConnector, mcp_app: Any) -> None:
    await connector.list_tools(_WS1)
    await connector.call("lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1")
    await connector.list_tools(_WS2)

    assert mcp_app.state.mock_mcp.initializations == 1


@pytest.mark.asyncio
async def test_call_returns_catalog_data(connector: McpConnector) -> None:
    result = await connector.call(
        "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
    )

    assert result.status == "succeeded"
    assert result.error is None
    assert result.data is not None
    assert result.data["tracking_number"] == "TRK8812"


@pytest.mark.asyncio
async def test_unreachable_server_maps_to_connector_not_connected() -> None:
    def raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(raise_connect_error),
        base_url="http://mock-mcp",
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)

        result = await adapter.call("lookup_order", {"order_number": "#1042"}, _WS1, "idem-1")
        tools = await adapter.list_tools(_WS1)

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "connector_not_connected"
    assert result.error.retryable is False
    assert tools == []


@pytest.mark.asyncio
async def test_timeout_maps_to_retryable_provider_error() -> None:
    def raise_timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(raise_timeout),
        base_url="http://mock-mcp",
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)

        result = await adapter.call("lookup_order", {"order_number": "#1042"}, _WS1, "idem-1")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "provider_error"
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_non_200_response_maps_to_structured_failure() -> None:
    def unauthorized(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid token"}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(unauthorized),
        base_url="http://mock-mcp",
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)

        result = await adapter.call("lookup_order", {"order_number": "#1042"}, _WS1, "idem-1")

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "connector_not_connected"
    assert result.error.message == "invalid token"


@pytest.mark.asyncio
async def test_aclose_does_not_close_injected_client() -> None:
    adapter, client = await _open_connector(_app())

    await adapter.aclose()

    assert client.is_closed is False
    await client.aclose()
