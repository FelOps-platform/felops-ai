"""Contract and behaviour tests for the MCP Connector adapter (task 1.10).

``TestMcpConnectorContract`` reuses the shared Connector contract suite
(``tests/unit/ports/test_connector_contract.py``) against a real HTTP round
trip through the mock MCP app, so isolation, idempotency-key forwarding and
error mapping cannot regress silently.  The remaining tests cover the parts
specific to this adapter: catalog metadata projection, per-workspace session
lifecycle, pagination, SSE decoding and the transport failure paths.
"""

from __future__ import annotations

import asyncio
import json
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
_PROTOCOL_VERSION = "2025-06-18"


def _app(failures: dict[str, dict[str, Any]] | None = None) -> Any:
    return create_app(MockMcpConfig(workspace_tools=_WORKSPACE_TOOLS, failures=failures))


async def _open_connector(app: Any) -> tuple[McpConnector, httpx.AsyncClient]:
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://mock-mcp",
    )
    return McpConnector("http://mock-mcp", client=client), client


def _request_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content) if request.content else {}


def _initialize_response(request: httpx.Request, session_id: str = "s1") -> httpx.Response:
    body = _request_body(request)
    return httpx.Response(
        200,
        json={
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "stub-mcp", "version": "0"},
            },
        },
        headers={"Mcp-Session-Id": session_id},
    )


def _result_response(request: httpx.Request, result: dict[str, Any]) -> httpx.Response:
    body = _request_body(request)
    return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": result})


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
async def test_session_is_created_once_per_workspace(
    connector: McpConnector, mcp_app: Any
) -> None:
    await connector.list_tools(_WS1)
    await connector.call("lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1")
    await connector.list_tools(_WS2)

    state = mcp_app.state.mock_mcp
    assert state.initializations == 2
    assert sorted(state.sessions.values()) == ["ws-1", "ws-2"]


@pytest.mark.asyncio
async def test_concurrent_calls_initialize_once(connector: McpConnector, mcp_app: Any) -> None:
    await asyncio.gather(*(connector.list_tools(_WS1) for _ in range(5)))

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
async def test_blank_idem_key_is_rejected(connector: McpConnector) -> None:
    with pytest.raises(ValueError):
        await connector.call("lookup_order", {"order_number": "#1042"}, _WS1, idem_key="  ")


@pytest.mark.asyncio
async def test_session_and_protocol_headers_are_forwarded() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        seen.append(request)
        return _result_response(request, {"tools": []})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        await adapter.list_tools(_WS1)

    assert seen[0].headers["X-Felops-Workspace"] == "ws-1"
    assert seen[0].headers["Mcp-Session-Id"] == "s1"
    assert seen[0].headers["MCP-Protocol-Version"] == _PROTOCOL_VERSION


@pytest.mark.asyncio
async def test_expired_session_is_reinitialized_and_retried() -> None:
    initializations = 0
    tool_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal initializations, tool_calls
        body = _request_body(request)
        method = body.get("method")
        if method == "initialize":
            initializations += 1
            return _initialize_response(request, session_id=f"s{initializations}")
        if "id" not in body:
            return httpx.Response(202)
        tool_calls += 1
        if request.headers.get("Mcp-Session-Id") == "s1":
            return httpx.Response(404, json={"error": {"message": "session expired"}})
        return _result_response(
            request,
            {"structuredContent": {"status": "succeeded", "data": {"ok": True}, "error": None}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "succeeded"
    assert initializations == 2
    assert tool_calls == 2


@pytest.mark.asyncio
async def test_sse_response_is_decoded_after_notifications() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        progress = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress"})
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": body["id"],
                "result": {
                    "structuredContent": {
                        "status": "succeeded",
                        "data": {"order_number": "#1042"},
                        "error": None,
                    }
                },
            }
        )
        return httpx.Response(
            200,
            text=f"event: message\ndata: {progress}\n\ndata: {response}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "succeeded"
    assert result.data == {"order_number": "#1042"}


@pytest.mark.asyncio
async def test_plain_text_tool_error_is_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(
            request,
            {"content": [{"type": "text", "text": "provider exploded"}], "isError": True},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.code == "provider_error"
    assert result.error.message == "provider exploded"


@pytest.mark.asyncio
async def test_structured_content_without_status_is_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(request, {"structuredContent": {"order_number": "#1042"}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "succeeded"
    assert result.data == {"order_number": "#1042"}


@pytest.mark.asyncio
async def test_json_text_content_fallback_is_decoded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        envelope = {"status": "succeeded", "data": {"tracking_number": "TRK8812"}, "error": None}
        return _result_response(
            request, {"content": [{"type": "text", "text": json.dumps(envelope)}]}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.data == {"tracking_number": "TRK8812"}


@pytest.mark.asyncio
async def test_json_rpc_error_on_http_200_is_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": body["id"],
                "error": {"code": -32603, "message": "internal boom"},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "internal boom"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(500, True), (429, True), (408, True), (400, False)],
)
async def test_http_error_retryability(status_code: int, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return httpx.Response(status_code, json={"error": {"message": "nope"}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.retryable is retryable


@pytest.mark.asyncio
async def test_list_tools_follows_pagination() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        if body["params"].get("cursor") == "page-2":
            return _result_response(
                request,
                {"tools": [{"name": "get_case", "description": "second", "inputSchema": {}}]},
            )
        return _result_response(
            request,
            {
                "tools": [{"name": "lookup_order", "description": "first", "inputSchema": {}}],
                "nextCursor": "page-2",
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        specs = await adapter.list_tools(_WS1)

    assert [spec.tool_id for spec in specs] == ["lookup_order", "get_case"]


@pytest.mark.asyncio
async def test_malformed_descriptor_is_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(
            request,
            {
                "tools": [
                    {"name": "lookup_order", "description": "ok", "inputSchema": {}},
                    {
                        "name": "broken",
                        "description": "bad version",
                        "inputSchema": {},
                        "_meta": {"felops": {"version": "not-a-number"}},
                    },
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        specs = await adapter.list_tools(_WS1)

    assert [spec.tool_id for spec in specs] == ["lookup_order"]


@pytest.mark.asyncio
async def test_blank_effect_summary_skips_descriptor() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(
            request,
            {
                "tools": [
                    {"name": "lookup_order", "description": "ok", "inputSchema": {}},
                    {
                        "name": "broken",
                        "description": "blank effect summary",
                        "inputSchema": {},
                        "_meta": {"felops": {"effect_summary": ""}},
                    },
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        specs = await adapter.list_tools(_WS1)

    assert [spec.tool_id for spec in specs] == ["lookup_order"]


@pytest.mark.asyncio
async def test_status_field_without_envelope_is_not_a_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(
            request,
            {"structuredContent": {"order_number": "#1042", "status": "shipped"}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "succeeded"
    assert result.data == {"order_number": "#1042", "status": "shipped"}


@pytest.mark.asyncio
async def test_is_error_overrides_a_succeeded_envelope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(
            request,
            {
                "structuredContent": {"status": "succeeded", "data": {"ok": True}, "error": None},
                "content": [{"type": "text", "text": "tool exploded"}],
                "isError": True,
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        adapter = McpConnector("http://mock-mcp", client=client)
        result = await adapter.call(
            "lookup_order", {"order_number": "#1042"}, _WS1, idem_key="idem-1"
        )

    assert result.status == "failed"
    assert result.error is not None
    assert result.error.message == "tool exploded"


@pytest.mark.asyncio
async def test_connector_id_falls_back_without_meta() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _request_body(request)
        if body.get("method") == "initialize":
            return _initialize_response(request)
        if "id" not in body:
            return httpx.Response(202)
        return _result_response(
            request, {"tools": [{"name": "lookup_order", "description": "", "inputSchema": {}}]}
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://mock-mcp"
    ) as client:
        explicit = McpConnector("http://mock-mcp", connector_id="shopify", client=client)
        defaulted = McpConnector("http://mock-mcp", client=client)
        explicit_specs = await explicit.list_tools(_WS1)
        default_specs = await defaulted.list_tools(_WS1)

    assert explicit_specs[0].connector_id == "shopify"
    assert default_specs[0].connector_id == "mcp"


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


@pytest.mark.asyncio
async def test_aclose_closes_owned_client() -> None:
    adapter = McpConnector("http://mock-mcp")
    client = adapter._client_or_create()

    await adapter.aclose()

    assert client.is_closed is True


def test_invalid_base_url_and_timeout_are_rejected() -> None:
    with pytest.raises(ValueError):
        McpConnector("   ")
    with pytest.raises(ValueError):
        McpConnector("http://mock-mcp", timeout=0)
