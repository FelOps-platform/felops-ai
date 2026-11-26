"""Tests for the mock MCP server (task 1.10, §4 MCP Client).

Every test builds a fresh app so call logs and sessions never leak between
tests.  The checks cover the MCP handshake, catalog projection, workspace
scoping, deterministic execution from ``fixtures/catalog/catalog.json`` and
the failure paths the Connector adapter must map.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mocks.mock_mcp.server import (
    IDEM_KEY_META,
    WORKSPACE_HEADER,
    MockMcpConfig,
    create_app,
)

_WORKSPACE = "ws-1"
_OTHER_WORKSPACE = "ws-2"
_ALL_TOOLS = (
    "lookup_order",
    "refund_order",
    "send_invoice",
    "get_invoice",
    "create_case",
    "get_case",
)
_WORKSPACE_TOOLS = {
    _WORKSPACE: _ALL_TOOLS,
    _OTHER_WORKSPACE: ("get_case",),
}
_FAILURES = {"refund_order": {"code": "provider_error", "message": "Shopify is down"}}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    config = MockMcpConfig(workspace_tools=_WORKSPACE_TOOLS, failures=_FAILURES)
    with TestClient(create_app(config)) as test_client:
        yield test_client


def _rpc(
    client: TestClient,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    workspace: str = _WORKSPACE,
    request_id: int = 1,
) -> Any:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", json=body, headers={WORKSPACE_HEADER: workspace})


def _tools_by_name(client: TestClient, workspace: str = _WORKSPACE) -> dict[str, Any]:
    response = _rpc(client, "tools/list", workspace=workspace)
    assert response.status_code == 200
    return {tool["name"]: tool for tool in response.json()["result"]["tools"]}


def _call(
    client: TestClient,
    tool_id: str,
    arguments: dict[str, Any],
    *,
    workspace: str = _WORKSPACE,
    idem_key: str = "idem-1",
) -> Any:
    return _rpc(
        client,
        "tools/call",
        {
            "name": tool_id,
            "arguments": arguments,
            "_meta": {IDEM_KEY_META: idem_key},
        },
        workspace=workspace,
    )


# ---------------------------------------------------------------------------
# Transport and handshake
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").json() == {"status": "ok"}


def test_initialize_returns_protocol_and_session(client: TestClient) -> None:
    response = _rpc(client, "initialize")

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert result["serverInfo"]["name"] == "felops-mock-mcp"
    assert result["capabilities"]["tools"]["listChanged"] is False
    assert response.headers["Mcp-Session-Id"]


def test_notification_is_accepted_without_body(client: TestClient) -> None:
    body = {"jsonrpc": "2.0", "method": "notifications/initialized"}

    response = client.post("/mcp", json=body, headers={WORKSPACE_HEADER: _WORKSPACE})

    assert response.status_code == 202
    assert response.content == b""


def test_missing_workspace_header_is_rejected(client: TestClient) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

    response = client.post("/mcp", json=body)

    assert response.status_code == 400
    assert WORKSPACE_HEADER in response.json()["error"]["message"]


def test_unknown_method_returns_json_rpc_error(client: TestClient) -> None:
    response = _rpc(client, "resources/list")

    assert response.status_code == 200
    assert response.json()["error"]["code"] == -32601


def test_invalid_json_returns_parse_error(client: TestClient) -> None:
    response = client.post(
        "/mcp",
        content=b"{not-json",
        headers={WORKSPACE_HEADER: _WORKSPACE, "Content-Type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32700


# ---------------------------------------------------------------------------
# tools/list
# ---------------------------------------------------------------------------


def test_tools_list_exposes_the_six_catalog_actions(client: TestClient) -> None:
    tools = _tools_by_name(client)

    assert set(tools) == set(_ALL_TOOLS)


def test_tools_list_projects_catalog_metadata(client: TestClient) -> None:
    tool = _tools_by_name(client)["lookup_order"]
    meta = tool["_meta"]["felops"]

    assert tool["description"].startswith("Read an order")
    assert tool["inputSchema"]["required"] == ["order_number"]
    assert meta["connector_id"] == "shopify"
    assert meta["version"] == 1
    assert meta["deprecated"] is False
    assert meta["effect_summary"] is None


def test_tools_list_is_scoped_by_workspace(client: TestClient) -> None:
    assert set(_tools_by_name(client, _OTHER_WORKSPACE)) == {"get_case"}


# ---------------------------------------------------------------------------
# tools/call
# ---------------------------------------------------------------------------


def test_tools_call_returns_the_catalog_example(client: TestClient) -> None:
    response = _call(client, "lookup_order", {"order_number": "#1042"})

    assert response.status_code == 200
    result = response.json()["result"]
    assert result["isError"] is False
    assert result["structuredContent"]["status"] == "succeeded"
    assert result["structuredContent"]["data"]["tracking_number"] == "TRK8812"
    assert json.loads(result["content"][0]["text"]) == result["structuredContent"]


def test_tools_call_records_the_forwarded_idem_key(client: TestClient) -> None:
    _call(client, "lookup_order", {"order_number": "#1042"}, idem_key="idem-forwarded")

    call = client.app.state.mock_mcp.calls[-1]
    assert call["idem_key"] == "idem-forwarded"
    assert call["workspace_id"] == _WORKSPACE
    assert call["arguments"] == {"order_number": "#1042"}


def test_tools_call_echoes_unknown_arguments(client: TestClient) -> None:
    response = _call(client, "lookup_order", {"order_number": "#9999"})

    data = response.json()["result"]["structuredContent"]["data"]
    assert data == {"tool_id": "lookup_order", "order_number": "#9999"}


def test_tool_not_available_in_workspace_is_structured_failure(client: TestClient) -> None:
    response = _call(client, "lookup_order", {"order_number": "#1042"}, workspace=_OTHER_WORKSPACE)

    result = response.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["status"] == "failed"
    assert result["structuredContent"]["error"]["code"] == "connector_not_connected"


def test_unknown_tool_is_structured_failure(client: TestClient) -> None:
    response = _call(client, "delete_everything", {})

    result = response.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "connector_not_connected"


def test_configured_failure_is_returned(client: TestClient) -> None:
    response = _call(
        client,
        "refund_order",
        {"order_number": "#1042", "amount": 120.0, "reason": "damaged_on_arrival"},
    )

    error = response.json()["result"]["structuredContent"]["error"]
    assert error == {"code": "provider_error", "message": "Shopify is down", "retryable": False}
