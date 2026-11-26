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
    SESSION_HEADER,
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
) -> Any:
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
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
) -> Any:
    return _rpc(
        client,
        "tools/call",
        {
            "name": tool_id,
            "arguments": arguments,
            "_meta": {IDEM_KEY_META: "idem-1"},
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
    _rpc(
        client,
        "tools/call",
        {
            "name": "lookup_order",
            "arguments": {"order_number": "#1042"},
            "_meta": {IDEM_KEY_META: "idem-forwarded"},
        },
    )

    call = client.app.state.mock_mcp.calls[-1]
    assert call["idem_key"] == "idem-forwarded"
    assert call["workspace_id"] == _WORKSPACE
    assert call["arguments"] == {"order_number": "#1042"}


def test_tools_call_echoes_unknown_arguments(client: TestClient) -> None:
    response = _call(client, "lookup_order", {"order_number": "#9999"})

    data = response.json()["result"]["structuredContent"]["data"]
    assert data == {"tool_id": "lookup_order", "order_number": "#9999"}


def test_echoed_tool_id_cannot_be_overridden(client: TestClient) -> None:
    response = _call(client, "lookup_order", {"order_number": "#9999", "tool_id": "evil"})

    data = response.json()["result"]["structuredContent"]["data"]
    assert data["tool_id"] == "lookup_order"


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


# ---------------------------------------------------------------------------
# Protocol validation and sessions
# ---------------------------------------------------------------------------


def _initialize(client: TestClient, workspace: str = _WORKSPACE) -> str:
    response = _rpc(client, "initialize", workspace=workspace)
    assert response.status_code == 200
    return response.headers[SESSION_HEADER]


def test_ping_returns_an_empty_result(client: TestClient) -> None:
    response = _rpc(client, "ping")

    assert response.json()["result"] == {}


def test_invalid_jsonrpc_version_is_rejected(client: TestClient) -> None:
    body = {"jsonrpc": "1.0", "id": 1, "method": "ping"}

    response = client.post("/mcp", json=body, headers={WORKSPACE_HEADER: _WORKSPACE})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600


def test_null_request_id_is_rejected(client: TestClient) -> None:
    body = {"jsonrpc": "2.0", "id": None, "method": "ping"}

    response = client.post("/mcp", json=body, headers={WORKSPACE_HEADER: _WORKSPACE})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32600


def test_invalid_params_type_is_rejected(client: TestClient) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": []}

    response = client.post("/mcp", json=body, headers={WORKSPACE_HEADER: _WORKSPACE})

    assert response.json()["error"]["code"] == -32602


def test_invalid_arguments_type_is_rejected(client: TestClient) -> None:
    response = _rpc(
        client,
        "tools/call",
        {"name": "lookup_order", "arguments": []},
    )

    assert response.json()["error"]["code"] == -32602


def test_unknown_session_is_rejected(client: TestClient) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

    response = client.post(
        "/mcp",
        json=body,
        headers={WORKSPACE_HEADER: _WORKSPACE, SESSION_HEADER: "unknown-session"},
    )

    assert response.status_code == 404


def test_session_is_bound_to_its_workspace(client: TestClient) -> None:
    session_id = _initialize(client, _WORKSPACE)
    body = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

    response = client.post(
        "/mcp",
        json=body,
        headers={WORKSPACE_HEADER: _OTHER_WORKSPACE, SESSION_HEADER: session_id},
    )

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Default configuration and catalog examples
# ---------------------------------------------------------------------------


def test_default_config_exposes_every_tool_to_every_workspace() -> None:
    with TestClient(create_app()) as default_client:
        tools = _tools_by_name(default_client, "any-workspace")
        result = _call(
            default_client,
            "lookup_order",
            {"order_number": "#1042"},
            workspace="any-workspace",
        )

    assert set(tools) == set(_ALL_TOOLS)
    assert result.json()["result"]["isError"] is False


def test_every_catalog_example_is_served() -> None:
    cases = [
        ("lookup_order", {"order_number": "#1042"}, {"tracking_number": "TRK8812"}),
        (
            "refund_order",
            {"order_number": "#1042", "amount": 120.0, "reason": "damaged_on_arrival"},
            {"refund_id": "RFD-3081"},
        ),
        (
            "send_invoice",
            {
                "invoice_number": "INV-2210",
                "amount": 349.0,
                "recipient_email": "sara.ali@example.com",
                "order_number": "#1042",
            },
            {"status": "sent"},
        ),
        ("get_invoice", {"invoice_number": "INV-2210"}, {"amount": 349.0}),
        (
            "create_case",
            {
                "subject": "Damaged lamp on order #1042",
                "description": (
                    "The ceramic base arrived cracked. The customer asks for a replacement "
                    "or a refund."
                ),
                "contact_email": "sara.ali@example.com",
                "priority": "high",
                "order_number": "#1042",
            },
            {"case_id": "CASE-5512"},
        ),
        ("get_case", {"case_id": "CASE-5512"}, {"status": "open"}),
    ]

    with TestClient(create_app()) as default_client:
        for tool_id, arguments, expected in cases:
            result = _call(default_client, tool_id, arguments).json()["result"]
            assert result["isError"] is False, tool_id
            data = result["structuredContent"]["data"]
            for key, value in expected.items():
                assert data[key] == value, (tool_id, key)

