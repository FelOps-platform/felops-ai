"""Mock MCP server for the P1 tool slice (task 1.10, §4 MCP Client).

The server speaks the JSON-RPC subset of the Model Context Protocol (MCP)
over HTTP:

* ``initialize`` / any notification — session handshake.
* ``tools/list`` — the catalog tools exposed to a workspace.
* ``tools/call`` — deterministic execution straight from
  ``fixtures/catalog/catalog.json`` (six actions).
* ``ping`` — liveness helper.

Workspace scoping (§2.3): the AI repo never handles connector credentials, so
the caller passes ``X-Felops-Workspace``.  A real deployment resolves the
connection on the backend; here the header only selects which catalog tools a
caller can see and call.  Any other tool answers ``connector_not_connected``,
which is the same failure the adapter must surface (§4.2).

Determinism: a call whose arguments match the catalog ``examples`` entry
returns that example result, so the demo and the contract tests assert on
fixed values (tracking ``TRK8812``, invoice ``INV-2210``, ...).  Calls with
other arguments succeed with the arguments echoed back as ``data``; this
keeps the mock usable while never inventing business data.

Idempotency (§4.3): the caller's ``idem_key`` is forwarded and recorded but
not enforced here.  Duplicate suppression belongs to the executor's ledger
(3.6) and the backend, not to the mock.
"""

from __future__ import annotations

import copy
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE_HEADER = "X-Felops-Workspace"
SESSION_HEADER = "Mcp-Session-Id"
PROTOCOL_VERSION = "2025-06-18"
IDEM_KEY_META = "felops/idem_key"
SERVER_NAME = "felops-mock-mcp"
SERVER_VERSION = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CATALOG_PATH = _REPO_ROOT / "fixtures" / "catalog" / "catalog.json"
_CATALOG_ENV = "MOCK_MCP_CATALOG_PATH"

_SERVER_META_FIELDS = (
    "tool_id",
    "connector_id",
    "version",
    "deprecated",
    "result_summary",
    "effect_summary",
)


# ---------------------------------------------------------------------------
# Configuration and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MockMcpConfig:
    """Process configuration for the mock MCP server.

    `catalog_path`
        Catalog snapshot to serve.  Defaults to ``fixtures/catalog/catalog.json``
        (overridable with ``MOCK_MCP_CATALOG_PATH``).
    `workspace_tools`
        Optional workspace -> tool-id allowlist.  ``None`` (the demo default)
        exposes every catalog tool to every workspace; tests use a mapping to
        prove workspace isolation.
    `failures`
        Optional tool-id -> execution error payload (``code``, ``message``,
        ``retryable``).  Used to exercise the failure path without lying about
        the catalog.
    `protocol_version`
        MCP protocol version reported by ``initialize``.
    """

    catalog_path: Path | None = None
    workspace_tools: Mapping[str, Sequence[str]] | None = None
    failures: Mapping[str, Mapping[str, Any]] | None = None
    protocol_version: str = PROTOCOL_VERSION


class _MockMcpState:
    """In-memory catalog, workspace policy and call log for one app instance."""

    def __init__(self, config: MockMcpConfig) -> None:
        configured_path = config.catalog_path or os.getenv(_CATALOG_ENV)
        self.catalog_path = Path(configured_path) if configured_path else _DEFAULT_CATALOG_PATH
        self.workspace_tools = config.workspace_tools
        self.failures = dict(config.failures or {})
        self.protocol_version = config.protocol_version

        catalog = _load_json(self.catalog_path)
        self.tools: dict[str, dict[str, Any]] = {tool["tool_id"]: tool for tool in catalog["tools"]}
        self.examples: dict[str, dict[str, Any]] = dict(catalog.get("examples") or {})

        self.calls: list[dict[str, Any]] = []
        self.sessions: dict[str, str] = {}
        self.initializations = 0

    def tool_ids_for(self, workspace_id: str) -> list[str]:
        """Return the tool ids exposed to *workspace_id* (catalog order kept)."""
        if self.workspace_tools is None:
            return list(self.tools)
        allowed = set(self.workspace_tools.get(workspace_id, ()))
        return [tool_id for tool_id in self.tools if tool_id in allowed]

    def list_tools(self, workspace_id: str) -> list[dict[str, Any]]:
        """Return MCP ``Tool`` descriptors for *workspace_id*."""
        return [self._mcp_tool(self.tools[tool_id]) for tool_id in self.tool_ids_for(workspace_id)]

    def call(
        self,
        workspace_id: str,
        tool_id: str,
        arguments: Mapping[str, Any],
        idem_key: str,
    ) -> dict[str, Any]:
        """Execute *tool_id* for *workspace_id* and return an execution envelope."""
        if tool_id not in self.tool_ids_for(workspace_id):
            envelope = _failure_envelope(
                "connector_not_connected",
                f"tool '{tool_id}' is not available in workspace '{workspace_id}'",
            )
        elif tool_id in self.failures:
            failure = self.failures[tool_id]
            envelope = _failure_envelope(
                str(failure.get("code") or "provider_error"),
                str(failure.get("message") or f"tool '{tool_id}' failed"),
                retryable=bool(failure.get("retryable", False)),
            )
        else:
            envelope = self._success_envelope(tool_id, arguments)

        self.calls.append(
            {
                "workspace_id": workspace_id,
                "tool_id": tool_id,
                "arguments": copy.deepcopy(dict(arguments)),
                "idem_key": idem_key,
                "result": copy.deepcopy(envelope),
            }
        )
        return envelope

    def _success_envelope(self, tool_id: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        example = self.examples.get(tool_id)
        if example is not None and example.get("arguments") == dict(arguments):
            return copy.deepcopy(example["result"])
        return {
            "status": "succeeded",
            "data": {**arguments, "tool_id": tool_id},
            "error": None,
        }

    @staticmethod
    def _mcp_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
        """Project a catalog entry onto the MCP ``Tool`` shape.

        Catalog-only fields travel in ``_meta.felops`` so the Connector adapter
        can rebuild a full ``ToolSpec`` without a second source of truth.
        """
        return {
            "name": tool["tool_id"],
            "description": tool.get("description", ""),
            "inputSchema": tool.get("parameters")
            or {"type": "object", "additionalProperties": False},
            "_meta": {"felops": {field: tool.get(field) for field in _SERVER_META_FIELDS}},
        }


def _failure_envelope(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "status": "failed",
        "data": None,
        "error": {"code": code, "message": message, "retryable": retryable},
    }


# ---------------------------------------------------------------------------
# JSON-RPC transport
# ---------------------------------------------------------------------------


def _rpc_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _rpc_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _read_idem_key(params: Mapping[str, Any]) -> str:
    meta = params.get("_meta")
    if isinstance(meta, Mapping):
        value = meta.get(IDEM_KEY_META)
        if isinstance(value, str) and value:
            return value
    return ""


def _handle_tools_call(
    state: _MockMcpState,
    request_id: Any,
    params: Mapping[str, Any],
    workspace_id: str,
) -> JSONResponse:
    name = params.get("name")
    raw_arguments = params.get("arguments")
    arguments = {} if raw_arguments is None else raw_arguments
    if not isinstance(name, str) or not name:
        return JSONResponse(_rpc_error(request_id, -32602, "tools/call requires a tool name"))
    if not isinstance(arguments, Mapping):
        return JSONResponse(
            _rpc_error(request_id, -32602, "tools/call arguments must be an object")
        )

    envelope = state.call(workspace_id, name, arguments, _read_idem_key(params))
    return JSONResponse(
        _rpc_result(
            request_id,
            {
                "content": [{"type": "text", "text": json.dumps(envelope, ensure_ascii=False)}],
                "structuredContent": envelope,
                "isError": envelope["status"] == "failed",
            },
        )
    )


async def handle_rpc(request: Request) -> Response:
    """Single JSON-RPC request handler for ``POST /mcp``."""
    state: _MockMcpState = request.app.state.mock_mcp

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(_rpc_error(None, -32700, "Parse error"), status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse(
            _rpc_error(None, -32600, "Invalid Request: batched requests are not supported"),
            status_code=400,
        )

    workspace_id = request.headers.get(WORKSPACE_HEADER, "").strip()
    if not workspace_id:
        return JSONResponse(
            _rpc_error(payload.get("id"), -32602, f"Missing {WORKSPACE_HEADER} header"),
            status_code=400,
        )

    method = payload.get("method")
    request_id = payload.get("id")
    if payload.get("jsonrpc") != "2.0":
        return JSONResponse(
            _rpc_error(request_id, -32600, "Invalid Request: jsonrpc must be '2.0'"),
            status_code=400,
        )
    if not isinstance(method, str) or not method:
        return JSONResponse(
            _rpc_error(request_id, -32600, "Invalid Request: method is required"),
            status_code=400,
        )
    if "id" in payload and not (
        isinstance(request_id, (str, int)) and not isinstance(request_id, bool)
    ):
        return JSONResponse(
            _rpc_error(None, -32600, "Invalid Request: id must be a string or integer"),
            status_code=400,
        )

    session_id = request.headers.get(SESSION_HEADER, "").strip()
    if method != "initialize" and session_id:
        session_workspace = state.sessions.get(session_id)
        if session_workspace is None:
            return JSONResponse(
                _rpc_error(request_id, -32000, "Unknown MCP session"), status_code=404
            )
        if session_workspace != workspace_id:
            return JSONResponse(
                _rpc_error(request_id, -32000, "MCP session belongs to another workspace"),
                status_code=403,
            )

    # MCP notifications (e.g. notifications/initialized) carry no id and get
    # an empty 202, exactly as a real streamable-HTTP server answers.
    if "id" not in payload:
        return Response(status_code=202)

    raw_params = payload.get("params")
    params = {} if raw_params is None else raw_params
    if not isinstance(params, dict):
        return JSONResponse(_rpc_error(request_id, -32602, "Invalid params"))

    if method == "initialize":
        session_id = uuid.uuid4().hex
        state.sessions[session_id] = workspace_id
        state.initializations += 1
        return JSONResponse(
            _rpc_result(
                request_id,
                {
                    "protocolVersion": state.protocol_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            ),
            headers={SESSION_HEADER: session_id},
        )

    if method == "ping":
        return JSONResponse(_rpc_result(request_id, {}))

    if method == "tools/list":
        return JSONResponse(_rpc_result(request_id, {"tools": state.list_tools(workspace_id)}))

    if method == "tools/call":
        return _handle_tools_call(state, request_id, params, workspace_id)

    return JSONResponse(_rpc_error(request_id, -32601, f"Method not found: {method}"))


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def health() -> dict[str, str]:
    return {"status": "ok"}


def create_app(config: MockMcpConfig | None = None) -> FastAPI:
    """Build a fresh mock MCP app (fresh in-memory state per instance)."""
    app = FastAPI(title="Felops mock MCP server", version=SERVER_VERSION)
    app.state.mock_mcp = _MockMcpState(config or MockMcpConfig())
    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/", health, methods=["GET"])
    app.add_api_route("/mcp", handle_rpc, methods=["POST"])
    return app


app = create_app()
