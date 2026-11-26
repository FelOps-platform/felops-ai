"""MCP implementation of the `Connector` port (§4 MCP Client, §5.1).

The adapter speaks the JSON-RPC subset of the Model Context Protocol over
HTTP (streamable HTTP transport, JSON responses).  It keeps one MCP session
per instance: ``initialize`` runs lazily on the first call, the
``Mcp-Session-Id`` is reused afterwards and every request carries the
protocol version header.

Workspace isolation (§2.3): the AI repo never sees connector credentials.  It
sends ``workspace_id`` on every request and the backend resolves the
connection; the mock server uses the same header to scope the exposed tools.

Errors (§4.2): this class never raises for an execution failure.  Timeouts map
to ``provider_error`` (retryable), unreachable or unauthorized connectors map
to ``connector_not_connected``, and a JSON-RPC error or an ``isError`` tool
result maps to ``ToolResult(status="failed")`` with a structured ``ToolError``.
``list_tools`` degrades to an empty list when the server is unreachable so a
disconnected connector cannot crash a run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from felops.domain.tools import ToolError, ToolResult, ToolSpec
from felops.domain.workspace import WorkspaceContext

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
WORKSPACE_HEADER = "X-Felops-Workspace"
SESSION_HEADER = "Mcp-Session-Id"
PROTOCOL_HEADER = "MCP-Protocol-Version"
IDEM_KEY_META = "felops/idem_key"
_CLIENT_INFO = {"name": "felops-ai", "version": "0.1.0"}
_DEFAULT_CONNECTOR_ID = "mcp"


class _McpFailure(Exception):
    """Internal carrier for a structured tool failure."""

    def __init__(self, error: ToolError) -> None:
        super().__init__(error.message)
        self.error = error


def _tool_error(code: str, message: str, *, retryable: bool = False) -> ToolError:
    return ToolError(code=code, message=message, retryable=retryable)


class McpConnector:
    """Connector adapter for one MCP server.

    Parameters
    ----------
    `base_url` : str
        Server root; requests go to ``{base_url}/mcp``.
    `connector_id` : str | None
        Fallback ``connector_id`` when the server does not publish one in
        ``_meta.felops``.
    `timeout` : float
        Per-request timeout in seconds (session pooling and retries land in
        7.5; this task only needs a bounded call).
    `client` : httpx.AsyncClient | None
        Injected client, used by tests with an ASGI transport.  When omitted
        the adapter owns a client bound to ``base_url``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        connector_id: str | None = None,
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        self._base_url = base_url.rstrip("/")
        self._connector_id = connector_id
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._initialized = False
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._request_seq = 0
        self._init_lock = asyncio.Lock()

    # -- Connector protocol -------------------------------------------------

    async def list_tools(self, ctx: WorkspaceContext) -> list[ToolSpec]:
        """Return the tools the server exposes to *ctx*.

        A disconnected connector yields an empty list (§4.2): the executor then
        denies every call and hands off instead of raising.
        """
        try:
            await self._ensure_initialized(ctx)
            result = await self._post(ctx, "tools/list", {})
        except _McpFailure as exc:
            logger.warning(
                "mcp tools/list failed for workspace %s: %s", ctx.workspace_id, exc.error.message
            )
            return []

        tools = result.get("tools")
        if not isinstance(tools, list):
            return []
        specs: list[ToolSpec] = []
        for tool in tools:
            if isinstance(tool, Mapping):
                spec = self._to_spec(tool)
                if spec is not None:
                    specs.append(spec)
        return specs

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str,
    ) -> ToolResult:
        """Invoke *tool* with *args*; ``idem_key`` is forwarded verbatim (§4.3)."""
        try:
            await self._ensure_initialized(ctx)
            result = await self._post(
                ctx,
                "tools/call",
                {
                    "name": tool,
                    "arguments": args,
                    "_meta": {IDEM_KEY_META: idem_key},
                },
            )
        except _McpFailure as exc:
            return ToolResult(status="failed", data=None, error=exc.error)
        return self._to_tool_result(result)

    async def aclose(self) -> None:
        """Close the owned HTTP client (no-op for an injected client)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Session ------------------------------------------------------------

    async def _ensure_initialized(self, ctx: WorkspaceContext) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            result, headers = await self._post_raw(
                ctx,
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
            )
            self._protocol_version = str(result.get("protocolVersion") or PROTOCOL_VERSION)
            self._session_id = headers.get(SESSION_HEADER) or None
            self._initialized = True
            await self._notify(ctx, "notifications/initialized")

    async def _notify(self, ctx: WorkspaceContext, method: str) -> None:
        client = self._client_or_create()
        try:
            await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "method": method},
                headers=self._headers(ctx),
            )
        except httpx.HTTPError as exc:  # notifications are fire-and-forget
            logger.debug("mcp notification %s failed: %s", method, exc)

    # -- JSON-RPC -----------------------------------------------------------

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    def _headers(self, ctx: WorkspaceContext) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            WORKSPACE_HEADER: ctx.workspace_id,
        }
        if self._session_id:
            headers[SESSION_HEADER] = self._session_id
        if self._protocol_version:
            headers[PROTOCOL_HEADER] = self._protocol_version
        return headers

    async def _post(
        self, ctx: WorkspaceContext, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        result, _ = await self._post_raw(ctx, method, params)
        return result

    async def _post_raw(
        self, ctx: WorkspaceContext, method: str, params: Mapping[str, Any]
    ) -> tuple[dict[str, Any], httpx.Headers]:
        client = self._client_or_create()
        self._request_seq += 1
        body: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._request_seq,
            "method": method,
            "params": dict(params),
        }
        try:
            response = await client.post("/mcp", json=body, headers=self._headers(ctx))
        except httpx.TimeoutException as exc:
            raise _McpFailure(
                _tool_error(
                    "provider_error",
                    f"MCP request '{method}' timed out: {exc}",
                    retryable=True,
                )
            ) from exc
        except httpx.HTTPError as exc:
            raise _McpFailure(
                _tool_error(
                    "connector_not_connected",
                    f"MCP server is not reachable: {exc}",
                )
            ) from exc

        if response.status_code >= 400:
            raise _McpFailure(_rpc_error_from_response(response))

        try:
            payload = _decode_json_rpc(response)
        except (ValueError, json.JSONDecodeError) as exc:
            raise _McpFailure(
                _tool_error("provider_error", f"MCP server returned invalid JSON: {exc}")
            ) from exc

        error = payload.get("error")
        if isinstance(error, Mapping):
            raise _McpFailure(
                _tool_error(
                    "provider_error",
                    str(error.get("message") or f"MCP method '{method}' failed"),
                )
            )
        result = payload.get("result")
        if not isinstance(result, dict):
            raise _McpFailure(
                _tool_error("provider_error", f"MCP method '{method}' returned no result")
            )
        return result, response.headers

    # -- Result mapping -----------------------------------------------------

    def _to_spec(self, tool: Mapping[str, Any]) -> ToolSpec | None:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            return None
        meta = tool.get("_meta")
        felops_meta = meta.get("felops") if isinstance(meta, Mapping) else None
        published = felops_meta if isinstance(felops_meta, Mapping) else {}

        tool_id = published.get("tool_id") or name
        connector_id = published.get("connector_id") or self._connector_id
        if not connector_id:
            connector_id = _DEFAULT_CONNECTOR_ID
        parameters = tool.get("inputSchema")
        return ToolSpec(
            tool_id=str(tool_id),
            connector_id=str(connector_id),
            description=str(tool.get("description") or ""),
            parameters=dict(parameters) if isinstance(parameters, Mapping) else {},
            result_summary=str(published.get("result_summary") or ""),
            effect_summary=published.get("effect_summary"),
            version=int(published.get("version") or 1),
            deprecated=bool(published.get("deprecated", False)),
        )

    @staticmethod
    def _to_tool_result(result: Mapping[str, Any]) -> ToolResult:
        envelope = result.get("structuredContent")
        if not isinstance(envelope, Mapping):
            envelope = _parse_text_content(result.get("content"))
        if not isinstance(envelope, Mapping):
            return ToolResult(status="succeeded", data={"result": dict(result)}, error=None)

        error_payload = envelope.get("error")
        if result.get("isError") or envelope.get("status") == "failed":
            error = error_payload if isinstance(error_payload, Mapping) else {}
            return ToolResult(
                status="failed",
                data=None,
                error=ToolError(
                    code=str(error.get("code") or "provider_error"),
                    message=str(error.get("message") or "MCP tool call failed"),
                    retryable=bool(error.get("retryable", False)),
                ),
            )

        data = envelope.get("data")
        if isinstance(data, Mapping):
            return ToolResult(status="succeeded", data=dict(data), error=None)
        return ToolResult(status="succeeded", data=None, error=None)


def _decode_json_rpc(response: httpx.Response) -> dict[str, Any]:
    """Decode a JSON body or the first ``data:`` frame of an SSE response."""
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("text/event-stream"):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data:") :].strip())
                if not isinstance(payload, dict):
                    raise ValueError("SSE data frame is not a JSON object")
                return payload
        raise ValueError("SSE response carried no data frame")
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("JSON-RPC response is not an object")
    return payload


def _parse_text_content(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    if not isinstance(first, Mapping) or first.get("type") != "text":
        return None
    text = first.get("text")
    if not isinstance(text, str):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _rpc_error_from_response(response: httpx.Response) -> ToolError:
    message = f"MCP server returned HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping) and error.get("message"):
            message = str(error["message"])
    if response.status_code in (401, 403, 404):
        return _tool_error("connector_not_connected", message)
    return _tool_error("provider_error", message, retryable=response.status_code >= 500)


__all__ = [
    "IDEM_KEY_META",
    "McpConnector",
    "PROTOCOL_VERSION",
    "SESSION_HEADER",
    "WORKSPACE_HEADER",
]
