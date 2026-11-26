"""MCP implementation of the `Connector` port (§4 MCP Client, §5.1).

The adapter speaks the JSON-RPC subset of the Model Context Protocol over
HTTP (streamable HTTP transport, JSON or SSE responses).  It keeps one MCP
session per workspace: ``initialize`` runs lazily on the first call, the
``Mcp-Session-Id`` is reused for that workspace afterwards, and every request
carries the protocol version header.  Sessions are never shared across
workspaces, so one workspace can never inherit another's connection state
(§2.3).

Workspace isolation (§2.3): the AI repo never sees connector credentials.  It
sends ``workspace_id`` on every request and the backend resolves the
connection; the mock server uses the same header to scope the exposed tools.

Errors (§4.2): this class never raises for an execution failure.  Timeouts map
to ``provider_error`` (retryable), unreachable or unauthorized connectors map
to ``connector_not_connected``, and a JSON-RPC error or an ``isError`` tool
result maps to ``ToolResult(status="failed")`` with a structured ``ToolError``.
An expired session (HTTP 404 on a request that carried a session id) is
re-initialized and retried once, because the request never executed.
``list_tools`` degrades to an empty list when the server is unreachable so a
disconnected connector cannot crash a run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import ValidationError

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
_MAX_TOOL_PAGES = 50


class _McpFailure(Exception):
    """Internal carrier for a structured tool failure."""

    def __init__(self, error: ToolError, *, status_code: int | None = None) -> None:
        super().__init__(error.message)
        self.error = error
        self.status_code = status_code


@dataclass(frozen=True)
class _Session:
    """State of one initialized MCP session for one workspace."""

    session_id: str | None
    protocol_version: str


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
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._base_url = base_url.strip().rstrip("/")
        self._endpoint = f"{self._base_url}/mcp"
        self._connector_id = connector_id.strip() if connector_id else None
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._sessions: dict[str, _Session] = {}
        self._request_seq = 0
        self._init_lock = asyncio.Lock()

    # -- Connector protocol -------------------------------------------------

    async def list_tools(self, ctx: WorkspaceContext) -> list[ToolSpec]:
        """Return the tools the server exposes to *ctx*.

        Pages through ``nextCursor`` until the server reports no more tools.  A
        disconnected connector yields an empty list (§4.2): the executor then
        denies every call and hands off instead of raising.
        """
        try:
            return await self._load_tool_pages(ctx)
        except _McpFailure as exc:
            logger.warning(
                "mcp tools/list failed for workspace %s: %s",
                ctx.workspace_id,
                exc.error.message,
            )
            return []

    async def _load_tool_pages(self, ctx: WorkspaceContext) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(_MAX_TOOL_PAGES):
            result = await self._request(
                ctx, "tools/list", {"cursor": cursor} if cursor else {}
            )
            self._append_specs(specs, result.get("tools"))
            cursor = _next_cursor(result, seen)
            if cursor is None:
                return specs
        logger.warning(
            "mcp tools/list stopped after %d pages for workspace %s",
            _MAX_TOOL_PAGES,
            ctx.workspace_id,
        )
        return specs

    def _append_specs(self, specs: list[ToolSpec], tools: Any) -> None:
        if not isinstance(tools, list):
            return
        for tool in tools:
            if isinstance(tool, Mapping):
                spec = self._to_spec(tool)
                if spec is not None:
                    specs.append(spec)

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str,
    ) -> ToolResult:
        """Invoke *tool* with *args*; ``idem_key`` is forwarded verbatim (§4.3)."""
        if not isinstance(idem_key, str) or not idem_key.strip():
            raise ValueError("idem_key is required for every connector call")
        try:
            result = await self._request(
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
        """Drop sessions and close the owned HTTP client (no-op if injected)."""
        self._sessions.clear()
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    # -- Session ------------------------------------------------------------

    async def _ensure_initialized(self, ctx: WorkspaceContext) -> _Session:
        session = self._sessions.get(ctx.workspace_id)
        if session is not None:
            return session
        async with self._init_lock:
            session = self._sessions.get(ctx.workspace_id)
            if session is not None:
                return session
            result, headers = await self._post_raw(
                ctx,
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": _CLIENT_INFO,
                },
                session=None,
            )
            session = _Session(
                session_id=headers.get(SESSION_HEADER) or None,
                protocol_version=str(result.get("protocolVersion") or PROTOCOL_VERSION),
            )
            await self._notify(ctx, session, "notifications/initialized")
            # Publish only after the handshake finishes, so a concurrent caller
            # can never send a request on a half-initialized session.
            self._sessions[ctx.workspace_id] = session
            return session

    async def _invalidate(self, ctx: WorkspaceContext, expected: _Session) -> None:
        """Drop *expected* only, so a concurrent re-init is not evicted."""
        async with self._init_lock:
            if self._sessions.get(ctx.workspace_id) == expected:
                self._sessions.pop(ctx.workspace_id, None)

    async def _notify(
        self, ctx: WorkspaceContext, session: _Session, method: str
    ) -> None:
        client = self._client_or_create()
        try:
            await client.post(
                self._endpoint,
                json={"jsonrpc": "2.0", "method": method},
                headers=self._headers(ctx, session),
            )
        except httpx.HTTPError as exc:  # notifications are fire-and-forget
            logger.debug("mcp notification %s failed: %s", method, exc)

    # -- JSON-RPC -----------------------------------------------------------

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    def _headers(self, ctx: WorkspaceContext, session: _Session | None) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            WORKSPACE_HEADER: ctx.workspace_id,
        }
        if session is not None and session.session_id:
            headers[SESSION_HEADER] = session.session_id
        if session is not None and session.protocol_version:
            headers[PROTOCOL_HEADER] = session.protocol_version
        return headers

    async def _request(
        self, ctx: WorkspaceContext, method: str, params: Mapping[str, Any]
    ) -> dict[str, Any]:
        session = await self._ensure_initialized(ctx)
        try:
            result, _ = await self._post_raw(ctx, method, params, session=session)
        except _McpFailure as exc:
            if exc.status_code != 404 or session.session_id is None:
                raise
            logger.info(
                "mcp session expired for workspace %s; re-initializing", ctx.workspace_id
            )
            await self._invalidate(ctx, session)
            session = await self._ensure_initialized(ctx)
            result, _ = await self._post_raw(ctx, method, params, session=session)
        return result

    async def _post_raw(
        self,
        ctx: WorkspaceContext,
        method: str,
        params: Mapping[str, Any],
        session: _Session | None,
    ) -> tuple[dict[str, Any], httpx.Headers]:
        body = self._request_body(method, params)
        response = await self._send(ctx, session, body)
        if response.status_code >= 400:
            raise _McpFailure(
                _rpc_error_from_response(response), status_code=response.status_code
            )
        payload = self._decode_payload(response, body["id"], method)
        return payload, response.headers

    def _request_body(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        self._request_seq += 1
        return {
            "jsonrpc": "2.0",
            "id": self._request_seq,
            "method": method,
            "params": dict(params),
        }

    async def _send(
        self,
        ctx: WorkspaceContext,
        session: _Session | None,
        body: Mapping[str, Any],
    ) -> httpx.Response:
        try:
            return await self._client_or_create().post(
                self._endpoint, json=body, headers=self._headers(ctx, session)
            )
        except httpx.TimeoutException as exc:
            method = body.get("method")
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

    @staticmethod
    def _decode_payload(
        response: httpx.Response, request_id: int, method: str
    ) -> dict[str, Any]:
        try:
            payload = _decode_json_rpc(response, request_id)
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
        return result

    # -- Result mapping -----------------------------------------------------

    def _to_spec(self, tool: Mapping[str, Any]) -> ToolSpec | None:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            return None
        try:
            return _tool_spec_from_mcp(tool, name, self._connector_id)
        except (ValidationError, TypeError, ValueError) as exc:
            logger.warning("skipping malformed MCP tool descriptor %r: %s", name, exc)
            return None

    @staticmethod
    def _to_tool_result(result: Mapping[str, Any]) -> ToolResult:
        envelope = _extract_envelope(result)
        if bool(result.get("isError")):
            return _error_result(result, envelope)
        if not isinstance(envelope, Mapping):
            return ToolResult(status="succeeded", data={"result": dict(result)}, error=None)
        if _is_execution_envelope(envelope):
            if envelope.get("status") == "failed":
                return _failure_from_envelope(envelope)
            return _success_from_envelope(envelope)
        return ToolResult(status="succeeded", data=dict(envelope), error=None)


def _next_cursor(result: Mapping[str, Any], seen: set[str]) -> str | None:
    """Return the next page cursor, or ``None`` when the page was the last."""
    cursor = result.get("nextCursor")
    if not isinstance(cursor, str) or not cursor:
        return None
    if cursor in seen:
        logger.warning("mcp tools/list repeated cursor %r", cursor)
        return None
    seen.add(cursor)
    return cursor


def _published_metadata(tool: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the catalog metadata a server publishes under ``_meta.felops``."""
    meta = tool.get("_meta")
    felops_meta = meta.get("felops") if isinstance(meta, Mapping) else None
    return felops_meta if isinstance(felops_meta, Mapping) else {}


def _tool_spec_from_mcp(
    tool: Mapping[str, Any], name: str, fallback_connector_id: str | None
) -> ToolSpec:
    """Build a ``ToolSpec`` from an MCP ``Tool`` descriptor.

    Raises ``ValidationError``/``TypeError``/``ValueError`` for a malformed
    descriptor; the caller skips that tool instead of failing discovery.
    """
    published = _published_metadata(tool)
    connector_id = (
        published.get("connector_id") or fallback_connector_id or _DEFAULT_CONNECTOR_ID
    )
    parameters = tool.get("inputSchema")
    return ToolSpec(
        tool_id=str(published.get("tool_id") or name),
        connector_id=str(connector_id),
        description=str(tool.get("description") or ""),
        parameters=dict(parameters) if isinstance(parameters, Mapping) else {},
        result_summary=str(published.get("result_summary") or ""),
        effect_summary=published.get("effect_summary"),
        version=int(published.get("version") or 1),
        deprecated=bool(published.get("deprecated", False)),
    )


def _decode_json_rpc(response: httpx.Response, request_id: int) -> dict[str, Any]:
    """Decode a JSON body or the SSE event carrying the matching response."""
    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("text/event-stream"):
        for payload in _parse_sse_events(response.text):
            if payload.get("id") == request_id:
                return payload
        raise ValueError(f"SSE stream carried no response for id {request_id}")

    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("JSON-RPC response is not an object")
    return payload


def _parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Parse SSE frames into JSON objects, ignoring non-JSON frames."""
    events: list[dict[str, Any]] = []
    data_lines: list[str] = []

    def flush() -> None:
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        data_lines.clear()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if isinstance(payload, dict):
            events.append(payload)

    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif not line.strip():
            flush()
    flush()
    return events


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


def _content_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts = [
        item["text"]
        for item in content
        if isinstance(item, Mapping)
        and item.get("type") == "text"
        and isinstance(item.get("text"), str)
    ]
    return "\n".join(parts).strip()


def _extract_envelope(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the structured payload of a tool result, if it has one."""
    envelope = result.get("structuredContent")
    if isinstance(envelope, Mapping):
        return envelope
    return _parse_text_content(result.get("content"))


def _error_result(
    result: Mapping[str, Any], envelope: Mapping[str, Any] | None
) -> ToolResult:
    """Map an ``isError`` tool result to a structured failure."""
    text = _content_text(result.get("content"))
    if envelope is not None:
        return _failure_from_envelope(envelope, fallback_message=text)
    return ToolResult(
        status="failed",
        data=None,
        error=_tool_error("provider_error", text or "MCP tool call failed"),
    )


def _success_from_envelope(envelope: Mapping[str, Any]) -> ToolResult:
    data = envelope.get("data")
    return ToolResult(
        status="succeeded",
        data=dict(data) if isinstance(data, Mapping) else None,
        error=None,
    )


def _is_execution_envelope(payload: Mapping[str, Any]) -> bool:
    """Return ``True`` only for the Felops ``{status, data, error}`` envelope."""
    return payload.get("status") in ("succeeded", "failed") and (
        "data" in payload or "error" in payload
    )


def _failure_from_envelope(
    envelope: Mapping[str, Any], fallback_message: str | None = None
) -> ToolResult:
    error = envelope.get("error")
    payload = error if isinstance(error, Mapping) else {}
    return ToolResult(
        status="failed",
        data=None,
        error=ToolError(
            code=str(payload.get("code") or "provider_error"),
            message=str(payload.get("message") or fallback_message or "MCP tool call failed"),
            retryable=bool(payload.get("retryable", False)),
        ),
    )


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
    retryable = response.status_code >= 500 or response.status_code in (408, 429)
    return _tool_error("provider_error", message, retryable=retryable)


__all__ = [
    "IDEM_KEY_META",
    "McpConnector",
    "PROTOCOL_VERSION",
    "SESSION_HEADER",
    "WORKSPACE_HEADER",
]
