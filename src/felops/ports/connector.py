"""Connector protocol — §5.1.

The single swap point for tool transport.  Adapters: MCP server (P1), REST
connector, native SDK.  The executor and Tool Registry depend on this port,
never on a vendor SDK.

Idempotency (§4.3, ``ai-service-contract.md`` §4.3): **every** call (reads and
writes) carries a required ``idem_key``.  The connector forwards the key
verbatim and never invents one.  Duplicate suppression belongs to the
executor's ledger and the backend, not to this port.
"""

from __future__ import annotations

from typing import Any, Protocol

from felops.domain.tools import ToolResult, ToolSpec
from felops.domain.workspace import WorkspaceContext


class Connector(Protocol):
    """Async tool connector.

    Methods
    -------
    `list_tools`
        Discover the tools this connector exposes for *ctx*.
    `call`
        Invoke *tool* (the stable ``ToolSpec.tool_id``) with *args* and a
        required *idem_key*, returning a structured ``ToolResult``.

    Isolation (§2.3): ``ctx`` scopes tools and workspace execution on every
    call.  Credentials never leave the backend.

    Errors (§4.2): execution failures — including provider errors, exhausted
    retries, and disconnected or unauthorized connectors
    (``code="connector_not_connected"``) — are returned as
    ``ToolResult(status="failed", data=None, error=ToolError(...))`` so the
    executor can route directly to the appropriate handoff path instead of
    crashing with an uncaught exception.
    """

    async def list_tools(self, ctx: WorkspaceContext) -> list[ToolSpec]: ...

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str,
    ) -> ToolResult: ...
