"""Connector protocol — §5.1.

The single swap point for tool transport.  Adapters: MCP server (P1), REST
connector, native SDK.  The executor, verifier and Tool Registry depend on
this port, never on a vendor SDK.

Idempotency (§4.1, §2.2): the caller passes ``idem_key`` iff the call is
side-effecting; a read-only call passes ``None`` and always runs live.  The
connector forwards the key verbatim and never invents one.  Duplicate
suppression belongs to the executor's ledger, not to this port.
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
        Invoke *tool* with *args* and return its result.

    ``tool`` is the connector-level name (``ToolSpec.name``), not the
    workspace ``action_id``; the Tool Registry maps one to the other.

    Isolation (§2.3): ``ctx`` scopes credentials and data on every call.

    Errors: a tool-level failure is returned as ``ToolResult(ok=False)``;
    transport failures (connection, auth, timeout) may raise.
    """

    async def list_tools(self, ctx: WorkspaceContext) -> list[ToolSpec]: ...

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str | None = None,
    ) -> ToolResult: ...
