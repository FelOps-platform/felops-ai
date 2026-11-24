from __future__ import annotations

import os
from contextvars import ContextVar
from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.graph.state import CompiledStateGraph

from felops.domain.workspace import WorkspaceContext

_workspace_ctx_var: ContextVar[WorkspaceContext | None] = ContextVar(
    "workspace_context", default=None
)

security = HTTPBearer(auto_error=False)


def get_current_workspace_context() -> WorkspaceContext:
    ctx = _workspace_ctx_var.get()
    if ctx is None:
        raise RuntimeError("WorkspaceContext has not been initialized for this request.")
    return ctx


async def verify_backend_identity(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> None:
    expected_token = os.getenv("BACKEND_IDENTITY_TOKEN")
    if not expected_token:
        # If token not configured in environment, allow request (development convenience)
        return
    if credentials is None or credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
        )


async def get_workspace_context(
    workspace_id: Annotated[str, Path(description="Workspace identifier from path")],
    _: Annotated[None, Depends(verify_backend_identity)],
) -> WorkspaceContext:
    if not workspace_id or not workspace_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="workspace_id path parameter cannot be empty.",
        )
    ctx = WorkspaceContext(workspace_id=workspace_id)
    _workspace_ctx_var.set(ctx)
    return ctx


def get_graph(request: Request) -> CompiledStateGraph:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise RuntimeError("Graph has not been initialized on app.state.")
    return graph
