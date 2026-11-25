from __future__ import annotations

import hmac
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


def _read_configured_token(env_var: str) -> str | None:
    token = os.getenv(env_var)
    if token is None:
        return None
    if not token.strip():
        return None
    return token


def _matches_bearer_token(
    credentials: HTTPAuthorizationCredentials | None, expected_token: str
) -> bool:
    if credentials is None:
        return False
    presented = credentials.credentials
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected_token.encode("utf-8"))


def get_current_workspace_context() -> WorkspaceContext:
    ctx = _workspace_ctx_var.get()
    if ctx is None:
        raise RuntimeError("WorkspaceContext has not been initialized for this request.")
    return ctx


async def verify_backend_identity(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(security)],
) -> None:
    expected_token = _read_configured_token("BACKEND_IDENTITY_TOKEN")
    if expected_token is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="BACKEND_IDENTITY_TOKEN is not configured.",
        )
    if not _matches_bearer_token(credentials, expected_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
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
