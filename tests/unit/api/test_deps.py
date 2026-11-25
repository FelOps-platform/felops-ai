"""Unit tests for felops.api.deps  get_workspace_context, verify_backend_identity."""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from felops.api.deps import (
    get_current_workspace_context,
    get_workspace_context,
    verify_backend_identity,
)


@pytest.mark.asyncio
async def test_get_workspace_context_valid():
    ctx = await get_workspace_context(workspace_id="ws-123", _=None)
    assert ctx.workspace_id == "ws-123"
    assert get_current_workspace_context().workspace_id == "ws-123"


@pytest.mark.asyncio
async def test_get_workspace_context_empty_raises():
    with pytest.raises(HTTPException) as exc_info:
        await get_workspace_context(workspace_id="   ", _=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_verify_backend_identity_when_unset_fails_closed(monkeypatch):
    monkeypatch.delenv("BACKEND_IDENTITY_TOKEN", raising=False)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="some-token")
    with pytest.raises(HTTPException) as exc_info:
        await verify_backend_identity(credentials=creds)
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize("token", ["", "   ", "\t\n"])
@pytest.mark.asyncio
async def test_verify_backend_identity_when_empty_or_whitespace_fails_closed(monkeypatch, token):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", token)
    with pytest.raises(HTTPException) as exc_info:
        await verify_backend_identity(credentials=None)
    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_verify_backend_identity_valid(monkeypatch):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", "secret-token")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="secret-token")
    await verify_backend_identity(credentials=creds)


@pytest.mark.asyncio
async def test_verify_backend_identity_invalid_token_raises(monkeypatch):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", "secret-token")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong-token")
    with pytest.raises(HTTPException) as exc_info:
        await verify_backend_identity(credentials=creds)
    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_verify_backend_identity_missing_creds_raises(monkeypatch):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", "secret-token")
    with pytest.raises(HTTPException) as exc_info:
        await verify_backend_identity(credentials=None)
    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


@pytest.mark.asyncio
async def test_verify_backend_identity_empty_bearer_creds_raises(monkeypatch):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", "secret-token")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="")
    with pytest.raises(HTTPException) as exc_info:
        await verify_backend_identity(credentials=creds)
    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}


def test_verify_backend_identity_empty_bearer_header_via_http(monkeypatch):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", "secret-token")
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(verify_backend_identity)])
    async def protected():
        return {"ok": True}

    response = TestClient(app).get("/protected", headers={"Authorization": "Bearer "})
    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
