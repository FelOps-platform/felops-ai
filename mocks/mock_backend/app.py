"""Mock backend for the AI repo (task 0.8).

The real backend owns auth, config persistence, billing and channel delivery
(design §1).  Until it exists, this service stands in for the three things
the AI repo calls:

* ``GET /connections/{connection_id}/token`` — short-lived connector
  credential, scoped to the ``X-Workspace-Id`` header (§2.3, §7.1).
* ``POST /webhooks/ai`` — sink for ``job.completed``, ``approval.required``,
  ``ingestion.status`` and ``usage.recorded`` (§7.1).  ``GET /webhooks/ai``
  exposes what was received so tests and demos can assert on it.
* ``GET /agents/{agent_id}/config`` and ``.../config/draft`` — the versioned
  runtime snapshot and wizard draft used by ``PUT /agents/{id}/config`` and
  ``config:validate`` (§7.1, §8.1).

State is in memory; restarting the mock clears the webhook sink.  It is a
mock, not a datastore.
"""

from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict

_FIXTURES = Path(__file__).parent / "fixtures"
_TOKEN_TTL_SECONDS = 300
_CONTRACT_EVENT_TYPES = {
    "job.completed",
    "approval.required",
    "ingestion.status",
    "usage.recorded",
}


def _load_fixture(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _require_workspace(workspace_id: str | None) -> str:
    if not workspace_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Workspace-Id header is required",
        )
    return workspace_id


class WebhookEvent(BaseModel):
    """A webhook as sent by the AI repo.

    Only ``type`` is required; the payload is event-specific (§7.2) and the
    mock keeps whatever it receives so demos can inspect it.
    """

    model_config = ConfigDict(extra="allow")

    type: str


def create_app() -> FastAPI:
    """Build a fresh mock backend, with its own in-memory webhook sink."""
    app = FastAPI(title="Felops mock backend", version="0.1.0")
    received_events: list[dict[str, Any]] = []

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/connections")
    def list_connections(
        x_workspace_id: str | None = Header(default=None, alias="X-Workspace-Id"),
    ) -> list[dict[str, Any]]:
        _require_workspace(x_workspace_id)
        return _load_fixture("connections.json")

    @app.get("/connections/{connection_id}/token")
    def connection_token(
        connection_id: str,
        x_workspace_id: str | None = Header(default=None, alias="X-Workspace-Id"),
    ) -> dict[str, Any]:
        workspace_id = _require_workspace(x_workspace_id)
        connections = _load_fixture("connections.json")
        if connection_id not in {connection["connection_id"] for connection in connections}:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown connection: {connection_id}",
            )

        now = datetime.now(timezone.utc)
        return {
            "access_token": secrets.token_urlsafe(32),
            "token_type": "Bearer",
            "expires_in": _TOKEN_TTL_SECONDS,
            "expires_at": (now + timedelta(seconds=_TOKEN_TTL_SECONDS)).isoformat(),
            "connection_id": connection_id,
            "workspace_id": workspace_id,
        }

    @app.get("/agents/{agent_id}/config")
    def agent_config(agent_id: str) -> dict[str, Any]:
        config = _load_fixture("agent_config.json")
        if config["agent_id"] != agent_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown agent: {agent_id}",
            )
        return config

    @app.get("/agents/{agent_id}/config/draft")
    def draft_config(agent_id: str) -> dict[str, Any]:
        draft = _load_fixture("draft_config.json")
        if draft["agent_id"] != agent_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown agent: {agent_id}",
            )
        return draft

    @app.post("/webhooks/ai", status_code=status.HTTP_202_ACCEPTED)
    def receive_webhook(event: WebhookEvent) -> dict[str, bool]:
        if event.type not in _CONTRACT_EVENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown event type: {event.type}",
            )
        payload = event.model_dump()
        payload["received_at"] = datetime.now(timezone.utc).isoformat()
        received_events.append(payload)
        return {"received": True}

    @app.get("/webhooks/ai")
    def list_webhooks() -> list[dict[str, Any]]:
        return received_events

    return app


app = create_app()
