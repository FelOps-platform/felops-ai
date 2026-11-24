"""Mock backend for the AI repo (task 0.8, §7.1, ``ai-service-contract.md``).

The real backend owns auth, workspaces, agent configs, credentials, the system
catalog, tool execution, and webhooks (§1).  Until it exists, this service
stands in for the backend-owned surface the AI repo and demo scripts use:

* ``GET /catalog`` (and ``POST /catalog/push``) — serves the global system
  catalog ``tools[]`` from ``fixtures/catalog/catalog.json`` and can push it to
  the AI Service's ``PUT /catalog`` endpoint (§4.3, §7).
* ``GET /fixtures/agent-config`` and ``GET /fixtures/draft-config`` — demo
  ``agent_config`` fixtures sent inside AI Service requests (§3.1, §7.1).
* ``GET /files`` and ``GET /files/{locale}/{filename}`` — mock presigned file
  URLs serving ``fixtures/kb/{en,ar}/`` for knowledge ingestion (§6.1, §8.5).
* ``POST /webhooks/ai`` (plus ``GET`` / ``DELETE``) — sink for
  ``job.completed``, ``approval.required``, ``ingestion.status``, and
  ``usage.recorded``, deduplicating by ``event_id`` when present (§9).

There is deliberately no token or credential endpoint: provider credentials
never leave the backend (§2).  State is in memory; restarting clears the
webhook sink.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_FIXTURES = Path(__file__).parent / "fixtures"
_DEFAULT_CATALOG_PATH = _REPO_ROOT / "fixtures" / "catalog" / "catalog.json"
_DEFAULT_KB_DIR = _REPO_ROOT / "fixtures" / "kb"
_ALLOWED_LOCALES = frozenset({"en", "ar"})
_SUMMARY_FIELDS = ("tool_id", "connector_id", "version", "deprecated")
_CONTRACT_EVENT_TYPES = frozenset(
    {
        "job.completed",
        "approval.required",
        "ingestion.status",
        "usage.recorded",
    }
)

_bearer = HTTPBearer(auto_error=False)

PushHandler = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_push_catalog(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {"status": resp.status}
    except (urllib.error.URLError, TimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"failed to push catalog to {url}: {exc}",
        ) from exc


class CatalogPushRequest(BaseModel):
    """Optional override for where to push ``PUT /catalog``."""

    target_url: str | None = None


class WebhookEvent(BaseModel):
    """A webhook envelope sent by the AI repo (§9).

    Only ``type`` is strictly required; ``event_id`` enables idempotent
    deduplication when supplied, and all remaining envelope/data fields are
    preserved for inspection.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    event_id: str | None = None


def create_app(
    *,
    catalog_path: Path | None = None,
    kb_dir: Path | None = None,
    fixtures_dir: Path | None = None,
    ai_identity_token: str | None = None,
    backend_identity_token: str | None = None,
    ai_service_url: str | None = None,
    push_handler: PushHandler | None = None,
) -> FastAPI:
    """Build a fresh mock backend with its own in-memory webhook sink."""
    app = FastAPI(title="Felops mock backend", version="0.2.0")

    resolved_catalog = catalog_path or _DEFAULT_CATALOG_PATH
    resolved_kb_dir = (kb_dir or _DEFAULT_KB_DIR).resolve()
    resolved_fixtures = fixtures_dir or _LOCAL_FIXTURES
    resolved_push = push_handler or _default_push_catalog

    received_events: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()

    def _expected_ai_token() -> str | None:
        if ai_identity_token is not None:
            return ai_identity_token
        return os.getenv("AI_IDENTITY_TOKEN")

    def _resolved_backend_token() -> str | None:
        if backend_identity_token is not None:
            return backend_identity_token
        return os.getenv("BACKEND_IDENTITY_TOKEN")

    def _resolved_ai_service_url() -> str:
        return (ai_service_url or os.getenv("AI_SERVICE_URL") or "http://api:8000").rstrip("/")

    def verify_ai_identity(
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> None:
        expected = _expected_ai_token()
        if not expected:
            return
        if credentials is None or credentials.credentials != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing AI identity token.",
            )

    def _catalog_tools() -> list[dict[str, Any]]:
        catalog = _load_json(resolved_catalog)
        return list(catalog["tools"])

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # System catalog (§4.3, §7)
    # ------------------------------------------------------------------

    @app.get("/catalog", dependencies=[Depends(verify_ai_identity)])
    def get_catalog(
        full: bool = Query(default=True),
    ) -> dict[str, Any]:
        tools = _catalog_tools()
        if not full:
            tools = [{field: tool[field] for field in _SUMMARY_FIELDS} for tool in tools]
        return {"tools": tools}

    @app.post("/catalog/push")
    def push_catalog(body: CatalogPushRequest | None = None) -> dict[str, Any]:
        base_url = (body.target_url if body and body.target_url else _resolved_ai_service_url())
        target = f"{base_url.rstrip('/')}/catalog"
        tools = _catalog_tools()
        headers = {"Content-Type": "application/json"}
        token = _resolved_backend_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"

        remote_response = resolved_push(target, {"tools": tools}, headers)
        return {
            "pushed": True,
            "target": target,
            "tool_count": len(tools),
            "response": remote_response,
        }

    # ------------------------------------------------------------------
    # Agent config fixtures (§3.1, §7.1)
    # ------------------------------------------------------------------

    @app.get("/fixtures/agent-config")
    def agent_config_fixture() -> dict[str, Any]:
        return _load_json(resolved_fixtures / "agent_config.json")

    @app.get("/fixtures/draft-config")
    def draft_config_fixture() -> dict[str, Any]:
        return _load_json(resolved_fixtures / "draft_config.json")

    # ------------------------------------------------------------------
    # Knowledge source file URLs (§6.1, §8.5)
    # ------------------------------------------------------------------

    @app.get("/files")
    def list_files(request: Request) -> dict[str, list[dict[str, str]]]:
        manifest = _load_json(resolved_kb_dir / "manifest.json")
        base_url = str(request.base_url).rstrip("/")
        entries: list[dict[str, str]] = []
        for doc in manifest["documents"]:
            for locale in ("en", "ar"):
                rel_path = Path(doc[f"{locale}_path"])
                filename = rel_path.name
                entries.append(
                    {
                        "doc_id": doc["doc_id"],
                        "locale": locale,
                        "title": doc[f"title_{locale}"],
                        "name": filename,
                        "type": "txt",
                        "ref": f"{base_url}/files/{locale}/{filename}",
                    }
                )
        return {"files": entries}

    @app.get("/files/{locale}/{filename}", response_class=PlainTextResponse)
    def get_file(locale: str, filename: str) -> PlainTextResponse:
        if locale not in _ALLOWED_LOCALES:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown locale: {locale}",
            )
        locale_root = (resolved_kb_dir / locale).resolve()
        candidate = (locale_root / filename).resolve()
        if not candidate.is_relative_to(locale_root) or not candidate.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown file: {locale}/{filename}",
            )
        return PlainTextResponse(
            content=candidate.read_text(encoding="utf-8"),
            media_type="text/plain; charset=utf-8",
        )

    # ------------------------------------------------------------------
    # Webhook sink (§7.1, §9)
    # ------------------------------------------------------------------

    @app.post(
        "/webhooks/ai",
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=[Depends(verify_ai_identity)],
    )
    def receive_webhook(
        event: WebhookEvent,
    ) -> dict[str, bool]:
        if event.type not in _CONTRACT_EVENT_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown event type: {event.type}",
            )
        if event.event_id is not None and event.event_id in seen_event_ids:
            return {"received": True, "duplicate": True}

        payload = event.model_dump(exclude_none=True)
        payload["received_at"] = datetime.now(timezone.utc).isoformat()
        received_events.append(payload)
        if event.event_id is not None:
            seen_event_ids.add(event.event_id)
        return {"received": True, "duplicate": False}

    @app.get("/webhooks/ai")
    def list_webhooks(
        event_type: str | None = Query(default=None, alias="type"),
    ) -> list[dict[str, Any]]:
        if event_type is None:
            return list(received_events)
        return [event for event in received_events if event["type"] == event_type]

    @app.delete("/webhooks/ai", status_code=status.HTTP_204_NO_CONTENT)
    def clear_webhooks() -> None:
        received_events.clear()
        seen_event_ids.clear()

    return app


app = create_app()
