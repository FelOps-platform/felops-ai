"""Mock backend for the AI repo (task 0.8, §7.1, ``ai-service-contract.md``)."""

from __future__ import annotations

import hmac
import http.client
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.responses import PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_FIXTURES = Path(__file__).parent / "fixtures"
_DEFAULT_CATALOG_PATH = _REPO_ROOT / "fixtures" / "catalog" / "catalog.json"
_DEFAULT_KB_DIR = _REPO_ROOT / "fixtures" / "kb"
_ALLOWED_LOCALES = frozenset({"en", "ar"})
_SUMMARY_FIELDS = ("tool_id", "connector_id", "version", "deprecated")
_CONTRACT_EVENT_TYPES = frozenset(
    {"job.completed", "approval.required", "ingestion.status", "usage.recorded"}
)
_bearer = HTTPBearer(auto_error=False)
PushHandler = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


@dataclass(frozen=True)
class MockBackendConfig:
    catalog_path: Path | None = None
    kb_dir: Path | None = None
    fixtures_dir: Path | None = None
    ai_identity_token: str | None = None
    backend_identity_token: str | None = None
    ai_service_url: str | None = None
    push_handler: PushHandler | None = None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_push_catalog(
    url: str, payload: dict[str, Any], headers: dict[str, str]
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="PUT")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        UnicodeDecodeError,
    ) as exc:
        raise HTTPException(
            status_code=502, detail=f"failed to push catalog to {url}: {exc}"
        ) from exc
    if not body:
        return {"status": status}
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"failed to push catalog to {url}: invalid JSON response ({exc})",
        ) from exc


class CatalogPushRequest(BaseModel):
    """Optional override for the mock-open dev trigger's ``PUT /catalog`` target."""

    target_url: str | None = None

    @field_validator("target_url")
    @classmethod
    def _validate_target_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        parsed = urlparse(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("target_url must be an http:// or https:// URL")
        return cleaned


class WebhookEvent(BaseModel):
    """A webhook envelope sent by the AI repo (§9)."""

    model_config = ConfigDict(extra="allow")
    event_id: str = Field(min_length=1)
    seq: int = Field(ge=0)
    ts: datetime
    type: str
    workspace_id: str | int
    agent_id: str | int
    sub_agent_id: str
    conversation_id: str | int | None = None
    job_id: str | int | None = None
    data: dict[str, Any]


def _resolve_env_override(explicit: str | None, env_var: str) -> str | None:
    if explicit is not None:
        return explicit
    return os.getenv(env_var)


def _read_configured_token(configured: str | None, env_var: str) -> str | None:
    token = _resolve_env_override(configured, env_var)
    if token is None or not token.strip():
        return None
    return token


def _resolve_ai_service_url(configured_url: str | None) -> str:
    value = configured_url
    if value is None:
        value = os.getenv("AI_SERVICE_URL")
    if value is None:
        value = "http://api:8000"
    if not value:
        value = "http://api:8000"
    return value.rstrip("/")


def _matches_bearer_token(
    credentials: HTTPAuthorizationCredentials | None, expected_token: str
) -> bool:
    if credentials is None:
        return False
    presented = credentials.credentials
    if not presented:
        return False
    return hmac.compare_digest(presented.encode("utf-8"), expected_token.encode("utf-8"))


def _build_file_entry(doc: dict[str, Any], locale: str, base_url: str) -> dict[str, str]:
    rel_path = Path(doc[f"{locale}_path"])
    filename = rel_path.name
    return {
        "doc_id": doc["doc_id"],
        "locale": locale,
        "title": doc[f"title_{locale}"],
        "name": filename,
        "type": "txt",
        "ref": f"{base_url}/files/{locale}/{filename}",
    }


class _MockBackendState:
    def __init__(self, config: MockBackendConfig):
        self.catalog_path = config.catalog_path or _DEFAULT_CATALOG_PATH
        self.kb_dir = (config.kb_dir or _DEFAULT_KB_DIR).resolve()
        self.fixtures_dir = config.fixtures_dir or _LOCAL_FIXTURES
        self.ai_token = _read_configured_token(config.ai_identity_token, "AI_IDENTITY_TOKEN")
        self.backend_token = _read_configured_token(
            config.backend_identity_token, "BACKEND_IDENTITY_TOKEN"
        )
        self.ai_service_url = _resolve_ai_service_url(config.ai_service_url)
        self.push_handler = config.push_handler or _default_push_catalog
        self.received_events: list[dict[str, Any]] = []
        self.seen_event_ids: set[str] = set()
        self.events_lock = threading.Lock()

    def catalog_tools(self) -> list[dict[str, Any]]:
        return list(_load_json(self.catalog_path)["tools"])

    def get_catalog(self, full: bool) -> dict[str, Any]:
        tools = self.catalog_tools()
        if not full:
            tools = [{field: tool[field] for field in _SUMMARY_FIELDS} for tool in tools]
        return {"tools": tools}

    def _resolve_push_base_url(self, body: CatalogPushRequest | None) -> str:
        if body is not None and body.target_url:
            return body.target_url
        return self.ai_service_url

    def push_catalog(self, body: CatalogPushRequest | None) -> dict[str, Any]:
        if self.backend_token is None:
            raise HTTPException(status_code=500, detail="BACKEND_IDENTITY_TOKEN is not configured.")
        target = f"{self._resolve_push_base_url(body).rstrip('/')}/catalog"
        tools = self.catalog_tools()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.backend_token}",
        }
        remote = self.push_handler(target, {"tools": tools}, headers)
        return {"pushed": True, "target": target, "tool_count": len(tools), "response": remote}

    def load_fixture(self, filename: str) -> dict[str, Any]:
        return _load_json(self.fixtures_dir / filename)

    def list_files(self, request: Request) -> dict[str, list[dict[str, str]]]:
        manifest = _load_json(self.kb_dir / "manifest.json")
        base_url = str(request.base_url).rstrip("/")
        entries = [
            _build_file_entry(doc, locale, base_url)
            for doc in manifest["documents"]
            for locale in ("en", "ar")
        ]
        return {"files": entries}

    def _resolve_kb_path(self, locale: str, filename: str) -> Path:
        if locale not in _ALLOWED_LOCALES:
            raise HTTPException(status_code=404, detail=f"unknown locale: {locale}")
        locale_root = (self.kb_dir / locale).resolve()
        candidate = (locale_root / filename).resolve()
        if not candidate.is_relative_to(locale_root):
            raise HTTPException(status_code=404, detail=f"unknown file: {locale}/{filename}")
        if not candidate.is_file():
            raise HTTPException(status_code=404, detail=f"unknown file: {locale}/{filename}")
        return candidate

    def read_kb_file(self, locale: str, filename: str) -> PlainTextResponse:
        path = self._resolve_kb_path(locale, filename)
        return PlainTextResponse(
            path.read_text(encoding="utf-8"), media_type="text/plain; charset=utf-8"
        )

    def receive_webhook(self, event: WebhookEvent) -> dict[str, bool]:
        if event.type not in _CONTRACT_EVENT_TYPES:
            raise HTTPException(status_code=400, detail=f"unknown event type: {event.type}")
        payload = event.model_dump(mode="json", exclude_none=False)
        payload["received_at"] = datetime.now(timezone.utc).isoformat()
        with self.events_lock:
            if event.event_id in self.seen_event_ids:
                return {"received": True, "duplicate": True}
            self.seen_event_ids.add(event.event_id)
            self.received_events.append(payload)
        return {"received": True, "duplicate": False}

    def list_webhooks(self, event_type: str | None) -> list[dict[str, Any]]:
        with self.events_lock:
            snapshot = list(self.received_events)
        if event_type is None:
            return snapshot
        return [event for event in snapshot if event["type"] == event_type]

    def clear_webhooks(self) -> None:
        with self.events_lock:
            self.received_events.clear()
            self.seen_event_ids.clear()


def _get_state(request: Request) -> _MockBackendState:
    return request.app.state.mock_backend


def verify_ai_identity(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Security(_bearer)
) -> None:
    expected = _get_state(request).ai_token
    if expected is None:
        raise HTTPException(status_code=500, detail="AI_IDENTITY_TOKEN is not configured.")
    if not _matches_bearer_token(credentials, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing AI identity token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def health() -> dict[str, str]:
    return {"status": "ok"}


def get_catalog(
    full: bool = Query(default=True), state: _MockBackendState = Depends(_get_state)
) -> dict[str, Any]:
    return state.get_catalog(full)


def push_catalog(
    body: CatalogPushRequest | None = None, state: _MockBackendState = Depends(_get_state)
) -> dict[str, Any]:
    return state.push_catalog(body)


def agent_config_fixture(state: _MockBackendState = Depends(_get_state)) -> dict[str, Any]:
    return state.load_fixture("agent_config.json")


def draft_config_fixture(state: _MockBackendState = Depends(_get_state)) -> dict[str, Any]:
    return state.load_fixture("draft_config.json")


def list_files(
    request: Request, state: _MockBackendState = Depends(_get_state)
) -> dict[str, list[dict[str, str]]]:
    return state.list_files(request)


def get_file(
    locale: str, filename: str, state: _MockBackendState = Depends(_get_state)
) -> PlainTextResponse:
    return state.read_kb_file(locale, filename)


def receive_webhook(
    event: WebhookEvent, state: _MockBackendState = Depends(_get_state)
) -> dict[str, bool]:
    return state.receive_webhook(event)


def list_webhooks(
    event_type: str | None = Query(default=None, alias="type"),
    state: _MockBackendState = Depends(_get_state),
) -> list[dict[str, Any]]:
    return state.list_webhooks(event_type)


def clear_webhooks(state: _MockBackendState = Depends(_get_state)) -> None:
    state.clear_webhooks()


def _register_catalog_routes(app: FastAPI) -> None:
    # GET /catalog is contract-authenticated; POST /catalog/push stays mock-open as a
    # dev/demo trigger, with validated HTTP(S) targets and authenticated outbound pushes.
    app.add_api_route(
        "/catalog", get_catalog, methods=["GET"], dependencies=[Depends(verify_ai_identity)]
    )
    app.add_api_route("/catalog/push", push_catalog, methods=["POST"])


def _register_fixture_routes(app: FastAPI) -> None:
    app.add_api_route("/fixtures/agent-config", agent_config_fixture, methods=["GET"])
    app.add_api_route("/fixtures/draft-config", draft_config_fixture, methods=["GET"])


def _register_file_routes(app: FastAPI) -> None:
    app.add_api_route("/files", list_files, methods=["GET"])
    app.add_api_route(
        "/files/{locale}/{filename}", get_file, methods=["GET"], response_class=PlainTextResponse
    )


def _register_webhook_routes(app: FastAPI) -> None:
    app.add_api_route(
        "/webhooks/ai",
        receive_webhook,
        methods=["POST"],
        status_code=202,
        dependencies=[Depends(verify_ai_identity)],
    )
    app.add_api_route("/webhooks/ai", list_webhooks, methods=["GET"])
    app.add_api_route("/webhooks/ai", clear_webhooks, methods=["DELETE"], status_code=204)


def create_app(config: MockBackendConfig | None = None) -> FastAPI:
    app = FastAPI(title="Felops mock backend", version="0.2.0")
    app.state.mock_backend = _MockBackendState(config or MockBackendConfig())
    app.add_api_route("/health", health, methods=["GET"])
    _register_catalog_routes(app)
    _register_fixture_routes(app)
    _register_file_routes(app)
    _register_webhook_routes(app)
    return app


app = create_app()
