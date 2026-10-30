"""Tests for the mock backend (task 0.8).

The mock stands in for the backend-owned surface the AI repo calls:
`GET /connections/{id}/token`, `POST /webhooks/ai`, and the agent config
fixtures.  Each test builds a fresh app so the in-memory webhook sink does
not leak between tests.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from mocks.mock_backend.app import create_app

_WORKSPACE = {"X-Workspace-Id": "ws-1"}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Token endpoint
# ---------------------------------------------------------------------------


def test_token_requires_workspace_header(client: TestClient) -> None:
    response = client.get("/connections/shopify/token")

    assert response.status_code == 400
    assert response.json()["detail"] == "X-Workspace-Id header is required"


def test_token_is_short_lived_and_workspace_scoped(client: TestClient) -> None:
    response = client.get("/connections/shopify/token", headers=_WORKSPACE)

    assert response.status_code == 200
    token = response.json()
    assert token["token_type"] == "Bearer"
    assert token["workspace_id"] == "ws-1"
    assert token["connection_id"] == "shopify"
    assert token["expires_in"] == 300
    assert token["access_token"]


def test_token_for_unknown_connection_returns_404(client: TestClient) -> None:
    response = client.get("/connections/nope/token", headers=_WORKSPACE)

    assert response.status_code == 404


def test_connections_are_listed(client: TestClient) -> None:
    response = client.get("/connections", headers=_WORKSPACE)

    assert response.status_code == 200
    names = {connection["connection_id"] for connection in response.json()}
    assert names == {"shopify", "quickbooks"}


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


def test_agent_config_fixture_is_served(client: TestClient) -> None:
    response = client.get("/agents/agent-sara/config")

    assert response.status_code == 200
    config = response.json()
    assert config["agent_id"] == "agent-sara"
    assert config["config_version"] == 1
    assert config["action_ids"] == ["shopify.lookup_order", "quickbooks.send_invoice"]
    assert config["sub_agents"] == ["customer_support"]


def test_draft_config_fixture_is_served(client: TestClient) -> None:
    response = client.get("/agents/agent-sara/config/draft")

    assert response.status_code == 200
    draft = response.json()
    assert draft["agent_id"] == "agent-sara"
    assert draft["channels"] == ["widget", "api"]


def test_config_for_unknown_agent_returns_404(client: TestClient) -> None:
    response = client.get("/agents/nobody/config")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Webhook sink
# ---------------------------------------------------------------------------


def test_webhook_sink_accepts_a_contract_event(client: TestClient) -> None:
    response = client.post(
        "/webhooks/ai",
        json={"type": "job.completed", "job_id": "job-1", "workspace_id": "ws-1"},
    )

    assert response.status_code == 202
    assert response.json() == {"received": True}


def test_webhook_sink_lists_received_events(client: TestClient) -> None:
    client.post("/webhooks/ai", json={"type": "job.completed", "job_id": "job-1"})
    client.post("/webhooks/ai", json={"type": "approval.required", "approval_id": "ap-1"})

    response = client.get("/webhooks/ai")

    assert response.status_code == 200
    events = response.json()
    assert [event["type"] for event in events] == ["job.completed", "approval.required"]
    assert all("received_at" in event for event in events)


def test_webhook_sink_rejects_unknown_event_type(client: TestClient) -> None:
    response = client.post("/webhooks/ai", json={"type": "not.a.real.event"})

    assert response.status_code == 400
    assert "not.a.real.event" in response.json()["detail"]


def test_webhook_sink_requires_a_type(client: TestClient) -> None:
    response = client.post("/webhooks/ai", json={"job_id": "job-1"})

    assert response.status_code == 422
