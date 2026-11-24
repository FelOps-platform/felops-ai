"""Tests for the mock backend (task 0.8, §7.1, ``ai-service-contract.md``).

The mock stands in for the backend-owned surface: system catalog (pull and
push), ``agent_config`` fixtures, KB file URLs for ingestion, and the webhook
sink.  Each test builds a fresh app so in-memory webhook state never leaks
between tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mocks.mock_backend.app import create_app

from felops.domain.state import AgentConfig

_CATALOG_FIELDS = {
    "tool_id",
    "connector_id",
    "description",
    "parameters",
    "result_summary",
    "effect_summary",
    "version",
    "deprecated",
}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(create_app(ai_identity_token="")) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health & removed token surface
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_endpoint_does_not_exist(client: TestClient) -> None:
    """Credentials never leave the backend (§2, execution plan)."""
    assert client.get("/connections/shopify/token").status_code == 404


# ---------------------------------------------------------------------------
# Authentication (AI_IDENTITY_TOKEN)
# ---------------------------------------------------------------------------


def test_ai_identity_token_enforced_when_configured() -> None:
    app = create_app(ai_identity_token="secret-ai-token")
    with TestClient(app) as authed_client:
        assert authed_client.get("/catalog").status_code == 401
        assert (
            authed_client.get(
                "/catalog",
                headers={"Authorization": "Bearer wrong-token"},
            ).status_code
            == 401
        )
        ok = authed_client.get(
            "/catalog",
            headers={"Authorization": "Bearer secret-ai-token"},
        )
        assert ok.status_code == 200

        assert authed_client.post("/webhooks/ai", json={"type": "job.completed"}).status_code == 401
        ok_hook = authed_client.post(
            "/webhooks/ai",
            json={"type": "job.completed"},
            headers={"Authorization": "Bearer secret-ai-token"},
        )
        assert ok_hook.status_code == 202


# ---------------------------------------------------------------------------
# Catalog endpoints (§4.3, §7)
# ---------------------------------------------------------------------------


def test_get_catalog_returns_full_six_tool_snapshot(client: TestClient) -> None:
    response = client.get("/catalog?full=true")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"tools"}
    tools = body["tools"]
    assert len(tools) == 6
    assert {tool["tool_id"] for tool in tools} == {
        "lookup_order",
        "refund_order",
        "send_invoice",
        "get_invoice",
        "create_case",
        "get_case",
    }
    for tool in tools:
        assert set(tool.keys()) == _CATALOG_FIELDS


def test_get_catalog_summary_mode(client: TestClient) -> None:
    response = client.get("/catalog?full=false")

    assert response.status_code == 200
    tools = response.json()["tools"]
    assert len(tools) == 6
    assert all(
        set(tool.keys()) == {"tool_id", "connector_id", "version", "deprecated"} for tool in tools
    )


def test_push_catalog_sends_tools_snapshot_to_ai_service() -> None:
    captured: list[dict[str, Any]] = []

    def fake_push(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        captured.append({"url": url, "payload": payload, "headers": headers})
        return {"applied": True}

    app = create_app(
        ai_identity_token="",
        backend_identity_token="secret-backend-token",
        ai_service_url="http://ai-service:8000",
        push_handler=fake_push,
    )
    with TestClient(app) as push_client:
        response = push_client.post("/catalog/push")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "pushed": True,
        "target": "http://ai-service:8000/catalog",
        "tool_count": 6,
        "response": {"applied": True},
    }
    assert len(captured) == 1
    assert captured[0]["url"] == "http://ai-service:8000/catalog"
    assert captured[0]["headers"]["Authorization"] == "Bearer secret-backend-token"
    assert set(captured[0]["payload"].keys()) == {"tools"}
    assert len(captured[0]["payload"]["tools"]) == 6


# ---------------------------------------------------------------------------
# Agent config fixtures (§3.1, §7.1)
# ---------------------------------------------------------------------------


def test_agent_config_fixture_matches_contract_and_catalog(client: TestClient) -> None:
    catalog_ids = {tool["tool_id"] for tool in client.get("/catalog").json()["tools"]}
    response = client.get("/fixtures/agent-config")

    assert response.status_code == 200
    cfg = AgentConfig.model_validate(response.json())
    assert cfg.name == "Sara Support"
    assert cfg.sub_agents == ["customer_support"]
    tool_flags = {tool.tool_id: tool.requires_approval for tool in cfg.tools}
    assert set(tool_flags).issubset(catalog_ids)
    assert tool_flags["lookup_order"] is False
    assert tool_flags["send_invoice"] is True
    assert "refund_order" not in tool_flags


def test_draft_config_fixture_matches_contract(client: TestClient) -> None:
    response = client.get("/fixtures/draft-config")

    assert response.status_code == 200
    cfg = AgentConfig.model_validate(response.json())
    assert cfg.sub_agents == ["customer_support"]
    assert [tool.tool_id for tool in cfg.tools] == ["lookup_order"]


# ---------------------------------------------------------------------------
# Knowledge source file URLs (§6.1, §8.5)
# ---------------------------------------------------------------------------


def test_list_files_exposes_all_bilingual_kb_refs(client: TestClient) -> None:
    response = client.get("/files")

    assert response.status_code == 200
    files = response.json()["files"]
    assert len(files) == 10
    assert {entry["locale"] for entry in files} == {"en", "ar"}
    assert all(entry["type"] == "txt" for entry in files)
    assert all("/files/" in entry["ref"] for entry in files)


def test_get_file_serves_en_and_ar_kb_documents(client: TestClient) -> None:
    en_resp = client.get("/files/en/shipping-delivery.txt")
    ar_resp = client.get("/files/ar/shipping-delivery.ar.txt")

    assert en_resp.status_code == 200
    assert "text/plain" in en_resp.headers["content-type"]
    assert "250" in en_resp.text

    assert ar_resp.status_code == 200
    assert "250" in ar_resp.text


def test_get_file_rejects_unknown_locale_missing_file_and_traversal(client: TestClient) -> None:
    assert client.get("/files/fr/shipping-delivery.txt").status_code == 404
    assert client.get("/files/en/does-not-exist.txt").status_code == 404
    assert client.get("/files/en/..%2Fmanifest.json").status_code == 404


# ---------------------------------------------------------------------------
# Webhook sink (§7.1, §9)
# ---------------------------------------------------------------------------


def test_webhook_sink_accepts_a_contract_event(client: TestClient) -> None:
    response = client.post(
        "/webhooks/ai",
        json={
            "event_id": "evt-1",
            "type": "job.completed",
            "job_id": "job-1",
            "workspace_id": "ws-1",
        },
    )

    assert response.status_code == 202
    assert response.json() == {"received": True, "duplicate": False}


def test_webhook_sink_deduplicates_by_event_id(client: TestClient) -> None:
    payload = {"event_id": "evt-dup", "type": "approval.required", "approval_id": "apr-1"}

    first = client.post("/webhooks/ai", json=payload)
    second = client.post("/webhooks/ai", json=payload)

    assert first.json() == {"received": True, "duplicate": False}
    assert second.json() == {"received": True, "duplicate": True}
    assert len(client.get("/webhooks/ai").json()) == 1


def test_webhook_sink_lists_filters_and_clears_events(client: TestClient) -> None:
    client.post("/webhooks/ai", json={"type": "job.completed", "job_id": "job-1"})
    client.post("/webhooks/ai", json={"type": "approval.required", "approval_id": "ap-1"})

    all_events = client.get("/webhooks/ai").json()
    assert [event["type"] for event in all_events] == ["job.completed", "approval.required"]
    assert all("received_at" in event for event in all_events)

    filtered = client.get("/webhooks/ai?type=approval.required").json()
    assert [event["type"] for event in filtered] == ["approval.required"]

    assert client.delete("/webhooks/ai").status_code == 204
    assert client.get("/webhooks/ai").json() == []


def test_webhook_sink_rejects_unknown_event_type(client: TestClient) -> None:
    response = client.post("/webhooks/ai", json={"type": "not.a.real.event"})

    assert response.status_code == 400
    assert "not.a.real.event" in response.json()["detail"]


def test_webhook_sink_requires_a_type(client: TestClient) -> None:
    response = client.post("/webhooks/ai", json={"job_id": "job-1"})

    assert response.status_code == 422
