"""HTTP integration tests for POST /workspaces/{workspace_id}/conversations/{id}/messages."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient

from felops.api.main import app


def _valid_payload(**overrides) -> dict:
    payload = {
        "agent_id": "agent-1",
        "channel": "widget",
        "sender": {"id": "cus_123", "name": "Sara Ali"},
        "content": "Where is my order?",
        "locale": "en",
        "agent_config": {
            "name": "Customer Experience",
            "instructions": {
                "role": "Help customers.",
                "language_constraints": "",
                "response_format": "",
            },
            "tools": [],
        },
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("BACKEND_IDENTITY_TOKEN", "test-backend-token")
    with TestClient(app) as c:
        yield c


def _headers(idem_key: str = "key-1", token: str = "test-backend-token") -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Idempotency-Key": idem_key,
    }


def test_submit_message_returns_202(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers=_headers("key-202"),
        json=_valid_payload(),
    )
    assert resp.status_code == 202
    assert "job_id" in resp.json()


def test_submit_message_requires_auth(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers={"Idempotency-Key": "key-no-auth"},
        json=_valid_payload(),
    )
    assert resp.status_code == 401

    resp_wrong = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers=_headers("key-bad-auth", token="wrong-token"),
        json=_valid_payload(),
    )
    assert resp_wrong.status_code == 401


def test_submit_message_requires_idempotency_key(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers={"Authorization": "Bearer test-backend-token"},
        json=_valid_payload(),
    )
    assert resp.status_code == 422


def test_submit_message_empty_content_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers=_headers("key-empty-content"),
        json=_valid_payload(content=""),
    )
    assert resp.status_code == 422


def test_submit_message_empty_agent_id_returns_422(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers=_headers("key-empty-agent"),
        json=_valid_payload(agent_id=""),
    )
    assert resp.status_code == 422


def test_submit_message_empty_workspace_id_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/%20%20%20/conversations/conv-1/messages",
        headers=_headers("key-empty-ws"),
        json=_valid_payload(),
    )
    assert resp.status_code == 400


def test_duplicate_idempotency_key_returns_same_job_id(client: TestClient) -> None:
    url = "/workspaces/ws-1/conversations/conv-dup/messages"
    r1 = client.post(url, headers=_headers("key-dup-1"), json=_valid_payload())
    r2 = client.post(
        url,
        headers=_headers("key-dup-1"),
        json=_valid_payload(content="Different content"),
    )
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["job_id"] == r2.json()["job_id"]


class _BlockingGraph:
    """Thread-safe blocking graph stub for testing in-flight job concurrency."""

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate

    async def ainvoke(self, *args, **kwargs):
        await asyncio.to_thread(self._gate.wait, 5.0)
        return {}


def test_submit_message_whitespace_idempotency_key_returns_400(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-1/messages",
        headers=_headers("   "),
        json=_valid_payload(),
    )
    assert resp.status_code == 400


def test_conversation_busy_returns_409(client: TestClient) -> None:
    gate = threading.Event()
    original_graph = app.state.graph
    app.state.graph = _BlockingGraph(gate)
    try:
        url = "/workspaces/ws-1/conversations/conv-busy/messages"
        r1 = client.post(url, headers=_headers("key-busy-1"), json=_valid_payload())
        assert r1.status_code == 202

        r2 = client.post(url, headers=_headers("key-busy-2"), json=_valid_payload())
        assert r2.status_code == 409
    finally:
        gate.set()
        app.state.graph = original_graph


def test_duplicate_idempotency_key_while_running_returns_202(client: TestClient) -> None:
    gate = threading.Event()
    original_graph = app.state.graph
    app.state.graph = _BlockingGraph(gate)
    try:
        url = "/workspaces/ws-1/conversations/conv-dup-running/messages"
        r1 = client.post(url, headers=_headers("key-dup-running"), json=_valid_payload())
        assert r1.status_code == 202

        r2 = client.post(url, headers=_headers("key-dup-running"), json=_valid_payload())
        assert r2.status_code == 202
        assert r2.json()["job_id"] == r1.json()["job_id"]
    finally:
        gate.set()
        app.state.graph = original_graph


def test_response_body_matches_schema(client: TestClient) -> None:
    resp = client.post(
        "/workspaces/ws-1/conversations/conv-schema/messages",
        headers=_headers("key-schema"),
        json=_valid_payload(),
    )
    assert resp.status_code == 202
    body = resp.json()
    assert set(body.keys()) == {"job_id"}
    assert isinstance(body["job_id"], str)
    assert len(body["job_id"]) > 0


def test_different_conversations_independent(client: TestClient) -> None:
    r1 = client.post(
        "/workspaces/ws-1/conversations/conv-a/messages",
        headers=_headers("key-a"),
        json=_valid_payload(),
    )
    r2 = client.post(
        "/workspaces/ws-1/conversations/conv-b/messages",
        headers=_headers("key-b"),
        json=_valid_payload(),
    )
    assert r1.status_code == 202
    assert r2.status_code == 202
    assert r1.json()["job_id"] != r2.json()["job_id"]
