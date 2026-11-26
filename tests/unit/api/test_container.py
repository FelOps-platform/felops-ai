"""Unit tests for app/container.py and api/deps.py get_graph, get_message_service."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from felops.api.deps import get_graph, get_message_service
from felops.api.main import app
from felops.app.container import Container, init_container
from felops.services.message_service import MessageService


def test_init_container():
    container = init_container()
    assert isinstance(container, Container)
    assert isinstance(container.checkpointer, InMemorySaver)
    assert isinstance(container.graph, CompiledStateGraph)
    assert container.graph.checkpointer is container.checkpointer


def test_get_graph_dependency():
    with TestClient(app):
        req = Request({"type": "http", "app": app})
        g = get_graph(req)
        assert isinstance(g, CompiledStateGraph)


def test_get_graph_uninitialized_raises():
    empty_app = FastAPI()
    req = Request({"type": "http", "app": empty_app})
    with pytest.raises(RuntimeError, match="Graph has not been initialized"):
        get_graph(req)


def test_get_message_service_dependency():
    with TestClient(app):
        req = Request({"type": "http", "app": app})
        svc = get_message_service(req)
        assert isinstance(svc, MessageService)


def test_get_message_service_uninitialized_container_raises():
    empty_app = FastAPI()
    req = Request({"type": "http", "app": empty_app})
    with pytest.raises(RuntimeError, match="Container has not been initialized"):
        get_message_service(req)


def test_get_message_service_missing_service_raises():
    partial_app = FastAPI()
    partial_app.state.container = SimpleNamespace(message_service=None)
    req = Request({"type": "http", "app": partial_app})
    with pytest.raises(RuntimeError, match="MessageService has not been initialized"):
        get_message_service(req)
