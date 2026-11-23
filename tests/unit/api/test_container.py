"""Unit tests for app/container.py and api/deps.py get_graph."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from felops.api.deps import get_graph
from felops.api.main import app
from felops.app.container import Container, init_container


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
