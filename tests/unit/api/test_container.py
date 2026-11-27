"""Unit tests for app/container.py and api/deps.py dependencies."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from felops.adapters.embedding import TeiEmbeddingModel
from felops.api.deps import get_embedding_model, get_graph
from felops.api.main import app
from felops.app.container import Container, init_container
from felops.app.settings import Settings


def test_init_container(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EMBEDDING_URL", raising=False)
    settings = Settings(_env_file=None)
    container = init_container(settings=settings)
    assert isinstance(container, Container)
    assert container.settings is settings
    assert isinstance(container.checkpointer, InMemorySaver)
    assert isinstance(container.graph, CompiledStateGraph)
    assert container.graph.checkpointer is container.checkpointer
    assert container.embedding is None


def test_init_container_wires_tei_when_embedding_url_set(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_URL", "http://tei:80")
    container = init_container()
    assert isinstance(container.embedding, TeiEmbeddingModel)
    assert container.embedding.model_id == "BAAI/bge-m3"
    assert container.embedding.dim == 1024


def test_get_graph_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EMBEDDING_URL", raising=False)
    with TestClient(app):
        req = Request({"type": "http", "app": app})
        g = get_graph(req)
        assert isinstance(g, CompiledStateGraph)


def test_get_embedding_model_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EMBEDDING_URL", "http://tei:80")
    with TestClient(app):
        req = Request({"type": "http", "app": app})
        emb = get_embedding_model(req)
        assert isinstance(emb, TeiEmbeddingModel)


def test_get_graph_uninitialized_raises():
    empty_app = FastAPI()
    req = Request({"type": "http", "app": empty_app})
    with pytest.raises(RuntimeError, match="Graph has not been initialized"):
        get_graph(req)


def test_get_embedding_model_uninitialized_raises():
    empty_app = FastAPI()
    req = Request({"type": "http", "app": empty_app})
    with pytest.raises(RuntimeError, match="EmbeddingModel has not been initialized"):
        get_embedding_model(req)
