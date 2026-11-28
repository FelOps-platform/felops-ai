"""Application dependency container (process-lifetime singletons)."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from felops.adapters.embedding import TeiEmbeddingModel
from felops.adapters.vector import QdrantVectorStore
from felops.app.settings import Settings
from felops.orchestrator.graph import build_graph
from felops.ports.embedding import EmbeddingModel
from felops.ports.vector import VectorStore


@dataclass
class Container:
    settings: Settings
    checkpointer: BaseCheckpointSaver
    graph: CompiledStateGraph
    embedding: EmbeddingModel | None = None
    vector_store: VectorStore | None = None


def init_container(
    settings: Settings | None = None,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    embedding: EmbeddingModel | None = None,
    vector_store: VectorStore | None = None,
) -> Container:
    """Build the process-level container once.

    Checkpointer construction is owned by build_graph().  When
    ``settings.embedding_url`` is configured and no ``embedding`` adapter is
    supplied, a ``TeiEmbeddingModel`` is wired from ``settings.to_tei_config()``.
    """
    resolved_settings = settings if settings is not None else Settings()

    graph = build_graph(checkpointer=checkpointer)
    if graph.checkpointer is None:
        raise RuntimeError("Graph was built without a checkpointer.")

    resolved_embedding = embedding
    if resolved_embedding is None and resolved_settings.embedding_url is not None:
        resolved_embedding = TeiEmbeddingModel(resolved_settings.to_tei_config())

    resolved_vector_store = vector_store
    if resolved_vector_store is None and resolved_settings.qdrant_url is not None:
        resolved_vector_store = QdrantVectorStore(resolved_settings.to_qdrant_config())

    return Container(
        settings=resolved_settings,
        checkpointer=graph.checkpointer,
        graph=graph,
        embedding=resolved_embedding,
        vector_store=resolved_vector_store,
    )
