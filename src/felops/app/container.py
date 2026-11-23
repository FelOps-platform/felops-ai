"""Application dependency container (process-lifetime singletons)."""

from __future__ import annotations

from dataclasses import dataclass

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph

from felops.orchestrator.graph import build_graph


@dataclass
class Container:
    checkpointer: BaseCheckpointSaver
    graph: CompiledStateGraph


def init_container(checkpointer: BaseCheckpointSaver | None = None) -> Container:
    """Build the process-level container once.

    Checkpointer construction is owned by build_graph().
    """
    graph = build_graph(checkpointer=checkpointer)
    if graph.checkpointer is None:
        raise RuntimeError("Graph was built without a checkpointer.")
    return Container(checkpointer=graph.checkpointer, graph=graph)
