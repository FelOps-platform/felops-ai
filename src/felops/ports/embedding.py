"""EmbeddingModel protocol — §5.1.

The single swap point for embedding generation.  Adapters: BGE-M3
(self-hosted, P1), OpenAI-compatible (P10).

The model runs in a thread pool so it never blocks the API event loop (§3).
"""

from __future__ import annotations

from typing import Protocol

from felops.domain.retrieval_types import Vector


class EmbeddingModel(Protocol):
    """Async embedding-model port.

    Attributes
    ----------
    `model_id` : str
        Identifier of the active model (e.g. ``"bge-m3"``).  Used for
        namespace-per-model versioning so a model change triggers re-indexing.
    `dim` : int
        Dimensionality of the dense output vector.

    Methods
    -------
    `embed_documents`
        Batch-embed a list of texts (ingestion path).
    `embed_query`
        Embed a single query string (retrieval path).
    """

    model_id: str
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    async def embed_query(self, text: str) -> Vector: ...
