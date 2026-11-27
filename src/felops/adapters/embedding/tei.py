"""HuggingFace TEI implementation of the `EmbeddingModel` port (§3, §4, §5.1, task 1.2).

Text Embeddings Inference (TEI) runs out-of-process in its own Docker container
(CPU or GPU profile) and exposes `POST /embed`.  This adapter communicates with
it over ``httpx.AsyncClient`` so embedding inference never blocks the API
asyncio event loop.

P1 produces dense vectors only (`sparse=None`).  TEI serves `BAAI/bge-m3`
dense embeddings (1024-dim, 8k context); lexical BM25 sparse vectors arrive
in P5 via Qdrant (task 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from felops.domain.retrieval_types import Vector

_RETRYABLE_STATUS = frozenset({408, 429})


class EmbeddingError(RuntimeError):
    """Raised when the embedding service fails or returns an invalid payload."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class TeiConfig:
    """Connection and model settings for `TeiEmbeddingModel`.

    Constructed centrally by `felops.app.settings.Settings.to_tei_config()`
    (or directly in tests).
    """

    model_id: str
    dim: int
    batch_size: int
    timeout: float
    truncate: bool = True
    url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


def _parse_vectors(payload: Any, expected_count: int, expected_dim: int) -> list[Vector]:
    """Validate TEI `/embed` JSON output (`list[list[float]]`) into `Vector`s."""
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise EmbeddingError(
            f"TEI returned invalid batch size: expected {expected_count}, "
            f"got {len(payload) if isinstance(payload, list) else type(payload).__name__}"
        )

    vectors: list[Vector] = []
    for row in payload:
        if not isinstance(row, list) or len(row) != expected_dim:
            actual = len(row) if isinstance(row, list) else type(row).__name__
            raise EmbeddingError(
                f"TEI returned unexpected vector dimension: expected {expected_dim}, got {actual}"
            )
        dense: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EmbeddingError("TEI returned a non-numeric vector component")
            dense.append(float(value))
        vectors.append(Vector(dense=dense, sparse=None))
    return vectors


class TeiEmbeddingModel:
    """Async `EmbeddingModel` adapter backed by HuggingFace TEI (`POST /embed`)."""

    def __init__(
        self,
        config: TeiConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self.model_id = config.model_id
        self.dim = config.dim

        if client is None:
            if config.url is None or not config.url.strip():
                raise ValueError(
                    "TeiConfig.url is required when no httpx.AsyncClient is provided"
                )
            self._base_url: str | None = config.url.strip().rstrip("/")
        else:
            self._base_url = None

        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            assert self._base_url is not None
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._config.timeout,
            )
        return self._client

    async def _embed_batch(self, batch: list[str]) -> list[Vector]:
        client = self._get_client()
        try:
            response = await client.post(
                "/embed",
                json={"inputs": batch, "truncate": self._config.truncate},
            )
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"TEI request timed out: {type(exc).__name__}", retryable=True
            ) from exc
        except httpx.TransportError as exc:
            raise EmbeddingError(
                f"TEI transport error: {type(exc).__name__}", retryable=True
            ) from exc

        if response.status_code >= 500 or response.status_code in _RETRYABLE_STATUS:
            raise EmbeddingError(
                f"TEI server error (status={response.status_code})",
                retryable=True,
            )
        if response.status_code >= 400:
            raise EmbeddingError(
                f"TEI rejected request (status={response.status_code})",
                retryable=False,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbeddingError("TEI returned invalid JSON", retryable=False) from exc

        return _parse_vectors(payload, expected_count=len(batch), expected_dim=self.dim)

    async def embed_documents(self, texts: list[str]) -> list[Vector]:
        """Batch-embed *texts* via TEI without blocking the event loop."""
        if not texts:
            return []

        vectors: list[Vector] = []
        batch_size = self._config.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors

    async def embed_query(self, text: str) -> Vector:
        """Embed a single query string via TEI."""
        vectors = await self._embed_batch([text])
        return vectors[0]

    async def close(self) -> None:
        """Close the underlying `httpx.AsyncClient` if owned by this adapter."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> TeiEmbeddingModel:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
