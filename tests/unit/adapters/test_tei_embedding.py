"""Unit and contract tests for `adapters/embedding/tei.py` (task 1.2).

Reuses the shared `TestEmbeddingModelContract` suite from
``tests/unit/ports/test_embedding_contract.py`` against a mocked TEI HTTP
transport, and verifies batching, error mapping, config validation, and that
async embedding never blocks the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest
import pytest_asyncio
from tests.unit.ports.test_embedding_contract import (
    TestEmbeddingModelContract as _EmbeddingModelContract,
)

from felops.adapters.embedding import EmbeddingError, TeiConfig, TeiEmbeddingModel
from felops.domain.retrieval_types import Vector
from felops.ports.embedding import EmbeddingModel


def _tei_config(
    *,
    model_id: str = "BAAI/bge-m3",
    dim: int = 4,
    batch_size: int = 32,
    timeout: float = 30.0,
    truncate: bool = True,
    url: str | None = None,
) -> TeiConfig:
    return TeiConfig(
        model_id=model_id,
        dim=dim,
        batch_size=batch_size,
        timeout=timeout,
        truncate=truncate,
        url=url,
    )


def _mock_tei_client(
    handler: Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Any],
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://tei-test:80",
    )


def _deterministic_handler(dim: int = 4) -> Callable[[httpx.Request], httpx.Response]:
    def _handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/embed"
        body = json.loads(request.content)
        inputs = body["inputs"]
        vectors = [
            [float(idx + 1) / (pos + 1) for pos in range(dim)]
            for idx, _ in enumerate(inputs)
        ]
        return httpx.Response(200, json=vectors)

    return _handle


class TestTeiEmbeddingModelContract(_EmbeddingModelContract):
    """Run the shared EmbeddingModel contract against `TeiEmbeddingModel`."""

    @pytest_asyncio.fixture
    async def model(self) -> AsyncIterator[TeiEmbeddingModel]:
        async with _mock_tei_client(_deterministic_handler(dim=4)) as client:
            yield TeiEmbeddingModel(_tei_config(dim=4), client=client)


def test_satisfies_port() -> None:
    client = _mock_tei_client(_deterministic_handler(dim=4))
    adapter = TeiEmbeddingModel(_tei_config(dim=4), client=client)
    port: EmbeddingModel = adapter
    assert isinstance(port, EmbeddingModel)
    assert port.model_id == "BAAI/bge-m3"
    assert port.dim == 4


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model_id": "  "}, "model_id"),
        ({"dim": 0}, "dim"),
        ({"batch_size": 0}, "batch_size"),
        ({"timeout": -1.0}, "timeout"),
    ],
)
def test_config_validation(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _tei_config(**kwargs)


def test_requires_url_when_no_client() -> None:
    with pytest.raises(ValueError, match="TeiConfig.url"):
        TeiEmbeddingModel(_tei_config(url=None))


@pytest.mark.asyncio
async def test_embed_documents_empty_makes_no_http_calls() -> None:
    called = False

    def _fail_if_called(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    async with _mock_tei_client(_fail_if_called) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=4), client=client)
        assert await adapter.embed_documents([]) == []
    assert not called


@pytest.mark.asyncio
async def test_embed_documents_splits_into_configured_batches() -> None:
    seen_batches: list[list[str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs = body["inputs"]
        seen_batches.append(inputs)
        assert body["truncate"] is True
        return httpx.Response(200, json=[[0.1, 0.2, 0.3] for _ in inputs])

    async with _mock_tei_client(_handler) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=3, batch_size=2), client=client)
        vectors = await adapter.embed_documents(["a", "b", "c", "d", "e"])

    assert seen_batches == [["a", "b"], ["c", "d"], ["e"]]
    assert len(vectors) == 5
    assert all(isinstance(v, Vector) and v.sparse is None for v in vectors)


@pytest.mark.asyncio
async def test_embed_query_returns_dense_vector_only() -> None:
    async with _mock_tei_client(_deterministic_handler(dim=4)) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=4), client=client)
        vec = await adapter.embed_query("how do I track my order?")

    assert len(vec.dense) == 4
    assert vec.sparse is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_retryable"),
    [(503, True), (429, True), (408, True), (422, False), (400, False)],
)
async def test_http_errors_map_to_embedding_error(
    status_code: int,
    expected_retryable: bool,
) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text="error")

    async with _mock_tei_client(_handler) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=4), client=client)
        with pytest.raises(EmbeddingError) as exc_info:
            await adapter.embed_query("hi")

    assert exc_info.value.retryable is expected_retryable


@pytest.mark.asyncio
async def test_transport_errors_are_retryable() -> None:
    def _timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async with _mock_tei_client(_timeout_handler) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=4), client=client)
        with pytest.raises(EmbeddingError) as exc_info:
            await adapter.embed_query("hi")

    assert exc_info.value.retryable is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [[0.1, 0.2]],  # wrong dim (expected 3)
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # wrong batch size (expected 1)
        [[0.1, True, 0.3]],  # bool component
        {"unexpected": "object"},
    ],
)
async def test_malformed_payload_raises_embedding_error(payload: Any) -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with _mock_tei_client(_handler) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=3), client=client)
        with pytest.raises(EmbeddingError) as exc_info:
            await adapter.embed_query("hi")

    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_invalid_json_raises_embedding_error() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    async with _mock_tei_client(_handler) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=3), client=client)
        with pytest.raises(EmbeddingError, match="invalid JSON"):
            await adapter.embed_query("hi")


@pytest.mark.asyncio
async def test_close_only_closes_owned_client() -> None:
    owned = TeiEmbeddingModel(_tei_config(url="http://tei:80", dim=4))
    _ = owned._get_client()
    assert owned._client is not None
    await owned.close()
    assert owned._client is None

    async with _mock_tei_client(_deterministic_handler(dim=4)) as injected:
        borrowed = TeiEmbeddingModel(_tei_config(dim=4), client=injected)
        await borrowed.close()
        assert not injected.is_closed


@pytest.mark.asyncio
async def test_embed_does_not_block_asyncio_event_loop() -> None:
    async def _slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        body = json.loads(request.content)
        return httpx.Response(200, json=[[0.5, 0.5, 0.5] for _ in body["inputs"]])

    ticks = 0
    stop = False

    async def _ticker() -> None:
        nonlocal ticks
        while not stop:
            ticks += 1
            await asyncio.sleep(0.005)

    async with _mock_tei_client(_slow_handler) as client:
        adapter = TeiEmbeddingModel(_tei_config(dim=3), client=client)
        ticker_task = asyncio.create_task(_ticker())
        try:
            vecs = await adapter.embed_documents(["hello", "مرحبا"])
        finally:
            stop = True
            await ticker_task

    assert len(vecs) == 2
    assert ticks >= 3
