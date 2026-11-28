"""Live HuggingFace TEI contract test (task 1.2).

Run with the TEI container from `docker compose`:

    EMBEDDING_URL=http://localhost:8081 pytest tests/integration/test_tei_embedding_live.py -q

Skipped when `EMBEDDING_URL` is unset, so the default offline suite stays green.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio

from felops.adapters.embedding import TeiEmbeddingModel
from felops.app.settings import Settings
from tests.unit.ports.test_embedding_contract import (
    TestEmbeddingModelContract as _EmbeddingModelContract,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("EMBEDDING_URL"),
    reason="EMBEDDING_URL is not set; start TEI with `make up`",
)


async def _skip_if_tei_unreachable(url: str) -> None:
    try:
        async with httpx.AsyncClient(base_url=url, timeout=5.0) as probe:
            resp = await probe.get("/health")
            resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - depends on environment
        pytest.skip(f"TEI is not reachable at {url}: {exc}")


class TestTeiEmbeddingModelLive(_EmbeddingModelContract):
    """The shared EmbeddingModel contract against a live TEI server."""

    @pytest_asyncio.fixture
    async def model(self) -> AsyncIterator[TeiEmbeddingModel]:
        cfg = Settings().to_tei_config()
        assert cfg.url is not None
        await _skip_if_tei_unreachable(cfg.url)

        adapter = TeiEmbeddingModel(cfg)
        try:
            yield adapter
        finally:
            await adapter.close()


@pytest.mark.asyncio
async def test_live_bge_m3_bilingual_and_non_blocking() -> None:
    cfg = Settings().to_tei_config()
    assert cfg.url is not None
    await _skip_if_tei_unreachable(cfg.url)

    async with TeiEmbeddingModel(cfg) as adapter:
        ticks = 0
        stop = False

        async def _ticker() -> None:
            nonlocal ticks
            while not stop:
                ticks += 1
                await asyncio.sleep(0.005)

        ticker = asyncio.create_task(_ticker())
        try:
            vecs = await adapter.embed_documents(
                [
                    "Where is my order #1042?",
                    "أين طلبي رقم ١٠٤٢؟",
                ]
            )
        finally:
            stop = True
            await ticker

        assert len(vecs) == 2
        for vec in vecs:
            assert len(vec.dense) == adapter.dim == 1024
            assert vec.sparse is None
            assert any(abs(v) > 0.0 for v in vec.dense)
        assert ticks >= 1
