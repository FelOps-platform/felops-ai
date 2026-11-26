"""Live Qdrant contract test (task 1.1).

Run with a Qdrant instance from `docker compose`:

    QDRANT_URL=http://localhost:6333 .venv/bin/pytest tests/integration -q

Skipped when `QDRANT_URL` is unset, so the default offline suite stays green.
The fixture uses a throw-away collection and removes it afterwards.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from felops.adapters.vector import QdrantVectorStore
from tests.unit.ports.test_vector_contract import (
    TestVectorStoreContract as _VectorStoreContract,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("QDRANT_URL"),
    reason="QDRANT_URL is not set; start Qdrant with `docker compose up -d qdrant`",
)


class TestQdrantVectorStoreLive(_VectorStoreContract):
    """The shared contract, against a real Qdrant server."""

    @pytest_asyncio.fixture
    async def store(self) -> AsyncIterator[QdrantVectorStore]:
        url = os.environ["QDRANT_URL"]
        client = AsyncQdrantClient(url=url, timeout=10)
        try:
            await client.collection_exists("__felops_health__")
        except Exception as exc:  # pragma: no cover - depends on the environment
            await client.close()
            pytest.skip(f"Qdrant is not reachable at {url}: {exc}")

        collection = f"felops_test_{uuid.uuid4().hex}"
        adapter = QdrantVectorStore(client=client, collection_name=collection)
        try:
            yield adapter
        finally:
            try:
                if await client.collection_exists(collection):
                    await client.delete_collection(collection)
            finally:
                await adapter.close()
