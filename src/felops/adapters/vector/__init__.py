"""Vector-store adapters. Selected by `vector.store` in config; wired only by
`app/container.py`."""

from felops.adapters.vector.qdrant import QdrantVectorStore

__all__ = ["QdrantVectorStore"]
