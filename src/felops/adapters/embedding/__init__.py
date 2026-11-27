"""Embedding adapters. Selected by `embedding.model` in config; wired by `app/container.py`."""

from felops.adapters.embedding.tei import EmbeddingError, TeiConfig, TeiEmbeddingModel

__all__ = ["EmbeddingError", "TeiConfig", "TeiEmbeddingModel"]
