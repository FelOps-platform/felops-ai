"""Centralized application settings (§5.2).

Merges configuration from:
1. Explicit constructor arguments (highest priority)
2. OS environment variables (for example from Docker Compose)
3. Local ``.env`` file
4. ``config/default.yaml`` (shared model and store defaults)
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from felops.adapters.embedding import TeiConfig
from felops.adapters.vector import QdrantConfig


def resolve_config_path() -> Path:
    """Locate ``config/default.yaml`` across repo, test, and container layouts."""
    configured = os.getenv("FELOPS_CONFIG_PATH")
    if configured and configured.strip():
        return Path(configured.strip())

    cwd_candidate = Path.cwd() / "config" / "default.yaml"
    if cwd_candidate.is_file():
        return cwd_candidate

    return Path(__file__).resolve().parents[3] / "config" / "default.yaml"


class EmbeddingSettings(BaseModel):
    """Embedding model defaults loaded from ``config/default.yaml``."""

    model_id: str = Field(min_length=1)
    dim: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    timeout: float = Field(gt=0)
    truncate: bool = True


class VectorSettings(BaseModel):
    """Vector store defaults loaded from ``config/default.yaml``."""

    collection_name: str = Field(min_length=1)
    timeout: int = Field(gt=0)


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


class Settings(BaseSettings):
    """Single source of truth for runtime configuration and adapter configs."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    embedding: EmbeddingSettings
    vector: VectorSettings

    embedding_url: str | None = None
    embedding_model_id: str | None = None
    embedding_dim: int | None = Field(default=None, gt=0)

    qdrant_url: str | None = None
    qdrant_api_key: str | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=resolve_config_path()),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _apply_flat_overrides(self) -> Settings:
        self.embedding_url = _blank_to_none(self.embedding_url)
        self.embedding_model_id = _blank_to_none(self.embedding_model_id)
        self.qdrant_url = _blank_to_none(self.qdrant_url)
        self.qdrant_api_key = _blank_to_none(self.qdrant_api_key)

        if self.embedding_model_id is not None:
            self.embedding.model_id = self.embedding_model_id
        if self.embedding_dim is not None:
            self.embedding.dim = self.embedding_dim
        return self

    def to_tei_config(self) -> TeiConfig:
        """Build `TeiConfig` from centralized YAML + env settings."""
        if self.embedding_url is None:
            raise ValueError("EMBEDDING_URL is not set; cannot build TeiConfig")
        return TeiConfig(
            url=self.embedding_url,
            model_id=self.embedding.model_id,
            dim=self.embedding.dim,
            batch_size=self.embedding.batch_size,
            timeout=self.embedding.timeout,
            truncate=self.embedding.truncate,
        )

    def to_qdrant_config(self) -> QdrantConfig:
        """Build `QdrantConfig` from centralized YAML + env settings."""
        if self.qdrant_url is None:
            raise ValueError("QDRANT_URL is not set; cannot build QdrantConfig")
        return QdrantConfig(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            collection_name=self.vector.collection_name,
            timeout=self.vector.timeout,
        )
