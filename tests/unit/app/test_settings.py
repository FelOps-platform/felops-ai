"""Unit tests for centralized `Settings` (YAML defaults + .env/env overrides)."""

from __future__ import annotations

from pathlib import Path

import pytest

from felops.app.settings import Settings


def test_loads_defaults_from_default_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "EMBEDDING_URL",
        "EMBEDDING_MODEL_ID",
        "EMBEDDING_DIM",
        "QDRANT_URL",
        "QDRANT_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)

    settings = Settings(_env_file=None)

    assert settings.embedding.model_id == "BAAI/bge-m3"
    assert settings.embedding.dim == 1024
    assert settings.embedding.batch_size == 32
    assert settings.embedding.timeout == 30.0
    assert settings.embedding.truncate is True
    assert settings.vector.collection_name == "felops_knowledge"
    assert settings.vector.timeout == 60
    assert settings.embedding_url is None
    assert settings.qdrant_url is None


def test_dotenv_and_env_override_yaml_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "EMBEDDING_URL=http://localhost:8081\n"
        "EMBEDDING_MODEL_ID=from-dotenv\n"
        "QDRANT_URL=http://localhost:6333\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMBEDDING_MODEL_ID", "from-os-env")
    monkeypatch.setenv("EMBEDDING__BATCH_SIZE", "16")
    monkeypatch.setenv("VECTOR__COLLECTION_NAME", "custom_collection")

    settings = Settings(_env_file=env_file)

    assert settings.embedding_url == "http://localhost:8081"
    assert settings.embedding.model_id == "from-os-env"
    assert settings.embedding.batch_size == 16
    assert settings.vector.collection_name == "custom_collection"

    tei_cfg = settings.to_tei_config()
    assert tei_cfg.url == "http://localhost:8081"
    assert tei_cfg.model_id == "from-os-env"
    assert tei_cfg.dim == 1024
    assert tei_cfg.batch_size == 16

    qdrant_cfg = settings.to_qdrant_config()
    assert qdrant_cfg.url == "http://localhost:6333"
    assert qdrant_cfg.collection_name == "custom_collection"
    assert qdrant_cfg.timeout == 60


def test_blank_urls_require_explicit_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_URL", "   ")
    monkeypatch.setenv("QDRANT_URL", "")

    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="EMBEDDING_URL"):
        settings.to_tei_config()
    with pytest.raises(ValueError, match="QDRANT_URL"):
        settings.to_qdrant_config()
