"""Loads `<repo>/.env` for the live LLM tests so `API_KEY` can live in a
git-ignored file instead of the shell. Real environment variables always win.
Minimal on purpose: no python-dotenv dependency."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


_load_env_file(_ENV_FILE)
