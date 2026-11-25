"""File-backed `PromptStore` (dev source, `prompts.source: file`, §5.2).

Layout under the prompts root (the repo's `prompts/` directory):

    prompts/<sub_agent_id>/<name>.md   -> PromptScope.PLATFORM_SUB_AGENT
    prompts/platform/<name>.md         -> PromptScope.PLATFORM

Resolution is the platform half of the §2.3 order: the platform sub-agent
default wins over the platform default. The two workspace-level scopes belong
to the Postgres store (P4), where per-workspace overrides live. This store
holds none, so it cannot return another workspace's override: `workspace_id`
never selects a file here.

Files are read on every call, so editing a template needs no restart or deploy.
The version is a content hash, so it is stable, changes whenever the text
changes, and can be recorded in `config_version`.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from felops.ports.prompts import (
    PromptContext,
    PromptNotFound,
    PromptScope,
    PromptVersion,
)

PLATFORM_DIR = "platform"
TEMPLATE_SUFFIX = ".md"

# Names become path segments, so only plain identifiers are accepted. This
# rejects "../x", absolute paths and separators before any filesystem access.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class FilePromptStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def resolve(self, name: str, ctx: PromptContext) -> PromptVersion:
        if not _SAFE_ID.fullmatch(name):
            raise PromptNotFound(f"invalid prompt name: {name!r}")

        sub_agent = ctx.sub_agent_id
        if sub_agent is not None and not _SAFE_ID.fullmatch(sub_agent):
            raise PromptNotFound(f"invalid sub-agent id: {sub_agent!r}")

        candidates: list[tuple[PromptScope, Path]] = []
        if sub_agent is not None and sub_agent.casefold() != PLATFORM_DIR:
            candidates.append(
                (
                    PromptScope.PLATFORM_SUB_AGENT,
                    self._root / sub_agent / f"{name}{TEMPLATE_SUFFIX}",
                )
            )
        candidates.append(
            (PromptScope.PLATFORM, self._root / PLATFORM_DIR / f"{name}{TEMPLATE_SUFFIX}")
        )

        root = self._root.resolve()
        for scope, path in candidates:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise PromptNotFound(
                    f"prompt path escapes the prompts root: {path!s}"
                )
            if not resolved.is_file():
                continue
            try:
                template = resolved.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
            except UnicodeDecodeError as exc:
                raise PromptNotFound(
                    f"prompt template is not valid UTF-8: {name!r}"
                ) from exc
            return PromptVersion(
                name=name,
                version="sha256:" + hashlib.sha256(template.encode("utf-8")).hexdigest()[:16],
                template=template,
                scope=scope,
            )
        raise PromptNotFound(f"no prompt template named {name!r} for sub-agent {sub_agent!r}")
