
from __future__ import annotations

from felops.ports.prompts import (
    PromptContext,
    PromptNotFound,
    PromptScope,
    PromptVersion,
)


class StubPromptStore:
    def __init__(self) -> None:
        self._items: dict[tuple, PromptVersion] = {}

    def add(
        self,
        scope: PromptScope,
        name: str,
        template: str,
        *,
        workspace_id: str | None = None,
        agent_id: str | None = None,
        sub_agent_id: str | None = None,
        version: str = "1",
    ) -> None:
        key = (scope, workspace_id, agent_id, sub_agent_id, name)
        self._items[key] = PromptVersion(name, version, template, scope)

    def resolve(self, name: str, ctx: PromptContext) -> PromptVersion:
        S = PromptScope
        candidates = (
            (S.WORKSPACE_AGENT_SUB_AGENT, ctx.workspace_id, ctx.agent_id, ctx.sub_agent_id),
            (S.WORKSPACE_SUB_AGENT, ctx.workspace_id, None, ctx.sub_agent_id),
            (S.PLATFORM_SUB_AGENT, None, None, ctx.sub_agent_id),
            (S.PLATFORM, None, None, None),
        )
        for scope, ws, agent, sub in candidates:
            found = self._items.get((scope, ws, agent, sub, name))
            if found is not None:
                return found
        raise PromptNotFound(name)