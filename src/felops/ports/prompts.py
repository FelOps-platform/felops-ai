
from __future__ import annotations
 
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol, runtime_checkable
 
 
class PromptNotFound(LookupError):
    """No template named `name` exists at any level of the resolution order."""
 
 
class PromptScope(IntEnum):
    """Level a resolved prompt came from. Lower value = more specific."""
 
    WORKSPACE_AGENT_SUB_AGENT = 1
    WORKSPACE_SUB_AGENT = 2
    PLATFORM_SUB_AGENT = 3
    PLATFORM = 4
 
 
@dataclass(frozen=True, slots=True)
class PromptContext:
    """Context supplied when resolving a prompt.

    `sub_agent_id` is the per-sub-agent seam (§4.2): V1 always passes
    `customer_support`. `node` is trace context, not a resolution key.
    """
 
    workspace_id: str
    agent_id: str
    sub_agent_id: str
    # Pin to a run execution (§2.3, §4). Postgres store (P4) pins template versions per run.
    run_id: str | None = None
    node: str | None = None
 
 
@dataclass(frozen=True, slots=True)
class PromptVersion:
    name: str
    version: str
    template: str
    scope: PromptScope
 
 
@runtime_checkable
class PromptStore(Protocol):  # versioned, per-workspace, per-agent, per-sub-agent
    def resolve(self, name: str, ctx: PromptContext) -> PromptVersion:
        """Return the most specific template for `name`.
 
        Raises `PromptNotFound` if no level has it. Must never return a
        template from another workspace's override.
        """
        ...
 