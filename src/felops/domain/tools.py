"""Tool value objects shared by the Connector port and the Tool Registry (§4.1, §5.1).

``ToolSpec`` is the canonical description of an action: the registry stores it,
the planner plans over it, and a connector reports what it can call.
``ToolResult`` is the transport-level outcome of one call; the executor wraps
it in a ``ToolOutcome``.  Neither type performs I/O.

``effective_tools(P, S)`` (P1, task 1.11) also lives in the domain layer and
will join this module when the Tool Registry lands.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from felops.domain.state import SideEffectClass


class ToolSpec(BaseModel, frozen=True):
    """A callable action as seen by the planner, policy and executor.

    Attributes
    ----------
    `action_id` : str
        Stable id of the workspace action.  The user's selection ``S`` and an
        approval both refer to it.
    `name` : str
        Connector-level tool name passed to ``Connector.call``.
    `description` : str
        Human- and LLM-readable summary.
    `input_schema` : dict
        JSON Schema for the arguments.
    `side_effect_class` : SideEffectClass
        Decides whether the executor allocates an ``idem_key``.
    """

    action_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    side_effect_class: SideEffectClass


class ToolResult(BaseModel, frozen=True):
    """Outcome of one connector call.

    ``ok=False`` is a tool-level failure (the remote answered with an error),
    not a transport failure: adapters raise for auth, timeout and connection
    problems.  ``error`` carries a short reason suitable for events and the
    trace.
    """

    ok: bool
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
