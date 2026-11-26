"""Tool value objects shared by the Connector port and the Tool Registry (§4.1, §4.3, §5.1).

``ToolSpec`` is the canonical system-catalog entry for a tool: the Tool Registry
stores it, the planner plans over it, and a connector reports what it can call.
Tool and action are the same item — ``tool_id`` is the single stable identifier.

``ToolResult`` and ``ToolError`` mirror the backend execution envelope
(``ai-service-contract.md`` §4.1–§4.2): every call returns either
``status="succeeded"`` with ``data`` or ``status="failed"`` with a structured
``ToolError`` (including ``connector_not_connected``).  Neither type performs I/O.

``effective_tools(P, S)`` (task 1.11) lives here too: the effective set is a
pure function of the sub-agent manifest ``P`` and the user's selection ``S``,
never an input field (§4.1 invariant 7).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ToolSpec(BaseModel, frozen=True):
    """Canonical system-catalog entry for a callable tool (§4.3).

    Attributes
    ----------
    `tool_id` : str
        Stable unique identifier used across the catalog, sub-agent manifests,
        policy, events, ``agent_config.tools``, and ``Connector.call``.
    `connector_id` : str
        Backend connector that serves this tool (e.g. ``"shopify"``).
    `description` : str
        Human- and LLM-readable summary of what the tool does.
    `parameters` : dict[str, Any]
        Input JSON Schema used for argument validation.
    `result_summary` : str
        Deterministic ``{field}`` template rendered from execution ``data``.
    `effect_summary` : str | None
        ``None`` for read-only tools (run live in preview); a deterministic
        ``{field}`` template for side-effecting tools (mocked in preview).
        Every call still carries an ``idem_key`` regardless of this field.
    `version` : int
        Per-tool schema/behavior version, emitted as ``tool_version`` on events.
    `deprecated` : bool
        ``True`` when retired from new use (ids are never reused).
    """

    tool_id: str = Field(min_length=1)
    connector_id: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    effect_summary: str | None = Field(default=None, min_length=1)
    version: int = Field(default=1, ge=1)
    deprecated: bool = False

    @property
    def is_side_effecting(self) -> bool:
        """Return ``True`` when the tool declares a non-null ``effect_summary``."""
        return self.effect_summary is not None


def effective_tools(
    predefined_tool_ids: Iterable[str],
    selected_tool_ids: Iterable[str],
) -> frozenset[str]:
    """Return the effective tool set ``E = P ∩ S`` (§4.1 invariant 7).

    ``P`` is the active sub-agent's predefined manifest, ``S`` the user's
    agent-level selection.  The function is pure — no I/O, no catalog lookup,
    no policy — and derives ``E`` every time instead of trusting an input
    field.  An id absent from either side can never be effective:

    * selecting a tool the sub-agent does not hold keeps it out of ``E``
      (the Tool Registry reports ``tool_unreachable`` separately);
    * dropping a tool from ``S`` revokes it immediately;
    * an id in neither set never appears.

    The result is a ``frozenset`` because ``E`` is a set: the contract is
    membership, ``E ⊆ P``, ``E ⊆ S``, monotonicity in ``S`` and idempotence.
    Callers that render tools sort the result by the catalog so display order
    stays deterministic without making order part of this function.
    """
    return frozenset(predefined_tool_ids).intersection(selected_tool_ids)


class ToolError(BaseModel, frozen=True):
    """Structured error returned on a failed tool execution (§4.2).

    ``code`` carries machine-readable failure codes such as
    ``connector_not_connected`` or ``provider_error`` so the executor can route
    to the right handoff path.
    """

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class ToolResult(BaseModel, frozen=True):
    """Structured execution outcome of one connector call (§4.1, §4.2).

    Mirrors the backend execution envelope:

    * ``status="succeeded"`` carries ``data`` and ``error=None``.
    * ``status="failed"`` carries a structured ``ToolError`` (including
      ``connector_not_connected`` for disconnected or unauthorized connectors)
      and ``data=None``.
    """

    status: Literal["succeeded", "failed"]
    data: dict[str, Any] | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def _check_status_invariants(self) -> ToolResult:
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("succeeded ToolResult must not carry an error")
        if self.status == "failed" and self.error is None:
            raise ValueError("failed ToolResult must carry a structured ToolError")
        if self.status == "failed" and self.data is not None:
            raise ValueError("failed ToolResult must not carry data")
        return self
