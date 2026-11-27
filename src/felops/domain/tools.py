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

``CatalogSnapshot`` (task 1.12) is the loader schema for the full catalog the
backend pushes as ``{"tools": [...]}``.  It intentionally carries no approval
flag and no verifier spec: approval is per (agent, tool) in ``agent_config``,
and verification is backend-owned (§4.1 inv. 5, §4.3).  The side-effect class
is derived from ``effect_summary`` (§4.3).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

_CATALOG_FIELDS: tuple[str, ...] = (
    "tool_id",
    "connector_id",
    "description",
    "parameters",
    "result_summary",
    "effect_summary",
    "version",
    "deprecated",
)


class SideEffectClass(StrEnum):
    """Whether a tool call can change external state (§4.3).

    Derived solely from ``effect_summary``: ``None`` is read-only and a
    deterministic template is side-effecting.  The class drives preview
    behaviour and the executor's retry policy; it never replaces the
    idempotency key, which every call carries.
    """

    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"


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
    def side_effect_class(self) -> SideEffectClass:
        """Return the class implied by ``effect_summary`` (§4.3)."""
        if self.effect_summary is None:
            return SideEffectClass.READ_ONLY
        return SideEffectClass.SIDE_EFFECTING

    @property
    def is_side_effecting(self) -> bool:
        """Return ``True`` when the tool declares a non-null ``effect_summary``."""
        return self.side_effect_class is SideEffectClass.SIDE_EFFECTING


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
        return self


class CatalogSnapshot(BaseModel, frozen=True):
    """Full system-catalog snapshot as pushed by the backend (§4.3, §7).

    The wire payload is exactly ``{"tools": [...]}``.  Extra keys — fixture
    metadata such as ``examples`` or a forward-compatible field from a newer
    backend — are ignored at both levels; unknown *tool* fields are ignored
    too, so a backend rollout cannot break the AI service.

    The snapshot adds the two invariants the domain owns:

    * ``tools`` is required, so a malformed push can never silently wipe the
      catalog;
    * ``tool_id`` is unique, because ids are stable primary keys for policy,
      events and ``agent_config`` (§4.1 inv. 8).

    It deliberately carries no approval flag and no verifier spec: approval is
    per (agent, tool) in ``agent_config``, and verification is backend-owned
    (§4.1 inv. 5, §4.3).  The side-effect class comes from ``effect_summary``.
    """

    tools: tuple[ToolSpec, ...]

    @model_validator(mode="before")
    @classmethod
    def _require_full_entries(cls, data: Any) -> Any:
        """Reject partial entries before defaults can hide a broken push.

        ``ToolSpec`` has defaults for connector-side discovery, but the
        backend's catalog contract (§4.3) is the full eight-field entry.  A
        summary payload from ``GET /catalog?full=false``, or an entry missing
        ``effect_summary``, would otherwise default to read-only and fail open
        in preview.
        """
        if not isinstance(data, Mapping):
            return data
        tools = data.get("tools")
        if not isinstance(tools, (list, tuple)):
            return data
        for index, tool in enumerate(tools):
            if not isinstance(tool, Mapping):
                continue
            missing = [field for field in _CATALOG_FIELDS if field not in tool]
            if missing:
                raise ValueError(
                    f"catalog entry {index} is missing required field(s): "
                    f"{', '.join(missing)}"
                )
        return data

    @model_validator(mode="after")
    def _check_unique_tool_ids(self) -> CatalogSnapshot:
        seen: set[str] = set()
        for tool in self.tools:
            if tool.tool_id in seen:
                raise ValueError(f"duplicate tool_id in catalog: {tool.tool_id}")
            seen.add(tool.tool_id)
        return self

    def get(self, tool_id: str) -> ToolSpec | None:
        """Return the catalog entry for *tool_id*, or ``None`` when absent."""
        return next((tool for tool in self.tools if tool.tool_id == tool_id), None)


class ToolOutcome(BaseModel, frozen=True):
    """Result of one executor run, including retry metadata (§4 Executor).

    ``side_effect_class`` makes the read/write split explicit for callers and
    events; ``attempts`` records how many connector calls were made with the
    same ``idem_key`` (0 for a pre-flight rejection or a ledger replay);
    ``deduplicated`` marks a replay served from the idempotency ledger.
    """

    call_id: str
    tool_id: str
    side_effect_class: SideEffectClass
    attempts: int = Field(ge=0)
    deduplicated: bool = False
    result: ToolResult
