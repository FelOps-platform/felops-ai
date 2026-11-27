"""Tool Registry v0 (task 2.1, §4.1).

The registry is the single owner of the system catalog and of each sub-agent's
manifest ``P_s``.  Every consumer (planner, ``tool_selector``, policy, executor)
obtains tools only through ``effective_for(agent_ref, agent_config)``; the
effective set is derived as ``E_s = P_s ∩ S`` (via the pure ``effective_tools``
function) and then narrowed to tools that still exist in the catalog and are
not deprecated (§4.1 invariants 1, 4 and 6).

Warnings such as ``unknown_tool`` and ``tool_removed`` are a
``config:validate`` concern (4.10) and are not emitted here yet; the registry
simply keeps those ids out of ``E``.  Argument validation uses the tool's
``parameters`` JSON Schema (Draft 2020-12).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from felops.domain.agent import AgentRef, SubAgentSpec
from felops.domain.state import AgentConfig, ToolCall
from felops.domain.tools import CatalogSnapshot, ToolSpec, effective_tools


def load_subagent_specs(manifest_dir: Path) -> dict[str, SubAgentSpec]:
    """Load every ``*.json`` sub-agent manifest in *manifest_dir*.

    Manifests are versioned with the code and never come from the backend
    (§4.1 inv. 3).  A missing or empty directory is a deployment error: booting
    with zero sub-agents would silently deny every tool call.  A duplicate
    ``sub_agent_id`` is a deployment error too.
    """
    if not manifest_dir.exists():
        raise FileNotFoundError(f"sub-agent manifest directory not found: {manifest_dir}")
    if not manifest_dir.is_dir():
        raise NotADirectoryError(
            f"sub-agent manifest path is not a directory: {manifest_dir}"
        )

    specs: dict[str, SubAgentSpec] = {}
    for path in sorted(manifest_dir.glob("*.json")):
        if not path.is_file():
            continue
        spec = SubAgentSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))
        if spec.sub_agent_id in specs:
            raise ValueError(f"duplicate sub_agent_id in manifests: {spec.sub_agent_id}")
        specs[spec.sub_agent_id] = spec
    if not specs:
        raise ValueError(f"no sub-agent manifests found in {manifest_dir}")
    return specs


def _validate_schemas(catalog: CatalogSnapshot) -> None:
    """Fail fast when a catalog entry is not a valid JSON Schema."""
    for tool in catalog.tools:
        try:
            Draft202012Validator.check_schema(tool.parameters)
        except SchemaError as exc:
            raise ValueError(
                f"invalid JSON Schema for tool {tool.tool_id}: {exc.message}"
            ) from exc


def _error_sort_key(error: ValidationError) -> tuple[tuple[str, ...], str]:
    return tuple(str(part) for part in error.absolute_path), error.message


def _format_error(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return f"{path}: {error.message}" if path else error.message


class ToolRegistry:
    """Read-only catalog copy plus per-sub-agent effective tool sets (§4.1)."""

    def __init__(
        self,
        catalog: CatalogSnapshot,
        subagents: Mapping[str, SubAgentSpec] | None = None,
    ) -> None:
        _validate_schemas(catalog)
        self._catalog = catalog
        self._subagents = dict(subagents or {})

    @classmethod
    def from_files(cls, catalog_path: Path, manifest_dir: Path) -> ToolRegistry:
        """Build a registry from the catalog snapshot and manifest files."""
        catalog = CatalogSnapshot.model_validate(
            json.loads(catalog_path.read_text(encoding="utf-8"))
        )
        return cls(catalog, load_subagent_specs(manifest_dir))

    @property
    def catalog(self) -> CatalogSnapshot:
        return self._catalog

    def apply_catalog(self, payload: Mapping[str, object]) -> None:
        """Replace the local catalog copy with a full backend snapshot (§4.3).

        An invalid snapshot raises and leaves the current copy untouched, so a
        bad push cannot corrupt the catalog.
        """
        catalog = CatalogSnapshot.model_validate(payload)
        _validate_schemas(catalog)
        self._catalog = catalog

    def get(self, tool_id: str) -> ToolSpec | None:
        """Return the catalog entry for *tool_id*, or ``None`` when absent."""
        return self._catalog.get(tool_id)

    def effective_for(
        self, agent_ref: AgentRef, agent_config: AgentConfig
    ) -> frozenset[str]:
        """Return ``E_s = P_s ∩ S`` for the sub-agent, minus unusable tools.

        Tools that are missing from the catalog or deprecated are dropped
        (§4.1 inv. 4); the caller may report them as config warnings.
        """
        subagent = self._subagents.get(agent_ref.sub_agent_id)
        if subagent is None or subagent.audience != agent_ref.audience:
            return frozenset()
        selected = {tool.tool_id for tool in agent_config.tools}
        candidates = effective_tools(subagent.predefined_tool_ids, selected)
        return frozenset(tool_id for tool_id in candidates if self._is_usable(tool_id))

    def _is_usable(self, tool_id: str) -> bool:
        spec = self._catalog.get(tool_id)
        return spec is not None and not spec.deprecated

    def validate_args(self, call: ToolCall) -> list[str]:
        """Validate *call* arguments against the tool's ``parameters`` schema.

        Returns a deterministic list of human-readable violations; an empty
        list means the call is valid.  An unknown tool id is itself a violation.
        """
        spec = self._catalog.get(call.tool_id)
        if spec is None:
            return [f"unknown tool: {call.tool_id}"]
        if spec.deprecated:
            return [f"deprecated tool: {call.tool_id}"]
        validator = Draft202012Validator(
            spec.parameters, format_checker=Draft202012Validator.FORMAT_CHECKER
        )
        errors = sorted(validator.iter_errors(call.arguments), key=_error_sort_key)
        return [_format_error(error) for error in errors]


__all__ = ["ToolRegistry", "load_subagent_specs"]
