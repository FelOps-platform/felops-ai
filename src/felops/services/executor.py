"""Executor v0 (task 2.2, §4 Executor).

Runs an authorised call through the ``Connector`` port, emits the
``tool.started`` / ``tool.completed`` events and retries transient failures with
simple exponential backoff.

Retry policy is selected by the tool's ``side_effect_class``.  Both reads and
writes carry the same ``idem_key`` on every attempt (the finalized contract
requires a key for every call), so a retry cannot duplicate a write: the
backend's ``UNIQUE (workspace_id, idempotency_key)`` suppresses it.  The split
lets callers tune reads and writes independently without changing the call
path.

Pre-flight failures (unknown tool, blank idempotency key) never reach the
connector and emit no ``tool.*`` events; they are returned as structured
``ToolResult`` failures like any other execution error.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from felops.domain.events import EventType
from felops.domain.state import ToolCall
from felops.domain.tools import (
    SideEffectClass,
    ToolError,
    ToolOutcome,
    ToolResult,
    ToolSpec,
)
from felops.domain.workspace import WorkspaceContext
from felops.ports.connector import Connector
from felops.ports.ledger import IdempotencyLedger
from felops.services.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

EventEmitter = Callable[[EventType, dict[str, Any]], None]
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential backoff for one side-effect class."""

    max_attempts: int = 3
    base_delay: float = 0.05
    max_delay: float = 1.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("retry delays must not be negative")

    def delay_after(self, attempt: int) -> float:
        """Delay after failed *attempt* (1-based), capped at ``max_delay``."""
        return min(self.base_delay * 2 ** (attempt - 1), self.max_delay)


@dataclass(frozen=True)
class ExecutorOptions:
    """Runtime options for one executor instance.

    Grouped so the constructor stays small and callers pass only what they
    change: per-class retry policies, an event sink and an async sleeper.
    """

    retry: Mapping[SideEffectClass, RetryPolicy] | None = None
    emit: EventEmitter | None = None
    sleep: Sleeper = asyncio.sleep
    ledger: IdempotencyLedger | None = None


class Executor:
    """Executes authorised calls and reports ``tool.*`` events."""

    def __init__(
        self,
        registry: ToolRegistry,
        connector: Connector,
        options: ExecutorOptions | None = None,
    ) -> None:
        options = options or ExecutorOptions()
        self._registry = registry
        self._connector = connector
        self._retry = {side_effect: RetryPolicy() for side_effect in SideEffectClass}
        if options.retry is not None:
            self._retry.update(options.retry)
        self._emit = options.emit
        self._sleep = options.sleep
        self._ledger = options.ledger

    async def execute(self, call: ToolCall, ctx: WorkspaceContext) -> ToolOutcome:
        """Run *call* and return the outcome with attempt metadata."""
        spec = self._registry.get(call.tool_id)
        if spec is None:
            return _preflight_failure(
                call,
                SideEffectClass.READ_ONLY,
                "unknown_tool",
                f"unknown tool: {call.tool_id}",
            )
        side_effect = spec.side_effect_class
        if spec.deprecated:
            return _preflight_failure(
                call,
                side_effect,
                "deprecated_tool",
                f"deprecated tool: {call.tool_id}",
            )
        if not call.idem_key.strip():
            return _preflight_failure(
                call,
                side_effect,
                "invalid_idem_key",
                "idem_key is required for every tool call",
            )

        cached = await self._lookup(call, ctx, side_effect)
        if cached is not None:
            return _replay(call, side_effect, cached)

        self._emit_started(call, spec)
        result, attempts = await self._call_with_retry(call, ctx, spec, side_effect)
        call.outcome = result.model_dump(mode="json")
        self._emit_completed(call, spec, result)
        await self._store(call, ctx, side_effect, result)
        return ToolOutcome(
            call_id=call.call_id,
            tool_id=call.tool_id,
            side_effect_class=side_effect,
            attempts=attempts,
            result=result,
        )

    async def _call_with_retry(
        self,
        call: ToolCall,
        ctx: WorkspaceContext,
        spec: ToolSpec,
        side_effect: SideEffectClass,
    ) -> tuple[ToolResult, int]:
        policy = self._retry[side_effect]
        attempts = 0
        while True:
            attempts += 1
            result = await self._connector.call(
                call.tool_id, call.arguments, ctx, call.idem_key
            )
            if not _should_retry(result, attempts, policy):
                return result, attempts
            await self._sleep(policy.delay_after(attempts))

    async def _lookup(
        self, call: ToolCall, ctx: WorkspaceContext, side_effect: SideEffectClass
    ) -> ToolResult | None:
        """Return a ledgered terminal result for a side-effecting retry."""
        if self._ledger is None or side_effect is not SideEffectClass.SIDE_EFFECTING:
            return None
        try:
            return await self._ledger.get(ctx.workspace_id, call.idem_key)
        except Exception as exc:  # the ledger is an optimisation, not a gate
            logger.warning(
                "idempotency ledger lookup failed for workspace %s: %s",
                ctx.workspace_id,
                exc,
            )
            return None

    async def _store(
        self,
        call: ToolCall,
        ctx: WorkspaceContext,
        side_effect: SideEffectClass,
        result: ToolResult,
    ) -> None:
        if self._ledger is None or side_effect is not SideEffectClass.SIDE_EFFECTING:
            return
        if not _is_terminal(result):
            # A retryable failure is not terminal: caching it would poison the
            # key and block the redelivery that could still succeed.
            return
        try:
            await self._ledger.put(ctx.workspace_id, call.idem_key, result)
        except Exception as exc:  # the backend still dedupes by idempotency key
            logger.warning(
                "idempotency ledger store failed for workspace %s: %s",
                ctx.workspace_id,
                exc,
            )

    # -- events -------------------------------------------------------------

    def _emit_started(self, call: ToolCall, spec: ToolSpec) -> None:
        self._emit_event(
            EventType.TOOL_STARTED,
            {
                "call_id": call.call_id,
                "tool_id": call.tool_id,
                "tool_version": spec.version,
                "action": spec.tool_id,
                "integration": spec.connector_id,
                "params_summary": _params_summary(spec, call.arguments),
            },
        )

    def _emit_completed(self, call: ToolCall, spec: ToolSpec, result: ToolResult) -> None:
        self._emit_event(
            EventType.TOOL_COMPLETED,
            {
                "call_id": call.call_id,
                "status": result.status,
                "result_summary": _summary(spec, result),
                "initiated_by": "auto",
            },
        )

    def _emit_event(self, event_type: EventType, data: dict[str, Any]) -> None:
        if self._emit is not None:
            self._emit(event_type, data)


def _is_terminal(result: ToolResult) -> bool:
    """Return ``True`` when a result must never be retried again."""
    if result.status == "succeeded":
        return True
    return result.error is not None and not result.error.retryable


def _replay(
    call: ToolCall, side_effect: SideEffectClass, result: ToolResult
) -> ToolOutcome:
    call.outcome = result.model_dump(mode="json")
    return ToolOutcome(
        call_id=call.call_id,
        tool_id=call.tool_id,
        side_effect_class=side_effect,
        attempts=0,
        deduplicated=True,
        result=result,
    )


def _should_retry(result: ToolResult, attempts: int, policy: RetryPolicy) -> bool:
    return (
        result.status == "failed"
        and result.error is not None
        and result.error.retryable
        and attempts < policy.max_attempts
    )


def _preflight_failure(
    call: ToolCall,
    side_effect: SideEffectClass,
    code: str,
    message: str,
) -> ToolOutcome:
    result = ToolResult(
        status="failed",
        data=None,
        error=ToolError(code=code, message=message, retryable=False),
    )
    call.outcome = result.model_dump(mode="json")
    return ToolOutcome(
        call_id=call.call_id,
        tool_id=call.tool_id,
        side_effect_class=side_effect,
        attempts=0,
        result=result,
    )


_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _render(template: str | None, values: Mapping[str, Any]) -> str:
    """Substitute flat ``{field}`` placeholders and leave everything else.

    Templates are backend-owned strings, so dotted paths, indexing and format
    specs are not evaluated: they stay literal instead of reaching into Python
    objects or raising after a side-effecting call has already run.
    """
    if not template:
        return ""
    return _PLACEHOLDER.sub(
        lambda match: (
            str(values[match.group(1)])
            if match.group(1) in values
            else match.group(0)
        ),
        template,
    )


def _params_summary(spec: ToolSpec, arguments: Mapping[str, Any]) -> str:
    """Render the effect template, or list read-only arguments deterministically."""
    if spec.effect_summary:
        return _render(spec.effect_summary, arguments)
    return ", ".join(f"{key}={value}" for key, value in sorted(arguments.items()))


def _summary(spec: ToolSpec, result: ToolResult) -> str:
    if result.status == "succeeded":
        return _render(spec.result_summary, result.data or {})
    return result.error.message if result.error is not None else ""


__all__ = ["EventEmitter", "Executor", "ExecutorOptions", "RetryPolicy"]
