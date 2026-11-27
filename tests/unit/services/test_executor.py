"""Tests for Executor v0 (task 2.2, §4 Executor).

Covers the connector round trip, ``tool.*`` event payloads, retry/backoff
behaviour, the per-side-effect-class retry split and the fail-closed pre-flight
paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from felops.domain.events import EventType
from felops.domain.state import ToolCall
from felops.domain.tools import CatalogSnapshot, SideEffectClass, ToolError, ToolResult
from felops.domain.workspace import WorkspaceContext
from felops.services.executor import Executor, ExecutorOptions, RetryPolicy, _render
from felops.services.tool_registry import ToolRegistry, load_subagent_specs

_ROOT = Path(__file__).resolve().parents[3]
_CATALOG_PATH = _ROOT / "fixtures" / "catalog" / "catalog.json"
_MANIFEST_DIR = _ROOT / "config" / "subagents"
_CTX = WorkspaceContext(workspace_id="ws-1")


class ScriptedConnector:
    """Connector double that returns scripted results and records calls."""

    def __init__(self, results: list[ToolResult]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    async def list_tools(self, ctx: WorkspaceContext) -> list[Any]:
        return []

    async def call(
        self,
        tool: str,
        args: dict[str, Any],
        ctx: WorkspaceContext,
        idem_key: str,
    ) -> ToolResult:
        self.calls.append(
            {"tool": tool, "args": args, "ctx": ctx, "idem_key": idem_key}
        )
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]


def _registry() -> ToolRegistry:
    catalog = CatalogSnapshot.model_validate(
        json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    )
    return ToolRegistry(catalog, load_subagent_specs(_MANIFEST_DIR))


def _call(tool_id: str, **arguments: Any) -> ToolCall:
    return ToolCall(tool_id=tool_id, arguments=arguments, idem_key="idem-1")


def _success(data: dict[str, Any]) -> ToolResult:
    return ToolResult(status="succeeded", data=data, error=None)


def _failure(*, retryable: bool = True, message: str = "provider error") -> ToolResult:
    return ToolResult(
        status="failed",
        data=None,
        error=ToolError(code="provider_error", message=message, retryable=retryable),
    )


async def _noop_sleep(delay: float) -> None:
    return None


def _registry_with(tool_id: str, **overrides: Any) -> ToolRegistry:
    payload = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    for tool in payload["tools"]:
        if tool["tool_id"] == tool_id:
            tool.update(overrides)
    return ToolRegistry(
        CatalogSnapshot.model_validate(payload), load_subagent_specs(_MANIFEST_DIR)
    )


def _options(
    *,
    retry: dict[SideEffectClass, RetryPolicy] | None = None,
    emit: Any = None,
    sleep: Any = None,
) -> ExecutorOptions:
    return ExecutorOptions(retry=retry, emit=emit, sleep=sleep or _noop_sleep)


def _executor(
    connector: ScriptedConnector,
    *,
    registry: ToolRegistry | None = None,
    options: ExecutorOptions | None = None,
) -> Executor:
    return Executor(registry or _registry(), connector, options or _options())


# ---------------------------------------------------------------------------
# The lookup runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_runs_through_the_connector() -> None:
    connector = ScriptedConnector(
        [
            _success(
                {
                    "order_number": "#1042",
                    "status": "shipped",
                    "tracking_number": "TRK8812",
                }
            )
        ]
    )
    call = _call("lookup_order", order_number="#1042")

    outcome = await _executor(connector).execute(call, _CTX)

    assert outcome.result.status == "succeeded"
    assert outcome.result.data is not None
    assert outcome.result.data["tracking_number"] == "TRK8812"
    assert outcome.side_effect_class is SideEffectClass.READ_ONLY
    assert outcome.attempts == 1
    assert connector.calls == [
        {
            "tool": "lookup_order",
            "args": {"order_number": "#1042"},
            "ctx": _CTX,
            "idem_key": "idem-1",
        }
    ]
    assert call.outcome == outcome.result.model_dump(mode="json")


@pytest.mark.asyncio
async def test_emits_tool_started_and_completed() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector(
        [
            _success(
                {
                    "refund_id": "RFD-3081",
                    "order_number": "#1042",
                    "amount": 120.0,
                    "status": "completed",
                    "currency": "AED",
                }
            )
        ]
    )
    call = _call("refund_order", order_number="#1042", amount=120.0, reason="damaged_on_arrival")

    await _executor(
        connector, options=_options(emit=lambda kind, data: events.append((kind, data)))
    ).execute(
        call, _CTX
    )

    assert [kind for kind, _ in events] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
    ]
    started = events[0][1]
    assert started == {
        "call_id": call.call_id,
        "tool_id": "refund_order",
        "tool_version": 1,
        "action": "refund_order",
        "integration": "shopify",
        "params_summary": (
            "Refund AED 120.0 on order #1042 to the original payment method"
        ),
    }
    completed = events[1][1]
    assert completed == {
        "call_id": call.call_id,
        "status": "succeeded",
        "result_summary": "Refund RFD-3081 · AED 120.0 · completed",
        "initiated_by": "auto",
    }


@pytest.mark.asyncio
async def test_emits_read_only_events_with_arguments() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector(
        [_success({"order_number": "#1042", "status": "shipped"})]
    )
    call = _call("lookup_order", order_number="#1042")

    await _executor(
        connector, options=_options(emit=lambda kind, data: events.append((kind, data)))
    ).execute(call, _CTX)

    assert events[0][1]["params_summary"] == "order_number=#1042"
    assert events[1][1]["result_summary"] == "Order #1042 · shipped"


@pytest.mark.asyncio
async def test_emits_once_per_execution_across_retries() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector([_failure(), _success({"order_number": "#1042"})])

    await _executor(
        connector, options=_options(emit=lambda kind, data: events.append((kind, data)))
    ).execute(_call("lookup_order", order_number="#1042"), _CTX)

    assert [kind for kind, _ in events] == [
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_failed_execution_emits_failed_completion() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector(
        [_failure(retryable=False, message="provider down")]
    )

    await _executor(
        connector, options=_options(emit=lambda kind, data: events.append((kind, data)))
    ).execute(_call("lookup_order"), _CTX)

    assert events[-1][0] is EventType.TOOL_COMPLETED
    assert events[-1][1]["status"] == "failed"
    assert events[-1][1]["result_summary"] == "provider down"


# ---------------------------------------------------------------------------
# Retry and backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retries_retryable_failures_with_the_same_idem_key() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    connector = ScriptedConnector([_failure(), _success({"ok": True})])
    call = _call("lookup_order", order_number="#1042")

    outcome = await _executor(connector, options=_options(sleep=record_sleep)).execute(
        call, _CTX
    )

    assert outcome.result.status == "succeeded"
    assert outcome.attempts == 2
    assert [record["idem_key"] for record in connector.calls] == ["idem-1", "idem-1"]
    assert delays == [0.05]


@pytest.mark.asyncio
async def test_non_retryable_failure_is_not_retried() -> None:
    connector = ScriptedConnector([_failure(retryable=False)])

    outcome = await _executor(connector).execute(_call("lookup_order"), _CTX)

    assert outcome.attempts == 1
    assert len(connector.calls) == 1
    assert outcome.result.status == "failed"


@pytest.mark.asyncio
async def test_exhausted_retries_return_the_last_failure() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    connector = ScriptedConnector([_failure(), _failure(), _failure(message="final")])

    outcome = await _executor(connector, options=_options(sleep=record_sleep)).execute(
        _call("lookup_order"), _CTX
    )

    assert outcome.attempts == 3
    assert outcome.result.status == "failed"
    assert outcome.result.error is not None
    assert outcome.result.error.message == "final"
    assert delays == [0.05, 0.1]


@pytest.mark.asyncio
async def test_backoff_is_capped() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    policy = RetryPolicy(max_attempts=4, base_delay=0.4, max_delay=0.5)
    connector = ScriptedConnector([_failure(), _failure(), _failure(), _failure()])

    await _executor(
        connector,
        options=_options(
            retry={side_effect: policy for side_effect in SideEffectClass},
            sleep=record_sleep,
        ),
    ).execute(_call("lookup_order"), _CTX)

    assert delays == [0.4, 0.5, 0.5]


@pytest.mark.asyncio
async def test_retry_policy_is_selected_by_side_effect_class() -> None:
    retry = {
        SideEffectClass.READ_ONLY: RetryPolicy(max_attempts=3),
        SideEffectClass.SIDE_EFFECTING: RetryPolicy(max_attempts=1),
    }

    read_connector = ScriptedConnector([_failure(), _failure(), _failure()])
    write_connector = ScriptedConnector([_failure(), _failure()])
    executor = _executor(read_connector, options=_options(retry=retry))
    write_executor = _executor(write_connector, options=_options(retry=retry))

    read_outcome = await executor.execute(_call("lookup_order"), _CTX)
    write_outcome = await write_executor.execute(_call("refund_order"), _CTX)

    assert read_outcome.side_effect_class is SideEffectClass.READ_ONLY
    assert read_outcome.attempts == 3
    assert write_outcome.side_effect_class is SideEffectClass.SIDE_EFFECTING
    assert write_outcome.attempts == 1


# ---------------------------------------------------------------------------
# Fail-closed pre-flight paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blank_idem_key_fails_closed() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector([_success({})])
    call = ToolCall(tool_id="lookup_order", idem_key="  ")

    outcome = await _executor(
        connector, options=_options(emit=lambda kind, data: events.append((kind, data)))
    ).execute(call, _CTX)

    assert outcome.result.status == "failed"
    assert outcome.result.error is not None
    assert outcome.result.error.code == "invalid_idem_key"
    assert outcome.attempts == 0
    assert call.outcome is not None
    assert connector.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_unknown_tool_fails_closed() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector([_success({})])
    call = _call("delete_everything")

    outcome = await _executor(
        connector, options=_options(emit=lambda kind, data: events.append((kind, data)))
    ).execute(call, _CTX)

    assert outcome.result.status == "failed"
    assert outcome.result.error is not None
    assert outcome.result.error.code == "unknown_tool"
    assert outcome.attempts == 0
    assert call.outcome is not None
    assert connector.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_empty_tool_id_fails_closed_without_a_schema_error() -> None:
    connector = ScriptedConnector([_success({})])
    call = ToolCall(tool_id="", idem_key="idem-1")

    outcome = await _executor(connector).execute(call, _CTX)

    assert outcome.result.status == "failed"
    assert outcome.result.error is not None
    assert outcome.result.error.code == "unknown_tool"
    assert outcome.attempts == 0
    assert connector.calls == []


@pytest.mark.asyncio
async def test_deprecated_tool_is_rejected_before_the_connector() -> None:
    events: list[tuple[EventType, dict[str, Any]]] = []
    connector = ScriptedConnector([_success({})])
    call = _call("lookup_order", order_number="#1042")

    outcome = await _executor(
        connector,
        registry=_registry_with("lookup_order", deprecated=True),
        options=_options(emit=lambda kind, data: events.append((kind, data))),
    ).execute(call, _CTX)

    assert outcome.result.status == "failed"
    assert outcome.result.error is not None
    assert outcome.result.error.code == "deprecated_tool"
    assert outcome.attempts == 0
    assert connector.calls == []
    assert events == []


@pytest.mark.asyncio
async def test_side_effecting_retries_reuse_the_idem_key() -> None:
    connector = ScriptedConnector(
        [
            _failure(),
            _success({"refund_id": "RFD-3081", "amount": 120.0, "status": "completed"}),
        ]
    )

    outcome = await _executor(connector).execute(
        _call("refund_order", order_number="#1042", amount=120.0, reason="defective"),
        _CTX,
    )

    assert outcome.attempts == 2
    assert [record["idem_key"] for record in connector.calls] == ["idem-1", "idem-1"]


@pytest.mark.asyncio
async def test_partial_retry_override_keeps_defaults_for_other_classes() -> None:
    retry = {SideEffectClass.SIDE_EFFECTING: RetryPolicy(max_attempts=1)}

    read_connector = ScriptedConnector([_failure(), _failure(), _failure()])
    write_connector = ScriptedConnector([_failure(), _failure()])

    read_outcome = await _executor(read_connector, options=_options(retry=retry)).execute(
        _call("lookup_order"), _CTX
    )
    write_outcome = await _executor(write_connector, options=_options(retry=retry)).execute(
        _call("refund_order"), _CTX
    )

    assert read_outcome.attempts == 3
    assert write_outcome.attempts == 1


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "values", "expected"),
    [
        ("Order {order_number} · {status}", {"order_number": "#1042"}, "Order #1042 · {status}"),
        ("Refund AED {amount}", {"amount": 120.0}, "Refund AED 120.0"),
        ("{missing}", {}, "{missing}"),
        ("{order.id}", {"order": {"id": 1}}, "{order.id}"),
        ("{items[0]}", {}, "{items[0]}"),
        ("{amount:>5}", {"amount": 1}, "{amount:>5}"),
        (None, {}, ""),
    ],
)
def test_render_only_substitutes_flat_placeholders(
    template: str | None, values: dict[str, Any], expected: str
) -> None:
    assert _render(template, values) == expected
