"""Property tests for ``effective_tools`` (task 1.11, §4.1 invariant 7).

``E_s = P_s ∩ S``.  The properties are checked exhaustively over the power set
of a small id universe instead of by random sampling: every possible
predefined set and every possible user selection is exercised, deterministically
and without adding a test dependency.  The §4.1 illustrative example and the
real catalog ids are covered separately.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import pytest

from felops.domain.tools import effective_tools

_UNIVERSE: tuple[str, ...] = tuple(f"tool_{index}" for index in range(8))
_SUBSETS: tuple[frozenset[str], ...] = tuple(
    frozenset(combo)
    for size in range(len(_UNIVERSE) + 1)
    for combo in combinations(_UNIVERSE, size)
)


def _extensions(subset: frozenset[str]) -> Iterable[frozenset[str]]:
    """Yield every selection obtained by adding one id to *subset*."""
    for tool_id in _UNIVERSE:
        if tool_id not in subset:
            yield subset | {tool_id}


# ---------------------------------------------------------------------------
# E ⊆ P and E ⊆ S (the issue's done-when)
# ---------------------------------------------------------------------------


def test_effective_tools_is_the_intersection_of_both_inputs() -> None:
    for predefined in _SUBSETS:
        for selected in _SUBSETS:
            result = effective_tools(predefined, selected)

            assert result == predefined & selected
            assert result <= predefined
            assert result <= selected


# ---------------------------------------------------------------------------
# Monotonicity in S and idempotence (§4.1 invariant 7)
# ---------------------------------------------------------------------------


def test_effective_tools_is_monotonic_in_selection() -> None:
    for predefined in _SUBSETS:
        for selected in _SUBSETS:
            smaller = effective_tools(predefined, selected)
            for larger_selection in _extensions(selected):
                assert smaller <= effective_tools(predefined, larger_selection)


def test_effective_tools_is_idempotent() -> None:
    for predefined in _SUBSETS:
        for selected in _SUBSETS:
            once = effective_tools(predefined, selected)

            assert effective_tools(predefined, once) == once
            assert effective_tools(once, selected) == once


# ---------------------------------------------------------------------------
# §4.1 illustrative example (P(customer_support) = [0, 2, 4, 6])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selected", "expected"),
    [
        ({"0", "2", "4", "5"}, {"0", "2", "4"}),
        ({"0", "2", "5"}, {"0", "2"}),
        ({"2", "5"}, {"2"}),
        ({"0", "1", "2", "3", "4", "5", "6", "7"}, {"0", "2", "4", "6"}),
        (set(), set()),
    ],
)
def test_design_table_examples(selected: set[str], expected: set[str]) -> None:
    predefined = ["0", "2", "4", "6"]

    assert effective_tools(predefined, selected) == expected


def test_tool_selected_outside_the_predefined_set_is_dropped() -> None:
    """Tool 7 is in no predefined set, so no sub-agent can use it (§4.1)."""
    predefined = ["0", "2", "4", "6"]

    assert "7" not in effective_tools(predefined, ["0", "2", "4", "5", "7"])


def test_real_catalog_ids() -> None:
    predefined = ["lookup_order", "refund_order", "send_invoice", "get_case"]
    selected = ["lookup_order", "refund_order", "send_invoice", "get_invoice", "create_case"]

    assert effective_tools(predefined, selected) == {
        "lookup_order",
        "refund_order",
        "send_invoice",
    }


# ---------------------------------------------------------------------------
# Purity and input shape
# ---------------------------------------------------------------------------


def test_inputs_are_order_and_duplicate_insensitive() -> None:
    assert effective_tools(["a", "a", "b"], ["b", "c", "b"]) == frozenset({"b"})
    assert effective_tools(["b", "a"], ["c", "b"]) == effective_tools(["a", "b"], ["b", "c"])


def test_empty_inputs_yield_an_empty_set() -> None:
    assert effective_tools([], ["a"]) == frozenset()
    assert effective_tools(["a"], []) == frozenset()
    assert effective_tools([], []) == frozenset()


def test_result_is_immutable_and_does_not_alias_inputs() -> None:
    predefined = ["a", "b"]
    selected = {"b"}
    result = effective_tools(predefined, selected)

    predefined.append("c")
    selected.add("c")

    assert result == frozenset({"b"})
    with pytest.raises(AttributeError):
        result.add("c")  # type: ignore[attr-defined]
