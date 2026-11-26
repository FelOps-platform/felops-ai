"""Shared recursive immutable container helpers for domain snapshots."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any


def _immutable_error(*_args: Any, **_kwargs: Any) -> Any:
    raise TypeError("frozen container is immutable")


class FrozenDict(dict[str, Any]):
    """A picklable immutable dict used for nested domain snapshots."""

    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    clear = _immutable_error
    pop = _immutable_error
    popitem = _immutable_error
    setdefault = _immutable_error
    update = _immutable_error
    __ior__ = _immutable_error

    def __reduce__(self) -> tuple[type[FrozenDict], tuple[list[tuple[str, Any]]]]:
        return (FrozenDict, (list(self.items()),))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenDict:
        copied = FrozenDict(
            (copy.deepcopy(k, memo), copy.deepcopy(v, memo)) for k, v in self.items()
        )
        memo[id(self)] = copied
        return copied


class FrozenList(list[Any]):
    """A picklable immutable list used for nested domain snapshots."""

    __setitem__ = _immutable_error
    __delitem__ = _immutable_error
    append = _immutable_error
    clear = _immutable_error
    extend = _immutable_error
    insert = _immutable_error
    pop = _immutable_error
    remove = _immutable_error
    reverse = _immutable_error
    sort = _immutable_error
    __iadd__ = _immutable_error
    __imul__ = _immutable_error

    def __reduce__(self) -> tuple[type[FrozenList], tuple[list[Any]]]:
        return (FrozenList, (list(self),))

    def __deepcopy__(self, memo: dict[int, Any]) -> FrozenList:
        copied = FrozenList(copy.deepcopy(item, memo) for item in self)
        memo[id(self)] = copied
        return copied


def _freeze_mapping(value: Mapping[Any, Any]) -> FrozenDict:
    return FrozenDict((key, freeze_value(item)) for key, item in value.items())


def _freeze_collection(factory: Callable[[Any], Any], value: Any) -> Any:
    return factory(freeze_value(item) for item in value)


_CONTAINER_FREEZERS: tuple[tuple[type[Any] | tuple[type[Any], ...], Callable[[Any], Any]], ...] = (
    (Mapping, _freeze_mapping),
    (list, partial(_freeze_collection, FrozenList)),
    (tuple, partial(_freeze_collection, tuple)),
    ((set, frozenset), partial(_freeze_collection, frozenset)),
)


def freeze_value(value: Any) -> Any:
    """Deep-copy a value while recursively freezing its containers."""
    for container_type, freezer in _CONTAINER_FREEZERS:
        if isinstance(value, container_type):
            return freezer(value)
    return copy.deepcopy(value)
