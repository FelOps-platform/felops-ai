"""Memory value objects and scoping (§2.3, §4, §5.1).

Memory is workspace-scoped (§2.3, §4, task 7.9).  A ``MemoryScope`` is the
partition key for reads and writes:

* ``agent_id=None`` (the default) addresses workspace-wide memory shared across
  all agents in the workspace.  Setting ``agent_id`` narrows writes to a single
  agent while keeping a non-breaking path for agent-specific memory.
* ``thread_id=None`` addresses long-term memory (customer facts, P7).  Setting
  ``thread_id`` addresses short-term thread memory (P4).  Callers that need
  both long-term and thread memory compose two reads.
"""

from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, ModelWrapValidatorHandler, field_validator, model_validator

_MEMORY_ITEM_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "felops.memory.item")


def _immutable_metadata_error(*_args: Any, **_kwargs: Any) -> Any:
    raise TypeError("MemoryItem.metadata is immutable")


class _FrozenDict(dict[str, Any]):
    """Immutable dict subclass for ``MemoryItem.metadata``."""

    __setitem__ = _immutable_metadata_error
    __delitem__ = _immutable_metadata_error
    clear = _immutable_metadata_error
    pop = _immutable_metadata_error
    popitem = _immutable_metadata_error
    setdefault = _immutable_metadata_error
    update = _immutable_metadata_error
    __ior__ = _immutable_metadata_error

    def __reduce__(self) -> tuple[type[_FrozenDict], tuple[list[tuple[str, Any]]]]:
        return (_FrozenDict, (list(self.items()),))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        copied = _FrozenDict(
            (copy.deepcopy(k, memo), copy.deepcopy(v, memo)) for k, v in self.items()
        )
        memo[id(self)] = copied
        return copied


class _FrozenList(list[Any]):
    """Immutable list subclass for nested lists inside ``MemoryItem.metadata``."""

    __setitem__ = _immutable_metadata_error
    __delitem__ = _immutable_metadata_error
    append = _immutable_metadata_error
    clear = _immutable_metadata_error
    extend = _immutable_metadata_error
    insert = _immutable_metadata_error
    pop = _immutable_metadata_error
    remove = _immutable_metadata_error
    reverse = _immutable_metadata_error
    sort = _immutable_metadata_error
    __iadd__ = _immutable_metadata_error
    __imul__ = _immutable_metadata_error

    def __reduce__(self) -> tuple[type[_FrozenList], tuple[list[Any]]]:
        return (_FrozenList, (list(self),))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenList:
        copied = _FrozenList(copy.deepcopy(item, memo) for item in self)
        memo[id(self)] = copied
        return copied


def _freeze_metadata_value(value: Any) -> Any:
    """Recursively deep-copy and freeze mutable containers in metadata."""
    if isinstance(value, Mapping):
        return _FrozenDict((k, _freeze_metadata_value(v)) for k, v in value.items())
    if isinstance(value, list):
        return _FrozenList(_freeze_metadata_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return copy.deepcopy(value)


def _derive_item_id(content: str, metadata: Mapping[str, Any]) -> str:
    """Derive a deterministic UUIDv5 from normalized content and metadata.

    Whitespace is collapsed and content is casefolded; metadata keys are sorted
    so dict ordering does not alter the id.  ``score`` is excluded because it is
    a read-time relevance value, not stored identity.
    """
    normalized_content = " ".join(content.split()).casefold()
    canonical_metadata = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    payload = f"{normalized_content}\x00{canonical_metadata}"
    return str(uuid.uuid5(_MEMORY_ITEM_NAMESPACE, payload))


class MemoryScope(BaseModel, frozen=True):
    """The scope key for one memory read or write.

    Attributes
    ----------
    `workspace_id` : str
        Workspace isolation boundary; never omitted (§2.3).
    `agent_id` : str | None
        ``None`` for workspace-wide memory readable by any agent in the
        workspace; set to restrict an item to a single agent.
    `thread_id` : str | None
        ``None`` for long-term workspace memory; a thread id for short-term
        conversation memory.
    """

    workspace_id: str = Field(min_length=1)
    agent_id: str | None = Field(default=None, min_length=1)
    thread_id: str | None = Field(default=None, min_length=1)


class MemoryItem(BaseModel, frozen=True):
    """One stored memory item.

    The upsert key for ``MemoryStore.write`` is the composite
    ``(scope, item_id)`` — the same ``item_id`` (whether explicit or derived)
    written under two different ``MemoryScope``s stores two independent records
    without colliding.

    ``metadata`` is recursively deep-copied and frozen on construction so
    neither caller-side mutations nor post-construction ``item.metadata[...]``
    mutations can alter stored metadata or desync a derived ``item_id``.

    Deduplication semantics:

    * **Default (derived id):** when ``item_id`` is omitted, it is derived
      deterministically (UUIDv5) from normalized ``content`` and ``metadata``
      (excluding ``score``).  Retried or duplicate writes of the same content
      collapse into one record, while rewording a fact (e.g. ``"prefers Arabic"``
      → ``"prefers English"``) produces a new id (append semantics).
    * **Explicit id:** passing ``item_id`` explicitly (for example keyed by a
      stable fact identity such as ``"customer:123:language"``) wins over the
      derived id and gives replace-in-place semantics on update.
    """

    item_id: str = Field(default_factory=str, min_length=1)
    content: str = Field(min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=_FrozenDict)
    score: float = 0.0

    @field_validator("content")
    @classmethod
    def _reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("metadata", mode="after")
    @classmethod
    def _freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_metadata_value(value)

    @model_validator(mode="wrap")
    @classmethod
    def _populate_item_id(
        cls,
        value: Any,
        handler: ModelWrapValidatorHandler[MemoryItem],
    ) -> MemoryItem:
        if isinstance(value, dict) and "item_id" not in value:
            item = handler({**value, "item_id": "_pending"})
            derived = _derive_item_id(item.content, item.metadata)
            object.__setattr__(item, "item_id", derived)
            return item
        return handler(value)
