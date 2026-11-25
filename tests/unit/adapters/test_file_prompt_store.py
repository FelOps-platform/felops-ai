from __future__ import annotations

from pathlib import Path

import pytest

from felops.adapters.prompts import FilePromptStore
from felops.ports.prompts import (
    PromptContext,
    PromptNotFound,
    PromptScope,
    PromptStore,
)

REPO_PROMPTS = Path(__file__).resolve().parents[3] / "prompts"


def _ctx(workspace: str = "w1", sub_agent: str = "customer_support") -> PromptContext:
    return PromptContext(workspace_id=workspace, agent_id="a1", sub_agent_id=sub_agent)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "platform").mkdir()
    (tmp_path / "customer_support").mkdir()
    (tmp_path / "platform" / "responder.md").write_text("platform responder", encoding="utf-8")
    return tmp_path


def test_satisfies_port(root: Path) -> None:
    assert isinstance(FilePromptStore(root), PromptStore)


def test_platform_default_is_used_when_nothing_more_specific(root: Path) -> None:
    pv = FilePromptStore(root).resolve("responder", _ctx())
    assert (pv.template, pv.scope) == ("platform responder", PromptScope.PLATFORM)


def test_sub_agent_default_beats_platform_default(root: Path) -> None:
    (root / "customer_support" / "responder.md").write_text("support responder", encoding="utf-8")
    pv = FilePromptStore(root).resolve("responder", _ctx())
    assert (pv.template, pv.scope) == ("support responder", PromptScope.PLATFORM_SUB_AGENT)
    # another sub-agent still gets the platform default
    assert (
        FilePromptStore(root).resolve("responder", _ctx(sub_agent="billing")).scope
        is PromptScope.PLATFORM
    )


def test_unknown_name_raises_prompt_not_found(root: Path) -> None:
    with pytest.raises(PromptNotFound):
        FilePromptStore(root).resolve("nope", _ctx())


@pytest.mark.parametrize("bad", ["../secret", "a/b", "..", "", "/etc/passwd", "a\\b"])
def test_path_traversal_is_rejected(root: Path, bad: str) -> None:
    (root.parent / "secret.md").write_text("leak", encoding="utf-8")
    with pytest.raises(PromptNotFound):
        FilePromptStore(root).resolve(bad, _ctx())
    # a hostile sub_agent_id must not escape the root either
    with pytest.raises(PromptNotFound):
        FilePromptStore(root).resolve("secret", _ctx(sub_agent="../" + root.name))
    # even when platform/secret.md exists, a hostile sub_agent_id is rejected
    (root / "platform" / "secret.md").write_text("platform secret", encoding="utf-8")
    with pytest.raises(PromptNotFound):
        FilePromptStore(root).resolve("secret", _ctx(sub_agent="../platform"))


@pytest.mark.parametrize("sub_agent", ["platform", "Platform"])
def test_platform_sub_agent_id_resolves_platform_scope(root: Path, sub_agent: str) -> None:
    (root / "Platform").mkdir(exist_ok=True)
    (root / "Platform" / "responder.md").write_text("should not win", encoding="utf-8")
    pv = FilePromptStore(root).resolve("responder", _ctx(sub_agent=sub_agent))
    assert (pv.template, pv.scope) == ("platform responder", PromptScope.PLATFORM)


def test_symlink_escape_outside_root_is_rejected(root: Path) -> None:
    outside = root.parent / "outside.md"
    outside.write_text("leak", encoding="utf-8")
    (root / "platform" / "escape.md").symlink_to(outside)
    with pytest.raises(PromptNotFound):
        FilePromptStore(root).resolve("escape", _ctx())


def test_invalid_utf8_file_raises_prompt_not_found(root: Path) -> None:
    (root / "platform" / "bad_utf8.md").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(PromptNotFound):
        FilePromptStore(root).resolve("bad_utf8", _ctx())


def test_workspace_never_selects_a_file(root: Path) -> None:
    store = FilePromptStore(root)
    assert store.resolve("responder", _ctx("w1")) == store.resolve("responder", _ctx("w2"))


def test_version_is_stable_and_follows_content(root: Path) -> None:
    store = FilePromptStore(root)
    v1 = store.resolve("responder", _ctx()).version
    assert v1.startswith("sha256:")
    assert store.resolve("responder", _ctx()).version == v1
    (root / "platform" / "responder.md").write_text("edited", encoding="utf-8")
    assert store.resolve("responder", _ctx()).version != v1  # picked up with no restart


def test_arabic_text_roundtrips(root: Path) -> None:
    (root / "platform" / "greeting.md").write_text("مرحبا بك", encoding="utf-8")
    assert FilePromptStore(root).resolve("greeting", _ctx()).template == "مرحبا بك"


@pytest.mark.parametrize("name", ["default", "router", "responder"])
def test_shipped_platform_templates_resolve(name: str) -> None:
    pv = FilePromptStore(REPO_PROMPTS).resolve(name, _ctx())
    assert pv.scope is PromptScope.PLATFORM
    assert pv.template.strip()
