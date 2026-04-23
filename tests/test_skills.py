"""Tests for src/skills/ — registry, CLI, and the 3 shipped skills."""
from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from src.skills.registry import (
    Skill,
    SkillRegistry,
    SkillResult,
    get_registry,
    load_skills_from_dir,
)


# ── Registry basics ─────────────────────────────────────────────────────


def _make_skill(name="test", **kwargs):
    return Skill(
        name=name,
        description=kwargs.pop("description", "test skill"),
        handler=kwargs.pop("handler", lambda args: SkillResult(ok=True, message="ok")),
        **kwargs,
    )


def test_registry_register_get_list():
    reg = SkillRegistry()
    reg.register(_make_skill("a"))
    reg.register(_make_skill("b"))
    assert set(reg.list_names()) == {"a", "b"}
    assert reg.get("a").name == "a"
    assert reg.get("missing") is None


def test_registry_last_writer_wins_on_duplicate():
    reg = SkillRegistry()
    reg.register(_make_skill("x", description="old"))
    reg.register(_make_skill("x", description="new"))
    assert reg.get("x").description == "new"


def test_registry_list_filter_by_tag():
    reg = SkillRegistry()
    reg.register(_make_skill("diag", tags=["diagnostic"]))
    reg.register(_make_skill("kill", tags=["destructive"]))
    diagnostic = [s.name for s in reg.list_skills(tag="diagnostic")]
    assert diagnostic == ["diag"]


def test_dispatch_unknown_returns_error():
    reg = SkillRegistry()
    r = reg.dispatch("nope")
    assert r.ok is False
    assert "unknown" in r.error


def test_dispatch_plain_return_wrapped():
    reg = SkillRegistry()
    reg.register(_make_skill("plain", handler=lambda args: "just a string"))
    r = reg.dispatch("plain")
    assert r.ok is True
    assert r.message == "just a string"


def test_dispatch_catches_handler_exception():
    reg = SkillRegistry()
    def _boom(_):
        raise RuntimeError("kaboom")
    reg.register(_make_skill("boom", handler=_boom))
    r = reg.dispatch("boom")
    assert r.ok is False
    assert "RuntimeError" in r.error


def test_dispatch_llm_blocked_without_confirm_on_confirmation_skill():
    reg = SkillRegistry()
    reg.register(_make_skill(
        "danger", requires_confirmation=True,
        handler=lambda args: SkillResult(ok=True, message="did it"),
    ))
    r = reg.dispatch("danger", {}, invoker="llm")
    assert r.ok is False
    assert "requires confirmation" in r.error


def test_dispatch_llm_passes_through_with_confirm():
    reg = SkillRegistry()
    reg.register(_make_skill(
        "danger", requires_confirmation=True,
        handler=lambda args: SkillResult(ok=True, message="did it"),
    ))
    r = reg.dispatch("danger", {"confirm": True}, invoker="llm")
    assert r.ok is True


def test_dispatch_operator_not_blocked_by_confirmation():
    reg = SkillRegistry()
    reg.register(_make_skill(
        "danger", requires_confirmation=True,
        handler=lambda args: SkillResult(ok=True, message="did it"),
    ))
    # Operator invocations don't need confirm at the registry layer — the
    # skill's own action.py may enforce it (as emergency-flatten does).
    r = reg.dispatch("danger", {}, invoker="operator")
    assert r.ok is True


# ── Directory loader ────────────────────────────────────────────────────


def test_load_skills_from_empty_dir(tmp_path):
    reg = load_skills_from_dir(str(tmp_path))
    assert reg.list_names() == []


def test_load_skills_from_valid_dir(tmp_path):
    skill_dir = tmp_path / "hello"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(textwrap.dedent("""\
        name: hello
        description: says hi
        tags: [test]
    """), encoding="utf-8")
    (skill_dir / "action.py").write_text(textwrap.dedent("""\
        from src.skills.registry import SkillResult
        def run(args):
            name = args.get("name", "world")
            return SkillResult(ok=True, message=f"hello, {name}!")
    """), encoding="utf-8")
    reg = load_skills_from_dir(str(tmp_path))
    assert "hello" in reg.list_names()
    skill = reg.get("hello")
    assert "test" in skill.tags
    r = reg.dispatch("hello", {"name": "ACT"})
    assert r.ok is True
    assert "hello, ACT" in r.message


def test_load_skills_ignores_incomplete_dir(tmp_path):
    # Only skill.yaml; missing action.py → skipped.
    d = tmp_path / "broken"
    d.mkdir()
    (d / "skill.yaml").write_text("name: broken\ndescription: x\n", encoding="utf-8")
    reg = load_skills_from_dir(str(tmp_path))
    assert "broken" not in reg.list_names()


def test_load_skills_survives_broken_action(tmp_path):
    d = tmp_path / "syntaxerror"
    d.mkdir()
    (d / "skill.yaml").write_text("name: syntaxerror\ndescription: x\n", encoding="utf-8")
    (d / "action.py").write_text("def run(args):\n    return SYNTAX ERROR\n", encoding="utf-8")
    # Should log a warning but not raise.
    reg = load_skills_from_dir(str(tmp_path))
    assert "syntaxerror" not in reg.list_names()


# ── Shipped skills (loaded from skills/ dir in repo) ────────────────────


def test_shipped_skills_are_discoverable():
    reg = get_registry(refresh=True)
    names = set(reg.list_names())
    required = {"readiness", "regime-check", "emergency-flatten"}
    assert required <= names


def test_shipped_readiness_skill_runs():
    reg = get_registry(refresh=True)
    r = reg.dispatch("readiness", {}, invoker="operator")
    # May or may not be OK depending on readiness_gate state, but must not
    # crash and must return a structured result.
    assert isinstance(r, SkillResult)
    assert "gate" in r.data or r.error is not None


def test_shipped_emergency_flatten_requires_confirm():
    reg = get_registry(refresh=True)
    # Operator without confirm — the skill itself enforces.
    r = reg.dispatch("emergency-flatten", {}, invoker="operator")
    assert r.ok is False
    assert "confirm" in (r.error or "").lower()


def test_shipped_emergency_flatten_runs_with_confirm(monkeypatch, tmp_path):
    # Point warm_store at a tmp DB so the incident row doesn't pollute
    # the real store.
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod
    store = WarmStore(str(tmp_path / "em.sqlite"))
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)

    reg = get_registry(refresh=True)
    r = reg.dispatch("emergency-flatten", {"confirm": True, "reason": "test"}, invoker="operator")
    assert r.ok is True
    assert os.environ.get("ACT_DISABLE_AGENTIC_LOOP") == "1"
    assert "incident_id" in r.data
    assert r.data["incident_id"].startswith("incident-")
    assert any("warm_store_logged" in a for a in r.data["actions_taken"])


# ── CLI ────────────────────────────────────────────────────────────────


def test_cli_list_prints_shipped_skills(capsys):
    from src.skills.cli import main
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/readiness" in out
    assert "/regime-check" in out
    assert "/emergency-flatten" in out
    assert "[CONFIRM]" in out  # emergency-flatten flagged
    assert "[NON-REVERSIBLE]" in out


def test_cli_list_with_tag_filter(capsys):
    from src.skills.cli import main
    rc = main(["list", "--tag", "destructive"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "/emergency-flatten" in out
    assert "/readiness" not in out


def test_cli_describe(capsys):
    from src.skills.cli import main
    rc = main(["describe", "readiness"])
    assert rc == 0
    meta = json.loads(capsys.readouterr().out)
    assert meta["name"] == "readiness"
    assert "ops" in meta["tags"]


def test_cli_run_unknown_exits_nonzero(capsys):
    from src.skills.cli import main
    rc = main(["run", "does-not-exist"])
    assert rc != 0


def test_cli_run_parses_kv_args(capsys, monkeypatch, tmp_path):
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod
    store = WarmStore(str(tmp_path / "clicli.sqlite"))
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)
    monkeypatch.delenv("ACT_DISABLE_AGENTIC_LOOP", raising=False)

    from src.skills.cli import main
    rc = main(["run", "emergency-flatten", "confirm=true", 'reason="cli test"'])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["data"]["reason"] == "cli test"
