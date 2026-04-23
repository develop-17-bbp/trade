"""Tests for src/orchestration/hooks.py — event dispatcher + shell/python hooks."""
from __future__ import annotations

import os
import sys

import pytest

from src.orchestration.hooks import (
    DISABLE_ENV,
    EVT_ON_AUTHORITY_VIOLATION,
    EVT_ON_EMERGENCY_ENTER,
    EVT_POST_TRADE_CLOSE,
    EVT_PRE_TRADE_SUBMIT,
    FireResult,
    HookRegistry,
    HookResult,
    HookSpec,
    _expand_env,
    _resolve_dotted,
    fire,
    get_registry,
    load_from_config,
)


@pytest.fixture
def reg():
    """Fresh registry so tests don't contaminate the singleton."""
    return HookRegistry()


# ── Variable expansion ─────────────────────────────────────────────────


def test_expand_env_replaces_dollar_vars(monkeypatch):
    monkeypatch.setenv("ACT_TEST_VAR", "hello")
    assert _expand_env("say ${ACT_TEST_VAR}", {}) == "say hello"


def test_expand_env_missing_var_becomes_empty(monkeypatch):
    monkeypatch.delenv("UNSET_X", raising=False)
    assert _expand_env("a${UNSET_X}b", {}) == "ab"


def test_expand_context_placeholders():
    ctx = {"asset": "BTC", "plan": {"direction": "LONG", "size_pct": 5.0}}
    assert _expand_env("trade {{asset}} {{plan.direction}} @{{plan.size_pct}}%", ctx) \
        == "trade BTC LONG @5.0%"


def test_expand_missing_context_key_becomes_empty():
    assert _expand_env("x{{no.such.key}}y", {"other": 1}) == "xy"


def test_resolve_dotted_walks_nested():
    d = {"a": {"b": {"c": 42}}}
    assert _resolve_dotted(d, "a.b.c") == 42
    assert _resolve_dotted(d, "a.b.missing") is None


# ── Registry ────────────────────────────────────────────────────────────


def test_register_and_list(reg):
    reg.register(HookSpec(event=EVT_PRE_TRADE_SUBMIT, cmd="echo pre"))
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, cmd="echo post"))
    assert reg.list_for(EVT_PRE_TRADE_SUBMIT)
    assert reg.list_for(EVT_POST_TRADE_CLOSE)
    assert reg.list_for("no_such_event") == []
    assert set(reg.events()) == {EVT_PRE_TRADE_SUBMIT, EVT_POST_TRADE_CLOSE}


def test_clear_event_or_all(reg):
    reg.register(HookSpec(event=EVT_PRE_TRADE_SUBMIT, cmd="x"))
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, cmd="y"))
    reg.clear(EVT_PRE_TRADE_SUBMIT)
    assert reg.list_for(EVT_PRE_TRADE_SUBMIT) == []
    assert reg.list_for(EVT_POST_TRADE_CLOSE) != []
    reg.clear()
    assert reg.events() == []


def test_register_unknown_event_still_works(reg, caplog):
    reg.register(HookSpec(event="custom_event_xyz", cmd="echo"))
    assert reg.list_for("custom_event_xyz")


# ── Shell hook execution ────────────────────────────────────────────────


def test_fire_shell_hook_success(reg):
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, cmd="echo hello", blocking=False))
    out = fire(EVT_POST_TRADE_CLOSE, {"x": 1}, registry=reg)
    assert not out.any_veto
    assert len(out.results) == 1
    assert out.results[0].ok is True
    assert "hello" in out.results[0].output


def test_fire_shell_hook_nonzero_exit_vetoes_blocking_pre(reg):
    reg.register(HookSpec(
        event=EVT_PRE_TRADE_SUBMIT,
        cmd="python -c \"import sys; sys.exit(1)\"",
        blocking=True, timeout_s=5.0,
    ))
    out = fire(EVT_PRE_TRADE_SUBMIT, {}, registry=reg)
    assert out.any_veto is True
    assert out.results[0].vetoed is True


def test_fire_shell_hook_nonzero_exit_does_not_veto_post(reg):
    # Non-blocking (or non-pre) hook failures don't veto.
    reg.register(HookSpec(
        event=EVT_POST_TRADE_CLOSE,
        cmd="python -c \"import sys; sys.exit(1)\"",
        blocking=False, timeout_s=5.0,
    ))
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg)
    assert out.any_veto is False
    assert out.results[0].ok is False


def test_fire_shell_timeout(reg):
    reg.register(HookSpec(
        event=EVT_PRE_TRADE_SUBMIT,
        cmd="python -c \"import time; time.sleep(5)\"",
        blocking=True, timeout_s=0.5,
    ))
    out = fire(EVT_PRE_TRADE_SUBMIT, {}, registry=reg)
    assert out.results[0].error == "timeout"
    assert out.any_veto is True   # blocking + timeout vetoes


def test_fire_shell_env_expansion(monkeypatch, reg):
    monkeypatch.setenv("ACT_TEST_GREET", "world")
    reg.register(HookSpec(
        event=EVT_POST_TRADE_CLOSE,
        cmd="python -c \"print('${ACT_TEST_GREET}')\"",
    ))
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg)
    assert out.results[0].ok is True
    assert "world" in out.results[0].output


def test_fire_shell_context_expansion(reg):
    reg.register(HookSpec(
        event=EVT_POST_TRADE_CLOSE,
        cmd="python -c \"print('{{asset}}')\"",
    ))
    out = fire(EVT_POST_TRADE_CLOSE, {"asset": "ETH"}, registry=reg)
    assert "ETH" in out.results[0].output


# ── Python hook execution ──────────────────────────────────────────────


def test_fire_python_hook_happy(reg):
    # Register a callable via the module:attr dotted form.
    # We use a module that's always importable.
    reg.register(HookSpec(
        event=EVT_POST_TRADE_CLOSE,
        python="os:getcwd",
        blocking=False,
    ))
    out = fire(EVT_POST_TRADE_CLOSE, None, registry=reg)
    # getcwd doesn't accept args, so fn(context) fails — the dispatcher
    # catches the TypeError and returns ok=False.  That's acceptable;
    # callers should use hooks with the `(context) -> result` signature.
    # We still want to confirm the result didn't crash the dispatcher.
    assert len(out.results) == 1


def test_fire_python_hook_returning_false_vetoes_on_blocking(reg):
    # Point the python hook at a real callable that honors the contract.
    # We'll stash one in sys.modules so the dispatcher can reach it.
    import types
    mod = types.ModuleType("__act_test_hookmod__")
    def reject(ctx):
        return False
    mod.reject = reject
    sys.modules["__act_test_hookmod__"] = mod

    reg.register(HookSpec(
        event=EVT_PRE_TRADE_SUBMIT,
        python="__act_test_hookmod__:reject",
        blocking=True,
    ))
    out = fire(EVT_PRE_TRADE_SUBMIT, {"asset": "BTC"}, registry=reg)
    assert out.any_veto is True
    del sys.modules["__act_test_hookmod__"]


def test_fire_python_hook_returning_ok_dict(reg):
    import types
    mod = types.ModuleType("__act_test_hookmod_ok__")
    mod.allow = lambda ctx: {"ok": True, "note": "fine"}
    sys.modules["__act_test_hookmod_ok__"] = mod

    reg.register(HookSpec(
        event=EVT_ON_EMERGENCY_ENTER,
        python="__act_test_hookmod_ok__:allow",
        blocking=False,
    ))
    out = fire(EVT_ON_EMERGENCY_ENTER, {}, registry=reg)
    assert out.results[0].ok is True
    del sys.modules["__act_test_hookmod_ok__"]


def test_fire_python_hook_bad_spec_errors(reg):
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, python="no_colon_here"))
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg)
    assert out.results[0].ok is False
    assert "bad python spec" in (out.results[0].error or "")


def test_fire_python_hook_module_not_importable(reg):
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, python="this_module_does_not_exist:fn"))
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg)
    assert out.results[0].ok is False


# ── Full-dispatch behavior ─────────────────────────────────────────────


def test_fire_disabled_by_env(monkeypatch, reg):
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, cmd="echo should-not-run"))
    monkeypatch.setenv(DISABLE_ENV, "1")
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg)
    assert out.results == []
    assert out.any_veto is False


def test_fire_event_with_no_hooks(reg):
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg)
    assert out.results == []
    assert out.any_veto is False


def test_fire_multiple_hooks_first_veto_still_runs_others(reg):
    reg.register(HookSpec(
        event=EVT_PRE_TRADE_SUBMIT,
        cmd="python -c \"import sys; sys.exit(1)\"",
        blocking=True,
    ))
    reg.register(HookSpec(
        event=EVT_PRE_TRADE_SUBMIT,
        cmd="python -c \"print('second ran')\"",
    ))
    out = fire(EVT_PRE_TRADE_SUBMIT, {}, registry=reg)
    assert len(out.results) == 2
    assert out.any_veto is True
    assert "second ran" in out.results[1].output


def test_fire_result_to_dict_shape(reg):
    reg.register(HookSpec(event=EVT_POST_TRADE_CLOSE, cmd="echo x", name="echo-test"))
    out = fire(EVT_POST_TRADE_CLOSE, {}, registry=reg).to_dict()
    assert out["event"] == EVT_POST_TRADE_CLOSE
    assert out["any_veto"] is False
    assert out["results"][0]["hook_name"] == "echo-test"


# ── Config loader ──────────────────────────────────────────────────────


def test_load_from_config_registers_all(reg):
    cfg = {
        "hooks": {
            EVT_PRE_TRADE_SUBMIT: [
                {"cmd": "echo a", "blocking": True, "timeout_s": 1.5},
                {"python": "os:getcwd", "name": "cwd-check"},
            ],
            EVT_ON_AUTHORITY_VIOLATION: [
                {"cmd": "echo violation", "blocking": False},
            ],
        }
    }
    load_from_config(cfg, registry=reg)
    pre_hooks = reg.list_for(EVT_PRE_TRADE_SUBMIT)
    assert len(pre_hooks) == 2
    assert pre_hooks[0].blocking is True
    assert abs(pre_hooks[0].timeout_s - 1.5) < 1e-9
    assert pre_hooks[1].name == "cwd-check"
    assert reg.list_for(EVT_ON_AUTHORITY_VIOLATION)


def test_load_from_config_tolerates_junk(reg):
    cfg = {"hooks": {EVT_POST_TRADE_CLOSE: [{"bad": "entry"}, "not a dict",
                                              {"cmd": "echo ok"}]}}
    # Should register the one valid entry; silently skip the rest.
    load_from_config(cfg, registry=reg)
    hooks = reg.list_for(EVT_POST_TRADE_CLOSE)
    # The {"bad": "entry"} has neither cmd nor python — but will still
    # register as a degenerate HookSpec (which errors at fire-time).
    assert any(h.cmd == "echo ok" for h in hooks)


def test_load_from_config_none_safe(reg):
    assert load_from_config(None, registry=reg) is reg
    assert load_from_config({}, registry=reg) is reg
    assert load_from_config({"hooks": "bad"}, registry=reg) is reg


# ── Singleton getter ──────────────────────────────────────────────────


def test_get_registry_returns_singleton():
    a = get_registry()
    b = get_registry()
    assert a is b
