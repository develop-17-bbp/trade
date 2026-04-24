"""Tests for src/ai/dual_brain.py — scanner (Qwen) / analyst (Devstral) router."""
from __future__ import annotations

import pytest

from src.ai.dual_brain import (
    ANALYST,
    BRAIN_PROFILES,
    BrainResponse,
    DEFAULT_ANALYST_MODEL,
    DEFAULT_ANALYST_TEMP,
    DEFAULT_PROFILE,
    DEFAULT_SCANNER_MODEL,
    DEFAULT_SCANNER_TEMP,
    DISABLE_ENV,
    PROFILE_ENV,
    SCANNER,
    _resolve,
    _resolve_profile,
    analyze,
    build_analyst_llm_call,
    call_brain,
    is_enabled,
    scan,
    strip_reasoning_tags,
)


# ── Profiles (C5d) ─────────────────────────────────────────────────────


def test_brain_profiles_has_named_set():
    # Post-2026-04 rankings the default moved to qwen3_r1; hybrid was
    # retired because devstral_qwen3coder serves the same "agentic"
    # niche with the models the operator actually has downloaded.
    assert {"qwen3_r1", "dense_r1", "moe_agentic", "devstral_qwen3coder"} <= set(BRAIN_PROFILES.keys())


def test_default_profile_is_qwen3_r1():
    assert DEFAULT_PROFILE == "qwen3_r1"
    # Each profile must carry the four required fields.
    for name, p in BRAIN_PROFILES.items():
        for k in ("scanner_model", "analyst_model",
                  "scanner_temperature", "analyst_temperature", "description"):
            assert k in p, f"profile {name!r} missing {k!r}"


def test_resolve_profile_default_when_env_and_config_empty(monkeypatch):
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    prof = _resolve_profile(None)
    assert prof is BRAIN_PROFILES[DEFAULT_PROFILE]


def test_resolve_profile_env_beats_config(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, "moe_agentic")
    cfg = {"ai": {"dual_brain": {"profile": "devstral_qwen3coder"}}}
    assert _resolve_profile(cfg) is BRAIN_PROFILES["moe_agentic"]


def test_resolve_profile_config_picks_when_no_env(monkeypatch):
    monkeypatch.delenv(PROFILE_ENV, raising=False)
    prof = _resolve_profile({"ai": {"dual_brain": {"profile": "devstral_qwen3coder"}}})
    assert prof is BRAIN_PROFILES["devstral_qwen3coder"]


def test_resolve_profile_unknown_falls_back(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, "definitely_not_a_profile")
    assert _resolve_profile(None) is BRAIN_PROFILES[DEFAULT_PROFILE]


def test_resolve_uses_profile_models_when_no_explicit_config(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, "moe_agentic")
    monkeypatch.delenv("ACT_SCANNER_MODEL", raising=False)
    monkeypatch.delenv("ACT_ANALYST_MODEL", raising=False)
    s = _resolve(None, SCANNER)
    a = _resolve(None, ANALYST)
    assert s.model == BRAIN_PROFILES["moe_agentic"]["scanner_model"]
    assert a.model == BRAIN_PROFILES["moe_agentic"]["analyst_model"]


def test_explicit_config_still_wins_over_profile(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, "moe_agentic")
    cfg = {"ai": {"dual_brain": {"scanner_model": "pinned:7b"}}}
    s = _resolve(cfg, SCANNER)
    a = _resolve(cfg, ANALYST)
    assert s.model == "pinned:7b"
    # analyst wasn't pinned, so profile wins.
    assert a.model == BRAIN_PROFILES["moe_agentic"]["analyst_model"]


def test_per_role_env_still_wins_over_profile(monkeypatch):
    monkeypatch.setenv(PROFILE_ENV, "devstral_qwen3coder")
    monkeypatch.setenv("ACT_SCANNER_MODEL", "env-scan:0.5b")
    s = _resolve(None, SCANNER)
    a = _resolve(None, ANALYST)
    assert s.model == "env-scan:0.5b"
    assert a.model == BRAIN_PROFILES["devstral_qwen3coder"]["analyst_model"]


# ── Config resolution ──────────────────────────────────────────────────


def test_resolve_defaults_when_no_config():
    s = _resolve(None, SCANNER)
    assert s.role == SCANNER
    assert s.model == DEFAULT_SCANNER_MODEL
    assert abs(s.temperature - DEFAULT_SCANNER_TEMP) < 1e-9
    assert "SCANNER" in s.system_prompt

    a = _resolve(None, ANALYST)
    assert a.role == ANALYST
    assert a.model == DEFAULT_ANALYST_MODEL
    assert abs(a.temperature - DEFAULT_ANALYST_TEMP) < 1e-9
    assert "ANALYST" in a.system_prompt


def test_resolve_uses_config_values():
    cfg = {"ai": {"dual_brain": {
        "scanner_model": "custom-scan:7b",
        "analyst_model": "custom-analyst:3b",
        "scanner_temperature": 0.9,
        "analyst_temperature": 0.05,
    }}}
    assert _resolve(cfg, SCANNER).model == "custom-scan:7b"
    assert _resolve(cfg, ANALYST).model == "custom-analyst:3b"
    assert abs(_resolve(cfg, SCANNER).temperature - 0.9) < 1e-9
    assert abs(_resolve(cfg, ANALYST).temperature - 0.05) < 1e-9


def test_env_override_beats_config(monkeypatch):
    monkeypatch.setenv("ACT_SCANNER_MODEL", "env-scan")
    monkeypatch.setenv("ACT_ANALYST_MODEL", "env-analyst")
    cfg = {"ai": {"dual_brain": {"scanner_model": "cfg-scan", "analyst_model": "cfg-analyst"}}}
    assert _resolve(cfg, SCANNER).model == "env-scan"
    assert _resolve(cfg, ANALYST).model == "env-analyst"


def test_resolve_rejects_invalid_brain():
    with pytest.raises(AssertionError):
        _resolve(None, "middlebrain")


# ── is_enabled ──────────────────────────────────────────────────────────


def test_is_enabled_env_kill_switch(monkeypatch):
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert is_enabled({"ai": {"dual_brain": {"enabled": True}}}) is False


def test_is_enabled_config(monkeypatch):
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    assert is_enabled({"ai": {"dual_brain": {"enabled": True}}}) is True
    assert is_enabled({"ai": {"dual_brain": {"enabled": False}}}) is False
    assert is_enabled({}) is False
    assert is_enabled(None) is False


# ── call_brain ──────────────────────────────────────────────────────────


def test_call_brain_primary_success_no_fallback():
    seen = {}

    def stub(model, prompt, sys_prompt, temp):
        seen["model"] = model
        seen["temp"] = temp
        return "scanner-ok"

    resp = call_brain(SCANNER, "hello", llm_call=stub)
    assert resp.ok is True
    assert resp.text == "scanner-ok"
    assert resp.fallback_used is False
    assert seen["model"] == DEFAULT_SCANNER_MODEL
    assert abs(seen["temp"] - DEFAULT_SCANNER_TEMP) < 1e-9


def test_call_brain_cross_fallback_fires_when_primary_empty():
    calls = []

    def stub(model, prompt, sys_prompt, temp):
        calls.append(model)
        # Primary (analyst) returns empty; fallback (scanner) returns text.
        if model == DEFAULT_ANALYST_MODEL:
            return ""
        return "scanner-saved-us"

    resp = call_brain(ANALYST, "compile plan", llm_call=stub)
    assert resp.ok is True
    assert resp.fallback_used is True
    assert resp.text == "scanner-saved-us"
    # Model reported is the fallback brain's model.
    assert resp.model == DEFAULT_SCANNER_MODEL
    # Both models got exactly one call.
    assert calls == [DEFAULT_ANALYST_MODEL, DEFAULT_SCANNER_MODEL]


def test_call_brain_cross_fallback_disabled_by_config():
    cfg = {"ai": {"dual_brain": {"fallback_cross": False}}}

    def stub(_m, _p, _s, _t):
        return ""

    resp = call_brain(SCANNER, "x", config=cfg, llm_call=stub)
    assert resp.ok is False
    assert resp.error == "primary_empty_no_fallback"
    assert resp.fallback_used is False


def test_call_brain_both_empty_returns_ok_false():
    def stub(*_):
        return ""

    resp = call_brain(ANALYST, "x", llm_call=stub)
    assert resp.ok is False
    assert resp.error == "both_brains_empty"


def test_call_brain_invalid_brain_returns_error_response():
    resp = call_brain("middlebrain", "x")
    assert resp.ok is False
    assert "invalid brain" in (resp.error or "")


def test_call_brain_extra_system_appended():
    seen = {}

    def stub(model, prompt, sys_prompt, temp):
        seen["sys"] = sys_prompt
        return "ok"

    call_brain(ANALYST, "p", extra_system="## TICK CONTEXT\nextra instructions", llm_call=stub)
    assert "extra instructions" in seen["sys"]
    assert "ANALYST" in seen["sys"]


def test_scan_and_analyze_convenience_wrappers():
    def stub(model, _p, _s, _t):
        return f"from {model}"

    r1 = scan("survey the market", llm_call=stub)
    r2 = analyze("compile plan", llm_call=stub)
    assert r1.brain == SCANNER
    assert r2.brain == ANALYST
    assert DEFAULT_SCANNER_MODEL in r1.text
    assert DEFAULT_ANALYST_MODEL in r2.text


# ── BrainResponse serialization ─────────────────────────────────────────


def test_brain_response_to_dict_shape():
    r = BrainResponse(brain=SCANNER, model="qwen", text="hi", ok=True)
    d = r.to_dict()
    assert d["brain"] == SCANNER
    assert d["model"] == "qwen"
    assert d["ok"] is True
    assert d["fallback_used"] is False


def test_brain_response_trims_long_text():
    r = BrainResponse(brain=ANALYST, model="x", text="y" * 5000, ok=True)
    assert len(r.to_dict()["text"]) <= 2000


# ── build_analyst_llm_call ──────────────────────────────────────────────


# ── strip_reasoning_tags ────────────────────────────────────────────────


def test_strip_think_noop_when_absent():
    assert strip_reasoning_tags("just a plain answer") == "just a plain answer"
    assert strip_reasoning_tags("") == ""


def test_strip_think_removes_closed_block():
    raw = "<think>step 1\nstep 2\nstep 3</think>\n{\"opportunity_score\": 72}"
    out = strip_reasoning_tags(raw)
    assert "step 1" not in out
    assert "opportunity_score" in out


def test_strip_think_case_insensitive():
    raw = "<THINK>reasoning</THINK>answer"
    assert strip_reasoning_tags(raw) == "answer"


def test_strip_think_removes_unclosed_block_at_end():
    raw = "final answer before reasoning\n<think>ran out of tokens mid-thought"
    out = strip_reasoning_tags(raw)
    assert out == "final answer before reasoning"


def test_strip_think_multiple_blocks():
    raw = "<think>a</think>part1<think>b</think>part2"
    assert strip_reasoning_tags(raw) == "part1part2"


def test_call_brain_strips_scanner_think_trace():
    def stub(model, prompt, sys_prompt, temp):
        return "<think>reasoning trace ...</think>\n{\"opportunity_score\": 60}"

    resp = call_brain(SCANNER, "scan", llm_call=stub)
    assert resp.ok is True
    assert "<think>" not in resp.text
    assert "opportunity_score" in resp.text


def test_call_brain_preserves_analyst_think_trace():
    def stub(model, prompt, sys_prompt, temp):
        return "<think>deliberation ...</think>\n{\"plan\": {\"asset\": \"BTC\"}}"

    resp = call_brain(ANALYST, "compile plan", llm_call=stub)
    assert resp.ok is True
    # Analyst keeps the <think> block so brain_memory + audit see it.
    assert "<think>" in resp.text


def test_analyst_llm_call_flattens_messages(monkeypatch):
    seen = {}

    def fake_llm(model, prompt, sys_prompt, temp):
        seen["model"] = model
        seen["prompt"] = prompt
        seen["sys"] = sys_prompt
        return "ok"

    # Swap the private _llm_call for a stub by monkeypatching the module.
    import src.ai.dual_brain as db
    monkeypatch.setattr(db, "_llm_call", fake_llm)

    closure = build_analyst_llm_call(None)
    out = closure([
        {"role": "system", "content": "sys — should be stripped"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ])
    assert out == "ok"
    assert seen["model"] == DEFAULT_ANALYST_MODEL
    assert "hello" in seen["prompt"]
    assert "hi there" in seen["prompt"]
    # Analyst's system prompt comes from dual_brain, not the message list.
    assert "ANALYST" in seen["sys"]
    assert "should be stripped" not in seen["prompt"]
