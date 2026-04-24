"""C21 bundle tests — FinToolBench metadata, news risk classifier,
persona prompts, output scrubber, privacy audit, instance lock,
chart vision stub. Safety-oriented features from the academic lit
review (FinVault / FinToolBench / GuruAgents / Beyond-Refusal /
AgentSCOPE / collusion / FinAgent)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

# ── tool_metadata (FinToolBench) ──────────────────────────────────────


def test_tool_metadata_classify_returns_unknown_for_missing():
    from src.ai.tool_metadata import classify
    c = classify("nonexistent_tool")
    assert c.timeliness == "unknown"
    assert c.intent_type == "unknown"
    assert c.regulatory == "unknown"
    assert c.is_fully_classified() is False


def test_tool_metadata_known_tools_fully_classified():
    from src.ai.tool_metadata import classify
    for name in ("submit_trade_plan", "get_recent_bars",
                 "fit_ou_process", "query_knowledge_graph"):
        c = classify(name)
        assert c.is_fully_classified(), name


def test_tool_metadata_filter_by_intent():
    from src.ai.tool_metadata import filter_tools_by_intent
    actions = filter_tools_by_intent("action")
    assert "submit_trade_plan" in actions
    # submit_trade_plan should be the ONLY classified action tool
    assert len(actions) <= 3


def test_tool_metadata_audit_coverage_has_expected_shape():
    from src.ai.tool_metadata import audit_coverage
    a = audit_coverage()
    assert a["total_tools"] >= 10
    assert a["fully_classified"] >= 10
    assert 0 <= a["coverage_pct"] <= 100
    assert "by_regulatory" in a


# ── news_risk_classifier ─────────────────────────────────────────────


def test_risk_hack_classification_with_dollar_figure():
    from src.ai.news_risk_classifier import classify_risk_event
    c = classify_risk_event("Major exchange hacked, $120 million stolen")
    assert c.severity == "critical"
    assert c.event_type == "hack"


def test_risk_depeg_critical():
    from src.ai.news_risk_classifier import classify_risk_event
    c = classify_risk_event("USDC broke its peg overnight as reserves questioned")
    assert c.severity == "critical"
    assert c.event_type == "depeg"


def test_risk_benign_headline_returns_none():
    from src.ai.news_risk_classifier import classify_risk_event
    c = classify_risk_event("BTC ETF sees record inflows this week")
    assert c.event_type == "none"
    assert c.severity == "none"


def test_risk_distribution_summary_spots_critical():
    from src.ai.news_risk_classifier import summarize_risk_distribution
    headlines = [
        "Exchange hacked, $50 million drained",
        "SEC sues major stablecoin issuer",
        "BTC up 2% on ETF optimism",
    ]
    d = summarize_risk_distribution(headlines)
    assert d["any_critical"] is True
    assert d["any_high"] is True
    assert d["worst_item"]["event_type"] == "hack"


# ── personality_prompts (GuruAgents) ─────────────────────────────────


def test_persona_empty_by_default(monkeypatch):
    monkeypatch.delenv("ACT_AGENT_PERSONAS", raising=False)
    from src.agents.personality_prompts import get_persona_prompt
    assert get_persona_prompt("trend_momentum") == ""


def test_persona_returns_snippet_when_enabled(monkeypatch):
    monkeypatch.setenv("ACT_AGENT_PERSONAS", "1")
    from src.agents.personality_prompts import get_persona_prompt
    s = get_persona_prompt("trend_momentum")
    assert "Paul Tudor Jones" in s


def test_persona_all_13_covered():
    from src.agents.personality_prompts import list_agents_with_personas
    mapping = list_agents_with_personas()
    # All mapped names should have a non-empty snippet
    assert all(mapping.values())
    # Coverage check — expected canonical 13
    canonical = {
        "trend_momentum", "mean_reversion", "risk_guardian",
        "loss_prevention_guardian", "market_structure",
        "regime_intelligence", "sentiment_decoder", "trade_timing",
        "portfolio_optimizer", "pattern_matcher", "decision_auditor",
        "data_integrity_validator", "polymarket_agent",
    }
    assert canonical <= set(mapping)


# ── output_scrubber (Beyond Refusal) ─────────────────────────────────


def test_scrubber_redacts_openai_key():
    from src.ai.output_scrubber import scrub
    text = "Here is your token: sk-abc123def456ghi789jklmnopqrstuvwx all done"
    r = scrub(text)
    assert "sk-abc" not in r.text
    assert "[REDACTED:OPENAI_KEY]" in r.text
    assert r.any_redacted is True


def test_scrubber_redacts_email():
    from src.ai.output_scrubber import scrub
    r = scrub("Contact admin at alice@example.com for details")
    assert "alice@example.com" not in r.text
    assert "[REDACTED:EMAIL]" in r.text


def test_scrubber_noop_on_clean_text():
    from src.ai.output_scrubber import scrub
    r = scrub("BTC is trading at 77000 with ETH lagging")
    assert r.any_redacted is False
    assert "BTC is trading" in r.text


def test_scrubber_dict_walks_nested():
    from src.ai.output_scrubber import scrub_dict
    d = {
        "thesis": "normal text",
        "secrets": {"key": "sk-proj-1234567890abcdefghij"},
        "mixed": ["sk-ant-01234567890abcdefghij", 42, None],
    }
    out = scrub_dict(d)
    assert "REDACTED" in out["secrets"]["key"]
    assert "REDACTED" in out["mixed"][0]
    assert out["mixed"][1] == 42
    assert out["thesis"] == "normal text"


def test_scrubber_none_and_non_string():
    from src.ai.output_scrubber import scrub
    assert scrub(None).text == ""
    assert scrub(42).text == "42"


# ── privacy_audit (AgentSCOPE) ──────────────────────────────────────


def test_privacy_audit_clean_payload_no_sensitive_tags():
    from src.ai.privacy_audit import scan_payload_for_sensitive_tags
    assert scan_payload_for_sensitive_tags({"trade": "BTC LONG"}) == []


def test_privacy_audit_spots_nested_sensitive_key():
    from src.ai.privacy_audit import scan_payload_for_sensitive_tags
    payload = {
        "thesis": "safe text",
        "config": {"robinhood_account_id": "311059868612"},
    }
    tags = scan_payload_for_sensitive_tags(payload)
    assert "robinhood_account_id" in tags


def test_privacy_audit_flows_from_producer():
    from src.ai.privacy_audit import audit_flow
    flows = audit_flow("scanner_brain", {"opportunity_score": 75})
    # scanner_brain has several known consumers
    assert len(flows) >= 3
    for f in flows:
        assert f.producer == "scanner_brain"
        assert f.risk_level() == "none"


def test_privacy_audit_summary_flags_high_risk():
    from src.ai.privacy_audit import audit_flow, summarize_audit
    payload = {
        "robinhood_account_id": "...",
        "wallet_private_key": "...",
        "operator_email": "...",
    }
    flows = audit_flow("scanner_brain", payload)
    s = summarize_audit(flows)
    assert s["any_high_risk"] is True


# ── instance_lock (collusion prevention) ────────────────────────────


def test_instance_lock_acquire_and_release(tmp_path):
    from src.orchestration.instance_lock import acquire, release, status
    lock = tmp_path / "act.lock"
    payload = acquire(str(lock), instance_id="test-1")
    assert payload["pid"] == os.getpid()
    s = status(str(lock))
    assert s["held"] is True
    assert s["owner_pid"] == os.getpid()
    assert release(str(lock)) is True
    s2 = status(str(lock))
    assert s2["held"] is False


def test_instance_lock_rejects_second_acquire_when_alive(tmp_path):
    from src.orchestration.instance_lock import acquire, InstanceLockError
    lock = tmp_path / "act.lock"
    acquire(str(lock), instance_id="first")
    with pytest.raises(InstanceLockError):
        acquire(str(lock), instance_id="second")


def test_instance_lock_force_reclaims(tmp_path):
    from src.orchestration.instance_lock import acquire
    lock = tmp_path / "act.lock"
    acquire(str(lock), instance_id="first")
    # force=True should succeed even if first still "holds"
    p2 = acquire(str(lock), instance_id="recovery", force=True)
    assert p2["instance_id"] == "recovery"


def test_instance_lock_status_on_missing_file(tmp_path):
    from src.orchestration.instance_lock import status
    s = status(str(tmp_path / "does_not_exist.lock"))
    assert s["held"] is False


# ── chart_vision stub ───────────────────────────────────────────────


def test_chart_vision_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ACT_CHART_VISION_MODEL", raising=False)
    monkeypatch.delenv("ACT_DISABLE_CHART_VISION", raising=False)
    from src.ai.chart_vision import is_enabled
    assert is_enabled() is False


def test_chart_vision_disabled_by_kill_switch(monkeypatch):
    monkeypatch.setenv("ACT_CHART_VISION_MODEL", "llama3.2-vision:11b")
    monkeypatch.setenv("ACT_DISABLE_CHART_VISION", "1")
    from src.ai.chart_vision import is_enabled
    assert is_enabled() is False


def test_chart_vision_summary_section_returns_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("ACT_CHART_VISION_MODEL", raising=False)
    from src.ai.chart_vision import chart_summary_section
    out = chart_summary_section("BTC")
    assert out["content"] == ""
    assert "disabled" in out["source"].lower()
