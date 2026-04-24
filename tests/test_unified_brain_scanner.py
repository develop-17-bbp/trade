"""Tests for C26 unified brain — MultiModelConsensus.query_two_pass
routes through dual_brain + publishes ScanReport to brain_memory."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest


def _fresh_consensus():
    from src.ai.trading_brain import MultiModelConsensus
    return MultiModelConsensus(ollama_base_url="http://localhost:11434")


def test_scanner_uses_dual_brain_not_direct_ollama(monkeypatch):
    """Verify scanner calls dual_brain.scan, not _call_ollama."""
    cons = _fresh_consensus()

    scan_called_with = {}
    def _fake_scan(prompt, **kwargs):
        scan_called_with["prompt"] = prompt
        resp = MagicMock()
        resp.text = '{"pattern_bias":"BULLISH","pattern_strength":7,"strongest_signal":"ema_cross"}'
        resp.model = "test-model"
        resp.error = ""
        return resp

    def _fake_analyze(prompt, **kwargs):
        resp = MagicMock()
        resp.text = '{"proceed":true,"confidence":0.7,"risk_score":4}'
        resp.model = "test-model"
        resp.error = ""
        return resp

    # Monkey-patch ollama call to catch if legacy path is used
    def _never_called(*a, **k):
        raise AssertionError("_call_ollama was called — unified path broken")
    monkeypatch.setattr(cons, "_call_ollama", _never_called)

    monkeypatch.setattr("src.ai.dual_brain.scan", _fake_scan)
    monkeypatch.setattr("src.ai.dual_brain.analyze", _fake_analyze)

    def _analyst_builder(pattern_scan_result, **kwargs):
        # analyst prompt just needs to be a string
        return f"analyst prompt with bias={pattern_scan_result.get('pattern_bias')}"

    result = cons.query_two_pass(
        scanner_prompt="scan BTC",
        analyst_prompt_builder=_analyst_builder,
        asset="BTC",
    )
    # Scanner was invoked
    assert "scan BTC" in scan_called_with.get("prompt", "")
    # Result carries through
    assert result["proceed"] is True


def test_scanner_publishes_to_brain_memory(tmp_path, monkeypatch):
    """Verify publish_scan is called with a valid ScanReport."""
    cons = _fresh_consensus()

    captured = {}

    def _fake_publish(report):
        captured["report"] = report

    def _fake_scan(prompt, **kwargs):
        resp = MagicMock()
        resp.text = '{"pattern_bias":"BEARISH","pattern_strength":8,"strongest_signal":"breakdown"}'
        resp.model = "dual-brain-test"
        resp.error = ""
        return resp

    def _fake_analyze(prompt, **kwargs):
        resp = MagicMock()
        resp.text = '{"proceed":false,"confidence":0.3}'
        resp.model = "dual-brain-test"
        resp.error = ""
        return resp

    monkeypatch.setattr("src.ai.dual_brain.scan", _fake_scan)
    monkeypatch.setattr("src.ai.dual_brain.analyze", _fake_analyze)
    monkeypatch.setattr("src.ai.brain_memory.publish_scan", _fake_publish)

    def _ab(pattern_scan_result, **kwargs):
        return "analyst prompt"

    cons.query_two_pass(
        scanner_prompt="scan ETH",
        analyst_prompt_builder=_ab,
        asset="ETH",
    )

    # ScanReport was published
    rep = captured.get("report")
    assert rep is not None
    assert rep.asset == "ETH"
    assert rep.proposed_direction == "SHORT"     # BEARISH → SHORT
    assert rep.opportunity_score == pytest.approx(80.0, abs=0.1)
    assert "bias=BEARISH" in rep.rationale


def test_empty_scanner_returns_falls_back_to_empty_patterns(monkeypatch):
    """If dual_brain returns empty text, scan result is the empty-default."""
    cons = _fresh_consensus()

    def _empty_scan(prompt, **kwargs):
        resp = MagicMock()
        resp.text = ""
        resp.model = "x"
        resp.error = "provider empty"
        return resp

    def _analyze_safe(prompt, **kwargs):
        resp = MagicMock()
        resp.text = '{"proceed":false,"confidence":0.2}'
        resp.model = "x"
        resp.error = ""
        return resp

    monkeypatch.setattr("src.ai.dual_brain.scan", _empty_scan)
    monkeypatch.setattr("src.ai.dual_brain.analyze", _analyze_safe)

    captured = []
    monkeypatch.setattr("src.ai.brain_memory.publish_scan",
                        lambda r: captured.append(r))

    def _ab(pattern_scan_result, **kwargs):
        # empty-default should still have pattern_bias=NEUTRAL
        assert pattern_scan_result["pattern_bias"] == "NEUTRAL"
        return "analyst prompt"

    result = cons.query_two_pass(
        scanner_prompt="scan BTC",
        analyst_prompt_builder=_ab,
        asset="BTC",
    )
    # Got safe-default
    assert result["proceed"] is False
    # publish_scan still called — with empty-default patterns
    assert len(captured) == 1
    assert captured[0].proposed_direction == "FLAT"


def test_empty_analyst_returns_safe_reject(monkeypatch):
    """If dual_brain analyst returns empty text, result is safe-reject."""
    cons = _fresh_consensus()

    def _scan_ok(prompt, **kwargs):
        resp = MagicMock()
        resp.text = '{"pattern_bias":"BULLISH","pattern_strength":6}'
        resp.model = "x"
        resp.error = ""
        return resp

    def _empty_analyze(prompt, **kwargs):
        resp = MagicMock()
        resp.text = ""
        resp.model = "x"
        resp.error = "oom"
        return resp

    monkeypatch.setattr("src.ai.dual_brain.scan", _scan_ok)
    monkeypatch.setattr("src.ai.dual_brain.analyze", _empty_analyze)
    monkeypatch.setattr("src.ai.brain_memory.publish_scan", lambda r: None)

    def _ab(pattern_scan_result, **kwargs):
        return "analyst prompt"

    result = cons.query_two_pass(
        scanner_prompt="scan BTC",
        analyst_prompt_builder=_ab,
        asset="BTC",
    )
    assert result["proceed"] is False
    assert "error" in result["facilitator_verdict"].lower() or \
           "empty" in result.get("bull_case", "").lower()


def test_dual_brain_exception_safely_returns_default(monkeypatch):
    """dual_brain raising is caught — scanner falls back to empty-default."""
    cons = _fresh_consensus()

    def _raises(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr("src.ai.dual_brain.scan", _raises)
    monkeypatch.setattr("src.ai.dual_brain.analyze", _raises)
    monkeypatch.setattr("src.ai.brain_memory.publish_scan", lambda r: None)

    def _ab(pattern_scan_result, **kwargs):
        return "analyst prompt"

    result = cons.query_two_pass(
        scanner_prompt="scan BTC",
        analyst_prompt_builder=_ab,
        asset="BTC",
    )
    # Scanner exception → empty pattern; analyst exception → safe reject.
    assert result["proceed"] is False


def test_legacy_model_constants_preserved_as_labels():
    """C26 removed fallback chains but kept legacy labels for audit."""
    from src.ai.trading_brain import MultiModelConsensus
    assert MultiModelConsensus.MODEL_SCANNER == "act-scanner"
    assert MultiModelConsensus.MODEL_ANALYST == "act-analyst"
    # Fallback lists are empty post-C26
    assert MultiModelConsensus.MODEL_SCANNER_FALLBACKS == []
    assert MultiModelConsensus.MODEL_ANALYST_FALLBACKS == []
