"""Tests for scenario_predictor + web_search modules."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ── Scenario Predictor ─────────────────────────────────────────────────


def test_scenario_predict_rejects_unsafe_dsl():
    from src.ai.scenario_predictor import predict_scenario
    r = predict_scenario(asset="BTC",
                          strategy_expr="__import__('os')",
                          direction="LONG")
    assert "abandon" in r.recommended_action
    assert "unsafe" in r.sample_warning.lower()


def test_scenario_predict_returns_structure():
    from src.ai.scenario_predictor import predict_scenario
    r = predict_scenario(
        asset="BTC",
        strategy_expr="ema_8 > ema_21",
        direction="LONG",
    )
    d = r.to_dict()
    assert "expected_profit_per_trade_pct" in d
    assert "confidence_label" in d
    assert "recommended_action" in d
    assert d["recommended_action"] in ("run", "refine", "abandon")
    assert d["confidence_label"] in ("high", "medium", "low", "low_sample")


def test_scenario_calibration_multiplier():
    from src.ai.scenario_predictor import _calibration_multiplier
    assert _calibration_multiplier("over_confident") == 0.7
    assert _calibration_multiplier("well_calibrated") == 1.0
    assert _calibration_multiplier("under_confident") == 1.2
    assert _calibration_multiplier("neutral") == 1.0


def test_scenario_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "query_scenario_prediction" in set(r.list_names())


# ── Web Search ─────────────────────────────────────────────────────────


def test_web_search_empty_query():
    from src.ai.web_search import search
    r = search("")
    assert r.error == "empty_query"
    assert r.n_results == 0


def test_web_search_daily_cap_hit(monkeypatch, tmp_path):
    """Force counter past cap → should return cap_hit error."""
    monkeypatch.setattr(
        "src.ai.web_search.DAILY_COUNTER_PATH",
        str(tmp_path / "counter.json"),
    )
    monkeypatch.setenv("ACT_WEB_SEARCH_DAILY_CAP", "0")  # cap is 0
    monkeypatch.setattr(
        "src.ai.web_search._query_cache", {},
    )
    from src.ai.web_search import search
    r = search("test query")
    assert r.error and "daily_cap_hit" in r.error


def test_web_search_cache_hit(monkeypatch, tmp_path):
    """Same query twice should hit cache the second time."""
    monkeypatch.setattr(
        "src.ai.web_search.DAILY_COUNTER_PATH",
        str(tmp_path / "counter.json"),
    )
    cache = {}
    monkeypatch.setattr("src.ai.web_search._query_cache", cache)
    # Mock the actual fetch
    from src.ai.web_search import WebSearchResult
    monkeypatch.setattr(
        "src.ai.web_search._ddg_html_search",
        lambda q, max_results=5: [
            WebSearchResult(title="Test", url="https://x.com", snippet="t"),
        ],
    )
    from src.ai.web_search import search
    r1 = search("test query")
    r2 = search("test query")
    assert r1.cache_hit is False
    assert r2.cache_hit is True
    assert r2.n_results == 1


def test_web_search_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "query_web_search" in set(r.list_names())


def test_web_search_response_serializes():
    from src.ai.web_search import WebSearchResponse, WebSearchResult
    r = WebSearchResponse(
        query="test", n_results=2,
        results=[
            WebSearchResult(title="A", url="https://a.com", snippet="snip"),
            WebSearchResult(title="B", url="https://b.com", snippet="snip2"),
        ],
        daily_cap_remaining=49,
    )
    d = r.to_dict()
    assert d["query"] == "test"
    assert len(d["results"]) == 2
    assert d["daily_cap_remaining"] == 49
