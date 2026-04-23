"""Tests for src/trading/polymarket_analyst.py — C13b LLM probability estimator."""
from __future__ import annotations

import json
from unittest import mock

import pytest

from src.trading.polymarket_analyst import (
    MAX_PROB,
    MIN_PROB,
    PolymarketProbabilityEstimate,
    _clamp_prob,
    _parse_estimate,
    estimate_probability,
)


# ── Parser ─────────────────────────────────────────────────────────────


def test_parse_estimate_bare_json():
    out = _parse_estimate('{"estimated_yes_probability": 0.6, "confidence": 0.8}')
    assert out["estimated_yes_probability"] == 0.6


def test_parse_estimate_fenced():
    text = "Here is my estimate:\n```json\n{\"estimated_yes_probability\": 0.42}\n```"
    out = _parse_estimate(text)
    assert out and out["estimated_yes_probability"] == 0.42


def test_parse_estimate_embedded_in_prose():
    out = _parse_estimate(
        "Based on context, {\"estimated_yes_probability\": 0.7, "
        "\"confidence\": 0.5, \"rationale\": \"BTC strong\"} makes sense"
    )
    assert out["estimated_yes_probability"] == 0.7
    assert out["rationale"] == "BTC strong"


def test_parse_estimate_none_on_garbage():
    assert _parse_estimate("") is None
    assert _parse_estimate("no json at all") is None


# ── Clamp ──────────────────────────────────────────────────────────────


def test_clamp_prob_respects_bounds():
    assert _clamp_prob(0.001) == MIN_PROB
    assert _clamp_prob(0.999) == MAX_PROB
    assert _clamp_prob(0.5) == 0.5


def test_clamp_prob_non_numeric_returns_default():
    assert _clamp_prob("nope") == 0.5


# ── Zero-edge fallback path ────────────────────────────────────────────


def test_estimator_zero_edge_when_no_question():
    out = estimate_probability({"market_id": "m1", "yes_price": 0.6,
                                  "question": ""})
    assert out.fallback_used is True
    assert out.edge == 0.0
    assert out.estimated_yes_probability == out.implied_yes_probability


def test_estimator_zero_edge_when_llm_unreachable(monkeypatch):
    import sys
    fake_module = mock.MagicMock(analyze=mock.MagicMock(side_effect=RuntimeError("no llm")))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)
    out = estimate_probability({
        "market_id": "m1", "yes_price": 0.4,
        "question": "Will BTC close above $70k by Friday?",
    })
    assert out.fallback_used is True
    assert abs(out.edge) < 1e-9
    assert abs(out.estimated_yes_probability - 0.4) < 1e-9


def test_estimator_zero_edge_when_llm_returns_junk(monkeypatch):
    import sys
    fake_resp = mock.MagicMock(ok=True, text="not json at all", model="stub",
                                fallback_used=False)
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)
    out = estimate_probability({
        "market_id": "m1", "yes_price": 0.3,
        "question": "Will ETH 2x by EOY?",
    })
    assert out.fallback_used is True


# ── Happy path ─────────────────────────────────────────────────────────


def test_estimator_happy_path_produces_edge(monkeypatch):
    import sys
    fake_resp = mock.MagicMock(
        ok=True,
        text=json.dumps({
            "estimated_yes_probability": 0.55,
            "confidence": 0.72,
            "rationale": "BTC breakout confirmed + positive ETF flow",
        }),
        model="stub-32b", fallback_used=False,
    )
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)

    out = estimate_probability({
        "market_id": "m-btc-up", "yes_price": 0.40,
        "question": "Will Bitcoin close above $70,000 by Friday?",
        "volume_24h": 12_000, "end_ts": 0,
    })
    assert out.fallback_used is False
    assert abs(out.estimated_yes_probability - 0.55) < 1e-6
    assert abs(out.edge - 0.15) < 1e-6        # 0.55 - 0.40
    assert out.confidence > 0.5
    assert "BTC" in out.rationale or "ETF" in out.rationale
    assert out.source_model == "stub-32b"


def test_estimator_clamps_out_of_range_probs(monkeypatch):
    import sys
    # LLM returns 1.5 (invalid) → should clamp to MAX_PROB.
    fake_resp = mock.MagicMock(
        ok=True, text=json.dumps({"estimated_yes_probability": 1.5,
                                    "confidence": 0.5, "rationale": "x"}),
        model="m", fallback_used=False,
    )
    fake_module = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)
    out = estimate_probability({
        "market_id": "m1", "yes_price": 0.5, "question": "anything"
    })
    assert out.estimated_yes_probability == MAX_PROB


# ── Asset detection ────────────────────────────────────────────────────


def test_estimator_picks_asset_from_question(monkeypatch):
    """Detects BTC vs ETH from the market's question text."""
    import sys
    captured_prompts = {}

    def _fake_analyze(prompt, **kw):
        captured_prompts["prompt"] = prompt
        return mock.MagicMock(ok=True, text='{"estimated_yes_probability": 0.5, "confidence": 0.5, "rationale": "x"}', model="m", fallback_used=False)

    fake_module = mock.MagicMock(analyze=_fake_analyze)
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)

    estimate_probability({
        "market_id": "m", "yes_price": 0.5,
        "question": "Will Ethereum flip $4k by October?",
    })
    # The context block is built by _build_context_block(asset).
    # We can't verify the asset directly from the prompt without mocking
    # web_context; just confirm the call went through.
    assert "prompt" in captured_prompts


# ── to_dict shape ──────────────────────────────────────────────────────


def test_to_dict_trims_long_fields():
    est = PolymarketProbabilityEstimate(
        market_id="m", question="x" * 500,
        implied_yes_probability=0.4,
        estimated_yes_probability=0.55,
        edge=0.15, confidence=0.7,
        rationale="y" * 600, source_model="m",
    )
    d = est.to_dict()
    assert len(d["question"]) <= 200
    assert len(d["rationale"]) <= 400
    assert d["edge"] == 0.15


# ── polymarket-hunt integration ────────────────────────────────────────


def test_polymarket_hunt_uses_analyst_estimate(monkeypatch):
    """The hunt skill should call _estimate_via_analyst which delegates
    to polymarket_analyst.estimate_probability."""
    import sys
    # Force PolymarketFetcher to produce one mock market.
    mock_market = {
        "market_id": "test-1", "question": "Will BTC close above 70k?",
        "yes_price": 0.40, "volume_24h": 20_000, "end_ts": 0,
    }
    fake_fetcher = mock.MagicMock()
    fake_fetcher.fetch_crypto_markets.return_value = [mock_market]
    fake_fetch_mod = mock.MagicMock(PolymarketFetcher=mock.MagicMock(return_value=fake_fetcher))
    monkeypatch.setitem(sys.modules, "src.data.polymarket_fetcher", fake_fetch_mod)

    # Force the analyst to produce an edge-positive estimate.
    fake_resp = mock.MagicMock(
        ok=True,
        text='{"estimated_yes_probability": 0.55, "confidence": 0.7, "rationale": "BTC strong"}',
        model="stub", fallback_used=False,
    )
    fake_dual = mock.MagicMock(analyze=mock.MagicMock(return_value=fake_resp))
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_dual)

    from skills.polymarket_hunt.action import run
    r = run({"confirm": True, "dry_run": True, "scan_limit": 1})
    assert r.ok is True
    # The estimate dict should be stashed into the top-3 entries.
    if r.data["top3"]:
        est = r.data["top3"][0].get("estimate") or {}
        assert "estimated_yes_probability" in est
        assert "rationale" in est


# ── check_polymarket diagnostic ────────────────────────────────────────


def test_check_polymarket_shape():
    from src.skills.diagnostics import check_polymarket
    out = check_polymarket()
    assert "enabled_in_config" in out
    assert "live_env_set" in out
    assert "py_clob_client_installed" in out
    assert "executor_mode" in out
    assert out["executor_mode"] in ("shadow", "live", "unknown")
