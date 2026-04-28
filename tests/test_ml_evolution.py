"""Tests for the four 2026 ML additions:
  - LLM alpha generator (5 guards enforced)
  - Foundation forecaster (Chronos + fallback)
  - TFT forecaster (with quantile fallback)
  - PPO RL scaffolding (with rule-based fallback)
+ profit_extraction_targets tool
"""
from __future__ import annotations

import pytest


# ── LLM Alpha Generator ─────────────────────────────────────────────────


def test_alpha_generator_safe_dsl_rejects_unsafe():
    from src.ai.llm_alpha_generator import _is_safe_expression
    assert _is_safe_expression("ema_8 > ema_21")
    assert _is_safe_expression("rsi_14 < 30 and close > ema_50")
    # Unsafe: imports
    assert not _is_safe_expression("__import__('os').system('ls')")
    # Unsafe: undefined feature
    assert not _is_safe_expression("magic_indicator > 0")


def test_alpha_generator_seed_library_present():
    from src.ai.llm_alpha_generator import list_seed_alphas
    seeds = list_seed_alphas()
    assert len(seeds) >= 5
    for s in seeds:
        assert "name" in s
        assert "expression" in s
        assert "direction_kind" in s


def test_alpha_generator_guard_constants_set():
    from src.ai.llm_alpha_generator import (
        DAILY_GENERATION_CAP, MAX_ACTIVE_LLM_ALPHAS,
        QUARANTINE_NEG_SHARPE_RUN, BATCH_PBO_REJECT_THRESHOLD,
    )
    assert DAILY_GENERATION_CAP == 1
    assert MAX_ACTIVE_LLM_ALPHAS == 5
    assert QUARANTINE_NEG_SHARPE_RUN == 5
    assert BATCH_PBO_REJECT_THRESHOLD == 0.5


def test_alpha_generator_evaluation_handles_short_history():
    from src.ai.llm_alpha_generator import AlphaFormula, _evaluate_alpha
    f = AlphaFormula(name="t", expression="ema_8 > ema_21",
                     direction_kind="long_only")
    # Only 10 bars — too short
    bars = [{"close": 100 + i, "high": 101 + i, "low": 99 + i,
             "open": 100 + i, "volume": 1000} for i in range(10)]
    r = _evaluate_alpha(f, bars)
    assert "insufficient" in r.sample_warning.lower() or r.n_signals == 0


def test_alpha_generator_parse_llm_output_filters_unsafe():
    from src.ai.llm_alpha_generator import parse_llm_generated_alphas
    text = '[{"name": "good", "expression": "ema_8 > ema_21"},' \
           '{"name": "bad", "expression": "__import__(\'os\')"}]'
    alphas = parse_llm_generated_alphas(text)
    # Only the safe one should pass
    assert len(alphas) == 1
    assert alphas[0].name == "good"


# ── Foundation Forecaster ───────────────────────────────────────────────


def test_foundation_forecaster_fallback_with_short_history():
    from src.ai.foundation_forecaster import forecast
    r = forecast("BTC", [100.0] * 10)
    assert r.method == "fallback_linear"
    assert r.confidence == 0.0


def test_foundation_forecaster_fallback_with_history():
    """Fallback should produce point + quantile forecasts when given
    sufficient history."""
    import math
    closes = [100 + 5 * math.sin(i / 10) + i * 0.1 for i in range(60)]
    from src.ai.foundation_forecaster import forecast
    r = forecast("BTC", closes, horizons=[1, 4])
    assert r.method == "fallback_linear"  # no chronos lib in test env
    assert 1 in r.horizon_forecasts
    assert 4 in r.horizon_forecasts
    assert "p50" in r.horizon_quantiles[1]


def test_foundation_forecaster_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("ACT_FOUNDATION_FORECAST", raising=False)
    from src.ai.foundation_forecaster import is_enabled
    assert is_enabled() is False


# ── TFT Forecaster ──────────────────────────────────────────────────────


def test_tft_forecaster_returns_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("ACT_TFT_FORECAST", raising=False)
    from src.ai.tft_forecaster import forecast
    r = forecast("BTC", [100.0] * 60)
    assert r.method == "tft_disabled"


def test_tft_forecaster_fallback_when_enabled_no_checkpoint(monkeypatch):
    monkeypatch.setenv("ACT_TFT_FORECAST", "1")
    import math
    closes = [100 + math.sin(i / 8) for i in range(60)]
    from src.ai.tft_forecaster import forecast
    r = forecast("BTC", closes, horizons=[1, 4])
    # No checkpoint → fallback
    assert r.method == "fallback_quantile_linear"
    assert "p10" in r.quantiles[1]


# ── PPO Agent ──────────────────────────────────────────────────────────


def test_ppo_disabled_when_env_unset(monkeypatch):
    monkeypatch.delenv("ACT_PPO_AGENT", raising=False)
    from src.ai.ppo_agent import infer_action, is_enabled, is_authoritative
    assert is_enabled() is False
    assert is_authoritative() is False
    r = infer_action([0.5, 0.3, 0.02, 0.5, 1])
    assert r.method == "disabled"
    assert r.action == "SKIP"


def test_ppo_shadow_mode(monkeypatch):
    monkeypatch.setenv("ACT_PPO_AGENT", "shadow")
    from src.ai.ppo_agent import is_enabled, is_authoritative
    assert is_enabled() is True
    assert is_authoritative() is False


def test_ppo_authoritative_one(monkeypatch):
    monkeypatch.setenv("ACT_PPO_AGENT", "1")
    from src.ai.ppo_agent import is_enabled, is_authoritative
    assert is_enabled() is True
    assert is_authoritative() is True


def test_ppo_reward_clip_bounded():
    from src.ai.ppo_agent import clip_reward, REWARD_CLIP_PCT
    # Outlier wins clipped
    assert clip_reward(50.0) == REWARD_CLIP_PCT
    assert clip_reward(-50.0) == -REWARD_CLIP_PCT
    # Within bounds preserved
    assert clip_reward(2.5) == 2.5
    assert clip_reward(-2.5) == -2.5


def test_ppo_fallback_proxy_returns_action(monkeypatch):
    monkeypatch.setenv("ACT_PPO_AGENT", "1")
    from src.ai.ppo_agent import infer_action, PPO_ACTIONS
    r = infer_action([0.7, 0.3, 0.01, 0.5, 0])
    # Strong trend + low vol → ENTER_LONG_SNIPER
    assert r.action in PPO_ACTIONS
    assert r.method == "fallback_qlearning_proxy"
    assert r.confidence >= 0.0


# ── Tool registration ──────────────────────────────────────────────────


def test_6_ml_evolution_tools_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    names = set(r.list_names())
    expected = {
        "query_alpha_seeds",
        "evaluate_alphas",
        "query_foundation_forecast",
        "query_tft_forecast",
        "query_ppo_action",
        "query_profit_extraction_targets",
    }
    missing = expected - names
    assert not missing, f"missing: {missing}"


def test_alpha_seeds_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_alpha_seeds", {})
    # Result is a digest string; verify it contains expected token
    s = str(res)
    assert "seed_alphas" in s
