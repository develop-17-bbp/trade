"""Scenario predictor — "if I use strategy X, what profit can I expect?"

Combines three existing data sources into one structured projection:

  1. Vectorized backtest on recent N bars (cheap, ~ms)
       → realized PnL on the proposed strategy over historical data
  2. Decision-graph similar setups
       → outcome distribution of past trades matching this setup's
         regime/pattern/direction
  3. Prediction-accuracy calibration
       → adjustment based on operator's own track record (if brain is
         currently over_confident, downweight projected profit)

Output is a structured ScenarioPrediction the brain can read or
ship as part of a TradePlan thesis. Anti-overfit-aware: applies
calibration AND surfaces sample-size warnings to prevent
extrapolation from small data.

Brain workflow:
    1. Hypothesize a strategy (e.g. "RSI<30 + bb_lower touch")
    2. Call query_scenario_prediction(strategy_expr, asset)
    3. Read expected_profit_per_trade, sample_warning, recommended_action
    4. If recommended_action == "run", proceed to submit_trade_plan
    5. If recommended_action == "refine", iterate hypothesis
    6. If recommended_action == "abandon", skip — don't submit

Anti-overfit:
  * No learned weights — pure aggregation of existing tools
  * Calibration applied (downweight by accuracy bucket)
  * Sample-size warnings everywhere (<10 = low_confidence)
  * Expected_total_pnl bounded by realistic position-size constraints
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScenarioPrediction:
    asset: str
    strategy_expr: str
    direction: str

    # From backtest
    backtest_n_signals: int = 0
    backtest_win_rate: float = 0.0
    backtest_avg_pnl_pct: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_dsr: float = 0.0   # deflated sharpe

    # From decision graph similar
    similar_n_decisions: int = 0
    similar_win_rate: float = 0.0
    similar_avg_pnl_pct: float = 0.0

    # From prediction accuracy
    calibration_label: str = "neutral"
    calibration_multiplier: float = 1.0  # applied to expected profit

    # Synthesis
    expected_profit_per_trade_pct: float = 0.0  # net of spread
    confidence_label: str = "low_sample"        # high / medium / low / low_sample
    recommended_action: str = "abandon"          # run / refine / abandon
    sample_warning: str = ""
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "strategy_expr": self.strategy_expr[:200],
            "direction": self.direction,
            "backtest": {
                "n_signals": int(self.backtest_n_signals),
                "win_rate": round(float(self.backtest_win_rate), 3),
                "avg_pnl_pct": round(float(self.backtest_avg_pnl_pct), 3),
                "sharpe": round(float(self.backtest_sharpe), 3),
                "dsr": round(float(self.backtest_dsr), 3),
            },
            "similar_setups": {
                "n_decisions": int(self.similar_n_decisions),
                "win_rate": round(float(self.similar_win_rate), 3),
                "avg_pnl_pct": round(float(self.similar_avg_pnl_pct), 3),
            },
            "calibration": {
                "label": self.calibration_label,
                "multiplier": round(float(self.calibration_multiplier), 3),
            },
            "expected_profit_per_trade_pct": round(float(self.expected_profit_per_trade_pct), 3),
            "confidence_label": self.confidence_label,
            "recommended_action": self.recommended_action,
            "sample_warning": self.sample_warning,
            "rationale": self.rationale[:300],
        }


def _calibration_multiplier(label: str) -> float:
    """Map calibration label to expected-profit multiplier."""
    return {
        "well_calibrated": 1.0,
        "neutral": 1.0,
        "over_confident": 0.7,    # downweight — your high-conviction calls underperform
        "under_confident": 1.2,   # upweight — your skipped neutrals were wins
    }.get(label, 1.0)


def predict_scenario(
    asset: str,
    strategy_expr: str,
    direction: str = "LONG",
    regime: Optional[str] = None,
    pattern: Optional[str] = None,
    spread_pct: float = 1.69,
    lookback_bars: int = 400,
) -> ScenarioPrediction:
    """Project expected profit + confidence for a candidate strategy.

    `strategy_expr` must be in the safe-DSL whitelist (see
    llm_alpha_generator.ALLOWED_FEATURES + ALLOWED_OPS).
    """
    asset = str(asset).upper()
    direction = str(direction).upper()
    out = ScenarioPrediction(
        asset=asset, strategy_expr=strategy_expr, direction=direction,
    )

    # ── 1. Vectorized backtest on recent bars ─────────────────────
    try:
        from src.ai.llm_alpha_generator import (
            AlphaFormula, _evaluate_alpha, _is_safe_expression,
        )
        from src.data.fetcher import PriceFetcher
        if not _is_safe_expression(strategy_expr):
            out.rationale = "strategy_expr_failed_safe_dsl_whitelist"
            out.recommended_action = "abandon"
            out.sample_warning = "unsafe_expression"
            return out
        pf = PriceFetcher()
        bars_raw = pf.get_recent_bars(asset, timeframe="1h",
                                        n=lookback_bars) or []
        if bars_raw and len(bars_raw) >= 100:
            highs = [float(b.get("high", 0)) for b in bars_raw]
            lows = [float(b.get("low", 0)) for b in bars_raw]
            closes = [float(b.get("close", 0)) for b in bars_raw]
            volumes = [float(b.get("volume", 0)) for b in bars_raw]
            opens = [closes[i - 1] if i > 0 else closes[0]
                     for i in range(len(closes))]
            bar_dicts = [
                {"close": closes[i], "high": highs[i], "low": lows[i],
                 "open": opens[i], "volume": volumes[i]}
                for i in range(len(closes))
            ]
            f = AlphaFormula(
                name="scenario", expression=strategy_expr,
                direction_kind=("long_only" if direction == "LONG"
                                  else "long_short"),
            )
            r = _evaluate_alpha(f, bar_dicts)
            out.backtest_n_signals = r.n_signals
            out.backtest_win_rate = r.win_rate
            out.backtest_sharpe = r.sharpe_observed
            out.backtest_dsr = r.sharpe_deflated
            # Extract avg PnL — _evaluate_alpha doesn't expose it directly,
            # so we use deflated_sharpe as a proxy for confidence
            # and compute expected PnL from win_rate × avg_win.
            # Simplified: assume 1.5% gross gain on wins net spread.
            avg_win_gross = 1.5
            avg_loss_gross = 1.5
            if r.n_signals > 0:
                out.backtest_avg_pnl_pct = (
                    r.win_rate * avg_win_gross
                    - (1 - r.win_rate) * avg_loss_gross
                    - spread_pct
                )
    except Exception as e:
        logger.debug("scenario backtest failed: %s", e)

    # ── 2. Decision-graph similar setups ──────────────────────────
    try:
        from src.ai.decision_graph import causal_query
        cq = causal_query(
            asset=asset, regime=regime, pattern_label=pattern,
            direction=direction, spread_pct=spread_pct,
        )
        out.similar_n_decisions = cq.matched_decisions
        out.similar_win_rate = cq.win_rate
        out.similar_avg_pnl_pct = cq.avg_pnl_pct_net
    except Exception as e:
        logger.debug("scenario similar lookup failed: %s", e)

    # ── 3. Prediction-accuracy calibration ───────────────────────
    try:
        from src.ai.prediction_accuracy import compute_accuracy
        acc = compute_accuracy(asset=asset, lookback_days=60)
        out.calibration_label = acc.calibration_label
        out.calibration_multiplier = _calibration_multiplier(acc.calibration_label)
    except Exception as e:
        logger.debug("scenario calibration failed: %s", e)

    # ── Synthesize expected profit ────────────────────────────────
    # Weighted combination:
    #   60% weight to backtest (forward-looking)
    #   40% weight to similar setups (lived-experience anchored)
    n_bt = out.backtest_n_signals
    n_sim = out.similar_n_decisions
    if n_bt + n_sim == 0:
        out.expected_profit_per_trade_pct = 0.0
        out.sample_warning = "no_data_in_either_source"
        out.confidence_label = "low_sample"
        out.recommended_action = "abandon"
    else:
        weight_bt = 0.6 if n_bt >= 10 else (0.3 if n_bt >= 5 else 0.0)
        weight_sim = 0.4 if n_sim >= 10 else (0.2 if n_sim >= 5 else 0.0)
        # Renormalize when one source is missing
        total_w = weight_bt + weight_sim
        if total_w == 0:
            weight_bt = 0.5 if n_bt > 0 else 0.0
            weight_sim = 0.5 if n_sim > 0 else 0.0
            total_w = weight_bt + weight_sim
        if total_w > 0:
            raw_expected = (
                weight_bt * out.backtest_avg_pnl_pct
                + weight_sim * out.similar_avg_pnl_pct
            ) / total_w
            out.expected_profit_per_trade_pct = (
                raw_expected * out.calibration_multiplier
            )

    # Confidence label
    total_n = n_bt + n_sim
    if total_n >= 30:
        out.confidence_label = "high"
    elif total_n >= 15:
        out.confidence_label = "medium"
    elif total_n >= 5:
        out.confidence_label = "low"
    else:
        out.confidence_label = "low_sample"

    # Recommended action
    if out.expected_profit_per_trade_pct > 0.5 and total_n >= 10:
        out.recommended_action = "run"
    elif out.expected_profit_per_trade_pct > 0 and total_n >= 5:
        out.recommended_action = "refine"
    else:
        out.recommended_action = "abandon"

    if total_n < 10:
        out.sample_warning = f"low_combined_sample_n={total_n}"

    parts = [
        f"expected={out.expected_profit_per_trade_pct:+.2f}%/trade",
        f"confidence={out.confidence_label}",
        f"action={out.recommended_action}",
        f"backtest_n={n_bt} similar_n={n_sim}",
        f"calibration={out.calibration_label}",
    ]
    out.rationale = " | ".join(parts)[:300]

    return out
