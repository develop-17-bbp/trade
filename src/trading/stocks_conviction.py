"""Stocks-tuned conviction gate.

Faithful copy of `conviction_gate.py`'s tier logic with coefficients
retuned for Alpaca's economics (~0.01% spread vs Robinhood crypto's
1.69%). The 169× spread reduction collapses the required edge by the
same factor; sniper threshold relaxes proportionally; trade rate
goes up roughly 10× post-recalibration.

Phase L caveat: during paper soak the bot reads IEX market data
(~2-8% of consolidated volume), so fill skew vs full SIP runs
~0.05-0.15%. The base `min_expected_move_pct` of 0.15 is for SIP;
the IEX bump is layered in via `ACT_STOCKS_DATA_FEED=iex` (default).

Operator basket has two risk classes:
  * Index ETFs (SPY, QQQ): full normal/sniper tiers, 15% intraday cap.
  * Leveraged ETFs (TQQQ, SOXL): tighter — 5% intraday cap (from
    SymbolMeta), no overnight (forced flat-by-EOD by authority overlay).

Inputs match conviction_gate.evaluate but include a `symbol` so we
can read the SymbolMeta and adjust per-symbol caps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Tunables ────────────────────────────────────────────────────────
# Calibrated to Alpaca SIP. IEX adjustment layered in below.
SIP_MIN_EXPECTED_MOVE_PCT = float(os.getenv("ACT_STOCKS_MIN_MOVE_SIP", "0.15"))
IEX_FILL_SKEW_PCT         = float(os.getenv("ACT_STOCKS_IEX_SKEW",    "0.10"))
SNIPER_MIN_STRATEGY_AGREEING = 4   # vs 5 for crypto — relaxed
NORMAL_MIN_STRATEGY_AGREEING = 2   # vs 3 for crypto

# Intraday position caps. Leveraged ETFs override via SymbolMeta.
DEFAULT_INTRADAY_PCT_MAX = 15.0
DEFAULT_OVERNIGHT_PCT_MAX = 5.0


@dataclass
class StocksConvictionResult:
    tier: str                                       # 'sniper' | 'normal' | 'reject'
    passed: bool
    direction: str                                  # LONG / SHORT / FLAT
    size_multiplier: float                          # 1.0 normal, 3.0 sniper, 0.0 reject
    intraday_pct_cap: float
    overnight_pct_cap: float
    min_expected_move_pct: float                    # sniper threshold actually applied
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier, "passed": self.passed,
            "direction": self.direction,
            "size_multiplier": round(self.size_multiplier, 3),
            "intraday_pct_cap": round(self.intraday_pct_cap, 3),
            "overnight_pct_cap": round(self.overnight_pct_cap, 3),
            "min_expected_move_pct": round(self.min_expected_move_pct, 4),
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


def _effective_min_move() -> float:
    """SIP threshold + IEX-fill-skew adder if running on the free tier."""
    feed = (os.getenv("ACT_ALPACA_DATA_FEED") or "iex").lower()
    if feed == "iex":
        return SIP_MIN_EXPECTED_MOVE_PCT + IEX_FILL_SKEW_PCT
    return SIP_MIN_EXPECTED_MOVE_PCT


def evaluate(
    *,
    symbol: str,
    direction: str,
    expected_move_pct: float,
    tf_5m_direction: Optional[str] = None,
    tf_15m_direction: Optional[str] = None,
    tf_1h_direction: Optional[str] = None,
    multi_strategy_counts: Optional[Dict[str, int]] = None,
    in_position: bool = False,
) -> StocksConvictionResult:
    """Compute stocks conviction tier + size multiplier + caps.

    Args mirror conviction_gate.evaluate where they overlap, with two
    additions:
      * `symbol` so we can resolve SymbolMeta + leveraged-ETF caps.
      * `expected_move_pct` is required — the analyst LLM emits this
        as part of every TradePlan; on stocks we hard-gate against
        the IEX-aware SIP threshold.

    Returns StocksConvictionResult with tier, size_multiplier, caps,
    and the actual min_expected_move_pct that was applied (so
    /diagnose-noop can show "rejected because move 0.12% < 0.25% IEX
    threshold" not just "rejected").
    """
    from src.models.asset_class import classify

    d = (direction or "").upper()
    if d in ("BUY", "LONG"):
        d = "LONG"
    elif d in ("SELL", "SHORT"):
        d = "SHORT"
    else:
        return StocksConvictionResult(
            tier="reject", passed=False, direction=d, size_multiplier=0.0,
            intraday_pct_cap=0.0, overnight_pct_cap=0.0,
            min_expected_move_pct=_effective_min_move(),
            reasons=[f"unknown_direction:{direction}"],
        )

    meta = classify(symbol, venue_hint="alpaca")
    intraday_cap = meta.intraday_pct_max(default=DEFAULT_INTRADAY_PCT_MAX)
    overnight_cap = meta.overnight_pct_max(default=DEFAULT_OVERNIGHT_PCT_MAX)
    min_move = _effective_min_move()

    counts = multi_strategy_counts or {}
    n_long = int(counts.get("long", 0))
    n_short = int(counts.get("short", 0))
    agreeing = n_long if d == "LONG" else n_short

    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    # ── Stocks-class blocker: not even a stock ──
    if not meta.asset_class.is_stock():
        reasons.append(f"not_a_stock:{meta.asset_class}")
        return StocksConvictionResult(
            tier="reject", passed=False, direction=d, size_multiplier=0.0,
            intraday_pct_cap=intraday_cap, overnight_pct_cap=overnight_cap,
            min_expected_move_pct=min_move, checks=checks, reasons=reasons,
        )

    # ── Expected-move floor ──
    move_ok = expected_move_pct >= min_move
    checks["expected_move_floor"] = move_ok
    if not move_ok:
        reasons.append(
            f"expected_move {expected_move_pct:.3f}% < min {min_move:.3f}%"
        )

    # ── Multi-strategy agreement (looser thresholds than crypto) ──
    sniper_agree = agreeing >= SNIPER_MIN_STRATEGY_AGREEING
    normal_agree = agreeing >= NORMAL_MIN_STRATEGY_AGREEING
    checks["multi_strategy_sniper"] = sniper_agree
    checks["multi_strategy_normal"] = normal_agree
    if not normal_agree:
        reasons.append(f"multistrat_weak:{d.lower()}={agreeing}<{NORMAL_MIN_STRATEGY_AGREEING}")

    # ── TF alignment (any 2-of-3 timeframes) ──
    bull_tags = ("RISING", "UP", "BULLISH", "LONG")
    bear_tags = ("FALLING", "DOWN", "BEARISH", "SHORT")

    def _aligned(tf: Optional[str]) -> bool:
        if tf is None:
            return False
        u = tf.upper()
        return u in bull_tags if d == "LONG" else u in bear_tags

    tf_aligned_count = sum(_aligned(t) for t in (tf_5m_direction, tf_15m_direction, tf_1h_direction))
    checks["tf_aligned_2_of_3"] = tf_aligned_count >= 2
    checks["tf_aligned_3_of_3"] = tf_aligned_count >= 3
    if tf_aligned_count < 2:
        reasons.append(
            f"tf_misaligned:5m={tf_5m_direction},15m={tf_15m_direction},1h={tf_1h_direction}"
        )

    # ── Tier assignment ──
    sniper_checks = (
        checks.get("expected_move_floor", False)
        and checks.get("multi_strategy_sniper", False)
        and checks.get("tf_aligned_3_of_3", False)
    )
    normal_checks = (
        checks.get("expected_move_floor", False)
        and checks.get("multi_strategy_normal", False)
        and checks.get("tf_aligned_2_of_3", False)
    )

    if sniper_checks:
        return StocksConvictionResult(
            tier="sniper", passed=True, direction=d, size_multiplier=3.0,
            intraday_pct_cap=intraday_cap, overnight_pct_cap=overnight_cap,
            min_expected_move_pct=min_move, checks=checks,
            reasons=reasons + [f"sniper:tf=3/3,strat={agreeing}"],
        )
    if normal_checks:
        return StocksConvictionResult(
            tier="normal", passed=True, direction=d, size_multiplier=1.0,
            intraday_pct_cap=intraday_cap, overnight_pct_cap=overnight_cap,
            min_expected_move_pct=min_move, checks=checks,
            reasons=reasons + [f"normal:tf=2/3,strat={agreeing}"],
        )

    return StocksConvictionResult(
        tier="reject", passed=False, direction=d, size_multiplier=0.0,
        intraday_pct_cap=intraday_cap, overnight_pct_cap=overnight_cap,
        min_expected_move_pct=min_move, checks=checks, reasons=reasons,
    )
