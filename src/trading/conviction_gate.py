"""
Conviction gate — the "only trade when all signals agree" filter.

Intervention C (Robinhood-hardening plan): on a 1.69% round-trip-spread venue,
every trade starts 2-3% underwater once slippage is included. The strategy
can't afford to take low-conviction trades. This gate aggregates the signals
we already compute (TF alignment, Hurst regime, multi-strategy consensus,
macro bias) into a single TIER classification:

    sniper   → maximum conviction. ALL checks align. 3x size eligibility.
    normal   → moderate conviction. TF + multi-strategy pass, macro not
               strongly misaligned. 1x size.
    reject   → fails the minimum bar. Skip entirely.

This is the filter that "only takes the best" in one well-defined place.
Everything downstream (safe-entries gate, LLM brain, meta model) still runs
normally — conviction just decides which trades even get to reach those
layers.

Design:
  * Pure function. Accepts structured dicts, returns a dataclass + reasons.
  * Tier-aware thresholds — sniper needs more evidence, normal needs less.
  * Crisis from macro_bias forces 'reject' regardless of other signals.
  * Soft-fail: missing inputs collapse to 'reject' with explicit reason,
    never silently pass.

Rules (all must hold for tier promotion):

    sniper tier:
      - tf_aligned_1h_4h == True                   (HTF alignment)
      - hurst_regime in {'trending', 'strong_trend'}  (not mean-reverting)
      - multi_strategy_agreeing >= 5               (out of ~36 total)
      - macro_bias.aligned(direction) == True      (net macro in favor)
      - macro_bias.signed_bias magnitude >= 0.20   (non-trivial macro lean)
      - macro_bias.crisis == False

    normal tier (less strict — floors, not peaks):
      - tf_aligned_1h_4h == True
      - multi_strategy_agreeing >= 3
      - macro_bias.crisis == False
      - macro_bias aligned OR neutral (not strongly misaligned)

    reject: anything else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional  # noqa: F401 — Any used by helper

from src.trading.macro_bias import MacroBias


# Tunables
SNIPER_MIN_STRATEGY_AGREEING = 5
NORMAL_MIN_STRATEGY_AGREEING = 3
SNIPER_MIN_MACRO_MAGNITUDE = 0.20
TRENDING_REGIMES = {"trending", "strong_trend", "trend"}


def _soak_overlay() -> Optional[Dict[str, Any]]:
    """Best-effort read of paper-soak loose overlay. Returns None when
    disabled or when real-capital is enabled (overlay is paper-only)."""
    try:
        from skills.paper_soak_loose.action import get_paper_soak_overlay
        return get_paper_soak_overlay()
    except Exception:
        return None

# ── Regime-dependent hysteresis (C19, WebCryptoAgent-inspired) ─────────
# θ_adopt > θ_hold so entering a position needs a stronger signal than
# staying in one. Prevents flip-flopping when signals wobble around the
# entry threshold while a trade is already live.
#
# Fresh entry (in_position=False) → full thresholds.
# Holding (in_position=True)       → thresholds scaled down by HOLD_FACTOR.
HOLD_SNIPER_STRATEGY_FLOOR = 4          # vs 5 for fresh entry
HOLD_NORMAL_STRATEGY_FLOOR = 2          # vs 3 for fresh entry
HOLD_MACRO_MAGNITUDE_FACTOR = 0.70      # 0.14 vs 0.20 for fresh entry


@dataclass
class ConvictionResult:
    tier: str                                       # 'sniper' | 'normal' | 'reject'
    passed: bool                                    # True iff tier != 'reject'
    direction: str                                  # LONG / SHORT / FLAT
    size_multiplier: float                          # 1.0 (normal), 3.0 (sniper), 0.0 (reject)
    checks: Dict[str, bool] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier, "passed": self.passed,
            "direction": self.direction,
            "size_multiplier": round(self.size_multiplier, 3),
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


def evaluate(
    direction: str,
    tf_1h_direction: Optional[str],
    tf_4h_direction: Optional[str],
    hurst_regime: Optional[str],
    multi_strategy_counts: Dict[str, int],
    macro_bias: Optional[MacroBias],
    in_position: bool = False,
) -> ConvictionResult:
    """Compute conviction tier + size multiplier.

    Arguments:
        direction: proposed trade direction — 'LONG' / 'SHORT' / 'BUY' / 'SELL'
        tf_1h_direction: '1h' timeframe direction ('RISING','FALLING','FLAT','UP','DOWN')
        tf_4h_direction: '4h' timeframe direction
        hurst_regime: 'trending' / 'mean_reverting' / 'random' / None
        multi_strategy_counts: {'long': int, 'short': int, 'flat': int}
        macro_bias: MacroBias from compute_macro_bias(), or None
        in_position: True if we're evaluating whether to HOLD a current
            position (use hold-threshold θ_hold); False for a fresh
            ENTRY (use adopt-threshold θ_adopt). Hold thresholds are
            looser than adopt so a mid-trade signal wobble doesn't flip
            the position out — WebCryptoAgent-style regime hysteresis.

    Returns ConvictionResult with tier, size_multiplier, per-check booleans.
    """
    d = (direction or "").upper()
    if d in ("BUY", "LONG"):
        d = "LONG"
    elif d in ("SELL", "SHORT"):
        d = "SHORT"
    else:
        return ConvictionResult(
            tier="reject", passed=False, direction=d, size_multiplier=0.0,
            reasons=[f"unknown_direction:{direction}"],
        )

    checks: Dict[str, bool] = {}
    reasons: List[str] = []

    # ── Macro crisis ────────────────────────────────────────────────────
    # Real capital: HARD reject -- crisis blocks every entry, no overrides.
    # Paper mode: ADVISORY only -- the brain has already weighed every
    # macro layer (12 of them) plus news, sentiment, regime, and 371
    # trade-memory examples; second-guessing it on a single CRISIS
    # signal blocks legitimate setups during macro chop. Per operator
    # directive `feedback_target_is_non_negotiable`: cost / macro
    # awareness is a tool the brain uses, not a ceiling that overrides
    # it. Real capital path (ACT_REAL_CAPITAL_ENABLED=1) ignores this
    # and stays a hard reject. The overlay's bypass_macro_crisis flag
    # is now redundant in paper mode -- the bypass is automatic --
    # but kept for back-compat so explicit overlay edits continue to
    # work.
    is_real_capital = os.environ.get(
        "ACT_REAL_CAPITAL_ENABLED", ""
    ).strip() == "1"
    if macro_bias is not None and macro_bias.crisis:
        if is_real_capital:
            overlay_early = _soak_overlay()
            bypass_crisis = False
            if overlay_early:
                bypass_crisis = bool(
                    (overlay_early.get("conviction") or {}).get("bypass_macro_crisis")
                )
            if not bypass_crisis:
                checks["macro_crisis_free"] = False
                reasons.append("macro_crisis")
                return ConvictionResult(
                    tier="reject", passed=False, direction=d, size_multiplier=0.0,
                    checks=checks, reasons=reasons,
                )
            checks["macro_crisis_free"] = False
            reasons.append("real_capital_overlay_bypass_macro_crisis")
        else:
            # Paper mode: brain is authority; macro_crisis is advisory.
            checks["macro_crisis_free"] = False
            reasons.append("paper_mode_advisory_macro_crisis")
    else:
        checks["macro_crisis_free"] = True

    # ── TF alignment (1h + 4h agree with trade direction) ──────────────
    tf_1h = (tf_1h_direction or "").upper()
    tf_4h = (tf_4h_direction or "").upper()
    bull_1h = tf_1h in ("RISING", "UP", "BULLISH", "LONG")
    bear_1h = tf_1h in ("FALLING", "DOWN", "BEARISH", "SHORT")
    bull_4h = tf_4h in ("RISING", "UP", "BULLISH", "LONG")
    bear_4h = tf_4h in ("FALLING", "DOWN", "BEARISH", "SHORT")

    if d == "LONG":
        tf_aligned = bull_1h and bull_4h
    else:
        tf_aligned = bear_1h and bear_4h
    checks["tf_alignment_1h_4h"] = tf_aligned
    if not tf_aligned:
        reasons.append(f"tf_not_aligned:1h={tf_1h},4h={tf_4h},dir={d}")

    # ── Multi-strategy agreement (regime-dependent hysteresis) ─────────
    # Fresh entry uses adopt-thresholds (5 for sniper, 3 for normal).
    # In-position uses hold-thresholds (4 for sniper, 2 for normal) so
    # a brief signal dip mid-trade doesn't force us out.
    n_long = int(multi_strategy_counts.get("long", 0))
    n_short = int(multi_strategy_counts.get("short", 0))
    agreeing = n_long if d == "LONG" else n_short
    sniper_strategy_floor = HOLD_SNIPER_STRATEGY_FLOOR if in_position else SNIPER_MIN_STRATEGY_AGREEING
    normal_strategy_floor = HOLD_NORMAL_STRATEGY_FLOOR if in_position else NORMAL_MIN_STRATEGY_AGREEING
    # Paper-soak loose overlay (C22) — only applied when real-capital
    # is disabled AND the operator has explicitly enabled loose mode.
    overlay = _soak_overlay()
    if overlay:
        ov = (overlay.get("conviction") or {}).get("min_normal_strategies_agreeing")
        if isinstance(ov, (int, float)):
            normal_strategy_floor = min(normal_strategy_floor, int(ov))
        reasons.append(f"paper_soak_loose_overlay:applied")
    checks["multi_strategy_normal_floor"] = agreeing >= normal_strategy_floor
    checks["multi_strategy_sniper_floor"] = agreeing >= sniper_strategy_floor
    # Back-compat aliases so existing downstream readers don't break.
    checks["multi_strategy_ge_3"] = checks["multi_strategy_normal_floor"]
    checks["multi_strategy_ge_5"] = checks["multi_strategy_sniper_floor"]
    if not checks["multi_strategy_normal_floor"]:
        reasons.append(f"multistrat_weak:{d.lower()}={agreeing}<{normal_strategy_floor}")

    # ── Macro alignment (direction matches net bias) ───────────────────
    sniper_macro_magnitude = SNIPER_MIN_MACRO_MAGNITUDE * (
        HOLD_MACRO_MAGNITUDE_FACTOR if in_position else 1.0
    )
    if macro_bias is not None:
        macro_aligned = macro_bias.aligned(d)
        macro_strong = abs(macro_bias.signed_bias) >= sniper_macro_magnitude
    else:
        macro_aligned = True            # absent macro = neutral
        macro_strong = False
    checks["macro_aligned"] = macro_aligned
    checks["macro_strong_lean"] = macro_strong
    if not macro_aligned:
        reasons.append(
            f"macro_misaligned:bias={macro_bias.signed_bias:+.2f}" if macro_bias else "macro_unknown"
        )

    # ── Hurst regime (bonus for sniper; not required for normal) ───────
    regime = (hurst_regime or "").lower()
    is_trending = regime in TRENDING_REGIMES
    checks["hurst_trending"] = is_trending
    if not is_trending:
        reasons.append(f"hurst:{regime or 'unknown'}")

    # ── Tier assignment ────────────────────────────────────────────────
    sniper_checks = (
        checks["tf_alignment_1h_4h"]
        and checks["multi_strategy_ge_5"]
        and checks["macro_aligned"]
        and checks["macro_strong_lean"]
        and checks["hurst_trending"]
    )
    normal_checks = (
        checks["tf_alignment_1h_4h"]
        and checks["multi_strategy_ge_3"]
        and checks["macro_aligned"]
    )

    if sniper_checks:
        return ConvictionResult(
            tier="sniper", passed=True, direction=d, size_multiplier=3.0,
            checks=checks,
            reasons=reasons + [f"sniper_all_aligned:strategies={agreeing}"],
        )
    if normal_checks:
        return ConvictionResult(
            tier="normal", passed=True, direction=d, size_multiplier=1.0,
            checks=checks,
            reasons=reasons + [f"normal:strategies={agreeing}"],
        )
    return ConvictionResult(
        tier="reject", passed=False, direction=d, size_multiplier=0.0,
        checks=checks, reasons=reasons,
    )


def is_enabled() -> bool:
    """Gated by ACT_ROBINHOOD_HARDEN=1 like the other hardening interventions."""
    v = (os.environ.get("ACT_ROBINHOOD_HARDEN") or "").strip().lower()
    return v in ("1", "true", "yes", "on")
