"""
Macro-bias aggregator — turns the 12 economic-intelligence layers into a single
signed bias and a size multiplier for the trade decision path.

Why this exists:
    EconomicIntelligence.get_macro_summary() produces a rich dict (composite
    BULLISH/BEARISH/CRISIS, bullish_count, bearish_count, usd_regime, macro_risk
    0-100, pre_event_flag, top_risks / top_tailwinds). The existing executor
    never consumes it into the entry decision — it just formats it as context
    for the LLM prompt. That means 12 layers of real-time macro telemetry have
    zero effect on whether we trade or how big.

Intervention B (Robinhood-hardening plan):
    Convert that summary into a numeric signed bias and a position-size
    multiplier the executor can multiply into the final notional.

Contract:
    compute_macro_bias(summary: dict) -> MacroBias
        .signed_bias         float in [-1.0, +1.0]   (+ = bullish, − = bearish)
        .crisis              bool                     (central_bank=CRISIS or macro_risk > 80)
        .size_multiplier     float in [0.0, 1.5]     (0 if crisis; 0.5-1.5 else)
        .aligned(direction)  bool                     (bias sign matches trade dir)
        .reasons             list[str]                (what drove the bias)

Design principles:
  * Soft-fail: missing keys, None inputs, or empty layers -> neutral bias (0.0).
  * No side effects. Pure function over a dict.
  * Size multiplier is bounded: never amplifies beyond 1.5x (keeps safe-entries
    risk budget intact).
  * Crisis overrides everything: size=0 means the executor should skip the
    trade entirely (honors the authority-rules "no new entries during macro
    crisis" intent).

Usage:
    from src.data.economic_intelligence import EconomicIntelligence
    from src.trading.macro_bias import compute_macro_bias

    ei = EconomicIntelligence()
    summary = ei.get_macro_summary()
    bias = compute_macro_bias(summary)

    if bias.crisis:
        logger.info(f"Skip: macro crisis")
        return
    position_size *= bias.size_multiplier
    if not bias.aligned(direction):
        position_size *= 0.6    # fade unaligned trades
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Tunables — keep them data not magic numbers so tests can inject
CRISIS_MACRO_RISK_THRESHOLD = 80            # macro_risk (0-100) >= this = crisis
CRISIS_CONFIDENCE_THRESHOLD = 0.70          # require non-trivial confidence to trigger crisis
BIAS_DOMINANCE_MIN = 2                      # bullish/bearish count must exceed the other by at least N
SIZE_MULT_AT_MAX_BIAS = 1.5                 # size multiplier when |bias| = 1.0
SIZE_MULT_AT_ZERO_BIAS = 1.0                # neutral macro = normal size
SIZE_MULT_FLOOR = 0.5                       # strong-misalignment floor (still trade, small)
SIZE_MULT_CRISIS = 0.0                      # crisis = skip
UNALIGNED_FADE_FACTOR = 0.6                 # when direction opposes bias sign


@dataclass
class MacroBias:
    signed_bias: float = 0.0                # -1.0 .. +1.0
    crisis: bool = False
    size_multiplier: float = 1.0            # 0.0 .. 1.5
    confidence: float = 0.0                 # avg confidence across active layers
    active_layers: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    composite_signal: str = "NEUTRAL"
    usd_regime: str = "neutral"
    pre_event_flag: bool = False
    reasons: List[str] = field(default_factory=list)

    def aligned(self, direction: str) -> bool:
        """Does the trade direction match the macro bias sign?

        LONG aligned iff signed_bias >= 0 (bullish or neutral).
        SHORT aligned iff signed_bias <= 0 (bearish or neutral).
        Neutral bias counts as aligned in BOTH directions — we don't fade
        neutral markets, we just don't amplify.
        """
        d = (direction or "").upper()
        if d in ("LONG", "BUY"):
            return self.signed_bias >= -0.05   # small tolerance
        if d in ("SHORT", "SELL"):
            return self.signed_bias <= 0.05
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signed_bias": round(self.signed_bias, 3),
            "crisis": self.crisis,
            "size_multiplier": round(self.size_multiplier, 3),
            "confidence": round(self.confidence, 3),
            "active_layers": self.active_layers,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "composite_signal": self.composite_signal,
            "usd_regime": self.usd_regime,
            "pre_event_flag": self.pre_event_flag,
            "reasons": list(self.reasons),
        }


def compute_macro_bias(summary: Optional[Dict[str, Any]]) -> MacroBias:
    """Turn EconomicIntelligence.get_macro_summary() into a MacroBias.

    Returns a neutral MacroBias on any malformed / missing input — callers
    can trust that this never raises.
    """
    if not summary or not isinstance(summary, dict):
        return MacroBias(reasons=["no_summary"])

    try:
        bullish = int(summary.get("bullish_count", 0))
        bearish = int(summary.get("bearish_count", 0))
        active = int(summary.get("active_layers", 0))
        composite = str(summary.get("composite_signal", "NEUTRAL"))
        confidence = float(summary.get("composite_confidence", 0.0))
        macro_risk = float(summary.get("macro_risk", 0))
        pre_event = bool(summary.get("pre_event_flag", False))
        usd_regime = str(summary.get("usd_regime", "neutral"))
        crisis_flag = bool(summary.get("crisis", False))
    except (TypeError, ValueError):
        return MacroBias(reasons=["malformed_summary"])

    reasons: List[str] = []

    # ── Crisis detection ────────────────────────────────────────────────
    # Trigger on any of:
    #   * explicit crisis flag (central_bank layer saw CRISIS-tier signal)
    #   * macro_risk score >= 80 AND confidence >= 0.70 (strong negative)
    is_crisis = crisis_flag or (
        macro_risk >= CRISIS_MACRO_RISK_THRESHOLD
        and confidence >= CRISIS_CONFIDENCE_THRESHOLD
    )
    if is_crisis:
        reasons.append(
            f"crisis:composite={composite},macro_risk={macro_risk:.0f},conf={confidence:.2f}"
        )
        return MacroBias(
            signed_bias=-1.0,
            crisis=True,
            size_multiplier=SIZE_MULT_CRISIS,
            confidence=confidence,
            active_layers=active,
            bullish_count=bullish,
            bearish_count=bearish,
            composite_signal=composite,
            usd_regime=usd_regime,
            pre_event_flag=pre_event,
            reasons=reasons,
        )

    # ── Signed bias ─────────────────────────────────────────────────────
    # Needs dominance — a 3-3 split is not a signal. Require bullish-bearish
    # delta >= BIAS_DOMINANCE_MIN before giving directional lean.
    delta = bullish - bearish
    if active == 0 or abs(delta) < BIAS_DOMINANCE_MIN:
        signed_bias = 0.0
        reasons.append(f"neutral:bullish={bullish},bearish={bearish},active={active}")
    else:
        # Scale by confidence so low-conf layers don't produce fake certainty
        signed_bias = max(-1.0, min(1.0, (delta / max(active, 4)) * confidence * 2.0))
        side = "bullish" if signed_bias > 0 else "bearish"
        reasons.append(f"{side}:delta={delta},conf={confidence:.2f},bias={signed_bias:+.2f}")

    # Pre-event flag: within 2h of a scheduled event → fade any directional bias
    if pre_event and abs(signed_bias) > 0:
        signed_bias *= 0.5
        reasons.append("pre_event:bias_halved")

    # ── Size multiplier ─────────────────────────────────────────────────
    # Linear: |bias|=0 → 1.0x; |bias|=1 → 1.5x. Floor at 0.5x.
    # This is the AMPLIFIER only — the alignment check below handles direction
    # (unaligned trades get further reduced by UNALIGNED_FADE_FACTOR).
    base_mult = SIZE_MULT_AT_ZERO_BIAS + abs(signed_bias) * (SIZE_MULT_AT_MAX_BIAS - SIZE_MULT_AT_ZERO_BIAS)
    size_mult = max(SIZE_MULT_FLOOR, min(SIZE_MULT_AT_MAX_BIAS, base_mult))

    return MacroBias(
        signed_bias=float(signed_bias),
        crisis=False,
        size_multiplier=float(size_mult),
        confidence=confidence,
        active_layers=active,
        bullish_count=bullish,
        bearish_count=bearish,
        composite_signal=composite,
        usd_regime=usd_regime,
        pre_event_flag=pre_event,
        reasons=reasons,
    )


def is_enabled() -> bool:
    """Master flag — macro bias only applied when the Robinhood-hardening
    intervention set is active. Defaults off for backwards compat."""
    v = (os.environ.get("ACT_ROBINHOOD_HARDEN") or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def apply_direction_alignment(bias: MacroBias, direction: str, size_mult: float) -> float:
    """Adjust size multiplier for alignment. Keep size_mult alone if aligned;
    fade if unaligned. Crisis already forced size=0 earlier."""
    if bias.crisis:
        return 0.0
    if bias.aligned(direction):
        return size_mult
    return size_mult * UNALIGNED_FADE_FACTOR
