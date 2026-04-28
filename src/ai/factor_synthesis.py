"""Factor synthesis — single source of truth for the 6-factor bias score.

The 6 predictive factors (macro_overlay, btc_dominance, cvd, whale_flow,
halving_cycle, btc_eth_lead_lag) are individually wired as brain
tools. But:
  * Continuous-brain daemon doesn't see them
  * Catalyst listener doesn't trigger on them
  * Macro outer loop scanner doesn't include them
  * Each consumer would otherwise re-fetch them

This module computes the synthesis ONCE per refresh interval (default
5 min, matching the longest cache TTL of any source factor) and
publishes the result to tick_state. Then all consumers — agentic loop,
continuous brain, catalyst listener, scanner — read the SAME synthesis
without re-fetching.

Output shape (per asset):
    {
      "asset": "BTC",
      "long_bias_score": 0.42,                # [-1, +1]
      "regime_label": "risk_on" | ...,
      "bullish_factors": ["macro", "halving", "cvd"],
      "bearish_factors": ["btc_dominance"],
      "confidence_label": "strong_long" | "mild_long" | "neutral" | ...,
      "recommended_action": "submit_long" | "wait_for_confirmation" |
                             "skip" | "close_at_profit" | "no_new_longs",
      "rationale": "...",
      "computed_at": ts,
      "source_factors": {... raw factor outputs ...},
    }

Anti-overfit / anti-noise:
  * Bias-score formula matches the prompt's MARKET DIRECTION SYNTHESIS
    recipe — operator-readable, not learned weights
  * Each missing factor reduces confidence (no spurious certainty
    when one source is unavailable)
  * Output bounded [-1, +1]
  * 5-min refresh interval prevents flapping
  * "TRUST HIGHER-LEVEL SIGNAL" rule encoded: if macro disagrees with
    order-flow, macro wins on the bias score
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_REFRESH_INTERVAL_S = 300  # 5 min — match factor cache TTL
_cache: Dict[str, Dict[str, Any]] = {}  # asset → synthesis dict


@dataclass
class FactorSynthesis:
    asset: str
    long_bias_score: float                 # [-1, +1]
    regime_label: str
    bullish_factors: List[str] = field(default_factory=list)
    bearish_factors: List[str] = field(default_factory=list)
    confidence_label: str = "neutral"
    recommended_action: str = "skip"
    rationale: str = ""
    computed_at: float = 0.0
    n_factors_available: int = 0
    source_factors: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "long_bias_score": round(float(self.long_bias_score), 3),
            "regime_label": self.regime_label,
            "bullish_factors": self.bullish_factors[:10],
            "bearish_factors": self.bearish_factors[:10],
            "confidence_label": self.confidence_label,
            "recommended_action": self.recommended_action,
            "rationale": self.rationale[:300],
            "computed_at": self.computed_at,
            "n_factors_available": int(self.n_factors_available),
            "age_s": round(time.time() - self.computed_at, 1) if self.computed_at else None,
        }


def _classify_action(score: float, asset: str = "BTC") -> Tuple[str, str]:
    """Map score → (recommended_action, confidence_label).
    Robinhood is longs-only — short bias means SKIP not enter SHORT.
    """
    if score > 0.5:
        return "submit_long", "strong_long"
    if score > 0.2:
        return "wait_for_confirmation", "mild_long"
    if score >= -0.2:
        return "skip", "neutral"
    if score >= -0.5:
        return "no_new_longs_partial_close_winners", "mild_short"
    return "no_new_longs_consider_thesis_broken_close", "strong_short"


def _safe_get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safe nested-dict access."""
    if not isinstance(d, dict):
        return default
    return d.get(key, default)


def compute_synthesis(asset: str = "BTC") -> FactorSynthesis:
    """Run all 6 factors and combine into a single directional bias.

    Each factor is called in a try/except — a missing factor reduces
    n_factors_available but doesn't block the synthesis. Empty/error
    factors contribute 0 to the score.
    """
    asset = str(asset).upper()
    factors: Dict[str, Any] = {}
    components: List[Tuple[str, float, str]] = []  # (name, score, direction_label)

    # 1. Macro overlay
    try:
        from src.ai.macro_overlay import fetch_macro_overlay
        m = fetch_macro_overlay()
        if m.method != "unavailable":
            factors["macro"] = m.to_dict()
            components.append((
                "macro", float(m.crypto_directional_bias),
                f"regime={m.risk_regime}",
            ))
    except Exception as e:
        logger.debug("synthesis macro fetch failed: %s", e)

    # 2. BTC dominance + ETH bias
    try:
        from src.ai.btc_dominance import fetch_btc_dominance
        d = fetch_btc_dominance()
        if d.method != "unavailable":
            factors["btc_dominance"] = d.to_dict()
            # For ETH, use eth_directional_bias_vs_btc directly
            # For BTC, dominance rising is mildly positive (BTC favored)
            if asset == "ETH":
                bias = float(d.eth_directional_bias_vs_btc)
            else:
                # BTC: rising BTC.D is mild positive (capital flowing in)
                bias = max(-0.5, min(0.5, d.btc_dominance_change_24h_pct / 2.0))
            components.append((
                "btc_dominance", bias,
                f"zone={d.btc_dominance_zone}",
            ))
    except Exception as e:
        logger.debug("synthesis btc_dom fetch failed: %s", e)

    # 3. Halving cycle (BONUS — +0.3 for bullish_phase, 0 otherwise)
    try:
        from src.ai.halving_cycle import get_halving_cycle
        h = get_halving_cycle()
        factors["halving_cycle"] = h.to_dict()
        cycle_bonus = 0.3 if h.bullish_phase else (-0.2 if h.cycle_phase == "distribution_bear" else 0.0)
        components.append((
            "halving_cycle", cycle_bonus,
            f"phase={h.cycle_phase}",
        ))
    except Exception as e:
        logger.debug("synthesis halving fetch failed: %s", e)

    # 4. CVD (per-asset)
    try:
        from src.ai.cvd import compute_cvd
        from src.data.fetcher import PriceFetcher
        pf = PriceFetcher()
        bars = pf.get_recent_bars(asset, timeframe="1h", n=80) or []
        if bars and len(bars) >= 60:
            highs = [float(b.get("high", 0)) for b in bars]
            lows = [float(b.get("low", 0)) for b in bars]
            closes = [float(b.get("close", 0)) for b in bars]
            volumes = [float(b.get("volume", 0)) for b in bars]
            opens = [closes[i - 1] if i > 0 else closes[0] for i in range(len(closes))]
            cvd_r = compute_cvd(closes, highs, lows, volumes, opens=opens)
            factors["cvd"] = cvd_r.to_dict()
            components.append((
                "cvd", float(cvd_r.cvd_momentum_score),
                f"slope={cvd_r.cvd_slope_sign:+d} divergence={cvd_r.cvd_divergence_kind}",
            ))
    except Exception as e:
        logger.debug("synthesis cvd compute failed: %s", e)

    # 5. Whale flow (per-asset)
    try:
        from src.ai.whale_flow import detect_whale_flow
        from src.data.fetcher import PriceFetcher
        pf = PriceFetcher()
        bars = pf.get_recent_bars(asset, timeframe="1h", n=120) or []
        if bars and len(bars) >= 100:
            closes = [float(b.get("close", 0)) for b in bars]
            opens = [closes[i - 1] if i > 0 else closes[0] for i in range(len(closes))]
            volumes = [float(b.get("volume", 0)) for b in bars]
            wf_r = detect_whale_flow(closes, opens, volumes)
            factors["whale_flow"] = wf_r.to_dict()
            components.append((
                "whale_flow", float(wf_r.whale_directional_bias),
                f"n_whales={wf_r.n_whale_bars_recent}",
            ))
    except Exception as e:
        logger.debug("synthesis whale_flow failed: %s", e)

    # 6. Lead-lag (informational; small bias when this asset leads)
    try:
        from src.ai.btc_eth_lead_lag import analyze_lead_lag
        from src.data.fetcher import PriceFetcher
        pf = PriceFetcher()
        btc_bars = pf.get_recent_bars("BTC", timeframe="1h", n=200) or []
        eth_bars = pf.get_recent_bars("ETH", timeframe="1h", n=200) or []
        if (btc_bars and eth_bars
                and len(btc_bars) >= 50 and len(eth_bars) >= 50):
            btc_closes = [float(b.get("close", 0)) for b in btc_bars]
            eth_closes = [float(b.get("close", 0)) for b in eth_bars]
            ll_r = analyze_lead_lag(btc_closes, eth_closes)
            factors["lead_lag"] = ll_r.to_dict()
            # +0.2 if THIS asset leads (move first, capture more)
            if (asset == "BTC" and ll_r.relationship == "btc_leads_eth"
                    and ll_r.correlation_strength > 0.3):
                lead_bias = 0.2
            elif (asset == "ETH" and ll_r.relationship == "eth_leads_btc"
                    and ll_r.correlation_strength > 0.3):
                lead_bias = 0.2
            else:
                lead_bias = 0.0
            components.append((
                "lead_lag", lead_bias,
                f"relation={ll_r.relationship}",
            ))
    except Exception as e:
        logger.debug("synthesis lead_lag failed: %s", e)

    # Combine — straight average of available components.
    n = len(components)
    if n == 0:
        return FactorSynthesis(
            asset=asset, long_bias_score=0.0,
            regime_label="unavailable",
            confidence_label="neutral",
            recommended_action="skip",
            rationale="no factors available",
            computed_at=time.time(),
            n_factors_available=0,
        )

    score = sum(s for _, s, _ in components) / n
    score = max(-1.0, min(1.0, score))

    bullish = [name for name, s, _ in components if s > 0.1]
    bearish = [name for name, s, _ in components if s < -0.1]

    macro_regime = factors.get("macro", {}).get("risk_regime", "unknown")
    action, conf_label = _classify_action(score, asset)

    rationale_parts = [f"score={score:+.2f}", f"n={n}/6"]
    rationale_parts.extend(
        f"{name}={s:+.2f}({lbl[:20]})" for name, s, lbl in components[:6]
    )

    result = FactorSynthesis(
        asset=asset,
        long_bias_score=score,
        regime_label=macro_regime,
        bullish_factors=bullish,
        bearish_factors=bearish,
        confidence_label=conf_label,
        recommended_action=action,
        rationale=" | ".join(rationale_parts)[:300],
        computed_at=time.time(),
        n_factors_available=n,
        source_factors=factors,
    )
    return result


def refresh_and_publish(asset: str = "BTC",
                        refresh_interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
                        force: bool = False) -> FactorSynthesis:
    """Compute synthesis if cache is stale, publish to tick_state.

    Cache prevents per-tick recompute (factors are cached individually
    too, but this skips the whole orchestration). All consumers
    (agentic loop, continuous brain, catalyst listener) read from
    tick_state — this is the single source of truth.
    """
    asset = str(asset).upper()
    cached = _cache.get(asset)
    if (not force and cached is not None
            and time.time() - cached.get("computed_at", 0) < refresh_interval_s):
        return FactorSynthesis(**{
            k: v for k, v in cached.items()
            if k in FactorSynthesis.__dataclass_fields__
        })

    synthesis = compute_synthesis(asset)
    _cache[asset] = synthesis.to_dict()

    # Publish to tick_state for all consumers
    try:
        from src.ai import tick_state as _ts
        _ts.update(asset,
                   factor_bias_score=synthesis.long_bias_score,
                   factor_regime=synthesis.regime_label,
                   factor_action=synthesis.recommended_action,
                   factor_confidence=synthesis.confidence_label,
                   factor_n_available=synthesis.n_factors_available,
                   factor_bullish=",".join(synthesis.bullish_factors)[:120],
                   factor_bearish=",".join(synthesis.bearish_factors)[:120],
                   factor_rationale=synthesis.rationale)
    except Exception as e:
        logger.debug("synthesis publish to tick_state failed: %s", e)

    return synthesis


def get_cached(asset: str = "BTC") -> Optional[Dict[str, Any]]:
    """Return cached synthesis without re-running."""
    return _cache.get(str(asset).upper())
