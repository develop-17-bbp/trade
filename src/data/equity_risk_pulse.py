"""Equity risk pulse — VIX term structure + SPY put-call ratio.

Crypto's analyst tool surface includes `get_fear_greed`, a single 0-100
risk-appetite reading. Equities don't have a Fear & Greed Index; the
canonical equivalents are:

  - VIX spot level (10-15 = complacent, 20-30 = elevated, 40+ = panic)
  - VIX term structure: VIX9D / VIX vs VIX / VIX3M
        > 1.0 (backwardation) → near-term stress priced higher than
                                further-dated; classic panic signal
        < 1.0 (contango)      → calm; structural risk premium
  - SPY put-call ratio: count of put OI vs call OI on near-the-money,
        near-dated contracts. > 1.0 = bearish positioning.

This module fuses them into a single `equity_risk_pulse()` reading the
brain can consume the same way it consumes `get_fear_greed`.

Sources (graceful fallback chain):
  1. yfinance for ^VIX, ^VIX9D, ^VIX3M (free, no key)
  2. AlpacaOptionsFetcher for SPY put/call chain (already wired)
  3. Hard fallback: returns risk_score=50 (neutral) so the gate never
     blocks the brain just because a fetcher is down.

Cache: 5 min — VIX moves but the brain runs every 60s; refreshing every
tick would burn yfinance.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Optional

from src.data.base_fetcher import CachedFetcher, CACHE_TTL_MEDIUM

logger = logging.getLogger(__name__)


# Risk-score buckets — analogous to the Fear & Greed labels.
# Lower = more fear, higher = more greed.
def _label_for_score(score: float) -> str:
    if score < 20:
        return "extreme_fear"
    if score < 40:
        return "fear"
    if score < 60:
        return "neutral"
    if score < 80:
        return "greed"
    return "extreme_greed"


class EquityRiskPulse(CachedFetcher):
    """VIX + put-call → unified risk pulse for equities."""

    def __init__(self) -> None:
        super().__init__(timeout=8)
        self._opts = None  # lazy

    # ── Public API ───────────────────────────────────────────────────────

    def vix_level(self) -> Optional[float]:
        cached = self._get_cached("vix_level")
        if cached is not None:
            return cached
        v = self._yf_last("^VIX")
        if v is not None:
            self._set_cached("vix_level", v, ttl=CACHE_TTL_MEDIUM)
        return v

    def vix_term_structure(self) -> Dict[str, float]:
        """Returns the two key ratios + raw legs.

        ratio_9d_to_vix > 1 → near-term stress > 30d implied; backwardation.
        ratio_vix_to_3m > 1 → 30d > 90d implied; broader-curve stress.
        """
        cached = self._get_cached("vix_ts")
        if cached is not None:
            return cached
        legs = {
            "vix9d": self._yf_last("^VIX9D"),
            "vix":   self._yf_last("^VIX"),
            "vix3m": self._yf_last("^VIX3M"),
        }
        out: Dict[str, float] = {}
        if legs["vix9d"] and legs["vix"] and legs["vix"] > 0:
            out["ratio_9d_to_vix"] = legs["vix9d"] / legs["vix"]
        if legs["vix"] and legs["vix3m"] and legs["vix3m"] > 0:
            out["ratio_vix_to_3m"] = legs["vix"] / legs["vix3m"]
        for k, v in legs.items():
            if v is not None:
                out[k] = v
        self._set_cached("vix_ts", out, ttl=CACHE_TTL_MEDIUM)
        return out

    def spy_put_call_ratio(self) -> Optional[float]:
        """Near-term SPY puts/calls ratio. > 1.0 = bearish positioning."""
        cached = self._get_cached("spy_pcr")
        if cached is not None:
            return cached

        try:
            if self._opts is None:
                from src.data.alpaca_options_fetcher import AlpacaOptionsFetcher
                self._opts = AlpacaOptionsFetcher(paper=True)
            calls = self._opts.chain("SPY", side="call", min_dte=1, max_dte=14, limit=100)
            puts  = self._opts.chain("SPY", side="put",  min_dte=1, max_dte=14, limit=100)
        except Exception as e:
            logger.debug("equity_risk_pulse: spy chain fetch failed: %s", e)
            return None

        if not calls or not puts:
            return None

        # Sum of OI is the standard PCR; if OI not in payload fall back
        # to contract count (less precise but directional).
        call_oi = sum(float(c.get("open_interest") or 0) for c in calls)
        put_oi  = sum(float(p.get("open_interest") or 0) for p in puts)
        if call_oi <= 0 and put_oi <= 0:
            ratio = float(len(puts)) / max(1.0, float(len(calls)))
        elif call_oi <= 0:
            ratio = math.inf
        else:
            ratio = put_oi / call_oi

        self._set_cached("spy_pcr", ratio, ttl=CACHE_TTL_MEDIUM)
        return ratio

    def equity_risk_pulse(self) -> Dict[str, object]:
        """One-shot read for the analyst tool surface.

        Returns:
            {
                'risk_score': 0..100,           # higher = more risk-on
                'label':      'extreme_fear'..'extreme_greed',
                'vix':        float | None,
                'vix_ts':     {...},            # term-structure dict
                'spy_pcr':    float | None,
                'reasons':    [str, ...],       # human-readable factors
            }
        """
        vix = self.vix_level()
        vix_ts = self.vix_term_structure()
        pcr = self.spy_put_call_ratio()
        reasons = []

        # Component 1: VIX absolute level. 10 → score 100, 50+ → score 0.
        if vix is None:
            vix_score = 50.0
            reasons.append("vix_unavailable")
        else:
            # Linear from VIX=10 (greed=100) to VIX=50 (fear=0).
            vix_score = max(0.0, min(100.0, 100.0 - ((vix - 10.0) / 40.0) * 100.0))
            reasons.append(f"vix={vix:.1f}->{vix_score:.0f}")

        # Component 2: Term-structure ratio_9d_to_vix.
        # > 1.0 = backwardation/panic; < 0.95 = contango/calm.
        ts_score = 50.0
        ratio = vix_ts.get("ratio_9d_to_vix")
        if ratio is not None:
            # 1.20 → 0 (panic); 0.85 → 100 (calm).
            ts_score = max(0.0, min(100.0, 100.0 * (1.20 - ratio) / (1.20 - 0.85)))
            reasons.append(f"vix_ts={ratio:.2f}->{ts_score:.0f}")
        else:
            reasons.append("vix_ts_unavailable")

        # Component 3: Put-call ratio.
        # PCR < 0.6 = greed (call-heavy); PCR > 1.4 = fear.
        pcr_score = 50.0
        if pcr is not None and math.isfinite(pcr):
            # 0.4 → 100 (greed); 1.6 → 0 (fear).
            pcr_score = max(0.0, min(100.0, 100.0 * (1.6 - pcr) / (1.6 - 0.4)))
            reasons.append(f"pcr={pcr:.2f}->{pcr_score:.0f}")
        else:
            reasons.append("pcr_unavailable")

        # Weighted blend — VIX absolute is the primary signal, term-
        # structure is secondary, PCR is tertiary (noisy on free tier).
        score = 0.5 * vix_score + 0.3 * ts_score + 0.2 * pcr_score
        return {
            "risk_score": round(score, 1),
            "label":      _label_for_score(score),
            "vix":        vix,
            "vix_ts":     vix_ts,
            "spy_pcr":    pcr,
            "reasons":    reasons,
        }

    # ── Source: yfinance ─────────────────────────────────────────────────

    def _yf_last(self, ticker: str) -> Optional[float]:
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            return None
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1d", interval="1m")
            if hist is None or hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.debug("equity_risk_pulse: yfinance %s failed: %s", ticker, e)
            return None


_singleton: Optional[EquityRiskPulse] = None


def get_equity_risk_pulse() -> EquityRiskPulse:
    global _singleton
    if _singleton is None:
        _singleton = EquityRiskPulse()
    return _singleton


def equity_risk_pulse() -> Dict[str, object]:
    """Convenience wrapper used by the LLM tool handler."""
    return get_equity_risk_pulse().equity_risk_pulse()
