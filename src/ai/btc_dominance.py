"""BTC Dominance + ETH/BTC ratio + alt rotation factor.

Per 2026 research:
  * BTC.D > 60% → no altseason expected (BeInCrypto 2026)
  * BTC.D 50-60% → BTC favored over alts (Phemex 2026)
  * BTC.D 45-50% → rotation zone
  * BTC.D < 45% → altseason likely
  * ETH/BTC ratio at ~0.0313 in April 2026, capped at descending
    trendline → ETH underperformance vs BTC

Fetches via CoinGecko free API (`/global` for total + dominance).
Cached 5 minutes.

Anti-overfit:
  * Thresholds research-grounded, not curve-fit
  * Only 24h delta + current level (no learned weights)
  * Bounded outputs
  * Graceful fallback when API unavailable
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CACHE_TTL_S = 300
_cache: Dict[str, Any] = {"ts": 0.0, "data": None}


@dataclass
class DominanceSignals:
    method: str
    btc_dominance_pct: Optional[float] = None
    btc_dominance_change_24h_pct: float = 0.0
    btc_dominance_zone: str = "unavailable"  # no_altseason / btc_favored / rotation_zone / altseason_likely
    eth_btc_ratio: Optional[float] = None
    total_market_cap_change_24h_pct: float = 0.0
    alt_rotation_signal: str = "unavailable"  # btc_favored / alt_favored / neutral
    eth_directional_bias_vs_btc: float = 0.0  # [-1, +1] negative = ETH underperforms
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "btc_dominance_pct": round(float(self.btc_dominance_pct), 3)
                if self.btc_dominance_pct is not None else None,
            "btc_dominance_change_24h_pct": round(float(self.btc_dominance_change_24h_pct), 4),
            "btc_dominance_zone": self.btc_dominance_zone,
            "eth_btc_ratio": round(float(self.eth_btc_ratio), 6)
                if self.eth_btc_ratio is not None else None,
            "total_market_cap_change_24h_pct": round(float(self.total_market_cap_change_24h_pct), 3),
            "alt_rotation_signal": self.alt_rotation_signal,
            "eth_directional_bias_vs_btc": round(float(self.eth_directional_bias_vs_btc), 3),
            "rationale": self.rationale[:300],
        }


def _classify_zone(btc_d: Optional[float]) -> str:
    if btc_d is None:
        return "unavailable"
    if btc_d > 60:
        return "no_altseason"
    if btc_d >= 50:
        return "btc_favored"
    if btc_d >= 45:
        return "rotation_zone"
    return "altseason_likely"


def _classify_rotation(btc_d_change: float, zone: str) -> str:
    """Direction of capital rotation."""
    if zone == "unavailable":
        return "unavailable"
    if btc_d_change > 0.5:
        return "btc_favored"
    if btc_d_change < -0.5:
        return "alt_favored"
    return "neutral"


def _eth_bias_score(btc_d_change: float, eth_btc_ratio_change: Optional[float],
                    zone: str) -> float:
    """Signed score [-1, +1] for ETH directional bias vs BTC.
    Negative = ETH underperforms; positive = ETH outperforms."""
    score = 0.0
    # BTC.D rising → ETH underperforms (negative bias)
    score -= max(-1.0, min(1.0, btc_d_change / 1.0))
    # ETH/BTC ratio direction
    if eth_btc_ratio_change is not None:
        score += max(-0.5, min(0.5, eth_btc_ratio_change * 5))
    # Zone bonus
    if zone == "altseason_likely":
        score += 0.3
    elif zone == "no_altseason":
        score -= 0.3
    return max(-1.0, min(1.0, score))


def _fetch_coingecko_global() -> Optional[Dict[str, Any]]:
    """CoinGecko /global endpoint (free, no auth)."""
    try:
        import requests
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=4)
        if r.status_code != 200:
            return None
        data = r.json().get("data", {})
        market_cap_pct = data.get("market_cap_percentage", {})
        return {
            "btc_dominance_pct": float(market_cap_pct.get("btc", 0)),
            "eth_dominance_pct": float(market_cap_pct.get("eth", 0)),
            "btc_dominance_change_24h_pct": float(
                data.get("market_cap_change_percentage_24h_usd", 0)
            ),
            "total_market_cap_change_24h_pct": float(
                data.get("market_cap_change_percentage_24h_usd", 0)
            ),
        }
    except Exception as e:
        logger.debug("coingecko global fetch failed: %s", e)
        return None


def _fetch_eth_btc_ratio() -> Optional[float]:
    """Compute ETH/BTC from CoinGecko spot prices."""
    try:
        import requests
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price"
            "?ids=bitcoin,ethereum&vs_currencies=usd",
            timeout=4,
        )
        if r.status_code != 200:
            return None
        j = r.json()
        btc = float(j.get("bitcoin", {}).get("usd", 0))
        eth = float(j.get("ethereum", {}).get("usd", 0))
        if btc <= 0 or eth <= 0:
            return None
        return eth / btc
    except Exception as e:
        logger.debug("eth_btc ratio fetch failed: %s", e)
        return None


def fetch_btc_dominance() -> DominanceSignals:
    """Fetch BTC.D + ETH/BTC + rotation signal. Cached 5 min."""
    now = time.time()
    if _cache["data"] is not None and now - _cache["ts"] < CACHE_TTL_S:
        return _cache["data"]

    g = _fetch_coingecko_global()
    eth_btc = _fetch_eth_btc_ratio()

    if g is None and eth_btc is None:
        result = DominanceSignals(method="unavailable",
                                    rationale="coingecko fetches failed")
        _cache["data"] = result
        _cache["ts"] = now
        return result

    btc_d = g.get("btc_dominance_pct") if g else None
    btc_d_change = g.get("btc_dominance_change_24h_pct", 0) if g else 0.0
    zone = _classify_zone(btc_d)
    rotation = _classify_rotation(btc_d_change, zone)
    # eth_btc ratio change rough estimate from 24h change in dominance
    eth_btc_change_est = -btc_d_change * 0.01 if btc_d_change else None
    bias = _eth_bias_score(btc_d_change, eth_btc_change_est, zone)

    parts = [f"zone={zone}", f"rotation={rotation}"]
    if btc_d is not None:
        parts.append(f"BTC.D={btc_d:.2f}%({btc_d_change:+.2f}%/24h)")
    if eth_btc is not None:
        parts.append(f"ETH/BTC={eth_btc:.6f}")

    result = DominanceSignals(
        method="coingecko",
        btc_dominance_pct=btc_d,
        btc_dominance_change_24h_pct=btc_d_change,
        btc_dominance_zone=zone,
        eth_btc_ratio=eth_btc,
        total_market_cap_change_24h_pct=g.get("total_market_cap_change_24h_pct", 0) if g else 0,
        alt_rotation_signal=rotation,
        eth_directional_bias_vs_btc=bias,
        rationale=" | ".join(parts)[:300],
    )
    _cache["data"] = result
    _cache["ts"] = now
    return result
