"""Data-layer health probe — Phase 1 observability extension.

Reports freshness + status of all 12 economic-intelligence layers. Used by:
  - startup pre-flight: refuses to run if > N critical layers are stale
  - Prometheus gauge `act_data_layer_fresh{layer}` (1=fresh, 0=stale)
  - `python -m src.data.health` CLI for ad-hoc inspection

A layer is "fresh" when its get_cached() returns a dict with stale=False and
the cached timestamp is within its configured TTL (default 15 min).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical layer names used across the bot. Keep in sync with
# src/data/economic_intelligence.py's registered layers.
CANONICAL_LAYERS: List[str] = [
    "social_sentiment",
    "derivatives",
    "onchain",
    "defi_liquidity",
    "institutional",
    "regulatory",
    "mining_economics",
    "equity_correlation",
    "central_bank",
    "geopolitical",
    "macro_indicators",
    "usd_strength",
]

# Layers we consider critical for the decision path. A stale critical layer
# is a warning; a stale non-critical layer is informational only.
CRITICAL_LAYERS = {
    "social_sentiment",   # fear/greed, social tone
    "derivatives",        # funding, OI
    "onchain",            # whale flow, exchange heatmap
}

DEFAULT_FRESH_TTL_S = 900.0  # 15 min


@dataclass
class LayerHealth:
    name: str
    fresh: bool
    stale_reason: Optional[str]
    age_s: Optional[float]
    critical: bool

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "fresh": self.fresh,
            "stale_reason": self.stale_reason,
            "age_s": None if self.age_s is None else round(self.age_s, 1),
            "critical": self.critical,
        }


def probe_one(layer_name: str, economic_intelligence) -> LayerHealth:
    """Inspect one layer's cache. Never raises."""
    is_critical = layer_name in CRITICAL_LAYERS
    try:
        layers = getattr(economic_intelligence, "_layers", {}) or {}
        layer = layers.get(layer_name)
        if layer is None:
            return LayerHealth(layer_name, False, "layer not registered", None, is_critical)
        if not hasattr(layer, "get_cached"):
            return LayerHealth(layer_name, False, "no get_cached()", None, is_critical)
        cached = layer.get_cached() or {}
        if not isinstance(cached, dict):
            return LayerHealth(layer_name, False, f"cache type={type(cached).__name__}", None, is_critical)
        if cached.get("stale", True):
            return LayerHealth(layer_name, False, "layer self-reports stale", None, is_critical)
        ts = cached.get("cached_at") or cached.get("timestamp") or cached.get("ts")
        if ts is None:
            # Layer says fresh but has no timestamp — trust it once, flag softly.
            return LayerHealth(layer_name, True, None, None, is_critical)
        age = time.time() - float(ts)
        if age > DEFAULT_FRESH_TTL_S:
            return LayerHealth(layer_name, False, f"age {age:.0f}s > TTL {DEFAULT_FRESH_TTL_S:.0f}s",
                               age, is_critical)
        return LayerHealth(layer_name, True, None, age, is_critical)
    except Exception as e:
        return LayerHealth(layer_name, False, f"probe raised {type(e).__name__}: {e}", None, is_critical)


def probe_all(economic_intelligence) -> List[LayerHealth]:
    """Return a LayerHealth for every canonical layer. Never raises."""
    out = []
    for name in CANONICAL_LAYERS:
        h = probe_one(name, economic_intelligence)
        out.append(h)
        _emit_metric(h)
    return out


def summary(healths: List[LayerHealth]) -> Dict:
    fresh = [h for h in healths if h.fresh]
    stale = [h for h in healths if not h.fresh]
    critical_stale = [h for h in stale if h.critical]
    return {
        "total": len(healths),
        "fresh": len(fresh),
        "stale": len(stale),
        "critical_stale": len(critical_stale),
        "critical_stale_names": [h.name for h in critical_stale],
    }


def is_healthy(healths: List[LayerHealth], max_critical_stale: int = 0) -> bool:
    """True iff no more than `max_critical_stale` critical layers are stale."""
    s = summary(healths)
    return s["critical_stale"] <= max_critical_stale


def _emit_metric(h: LayerHealth) -> None:
    try:
        from src.orchestration.metrics import record_data_layer_fresh
        record_data_layer_fresh(layer=h.name, fresh=h.fresh, critical=h.critical)
    except Exception:
        pass


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> int:
    """`python -m src.data.health` — print a table of layer freshness."""
    try:
        from src.data.economic_intelligence import EconomicIntelligence
        ei = EconomicIntelligence()
    except Exception as e:
        print(f"[ERROR] could not instantiate EconomicIntelligence: {e}")
        return 2
    healths = probe_all(ei)
    print(f"{'LAYER':<22} {'FRESH':<6} {'AGE':<10} {'CRITICAL':<9} REASON")
    print("-" * 70)
    for h in healths:
        age_s = f"{h.age_s:.0f}s" if h.age_s is not None else "—"
        print(f"{h.name:<22} {'yes' if h.fresh else 'NO':<6} {age_s:<10} "
              f"{'Y' if h.critical else '-':<9} {h.stale_reason or ''}")
    s = summary(healths)
    print(f"\n{s['fresh']}/{s['total']} fresh, {s['critical_stale']} critical stale "
          f"({', '.join(s['critical_stale_names']) or 'none'})")
    return 0 if is_healthy(healths) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
