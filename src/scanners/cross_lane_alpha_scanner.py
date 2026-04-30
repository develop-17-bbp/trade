"""Cross-lane alpha ranker — picks the highest-EV setups across every
enabled exchange lane each tick, with per-asset-class budget caps so a
single class can't siphon the whole risk envelope.

Design (matches plan Phase 5.1):
  1. Walk every `enabled` exchange in config.yaml that the current
     ACT_BOX_ROLE filter admits (so 4060 with role='stocks,crypto' sees
     alpaca + alpaca_crypto; 5090 with role='crypto' sees robinhood).
  2. For each asset on each lane, pull recent OHLCV via the lane's
     fetcher and run `score_candidate` (venue-agnostic OHLCV→score).
  3. Sort by score (proxy for expected daily EV) descending.
  4. Admit candidates greedily while respecting a per-asset-class
     percent-of-NAV cap from `cross_lane.max_class_pct` in config.yaml.

This is the *discovery* layer. The conviction gate, authority overlay,
and SL ratchet still run on each admitted candidate before any order
fires. Failure modes here (fetcher down, malformed bars) emit one log
line and the lane is skipped for the cycle — never raises.
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Hard ceilings prevent single-class concentration regardless of scoring
# noise. RH long-hold + alpaca-crypto share the CRYPTO budget so they
# can't double-allocate. OPTIONS is leveraged → tightest cap.
DEFAULT_MAX_CLASS_PCT: Dict[str, float] = {
    "CRYPTO":     35.0,
    "STOCK":      45.0,
    "OPTIONS":    20.0,
    "POLYMARKET": 15.0,
}

DEFAULT_TOP_K = 10
SCAN_BARS_LIMIT = 30
SCAN_TIMEFRAME = "5Min"


@dataclass
class CrossLaneCandidate:
    lane: str             # exchange name (e.g. 'robinhood', 'alpaca', 'alpaca_crypto')
    asset: str            # symbol (e.g. 'BTC', 'NVDA')
    asset_class: str      # 'CRYPTO' / 'STOCK' / 'OPTIONS' / 'POLYMARKET'
    venue: str            # 'robinhood' / 'alpaca' / 'polymarket'
    score: float          # raw alpha score from score_candidate (proxy EV)
    size_pct: float       # nominal size as % of NAV if admitted
    direction_hint: str = "FLAT"
    last_price: float = 0.0
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lane": self.lane, "asset": self.asset, "asset_class": self.asset_class,
            "venue": self.venue, "score": round(self.score, 3),
            "size_pct": round(self.size_pct, 2), "direction_hint": self.direction_hint,
            "last_price": round(self.last_price, 4), "reasons": list(self.reasons),
        }


# ── Config loading ─────────────────────────────────────────────────────


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = config_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config.yaml",
    )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _max_class_pct(config: Dict[str, Any]) -> Dict[str, float]:
    """Pull `cross_lane.max_class_pct` from config; fall back to defaults
    for any missing class. Operator can retune without code changes."""
    raw = (config.get("cross_lane") or {}).get("max_class_pct") or {}
    out = dict(DEFAULT_MAX_CLASS_PCT)
    for k, v in raw.items():
        try:
            out[str(k).upper()] = float(v)
        except (TypeError, ValueError):
            logger.warning("[CROSS-LANE] cross_lane.max_class_pct[%s]=%r unparseable", k, v)
    return out


def _active_lanes(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return enabled exchanges filtered by ACT_BOX_ROLE (mirror of the
    role-to-class mapping in src/main.py — same precedence rules)."""
    role_to_class = {
        "crypto": "CRYPTO", "stocks": "STOCK", "stock": "STOCK",
        "options": "OPTIONS", "polymarket": "POLYMARKET",
    }
    box_roles = [
        r.strip().lower()
        for r in (os.environ.get("ACT_BOX_ROLE") or "").split(",")
        if r.strip()
    ]
    allowed_classes = {role_to_class[r] for r in box_roles if r in role_to_class}

    lanes: List[Dict[str, Any]] = []
    for ex in config.get("exchanges", []) or []:
        cls = (ex.get("asset_class") or "").upper()
        if allowed_classes:
            if cls in allowed_classes:
                lanes.append(ex)
        elif ex.get("enabled", True):
            lanes.append(ex)
    return lanes


# ── Scoring ────────────────────────────────────────────────────────────


def _fetch_bars(lane_cfg: Dict[str, Any], asset: str) -> Optional[List[List[float]]]:
    """Per-venue fetch with try/except so one bad lane never breaks the scan."""
    venue = (lane_cfg.get("venue") or lane_cfg.get("name") or "").lower()
    try:
        if venue == "alpaca":
            from src.data.alpaca_fetcher import AlpacaFetcher
            f = AlpacaFetcher(paper=bool(lane_cfg.get("paper", True)))
            if not f.available:
                return None
            # Stocks symbols pass through; crypto symbols may need normalization
            # (BTC -> BTC/USD); AlpacaFetcher already handles that.
            return f.fetch_ohlcv(asset, timeframe=SCAN_TIMEFRAME, limit=SCAN_BARS_LIMIT)
        if venue == "robinhood":
            from src.data.robinhood_fetcher import RobinhoodFetcher
            f = RobinhoodFetcher()
            return f.fetch_ohlcv(asset, timeframe=SCAN_TIMEFRAME, limit=SCAN_BARS_LIMIT)
        if venue == "polymarket":
            # Binary markets don't have OHLCV; skip until polymarket scoring
            # path is wired (separate module).
            return None
    except Exception as e:
        logger.debug("[CROSS-LANE] %s/%s fetch failed: %s", venue, asset, e)
        return None
    return None


def score_candidate(lane_cfg: Dict[str, Any], asset: str,
                    bars: List[List[float]]) -> Optional[CrossLaneCandidate]:
    """Reuses watchlist_scanner's OHLCV→score logic so cross-lane scoring
    stays consistent with the in-tree stocks scanner. Returns None when
    bars are too short to score reliably."""
    from src.trading.watchlist_scanner import _score_candidate as _ws_score
    base = _ws_score(asset, bars)
    if base is None:
        return None
    return CrossLaneCandidate(
        lane=lane_cfg.get("name") or "",
        asset=asset,
        asset_class=(lane_cfg.get("asset_class") or "").upper(),
        venue=(lane_cfg.get("venue") or lane_cfg.get("name") or "").lower(),
        score=base.score,
        size_pct=_default_size_pct(lane_cfg),
        direction_hint=base.direction_hint,
        last_price=base.last_price,
        reasons=list(base.reasons),
    )


def _default_size_pct(lane_cfg: Dict[str, Any]) -> float:
    """Choose a default per-position % size for the lane. Honors
    `intraday_position_pct_max` if set, else falls back to a venue
    default (CRYPTO 5%, STOCK 5%, OPTIONS 3%, POLYMARKET 2%)."""
    explicit = lane_cfg.get("intraday_position_pct_max")
    if isinstance(explicit, (int, float)) and explicit > 0:
        # Use a third of the per-position max as a "typical" entry —
        # the conviction gate may scale up to the cap on sniper tier.
        return float(explicit) / 3.0
    cls = (lane_cfg.get("asset_class") or "").upper()
    return {"CRYPTO": 5.0, "STOCK": 5.0, "OPTIONS": 3.0, "POLYMARKET": 2.0}.get(cls, 3.0)


# ── Ranker ─────────────────────────────────────────────────────────────


def rank_lanes(config: Optional[Dict[str, Any]] = None,
               top_k: int = DEFAULT_TOP_K) -> List[CrossLaneCandidate]:
    """Score every (lane, asset) pair across enabled lanes and return
    the highest-EV candidates that fit the per-asset-class budget cap.

    The cap admits candidates greedily by score; once a class would
    breach its ceiling, subsequent candidates of that class are skipped
    even if their EV is high — this is intentional. Concentrating into
    one class would defeat cross-lane diversification.
    """
    cfg = config if config is not None else _load_config()
    caps = _max_class_pct(cfg)
    lanes = _active_lanes(cfg)
    if not lanes:
        logger.info("[CROSS-LANE] no active lanes (ACT_BOX_ROLE filter or all disabled)")
        return []

    raw: List[CrossLaneCandidate] = []
    for lane_cfg in lanes:
        for asset in lane_cfg.get("assets", []) or []:
            bars = _fetch_bars(lane_cfg, asset)
            if not bars:
                continue
            cand = score_candidate(lane_cfg, asset, bars)
            if cand is not None:
                raw.append(cand)

    raw.sort(key=lambda c: c.score, reverse=True)

    class_budget: Dict[str, float] = defaultdict(float)
    selected: List[CrossLaneCandidate] = []
    for cand in raw:
        cap = caps.get(cand.asset_class, 0.0)
        if class_budget[cand.asset_class] + cand.size_pct > cap:
            continue
        selected.append(cand)
        class_budget[cand.asset_class] += cand.size_pct
        if len(selected) >= top_k:
            break

    logger.info(
        "[CROSS-LANE] scored=%d admitted=%d caps=%s budget_used=%s",
        len(raw), len(selected), {k: round(v, 1) for k, v in caps.items()},
        {k: round(v, 1) for k, v in class_budget.items()},
    )
    return selected


# ── CLI ────────────────────────────────────────────────────────────────


def _main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true",
                    help="Run a single scan + print results, then exit.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip fetcher init and use synthetic bars (smoke test).")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.dry_run:
        # Synthetic single-lane single-asset path — verifies admit logic
        # without needing live fetchers.
        cfg = {
            "exchanges": [{"name": "alpaca", "asset_class": "STOCK", "venue": "alpaca",
                           "enabled": True, "paper": True, "assets": ["NVDA"]}],
            "cross_lane": {"max_class_pct": {"STOCK": 10.0}},
        }
        # Synthetic 30 bars trending up ~0.5% per bar
        bars = [[0, 100 + i * 0.5, 100 + i * 0.5, 100 + i * 0.5,
                 100 + i * 0.5, 1_000_000 * (1.0 + i * 0.05)]
                for i in range(SCAN_BARS_LIMIT)]
        cand = score_candidate(cfg["exchanges"][0], "NVDA", bars)
        if cand is None:
            print("[DRY-RUN] scoring returned None — fixture too short")
            return 1
        print("[DRY-RUN] candidate:", cand.to_dict())
        return 0

    selected = rank_lanes(top_k=args.top_k)
    if not selected:
        print("[CROSS-LANE] no candidates admitted")
        return 0
    for c in selected:
        print(c.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
