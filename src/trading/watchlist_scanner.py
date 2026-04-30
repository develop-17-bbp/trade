"""US large-cap watchlist scanner.

Polls a static top-100 large-cap basket every 5 minutes, scores each name
by a simple multi-factor heuristic, and surfaces the top-N candidates
back to the agentic loop. The active 1-minute trading pipeline keeps
running on a small 'always-on' basket (config.yaml `assets`); the
watchlist is the *opportunity discovery* layer that says "NVDA just moved
3% on 2x volume — analyst, take a look."

Why this design vs trading 100 symbols at 1-minute cadence:
  * Alpaca free tier: 200 req/min. 100 symbols × 1Min = 100 bar calls/min
    PLUS quotes/orders blows past the cap. 100 × 5Min = 20 calls/min,
    well under cap with room for the active 1Min pipeline.
  * Per-symbol attention: 100 names at 1Min means each scanner cycle
    averages ≤0.6s/symbol on a 4060 — the LLM analyst can't keep up.
    Watchlist screens fast (no LLM), only escalates the top 5-10 to the
    full pipeline.
  * Risk: 100 small correlated positions are NOT diversification. The
    watchlist surfaces opportunities, but the executor still respects
    ACT_MAX_OPEN_POSITIONS_PER_ASSET (3) plus a new
    ACT_MAX_OPEN_POSITIONS_TOTAL cap (default 10) so the bot can't
    accumulate 50 small longs across the basket.

The score combines:
  pct_move_5m   (×3 weight) — how much it ran in the last 5 minutes
  vol_ratio     (×2 weight) — volume vs prior-day-rolling-avg ratio
  rsi_extreme   (×1 weight) — bonus for RSI > 70 or < 30 (mean-revert
                              or breakout candidates)

Result is a `List[Candidate]` sorted desc; the agentic loop reads
`get_top_candidates(n)` once per scanner tick.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── The list ─────────────────────────────────────────────────────────────
# Static top-100 US large-caps (S&P 500 leaders) approximate as of 2026.
# Operator can override via ACT_WATCHLIST_OVERRIDE env (comma-separated)
# or by editing config.yaml:watchlist.symbols.
TOP_100_LARGE_CAPS: List[str] = [
    # Mega-caps (1-20)
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "LLY", "UNH", "V", "XOM", "MA", "JNJ", "COST", "HD", "PG", "WMT",
    # Tier 2 (21-50)
    "ABBV", "BAC", "NFLX", "CVX", "KO", "MRK", "AMD", "CRM", "ORCL", "TMO",
    "ABT", "ADBE", "PEP", "ACN", "MCD", "LIN", "CSCO", "NKE", "DIS", "WFC",
    "INTC", "IBM", "CAT", "T", "AXP", "GS", "VZ", "GE", "BX", "TXN",
    # Tier 3 (51-80)
    "QCOM", "BLK", "NOW", "ISRG", "INTU", "AMGN", "BKNG", "BA", "GILD", "DE",
    "AMAT", "SYK", "NEE", "SCHW", "BMY", "RTX", "AAL", "LMT", "SBUX", "PFE",
    "MDT", "COP", "MO", "USB", "SO", "DUK", "TJX", "HON", "MMC", "SPGI",
    # Tier 4 (81-100)
    "CB", "MDLZ", "PLD", "ICE", "MMM", "ELV", "ETN", "REGN", "CL", "DHR",
    "BSX", "ADP", "PYPL", "FDX", "EMR", "ZTS", "NOC", "GD", "KKR", "GOOG",
]


@dataclass
class Candidate:
    symbol: str
    score: float = 0.0
    pct_move_5m: float = 0.0
    vol_ratio: float = 0.0
    rsi: float = 50.0
    last_price: float = 0.0
    direction_hint: str = "FLAT"   # 'LONG' / 'SHORT' / 'FLAT'
    reasons: List[str] = field(default_factory=list)
    # Multi-timeframe alignment (2026-04-30 upgrade): pre-fetched 1h + 4h
    # slope so the LLM analyst's first ReAct turn already has TF-aligned
    # context instead of spending 2-3 extra steps calling get_recent_bars
    # for each candidate. None until pre-fetched.
    slope_1h_pct: Optional[float] = None
    slope_4h_pct: Optional[float] = None
    mtf_aligned: Optional[bool] = None        # True when 5m mom + 1h slope + 4h slope all agree

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "symbol": self.symbol,
            "score": round(self.score, 3),
            "pct_move_5m": round(self.pct_move_5m, 3),
            "vol_ratio": round(self.vol_ratio, 2),
            "rsi": round(self.rsi, 1),
            "last_price": round(self.last_price, 2),
            "direction_hint": self.direction_hint,
            "reasons": list(self.reasons),
        }
        if self.slope_1h_pct is not None:
            out["slope_1h_pct"] = round(self.slope_1h_pct, 3)
        if self.slope_4h_pct is not None:
            out["slope_4h_pct"] = round(self.slope_4h_pct, 3)
        if self.mtf_aligned is not None:
            out["mtf_aligned"] = self.mtf_aligned
        return out


# ── Scoring ─────────────────────────────────────────────────────────────


def _rsi_from_closes(closes: List[float], period: int = 14) -> float:
    """Plain Wilder RSI. Returns 50 if not enough data."""
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    # Perfectly flat = no signal; treat as neutral 50, not max 100.
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _score_candidate(symbol: str, bars: List[List[float]]) -> Optional[Candidate]:
    """`bars` is the list returned by AlpacaFetcher.fetch_ohlcv (timeframe=5Min,
    limit ~30). Returns None if data is insufficient.
    """
    if not bars or len(bars) < 5:
        return None

    closes = [float(b[4]) for b in bars]
    volumes = [float(b[5]) for b in bars]
    last_price = closes[-1]
    prev_5m_close = closes[-2] if len(closes) > 1 else last_price
    pct_move_5m = ((last_price - prev_5m_close) / prev_5m_close * 100.0) if prev_5m_close > 0 else 0.0

    avg_vol = sum(volumes[:-1]) / max(1, len(volumes) - 1) if len(volumes) > 1 else 0.0
    vol_ratio = (volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

    rsi = _rsi_from_closes(closes, period=14)
    rsi_extreme = max(0.0, abs(rsi - 50.0) - 20.0) / 30.0  # 0 if 30<=RSI<=70, ramps up past extremes

    direction_hint = "FLAT"
    if pct_move_5m > 0.3 and vol_ratio > 1.3:
        direction_hint = "LONG"
    elif pct_move_5m < -0.3 and vol_ratio > 1.3:
        direction_hint = "SHORT"

    score = 3.0 * abs(pct_move_5m) + 2.0 * (vol_ratio - 1.0) + 1.0 * rsi_extreme

    reasons: List[str] = []
    if abs(pct_move_5m) >= 1.0:
        reasons.append(f"move_5m={pct_move_5m:+.2f}%")
    if vol_ratio >= 2.0:
        reasons.append(f"vol={vol_ratio:.1f}×")
    if rsi >= 70:
        reasons.append(f"RSI={rsi:.0f}_overbought")
    elif rsi <= 30:
        reasons.append(f"RSI={rsi:.0f}_oversold")

    return Candidate(
        symbol=symbol,
        score=score,
        pct_move_5m=pct_move_5m,
        vol_ratio=vol_ratio,
        rsi=rsi,
        last_price=last_price,
        direction_hint=direction_hint,
        reasons=reasons,
    )


# ── Scanner ─────────────────────────────────────────────────────────────


class WatchlistScanner:
    """5-minute screener over a 100-name large-cap universe.

    Usage:
        scanner = WatchlistScanner(symbols=TOP_100_LARGE_CAPS, refresh_s=300)
        scanner.start()    # spawns a daemon thread
        ...
        top = scanner.top(10)   # called from agentic loop
    """

    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 refresh_s: float = 300.0,
                 timeframe: str = "5Min"):
        env_override = os.getenv("ACT_WATCHLIST_OVERRIDE", "").strip()
        if env_override:
            self.symbols = [s.strip().upper() for s in env_override.split(",") if s.strip()]
        else:
            self.symbols = list(symbols or TOP_100_LARGE_CAPS)
        self.refresh_s = float(refresh_s)
        self.timeframe = timeframe
        self._candidates: List[Candidate] = []
        self._last_scan_ts: float = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="watchlist-scanner", daemon=True,
        )
        self._thread.start()
        logger.info(
            "[WATCHLIST] scanner started — %d symbols, refresh every %.0fs",
            len(self.symbols), self.refresh_s,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.scan_once()
            except Exception as e:
                logger.warning("[WATCHLIST] scan failed: %s", e)
            self._stop.wait(self.refresh_s)

    def scan_once(self) -> List[Candidate]:
        """Pull bars for every watchlist symbol and rebuild the top list."""
        try:
            from src.data.alpaca_fetcher import AlpacaFetcher
            fetcher = AlpacaFetcher(paper=True)
        except Exception as e:
            logger.warning("[WATCHLIST] AlpacaFetcher unavailable: %s", e)
            return []

        if not fetcher.available:
            logger.debug("[WATCHLIST] Alpaca creds missing — scan skipped")
            return []

        new_candidates: List[Candidate] = []
        for sym in self.symbols:
            try:
                bars = fetcher.fetch_ohlcv(sym, timeframe=self.timeframe, limit=30)
                cand = _score_candidate(sym, bars)
                if cand is not None:
                    # Pre-fetch 1h + 4h slope so the analyst's FIRST
                    # turn already has multi-timeframe alignment without
                    # needing extra get_recent_bars tool calls. Saves
                    # ~2-3 ReAct steps per analyzed candidate. We only
                    # do this for the highest-score candidates (top 20)
                    # to bound API calls — 100 symbols × 3 timeframes
                    # would push past Alpaca's 200/min cap.
                    new_candidates.append(cand)
            except Exception as e:
                logger.debug("[WATCHLIST] %s scan error: %s", sym, e)
                continue

        # Pre-fetch 1h + 4h slope for the top-K candidates only so the
        # analyst sees pre-computed MTF alignment in its seed context.
        # Top-20 × 2 extra calls = 40 extra/scan; well under the
        # Alpaca free-tier 200/min cap.
        new_candidates.sort(key=lambda c: c.score, reverse=True)
        for cand in new_candidates[:20]:
            try:
                bars_1h = fetcher.fetch_ohlcv(cand.symbol, timeframe="1Hour", limit=8)
                if bars_1h and len(bars_1h) >= 4:
                    closes_1h = [float(b[4]) for b in bars_1h]
                    if closes_1h[0] > 0:
                        cand.slope_1h_pct = (
                            (closes_1h[-1] - closes_1h[0]) / closes_1h[0] * 100.0
                        )
            except Exception as e:
                logger.debug("[WATCHLIST] %s 1h fetch failed: %s", cand.symbol, e)
            try:
                bars_4h = fetcher.fetch_ohlcv(cand.symbol, timeframe="1Hour", limit=24)
                if bars_4h and len(bars_4h) >= 12:
                    # Use 24×1Hour as proxy for 4h slope (free tier doesn't
                    # always serve 4Hour cleanly; rolling-24h covers the
                    # equivalent regime view).
                    closes_4h = [float(b[4]) for b in bars_4h]
                    if closes_4h[0] > 0:
                        cand.slope_4h_pct = (
                            (closes_4h[-1] - closes_4h[0]) / closes_4h[0] * 100.0
                        )
            except Exception as e:
                logger.debug("[WATCHLIST] %s 4h fetch failed: %s", cand.symbol, e)
            # Compute MTF alignment: True iff direction_hint matches
            # 1h slope sign AND 4h slope sign (all 3 same direction).
            if (cand.slope_1h_pct is not None and cand.slope_4h_pct is not None
                    and cand.direction_hint != "FLAT"):
                want_up = cand.direction_hint == "LONG"
                ok_1h = (cand.slope_1h_pct > 0) if want_up else (cand.slope_1h_pct < 0)
                ok_4h = (cand.slope_4h_pct > 0) if want_up else (cand.slope_4h_pct < 0)
                cand.mtf_aligned = ok_1h and ok_4h
                if cand.mtf_aligned:
                    cand.reasons.append(
                        f"MTF_aligned (1h={cand.slope_1h_pct:+.1f}%/4h={cand.slope_4h_pct:+.1f}%)"
                    )
                    # Bonus score for full alignment so MTF-aligned setups
                    # surface ahead of pure 5m-momentum noise.
                    cand.score += 1.5

        new_candidates.sort(key=lambda c: c.score, reverse=True)
        with self._lock:
            self._candidates = new_candidates
            self._last_scan_ts = time.time()

        if new_candidates:
            top3 = ", ".join(
                f"{c.symbol}({c.score:.1f}/{c.direction_hint})"
                for c in new_candidates[:3]
            )
            logger.info("[WATCHLIST] scan complete — %d ranked, top: %s",
                        len(new_candidates), top3)
        return new_candidates

    def top(self, n: int = 10, min_score: float = 0.5) -> List[Candidate]:
        """Read the most recent scan's top N. Thread-safe.

        `min_score` filters out flat / no-signal names so a quiet day
        returns an empty list instead of 100 zero-score noise.
        """
        with self._lock:
            return [c for c in self._candidates if c.score >= min_score][:n]

    def last_scan_age_s(self) -> Optional[float]:
        if not self._last_scan_ts:
            return None
        return max(0.0, time.time() - self._last_scan_ts)


# ── Module-level singleton ──────────────────────────────────────────────

_SINGLETON: Optional[WatchlistScanner] = None
_SINGLETON_LOCK = threading.Lock()


def get_watchlist_scanner() -> WatchlistScanner:
    """Process-wide singleton. Constructed lazily on first use."""
    global _SINGLETON
    if _SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SINGLETON is None:
                _SINGLETON = WatchlistScanner()
    return _SINGLETON


def get_top_candidates(n: int = 10) -> List[Dict[str, Any]]:
    """Convenience wrapper for tools/skills — returns top-N candidates as
    plain dicts (LLM-friendly)."""
    scanner = get_watchlist_scanner()
    return [c.to_dict() for c in scanner.top(n)]
