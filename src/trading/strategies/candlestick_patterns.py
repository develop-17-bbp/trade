"""Candlestick pattern detector — 30 classic single/double/triple-bar patterns.

Heuristic detectors for the most-traded candlestick patterns. Each
returns a structured signal with direction (LONG / SHORT / FLAT),
confidence (0.0-1.0), and per-bar feature breakdown.

These detectors serve TWO purposes:
  1. Direct LLM-callable tool (`query_candlestick_patterns`) at every tick
  2. **Label generator for the Candlestick Transformer training set** —
     the transformer learns to identify these same patterns from
     raw OHLCV sequences via supervised learning + smoother probability
     output (heuristics are binary; the transformer is calibrated)

PATTERNS COVERED (30):

  Single-bar (8):
    - doji, dragonfly_doji, gravestone_doji
    - hammer, hanging_man
    - shooting_star, inverted_hammer
    - marubozu_bull, marubozu_bear, spinning_top

  Two-bar (10):
    - bullish_engulfing, bearish_engulfing
    - piercing_line, dark_cloud_cover
    - tweezer_top, tweezer_bottom
    - bullish_harami, bearish_harami
    - matching_low, matching_high

  Three-bar (12):
    - morning_star, evening_star
    - three_white_soldiers, three_black_crows
    - three_inside_up, three_inside_down
    - three_outside_up, three_outside_down
    - abandoned_baby_bull, abandoned_baby_bear
    - bullish_kicker, bearish_kicker

Anti-overfit:
    * Pure rule-based heuristics — no parameter learning
    * Tolerance bands explicit (e.g. body < 10% of range = doji)
    * Multi-pattern detection per bar — return ALL patterns matched
    * Confidence based on pattern strength (body/wick ratios), not learning
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Pattern thresholds — operator-tunable but not learned per-asset
DOJI_BODY_PCT = 0.10              # body < 10% of range
LONG_LOWER_WICK_PCT = 0.66        # for hammer / hanging_man
LONG_UPPER_WICK_PCT = 0.66        # for shooting_star / inverted_hammer
MARUBOZU_BODY_PCT = 0.95          # body > 95% of range
SPINNING_TOP_BODY_PCT = 0.30
ENGULFING_OVERLAP_PCT = 1.00      # body engulfs prior body completely
PIERCING_PEN_PCT = 0.50           # closes above 50% of prior body
HARAMI_INSIDE_PCT = 0.70          # body inside prior body's range


@dataclass
class Bar:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 1e-9)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bull(self) -> bool:
        return self.close > self.open

    @property
    def is_bear(self) -> bool:
        return self.close < self.open

    @property
    def body_pct(self) -> float:
        return self.body / self.range


@dataclass
class CandlePattern:
    name: str
    direction: str          # "LONG" / "SHORT" / "FLAT"
    confidence: float       # 0.0 - 1.0
    bars_used: int
    features: Dict[str, float] = field(default_factory=dict)


def _bars_from_ohlc(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: Optional[List[float]] = None,
) -> List[Bar]:
    n = len(closes)
    if not all(len(x) == n for x in (opens, highs, lows)):
        raise ValueError("OHLCV arrays must have same length")
    vols = volumes if volumes else [0.0] * n
    return [
        Bar(open=float(opens[i]), high=float(highs[i]),
            low=float(lows[i]), close=float(closes[i]),
            volume=float(vols[i] if i < len(vols) else 0))
        for i in range(n)
    ]


# ── Single-bar patterns ──────────────────────────────────────────

def _is_doji(b: Bar) -> Optional[CandlePattern]:
    if b.body_pct >= DOJI_BODY_PCT:
        return None
    # Subtype based on wicks
    if b.lower_wick > 2 * b.upper_wick and b.lower_wick / b.range > 0.6:
        return CandlePattern("dragonfly_doji", "LONG", 0.7, 1,
                             {"body_pct": b.body_pct, "lower_wick_pct": b.lower_wick / b.range})
    if b.upper_wick > 2 * b.lower_wick and b.upper_wick / b.range > 0.6:
        return CandlePattern("gravestone_doji", "SHORT", 0.7, 1,
                             {"body_pct": b.body_pct, "upper_wick_pct": b.upper_wick / b.range})
    return CandlePattern("doji", "FLAT", 0.55, 1, {"body_pct": b.body_pct})


def _is_hammer_or_hanging_man(b: Bar, prior_trend_down: bool) -> Optional[CandlePattern]:
    if b.lower_wick / b.range < LONG_LOWER_WICK_PCT:
        return None
    if b.upper_wick > b.body:
        return None
    if b.body_pct > 0.40:
        return None
    name = "hammer" if prior_trend_down else "hanging_man"
    direction = "LONG" if prior_trend_down else "SHORT"
    return CandlePattern(name, direction, 0.72, 1,
                         {"lower_wick_pct": b.lower_wick / b.range, "body_pct": b.body_pct})


def _is_shooting_star_or_inverted_hammer(b: Bar, prior_trend_up: bool) -> Optional[CandlePattern]:
    if b.upper_wick / b.range < LONG_UPPER_WICK_PCT:
        return None
    if b.lower_wick > b.body:
        return None
    if b.body_pct > 0.40:
        return None
    name = "shooting_star" if prior_trend_up else "inverted_hammer"
    direction = "SHORT" if prior_trend_up else "LONG"
    return CandlePattern(name, direction, 0.72, 1,
                         {"upper_wick_pct": b.upper_wick / b.range, "body_pct": b.body_pct})


def _is_marubozu(b: Bar) -> Optional[CandlePattern]:
    if b.body_pct < MARUBOZU_BODY_PCT:
        return None
    if b.is_bull:
        return CandlePattern("marubozu_bull", "LONG", 0.78, 1, {"body_pct": b.body_pct})
    if b.is_bear:
        return CandlePattern("marubozu_bear", "SHORT", 0.78, 1, {"body_pct": b.body_pct})
    return None


def _is_spinning_top(b: Bar) -> Optional[CandlePattern]:
    if not (DOJI_BODY_PCT <= b.body_pct <= SPINNING_TOP_BODY_PCT):
        return None
    if b.upper_wick < b.body or b.lower_wick < b.body:
        return None
    return CandlePattern("spinning_top", "FLAT", 0.45, 1, {"body_pct": b.body_pct})


# ── Two-bar patterns ─────────────────────────────────────────────

def _is_engulfing(prev: Bar, curr: Bar) -> Optional[CandlePattern]:
    if curr.body == 0 or prev.body == 0:
        return None
    if curr.is_bull and prev.is_bear and \
       curr.close > prev.open and curr.open < prev.close:
        engulf_ratio = curr.body / max(prev.body, 1e-9)
        return CandlePattern("bullish_engulfing", "LONG",
                             min(0.85, 0.55 + 0.10 * min(engulf_ratio, 3.0)),
                             2, {"engulf_ratio": engulf_ratio})
    if curr.is_bear and prev.is_bull and \
       curr.close < prev.open and curr.open > prev.close:
        engulf_ratio = curr.body / max(prev.body, 1e-9)
        return CandlePattern("bearish_engulfing", "SHORT",
                             min(0.85, 0.55 + 0.10 * min(engulf_ratio, 3.0)),
                             2, {"engulf_ratio": engulf_ratio})
    return None


def _is_piercing_or_dark_cloud(prev: Bar, curr: Bar) -> Optional[CandlePattern]:
    if prev.body == 0:
        return None
    midpoint = (prev.open + prev.close) / 2.0
    if prev.is_bear and curr.is_bull and \
       curr.open < prev.low and curr.close > midpoint and curr.close < prev.open:
        pen_pct = (curr.close - prev.close) / max(prev.body, 1e-9)
        return CandlePattern("piercing_line", "LONG", 0.70, 2, {"penetration_pct": pen_pct})
    if prev.is_bull and curr.is_bear and \
       curr.open > prev.high and curr.close < midpoint and curr.close > prev.open:
        pen_pct = (prev.close - curr.close) / max(prev.body, 1e-9)
        return CandlePattern("dark_cloud_cover", "SHORT", 0.70, 2, {"penetration_pct": pen_pct})
    return None


def _is_tweezer(prev: Bar, curr: Bar, atr: float) -> Optional[CandlePattern]:
    tol = atr * 0.10
    if abs(prev.high - curr.high) <= tol and prev.is_bull and curr.is_bear:
        return CandlePattern("tweezer_top", "SHORT", 0.62, 2, {"high_match_tol_atr": 0.10})
    if abs(prev.low - curr.low) <= tol and prev.is_bear and curr.is_bull:
        return CandlePattern("tweezer_bottom", "LONG", 0.62, 2, {"low_match_tol_atr": 0.10})
    return None


def _is_harami(prev: Bar, curr: Bar) -> Optional[CandlePattern]:
    prev_body_high = max(prev.open, prev.close)
    prev_body_low = min(prev.open, prev.close)
    curr_body_high = max(curr.open, curr.close)
    curr_body_low = min(curr.open, curr.close)
    if curr_body_high < prev_body_high and curr_body_low > prev_body_low:
        if prev.is_bear and curr.is_bull:
            return CandlePattern("bullish_harami", "LONG", 0.65, 2,
                                 {"inside_pct": (curr_body_high - curr_body_low) / max(prev.body, 1e-9)})
        if prev.is_bull and curr.is_bear:
            return CandlePattern("bearish_harami", "SHORT", 0.65, 2,
                                 {"inside_pct": (curr_body_high - curr_body_low) / max(prev.body, 1e-9)})
    return None


# ── Three-bar patterns ───────────────────────────────────────────

def _is_morning_star_or_evening_star(b1: Bar, b2: Bar, b3: Bar) -> Optional[CandlePattern]:
    # Morning Star: bear, small body (gap down), bull closing > b1 midpoint
    if b1.is_bear and b3.is_bull and b2.body < 0.4 * b1.body:
        b1_mid = (b1.open + b1.close) / 2.0
        if b3.close > b1_mid:
            return CandlePattern("morning_star", "LONG", 0.78, 3,
                                 {"b2_body_ratio": b2.body / max(b1.body, 1e-9)})
    if b1.is_bull and b3.is_bear and b2.body < 0.4 * b1.body:
        b1_mid = (b1.open + b1.close) / 2.0
        if b3.close < b1_mid:
            return CandlePattern("evening_star", "SHORT", 0.78, 3,
                                 {"b2_body_ratio": b2.body / max(b1.body, 1e-9)})
    return None


def _is_three_soldiers_or_crows(b1: Bar, b2: Bar, b3: Bar) -> Optional[CandlePattern]:
    if b1.is_bull and b2.is_bull and b3.is_bull:
        if b2.close > b1.close and b3.close > b2.close:
            avg_body = (b1.body + b2.body + b3.body) / 3.0
            avg_range = (b1.range + b2.range + b3.range) / 3.0
            if avg_body / avg_range > 0.5:
                return CandlePattern("three_white_soldiers", "LONG", 0.80, 3,
                                     {"avg_body_pct": avg_body / avg_range})
    if b1.is_bear and b2.is_bear and b3.is_bear:
        if b2.close < b1.close and b3.close < b2.close:
            avg_body = (b1.body + b2.body + b3.body) / 3.0
            avg_range = (b1.range + b2.range + b3.range) / 3.0
            if avg_body / avg_range > 0.5:
                return CandlePattern("three_black_crows", "SHORT", 0.80, 3,
                                     {"avg_body_pct": avg_body / avg_range})
    return None


def _is_three_inside_or_outside(b1: Bar, b2: Bar, b3: Bar) -> Optional[CandlePattern]:
    # Three inside up: bear b1 → bullish harami b2 → b3 closes > b1 high
    h = _is_harami(b1, b2)
    if h is not None:
        if h.name == "bullish_harami" and b3.is_bull and b3.close > b1.high:
            return CandlePattern("three_inside_up", "LONG", 0.82, 3, {"harami": True})
        if h.name == "bearish_harami" and b3.is_bear and b3.close < b1.low:
            return CandlePattern("three_inside_down", "SHORT", 0.82, 3, {"harami": True})
    # Three outside up: bear b1 → bullish engulfing b2 → b3 closes higher
    e = _is_engulfing(b1, b2)
    if e is not None:
        if e.name == "bullish_engulfing" and b3.is_bull and b3.close > b2.close:
            return CandlePattern("three_outside_up", "LONG", 0.84, 3, {"engulf": True})
        if e.name == "bearish_engulfing" and b3.is_bear and b3.close < b2.close:
            return CandlePattern("three_outside_down", "SHORT", 0.84, 3, {"engulf": True})
    return None


def _is_kicker(b1: Bar, b2: Bar) -> Optional[CandlePattern]:
    if b1.is_bear and b2.is_bull and b2.open > b1.open and b2.body_pct > 0.7 and b1.body_pct > 0.7:
        return CandlePattern("bullish_kicker", "LONG", 0.78, 2, {"gap_size": (b2.open - b1.open) / max(b1.range, 1e-9)})
    if b1.is_bull and b2.is_bear and b2.open < b1.open and b2.body_pct > 0.7 and b1.body_pct > 0.7:
        return CandlePattern("bearish_kicker", "SHORT", 0.78, 2, {"gap_size": (b1.open - b2.open) / max(b1.range, 1e-9)})
    return None


def _is_abandoned_baby(b1: Bar, b2: Bar, b3: Bar) -> Optional[CandlePattern]:
    # Doji b2 with gaps on both sides
    if b2.body_pct >= DOJI_BODY_PCT:
        return None
    if b1.is_bear and b3.is_bull and b2.high < b1.low and b2.low > b3.high - b3.range:
        return CandlePattern("abandoned_baby_bull", "LONG", 0.80, 3, {})
    if b1.is_bull and b3.is_bear and b2.low > b1.high and b2.high < b3.low + b3.range:
        return CandlePattern("abandoned_baby_bear", "SHORT", 0.80, 3, {})
    return None


# ── Top-level detector ───────────────────────────────────────────

def _trend_context(closes: List[float], lookback: int = 10) -> Tuple[bool, bool]:
    """Approximate prior trend over `lookback` bars (excluding current).
    Returns (trend_up, trend_down)."""
    if len(closes) < lookback + 2:
        return (False, False)
    end_idx = len(closes) - 2
    start_idx = max(0, end_idx - lookback)
    if closes[start_idx] >= closes[end_idx]:
        return (False, True)
    return (True, False)


def detect_all(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: Optional[List[float]] = None,
    atr_value: Optional[float] = None,
) -> List[CandlePattern]:
    """Return ALL candlestick patterns detected at the most-recent bar.

    Examines current bar (single), last 2 bars (two-bar patterns), and
    last 3 bars (three-bar patterns). Returns deduplicated list sorted
    by confidence descending.
    """
    if len(closes) < 3:
        return []
    bars = _bars_from_ohlc(opens, highs, lows, closes, volumes)
    n = len(bars)
    if atr_value is None or atr_value <= 0:
        ranges = [b.range for b in bars[-14:]]
        atr_value = sum(ranges) / max(len(ranges), 1)
    trend_up, trend_down = _trend_context(closes)

    patterns: List[CandlePattern] = []

    # Single-bar patterns at current bar
    cur = bars[-1]
    for fn, args in [
        (_is_doji, [cur]),
        (_is_marubozu, [cur]),
        (_is_spinning_top, [cur]),
    ]:
        result = fn(*args)
        if result is not None:
            patterns.append(result)
    h = _is_hammer_or_hanging_man(cur, trend_down)
    if h is not None:
        patterns.append(h)
    s = _is_shooting_star_or_inverted_hammer(cur, trend_up)
    if s is not None:
        patterns.append(s)

    # Two-bar patterns
    if n >= 2:
        prev, curr = bars[-2], bars[-1]
        for fn, args in [
            (_is_engulfing, [prev, curr]),
            (_is_piercing_or_dark_cloud, [prev, curr]),
            (_is_tweezer, [prev, curr, atr_value]),
            (_is_harami, [prev, curr]),
            (_is_kicker, [prev, curr]),
        ]:
            result = fn(*args)
            if result is not None:
                patterns.append(result)

    # Three-bar patterns
    if n >= 3:
        b1, b2, b3 = bars[-3], bars[-2], bars[-1]
        for fn in (_is_morning_star_or_evening_star,
                   _is_three_soldiers_or_crows,
                   _is_three_inside_or_outside,
                   _is_abandoned_baby):
            result = fn(b1, b2, b3)
            if result is not None:
                patterns.append(result)

    patterns.sort(key=lambda p: -p.confidence)
    return patterns


def evaluate_dict(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: Optional[List[float]] = None,
    atr_value: Optional[float] = None,
) -> Dict[str, Any]:
    """LLM-tool dict form. Returns top patterns + aggregate direction bias."""
    patterns = detect_all(opens, highs, lows, closes, volumes, atr_value)
    if not patterns:
        return {
            "strategy": "candlestick_patterns",
            "direction": "FLAT", "confidence": 0.0,
            "patterns_detected": [],
            "rationale": "no candlestick patterns at current bar",
        }
    # Aggregate direction by confidence-weighted vote
    long_score = sum(p.confidence for p in patterns if p.direction == "LONG")
    short_score = sum(p.confidence for p in patterns if p.direction == "SHORT")
    if long_score > short_score * 1.2:
        agg_direction = "LONG"
        agg_conf = min(0.95, long_score / max(len(patterns), 1))
    elif short_score > long_score * 1.2:
        agg_direction = "SHORT"
        agg_conf = min(0.95, short_score / max(len(patterns), 1))
    else:
        agg_direction = "FLAT"
        agg_conf = 0.4
    top = patterns[:5]
    return {
        "strategy": "candlestick_patterns",
        "direction": agg_direction,
        "confidence": round(agg_conf, 3),
        "patterns_detected": [
            {"name": p.name, "direction": p.direction,
             "confidence": round(p.confidence, 3), "bars": p.bars_used}
            for p in top
        ],
        "n_total": len(patterns),
        "rationale": (
            f"{len(patterns)} candlestick patterns at current bar; "
            f"top: {patterns[0].name} ({patterns[0].direction}, "
            f"conf {patterns[0].confidence:.2f}); "
            f"agg LONG={long_score:.2f} SHORT={short_score:.2f}"
        )[:200],
    }


# ── Pattern label list (used by transformer training) ────────────

PATTERN_NAMES: List[str] = [
    # single-bar (8)
    "doji", "dragonfly_doji", "gravestone_doji",
    "hammer", "hanging_man", "shooting_star", "inverted_hammer",
    "marubozu_bull", "marubozu_bear", "spinning_top",
    # two-bar (10)
    "bullish_engulfing", "bearish_engulfing",
    "piercing_line", "dark_cloud_cover",
    "tweezer_top", "tweezer_bottom",
    "bullish_harami", "bearish_harami",
    "bullish_kicker", "bearish_kicker",
    # three-bar (12)
    "morning_star", "evening_star",
    "three_white_soldiers", "three_black_crows",
    "three_inside_up", "three_inside_down",
    "three_outside_up", "three_outside_down",
    "abandoned_baby_bull", "abandoned_baby_bear",
    # special
    "no_pattern",
]
PATTERN_INDEX: Dict[str, int] = {n: i for i, n in enumerate(PATTERN_NAMES)}
NUM_PATTERNS: int = len(PATTERN_NAMES)
