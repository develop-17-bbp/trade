"""
Trendline Detection & Breakout System
=======================================
Draws diagonal trendlines connecting swing highs and swing lows,
just like a human trader draws on charts.

Multi-timeframe trendlines:
  - 5m: Short-term micro trendlines (10-50 bars)
  - 1h: Medium-term trendlines (20-100 bars)
  - 4h: Major trendlines (20-100 bars)

Breakout signals:
  - Price crossing ABOVE a downtrend resistance line = BULLISH breakout
  - Price crossing BELOW an uptrend support line = BEARISH breakout
  - Confirmed when close is beyond the line (not just wick)

Combined with EMA(8) strategy:
  - Trendline breakout + EMA new line = HIGH CONFIDENCE entry
  - Trendline breakout alone = entry score boost
  - Trading against a trendline = entry score penalty
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Trendline:
    """A diagonal line connecting two swing points."""
    start_idx: int          # Bar index of first point
    start_price: float      # Price at first point
    end_idx: int            # Bar index of second point
    end_price: float        # Price at second point
    slope: float            # Price change per bar
    intercept: float        # Price at bar 0
    line_type: str          # 'resistance' (connects highs) or 'support' (connects lows)
    touches: int            # How many times price touched this line (more = stronger)
    timeframe: str          # Which TF this was detected on
    strength: float         # 0-1 strength score

    def price_at(self, bar_idx: int) -> float:
        """Get the trendline price at any bar index."""
        return self.slope * bar_idx + self.intercept

    def distance_pct(self, bar_idx: int, current_price: float) -> float:
        """Distance from price to trendline as percentage."""
        line_price = self.price_at(bar_idx)
        if line_price == 0:
            return 0.0
        return (current_price - line_price) / line_price * 100


@dataclass
class TrendlineBreakout:
    """A detected trendline breakout event."""
    direction: int          # +1 = bullish break above resistance, -1 = bearish break below support
    trendline: Trendline    # The broken trendline
    break_price: float      # Price where break occurred
    break_bar: int          # Bar index of the break
    distance_pct: float     # How far past the line (% of price)
    strength: float         # Breakout strength (0-1)


def find_swing_highs(highs: List[float], window: int = 5) -> List[Tuple[int, float]]:
    """Find local swing highs (peaks)."""
    swings = []
    n = len(highs)
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window:i + window + 1]):
            swings.append((i, highs[i]))
    return swings


def find_swing_lows(lows: List[float], window: int = 5) -> List[Tuple[int, float]]:
    """Find local swing lows (troughs)."""
    swings = []
    n = len(lows)
    for i in range(window, n - window):
        if lows[i] == min(lows[i - window:i + window + 1]):
            swings.append((i, lows[i]))
    return swings


def fit_trendline(point1: Tuple[int, float], point2: Tuple[int, float]) -> Tuple[float, float]:
    """
    Fit a line through two points.
    Returns (slope, intercept) where price = slope * bar_idx + intercept.
    """
    idx1, price1 = point1
    idx2, price2 = point2
    if idx2 == idx1:
        return 0.0, price1
    slope = (price2 - price1) / (idx2 - idx1)
    intercept = price1 - slope * idx1
    return slope, intercept


def count_touches(highs: List[float], lows: List[float], closes: List[float],
                  slope: float, intercept: float, line_type: str,
                  start_idx: int, end_idx: int,
                  tolerance_pct: float = 0.15) -> int:
    """
    Count how many times price touched or came within tolerance of the trendline.
    More touches = stronger trendline.
    """
    touches = 0
    for i in range(start_idx, min(end_idx + 1, len(closes))):
        line_price = slope * i + intercept
        if line_price <= 0:
            continue
        tol = line_price * tolerance_pct / 100

        if line_type == 'resistance':
            # Price high came close to or touched the line from below
            if abs(highs[i] - line_price) <= tol or (highs[i] >= line_price - tol and closes[i] <= line_price + tol):
                touches += 1
        else:
            # Price low came close to or touched the line from above
            if abs(lows[i] - line_price) <= tol or (lows[i] <= line_price + tol and closes[i] >= line_price - tol):
                touches += 1
    return touches


def detect_trendlines(highs: List[float], lows: List[float], closes: List[float],
                      window: int = 5, min_bars_apart: int = 10,
                      max_lines: int = 6, timeframe: str = '5m') -> List[Trendline]:
    """
    Detect significant trendlines by connecting swing points.

    Strategy:
    1. Find swing highs -> draw RESISTANCE lines (descending = downtrend resistance)
    2. Find swing lows -> draw SUPPORT lines (ascending = uptrend support)
    3. Score each line by number of touches
    4. Return the strongest lines

    Args:
        highs, lows, closes: OHLC data
        window: Swing detection window
        min_bars_apart: Minimum bars between two points forming a line
        max_lines: Maximum trendlines to return
        timeframe: Label for which TF

    Returns:
        List of Trendline objects, sorted by strength
    """
    n = len(closes)
    if n < window * 3:
        return []

    swing_highs = find_swing_highs(highs, window)
    swing_lows = find_swing_lows(lows, window)

    trendlines = []

    # --- RESISTANCE lines (connecting swing highs) ---
    for i in range(len(swing_highs)):
        for j in range(i + 1, len(swing_highs)):
            idx1, p1 = swing_highs[i]
            idx2, p2 = swing_highs[j]

            # Must be far enough apart
            if idx2 - idx1 < min_bars_apart:
                continue

            # Only use recent swing points (last 60% of data for relevance)
            if idx2 < n * 0.4:
                continue

            slope, intercept = fit_trendline((idx1, p1), (idx2, p2))

            # Validate: line shouldn't be crossed by too many bars between the points
            violations = 0
            for k in range(idx1, idx2):
                if highs[k] > slope * k + intercept + p1 * 0.003:
                    violations += 1
            violation_rate = violations / max(1, idx2 - idx1)
            if violation_rate > 0.15:
                continue  # Too many violations — not a valid trendline

            touches = count_touches(highs, lows, closes, slope, intercept,
                                   'resistance', idx1, idx2)

            # Strength: based on touches + length + recency
            length_score = min((idx2 - idx1) / (n * 0.3), 1.0)
            recency_score = min(idx2 / n, 1.0)
            touch_score = min(touches / 5, 1.0)
            strength = touch_score * 0.5 + length_score * 0.25 + recency_score * 0.25

            trendlines.append(Trendline(
                start_idx=idx1, start_price=p1,
                end_idx=idx2, end_price=p2,
                slope=slope, intercept=intercept,
                line_type='resistance',
                touches=touches, timeframe=timeframe,
                strength=strength,
            ))

    # --- SUPPORT lines (connecting swing lows) ---
    for i in range(len(swing_lows)):
        for j in range(i + 1, len(swing_lows)):
            idx1, p1 = swing_lows[i]
            idx2, p2 = swing_lows[j]

            if idx2 - idx1 < min_bars_apart:
                continue
            if idx2 < n * 0.4:
                continue

            slope, intercept = fit_trendline((idx1, p1), (idx2, p2))

            # Validate: line shouldn't be crossed by too many bars
            violations = 0
            for k in range(idx1, idx2):
                if lows[k] < slope * k + intercept - p1 * 0.003:
                    violations += 1
            violation_rate = violations / max(1, idx2 - idx1)
            if violation_rate > 0.15:
                continue

            touches = count_touches(highs, lows, closes, slope, intercept,
                                   'support', idx1, idx2)

            length_score = min((idx2 - idx1) / (n * 0.3), 1.0)
            recency_score = min(idx2 / n, 1.0)
            touch_score = min(touches / 5, 1.0)
            strength = touch_score * 0.5 + length_score * 0.25 + recency_score * 0.25

            trendlines.append(Trendline(
                start_idx=idx1, start_price=p1,
                end_idx=idx2, end_price=p2,
                slope=slope, intercept=intercept,
                line_type='support',
                touches=touches, timeframe=timeframe,
                strength=strength,
            ))

    # Sort by strength and return top N
    trendlines.sort(key=lambda t: t.strength, reverse=True)
    return trendlines[:max_lines]


def detect_breakouts(highs: List[float], lows: List[float], closes: List[float],
                     trendlines: List[Trendline], bar_idx: int,
                     min_break_pct: float = 0.05) -> List[TrendlineBreakout]:
    """
    Detect if current bar breaks any active trendlines.

    Breakout conditions:
    - BULLISH: Close > resistance trendline + buffer
    - BEARISH: Close < support trendline - buffer

    Only counts as breakout if:
    1. Close (not just wick) is past the line
    2. Distance past the line > min_break_pct
    3. Trendline extends to current bar (extrapolated)

    Args:
        bar_idx: Current bar index
        min_break_pct: Minimum % past the line to confirm break

    Returns:
        List of breakout events
    """
    if bar_idx >= len(closes):
        return []

    close = closes[bar_idx]
    breakouts = []

    for tl in trendlines:
        # Only check lines that extend to or past current bar
        if tl.end_idx > bar_idx:
            continue  # Line hasn't formed yet

        line_price = tl.price_at(bar_idx)
        if line_price <= 0:
            continue

        dist_pct = (close - line_price) / line_price * 100

        if tl.line_type == 'resistance' and dist_pct > min_break_pct:
            # Bullish breakout: price broke above resistance
            breakouts.append(TrendlineBreakout(
                direction=1,
                trendline=tl,
                break_price=close,
                break_bar=bar_idx,
                distance_pct=dist_pct,
                strength=min(1.0, tl.strength * (1 + dist_pct / 2)),
            ))

        elif tl.line_type == 'support' and dist_pct < -min_break_pct:
            # Bearish breakout: price broke below support
            breakouts.append(TrendlineBreakout(
                direction=-1,
                trendline=tl,
                break_price=close,
                break_bar=bar_idx,
                distance_pct=abs(dist_pct),
                strength=min(1.0, tl.strength * (1 + abs(dist_pct) / 2)),
            ))

    return breakouts


def get_trendline_context(highs: List[float], lows: List[float], closes: List[float],
                          bar_idx: int, timeframe: str = '5m',
                          window: int = 5) -> Dict:
    """
    Get trendline analysis for the current bar.
    Returns context dict that can be used for entry scoring.

    This is the main function called by the trading system.
    """
    n = len(closes)
    if n < 50 or bar_idx < 50:
        return {
            'trendline_breakout': 0,
            'trendline_strength': 0.0,
            'nearest_resistance_pct': 999.0,
            'nearest_support_pct': -999.0,
            'trendline_count': 0,
            'breakout_details': '',
        }

    # Use data up to current bar for detection
    h = highs[:bar_idx + 1]
    l = lows[:bar_idx + 1]
    c = closes[:bar_idx + 1]

    # Detect trendlines
    trendlines = detect_trendlines(h, l, c, window=window,
                                    min_bars_apart=max(10, window * 2),
                                    max_lines=8, timeframe=timeframe)

    # Detect breakouts at current bar
    breakouts = detect_breakouts(h, l, c, trendlines, bar_idx)

    # Find nearest resistance and support distances
    nearest_res_pct = 999.0
    nearest_sup_pct = -999.0
    close = closes[bar_idx]

    for tl in trendlines:
        line_price = tl.price_at(bar_idx)
        if line_price <= 0:
            continue
        dist = (line_price - close) / close * 100

        if tl.line_type == 'resistance' and 0 < dist < nearest_res_pct:
            nearest_res_pct = dist
        elif tl.line_type == 'support' and dist < 0 and dist > nearest_sup_pct:
            nearest_sup_pct = dist

    # Determine strongest breakout signal
    breakout_dir = 0
    breakout_strength = 0.0
    breakout_detail = ''
    if breakouts:
        best = max(breakouts, key=lambda b: b.strength)
        breakout_dir = best.direction
        breakout_strength = best.strength
        tl = best.trendline
        breakout_detail = (f"{'BULL' if best.direction > 0 else 'BEAR'} break "
                          f"{tl.line_type} ({tl.touches} touches, "
                          f"{tl.timeframe}) +{best.distance_pct:.2f}%")

    return {
        'trendline_breakout': breakout_dir,        # +1=bullish, -1=bearish, 0=none
        'trendline_strength': breakout_strength,    # 0-1
        'nearest_resistance_pct': round(nearest_res_pct, 3),
        'nearest_support_pct': round(nearest_sup_pct, 3),
        'trendline_count': len(trendlines),
        'breakout_details': breakout_detail,
        'active_trendlines': trendlines,            # For downstream use
    }


def multi_timeframe_trendlines(ohlcv_by_tf: Dict[str, Dict],
                                bar_idx_5m: int) -> Dict:
    """
    Compute trendline context across multiple timeframes.

    Trendlines are drawn across the FULL dataset for each timeframe,
    connecting MAJOR swing points — not just local micro-swings.

    Each TF uses a swing window proportional to its period:
    - 5m: window=10 (50 min between swings) — short-term micro structure
    - 15m: window=10 (2.5 hours) — intraday trends
    - 1h: window=15 (15 hours) — daily trend structure
    - 4h: window=15 (2.5 days) — weekly/major trendlines (strongest)

    The 4h trendlines carry most weight because they represent the major
    market structure lines that institutional traders draw on charts.

    Args:
        ohlcv_by_tf: Dict of timeframe -> {'highs': [], 'lows': [], 'closes': [], ...}
        bar_idx_5m: Current bar index on 5m timeframe

    Returns:
        Combined trendline context with multi-TF consensus
    """
    # Larger windows = connecting MAJOR peaks/valleys across full timeframe
    # Weight: 4h lines matter most (like drawn on daily chart)
    tf_configs = {
        '5m':  {'window': 10, 'min_apart': 20,  'weight': 0.15},
        '15m': {'window': 10, 'min_apart': 15,  'weight': 0.20},
        '1h':  {'window': 15, 'min_apart': 15,  'weight': 0.30},
        '4h':  {'window': 15, 'min_apart': 10,  'weight': 0.35},
    }

    # Time ratio for bar index conversion
    tf_bar_ratios = {'5m': 1, '15m': 3, '1h': 12, '4h': 48}

    contexts = {}
    weighted_breakout = 0.0
    total_weight = 0.0

    for tf, cfg in tf_configs.items():
        if tf not in ohlcv_by_tf:
            continue

        data = ohlcv_by_tf[tf]
        if not data or 'closes' not in data:
            continue

        # Convert 5m bar index to this TF's bar index
        ratio = tf_bar_ratios.get(tf, 1)
        tf_bar_idx = bar_idx_5m // ratio

        h = data.get('highs', data.get('closes', []))
        l = data.get('lows', data.get('closes', []))
        c = data['closes']

        if len(c) < 50:
            continue

        tf_bar_idx = min(tf_bar_idx, len(c) - 1)

        # Use full data up to current bar for trendline detection
        # Larger window = connecting major swing points across entire dataset
        tls = detect_trendlines(h[:tf_bar_idx+1], l[:tf_bar_idx+1], c[:tf_bar_idx+1],
                                 window=cfg['window'],
                                 min_bars_apart=cfg.get('min_apart', 10),
                                 max_lines=8, timeframe=tf)

        breakouts = detect_breakouts(h, l, c, tls, tf_bar_idx) if tls else []

        # Build context
        close = c[tf_bar_idx] if tf_bar_idx < len(c) else c[-1]
        nearest_res = 999.0
        nearest_sup = -999.0
        for tl in tls:
            lp = tl.price_at(tf_bar_idx)
            if lp <= 0: continue
            d = (lp - close) / close * 100
            if tl.line_type == 'resistance' and 0 < d < nearest_res:
                nearest_res = d
            elif tl.line_type == 'support' and d < 0 and d > nearest_sup:
                nearest_sup = d

        bo_dir = 0
        bo_str = 0.0
        if breakouts:
            best = max(breakouts, key=lambda b: b.strength)
            bo_dir = best.direction
            bo_str = best.strength

        ctx = {
            'trendline_breakout': bo_dir,
            'trendline_strength': bo_str,
            'nearest_resistance_pct': round(nearest_res, 3),
            'nearest_support_pct': round(nearest_sup, 3),
            'trendline_count': len(tls),
            'breakout_details': '',
            'active_trendlines': tls,
        }
        contexts[tf] = ctx

        # Weighted breakout consensus
        w = cfg['weight']
        weighted_breakout += ctx['trendline_breakout'] * ctx['trendline_strength'] * w
        total_weight += w

    # Consensus
    if total_weight > 0:
        consensus = weighted_breakout / total_weight
    else:
        consensus = 0.0

    # Breakout confirmed across multiple TFs = very strong
    breakout_tfs = sum(1 for ctx in contexts.values() if ctx['trendline_breakout'] != 0)

    return {
        'trendline_consensus': round(consensus, 3),     # +1=strong bullish, -1=strong bearish
        'trendline_breakout_tfs': breakout_tfs,          # How many TFs show breakout
        'trendline_contexts': contexts,                  # Per-TF details
        'trendline_score_adj': _compute_score_adjustment(contexts),
    }


def detect_sr_levels(highs: List[float], lows: List[float], closes: List[float],
                     lookback: int = 100, zone_pct: float = 0.3,
                     min_touches: int = 3) -> List[Dict]:
    """Detect horizontal support/resistance levels from swing point clusters.

    Groups swing highs and lows into price zones. Zones with multiple touches
    (price reversals at similar levels) become S/R levels.

    Args:
        lookback: bars to analyze
        zone_pct: % width of each zone (prices within this band count as same level)
        min_touches: minimum touches to qualify as S/R level

    Returns:
        List of {price, type, touches, strength, distance_pct} sorted by strength
    """
    n = len(closes)
    if n < 30:
        return []

    start = max(0, n - lookback)
    h = highs[start:]
    l = lows[start:]
    c = closes[start:]
    current = c[-1]

    # Find swing highs and swing lows (window=3)
    swing_highs = []
    swing_lows = []
    for i in range(2, len(h) - 2):
        if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
            swing_highs.append(h[i])
        if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
            swing_lows.append(l[i])

    # Combine all swing points
    all_points = [(p, 'high') for p in swing_highs] + [(p, 'low') for p in swing_lows]
    if not all_points:
        return []

    # Cluster points into zones
    all_points.sort(key=lambda x: x[0])
    zones = []
    used = [False] * len(all_points)

    for i in range(len(all_points)):
        if used[i]:
            continue
        center = all_points[i][0]
        zone_width = center * zone_pct / 100.0
        zone_points = []

        for j in range(i, len(all_points)):
            if abs(all_points[j][0] - center) <= zone_width:
                zone_points.append(all_points[j])
                used[j] = True

        if len(zone_points) >= min_touches:
            avg_price = sum(p for p, _ in zone_points) / len(zone_points)
            n_highs = sum(1 for _, t in zone_points if t == 'high')
            n_lows = sum(1 for _, t in zone_points if t == 'low')

            if avg_price > current:
                level_type = 'resistance'
            else:
                level_type = 'support'

            dist_pct = (avg_price - current) / current * 100

            zones.append({
                'price': round(avg_price, 2),
                'type': level_type,
                'touches': len(zone_points),
                'strength': len(zone_points) / max(1, len(all_points)) * 10,
                'distance_pct': round(dist_pct, 3),
            })

    zones.sort(key=lambda z: abs(z['distance_pct']))
    return zones[:10]


def get_sr_score_adjustment(highs: List[float], lows: List[float],
                            closes: List[float], signal: str,
                            lookback: int = 100) -> Dict:
    """Get S/R-based entry score adjustment.

    Penalizes entries near opposing S/R levels:
    - LONG near strong resistance → penalty
    - SHORT near strong support → penalty
    - Breakout through S/R → bonus

    Returns:
        Dict with sr_score_adj, nearest_resistance, nearest_support, sr_details
    """
    levels = detect_sr_levels(highs, lows, closes, lookback=lookback)
    if not levels:
        return {'sr_score_adj': 0, 'sr_details': ''}

    current = closes[-1]
    score_adj = 0
    details = []

    nearest_res = None
    nearest_sup = None

    for lv in levels:
        if lv['type'] == 'resistance' and (nearest_res is None or abs(lv['distance_pct']) < abs(nearest_res['distance_pct'])):
            nearest_res = lv
        if lv['type'] == 'support' and (nearest_sup is None or abs(lv['distance_pct']) < abs(nearest_sup['distance_pct'])):
            nearest_sup = lv

    # LONG near resistance = bad (price likely to bounce down)
    if signal == 'BUY' and nearest_res and 0 < nearest_res['distance_pct'] < 0.5:
        penalty = -1 if nearest_res['touches'] < 4 else -2
        score_adj += penalty
        details.append(f"near_res({nearest_res['price']:.0f},{nearest_res['touches']}t,{nearest_res['distance_pct']:+.2f}%)")

    # SHORT near support = bad (price likely to bounce up)
    if signal == 'SELL' and nearest_sup and -0.5 < nearest_sup['distance_pct'] < 0:
        penalty = -1 if nearest_sup['touches'] < 4 else -2
        score_adj += penalty
        details.append(f"near_sup({nearest_sup['price']:.0f},{nearest_sup['touches']}t,{nearest_sup['distance_pct']:+.2f}%)")

    # LONG breaking above resistance = strong signal
    if signal == 'BUY' and nearest_res and -0.3 < nearest_res['distance_pct'] < 0:
        # Price just broke above resistance
        bonus = 1 if nearest_res['touches'] >= 3 else 0
        score_adj += bonus
        if bonus:
            details.append(f"break_res({nearest_res['price']:.0f},{nearest_res['touches']}t)")

    # SHORT breaking below support = strong signal
    if signal == 'SELL' and nearest_sup and 0 < nearest_sup['distance_pct'] < 0.3:
        bonus = 1 if nearest_sup['touches'] >= 3 else 0
        score_adj += bonus
        if bonus:
            details.append(f"break_sup({nearest_sup['price']:.0f},{nearest_sup['touches']}t)")

    return {
        'sr_score_adj': score_adj,
        'nearest_resistance': nearest_res,
        'nearest_support': nearest_sup,
        'sr_levels': levels,
        'sr_details': ', '.join(details),
    }


def _compute_score_adjustment(contexts: Dict) -> int:
    """
    Compute entry score adjustment based on trendline analysis.

    +2: Multi-TF trendline breakout aligned with trade direction
    +1: Single TF breakout aligned
     0: No trendline signal
    -1: Trading against a trendline (price approaching resistance for LONG)
    -2: Trading against multi-TF trendlines
    """
    breakout_count = 0
    against_count = 0

    for tf, ctx in contexts.items():
        bo = ctx.get('trendline_breakout', 0)
        strength = ctx.get('trendline_strength', 0)

        if bo != 0 and strength > 0.3:
            breakout_count += 1

        # Check if near resistance (bad for longs) or near support (bad for shorts)
        res_dist = ctx.get('nearest_resistance_pct', 999)
        sup_dist = ctx.get('nearest_support_pct', -999)

        if 0 < res_dist < 0.5:
            against_count += 1  # Very close to resistance
        if -0.5 < sup_dist < 0:
            against_count += 1  # Very close to support

    if breakout_count >= 2:
        return +2
    elif breakout_count >= 1:
        return +1
    elif against_count >= 2:
        return -2
    elif against_count >= 1:
        return -1
    return 0
