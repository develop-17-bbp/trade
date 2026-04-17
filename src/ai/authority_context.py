"""
Authority Context Builder
===========================
Maps live indicator data + OHLCV into the fields the Authority Compliance
Guardian expects. Runs once per cycle inside the orchestrator so every
caller automatically gets authoritative context without having to compute
it themselves.

Fields produced (best-effort — missing data → None, guardian treats as
unknown and does not veto on it):
  - htf_trend_direction     (+1 up, -1 down, 0 neutral)
  - candle_body_pct         (|close-open| / close of last candle)
  - avg_body_pct_10_50      (mean body pct over last 10-50 bars)
  - entry_on_wick           (True if last candle body < 30% of its range)
  - fakeout_filters_passed  (list of filter names the last bar cleared)
  - trade_type              (inferred from strategy hint or default)
  - entry_tf                (passed through from caller if known)
"""

from typing import Dict, Any, List, Optional


# Default timeframe→trade_type mapping from the authority PDF hierarchy
_TF_TO_TRADE_TYPE = {
    '1d': 'swing',
    '1D': 'swing',
    'D': 'swing',
    '4h': 'intraday',
    '4H': 'intraday',
    '1h': 'scalp',
    '1H': 'scalp',
    '15m': 'intraday',  # 15m is typically the entry TF for intraday
    '5m': 'scalp',
}


def _dir_to_int(direction_str: Any) -> int:
    """Convert 'UP'/'DOWN'/etc to +1/-1/0."""
    if direction_str is None:
        return 0
    s = str(direction_str).upper().strip()
    if s in ('UP', 'BULL', 'BULLISH', 'LONG', '1', 'POSITIVE'):
        return 1
    if s in ('DOWN', 'BEAR', 'BEARISH', 'SHORT', '-1', 'NEGATIVE'):
        return -1
    return 0


def _last_n(seq: Optional[list], n: int) -> list:
    if not seq:
        return []
    return list(seq[-n:])


def _compute_body_pct(opens: list, closes: list, idx: int = -1) -> Optional[float]:
    """Return |close - open| / close at index idx (default: last bar)."""
    if not opens or not closes:
        return None
    if abs(idx) > len(opens) or abs(idx) > len(closes):
        return None
    try:
        o = float(opens[idx])
        c = float(closes[idx])
        if c <= 0:
            return None
        return abs(c - o) / c
    except (ValueError, TypeError):
        return None


def _compute_avg_body_pct(opens: list, closes: list, window: int = 30) -> Optional[float]:
    """Mean body percentage over the last `window` bars (authority calls for 10-50)."""
    if not opens or not closes:
        return None
    n = min(len(opens), len(closes), window)
    if n < 5:
        return None
    body_pcts = []
    for i in range(-n, 0):
        bp = _compute_body_pct(opens, closes, i)
        if bp is not None:
            body_pcts.append(bp)
    if not body_pcts:
        return None
    return sum(body_pcts) / len(body_pcts)


def _detect_entry_on_wick(
    opens: list, highs: list, lows: list, closes: list,
    body_ratio_threshold: float = 0.30,
) -> Optional[bool]:
    """Heuristic: True if the last candle's body is <30% of its total range.

    Authority rule: "never enter on a wick — always wait for close". A small
    body with long wicks suggests the close has not confirmed the move.
    """
    if not (opens and highs and lows and closes):
        return None
    try:
        o = float(opens[-1])
        h = float(highs[-1])
        l = float(lows[-1])
        c = float(closes[-1])
    except (ValueError, TypeError, IndexError):
        return None

    total_range = h - l
    if total_range <= 0:
        return None
    body = abs(c - o)
    return (body / total_range) < body_ratio_threshold


def _compute_fakeout_filters(
    highs: list, lows: list, opens: list, closes: list, volumes: list,
    lookback: int = 30,
) -> List[str]:
    """Which of the 4 authority fakeout filters cleared on the last bar.

    Authority requires all 4 on 5m/15m:
      1. unusual_candle    — body > 3x avg OR volume > 5x avg
      2. liquidity_sweep   — break + close back inside invalidates (we pass
                             the filter if NO sweep was detected)
      3. double_top_bottom — no incomplete double-top/bottom pattern
      4. back_to_zone_reentry — after a sweep, require clean rejection

    Returns the names of filters that currently pass. Guardian expects all
    4 names present before allowing 5m/15m entries.
    """
    passed: List[str] = []
    if not (highs and lows and closes and opens):
        return passed

    n = min(len(highs), len(lows), len(closes), len(opens), lookback)
    if n < 5:
        return passed

    try:
        # ── Filter 1: unusual candle (body/volume spike) ──
        last_body = abs(float(closes[-1]) - float(opens[-1]))
        prior_bodies = [abs(float(closes[i]) - float(opens[i])) for i in range(-n, -1)]
        avg_body = sum(prior_bodies) / len(prior_bodies) if prior_bodies else 0
        unusual_body = avg_body > 0 and last_body > 3 * avg_body

        unusual_volume = False
        if volumes and len(volumes) >= n:
            last_vol = float(volumes[-1])
            prior_vols = [float(v) for v in volumes[-n:-1]]
            avg_vol = sum(prior_vols) / len(prior_vols) if prior_vols else 0
            unusual_volume = avg_vol > 0 and last_vol > 5 * avg_vol

        # Filter passes if the entry candle is NOT suspiciously oversized
        # (oversized candles are typical fakeout flags)
        if not (unusual_body or unusual_volume):
            passed.append('unusual_candle')

        # ── Filter 2: liquidity sweep (meaningful break + close back inside) ──
        # Require the break to exceed prior extreme by at least 0.2% to avoid
        # false positives on monotonic trends where every bar nicks the prior high.
        prior_high = max(float(h) for h in highs[-n:-1])
        prior_low = min(float(l) for l in lows[-n:-1])
        last_high = float(highs[-1])
        last_low = float(lows[-1])
        last_close = float(closes[-1])
        sweep_margin = 0.002  # 0.2% minimum break to be considered a sweep

        swept_high = (
            last_high > prior_high * (1 + sweep_margin)
            and last_close < prior_high
        )
        swept_low = (
            last_low < prior_low * (1 - sweep_margin)
            and last_close > prior_low
        )
        if not (swept_high or swept_low):
            passed.append('liquidity_sweep')

        # ── Filter 3: double top/bottom — require two distinct touches with
        # a meaningful retracement between them (not just any cluster near
        # the extreme). Pattern needs ≥3 bars between touches and a pullback
        # of at least 0.5% between them to qualify as a real double-top. ──
        closes_tail = [float(c) for c in closes[-n:]]
        highs_tail = [float(h) for h in highs[-n:]]
        lows_tail = [float(l) for l in lows[-n:]]
        recent_high = max(highs_tail)
        recent_low = min(lows_tail)
        touch_tol = 0.003          # 0.3% tolerance for a "touch"
        pullback_min = 0.005       # 0.5% minimum intervening pullback
        min_separation = 3          # bars between touches

        def _has_double_top():
            touch_idxs = [i for i, h in enumerate(highs_tail)
                          if abs(h - recent_high) / (recent_high + 1e-12) < touch_tol]
            if len(touch_idxs) < 2:
                return False
            for i in range(len(touch_idxs) - 1):
                a, b = touch_idxs[i], touch_idxs[i + 1]
                if b - a < min_separation:
                    continue
                between_low = min(lows_tail[a + 1:b]) if b > a + 1 else recent_high
                if (recent_high - between_low) / (recent_high + 1e-12) >= pullback_min:
                    return True
            return False

        def _has_double_bottom():
            touch_idxs = [i for i, l in enumerate(lows_tail)
                          if abs(l - recent_low) / (recent_low + 1e-12) < touch_tol]
            if len(touch_idxs) < 2:
                return False
            for i in range(len(touch_idxs) - 1):
                a, b = touch_idxs[i], touch_idxs[i + 1]
                if b - a < min_separation:
                    continue
                between_high = max(highs_tail[a + 1:b]) if b > a + 1 else recent_low
                if (between_high - recent_low) / (recent_low + 1e-12) >= pullback_min:
                    return True
            return False

        if not (_has_double_top() or _has_double_bottom()):
            passed.append('double_top_bottom')

        # ── Filter 4: back-to-zone re-entry — require a clean rejection bar
        # after any recent sweep. If no sweep happened, we trivially pass.
        prior_sweep = False
        for i in range(-min(n, 5), -1):
            sw_h = float(highs[i]) > prior_high and float(closes[i]) < prior_high
            sw_l = float(lows[i]) < prior_low and float(closes[i]) > prior_low
            if sw_h or sw_l:
                prior_sweep = True
                break
        if not prior_sweep:
            passed.append('back_to_zone_reentry')
        else:
            # Had a recent sweep — require last bar to be a clean rejection
            body_direction = 1 if float(closes[-1]) > float(opens[-1]) else -1
            # Passes only if the last bar's direction opposes the sweep
            # (heuristic: sweep high then bear candle = rejection)
            if (swept_high and body_direction < 0) or (swept_low and body_direction > 0):
                passed.append('back_to_zone_reentry')
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        pass

    return passed


def build_authority_context(
    quant_state: Dict[str, Any],
    ohlcv_data: Optional[Dict[str, Any]] = None,
    asset: str = '',
    trade_type_hint: Optional[str] = None,
    entry_tf: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract authority-relevant fields from live pipeline data.

    Returns a dict safe to merge into the orchestrator context. Fields that
    can't be computed from the available data default to None so the
    guardian treats them as unknown rather than false.
    """
    ctx: Dict[str, Any] = {}

    # ── Higher-TF trend direction from the trend indicator block ──
    trend = quant_state.get('trend', {}) if isinstance(quant_state.get('trend'), dict) else {}
    htf_dir = _dir_to_int(trend.get('trend_direction'))
    # Only mark a strong trend — weak ADX = neutral (don't trigger HTF veto)
    adx_value = trend.get('adx')
    if isinstance(adx_value, (int, float)) and adx_value < 20:
        htf_dir = 0
    ctx['htf_trend_direction'] = htf_dir

    # ── Candle body metrics ──
    ohlcv = ohlcv_data or {}
    opens = ohlcv.get('opens') or []
    highs = ohlcv.get('highs') or []
    lows = ohlcv.get('lows') or []
    closes = ohlcv.get('closes') or ohlcv.get('prices') or []
    volumes = ohlcv.get('volumes') or []

    ctx['candle_body_pct'] = _compute_body_pct(opens, closes)
    ctx['avg_body_pct_10_50'] = _compute_avg_body_pct(opens, closes, window=30)

    # ── Wick entry detection ──
    ctx['entry_on_wick'] = _detect_entry_on_wick(opens, highs, lows, closes)

    # ── Fakeout filters ──
    ctx['fakeout_filters_passed'] = _compute_fakeout_filters(
        highs, lows, opens, closes, volumes
    )

    # ── Trade type + entry TF ──
    # Caller hint takes precedence; else map entry_tf via authority hierarchy
    if trade_type_hint:
        ctx['trade_type'] = trade_type_hint.lower()
    elif entry_tf and entry_tf in _TF_TO_TRADE_TYPE:
        ctx['trade_type'] = _TF_TO_TRADE_TYPE[entry_tf]
    else:
        # Default: 'intraday' — most conservative (blocks ETH swing,
        # allows BTC/ETH intraday+scalp which are always permitted)
        ctx['trade_type'] = 'intraday'

    if entry_tf:
        ctx['entry_tf'] = entry_tf

    return ctx
