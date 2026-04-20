"""
Authority Rules — Non-Negotiable Trading Directives
=====================================================
Official strategies and universal rules from the authority PDF.
This module is the single source of truth for ACT's authority compliance.

Used by:
  - prompt_constraints.py  → injects AUTHORITY_SYSTEM_PROMPT into every LLM call
  - authority_compliance_guardian.py → validates each trade against these rules (VETO power)
  - genetic_strategy_engine.py → seeds initial DNA from authority strategies
"""

from typing import Dict, Any, List, Tuple


# ═══════════════════════════════════════════════════════════════
# TRADE TYPE PERMISSIONS PER ASSET
# ═══════════════════════════════════════════════════════════════

ASSET_TRADE_PERMISSIONS: Dict[str, List[str]] = {
    # BTC: all trade types allowed
    'BTC': ['scalp', 'intraday', 'swing'],
    'BTC-USD': ['scalp', 'intraday', 'swing'],
    'BTCUSD': ['scalp', 'intraday', 'swing'],
    'BTCUSDT': ['scalp', 'intraday', 'swing'],
    # ETH + alts: day trades ONLY, never swing
    'ETH': ['scalp', 'intraday'],
    'ETH-USD': ['scalp', 'intraday'],
    'ETHUSD': ['scalp', 'intraday'],
    'ETHUSDT': ['scalp', 'intraday'],
}


# ═══════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME HIERARCHY
# ═══════════════════════════════════════════════════════════════

TIMEFRAME_HIERARCHY: Dict[str, Dict[str, str]] = {
    'swing':    {'trend_tf': '1d',  'entry_tf': '1h',  'hold_range': '3-10 days'},
    'intraday': {'trend_tf': '4h',  'entry_tf': '15m', 'hold_range': '12-48h'},
    'scalp':    {'trend_tf': '1h',  'entry_tf': '5m',  'hold_range': '2-8h'},
}


# ═══════════════════════════════════════════════════════════════
# MAX-HOLD CAPS PER ASSET (hard exit regardless of PnL)
# ═══════════════════════════════════════════════════════════════
#
# Derived from ASSET_TRADE_PERMISSIONS + TIMEFRAME_HIERARCHY top-bound:
#   BTC can swing  → top bound = 10 days  (swing: 3-10 days)
#   ETH/alts intraday only → top bound = 48h (intraday: 12-48h)
#
# The executor MUST close any position that exceeds this cap even if the
# trade is profitable — otherwise an ETH position held 3 days is by
# definition a swing trade on an asset that's banned from swinging.
AUTHORITY_MAX_HOLD_HOURS: Dict[str, float] = {
    'BTC':    240.0,   # 10 days
    'BTCUSD': 240.0, 'BTC-USD': 240.0, 'BTCUSDT': 240.0,
    'ETH':    48.0,    # 48 hours — intraday cap, never swing
    'ETHUSD': 48.0, 'ETH-USD': 48.0, 'ETHUSDT': 48.0,
}

# Default for any unlisted alt — 48h, the intraday ceiling.
DEFAULT_MAX_HOLD_HOURS: float = 48.0


def get_max_hold_hours(asset: str) -> float:
    """Return the authority-mandated max hold in hours for `asset`.

    Unknown assets default to DEFAULT_MAX_HOLD_HOURS (intraday ceiling) so
    the safest interpretation applies when a caller passes something like
    SOL or ARB that hasn't been explicitly listed.
    """
    key = str(asset or '').upper().strip()
    return AUTHORITY_MAX_HOLD_HOURS.get(key, DEFAULT_MAX_HOLD_HOURS)


# ═══════════════════════════════════════════════════════════════
# OFFICIAL STRATEGIES (authority-blessed — seed for genetic engine)
# ═══════════════════════════════════════════════════════════════

STRATEGY_400_EMA_TWO_CANDLE = {
    'name': '400_ema_two_candle_closure',
    'description': 'Two consecutive full-body candles close above/below 400 EMA on trend TF',
    'indicators': ['EMA_400'],
    'entry_rule': 'Two full-body candles close beyond 400 EMA, mark their high/low range, '
                  'enter on lower TF when strong-body candle closes beyond signal range',
    'sl_rule': 'Beyond opposite side of signal range + 0.3-0.5% buffer (BTC)',
    'invalidation': 'Close back inside signal range',
    'retest_allowed': True,
}

STRATEGY_THREE_CANDLE = {
    'name': 'three_candle_formation',
    'description': 'Three consecutive candles define clear high/low range',
    'indicators': ['CANDLE_BODY_AVG_10_50'],
    'entry_rule': 'Breakout or retracement entry on lower TF, strong-body filter required',
    'sl_rule': 'Beyond opposite side of 3-candle range + volatility buffer',
    'strong_body_filter': 'body > average of last 10-50 bars',
}

STRATEGY_REGIME_MEAN_REVERSION = {
    'name': 'regime_gated_mean_reversion',
    'description': 'Mean reversion ONLY in CHOP or LOW_VOL regime — non-negotiable',
    'indicators': ['RSI_14', 'BBANDS_20', 'MA_20'],
    'allowed_regimes': ['CHOP', 'LOW_VOL', 'RANGING'],
    'long_entry': 'RSI(14) < 25 AND price at lower Bollinger Band on 15m',
    'short_entry': 'RSI(14) > 75 AND price at upper Bollinger Band on 15m',
    'target': 'Return to 20-period MA (exit at mean, NOT opposite band)',
    'size_multiplier': 0.5,  # half of trend-following size
}

AUTHORITY_STRATEGIES = [
    STRATEGY_400_EMA_TWO_CANDLE,
    STRATEGY_THREE_CANDLE,
    STRATEGY_REGIME_MEAN_REVERSION,
]


# ═══════════════════════════════════════════════════════════════
# FAKEOUT FILTERS (all 4 required on 5m/15m)
# ═══════════════════════════════════════════════════════════════

FAKEOUT_FILTERS = [
    'unusual_candle',       # body > 3x avg OR volume > 5x avg
    'liquidity_sweep',      # break + close back inside = skip
    'double_top_bottom',    # wait for pattern completion
    'back_to_zone_reentry', # after sweep, only enter on clean rejection
]


# ═══════════════════════════════════════════════════════════════
# UNIVERSAL RULES (never violate)
# ═══════════════════════════════════════════════════════════════

UNIVERSAL_RULES = [
    'higher_tf_trend_must_agree',     # Never trade without higher TF trend agreeing
    'never_enter_on_wick',            # Always wait for close
    'never_enter_small_body',         # Body must exceed avg
    'never_widen_stop',               # Stop never moves against position
    'never_add_to_losing',            # Scale in only on winners
    'news_blackout',                  # Flatten 15min before major news, wait 2 bars after
    'parameter_change_requires_approval',  # Every config change needs authority approval
]


# ═══════════════════════════════════════════════════════════════
# AUTHORITY SYSTEM PROMPT (injected into every LLM call)
# ═══════════════════════════════════════════════════════════════

AUTHORITY_SYSTEM_PROMPT = """
═══════════════════════════════════════════════════════════════
AUTHORITY DIRECTIVES (NON-NEGOTIABLE — from superiors)
═══════════════════════════════════════════════════════════════

These rules come from the AUTHORITY / SUPERIORS and override all other
strategy guidance. Violating any of these is grounds for automatic VETO
by the Authority Compliance Guardian agent.

## ASSET-SPECIFIC PERMISSIONS
- BTC: scalp, intraday, AND swing trades allowed
- ETH and high-beta alts: DAY TRADES ONLY — NEVER swing trade
- Parameters are tuned per asset; do not cross-contaminate

## MULTI-TIMEFRAME HIERARCHY (higher TF always wins conflicts)
| Trade Type | Trend TF | Entry TF | Hold Range |
|------------|----------|----------|------------|
| Swing      | Daily    | H1       | 3-10 days  |
| Intraday   | H4       | 15m      | 12-48h     |
| Scalp      | H1       | 5m       | 2-8h       |

## OFFICIAL STRATEGIES (use these by name)

**S1: 400 EMA Two-Candle Closure**
Two consecutive full-body candles close beyond 400 EMA on trend TF.
Mark high/low as signal range. Enter on lower TF when strong-body
candle closes beyond signal range. SL beyond opposite side + 0.3-0.5%
buffer (BTC). Retest is a valid second entry. Close back inside =
invalidated.

**S2: Three-Candle Formation**
Three consecutive candles define a high/low range. Entry same as S1.
Strong-body filter: body > average of last 10-50 bars.

**S3: Regime-Gated Mean Reversion**
ONLY in CHOP or LOW_VOL regime. Non-negotiable.
Long: RSI(14) < 25 + lower Bollinger Band on 15m.
Short: RSI(14) > 75 + upper Bollinger Band on 15m.
Target: return to 20-period MA (exit at mean, NOT opposite band).
Size: HALF of trend-following size.

## FAKEOUT FILTERS (all 4 required on 5m/15m before entry)
1. Unusual candle detection (body > 3x avg OR volume > 5x avg)
2. Liquidity sweep recognition (break + close back inside = skip)
3. Double top/bottom (wait for pattern completion)
4. Back-to-zone re-entry (only enter on clean rejection after sweep)

## UNIVERSAL RULES (NEVER VIOLATE)
- Never trade without higher TF trend agreeing
- Never enter on a wick — wait for close
- Never enter on a small-body candle (body must exceed recent avg)
- Never widen a stop after entry
- Never add to a losing position (scale in on winners only)
- Never trade during major scheduled news on lower TFs
  (flatten 15min before, wait 2 bars after)
- Every parameter change requires approval

## BTC-ALT CORRELATION (observational only, until walk-forward validated)
Track rolling correlation 1d/7d/30d + lag correlation at 5m/15m/1H/4H.
When promoted: modifies SIZE on existing entries, does NOT generate
independent trades.

═══════════════════════════════════════════════════════════════
END AUTHORITY DIRECTIVES
═══════════════════════════════════════════════════════════════
"""


# ═══════════════════════════════════════════════════════════════
# VALIDATOR (used by AuthorityComplianceGuardian)
# ═══════════════════════════════════════════════════════════════

def validate_authority_entry(
    quant_state: Dict[str, Any],
    context: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """Check a proposed trade entry against the authority rules.

    Returns (ok, violations). If ok=False, the trade should be vetoed.
    Missing context fields are treated as unknown — rules that cannot be
    verified from available data emit a soft warning but do not veto,
    except the hard rules (asset permission, wick entry, small body,
    widening stop, adding to loser).
    """
    violations: List[str] = []

    raw_signal = context.get('raw_signal', 0)
    if raw_signal == 0:
        return True, []  # No trade proposed — nothing to check

    asset = str(context.get('asset', '')).upper().strip()
    trade_type = str(context.get('trade_type', '')).lower().strip()

    # ── Rule 1: Asset-specific trade-type permission ──
    if asset and trade_type:
        allowed = ASSET_TRADE_PERMISSIONS.get(asset)
        if allowed is not None and trade_type not in allowed:
            violations.append(
                f"ASSET_PERMISSION: {asset} not permitted to {trade_type} "
                f"(allowed: {allowed})"
            )

    # ── Rule 2: Higher-TF trend must agree with entry direction ──
    htf_dir = context.get('htf_trend_direction')  # +1 up, -1 down, 0 neutral
    if htf_dir is not None and htf_dir != 0:
        if (raw_signal > 0 and htf_dir < 0) or (raw_signal < 0 and htf_dir > 0):
            violations.append(
                f"HTF_DISAGREEMENT: entry direction={raw_signal} "
                f"conflicts with higher-TF trend={htf_dir}"
            )

    # ── Rule 3: Never enter on a wick ──
    if context.get('entry_on_wick', False):
        violations.append("WICK_ENTRY: entry based on wick — wait for close")

    # ── Rule 4: Never enter on a small-body candle ──
    body_pct = context.get('candle_body_pct')
    avg_body_pct = context.get('avg_body_pct_10_50')
    if body_pct is not None and avg_body_pct is not None and avg_body_pct > 0:
        if body_pct < avg_body_pct:
            violations.append(
                f"SMALL_BODY: candle body {body_pct:.4f} below avg {avg_body_pct:.4f}"
            )

    # ── Rule 5: Never widen a stop ──
    if context.get('stop_widened', False):
        violations.append("STOP_WIDENED: stop moved further from entry — forbidden")

    # ── Rule 6: Never add to a losing position ──
    if context.get('adding_to_position', False) and context.get('position_is_losing', False):
        violations.append("ADD_TO_LOSER: scaling into losing position — forbidden")

    # ── Rule 7: News blackout window ──
    if context.get('news_blackout_active', False):
        violations.append("NEWS_BLACKOUT: major news window active — no entries")

    # ── Rule 8: Mean-reversion strategy used outside allowed regime ──
    strategy = str(context.get('strategy', '')).lower()
    if 'mean_reversion' in strategy or 'reversion' in strategy:
        regime = str(quant_state.get('regime', context.get('regime', ''))).upper()
        allowed_regimes = STRATEGY_REGIME_MEAN_REVERSION['allowed_regimes']
        if regime and regime not in [r.upper() for r in allowed_regimes]:
            violations.append(
                f"REVERSION_REGIME: mean-reversion not allowed in {regime} "
                f"(allowed: {allowed_regimes})"
            )

    # ── Rule 9: Fakeout filters (all 4 required on 5m/15m) ──
    entry_tf = str(context.get('entry_tf', '')).lower()
    if entry_tf in ('5m', '15m'):
        filters_passed = context.get('fakeout_filters_passed', None)
        if filters_passed is not None:
            # Either a list of filter names, or a count
            if isinstance(filters_passed, (list, tuple, set)):
                missing = [f for f in FAKEOUT_FILTERS if f not in filters_passed]
                if missing:
                    violations.append(f"FAKEOUT_FILTERS: missing {missing}")
            elif isinstance(filters_passed, int) and filters_passed < len(FAKEOUT_FILTERS):
                violations.append(
                    f"FAKEOUT_FILTERS: only {filters_passed}/{len(FAKEOUT_FILTERS)} passed"
                )

    return (len(violations) == 0, violations)


def get_asset_permitted_types(asset: str) -> List[str]:
    """Return list of trade types permitted for an asset (empty = unknown asset)."""
    return list(ASSET_TRADE_PERMISSIONS.get(str(asset).upper().strip(), []))
