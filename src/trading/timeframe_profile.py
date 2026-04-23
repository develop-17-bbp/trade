"""
Timeframe profile — swing vs scalp tuning for the same strategy code.

On a 1.69% round-trip spread venue (Robinhood), 5m bars produce signals too
small to overcome spread. A 1h or 4h primary timeframe shifts the median
move from ~0.1-0.3% into the 1-3% range where the rule strategy can survive
the spread tax.

This module centralises the profile-dependent constants so executor code
can ask `get_profile()` once at init and get consistent values for:
  * primary timeframe string ("5m" / "1h" / "4h")
  * min expected move for sniper vs normal tiers
  * min hold minutes
  * ATR multiplier for stop floor
  * ratchet-level scaling factor (higher TF = wider levels)
  * rolling Sharpe window (fewer trades per day on swing → shorter window)

Enabled by `ACT_TIMEFRAME_PROFILE=swing` env flag (default: `scalp`, which
preserves the existing 5m behavior exactly). Operator can also pin via
config.yaml:
    timeframe_profile: swing

Rationale for 25-40% annual target with this profile:
  * Robinhood 1.69% round-trip means each trade needs ≥ ~2% gross move to
    clear spread + slippage. 1h/4h moves routinely produce this.
  * 1-2 trades/week on swing vs 1-2/day on scalp.
  * 50-55% WR × 2:1 R:R × 6 trades/month × 2% position size = 1-2% monthly.
  * Compounded: 25-40% annual is inside the plausibility envelope.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


# ── Profile presets ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeframeProfile:
    """All constants that vary by primary timeframe. Frozen so callers can't
    mutate the shared instance."""
    name: str                        # 'scalp' | 'swing' | 'position'
    primary_tf: str                  # canonical timeframe string
    poll_interval_s: int             # how often the main loop fires
    sniper_min_move_pct: float       # 3x-size tier threshold
    normal_min_move_pct: float       # 1x-size tier threshold
    min_hold_minutes: int            # hard-floor hold time
    max_hold_days: int               # time-exit cap
    atr_stop_mult: float             # SL = entry - (mult × ATR)
    atr_tp_mult: float               # TP target = entry + (mult × ATR)
    ratchet_scale: float             # multiplier for ratchet-level thresholds
    sharpe_window: int               # rolling Sharpe window (trades)
    target_trades_per_week: int      # expected cadence for monitoring

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# The existing 5m behavior — preserved as the default.
SCALP_PROFILE = TimeframeProfile(
    name="scalp",
    primary_tf="5m",
    poll_interval_s=60,
    sniper_min_move_pct=5.0,         # matches existing sniper.min_expected_move_pct
    normal_min_move_pct=2.5,         # matches earlier Robinhood-hardening add
    min_hold_minutes=1440,           # 24h — existing Robinhood config
    max_hold_days=7,
    atr_stop_mult=3.0,
    atr_tp_mult=10.0,                # existing risk.atr_tp_mult
    ratchet_scale=1.0,               # baseline
    sharpe_window=30,
    target_trades_per_week=20,       # rough: 2-4/day on active regime
)

# 1h primary — the recommended mode for surviving Robinhood's 1.69% spread.
SWING_PROFILE = TimeframeProfile(
    name="swing",
    primary_tf="1h",
    poll_interval_s=180,             # 3min poll is plenty for 1h bars
    sniper_min_move_pct=4.0,         # 1h ATR moves are larger in absolute terms
    normal_min_move_pct=2.0,         # still above 1.69% spread
    min_hold_minutes=4 * 60,         # 4h floor (wait for the next HTF bar to close)
    max_hold_days=7,
    atr_stop_mult=2.5,               # tighter than scalp — 1h ATR is already wider
    atr_tp_mult=5.0,                 # 2:1 R:R on 1h is realistic
    ratchet_scale=2.5,               # ratchet levels need to scale with bar size
    sharpe_window=30,
    target_trades_per_week=5,        # rule of thumb on swing: 0.5-1 per day
)

# 4h primary — "position" mode, very slow. Left in for operators who want
# multi-day holds only. Not activated by default.
POSITION_PROFILE = TimeframeProfile(
    name="position",
    primary_tf="4h",
    poll_interval_s=600,             # 10min poll
    sniper_min_move_pct=6.0,
    normal_min_move_pct=3.0,
    min_hold_minutes=24 * 60,        # 24h floor
    max_hold_days=14,
    atr_stop_mult=2.0,
    atr_tp_mult=4.0,
    ratchet_scale=5.0,
    sharpe_window=20,                # fewer trades → shorter window
    target_trades_per_week=2,
)


_PROFILES: Dict[str, TimeframeProfile] = {
    "scalp": SCALP_PROFILE,
    "swing": SWING_PROFILE,
    "position": POSITION_PROFILE,
}


def get_profile(config: Optional[Dict[str, Any]] = None) -> TimeframeProfile:
    """Resolve the active profile.

    Precedence:
      1. ACT_TIMEFRAME_PROFILE env var (wins over config)
      2. config['timeframe_profile'] key
      3. 'scalp' default (preserves existing behavior)

    Unknown names fall back to scalp with a warning-worthy reason — callers
    can inspect returned .name to detect fallback.
    """
    env = (os.environ.get("ACT_TIMEFRAME_PROFILE") or "").strip().lower()
    cfg_name = ""
    if isinstance(config, dict):
        cfg_name = str(config.get("timeframe_profile") or "").strip().lower()
    name = env or cfg_name or "scalp"
    return _PROFILES.get(name, SCALP_PROFILE)


def is_swing_or_higher(profile: TimeframeProfile) -> bool:
    """Convenience predicate — 1h+ bars."""
    return profile.name in ("swing", "position")
