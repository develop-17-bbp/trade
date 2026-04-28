"""Session-based volume bias.

Crypto trading volume is heavily concentrated in specific hour
ranges (UTC). The 2026 literature consistently reports US session
holds ~60% of BTC daily volume, EU session ~30%, Asia ~10%
(approximate; varies by week).

This module provides a session-aware multiplier the brain can use to
size up during high-volume sessions and down during low-volume ones,
without changing the underlying signals.

Anti-overfit design:
  * Multipliers are bounded [0.5, 1.0] — never >1.0 (avoids
    amplifying a high-confidence trade beyond the base risk budget)
  * Sessions are deterministic UTC ranges, not learned — no fitting
    risk
  * Output is advisory; brain reads via tool, decides whether to
    apply
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict


# UTC ranges. Crypto trades 24/7 but volume profile concentrates
# during these windows.
SESSION_RANGES = (
    # (name, start_hour, end_hour, volume_share, conviction_multiplier)
    ("ASIA",     0,  8,  0.10, 0.6),
    ("EU",       7, 14,  0.30, 0.85),
    ("US",      13, 21,  0.55, 1.0),
    ("LATE_US", 21, 24,  0.15, 0.75),
)


def current_session(now_utc: datetime = None) -> Dict[str, Any]:
    """Return the dominant session for the current UTC hour with
    associated volume share + conviction multiplier.

    Overlapping windows (e.g. 13:00 UTC = end of EU + start of US)
    pick the higher-volume session.
    """
    if now_utc is None:
        now_utc = datetime.now(tz=timezone.utc)
    h = now_utc.hour
    # Score each session by whether the hour falls inside it; prefer
    # the highest-volume-share match.
    best = ("UNKNOWN", 0.0, 0.5)
    for name, start, end, share, mult in SESSION_RANGES:
        if start <= h < end:
            if share > best[1]:
                best = (name, share, mult)
    return {
        "session": best[0],
        "hour_utc": h,
        "volume_share": round(float(best[1]), 3),
        "conviction_multiplier": round(float(best[2]), 3),
        "advisory": (
            "Use conviction_multiplier as a SCALE on your confidence "
            "before submit_trade_plan. Range [0.5-1.0] — never amplifies "
            "above base. Skip propositions on ASIA session unless "
            "sniper-tier confluence."
        ),
    }


def is_active_trading_hour(now_utc: datetime = None,
                            min_volume_share: float = 0.25) -> bool:
    """True when the current session has sustained trading volume
    (>= min_volume_share of daily volume). Use as a soft gate when
    reasoning about marginal setups."""
    return current_session(now_utc)["volume_share"] >= min_volume_share
