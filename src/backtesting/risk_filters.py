"""
Backtest Risk Filters — Stateful Filter Chain
===============================================
Replicates the 6-layer filter chain from executor.py.
Uses bar timestamps instead of time.time() for deterministic replay.
"""

from datetime import datetime, timezone
from typing import Dict, Optional


class BacktestFilterChain:
    """Stateful filter chain for backtesting.

    All time-based checks use bar timestamps (from OHLCV data)
    rather than wall clock, enabling deterministic replay.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        filters = config.get('filters', {})

        # Dangerous hours (UTC) — disabled for crypto (24/7 markets)
        self.dangerous_hours = filters.get('dangerous_hours', [])
        self.block_dangerous_hours = len(self.dangerous_hours) > 0

        # Range filter
        self.range_threshold = 0.15  # % — matches executor fix

        # Cooldowns (seconds)
        self.trade_cooldown = 300.0     # 5 min between trades
        cooldown_min = config.get('post_close_cooldown_min', 10)
        self.post_close_cooldown = cooldown_min * 60.0  # configurable
        self.max_loss_streak_cooldown = 900.0  # 15 min max

        # State tracking
        self.last_trade_time: Dict[str, float] = {}  # asset -> timestamp_s
        self.last_close_time: Dict[str, float] = {}
        self.loss_streak: Dict[str, int] = {}
        self.cooldown_until: Dict[str, float] = {}

        # Stats
        self.filter_stats = {
            'dangerous_hours': 0,
            'ranging': 0,
            'trade_cooldown': 0,
            'post_close_cooldown': 0,
            'loss_streak': 0,
        }

    def check_all(self, asset: str, bar_timestamp_ms: int,
                  closes: list) -> tuple:
        """Run all filters. Returns (passed: bool, reason: str)."""

        ts_s = bar_timestamp_ms / 1000.0

        # 1. Dangerous hours
        if self.block_dangerous_hours:
            dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
            if dt.hour in self.dangerous_hours:
                self.filter_stats['dangerous_hours'] += 1
                return False, f"HOUR_BLOCK UTC {dt.hour}:00"

        # 2. Ranging filter
        if len(closes) >= 12:
            last_10 = closes[-12:-2]
            if len(last_10) >= 10:
                range_high = max(last_10)
                range_low = min(last_10)
                range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0
                if range_pct < self.range_threshold:
                    self.filter_stats['ranging'] += 1
                    return False, f"RANGING {range_pct:.2f}%"

        # 3. Post-close cooldown
        if asset in self.last_close_time:
            since_close = ts_s - self.last_close_time[asset]
            if since_close < self.post_close_cooldown:
                self.filter_stats['post_close_cooldown'] += 1
                return False, f"POST_CLOSE_COOLDOWN {int(self.post_close_cooldown - since_close)}s"

        # 4. Trade cooldown
        if asset in self.last_trade_time:
            since_trade = ts_s - self.last_trade_time[asset]
            if since_trade < self.trade_cooldown:
                self.filter_stats['trade_cooldown'] += 1
                return False, f"TRADE_COOLDOWN {int(self.trade_cooldown - since_trade)}s"

        # 5. Loss streak cooldown
        cooldown_until = self.cooldown_until.get(asset, 0)
        if ts_s < cooldown_until:
            self.filter_stats['loss_streak'] += 1
            remaining = int(cooldown_until - ts_s)
            return False, f"LOSS_STREAK_COOLDOWN {remaining}s"

        return True, "PASS"

    def record_trade_open(self, asset: str, bar_timestamp_ms: int):
        """Record that a trade was opened."""
        self.last_trade_time[asset] = bar_timestamp_ms / 1000.0

    def record_trade_close(self, asset: str, bar_timestamp_ms: int, is_loss: bool):
        """Record trade close and update streaks."""
        ts_s = bar_timestamp_ms / 1000.0
        self.last_close_time[asset] = ts_s
        self.last_trade_time[asset] = ts_s

        if is_loss:
            streak = self.loss_streak.get(asset, 0) + 1
            self.loss_streak[asset] = streak
            if streak >= 3:
                cooldown_min = min(15, 5 * (streak - 2))
                self.cooldown_until[asset] = ts_s + cooldown_min * 60
        else:
            self.loss_streak[asset] = 0

    def get_stats(self) -> dict:
        return dict(self.filter_stats)
