"""
Trade Protection Mechanisms
============================
Inspired by freqtrade's protection system. Implements stoploss guards,
drawdown limits, pair locks, ROI tables, pairlist filters, entry confirmation,
trade tagging, position adjustment (DCA/partial exits), and order management.

Standard library only. Thread-safe.
"""

import time
import threading
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. StoplossGuard
# ---------------------------------------------------------------------------
class StoplossGuard:
    """Locks all trading when too many stoploss exits happen in a window."""

    def __init__(self, trade_limit=3, lookback_minutes=60, cooldown_minutes=30):
        self.trade_limit = trade_limit
        self.lookback_minutes = lookback_minutes
        self.cooldown_minutes = cooldown_minutes
        self.sl_exits = []  # list of (timestamp, asset, pnl)
        self.locked_until = 0
        self._lock = threading.Lock()

    def record_sl_exit(self, asset, pnl, timestamp=None):
        """Record a stoploss exit."""
        ts = timestamp or time.time()
        with self._lock:
            self.sl_exits.append((ts, asset, pnl))
            # Prune old entries beyond 2x lookback to save memory
            cutoff = ts - self.lookback_minutes * 60 * 2
            self.sl_exits = [(t, a, p) for t, a, p in self.sl_exits if t >= cutoff]

    def is_locked(self) -> dict:
        """Return {locked: bool, reason: str, remaining_min: float}"""
        try:
            now = time.time()
            with self._lock:
                # Check existing lock
                if now < self.locked_until:
                    remaining = (self.locked_until - now) / 60.0
                    return {
                        "locked": True,
                        "reason": f"StoplossGuard: cooling down ({remaining:.1f}min left)",
                        "remaining_min": round(remaining, 1),
                    }

                # Count SL exits in lookback window
                cutoff = now - self.lookback_minutes * 60
                recent = [e for e in self.sl_exits if e[0] >= cutoff]

                if len(recent) >= self.trade_limit:
                    self.locked_until = now + self.cooldown_minutes * 60
                    return {
                        "locked": True,
                        "reason": f"StoplossGuard: {len(recent)} SL exits in {self.lookback_minutes}min (limit {self.trade_limit}), locked for {self.cooldown_minutes}min",
                        "remaining_min": float(self.cooldown_minutes),
                    }

            return {"locked": False, "reason": "", "remaining_min": 0.0}
        except Exception as e:
            print(f"[PROTECT:SLGuard] Error: {e}")
            return {"locked": False, "reason": "", "remaining_min": 0.0}


# ---------------------------------------------------------------------------
# 2. MaxDrawdownProtection
# ---------------------------------------------------------------------------
class MaxDrawdownProtection:
    """Halts trading when rolling drawdown exceeds threshold."""

    def __init__(self, max_drawdown_pct=8.0, lookback_minutes=120, cooldown_minutes=60):
        self.max_drawdown_pct = max_drawdown_pct
        self.lookback_minutes = lookback_minutes
        self.cooldown_minutes = cooldown_minutes
        self.equity_snapshots = []  # list of (timestamp, equity)
        self.locked_until = 0
        self._lock = threading.Lock()

    def record_equity(self, equity, timestamp=None):
        """Record equity snapshot."""
        ts = timestamp or time.time()
        with self._lock:
            self.equity_snapshots.append((ts, equity))
            # Prune old snapshots
            cutoff = ts - self.lookback_minutes * 60 * 2
            self.equity_snapshots = [(t, e) for t, e in self.equity_snapshots if t >= cutoff]

    def is_locked(self) -> dict:
        """Check if drawdown exceeded threshold."""
        try:
            now = time.time()
            with self._lock:
                if now < self.locked_until:
                    remaining = (self.locked_until - now) / 60.0
                    return {
                        "locked": True,
                        "reason": f"MaxDrawdown: cooling down ({remaining:.1f}min left)",
                        "remaining_min": round(remaining, 1),
                    }

                cutoff = now - self.lookback_minutes * 60
                recent = [(t, e) for t, e in self.equity_snapshots if t >= cutoff]

                if len(recent) < 2:
                    return {"locked": False, "reason": "", "remaining_min": 0.0}

                peak = max(e for _, e in recent)
                current = recent[-1][1]

                if peak <= 0:
                    return {"locked": False, "reason": "", "remaining_min": 0.0}

                drawdown_pct = (peak - current) / peak * 100.0

                if drawdown_pct > self.max_drawdown_pct:
                    self.locked_until = now + self.cooldown_minutes * 60
                    return {
                        "locked": True,
                        "reason": f"MaxDrawdown: {drawdown_pct:.1f}% drawdown exceeds {self.max_drawdown_pct}% limit, locked for {self.cooldown_minutes}min",
                        "remaining_min": float(self.cooldown_minutes),
                    }

            return {"locked": False, "reason": "", "remaining_min": 0.0}
        except Exception as e:
            print(f"[PROTECT:MaxDD] Error: {e}")
            return {"locked": False, "reason": "", "remaining_min": 0.0}


# ---------------------------------------------------------------------------
# 3. LowProfitPairLock
# ---------------------------------------------------------------------------
class LowProfitPairLock:
    """Locks specific pairs that have been consistently losing."""

    def __init__(self, min_profit_pct=-2.0, lookback_trades=10, lock_hours=4):
        self.min_profit_pct = min_profit_pct
        self.lookback_trades = lookback_trades
        self.lock_hours = lock_hours
        self.pair_trades = {}   # asset -> list of (timestamp, pnl_pct)
        self.pair_locks = {}    # asset -> locked_until timestamp
        self._lock = threading.Lock()

    def record_trade(self, asset, pnl_pct, timestamp=None):
        """Record trade result for pair."""
        ts = timestamp or time.time()
        with self._lock:
            if asset not in self.pair_trades:
                self.pair_trades[asset] = []
            self.pair_trades[asset].append((ts, pnl_pct))
            # Keep only last 2x lookback_trades
            max_keep = self.lookback_trades * 2
            if len(self.pair_trades[asset]) > max_keep:
                self.pair_trades[asset] = self.pair_trades[asset][-max_keep:]

    def is_pair_locked(self, asset) -> dict:
        """Check if pair is locked due to low profit."""
        try:
            now = time.time()
            with self._lock:
                # Check existing lock
                if asset in self.pair_locks and now < self.pair_locks[asset]:
                    remaining = (self.pair_locks[asset] - now) / 3600.0
                    return {
                        "locked": True,
                        "reason": f"PairLock: {asset} locked ({remaining:.1f}h left)",
                        "remaining_hours": round(remaining, 1),
                    }

                # Clear expired lock
                if asset in self.pair_locks and now >= self.pair_locks[asset]:
                    del self.pair_locks[asset]

                trades = self.pair_trades.get(asset, [])
                if len(trades) < self.lookback_trades:
                    return {"locked": False, "reason": "", "remaining_hours": 0.0}

                recent = trades[-self.lookback_trades:]
                cumulative_pnl = sum(p for _, p in recent)

                if cumulative_pnl < self.min_profit_pct:
                    self.pair_locks[asset] = now + self.lock_hours * 3600
                    print(f"[PROTECT:{asset}] Pair locked: cumulative PnL {cumulative_pnl:.2f}% over last {self.lookback_trades} trades")
                    return {
                        "locked": True,
                        "reason": f"PairLock: {asset} cumulative PnL {cumulative_pnl:.2f}% < {self.min_profit_pct}%, locked for {self.lock_hours}h",
                        "remaining_hours": float(self.lock_hours),
                    }

            return {"locked": False, "reason": "", "remaining_hours": 0.0}
        except Exception as e:
            print(f"[PROTECT:{asset}] PairLock error: {e}")
            return {"locked": False, "reason": "", "remaining_hours": 0.0}


# ---------------------------------------------------------------------------
# 4. ROITable
# ---------------------------------------------------------------------------
class ROITable:
    """Time-decaying profit targets — forces exits when trades stall."""

    def __init__(self, roi_table=None):
        self.roi_table = roi_table or {
            0: 0.08,
            15: 0.04,
            30: 0.02,
            60: 0.005,
            120: 0.0,
        }
        # Pre-sort: highest duration first for reverse walk
        self._sorted_entries = sorted(self.roi_table.items(), key=lambda x: x[0], reverse=True)

    def should_exit(self, trade_duration_min, current_profit_pct) -> dict:
        """Check if trade should exit based on ROI table.
        Returns {exit: bool, reason: str, min_roi: float}"""
        try:
            for duration_threshold, min_roi in self._sorted_entries:
                if trade_duration_min >= duration_threshold and current_profit_pct >= min_roi * 100:
                    return {
                        "exit": True,
                        "reason": f"ROI: profit {current_profit_pct:.2f}% >= {min_roi*100:.1f}% target at {duration_threshold}min",
                        "min_roi": min_roi,
                    }
            return {"exit": False, "reason": "", "min_roi": 0.0}
        except Exception as e:
            print(f"[PROTECT:ROI] Error: {e}")
            return {"exit": False, "reason": "", "min_roi": 0.0}


# ---------------------------------------------------------------------------
# 5. DynamicPairlistFilter
# ---------------------------------------------------------------------------
class DynamicPairlistFilter:
    """Filters assets before trading based on volume, volatility, spread."""

    def __init__(self, min_volume_24h=0, max_spread_pct=0.5,
                 min_volatility_pct=0.5, max_volatility_pct=20.0):
        self.min_volume_24h = min_volume_24h
        self.max_spread_pct = max_spread_pct
        self.min_volatility_pct = min_volatility_pct
        self.max_volatility_pct = max_volatility_pct

    def is_tradeable(self, asset, ticker_data, ohlcv_data) -> dict:
        """Check if asset passes all filters.
        ticker_data: {bid, ask, volume_24h, last}
        ohlcv_data: {closes: list, highs: list, lows: list}
        Returns {allowed: bool, reasons: list}"""
        reasons = []
        try:
            ticker_data = ticker_data or {}
            ohlcv_data = ohlcv_data or {}

            bid = ticker_data.get("bid", 0)
            ask = ticker_data.get("ask", 0)
            volume = ticker_data.get("volume_24h", 0)

            # Spread check
            if ask and ask > 0:
                spread_pct = (ask - bid) / ask * 100.0
                if spread_pct > self.max_spread_pct:
                    reasons.append(f"Spread {spread_pct:.2f}% > {self.max_spread_pct}%")

            # Volume check
            if volume < self.min_volume_24h:
                reasons.append(f"Volume {volume:.0f} < {self.min_volume_24h}")

            # Volatility check (range of last 20 candles relative to price)
            highs = ohlcv_data.get("highs", [])
            lows = ohlcv_data.get("lows", [])
            closes = ohlcv_data.get("closes", [])

            if highs and lows and closes and len(highs) >= 2:
                window = min(20, len(highs))
                recent_highs = highs[-window:]
                recent_lows = lows[-window:]
                high_val = max(recent_highs)
                low_val = min(recent_lows)
                mid_price = closes[-1] if closes[-1] > 0 else (high_val + low_val) / 2

                if mid_price > 0:
                    volatility_pct = (high_val - low_val) / mid_price * 100.0

                    if volatility_pct < self.min_volatility_pct:
                        reasons.append(f"Volatility {volatility_pct:.2f}% < {self.min_volatility_pct}%")
                    if volatility_pct > self.max_volatility_pct:
                        reasons.append(f"Volatility {volatility_pct:.2f}% > {self.max_volatility_pct}%")

            allowed = len(reasons) == 0
            if not allowed:
                print(f"[PROTECT:{asset}] Pairlist filtered: {', '.join(reasons)}")
            return {"allowed": allowed, "reasons": reasons}

        except Exception as e:
            print(f"[PROTECT:{asset}] Pairlist filter error: {e}")
            return {"allowed": True, "reasons": []}


# ---------------------------------------------------------------------------
# 6. ConfirmTradeEntry
# ---------------------------------------------------------------------------
class ConfirmTradeEntry:
    """Last-second veto before placing an order."""

    def __init__(self, max_spread_pct=0.3, max_price_drift_pct=0.3,
                 max_concurrent_trades=4):
        self.max_spread_pct = max_spread_pct
        self.max_price_drift_pct = max_price_drift_pct
        self.max_concurrent_trades = max_concurrent_trades

    def confirm(self, asset, signal_price, current_price,
                bid, ask, open_trade_count) -> dict:
        """Last-second check before placing order.
        Returns {confirmed: bool, reason: str}"""
        try:
            # Concurrent trades check
            if open_trade_count >= self.max_concurrent_trades:
                reason = f"Max concurrent trades reached ({open_trade_count}/{self.max_concurrent_trades})"
                print(f"[PROTECT:{asset}] Entry vetoed: {reason}")
                return {"confirmed": False, "reason": reason}

            # Spread check
            if ask and ask > 0 and bid and bid > 0:
                spread_pct = (ask - bid) / ask * 100.0
                if spread_pct > self.max_spread_pct:
                    reason = f"Spread too wide: {spread_pct:.2f}% > {self.max_spread_pct}%"
                    print(f"[PROTECT:{asset}] Entry vetoed: {reason}")
                    return {"confirmed": False, "reason": reason}

            # Price drift check
            if signal_price and signal_price > 0 and current_price and current_price > 0:
                drift_pct = abs(current_price - signal_price) / signal_price * 100.0
                if drift_pct > self.max_price_drift_pct:
                    reason = f"Price drifted {drift_pct:.2f}% from signal ({signal_price} -> {current_price})"
                    print(f"[PROTECT:{asset}] Entry vetoed: {reason}")
                    return {"confirmed": False, "reason": reason}

            return {"confirmed": True, "reason": ""}

        except Exception as e:
            print(f"[PROTECT:{asset}] ConfirmEntry error: {e}")
            return {"confirmed": True, "reason": ""}


# ---------------------------------------------------------------------------
# 7. EntryExitTagger
# ---------------------------------------------------------------------------
class EntryExitTagger:
    """Tags every trade with WHY it entered and exited for analysis."""

    def tag_entry(self, signal, entry_score, regime, htf_alignment,
                  is_reversal, ema_slope, consensus) -> str:
        """Generate entry tag like 'trend_strong_aligned_8' or 'reversal_ranging_5'."""
        try:
            parts = []

            # Trade type
            if is_reversal:
                parts.append("reversal")
            else:
                parts.append("trend")

            # Strength
            if entry_score is not None:
                if entry_score >= 7:
                    parts.append("strong")
                elif entry_score >= 5:
                    parts.append("med")
                else:
                    parts.append("weak")

            # Regime
            regime_str = str(regime).lower().replace(" ", "_") if regime else "unknown"
            parts.append(regime_str)

            # HTF alignment
            if htf_alignment:
                parts.append("aligned")
            else:
                parts.append("misaligned")

            # Score
            score_val = int(entry_score) if entry_score is not None else 0
            parts.append(str(score_val))

            return "_".join(parts)

        except Exception as e:
            print(f"[PROTECT:Tagger] Entry tag error: {e}")
            return "unknown_entry"

    def tag_exit(self, exit_reason, sl_level, pnl_pct, duration_min,
                 roi_exit=False) -> str:
        """Generate exit tag like 'sl_L3_+2.1%_45min' or 'roi_+4.0%_30min'."""
        try:
            pnl_str = f"{pnl_pct:+.1f}%"
            dur_str = f"{int(duration_min)}min"

            if roi_exit:
                return f"roi_{pnl_str}_{dur_str}"

            reason_str = str(exit_reason).lower().replace(" ", "_") if exit_reason else "manual"
            sl_str = f"L{sl_level}" if sl_level is not None else ""

            parts = [reason_str]
            if sl_str:
                parts.append(sl_str)
            parts.append(pnl_str)
            parts.append(dur_str)

            return "_".join(parts)

        except Exception as e:
            print(f"[PROTECT:Tagger] Exit tag error: {e}")
            return "unknown_exit"

    def analyze_tags(self, trades_with_tags) -> dict:
        """Analyze which entry/exit tag combinations are profitable.
        trades_with_tags: list of {entry_tag, exit_tag, pnl_pct}
        Returns dict sorted by profitability."""
        try:
            if not trades_with_tags:
                return {}

            by_entry = defaultdict(list)
            for t in trades_with_tags:
                entry_tag = t.get("entry_tag", "unknown")
                pnl = t.get("pnl_pct", 0.0)
                by_entry[entry_tag].append(pnl)

            results = {}
            for tag, pnls in by_entry.items():
                wins = [p for p in pnls if p > 0]
                results[tag] = {
                    "count": len(pnls),
                    "win_rate": len(wins) / len(pnls) * 100.0 if pnls else 0.0,
                    "avg_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
                    "total_pnl": sum(pnls),
                    "best": max(pnls) if pnls else 0.0,
                    "worst": min(pnls) if pnls else 0.0,
                }

            # Sort by total_pnl descending
            sorted_results = dict(
                sorted(results.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
            )
            return sorted_results

        except Exception as e:
            print(f"[PROTECT:Tagger] Analyze error: {e}")
            return {}


# ---------------------------------------------------------------------------
# 8. PositionAdjuster (DCA + Partial Exits)
# ---------------------------------------------------------------------------
class PositionAdjuster:
    """Manages DCA (dollar cost averaging) entries and partial profit exits."""

    def __init__(self, max_dca_entries=2, dca_threshold_pct=-2.0,
                 dca_multiplier=0.5, partial_exit_levels=None):
        self.max_dca_entries = max_dca_entries
        self.dca_threshold_pct = dca_threshold_pct
        self.dca_multiplier = dca_multiplier
        self.partial_exit_levels = partial_exit_levels or [
            (3.0, 0.25),   # At +3%, exit 25%
            (5.0, 0.25),   # At +5%, exit another 25%
            (8.0, 0.25),   # At +8%, exit another 25%
            # Last 25% rides with trailing SL
        ]
        self.dca_history = {}    # asset -> list of {ts, amount, price}
        self.partial_exits = {}  # asset -> set of levels already taken
        self._lock = threading.Lock()

    def should_dca(self, asset, current_pnl_pct, entry_count, regime) -> dict:
        """Check if we should add to position (DCA).
        Returns {adjust: bool, amount_multiplier: float, reason: str}"""
        try:
            regime_str = str(regime).lower() if regime else ""

            # Don't DCA in volatile or choppy regimes
            blocked_regimes = ("volatile", "choppy", "chaotic")
            for br in blocked_regimes:
                if br in regime_str:
                    return {
                        "adjust": False,
                        "amount_multiplier": 0.0,
                        "reason": f"DCA blocked in {regime_str} regime",
                    }

            # Check entry count
            if entry_count >= self.max_dca_entries:
                return {
                    "adjust": False,
                    "amount_multiplier": 0.0,
                    "reason": f"Max DCA entries reached ({entry_count}/{self.max_dca_entries})",
                }

            # Check PnL threshold
            if current_pnl_pct >= self.dca_threshold_pct:
                return {
                    "adjust": False,
                    "amount_multiplier": 0.0,
                    "reason": f"PnL {current_pnl_pct:.2f}% not below DCA threshold {self.dca_threshold_pct}%",
                }

            print(f"[PROTECT:{asset}] DCA signal: PnL {current_pnl_pct:.2f}% < {self.dca_threshold_pct}%, entry #{entry_count+1}")
            return {
                "adjust": True,
                "amount_multiplier": self.dca_multiplier,
                "reason": f"DCA: PnL {current_pnl_pct:.2f}% below {self.dca_threshold_pct}%, adding {self.dca_multiplier*100:.0f}%",
            }

        except Exception as e:
            print(f"[PROTECT:{asset}] DCA error: {e}")
            return {"adjust": False, "amount_multiplier": 0.0, "reason": f"Error: {e}"}

    def should_partial_exit(self, asset, current_pnl_pct) -> dict:
        """Check if we should take partial profits.
        Returns {exit: bool, exit_fraction: float, reason: str}"""
        try:
            with self._lock:
                taken = self.partial_exits.get(asset, set())

                for level_pct, fraction in self.partial_exit_levels:
                    if current_pnl_pct >= level_pct and level_pct not in taken:
                        print(f"[PROTECT:{asset}] Partial exit: PnL {current_pnl_pct:.2f}% >= {level_pct}%, exiting {fraction*100:.0f}%")
                        return {
                            "exit": True,
                            "exit_fraction": fraction,
                            "level": level_pct,
                            "reason": f"Partial exit: {fraction*100:.0f}% at +{level_pct}% profit",
                        }

            return {"exit": False, "exit_fraction": 0.0, "reason": ""}

        except Exception as e:
            print(f"[PROTECT:{asset}] Partial exit error: {e}")
            return {"exit": False, "exit_fraction": 0.0, "reason": f"Error: {e}"}

    def record_dca(self, asset, amount, price):
        """Record a DCA entry."""
        try:
            with self._lock:
                if asset not in self.dca_history:
                    self.dca_history[asset] = []
                self.dca_history[asset].append({
                    "ts": time.time(),
                    "amount": amount,
                    "price": price,
                })
        except Exception as e:
            print(f"[PROTECT:{asset}] Record DCA error: {e}")

    def record_partial_exit(self, asset, level):
        """Record a partial exit level was taken."""
        try:
            with self._lock:
                if asset not in self.partial_exits:
                    self.partial_exits[asset] = set()
                self.partial_exits[asset].add(level)
        except Exception as e:
            print(f"[PROTECT:{asset}] Record partial exit error: {e}")

    def reset(self, asset):
        """Reset DCA/partial state when position fully closed."""
        try:
            with self._lock:
                self.dca_history.pop(asset, None)
                self.partial_exits.pop(asset, None)
        except Exception as e:
            print(f"[PROTECT:{asset}] Reset error: {e}")


# ---------------------------------------------------------------------------
# 9. OrderPriceAdjuster
# ---------------------------------------------------------------------------
class OrderPriceAdjuster:
    """Manages unfilled orders — chases price or cancels."""

    def __init__(self, max_age_seconds=120, chase_threshold_pct=0.3,
                 max_chase_attempts=2):
        self.max_age_seconds = max_age_seconds
        self.chase_threshold_pct = chase_threshold_pct
        self.max_chase_attempts = max_chase_attempts
        self.chase_count = {}  # order_id -> count
        self._lock = threading.Lock()

    def should_adjust(self, order_id, order_price, current_price,
                      order_age_seconds, side) -> dict:
        """Check if unfilled order should be adjusted.
        Returns {action: 'keep'|'adjust'|'cancel', new_price: float, reason: str}"""
        try:
            with self._lock:
                chases = self.chase_count.get(order_id, 0)

            # Too old -> cancel
            if order_age_seconds > self.max_age_seconds:
                self._clear_order(order_id)
                return {
                    "action": "cancel",
                    "new_price": 0.0,
                    "reason": f"Order aged out ({order_age_seconds:.0f}s > {self.max_age_seconds}s)",
                }

            # Already chased max times -> cancel
            if chases >= self.max_chase_attempts:
                self._clear_order(order_id)
                return {
                    "action": "cancel",
                    "new_price": 0.0,
                    "reason": f"Max chase attempts reached ({chases}/{self.max_chase_attempts})",
                }

            # Price drift check
            if order_price and order_price > 0 and current_price and current_price > 0:
                drift_pct = abs(current_price - order_price) / order_price * 100.0

                if drift_pct > self.chase_threshold_pct:
                    # Chase: adjust price toward current market
                    # For buy orders, move price up; for sell orders, move price down
                    side_lower = str(side).lower()
                    if side_lower in ("buy", "long"):
                        new_price = current_price  # Match current ask
                    else:
                        new_price = current_price  # Match current bid

                    with self._lock:
                        self.chase_count[order_id] = chases + 1

                    return {
                        "action": "adjust",
                        "new_price": new_price,
                        "reason": f"Price drifted {drift_pct:.2f}% (chase #{chases+1})",
                    }

            return {"action": "keep", "new_price": 0.0, "reason": "Order still valid"}

        except Exception as e:
            print(f"[PROTECT:OrderAdj] Error for {order_id}: {e}")
            return {"action": "keep", "new_price": 0.0, "reason": f"Error: {e}"}

    def _clear_order(self, order_id):
        """Remove tracking for completed/cancelled order."""
        with self._lock:
            self.chase_count.pop(order_id, None)

    def reset_order(self, order_id):
        """Public method to clear order tracking."""
        self._clear_order(order_id)


# ---------------------------------------------------------------------------
# Main: TradeProtections
# ---------------------------------------------------------------------------
class TradeProtections:
    """Combines all protection modules into a single interface."""

    def __init__(self, config=None):
        config = config or {}
        self.sl_guard = StoplossGuard(**config.get("sl_guard", {}))
        self.drawdown = MaxDrawdownProtection(**config.get("max_drawdown", {}))
        self.pair_lock = LowProfitPairLock(**config.get("pair_lock", {}))
        self.roi = ROITable(config.get("roi_table"))
        self.pairlist = DynamicPairlistFilter(**config.get("pairlist", {}))
        self.confirm = ConfirmTradeEntry(**config.get("confirm", {}))
        self.tagger = EntryExitTagger()
        self.adjuster = PositionAdjuster(**config.get("position_adjust", {}))
        self.order_adjust = OrderPriceAdjuster(**config.get("order_adjust", {}))

    def pre_entry_check(self, asset, signal_price, current_price,
                        bid, ask, open_trade_count, equity,
                        ticker_data=None, ohlcv_data=None) -> dict:
        """Run ALL pre-entry checks. Returns {allowed: bool, reasons: list}"""
        reasons = []
        try:
            # 1. StoplossGuard
            sl_check = self.sl_guard.is_locked()
            if sl_check["locked"]:
                reasons.append(sl_check["reason"])

            # 2. MaxDrawdown
            dd_check = self.drawdown.is_locked()
            if dd_check["locked"]:
                reasons.append(dd_check["reason"])

            # 3. Pair lock
            pair_check = self.pair_lock.is_pair_locked(asset)
            if pair_check["locked"]:
                reasons.append(pair_check["reason"])

            # 4. Pairlist filter
            if ticker_data or ohlcv_data:
                pl_check = self.pairlist.is_tradeable(asset, ticker_data, ohlcv_data)
                if not pl_check["allowed"]:
                    reasons.extend(pl_check["reasons"])

            # 5. ConfirmTradeEntry
            ce_check = self.confirm.confirm(
                asset, signal_price, current_price, bid, ask, open_trade_count
            )
            if not ce_check["confirmed"]:
                reasons.append(ce_check["reason"])

            # Record equity snapshot if provided
            if equity is not None and equity > 0:
                self.drawdown.record_equity(equity)

            allowed = len(reasons) == 0
            if not allowed:
                print(f"[PROTECT:{asset}] Entry BLOCKED: {'; '.join(reasons)}")

            return {"allowed": allowed, "reasons": reasons}

        except Exception as e:
            print(f"[PROTECT:{asset}] pre_entry_check error: {e}")
            return {"allowed": True, "reasons": []}

    def check_exit(self, asset, current_pnl_pct, trade_duration_min,
                   sl_level) -> dict:
        """Check ROI table + partial exits.
        Returns {exit: bool, partial: bool, fraction: float, reason: str}"""
        try:
            # Check ROI table first
            roi_check = self.roi.should_exit(trade_duration_min, current_pnl_pct)
            if roi_check["exit"]:
                return {
                    "exit": True,
                    "partial": False,
                    "fraction": 1.0,
                    "reason": roi_check["reason"],
                }

            # Check partial exits
            partial_check = self.adjuster.should_partial_exit(asset, current_pnl_pct)
            if partial_check["exit"]:
                return {
                    "exit": True,
                    "partial": True,
                    "fraction": partial_check["exit_fraction"],
                    "reason": partial_check["reason"],
                }

            return {"exit": False, "partial": False, "fraction": 0.0, "reason": ""}

        except Exception as e:
            print(f"[PROTECT:{asset}] check_exit error: {e}")
            return {"exit": False, "partial": False, "fraction": 0.0, "reason": ""}

    def record_close(self, asset, pnl_pct, pnl_usd, exit_reason,
                     sl_hit=False, equity=None):
        """Record trade close across all protections."""
        try:
            # Record stoploss if applicable
            if sl_hit:
                self.sl_guard.record_sl_exit(asset, pnl_usd)

            # Record pair trade
            self.pair_lock.record_trade(asset, pnl_pct)

            # Record equity
            if equity is not None and equity > 0:
                self.drawdown.record_equity(equity)

            # Reset position adjuster for this asset
            self.adjuster.reset(asset)

            print(f"[PROTECT:{asset}] Trade closed: {pnl_pct:+.2f}% (${pnl_usd:+.2f}) reason={exit_reason} sl_hit={sl_hit}")

        except Exception as e:
            print(f"[PROTECT:{asset}] record_close error: {e}")
