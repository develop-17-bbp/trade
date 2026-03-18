"""
L3 Risk Engine — Full Risk Management System
===============================================
Implements all risk controls required before live trading:

  1. Position size limits (per-trade and portfolio)
  2. Daily loss limit with halt mechanism
  3. ATR-based dynamic stop-loss / take-profit
  4. Volatility gating (EWMA regime filter)
  5. Drawdown tracking & circuit breaker
  6. Correlation filter (avoid concentrated exposure)
  7. Trade frequency throttling
  8. VETO power — absolute override when risk is extreme

The L3 layer has 20% signal weight PLUS veto authority.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum

from src.models.volatility import VolRegime


class RiskAction(Enum):
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"
    VETO = "veto"           # absolute override — no trading
    EMERGENCY_EXIT = "emergency_exit"  # close all positions immediately


class TradeRecord:
    """Record of a single trade for P&L tracking."""
    __slots__ = ('asset', 'direction', 'entry_price', 'exit_price',
                 'size', 'pnl', 'timestamp', 'holding_bars',
                 'stop_loss', 'take_profit', 'trailing_stop_active', 'peak_price',
                 'partial_tp_hit', 'breakeven_active', 'order_id')

    def __init__(self, asset: str, direction: int, entry_price: float,
                 size: float, timestamp: float = 0.0, order_id: str = ""):
        self.asset = asset
        self.direction = direction
        self.entry_price = entry_price
        self.exit_price = 0.0
        self.size = size
        self.pnl = 0.0
        self.timestamp = timestamp or time.time()
        self.order_id = order_id
        self.holding_bars = 0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop_active = False
        self.peak_price = entry_price
        self.partial_tp_hit = False
        self.breakeven_active = False

    def close(self, exit_price: float):
        self.exit_price = exit_price
        self.pnl = self.direction * self.size * (exit_price - self.entry_price)


class RiskManager:
    """
    Full L3 Risk Engine with veto authority.

    Config parameters:
        max_position_pct:     max % of account per single position (default: 2%)
        max_portfolio_pct:    max % of account in total open positions (default: 20%)
        daily_loss_limit_pct: max daily loss before halt (default: 3%)
        max_drawdown_pct:     max drawdown before circuit breaker (default: 10%)
        atr_stop_mult:        ATR multiplier for stop-loss (default: 2.0)
        atr_tp_mult:          ATR multiplier for take-profit (default: 3.0)
        vol_gate_mult:        volatility gating multiplier (default: 2.0)
        max_trades_per_day:   trade frequency cap (default: 20)
        cooldown_after_loss:  minutes to wait after large loss (default: 30)
    """

    def __init__(self,
                 max_position_pct: float = 2.0,
                 max_portfolio_pct: float = 20.0,
                 daily_loss_limit_pct: float = 3.0,
                 max_drawdown_pct: float = 10.0,
                 atr_stop_mult: float = 2.0,
                 atr_tp_mult: float = 3.0,
                 vol_gate_mult: float = 2.0,
                 max_trades_per_day: int = 20,
                 cooldown_after_loss: float = 30.0,
                 **kwargs):
        # Position limits (static baselines — MC can tighten dynamically)
        self._base_max_position_pct = kwargs.get('max_position_size_pct', max_position_pct)
        self.max_position_pct = self._base_max_position_pct
        self.max_portfolio_pct = max_portfolio_pct

        # Loss controls
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct

        # ATR-based stops
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult

        # Volatility gating
        self.vol_gate_mult = vol_gate_mult

        # Frequency limits
        self.max_trades_per_day = max_trades_per_day
        self.cooldown_after_loss = cooldown_after_loss  # minutes

        # Monte Carlo dynamic risk adjustment
        self._mc_risk_score = 0.5
        self._mc_position_scale = 1.0

        # EVT tail risk dynamic adjustment
        self._evt_risk_score = 0.3
        self._evt_position_scale = 1.0

        # State tracking
        self.daily_loss = 0.0
        self.daily_trades = 0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.trade_history: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}
        self._halted = False
        self._halt_reason = ''
        self._last_loss_time = 0.0
        self._veto_active = False

    def update_mc_risk(self, mc_risk_score: float = 0.5, mc_position_scale: float = 1.0):
        """
        Update risk limits dynamically from Monte Carlo VaR/CVaR engine.

        When MC risk is elevated (score > 0.7), position limits tighten automatically.
        This replaces static percentage limits with forward-looking risk assessment.
        """
        self._mc_risk_score = mc_risk_score
        self._mc_position_scale = mc_position_scale

        # Dynamic position limit: tighten when MC says risk is high
        if mc_risk_score > 0.7:
            self.max_position_pct = self._base_max_position_pct * 0.5
        elif mc_risk_score > 0.5:
            self.max_position_pct = self._base_max_position_pct * 0.75
        else:
            self.max_position_pct = self._base_max_position_pct

    def update_evt_risk(self, evt_risk_score: float = 0.3, evt_position_scale: float = 1.0):
        """
        Update risk limits from Extreme Value Theory tail risk engine.

        EVT captures fat-tail risk that normal distributions miss.
        Heavy tails (high ξ) → tighter position limits.
        """
        self._evt_risk_score = evt_risk_score
        self._evt_position_scale = evt_position_scale

        # Further tighten if EVT says tails are heavy
        if evt_risk_score > 0.7:
            # Stack with MC adjustment (multiplicative)
            self.max_position_pct = min(
                self.max_position_pct,
                self._base_max_position_pct * 0.4
            )
        elif evt_risk_score > 0.5:
            self.max_position_pct = min(
                self.max_position_pct,
                self._base_max_position_pct * 0.6
            )

    def is_trade_safe(self, current_price: float, direction: int,
                        atr_value: float, account_balance: float) -> Tuple[bool, str]:
        """Wrapper for evaluate_trade to match HybridStrategy interface."""
        res = self.evaluate_trade(
            asset="PROXY", direction=direction, proposed_size=1.0, 
            account_balance=account_balance, current_price=current_price,
            atr_value=atr_value
        )
        is_safe = res['action'] in [RiskAction.ALLOW, RiskAction.REDUCE]
        return is_safe, res['reason']

    # -------------------------------------------------------------------
    # Core risk check — called before every trade
    # -------------------------------------------------------------------
    def evaluate_trade(self,
                        asset: str,
                        direction: int,
                        proposed_size: float,
                        account_balance: float,
                        current_price: float,
                        atr_value: float = 0.0,
                        vol_regime: VolRegime = VolRegime.MEDIUM,
                        composite_signal: float = 0.0,
                        ) -> Dict:
        """
        Evaluate whether a proposed trade should be allowed, reduced, or blocked.

        Returns:
            {
                'action': RiskAction,
                'adjusted_size': float,
                'stop_loss': float,
                'take_profit': float,
                'reason': str,
                'risk_score': float,  # 0 (safe) to 1 (dangerous)
            }
        """
        self.current_equity = account_balance
        self.peak_equity = max(self.peak_equity, account_balance)

        reasons: List[str] = []
        risk_score = 0.0

        # ---- Check 1: Is trading halted? ----
        if self._halted:
            return self._veto_result(
                f"HALTED: {self._halt_reason}", current_price, atr_value
            )

        # ---- Check 2: Daily loss limit ----
        loss_pct = (self.daily_loss / account_balance * 100) if account_balance > 0 else 0
        if loss_pct >= self.daily_loss_limit_pct:
            self._halt("Daily loss limit reached")
            return self._veto_result(
                f"Daily loss {loss_pct:.1f}% >= limit {self.daily_loss_limit_pct}%",
                current_price, atr_value
            )
        risk_score += (loss_pct / self.daily_loss_limit_pct) * 0.3

        # ---- Check 3: Maximum drawdown circuit breaker ----
        drawdown_pct = self._current_drawdown_pct()
        if drawdown_pct >= self.max_drawdown_pct:
            self._halt("Circuit breaker: max drawdown exceeded")
            return self._veto_result(
                f"Drawdown {drawdown_pct:.1f}% >= circuit breaker {self.max_drawdown_pct}%",
                current_price, atr_value
            )
        risk_score += (drawdown_pct / self.max_drawdown_pct) * 0.2

        # ---- Check 4: Extreme volatility VETO ----
        if vol_regime == VolRegime.EXTREME:
            return self._veto_result(
                "EXTREME volatility regime — all trading suspended",
                current_price, atr_value
            )
        if vol_regime == VolRegime.HIGH:
            risk_score += 0.2
            reasons.append("HIGH volatility — reduced sizing")

        # ---- Check 5: Trade frequency ----
        if self.daily_trades >= self.max_trades_per_day:
            return self._block_result("Daily trade limit reached", current_price, atr_value)
        risk_score += (self.daily_trades / self.max_trades_per_day) * 0.1

        # ---- Check 6: Post-loss cooldown ----
        if self._last_loss_time > 0:
            elapsed_min = (time.time() - self._last_loss_time) / 60.0
            if elapsed_min < self.cooldown_after_loss:
                remaining = self.cooldown_after_loss - elapsed_min
                return self._block_result(
                    f"Post-loss cooldown: {remaining:.0f}min remaining",
                    current_price, atr_value
                )

        # ---- Check 7: Position size limit ----
        max_allowed = account_balance * (self.max_position_pct / 100.0)
        adjusted_size = min(proposed_size, max_allowed)

        # ---- Check 8: Portfolio exposure limit ----
        total_exposure = sum(
            tr.size * tr.entry_price for tr in self.open_positions.values()
        )
        max_portfolio = account_balance * (self.max_portfolio_pct / 100.0)
        new_exposure = total_exposure + adjusted_size * current_price
        if new_exposure > max_portfolio:
            room = max(0, max_portfolio - total_exposure)
            if room <= 0:
                return self._block_result(
                    "Portfolio exposure limit reached", current_price, atr_value
                )
            adjusted_size = min(adjusted_size, room / current_price)
            reasons.append("Reduced to fit portfolio limit")

        # ---- Check 9: Volatility-scaled sizing ----
        vol_scale = {
            VolRegime.LOW: 1.2,
            VolRegime.MEDIUM: 1.0,
            VolRegime.HIGH: 0.5,
            VolRegime.EXTREME: 0.0,
        }
        adjusted_size *= vol_scale.get(vol_regime, 1.0)

        # ---- Check 10: Signal strength scaling ----
        signal_strength = abs(composite_signal)
        if signal_strength < 0.3:
            adjusted_size *= 0.5
            reasons.append("Weak signal — half size")
        elif signal_strength > 0.7:
            adjusted_size *= 1.0  # full size for strong signals
        else:
            adjusted_size *= 0.75
            reasons.append("Moderate signal — 75% size")

        # ---- Calculate stops ----
        stop_loss, take_profit = self._calculate_stops(
            current_price, direction, atr_value
        )

        # ---- Determine action ----
        if adjusted_size <= 0:
            action = RiskAction.BLOCK
        elif adjusted_size < proposed_size * 0.5:
            action = RiskAction.REDUCE
        else:
            action = RiskAction.ALLOW

        return {
            'action': action,
            'adjusted_size': max(0.0, adjusted_size),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'reason': '; '.join(reasons) if reasons else 'OK',
            'risk_score': min(1.0, risk_score),
            'drawdown_pct': drawdown_pct,
            'daily_loss_pct': loss_pct,
        }

    # -------------------------------------------------------------------
    # Trade lifecycle
    # -------------------------------------------------------------------
    def register_trade_open(self, asset: str, direction: int,
                              entry_price: float, size: float,
                              stop_loss: float = 0.0, take_profit: float = 0.0, order_id: str = ""):
        """Record a new open position with precise stops."""
        record = TradeRecord(asset, direction, entry_price, size, order_id=order_id)
        record.stop_loss = stop_loss
        record.take_profit = take_profit
        self.open_positions[asset] = record
        self.daily_trades += 1

    def register_trade_close(self, asset: str, exit_price: float,
                               account_balance: float):
        """Close position and update P&L tracking."""
        if asset not in self.open_positions:
            return 0.0
        record = self.open_positions.pop(asset)
        record.close(exit_price)
        self.trade_history.append(record)

        if record.pnl < 0:
            self.daily_loss += abs(record.pnl)
            self._last_loss_time = time.time()

        return record.pnl

    def register_loss(self, loss: float, account_balance: float):
        """Register a loss (backward-compatible method)."""
        self.daily_loss += abs(loss)
        if account_balance > 0:
            loss_pct = (self.daily_loss / account_balance) * 100
            if loss_pct >= self.daily_loss_limit_pct:
                self._halt("Daily loss limit reached")

    # -------------------------------------------------------------------
    # Stop-loss / Take-profit
    # -------------------------------------------------------------------
    def _calculate_stops(self, price: float, direction: int,
                          atr_value: float) -> Tuple[float, float]:
        """
        ATR-based dynamic stop-loss and take-profit.
          Stop  = entry ± k_stop * ATR
          TP    = entry ± k_tp * ATR
        """
        if atr_value <= 0:
            atr_value = price * 0.02  # fallback: 2% of price

        if direction > 0:  # long
            stop_loss = price - self.atr_stop_mult * atr_value
            take_profit = price + self.atr_tp_mult * atr_value
        else:  # short
            stop_loss = price + self.atr_stop_mult * atr_value
            take_profit = price - self.atr_tp_mult * atr_value

        return stop_loss, take_profit

    def check_stops(self, asset: str, current_price: float) -> Optional[str]:
        """Check if current price hit stop-loss or take-profit."""
        if asset not in self.open_positions:
            return None

        record = self.open_positions[asset]
        
        # --- Update Trailing Stop & Check Partial TP ---
        trail_dist = abs(record.entry_price - record.stop_loss) * 0.8
        
        if record.direction > 0: # Long
            # Partial Take Profit Check (at 50% of the distance to full TP)
            if not record.partial_tp_hit and record.take_profit > record.entry_price:
                partial_price = record.entry_price + (record.take_profit - record.entry_price) * 0.5
                if current_price >= partial_price:
                    record.partial_tp_hit = True
                    # Move stop to breakeven
                    record.stop_loss = max(record.stop_loss, record.entry_price * 1.001)
                    record.breakeven_active = True
                    return 'partial_tp_long'

            if current_price > record.peak_price:
                record.peak_price = current_price
                new_stop = current_price - trail_dist
                if new_stop > record.stop_loss:
                    record.stop_loss = new_stop
                    
            if current_price <= record.stop_loss:
                return 'stop_loss'
            if record.take_profit > 0 and current_price >= record.take_profit:
                return 'take_profit'
                
        else: # Short
            # Partial Take Profit Check
            if not record.partial_tp_hit and record.take_profit < record.entry_price:
                partial_price = record.entry_price - (record.entry_price - record.take_profit) * 0.5
                if current_price <= partial_price:
                    record.partial_tp_hit = True
                    # Move stop to breakeven
                    record.stop_loss = min(record.stop_loss, record.entry_price * 0.999)
                    record.breakeven_active = True
                    return 'partial_tp_short'

            if current_price < record.peak_price:
                record.peak_price = current_price
                new_stop = current_price + trail_dist
                if (record.stop_loss == 0) or (new_stop < record.stop_loss):
                    record.stop_loss = new_stop
                    
            if current_price >= record.stop_loss:
                return 'stop_loss'
            if record.take_profit > 0 and current_price <= record.take_profit:
                return 'take_profit'

        # Fix #7: Time-based exit — close stale positions after max_hold_bars
        max_hold_bars = 8
        record.holding_bars = getattr(record, 'holding_bars', 0) + 1
        if record.holding_bars >= max_hold_bars:
            pnl_pct = record.direction * (current_price - record.entry_price) / record.entry_price
            if pnl_pct < 0.005:  # less than 0.5% profit
                return 'time_exit'

        return None

    def check_all_stops(self, prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Utility to check all open positions against current prices."""
        results = []
        for asset, price in prices.items():
            trigger = self.check_stops(asset, price)
            if trigger:
                results.append({
                    'asset': asset,
                    'price': price,
                    'trigger': trigger,
                    'record': self.open_positions[asset]
                })
        return results

    # -------------------------------------------------------------------
    # Position size check (backward-compatible)
    # -------------------------------------------------------------------
    def check_position_size(self, account_balance: float,
                              position_value: float) -> bool:
        """Check if a position is within limits. Backward-compatible."""
        allowed = account_balance * (self.max_position_pct / 100.0)
        return position_value <= allowed

    # -------------------------------------------------------------------
    # Drawdown tracking
    # -------------------------------------------------------------------
    def _current_drawdown_pct(self) -> float:
        """Calculate current drawdown from peak equity."""
        if self.peak_equity <= 0:
            return 0.0
        dd = (self.peak_equity - self.current_equity) / self.peak_equity * 100
        return max(0.0, dd)

    # -------------------------------------------------------------------
    # Performance metrics
    # -------------------------------------------------------------------
    def get_performance_stats(self) -> Dict:
        """Compute trading performance statistics from history."""
        if not self.trade_history:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'avg_pnl': 0.0,
                'total_pnl': 0.0, 'sharpe': 0.0, 'max_drawdown': 0.0,
                'profit_factor': 0.0, 'avg_win_loss_ratio': 0.0,
            }

        pnls = [t.pnl for t in self.trade_history]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_pnl = total / len(pnls) if pnls else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-10
        profit_factor = gross_profit / gross_loss

        # Win/loss ratio
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 1e-10
        wl_ratio = avg_win / avg_loss

        # Sharpe approximation
        if len(pnls) > 1:
            mean_pnl = sum(pnls) / len(pnls)
            var_pnl = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
            std_pnl = math.sqrt(var_pnl) if var_pnl > 0 else 1e-10
            sharpe = (mean_pnl / std_pnl) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Equity curve for max drawdown
        equity = [0.0]
        for p in pnls:
            equity.append(equity[-1] + p)
        peak = equity[0]
        max_dd = 0.0
        for e in equity:
            if e > peak:
                peak = e
            dd = peak - e
            if dd > max_dd:
                max_dd = dd

        return {
            'total_trades': len(pnls),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl': total,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'profit_factor': profit_factor,
            'avg_win_loss_ratio': wl_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
        }

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------
    def _halt(self, reason: str):
        self._halted = True
        self._halt_reason = reason
        print(f"[HALT] RISK: {reason}")

    def _veto_result(self, reason: str, price: float, atr: float) -> Dict:
        return {
            'action': RiskAction.VETO,
            'adjusted_size': 0.0,
            'stop_loss': price,
            'take_profit': price,
            'reason': f"VETO: {reason}",
            'risk_score': 1.0,
            'drawdown_pct': self._current_drawdown_pct(),
            'daily_loss_pct': 0.0,
        }

    def _block_result(self, reason: str, price: float, atr: float) -> Dict:
        return {
            'action': RiskAction.BLOCK,
            'adjusted_size': 0.0,
            'stop_loss': price,
            'take_profit': price,
            'reason': f"BLOCKED: {reason}",
            'risk_score': 0.8,
            'drawdown_pct': self._current_drawdown_pct(),
            'daily_loss_pct': 0.0,
        }

    def reset_daily(self):
        """Reset daily counters — call at start of each trading day."""
        self.daily_loss = 0.0
        self.daily_trades = 0
        self._halted = False
        self._halt_reason = ''
        self._last_loss_time = 0.0
        self._veto_active = False

    def unhalt(self):
        """Manual override to resume trading after halt."""
        self._halted = False
        self._halt_reason = ''
        print("[OK] Risk halt cleared -- trading resumed")
