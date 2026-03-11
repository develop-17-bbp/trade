"""
Profit Protector & Loss Aversion Engine
=========================================
Prevents loss trades and protects profits at all costs.

Core Rules:
1. If portfolio in profit → NEVER trade unless HIGH confidence (>0.75)
2. Predict losses BEFORE entry → Block trade if P(loss) > 0.35
3. Always lock breakeven → Stop loss at entry price minimum
4. Position size ∝ confidence → High confidence = bigger position
5. Wait when uncertain → No trade = 0% loss vs risky trade = -2% loss
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

@dataclass
class TradeQualityScore:
    """Rates the quality of a potential trade"""
    confidence: float           # 0-1: Model confidence in direction
    win_probability: float      # 0-1: Likelihood this trade wins
    profit_expectancy: float    # Expected profit in $ if trade executes
    risk_reward_ratio: float    # Reward / Risk (want > 2.0)
    is_profitable: bool         # True if expected profit > 0
    quality_score: float        # 0-100: Overall trade quality
    recommendation: str         # "STRONG_BUY", "BUY", "HOLD", "AVOID"

class ProfitProtector:
    """
    Prevents the system from trading at a loss when already profitable.
    Implements loss aversion and profit lock strategies.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.peak_balance = initial_capital
        self.current_balance = initial_capital
        self.total_pnl = 0.0
        
        # Loss aversion parameters
        self.min_confidence_holding_profit = 0.75  # Must be 75% confident if in profit
        self.min_confidence_breakeven = 0.60       # 60% confident if breakeven
        self.min_win_rate_for_trading = 0.52       # Need 52% win rate (very conservative)
        
        # Trade history for learning
        self.completed_trades = []  # (entry_price, exit_price, pnl, confidence)
        self.win_rate = 0.5
        self.avg_win = 0.0
        self.avg_loss = 0.0
    
    def update_balance(self, new_balance: float):
        """Update current balance and peak balance"""
        self.current_balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)
        self.total_pnl = new_balance - self.initial_capital
    
    def get_profit_status(self) -> Dict:
        """Returns current profit/loss status"""
        return {
            'total_pnl': self.total_pnl,
            'total_pnl_pct': (self.total_pnl / self.initial_capital * 100),
            'current_balance': self.current_balance,
            'is_profitable': self.total_pnl > 10,  # Need $10+ profit to be "profitable"
            'underwater_pct': ((self.peak_balance - self.current_balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0,
        }
    
    def rate_trade_quality(self, 
                          signal_confidence: float,
                          model_win_rate: float,
                          entry_price: float,
                          stop_loss: float,
                          take_profit: float,
                          position_size: float,
                          current_balance: float) -> TradeQualityScore:
        """
        Rate a potential trade from 0-100 (quality score).
        
        Args:
            signal_confidence: Model's confidence in direction (0-1)
            model_win_rate: Historical win rate of this model/signal
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            position_size: Position size in contracts/shares
            current_balance: Current account balance
        
        Returns:
            TradeQualityScore with recommendation
        """
        profit_status = self.get_profit_status()
        
        # 1. CONFIDENCE CHECK
        # If holding profit, require higher confidence
        if profit_status['is_profitable']:
            required_confidence = self.min_confidence_holding_profit
        else:
            required_confidence = self.min_confidence_breakeven
        
        confidence_ok = signal_confidence >= required_confidence
        confidence_score = min(100, signal_confidence * 100)
        
        # 2. WIN RATE CHECK
        # Model's historical accuracy must support trading
        win_rate_ok = model_win_rate >= self.min_win_rate_for_trading
        win_rate_score = min(100, model_win_rate * 100)
        
        # 3. RISK/REWARD ANALYSIS
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk > 0:
            risk_reward_ratio = reward / risk
        else:
            risk_reward_ratio = 0.0
        
        # Want at least 2:1 reward/risk
        rr_ok = risk_reward_ratio >= 2.0
        rr_score = min(100, (risk_reward_ratio / 2.0) * 100)
        
        # 4. LOSS PROBABILITY PREDICTION
        # Use confidence and win rate to predict P(loss)
        p_win = signal_confidence * model_win_rate
        p_loss = 1.0 - p_win
        loss_prob_ok = p_loss < 0.35  # Don't trade if >35% chance of loss
        
        # 5. PROFIT EXPECTANCY
        avg_win_dollars = reward * position_size if reward > 0 else 0
        avg_loss_dollars = -risk * position_size if risk > 0 else 0
        
        expected_value = (p_win * avg_win_dollars) + (p_loss * avg_loss_dollars)
        expected_pct = (expected_value / current_balance * 100) if current_balance > 0 else 0
        
        profit_ok = expected_value > 5.0  # Need at least $5 expected profit
        
        # 6. AGGREGATE QUALITY SCORE
        scores = [confidence_score, win_rate_score, rr_score]
        quality_score = np.mean(scores)
        
        # PENALTIES for being in profit
        if profit_status['is_profitable']:
            if signal_confidence < 0.75:
                quality_score *= 0.7  # Heavy penalty for low confidence in profit
            if model_win_rate < 0.55:
                quality_score *= 0.6  # Heavy penalty for low win rate
        
        # 7. FINAL RECOMMENDATION
        if quality_score >= 75 and confidence_ok and profit_ok and loss_prob_ok:
            recommendation = "STRONG_BUY"
        elif quality_score >= 60 and confidence_ok and profit_ok:
            recommendation = "BUY"
        elif quality_score >= 40:
            recommendation = "HOLD"
        else:
            recommendation = "AVOID"
        
        # Override: If in profit and high risk, always avoid
        if profit_status['is_profitable'] and (signal_confidence < 0.65 or p_loss > 0.40):
            recommendation = "AVOID"
        
        return TradeQualityScore(
            confidence=signal_confidence,
            win_probability=p_win,
            profit_expectancy=expected_value,
            risk_reward_ratio=risk_reward_ratio,
            is_profitable=expected_value > 0,
            quality_score=quality_score,
            recommendation=recommendation
        )
    
    def should_enter_trade(self,
                          trade_quality: TradeQualityScore,
                          current_balance: float,
                          force_hold_if_profitable: bool = True) -> Tuple[bool, str]:
        """
        Decide whether to enter a trade based on quality score.
        
        Args:
            trade_quality: TradeQualityScore from rate_trade_quality()
            current_balance: Current account balance
            force_hold_if_profitable: If True, never trade when in profit unless STRONG_BUY
        
        Returns:
            (should_trade, reason)
        """
        profit_status = self.get_profit_status()
        
        # Rule 1: If quality is too low, don't trade
        if trade_quality.recommendation == "AVOID":
            return False, f"Poor trade quality ({trade_quality.quality_score:.0f}/100). P(loss)={1-trade_quality.win_probability:.1%}"
        
        # Rule 2: If in profit, require STRONG_BUY
        if profit_status['is_profitable'] and force_hold_if_profitable:
            if trade_quality.recommendation != "STRONG_BUY":
                return False, f"In profit (${profit_status['total_pnl']:+.2f}). Need STRONG_BUY, got {trade_quality.recommendation}"
        
        # Rule 3: Expected profit must be positive
        if trade_quality.profit_expectancy <= 0:
            return False, f"Negative expectancy: ${trade_quality.profit_expectancy:.2f}"
        
        # Rule 4: Win probability must be >50%
        if trade_quality.win_probability < 0.50:
            return False, f"P(win)={trade_quality.win_probability:.1%} < 50%"
        
        # Rule 5: If underwater (below peak), be cautious
        if profit_status['underwater_pct'] > 5:
            if trade_quality.confidence < 0.70:
                return False, f"Down {profit_status['underwater_pct']:.1f}% from peak and low confidence"
        
        return True, f"✅ Trade approved: {trade_quality.recommendation} (Quality {trade_quality.quality_score:.0f}/100)"
    
    def log_trade_result(self, entry_price: float, exit_price: float, 
                         confidence: float, position_size: float = 1.0):
        """Record a completed trade for learning"""
        pnl = (exit_price - entry_price) * position_size
        pnl_pct = ((exit_price - entry_price) / entry_price * 100)
        
        self.completed_trades.append({
            'entry': entry_price,
            'exit': exit_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'confidence': confidence,
            'won': pnl > 0
        })
        
        # Update win rate
        recent_trades = self.completed_trades[-50:]  # Last 50 trades
        wins = sum(1 for t in recent_trades if t['won'])
        self.win_rate = wins / len(recent_trades) if recent_trades else 0.5
        
        # Update avg win/loss
        wins_list = [t['pnl'] for t in recent_trades if t['won']]
        losses_list = [t['pnl'] for t in recent_trades if not t['won']]
        
        self.avg_win = np.mean(wins_list) if wins_list else 0.0
        self.avg_loss = np.mean(losses_list) if losses_list else 0.0
    
    def get_adaptive_position_size(self,
                                  base_risk_pct: float = 1.0,
                                  trade_quality: Optional[TradeQualityScore] = None) -> float:
        """
        Adapt position size based on confidence and win rate.
        High confidence + high win rate = bigger position
        Low confidence = tiny position
        """
        if trade_quality is None:
            return base_risk_pct
        
        # Confidence multiplier: 0.5x to 2.0x
        confidence_mult = 0.5 + (trade_quality.confidence * 1.5)
        
        # Win rate multiplier: 0.3x to 1.5x
        win_rate_adj = max(0.0, self.win_rate - 0.50) * 10  # 0 if 50%, 1.0 if 60%
        win_rate_mult = 0.3 + (win_rate_adj * 1.2)
        
        # Risk reward multiplier: reward/risk ratio
        rr_mult = min(2.0, max(0.5, trade_quality.risk_reward_ratio))
        
        # Combine
        adaptive_size = base_risk_pct * confidence_mult * win_rate_mult * rr_mult
        
        # Cap at 2x base (never more than double)
        return min(base_risk_pct * 2.0, max(base_risk_pct * 0.1, adaptive_size))
    
    def generate_breakeven_stop(self, entry_price: float, slippage_bps: int = 10) -> float:
        """
        Generate a stop loss that protects entry (breakeven + tiny margin).
        
        Args:
            entry_price: Entry price
            slippage_bps: Slippage in basis points (10 = 0.1%)
        
        Returns:
            Stop loss price that locks breakeven
        """
        slippage_pct = slippage_bps / 10000.0
        return entry_price * (1.0 - slippage_pct)


class LossAversionFilter:
    """
    Filters trades to ONLY enter high-probability winning trades.
    Uses multiple signals to confirm trade quality.
    """
    
    def __init__(self):
        self.min_signals_aligned = 2  # At least 2 signals must agree
        self.min_cross_confirmation = 0.70  # % of signals agreeing
    
    def check_multi_signal_alignment(self,
                                     signal_dict: Dict[str, float]) -> Tuple[float, float]:
        """
        Check if multiple signals (L1, L2, L3) are aligned.
        
        Args:
            signal_dict: {'L1_confidence': 0.8, 'L2_score': 0.6, 'L3_ok': 1.0}
        
        Returns:
            (alignment_score, consensus_direction)
        """
        signals = list(signal_dict.values())
        if not signals:
            return 0.0, 0.0
        
        # All signals in same direction?
        pos_signals = sum(1 for s in signals if s > 0.5)
        neg_signals = sum(1 for s in signals if s < 0.5)
        alignment = max(pos_signals, neg_signals) / len(signals)
        
        consensus = np.mean(signals)
        
        return alignment, consensus
    
    def should_block_for_uncertainty(self, 
                                    alignment_score: float,
                                    confidence: float,
                                    in_profit: bool = False) -> bool:
        """
        Block trade if signals are not well-aligned or confidence is low.
        Stricter when in profit (loss aversion).
        """
        if in_profit:
            return alignment_score < 0.80 or confidence < 0.72
        else:
            return alignment_score < 0.65 or confidence < 0.55
