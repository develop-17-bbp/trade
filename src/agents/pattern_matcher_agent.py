"""
Pattern Matcher Agent
======================
Analyzes recent trade history for streaks and patterns, combines with
LSTM ensemble predictions (when available), and adjusts confidence
based on historical pattern quality and data reliability.
"""

from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, AgentVote


class PatternMatcherAgent(BaseAgent):
    """Matches historical trade patterns and LSTM predictions to current setup."""

    def __init__(self, name: str = 'pattern_matcher', config: Dict = None):
        super().__init__(name=name, config=config)

    def analyze(self, quant_state: Dict, context: Dict) -> AgentVote:
        # --- Extract inputs ---
        trade_history: List[Dict] = context.get('trade_history', [])
        raw_signal = context.get('raw_signal', 0)

        lstm = quant_state.get('lstm_ensemble', {})
        lstm_prediction = lstm.get('prediction', None)
        lstm_confidence = lstm.get('confidence', 0.0)

        similar_avg_pnl = context.get('similar_trade_avg_pnl', None)

        reasons = []
        direction = raw_signal
        position_scale = 1.0

        # --- Analyze recent trade history ---
        recent = trade_history[-10:] if len(trade_history) >= 10 else trade_history
        num_recent = len(recent)

        wins = 0
        losses = 0
        win_rate = 0.5
        consecutive_wins = 0
        consecutive_losses = 0
        streak_flag = None

        if num_recent > 0:
            results = []
            for t in recent:
                profitable = t.get('profitable', t.get('pnl', 0) > 0)
                results.append(bool(profitable))

            wins = sum(results)
            losses = num_recent - wins
            win_rate = wins / num_recent

            # Count consecutive streak from most recent
            if results:
                streak_val = results[-1]
                streak_count = 0
                for r in reversed(results):
                    if r == streak_val:
                        streak_count += 1
                    else:
                        break

                if streak_val and streak_count >= 5:
                    consecutive_wins = streak_count
                    streak_flag = 'winning_streak'
                    position_scale = 0.7  # Reversion caution
                    reasons.append(f"Winning streak [{consecutive_wins} consecutive] scale=0.7 (luck reversion caution)")

                elif not streak_val and streak_count >= 5:
                    consecutive_losses = streak_count
                    streak_flag = 'losing_streak'
                    position_scale = 0.3  # High caution
                    reasons.append(f"Losing streak [{consecutive_losses} consecutive] scale=0.3 (protective)")

            reasons.append(f"[WIN_RATE_10={win_rate:.2f}] ({wins}W/{losses}L)")

        # --- LSTM ensemble integration ---
        lstm_alignment_bonus = 1.0
        lstm_available = lstm_prediction is not None and lstm_prediction != 0

        if lstm_available:
            if (lstm_prediction > 0 and raw_signal > 0) or (lstm_prediction < 0 and raw_signal < 0):
                # LSTM aligns with raw signal -> boost
                lstm_alignment_bonus = 1.0 + min(lstm_confidence, 0.5)
                reasons.append(f"LSTM aligned [PRED={lstm_prediction}] [CONF={lstm_confidence:.2f}] boosting")
            elif (lstm_prediction > 0 and raw_signal < 0) or (lstm_prediction < 0 and raw_signal > 0):
                # LSTM opposes raw signal -> reduce confidence
                lstm_alignment_bonus = 0.7
                reasons.append(f"LSTM opposes signal [PRED={lstm_prediction}] vs [RAW={raw_signal}] reducing 30%")
            else:
                reasons.append(f"LSTM neutral [PRED={lstm_prediction}]")
        else:
            reasons.append("LSTM unavailable, using trade history only")

        # --- Similar trade average PnL ---
        data_quality = 0.5
        if similar_avg_pnl is not None:
            if similar_avg_pnl > 0:
                data_quality = min(1.0, 0.5 + similar_avg_pnl * 10)
                reasons.append(f"Similar trades profitable [AVG_PNL={similar_avg_pnl:.4f}]")
            else:
                data_quality = max(0.1, 0.5 + similar_avg_pnl * 10)
                reasons.append(f"Similar trades unprofitable [AVG_PNL={similar_avg_pnl:.4f}]")
        else:
            data_quality = 0.5  # No data, neutral

        # Increase data quality if we have enough trade history
        if num_recent >= 8:
            data_quality = min(1.0, data_quality + 0.2)

        # --- Pattern quality ---
        pattern_quality = win_rate * lstm_alignment_bonus
        pattern_quality = max(0.0, min(2.0, pattern_quality))

        # --- Final confidence ---
        confidence = pattern_quality * data_quality
        confidence = max(0.05, min(1.0, confidence))

        # Override direction to flat if losing streak is severe
        if streak_flag == 'losing_streak' and consecutive_losses >= 7:
            direction = 0
            position_scale = 0.1
            reasons.append("Severe losing streak: going flat")

        # Clamp
        position_scale = max(0.0, min(1.0, position_scale))

        reasoning = "; ".join(reasons) if reasons else "No pattern data available"

        return AgentVote(
            direction=direction,
            confidence=round(confidence, 4),
            position_scale=round(position_scale, 4),
            reasoning=reasoning,
            metadata={
                'streak_flag': streak_flag,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses,
                'win_rate_last_10': round(win_rate, 4),
                'lstm_available': lstm_available,
                'lstm_alignment_bonus': round(lstm_alignment_bonus, 4),
                'pattern_quality': round(pattern_quality, 4),
                'data_quality': round(data_quality, 4),
            },
        )
