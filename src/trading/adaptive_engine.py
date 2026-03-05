"""
Adaptive Strategy Engine -- Dynamic Strategy Selection & Learning
=============================================================
Chooses the optimal sub-strategy based on:
1. Sentiment (FinBERT)
2. Volatility Regime
3. Momentum / Trend Strength
4. Historical performance (Auto-tuning weights)
"""
import numpy as np
import os
import json
from typing import List, Dict, Optional
from src.trading.sub_strategies import (
    MeanReversionStrategy, 
    TrendFollowingStrategy, 
    VolatilityBreakoutStrategy,
    ScalpingStrategy
)

class AdaptiveEngine:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.strategies = {
            'mean_reversion': MeanReversionStrategy(),
            'trend_following': TrendFollowingStrategy(),
            'volatility_breakout': VolatilityBreakoutStrategy(),
            'scalping': ScalpingStrategy()
        }
        
        # Performance memory (learning system)
        # Higher score = more weight to that strategy
        self.perf_scores = {k: 1.0 for k in self.strategies.keys()}
        self.memory_file = "c:/Users/convo/trade/strategy_memory.json"
        self._load_memory()

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.perf_scores.update(data)
            except: pass

    def _save_memory(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.perf_scores, f)

    def select_strategy(self, features: Dict[str, float], sentiment_data: Dict) -> str:
        """Helper to interface with HybridStrategy's feature format."""
        sentiment_score = sentiment_data.get('aggregate_score', 0.0)
        vol = features.get('ewma_vol', 0.02)
        trend = features.get('ema_10_slope', 0.0)
        chop = features.get('adx_strength', 25.0) # ADX as proxy for non-chop trendiness
        
        return self.select_best_strategy(
            sentiment_score=sentiment_score,
            volatility=vol,
            trend_strength=trend,
            chop_index=100 - chop, # Invert ADX for chop approx
            asset_type='BTC'
        )

    def select_best_strategy(self, 
                             sentiment_score: float, 
                             volatility: float, 
                             trend_strength: float,
                             chop_index: float = 50.0,
                             asset_type: str = 'BTC') -> str:
        """
        Logic to pick the core strategy for the current market state.
        """
        # Logic 1: Extreme Volatility -> Breakout
        if volatility > 0.05:
            return 'volatility_breakout'
        
        # Logic 2: Choppy Market (CHOP > 60) -> Scalping
        if chop_index > 60.0:
            return 'scalping'
        
        # Logic 3: Strong Trend + Positive Sentiment -> Trend Following
        if abs(trend_strength) > 0.7 and abs(sentiment_score) > 0.2:
            return 'trend_following'
        
        # Logic 4: Low Volatility + Neutral Sentiment -> Mean Reversion
        if volatility < 0.02 and abs(sentiment_score) < 0.1:
            return 'mean_reversion'
        
        # Default: Pick based on historical performance (Learning)
        best_strat = max(self.perf_scores, key=self.perf_scores.get)
        return best_strat

    def update_learning(self, strategy_used: str, pnl: float):
        """
        Reinforcement Learning: Update strategy weights based on actual P&L.
        """
        if strategy_used in self.perf_scores:
            # Simple reward system
            learning_rate = 0.05
            reward = 1.0 if pnl > 0 else -1.0 if pnl < 0 else 0
            self.perf_scores[strategy_used] += reward * learning_rate
            
            # Keep scores within reasonable bounds
            self.perf_scores[strategy_used] = max(0.1, min(2.0, self.perf_scores[strategy_used]))
            self._save_memory()

    def generate_adaptive_signal(self, 
                                 prices: List[float], 
                                 highs: List[float], 
                                 lows: List[float], 
                                 volumes: List[float],
                                 sentiment_score: float,
                                 asset: str = 'BTC') -> Dict:
        """
        The main entry point: Analyzes market -> Picks Strategy -> Generates Signal.
        """
        # Calculate market state
        arr_prices = np.asarray(prices)
        if len(arr_prices) < 20:
             return {
                'strategy_selected': 'mean_reversion',
                'signal': 0,
                'market_state': {'volatility': 0, 'trend': 0, 'sentiment': sentiment_score}
            }

        vol = (np.max(arr_prices[-20:]) - np.min(arr_prices[-20:])) / np.mean(arr_prices[-20:])
        
        from src.indicators.indicators import sma, choppiness_index
        ma20 = sma(prices, 20)
        ma50 = sma(prices, 50) if len(prices) >= 50 else ma20
        trend = 1.0 if ma20[-1] > ma50[-1] else -1.0
        
        chop_vals = choppiness_index(highs, lows, prices, 14)
        chop = chop_vals[-1] if len(chop_vals) > 0 else 50.0
        
        strategy_name = self.select_best_strategy(sentiment_score, vol, trend, chop, asset)
        strategy = self.strategies[strategy_name]
        
        raw_signal = strategy.generate_signal(prices, highs, lows, volumes)
        
        return {
            'strategy_selected': strategy_name,
            'signal': raw_signal,
            'market_state': {
                'volatility': vol,
                'trend': trend,
                'sentiment': sentiment_score
            }
        }
