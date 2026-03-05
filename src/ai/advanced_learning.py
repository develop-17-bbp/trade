"""
PHASE 6: Advanced Learning (Meta-Learning) - Layer 9
=======================================================
Self-adapting intelligent trading system that:
- Learns cross-market patterns
- Generates adaptive strategies dynamically
- Classifies market regimes in real-time
- Self-modifies algorithms based on performance
- Meta-learns optimal hyperparameters across markets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import pickle
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketRegime:
    """Identified market regime state."""
    regime_type: str  # TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, MEAN_REVERTING
    confidence: float  # 0-100%
    volatility: float  # Current volatility
    trend_strength: float  # -1.0 to 1.0
    regime_duration_bars: int
    optimal_strategy: str  # Strategy suited for this regime
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class GeneratedStrategy:
    """Dynamically generated trading strategy."""
    strategy_id: str
    name: str
    entry_signals: Dict[str, Any]  # Dynamic entry conditions
    exit_signals: Dict[str, Any]   # Dynamic exit conditions
    position_sizing: Dict[str, float]  # Dynamic sizing rules
    risk_params: Dict[str, float]  # Dynamic risk parameters
    performance_score: float  # How well it performed
    applicable_regimes: List[str]
    creation_timestamp: str
    last_updated: str


class CrossMarketPatternRecognizer:
    """
    Identifies patterns that work across multiple markets.
    Learns what conditions lead to profitable trades regardless of asset.
    """
    
    def __init__(self, lookback_periods: int = 500):
        self.lookback_periods = lookback_periods
        self.pattern_memory = defaultdict(list)  # Store detected patterns
        self.cross_market_correlations = {}
        self.pattern_success_rate = defaultdict(lambda: {"wins": 0, "losses": 0})
        
    def extract_market_features(self, ohlcv_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract key features from market data that define conditions.
        """
        close = ohlcv_data['close'].values
        high = ohlcv_data['high'].values
        low = ohlcv_data['low'].values
        volume = ohlcv_data['volume'].values
        
        # Momentum features
        rocs = [(close[i] - close[i-20]) / close[i-20] for i in range(20, len(close))]
        momentum = np.mean(rocs[-50:]) if len(rocs) >= 50 else 0
        
        # Volatility features
        returns = np.diff(close) / close[:-1]
        volatility = np.std(returns[-50:]) if len(returns) >= 50 else 0
        
        # Trend features
        sma20 = np.mean(close[-20:])
        sma50 = np.mean(close[-50:])
        trend = (sma20 - sma50) / sma50 if sma50 != 0 else 0
        
        # Volume features
        avg_vol = np.mean(volume[-20:])
        vol_ratio = volume[-1] / avg_vol if avg_vol != 0 else 1
        
        # Mean reversion features
        zscore = (close[-1] - np.mean(close[-50:])) / np.std(close[-50:]) if np.std(close[-50:]) > 0 else 0
        
        # Extreme moves
        recent_high = np.max(high[-20:])
        recent_low = np.min(low[-20:])
        position_in_range = (close[-1] - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5
        
        return {
            "momentum": momentum,
            "volatility": volatility,
            "trend": trend,
            "volume_ratio": vol_ratio,
            "zscore": zscore,
            "position_in_range": position_in_range,
            "sma_ratio": sma20 / sma50 if sma50 != 0 else 1,
        }
    
    def recognize_patterns(self, multi_asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Identify recurring profitable patterns across assets.
        """
        patterns = {
            "momentum_breakout": [],
            "mean_reversion": [],
            "volatility_expansion": [],
            "trend_continuation": [],
            "regime_shift": []
        }
        
        for asset, data in multi_asset_data.items():
            features = self.extract_market_features(data)
            
            # Pattern detection logic
            if features["momentum"] > 0.05 and features["volume_ratio"] > 1.5:
                patterns["momentum_breakout"].append((asset, features["momentum"]))
            
            if abs(features["zscore"]) > 2.0 and features["volatility"] < 0.05:
                patterns["mean_reversion"].append((asset, abs(features["zscore"])))
            
            if features["volatility"] > 0.08 and features["volume_ratio"] > 2.0:
                patterns["volatility_expansion"].append((asset, features["volatility"]))
            
            if features["trend"] > 0.02 and features["sma_ratio"] > 1.01:
                patterns["trend_continuation"].append((asset, features["trend"]))
        
        # Store in memory for meta-learning
        self.pattern_memory[datetime.now().isoformat()] = patterns
        
        return patterns
    
    def compute_cross_market_correlation(self, multi_asset_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute which assets move together - for hedging and correlation-based strategies.
        """
        prices = {}
        for asset, data in multi_asset_data.items():
            prices[asset] = data['close'].pct_change().values
        
        assets = list(prices.keys())
        correlations = {}
        
        for i, asset1 in enumerate(assets):
            for asset2 in assets[i+1:]:
                corr = np.corrcoef(prices[asset1], prices[asset2])[0, 1]
                pair = f"{asset1}_{asset2}"
                correlations[pair] = corr
        
        self.cross_market_correlations = correlations
        return correlations
    
    def get_correlated_assets(self, primary_asset: str, correlation_threshold: float = 0.7) -> List[str]:
        """
        Find assets that move with the primary asset (good hedges or signals).
        """
        correlated = []
        for pair, corr in self.cross_market_correlations.items():
            if primary_asset in pair and abs(corr) > correlation_threshold:
                other = pair.replace(primary_asset, "").replace("_", "")
                correlated.append((other, corr))
        return correlated


class MarketRegimeClassifier:
    """
    Real-time market regime identification.
    Determines optimal strategy for current market conditions.
    """
    
    def __init__(self, lookback_bars: int = 100):
        self.lookback_bars = lookback_bars
        self.regime_history = []
        self.regime_transition_probabilities = defaultdict(lambda: defaultdict(int))
        
    def classify_regime(self, close: np.ndarray, high: np.ndarray, 
                       low: np.ndarray, volume: np.ndarray, 
                       onchain_signals: Optional[Dict[str, float]] = None) -> MarketRegime:
        """
        Classify current market regime using multi-factor analysis.
        """
        if len(close) < self.lookback_bars:
            return MarketRegime("UNKNOWN", 0, 0, 0, 0, "HOLD")
        
        recent_close = close[-self.lookback_bars:]
        recent_high = high[-self.lookback_bars:]
        recent_low = low[-self.lookback_bars:]
        recent_vol = volume[-self.lookback_bars:]
        
        # Calculate metrics
        returns = np.diff(recent_close) / recent_close[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Trend strength (using linear regression)
        x = np.arange(len(recent_close))
        slope = np.polyfit(x, recent_close, 1)[0]
        trend_strength = slope / np.mean(recent_close) * 100
        
        # Range metrics
        range_high = np.max(recent_high)
        range_low = np.min(recent_low)
        price_position = (close[-1] - range_low) / (range_high - range_low)
        
        # Mean reversion metrics
        mean_price = np.mean(recent_close)
        std_price = np.std(recent_close)
        z_score = (close[-1] - mean_price) / std_price if std_price > 0 else 0
        
        # Volume trend
        avg_vol = np.mean(recent_vol)
        vol_expansion = recent_vol[-1] / avg_vol if avg_vol > 0 else 1
        
        # Incorporate onchain signals into regime classification
        onchain_bias = 0.0
        whale_influence = 0.0
        if onchain_signals:
            whale_score = onchain_signals.get('whale_score', 0.0)
            onchain_momentum = onchain_signals.get('on_chain_momentum', 0.0)
            liquidation_risk = onchain_signals.get('liquidation_risk', 50.0)
            
            # Whale activity influences regime confidence
            whale_influence = abs(whale_score) * 0.3
            
            # Onchain momentum adjusts trend strength
            onchain_bias = onchain_momentum * 0.2
            
            # High liquidation risk suggests potential volatility
            if liquidation_risk > 70:
                volatility += 0.02  # Increase perceived volatility
        
        # Adjust trend strength with onchain momentum
        adjusted_trend_strength = trend_strength + onchain_bias
        
        # Regime classification logic (enhanced with onchain)
        if volatility > 0.05 and vol_expansion > 1.5:
            regime_type = "VOLATILE"
            optimal_strategy = "RANGE_BOUND"
        elif abs(adjusted_trend_strength) > 2.0 and vol_expansion > 1.0:
            regime_type = "TRENDING_UP" if adjusted_trend_strength > 0 else "TRENDING_DOWN"
            optimal_strategy = "TREND_FOLLOWING"
        elif abs(z_score) > 2.0 and volatility < 0.03:
            regime_type = "MEAN_REVERTING"
            optimal_strategy = "MEAN_REVERSION"
        elif volatility < 0.02 and abs(adjusted_trend_strength) < 1.0:
            regime_type = "RANGING"
            optimal_strategy = "SCALPING"
        else:
            regime_type = "NEUTRAL"
            optimal_strategy = "HOLD"
        
        # Calculate confidence (0-100%) - enhanced with whale influence
        confidence = min(100, 50 + abs(adjusted_trend_strength) * 10 + (volatility - 0.02) * 1000 + whale_influence * 100)
        confidence = max(0, confidence)
        
        regime = MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            volatility=volatility,
            trend_strength=trend_strength / 100,  # Normalize to -1 to 1
            regime_duration_bars=len(recent_close),
            optimal_strategy=optimal_strategy
        )
        
        self.regime_history.append(regime)
        return regime


class AdaptiveStrategyGenerator:
    """
    Generates trading strategies dynamically based on current market conditions.
    Self-modifying algorithms that adapt in real-time.
    """
    
    def __init__(self):
        self.strategy_library = {}
        self.performance_history = defaultdict(list)
        self.hyperparameter_space = self._define_hyperparameter_space()
        self.generated_strategies = {}
        
    def _define_hyperparameter_space(self) -> Dict[str, Tuple[float, float]]:
        """
        Define the search space for dynamic strategy generation.
        """
        return {
            "rsi_period": (7, 28),
            "rsi_oversold": (20, 40),
            "rsi_overbought": (60, 80),
            "ma_fast": (5, 20),
            "ma_slow": (30, 100),
            "breakout_atr_mult": (1.5, 3.0),
            "stop_loss_atr_mult": (1.5, 3.0),
            "take_profit_mult": (1.5, 4.0),
            "position_size_pct": (0.5, 2.5),
            "volatility_filter": (0.01, 0.1),
        }
    
    def generate_strategy_for_regime(self, regime: MarketRegime, 
                                     recent_performance: Dict[str, float]) -> GeneratedStrategy:
        """
        Generate optimal strategy parameters for the identified regime.
        """
        strategy_id = f"{regime.regime_type}_{datetime.now().isoformat()}"
        
        if regime.regime_type == "TRENDING_UP":
            strategy = self._generate_trend_following_strategy(regime, recent_performance)
        elif regime.regime_type == "TRENDING_DOWN":
            strategy = self._generate_short_strategy(regime, recent_performance)
        elif regime.regime_type == "MEAN_REVERTING":
            strategy = self._generate_mean_reversion_strategy(regime, recent_performance)
        elif regime.regime_type == "VOLATILE":
            strategy = self._generate_range_bound_strategy(regime, recent_performance)
        else:
            strategy = self._generate_hold_strategy(regime)
        
        strategy.strategy_id = strategy_id
        self.generated_strategies[strategy_id] = strategy
        return strategy
    
    def _generate_trend_following_strategy(self, regime: MarketRegime, 
                                          perf: Dict[str, float]) -> GeneratedStrategy:
        """Generate trend-following strategy."""
        # Adapt parameters based on recent performance
        ma_fast = int(10 + (perf.get("sharpe", 0.5) * 5))
        ma_slow = int(50 + (perf.get("sharpe", 0.5) * 20))
        atr_mult = 2.0 + (perf.get("volatility", 0.03) * 100)
        
        return GeneratedStrategy(
            strategy_id="",
            name=f"Trend Following ({regime.regime_type})",
            entry_signals={
                "type": "ma_crossover",
                "fast_period": ma_fast,
                "slow_period": ma_slow,
                "confirmation": "volume_surge"
            },
            exit_signals={
                "type": "atr_trailing_stop",
                "atr_period": 14,
                "atr_multiplier": atr_mult,
                "take_profit_multiplier": 2.5
            },
            position_sizing={
                "method": "volatility_adjusted",
                "base_percent": 1.5,
                "max_percent": 2.5
            },
            risk_params={
                "max_position_size": 2.5,
                "stop_loss_atr": atr_mult,
                "daily_loss_limit": 3.0
            },
            performance_score=0.0,
            applicable_regimes=["TRENDING_UP", "TRENDING_DOWN"],
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_short_strategy(self, regime: MarketRegime, 
                                perf: Dict[str, float]) -> GeneratedStrategy:
        """Generate short-selling strategy."""
        return GeneratedStrategy(
            strategy_id="",
            name=f"Short Strategy ({regime.regime_type})",
            entry_signals={
                "type": "high_rsi_short",
                "rsi_period": 14,
                "rsi_overbought": 65
            },
            exit_signals={
                "type": "rsi_recovery",
                "rsi_recovery_level": 50
            },
            position_sizing={
                "method": "fixed",
                "percent": 1.0
            },
            risk_params={
                "max_position_size": 1.5,
                "stop_loss_pct": 2.0
            },
            performance_score=0.0,
            applicable_regimes=["TRENDING_DOWN"],
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_mean_reversion_strategy(self, regime: MarketRegime, 
                                         perf: Dict[str, float]) -> GeneratedStrategy:
        """Generate mean reversion strategy."""
        return GeneratedStrategy(
            strategy_id="",
            name=f"Mean Reversion ({regime.regime_type})",
            entry_signals={
                "type": "zscore_extreme",
                "zscore_threshold": 2.0,
                "window": 20
            },
            exit_signals={
                "type": "revert_to_mean",
                "exit_zscore": 0.5
            },
            position_sizing={
                "method": "volatility_adjusted",
                "base_percent": 2.0
            },
            risk_params={
                "max_position_size": 2.5,
                "stop_loss_zscore": 3.0
            },
            performance_score=0.0,
            applicable_regimes=["MEAN_REVERTING", "RANGING"],
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_range_bound_strategy(self, regime: MarketRegime, 
                                       perf: Dict[str, float]) -> GeneratedStrategy:
        """Generate range-bound / scalping strategy."""
        return GeneratedStrategy(
            strategy_id="",
            name=f"Range Bound ({regime.regime_type})",
            entry_signals={
                "type": "support_resistance",
                "lookback": 50,
                "entry_distance_pct": 0.5
            },
            exit_signals={
                "type": "midpoint_exit",
                "exit_at_midpoint": True
            },
            position_sizing={
                "method": "micro_position",
                "percent": 0.5
            },
            risk_params={
                "max_position_size": 1.0,
                "tight_stops": True
            },
            performance_score=0.0,
            applicable_regimes=["RANGING", "VOLATILE"],
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def _generate_hold_strategy(self, regime: MarketRegime) -> GeneratedStrategy:
        """Generate hold/wait strategy for uncertain conditions."""
        return GeneratedStrategy(
            strategy_id="",
            name="Hold/Wait",
            entry_signals={"type": "none"},
            exit_signals={"type": "none"},
            position_sizing={"method": "zero"},
            risk_params={"max_position_size": 0},
            performance_score=0.0,
            applicable_regimes=["NEUTRAL", "UNKNOWN"],
            creation_timestamp=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
    
    def optimize_strategy_hyperparameters(self, strategy: GeneratedStrategy, 
                                         backtest_results: Dict[str, float]) -> GeneratedStrategy:
        """
        Optimize strategy hyperparameters based on backtest results.
        Self-modifying algorithm.
        """
        sharpe = backtest_results.get("sharpe_ratio", 0)
        win_rate = backtest_results.get("win_rate", 0.5)
        max_dd = backtest_results.get("max_drawdown", 0)
        
        # Adjust parameters based on performance
        if sharpe < 0.5:
            # Reduce position size if returns per unit of risk are low
            strategy.position_sizing["base_percent"] *= 0.9
        elif sharpe > 2.0:
            # Increase position size if risk-adjusted returns are high
            strategy.position_sizing["base_percent"] *= 1.1
        
        # Adjust stops based on drawdown
        if max_dd > 0.15:
            # Tighten stops if drawdown is excessive
            strategy.risk_params["stop_loss_atr_mult"] *= 0.8
        
        # Update strategy
        strategy.last_updated = datetime.now().isoformat()
        return strategy


class MetaLearningEngine:
    """
    Meta-learning: learn how to learn better trading strategies.
    Learns optimal hyperparameters across different markets and time periods.
    """
    
    def __init__(self):
        self.meta_model_history = []
        self.hyperparameter_effectiveness = defaultdict(list)
        self.strategy_effectiveness_db = {}
        self.market_to_strategy_map = {}
        
    def learn_optimal_hyperparameters(self, 
                                     asset: str,
                                     tested_strategies: List[GeneratedStrategy],
                                     results: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Meta-learn which hyperparameters work best for this asset/regime.
        """
        # Track effectiveness for each hyperparameter combination
        best_hyperparams = {}
        best_score = -float('inf')
        
        for strategy, result in zip(tested_strategies, results):
            # Compute meta-score (combination of returns, Sharpe, drawdown)
            meta_score = (
                result.get("total_return", 0) * 0.4 +
                result.get("sharpe_ratio", 0) * 0.35 +
                (1 - min(result.get("max_drawdown", 1), 0.5)) * 0.25
            )
            
            # Store effectiveness
            self.hyperparameter_effectiveness[asset].append({
                "hyperparams": strategy.entry_signals,
                "score": meta_score
            })
            
            if meta_score > best_score:
                best_score = meta_score
                best_hyperparams = strategy.entry_signals
        
        # Store the asset-specific strategy map
        self.market_to_strategy_map[asset] = best_hyperparams
        logger.info(f"[META-LEARNING] Optimal hyperparams for {asset}: {best_hyperparams}")
        
        return best_hyperparams
    
    def transfer_learning(self, source_asset: str, target_asset: str) -> Dict[str, float]:
        """
        Transfer learned hyperparameters from one asset to another.
        Useful when starting to trade a new asset.
        """
        if source_asset in self.market_to_strategy_map:
            source_params = self.market_to_strategy_map[source_asset].copy()
            logger.info(f"[TRANSFER-LEARNING] Transferring params from {source_asset} to {target_asset}")
            return source_params
        return {}
    
    def predict_strategy_performance(self, asset: str, strategy: GeneratedStrategy) -> float:
        """
        Predict how well a strategy will perform on an asset using meta-learning.
        """
        if asset not in self.hyperparameter_effectiveness:
            return 0.5  # Base confidence
        
        # Find similar historical strategies
        historical = self.hyperparameter_effectiveness[asset]
        similarities = []
        
        for hist in historical:
            similarity = self._compute_param_similarity(
                strategy.entry_signals, 
                hist["hyperparams"]
            )
            weighted_score = similarity * hist["score"]
            similarities.append(weighted_score)
        
        predicted_performance = np.mean(similarities) if similarities else 0.5
        return predicted_performance
    
    def _compute_param_similarity(self, params1: Dict, params2: Dict) -> float:
        """
        Compute similarity between two hyperparameter sets.
        """
        common_keys = set(params1.keys()) & set(params2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            v1, v2 = params1[key], params2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # Numeric similarity
                similarity = 1 - (abs(v1 - v2) / (abs(v1) + abs(v2) + 1))
                similarities.append(max(0, similarity))
            elif v1 == v2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def save_meta_model(self, filepath: str):
        """Save learned meta-model for persistence."""
        meta_model = {
            "hyperparameter_effectiveness": dict(self.hyperparameter_effectiveness),
            "market_to_strategy_map": self.market_to_strategy_map,
            "timestamp": datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(meta_model, f, indent=2, default=str)
        logger.info(f"[META-LEARNING] Model saved to {filepath}")
    
    def load_meta_model(self, filepath: str):
        """Load previously learned meta-model."""
        try:
            with open(filepath, 'r') as f:
                meta_model = json.load(f)
            self.hyperparameter_effectiveness = defaultdict(list, meta_model.get("hyperparameter_effectiveness", {}))
            self.market_to_strategy_map = meta_model.get("market_to_strategy_map", {})
            logger.info(f"[META-LEARNING] Model loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"[META-LEARNING] No saved model found at {filepath}")


class AdvancedLearningEngine:
    """
    Main orchestrator for Phase 6 Advanced Learning.
    Combines pattern recognition, regime classification, strategy generation, and meta-learning.
    """
    
    def __init__(self, meta_model_path: str = "models/meta_learning_model.json"):
        self.pattern_recognizer = CrossMarketPatternRecognizer()
        self.regime_classifier = MarketRegimeClassifier()
        self.strategy_generator = AdaptiveStrategyGenerator()
        self.meta_learner = MetaLearningEngine()
        
        # Try to load existing meta-model
        self.meta_learner.load_meta_model(meta_model_path)
        self.meta_model_path = meta_model_path
        
        self.active_strategies = {}  # asset -> GeneratedStrategy
        self.performance_tracker = defaultdict(list)
        
    def process_market_data(self, multi_asset_data: Dict[str, pd.DataFrame], 
                           onchain_data: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Main entry point: process market data and generate adaptive strategies.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "regimes": {},
            "strategies": {},
            "patterns": {},
            "correlations": {}
        }
        
        # Step 1: Recognize cross-market patterns
        patterns = self.pattern_recognizer.recognize_patterns(multi_asset_data)
        result["patterns"] = patterns
        
        # Step 2: Compute cross-market correlations
        correlations = self.pattern_recognizer.compute_cross_market_correlation(multi_asset_data)
        result["correlations"] = correlations
        
        # Step 3: Classify regime for each asset
        for asset, data in multi_asset_data.items():
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            volume = data['volume'].values
            
            # Incorporate onchain data if available
            onchain_signals = {}
            if onchain_data and asset in onchain_data:
                onchain = onchain_data[asset]
                onchain_signals = {
                    'whale_score': onchain.get('whale_score', 0.0),
                    'on_chain_momentum': onchain.get('on_chain_momentum', 0.0),
                    'liquidation_risk': onchain.get('liquidation_risk', 50.0),
                    'network_health': onchain.get('network_health', 50.0),
                    'exchange_flow': onchain.get('exchange_flow', 0.0)
                }
            
            regime = self.regime_classifier.classify_regime(close, high, low, volume, onchain_signals)
            result["regimes"][asset] = regime
            
            # Step 4: Generate adaptive strategy for this regime
            recent_perf = self.performance_tracker[asset][-20:] if asset in self.performance_tracker else []
            perf_dict = {
                "sharpe": np.mean([p.get("sharpe_ratio", 0) for p in recent_perf]) if recent_perf else 0,
                "volatility": np.mean([p.get("volatility", 0.03) for p in recent_perf]) if recent_perf else 0.03
            }
            
            strategy = self.strategy_generator.generate_strategy_for_regime(regime, perf_dict)
            
            # Step 5: Meta-learning optimization
            predicted_perf = self.meta_learner.predict_strategy_performance(asset, strategy)
            strategy.performance_score = predicted_perf
            
            self.active_strategies[asset] = strategy
            result["strategies"][asset] = {
                "strategy_name": strategy.name,
                "predicted_performance": predicted_perf,
                "entry_signals": strategy.entry_signals,
                "position_size": strategy.position_sizing
            }
        
        return result
    
    def update_with_backtest_results(self, asset: str, backtest_results: Dict[str, float]):
        """
        Update meta-learning with actual backtest results.
        Self-optimization loop.
        """
        self.performance_tracker[asset].append(backtest_results)
        
        if asset in self.active_strategies:
            strategy = self.active_strategies[asset]
            
            # Optimize strategy based on results
            optimized = self.strategy_generator.optimize_strategy_hyperparameters(
                strategy, backtest_results
            )
            self.active_strategies[asset] = optimized
            
            logger.info(f"[ADVANCED-LEARNING] Updated strategy for {asset} based on backtest")
    
    def discover_new_trading_patterns(self, asset: str, lookback_days: int = 365) -> Dict[str, Any]:
        """
        Discover new profitable patterns in an asset's history.
        """
        # This would load historical data and run meta-learning
        return {
            "high_probability_patterns": [],
            "new_entry_signals": [],
            "optimal_timeframes": [],
            "confidence_score": 0.0
        }
    
    def save_learned_models(self):
        """Persist all learned models."""
        self.meta_learner.save_meta_model(self.meta_model_path)
        logger.info("[ADVANCED-LEARNING] All models saved successfully")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the advanced learning system."""
        return {
            "active_strategies": len(self.active_strategies),
            "patterns_discovered": len(self.pattern_recognizer.pattern_memory),
            "meta_model_assets": len(self.meta_learner.market_to_strategy_map),
            "performance_history_events": len(self.performance_tracker),
            "active_regimes": {k: v.regime_type for k, v in self.regime_classifier.regime_history[-5:]},
            "timestamp": datetime.now().isoformat()
        }

