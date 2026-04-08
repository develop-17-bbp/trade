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


class PipelineOverlay:
    """Runtime parameter overlay — adjusts SECONDARY pipeline knobs only.
    NEVER touches v13/v14 core: min_entry_score, max_entry_score, short_score_penalty, sr_assets.
    """
    __slots__ = ('bear_veto_threshold', 'bear_reduce_threshold', 'min_confidence',
                 'ml_weights', 'regime', 'regime_confidence', 'risk_multiplier',
                 'hold_strategy', 'reasoning', 'timestamp')

    def __init__(self, **kwargs):
        self.bear_veto_threshold: int = kwargs.get('bear_veto_threshold', 9)
        self.bear_reduce_threshold: int = kwargs.get('bear_reduce_threshold', 7)
        self.min_confidence: float = kwargs.get('min_confidence', 0.60)
        self.ml_weights: Dict[str, float] = kwargs.get('ml_weights', {
            'lgbm': 1.0, 'lstm': 1.0, 'patchtst': 1.0, 'rl': 1.0, 'hmm': 1.0,
        })
        self.regime: str = kwargs.get('regime', 'NEUTRAL')
        self.regime_confidence: float = kwargs.get('regime_confidence', 0.5)
        self.risk_multiplier: float = kwargs.get('risk_multiplier', 1.0)
        self.hold_strategy: bool = kwargs.get('hold_strategy', False)
        self.reasoning: str = kwargs.get('reasoning', '')
        self.timestamp: str = kwargs.get('timestamp', datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__slots__}


class AdaptiveStrategyGenerator:
    """
    Runtime meta-optimizer: observes pipeline decisions + trade outcomes,
    outputs PipelineOverlay adjustments (secondary knobs only).

    SAFE: Never touches v13/v14 core params (min/max_entry_score, short_score_penalty, sr_assets).
    Instead adjusts: bear thresholds, ML model weights, confidence gates, risk multipliers.
    """

    def __init__(self):
        self.performance_history: Dict[str, List[Dict]] = defaultdict(list)
        self.overlay_history: Dict[str, List[PipelineOverlay]] = defaultdict(list)
        # EWMA smoothing factor for online learning
        self._ewma_alpha: float = 0.15
        # Per-asset running stats (online — no batch needed)
        self._running_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            'win_rate': 0.5, 'avg_pnl': 0.0, 'loss_streak': 0, 'win_streak': 0,
            'total_trades': 0, 'last_regime': 'NEUTRAL',
        })

    def generate_overlay(self, regime: MarketRegime, asset: str,
                         recent_performance: Dict[str, float]) -> PipelineOverlay:
        """
        Generate pipeline parameter overlay based on current regime + performance.
        Returns SECONDARY knobs only — v13/v14 core is NEVER modified.
        """
        stats = self._running_stats[asset]
        win_rate = stats['win_rate']
        loss_streak = stats['loss_streak']

        # Base overlay — defaults match current pipeline config
        overlay = PipelineOverlay(regime=regime.regime_type,
                                  regime_confidence=regime.confidence / 100.0)

        # ── Regime-based adjustments (SECONDARY knobs only) ──
        if regime.regime_type == "TRENDING_UP":
            # Strong trend: loosen bear veto (let good trends run), boost trend-following models
            overlay.bear_veto_threshold = 9
            overlay.bear_reduce_threshold = 7
            overlay.min_confidence = 0.55  # Allow slightly lower confidence in strong trends
            overlay.ml_weights = {'lgbm': 1.0, 'lstm': 1.2, 'patchtst': 1.2, 'rl': 1.0, 'hmm': 0.8}
            overlay.risk_multiplier = 1.1
            overlay.reasoning = f"TRENDING_UP: loosened bear, boosted trend models"

        elif regime.regime_type == "TRENDING_DOWN":
            # Downtrend: tighten everything for SHORTs
            overlay.bear_veto_threshold = 8
            overlay.bear_reduce_threshold = 6
            overlay.min_confidence = 0.65
            overlay.ml_weights = {'lgbm': 1.2, 'lstm': 1.0, 'patchtst': 1.0, 'rl': 0.8, 'hmm': 1.2}
            overlay.risk_multiplier = 0.8
            overlay.reasoning = f"TRENDING_DOWN: tightened bear, boosted regime models"

        elif regime.regime_type in ("VOLATILE", "ANOMALY"):
            # High vol: defensive — tighten everything, reduce risk
            overlay.bear_veto_threshold = 7
            overlay.bear_reduce_threshold = 5
            overlay.min_confidence = 0.70
            overlay.ml_weights = {'lgbm': 1.3, 'lstm': 0.8, 'patchtst': 0.8, 'rl': 0.5, 'hmm': 1.5}
            overlay.risk_multiplier = 0.5
            overlay.reasoning = f"VOLATILE: defensive mode, tightened all gates"

        elif regime.regime_type == "MEAN_REVERTING":
            # Mean reverting: EMA signals chop, be cautious
            overlay.bear_veto_threshold = 8
            overlay.bear_reduce_threshold = 6
            overlay.min_confidence = 0.65
            overlay.ml_weights = {'lgbm': 1.0, 'lstm': 0.8, 'patchtst': 1.0, 'rl': 1.2, 'hmm': 1.0}
            overlay.risk_multiplier = 0.7
            overlay.reasoning = f"MEAN_REVERTING: EMA chops, reduced trend-follower weights"

        elif regime.regime_type == "RANGING":
            # Ranging: similar to mean reverting but even more cautious
            overlay.bear_veto_threshold = 8
            overlay.bear_reduce_threshold = 6
            overlay.min_confidence = 0.70
            overlay.hold_strategy = True
            overlay.risk_multiplier = 0.5
            overlay.reasoning = f"RANGING: hold strategy, minimal entries"

        else:
            overlay.reasoning = f"NEUTRAL: using defaults"

        # ── Performance-based adjustments (online learning) ──
        if loss_streak >= 4:
            # Consecutive losses: tighten everything regardless of regime
            overlay.bear_veto_threshold = min(overlay.bear_veto_threshold, 7)
            overlay.min_confidence = max(overlay.min_confidence, 0.70)
            overlay.risk_multiplier *= 0.6
            overlay.reasoning += f" | LOSS_STREAK={loss_streak}: defensive override"
        elif win_rate > 0.6 and stats['total_trades'] >= 10:
            # Proven edge: slightly loosen to capture more
            overlay.risk_multiplier = min(overlay.risk_multiplier * 1.1, 1.2)
            overlay.reasoning += f" | HIGH_WR={win_rate:.0%}: slight boost"
        elif win_rate < 0.35 and stats['total_trades'] >= 10:
            # Poor performance: tighten hard
            overlay.bear_veto_threshold = min(overlay.bear_veto_threshold, 7)
            overlay.min_confidence = max(overlay.min_confidence, 0.70)
            overlay.risk_multiplier *= 0.5
            overlay.reasoning += f" | LOW_WR={win_rate:.0%}: hard tighten"

        self.overlay_history[asset].append(overlay)
        # Keep last 100 overlays per asset
        if len(self.overlay_history[asset]) > 100:
            self.overlay_history[asset] = self.overlay_history[asset][-50:]

        return overlay

    def update_from_trade(self, asset: str, pnl_usd: float, pnl_pct: float):
        """Online learning: update running stats from each completed trade."""
        stats = self._running_stats[asset]
        stats['total_trades'] += 1

        if pnl_usd > 0:
            stats['win_streak'] += 1
            stats['loss_streak'] = 0
        else:
            stats['loss_streak'] += 1
            stats['win_streak'] = 0

        # EWMA update of win rate (online, no batch needed)
        win = 1.0 if pnl_usd > 0 else 0.0
        stats['win_rate'] = stats['win_rate'] * (1 - self._ewma_alpha) + win * self._ewma_alpha

        # EWMA update of avg PnL
        stats['avg_pnl'] = stats['avg_pnl'] * (1 - self._ewma_alpha) + pnl_pct * self._ewma_alpha

        self.performance_history[asset].append({
            'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct,
            'timestamp': datetime.now().isoformat(),
        })
        # Keep last 200 trades per asset
        if len(self.performance_history[asset]) > 200:
            self.performance_history[asset] = self.performance_history[asset][-100:]


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
        except Exception as e:
            logger.error(f"Error loading meta-model from {filepath}: {e}")

class MarketAnomalyDetector:
    """
    Detects extreme market outliers (Black Swans, Flash Crashes, Liquidity Sweeps)
    using statistical velocity and volume profile deviations.
    """
    def __init__(self, deviation_threshold: float = 3.5):
        self.deviation_threshold = deviation_threshold
        
    def detect_anomalies(self, close: np.ndarray, volume: np.ndarray) -> Dict[str, Any]:
        """Returns anomaly detection flags and scores."""
        if len(close) < 50:
            return {"is_anomaly": False, "type": "NONE", "severity": 0.0}
            
        recent_returns = np.diff(close[-50:]) / close[-50:-1]
        mean_ret = np.mean(recent_returns[:-1])
        std_ret = np.std(recent_returns[:-1])
        
        # Z-score of the most recent bar's return
        if std_ret > 0:
            z_score = abs(recent_returns[-1] - mean_ret) / std_ret
        else:
            z_score = 0.0
            
        avg_vol = np.mean(volume[-50:-1])
        vol_spike = volume[-1] / avg_vol if avg_vol > 0 else 1.0
        
        is_anomaly = z_score > self.deviation_threshold and vol_spike > 2.0
        
        anomaly_type = "NONE"
        if is_anomaly:
            if recent_returns[-1] < 0:
                anomaly_type = "FLASH_CRASH"
            else:
                anomaly_type = "LIQUIDITY_SWEEP_UP"
                
        return {
            "is_anomaly": is_anomaly,
            "type": anomaly_type,
            "severity": round(z_score * vol_spike, 2),
            "z_score": round(z_score, 2),
            "vol_spike": round(vol_spike, 2)
        }

class AlphaDecayTracker:
    """
    Tracks the 'half-life' of a generated strategy's edge.
    If a strategy's expected EV diverges too far from actual Out-of-Sample PnL,
    it forces a deprecation of that strategy hyperparameter set.
    """
    def __init__(self, decay_threshold: float = 0.6):
        self.decay_threshold = decay_threshold
        self.strategy_performance_history = defaultdict(list)
        
    def update_edge_retention(self, strategy_id: str, predicted_perf: float, actual_out_of_sample: float) -> float:
        """Calculate the Remaining Edge Ratio (0.0 to 1.0)"""
        if predicted_perf <= 0:
            return 1.0
            
        ratio = max(0.0, actual_out_of_sample / predicted_perf)
        self.strategy_performance_history[strategy_id].append(ratio)
        
        # EWMA of edge retention
        history = self.strategy_performance_history[strategy_id][-10:]
        weights = np.exp(np.linspace(-1., 0., len(history)))
        weights /= weights.sum()
        
        edge_remaining = np.sum(np.array(history) * weights)
        return float(edge_remaining)
        
    def requires_retirement(self, edge_remaining: float) -> bool:
        """Returns True if the strategy's alpha has decayed past the acceptable threshold."""
        return edge_remaining < self.decay_threshold


class AdvancedLearningEngine:
    """
    Runtime meta-optimizer orchestrator.
    Observes pipeline decisions + trade outcomes and dynamically adjusts
    SECONDARY pipeline parameters via PipelineOverlay.

    SAFE: v13/v14 core (min/max_entry_score, short_score_penalty, sr_assets)
    is NEVER touched. Only adjusts: bear thresholds, ML weights, confidence, risk_mult.

    Online training: learns from EVERY bar and EVERY trade — no batch needed.
    """

    def __init__(self, meta_model_path: str = "models/meta_learning_model.json"):
        self.pattern_recognizer = CrossMarketPatternRecognizer()
        self.regime_classifier = MarketRegimeClassifier()
        self.strategy_generator = AdaptiveStrategyGenerator()
        self.meta_learner = MetaLearningEngine()
        self.anomaly_detector = MarketAnomalyDetector()
        self.alpha_tracker = AlphaDecayTracker()

        self.active_overlays: Dict[str, PipelineOverlay] = {}
        self.performance_tracker: Dict[str, List[Dict]] = defaultdict(list)
        self.edge_retention: Dict[str, float] = {}
        self.meta_model_path = meta_model_path

        # Auto-save interval (save learned meta-model every N bars)
        self._bars_since_save: int = 0
        self._save_interval: int = 360  # ~30 min at 5s poll, ~3h at 30s poll

        # Try loading existing meta-model
        self.meta_learner.load_meta_model(self.meta_model_path)

    # ------------------------------------------------------------------
    # ONLINE: called every bar from main loop
    # ------------------------------------------------------------------
    def process_bar(self, asset: str, close: np.ndarray, high: np.ndarray,
                    low: np.ndarray, volume: np.ndarray,
                    onchain_signals: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Per-bar online processing. Called from executor main loop.
        Returns: {overlay, anomaly, regime, edge_ratio, patterns}
        """
        result = {'asset': asset, 'anomaly': None, 'overlay': None,
                  'regime': None, 'edge_ratio': 1.0}

        if len(close) < 50:
            return result

        # 1. Anomaly detection (flash crash / liquidity sweep)
        anomaly = self.anomaly_detector.detect_anomalies(close, volume)
        result['anomaly'] = anomaly

        # 2. Regime classification
        regime = self.regime_classifier.classify_regime(
            close, high, low, volume, onchain_signals)
        result['regime'] = regime

        # 3. Generate pipeline overlay (SECONDARY knobs only)
        recent_perf = {
            'win_rate': self.strategy_generator._running_stats[asset]['win_rate'],
            'avg_pnl': self.strategy_generator._running_stats[asset]['avg_pnl'],
        }

        # If anomaly, force ANOMALY regime
        if anomaly and anomaly.get('is_anomaly'):
            regime = MarketRegime("ANOMALY", 100.0, 1.0, 0.0, 1, "HOLD")
            result['regime'] = regime
            logger.warning(f"[ANOMALY] {asset}: {anomaly['type']} severity={anomaly['severity']}")

        overlay = self.strategy_generator.generate_overlay(regime, asset, recent_perf)
        self.active_overlays[asset] = overlay
        result['overlay'] = overlay

        # 4. Alpha decay check
        edge_ratio = self.edge_retention.get(asset, 1.0)
        result['edge_ratio'] = edge_ratio
        if edge_ratio < self.alpha_tracker.decay_threshold:
            logger.warning(f"[ALPHA DECAY] {asset} edge degraded ({edge_ratio:.2f}) — tightening overlay")
            overlay.risk_multiplier *= 0.5
            overlay.min_confidence = max(overlay.min_confidence, 0.70)
            overlay.reasoning += f" | ALPHA_DECAY={edge_ratio:.2f}"

        # 5. Auto-save meta-model periodically
        self._bars_since_save += 1
        if self._bars_since_save >= self._save_interval:
            self._bars_since_save = 0
            try:
                self.save_learned_models()
            except Exception:
                pass

        return result

    # ------------------------------------------------------------------
    # ONLINE: called after every trade close
    # ------------------------------------------------------------------
    def on_trade_close(self, asset: str, pnl_usd: float, pnl_pct: float,
                       predicted_l_level: int = 0, actual_l_level: int = 0):
        """
        Online learning from each completed trade.
        Updates strategy generator, alpha tracker, and meta-learner.
        """
        # 1. Update strategy generator (online EWMA stats)
        self.strategy_generator.update_from_trade(asset, pnl_usd, pnl_pct)

        # 2. Track alpha decay (how well our predicted L-level matches actual)
        predicted_perf = max(0.1, predicted_l_level)  # predicted L-level as proxy for expected perf
        actual_perf = max(0.0, actual_l_level)
        edge = self.alpha_tracker.update_edge_retention(
            f"ema8_{asset}", predicted_perf, actual_perf)
        self.edge_retention[asset] = edge

        # 3. Store performance for meta-learner
        self.performance_tracker[asset].append({
            'pnl_usd': pnl_usd, 'pnl_pct': pnl_pct,
            'predicted_l': predicted_l_level, 'actual_l': actual_l_level,
            'edge_remaining': edge,
            'timestamp': datetime.now().isoformat(),
        })
        if len(self.performance_tracker[asset]) > 500:
            self.performance_tracker[asset] = self.performance_tracker[asset][-250:]

        logger.info(f"[META-LEARN] {asset}: pnl=${pnl_usd:+.2f} edge={edge:.2f} "
                     f"WR={self.strategy_generator._running_stats[asset]['win_rate']:.0%}")

    # ------------------------------------------------------------------
    # Full multi-asset analysis (periodic, not every bar)
    # ------------------------------------------------------------------
    def process_market_data(self, multi_asset_data: Dict[str, pd.DataFrame],
                            onchain_data: Optional[Dict[str, Dict]] = None) -> Dict[str, Any]:
        """
        Full cross-asset analysis. Called periodically (every 30-60 min).
        Computes correlations, patterns, and updates meta-learner across assets.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "regimes": {}, "overlays": {}, "patterns": {}, "correlations": {}
        }

        # Cross-market patterns + correlations
        if len(multi_asset_data) >= 2:
            try:
                patterns = self.pattern_recognizer.recognize_patterns(multi_asset_data)
                result["patterns"] = patterns
                correlations = self.pattern_recognizer.compute_cross_market_correlation(multi_asset_data)
                result["correlations"] = correlations
            except Exception as e:
                logger.debug(f"Pattern/correlation error: {e}")

        # Per-asset regime + overlay
        for asset, data in multi_asset_data.items():
            try:
                close = data['close'].values
                high = data['high'].values
                low = data['low'].values
                volume = data['volume'].values

                onchain_signals = {}
                if onchain_data and asset in onchain_data:
                    onchain_signals = onchain_data[asset]

                bar_result = self.process_bar(asset, close, high, low, volume, onchain_signals)
                result["regimes"][asset] = bar_result.get('regime')
                result["overlays"][asset] = bar_result.get('overlay')
            except Exception as e:
                logger.debug(f"Asset {asset} processing error: {e}")

        return result

    def save_learned_models(self):
        """Persist all learned meta-models."""
        self.meta_learner.save_meta_model(self.meta_model_path)
        logger.info("[META-LEARN] Models saved to %s", self.meta_model_path)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the meta-learning system."""
        return {
            "active_overlays": {k: v.to_dict() for k, v in self.active_overlays.items()},
            "edge_retention": dict(self.edge_retention),
            "running_stats": dict(self.strategy_generator._running_stats),
            "patterns_discovered": len(self.pattern_recognizer.pattern_memory),
            "meta_model_assets": len(self.meta_learner.market_to_strategy_map),
            "bars_since_save": self._bars_since_save,
            "timestamp": datetime.now().isoformat(),
        }
