"""
Hybrid Trading Strategy -- FinBERT + LightGBM Ensemble
========================================================
Three-Layer Hybrid Signal Architecture with ML integration:

  L1 (50%): LightGBM 3-class classifier consuming 80+ features
            + FinBERT sentiment features fused into feature vector
  L2 (30%): FinBERT sentiment scoring with time-decay aggregation
  L3 (20% + VETO): Risk Engine with 5-tier stop system

Signal Fusion:
  - LightGBM predicts direction (Long/Flat/Short) with confidence
  - FinBERT amplifies or dampens signals by up to 30%
  - Directional disagreement halves position or cancels trade
  - Confidence gating: only signals with conf > 0.65 pass
"""

import time
from typing import List, Dict, Optional

from src.models.lightgbm_classifier import LightGBMClassifier
from src.models.rl_agent import RLAgent
from src.ai.finbert_service import FinBERTService
from src.ai.sentiment import SentimentPipeline
from src.risk.manager import RiskManager
from src.trading.meta_controller import MetaController
from src.trading.adaptive_engine import AdaptiveEngine
from src.trading.signal_combiner import SignalCombiner
from src.models.volatility import VolRegime


class HybridStrategy:
    """
    FinBERT + LightGBM Hybrid Ensemble Strategy.

    Pipeline:
      1. FinBERT scores news headlines -> sentiment features
      2. LightGBM extracts 80+ technical features + sentiment features
      3. LightGBM classifies: Long/Flat/Short with confidence
      4. Signal fusion checks directional agreement
      5. L3 risk engine evaluates with 5-tier stops + VETO
      6. Final signal with position sizing and stops
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # L3: LightGBM classifier (v5.5 core)
        lgb_config = cfg.get('l1', {})
        # Merge in model path from models section if available
        if 'models' in cfg and 'lightgbm' in cfg['models']:
            lgb_config = {**lgb_config, **cfg['models']['lightgbm']}
        self.classifier = LightGBMClassifier(lgb_config)

        # L3: RL Agent (v6.0 core)
        self.rl_agent = RLAgent(cfg.get('rl', {}))
        # RL experience buffer (state, action, reward, next_state)
        self._rl_experience: List[Tuple[Dict[str,float], int, float, Dict[str,float]]] = []

        # L4: Meta-Controller
        self.meta_controller = MetaController(cfg.get('meta', {}))
        
        # New L5: Adaptive Strategy Engine
        self.adaptive_engine = AdaptiveEngine(cfg.get('adaptive', {}))

        # L2: FinBERT service
        ai_cfg = cfg.get('ai', {})
        self.finbert = FinBERTService(
            model_name=ai_cfg.get('finbert_model', 'finbert'),
            device=ai_cfg.get('device', 'cpu'),
        )

        # Legacy sentiment (for backward compat + fallback)
        self.sentiment = SentimentPipeline(
            use_transformer=ai_cfg.get('use_transformer', False),
            decay_gamma=cfg.get('decay_gamma', 0.001),
        )

        # L3: Risk engine
        risk_cfg = cfg.get('risk', {})
        self.risk_manager = RiskManager(
            max_position_pct=risk_cfg.get('max_position_size_pct', 2.0),
            max_portfolio_pct=risk_cfg.get('max_portfolio_pct', 20.0),
            daily_loss_limit_pct=risk_cfg.get('daily_loss_limit_pct', 3.0),
            max_drawdown_pct=risk_cfg.get('max_drawdown_pct', 10.0),
            atr_stop_mult=risk_cfg.get('atr_stop_mult', 2.0),
            atr_tp_mult=risk_cfg.get('atr_tp_mult', 3.0),
        )

        # Signal combiner
        self.combiner = SignalCombiner(cfg.get('combiner', {}))

        # track trades for feedback learning
        self._trade_history: List[Dict] = []

    def generate_signals(self, prices: List[float],
                          highs: Optional[List[float]] = None,
                          lows: Optional[List[float]] = None,
                          volumes: Optional[List[float]] = None,
                          headlines: Optional[List[str]] = None,
                          headline_timestamps: Optional[List[float]] = None,
                          headline_sources: Optional[List[str]] = None,
                          headline_event_types: Optional[List[str]] = None,
                          external_features: Optional[Dict[str, float]] = None,
                          account_balance: float = 100_000.0,
                          ) -> Dict:
        """
        Run the full FinBERT + LightGBM hybrid pipeline.

        Returns:
          - 'signals':           List[int]   -- discrete trade signals
          - 'forecast_signal':   List[float] -- continuous LightGBM forecast
          - 'l1_data':           dict        -- all L1 indicators + features
          - 'l2_data':           dict        -- sentiment aggregate
          - 'final_decisions':   List[Dict]  -- per-bar combined decisions
        """
        n = len(prices)
        if highs is None:
            highs = prices
        if lows is None:
            lows = prices
        if volumes is None:
            volumes = [1.0] * n

        # ================================================================
        # STEP 1: FinBERT Sentiment Scoring (L2)
        # ================================================================
        sentiment_features = self.finbert.get_sentiment_features(
            headlines or []
        )

        # Also compute aggregate via legacy pipeline for combiner
        if headlines:
            legacy_sentiments = self.sentiment.analyze(
                headlines,
                timestamps=headline_timestamps,
                sources=headline_sources,
                event_types=headline_event_types,
            )
            l2_aggregate = self.sentiment.aggregate_sentiment(
                legacy_sentiments,
                timestamps=headline_timestamps,
            )
            l2_timestamp = time.time()
        else:
            l2_aggregate = {
                'aggregate_score': 0.0, 'aggregate_label': 'NEUTRAL',
                'confidence': 0.0, 'num_sources': 0, 'freshness': 0.0,
            }
            l2_timestamp = None

        # ================================================================
        # STEP 2: LightGBM Feature Extraction + Classification (L1)
        # ================================================================
        features = self.classifier.extract_features(
            closes=prices, highs=highs, lows=lows, volumes=volumes,
            sentiment_features=sentiment_features,
            external_features=external_features,
        )

        # Get LightGBM predictions (class + confidence)
        lgb_predictions = self.classifier.predict(features)

        # Get RL Agent predictions (action + probability)
        rl_predictions = self.rl_agent.predict(features)

        # Continuous forecast signal for display
        forecast_sig = self.classifier.forecast_signal(features)

        # store last features for potential trade logging
        self._last_features = features[-1] if features else {}

        # Build L1 data dict with indicator values from last feature
        l1_data = self._build_l1_data(features, lgb_predictions, prices, highs, lows, volumes)
        l1_data['forecast_signal'] = forecast_sig

        # ================================================================
        # STEP 3: Signal Fusion with Directional Disagreement Check
        # ================================================================
        from src.indicators.indicators import atr as compute_atr
        atr_vals = compute_atr(highs, lows, prices)

        final_signals: List[int] = []
        final_decisions: List[Dict] = []

        for i in range(n):
            lgb_class, lgb_conf = lgb_predictions[i] if i < len(lgb_predictions) else (0, 0.0)
            rl_action, rl_prob = rl_predictions[i] if i < len(rl_predictions) else (0, 0.0)

            # Extract features for L4 Arbitrator
            f_current = features[i] if i < len(features) else {}

            # Use new purely L4 Arbitrator logic 
            final_class, final_conf, position_scale = self.meta_controller.arbitrate(
                lgb_class=lgb_class,
                lgb_conf=lgb_conf,
                rl_action=rl_action,
                rl_prob=rl_prob,
                features=f_current,
                finbert_score=sentiment_features.get('sentiment_mean', 0.0)
            )

            # ================================================================
            # NEW STEP: Adaptive Strategy Selection
            # ================================================================
            sent_val = sentiment_features.get('sentiment_mean', 0.0)
            adaptive_package = self.adaptive_engine.generate_adaptive_signal(
                prices=prices[:i+1], highs=highs[:i+1], lows=lows[:i+1], volumes=volumes[:i+1],
                sentiment_score=sent_val
            )
            adaptive_signal = adaptive_package['signal']
            strat_chosen = adaptive_package['strategy_selected']

            # Fuse adaptive tactical signal with ML ensemble (70% ML, 30% Tactical)
            l1_signal = final_class * final_conf
            l1_signal = (l1_signal * 0.7) + (adaptive_signal * 0.3)
            
            # L3 risk evaluation
            current_atr = atr_vals[i] if i < len(atr_vals) else prices[i] * 0.02
            vol_regime = VolRegime.MEDIUM
            if features and i < len(features) and features[i]:
                vr = features[i].get('vol_regime_encoded', 1)
                vol_regime = [VolRegime.LOW, VolRegime.MEDIUM, VolRegime.HIGH, VolRegime.EXTREME][min(int(vr), 3)]

            if final_class != 0 and position_scale > 0:
                from src.risk.manager import RiskAction
                l3_eval = self.risk_manager.evaluate_trade(
                    asset='COMPOSITE',
                    direction=final_class,
                    proposed_size=account_balance * 0.02 / max(prices[i], 1) * position_scale,
                    account_balance=account_balance,
                    current_price=prices[i],
                    atr_value=current_atr,
                    vol_regime=vol_regime,
                    composite_signal=l1_signal,
                )
            else:
                from src.risk.manager import RiskAction
                l3_eval = {
                    'action': RiskAction.ALLOW, 'adjusted_size': 0.0,
                    'stop_loss': 0.0, 'take_profit': 0.0,
                    'reason': 'No signal', 'risk_score': 0.0,
                }

            # Combine L1 + L2 + L3
            decision = self.combiner.combine(
                l1_signal=l1_signal,
                l2_sentiment=l2_aggregate,
                l3_evaluation=l3_eval,
                l2_timestamp=l2_timestamp,
            )

            # Ensure scale is correctly zeroed out in actions if meta-controller vetoes
            if position_scale == 0.0:
                decision['action'] = 'hold'
                decision['final_signal'] = 0.0
            elif position_scale < 1.0:
                decision['position_size'] *= position_scale

            # Map to discrete signal
            action = decision.get('action', 'hold')
            if action == 'buy':
                final_signals.append(1)
            elif action == 'sell':
                final_signals.append(-1)
            else:
                final_signals.append(0)

            final_decisions.append(decision)

        result = {
            'signals': final_signals,
            'forecast_signal': forecast_sig,
            'composite': [fs for fs in forecast_sig],
            'l1_data': l1_data,
            'l2_data': l2_aggregate,
            'sentiment_features': sentiment_features,
            'final_decisions': final_decisions,
            'risk_stats': self.risk_manager.get_performance_stats(),
            'features': features,
            'adaptive_info': adaptive_package if 'adaptive_package' in locals() else {} # type: ignore
        }
        return result

    # -------------------------------------------------------------
    # Feedback / learning helpers
    # -------------------------------------------------------------
    def record_backtest(self, bt_result: 'BacktestResult') -> None:
        """Log trades from a backtest for incremental learning."""
        for trade in bt_result.trades:
            feats = getattr(trade, 'entry_features', {}) or {}
            self.classifier.log_trade(
                feats, 
                trade.direction, 
                trade.net_pnl,
                getattr(trade, 'entry_price', 0.0),
                getattr(trade, 'exit_price', 0.0),
                getattr(trade, 'bars_held', 1)
            )
            
            # Learn from adaptive strategy if it was recorded
            strat_used = getattr(trade, 'strategy_selected', None)
            if strat_used:
                self.adaptive_engine.update_learning(strat_used, trade.net_pnl)

            # RL: record a simple tuple (state, action, reward, next_state)
            self._rl_experience.append((feats, trade.direction, trade.net_pnl, {}))

    def retrain_models(self) -> None:
        """Trigger retraining for L1/RL models based on logged experience."""
        self.classifier.retrain_from_log(max_examples=2000)
        if hasattr(self.rl_agent, 'train_from_experience'):
            self.rl_agent.train_from_experience(self._rl_experience)

    def _build_l1_data(self, features, predictions, prices, highs, lows, volumes):
        """Build L1 data dict from features for executor display."""
        from src.indicators.indicators import sma, rsi, macd, bollinger_bands
        from src.models.volatility import ewma_volatility, classify_volatility_regime
        from src.models.cycle_detector import detect_dominant_cycles, detect_cycle_phase

        n = len(prices)
        # Quick indicator computation for display
        short_ma = sma(prices, 10)
        long_ma = sma(prices, 50)
        rsi_vals = rsi(prices, 14)
        _, _, macd_hist = macd(prices)
        ewma_vol = ewma_volatility(prices)
        vol_regimes = classify_volatility_regime(ewma_vol)
        cycles = detect_dominant_cycles(prices)
        phase = detect_cycle_phase(prices)

        # Z-score from features
        zscore_vals = [f.get('zscore_20', 0.0) if f else 0.0 for f in features]

        # BB bands
        bb_up = [f.get('price_vs_bb_upper', 0.0) if f else 0.0 for f in features]
        bb_lo = [f.get('price_vs_bb_lower', 0.0) if f else 0.0 for f in features]

        dom_period = cycles[0][0] if cycles else 30
        from src.models.cycle_detector import adaptive_holding_period
        holding = adaptive_holding_period(dom_period, phase)

        return {
            'short_ma': short_ma,
            'long_ma': long_ma,
            'rsi': rsi_vals,
            'macd_hist': macd_hist,
            'zscore': zscore_vals,
            'bb_upper': bb_up,
            'bb_lower': bb_lo,
            'ewma_vol': ewma_vol,
            'garch_vol': [f.get('garch_vol', 0.0) if f else 0.0 for f in features],
            'vol_regimes': vol_regimes,
            'cycle_phase': phase,
            'dominant_cycles': cycles,
            'holding_period': holding,
            'predictions': predictions,
        }


class SimpleStrategy:
    """
    Backward-compatible simple strategy.
    Wraps the LightGBM classifier for basic signal generation.
    """

    def __init__(self, short: int = 5, long: int = 20, z_window: int = 20):
        self.classifier = LightGBMClassifier({
            'sma_short': short,
            'sma_long': long,
            'z_window': z_window,
        })

    def generate_signals(self, prices: List[float]) -> Dict[str, List]:
        """Backward-compatible interface."""
        from src.indicators.indicators import sma
        from src.models.numerical_models import zscore

        features = self.classifier.extract_features(prices)
        predictions = self.classifier.predict(features)
        signals = [cls for cls, _ in predictions]

        return {
            'signals': signals,
            'short_ma': sma(prices, 10),
            'long_ma': sma(prices, 50),
            'zscore': zscore(prices, 20),
        }
