"""
Hybrid Strategy -- FinBERT (L2) + LightGBM (L1) Fusion
======================================================
The high-fidelity brain of the system.
v1.0: Combines Institutional Sentiment with Quantitative Boosted Trees.
"""

import time
from typing import List, Dict, Optional, Tuple

from src.models.lightgbm_classifier import LightGBMClassifier
from src.models.rl_agent import RLAgent
from src.ai.finbert_service import FinBERTService
from src.ai.sentiment import SentimentPipeline
from src.ai.patchtst_model import PatchTSTClassifier
from src.risk.manager import RiskManager
from src.trading.meta_controller import MetaController
from src.trading.adaptive_engine import AdaptiveEngine
from src.trading.signal_combiner import SignalCombiner
from src.models.volatility import EWMAVolatility


class HybridStrategy:
    """
    Orchestrates the multi-layered signal generation.
    - Tier 1: LightGBM (Quant Classification)
    - Tier 2: FinBERT (Institutional Sentiment)
    - Tier 3: Adaptive Engine (Regime Filtering)
    - Tier 4: Meta-Controller (Weighted Arbitration + Veto)
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}

        # L1: LightGBM classifier (v5.5 core)
        lgb_config = cfg.get('l1', {})
        if 'models' in cfg and 'lightgbm' in cfg['models']:
            lgb_config = {**lgb_config, **cfg['models']['lightgbm']}
        self.classifier = LightGBMClassifier(lgb_config)

        # L1: RL Agent (v6.0 core)
        self.rl_agent = RLAgent(cfg.get('rl', {}))

        # L2: FinBERT Sentiment
        self.finbert = FinBERTService(device='cpu') # FinBERT is L2 institutional layer

        # Legacy sentiment layer for backward compatibility
        self.sentiment = SentimentPipeline(
            use_transformer=cfg.get('ai', {}).get('use_transformer', False)
        )

        # L3: Risk and Regime
        self.risk_manager = RiskManager(**cfg.get('risk', {}))
        self.adaptive_engine = AdaptiveEngine(cfg.get('adaptive', {}))
        self.volatility_model = EWMAVolatility()

        # L4: Meta-Controller (The Judge)
        self.meta_controller = MetaController(cfg.get('meta', {}))

        # Signal combiner
        self.combiner = SignalCombiner(cfg.get('combiner', {}))
        
        # L7: PatchTST Transformer
        self.patch_tst = PatchTSTClassifier(cfg.get('models', {}).get('patchtst_path', 'models/patchtst_v1.pt'))

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
                          agentic_bias: float = 0.0,
                          account_balance: float = 100_000.0,
                          ) -> Dict:
        """
        Run the full FinBERT + LightGBM hybrid pipeline.
        """
        n = len(prices)
        if highs is None: highs = prices
        if lows is None: lows = prices
        if volumes is None: volumes = [1.0] * n

        # STEP 1: FinBERT Sentiment Scoring (L2)
        sentiment_features = self.finbert.get_sentiment_features(headlines or [])

        # Legacy aggregate
        if headlines:
            legacy_sentiments = self.sentiment.analyze(
                headlines, timestamps=headline_timestamps,
                sources=headline_sources, event_types=headline_event_types
            )
            l2_aggregate = self.sentiment.aggregate_sentiment(legacy_sentiments, timestamps=headline_timestamps)
        else:
            l2_aggregate = {'aggregate_score': 0.0, 'aggregate_label': 'NEUTRAL', 'confidence': 0.0}

        # STEP 2: LightGBM Feature Extraction + Classification (L1)
        features = self.classifier.extract_features(
            closes=prices, highs=highs, lows=lows, volumes=volumes,
            sentiment_features=sentiment_features,
            external_features=external_features,
        )

        lgb_predictions = self.classifier.predict(features)
        rl_predictions = self.rl_agent.predict(features)
        forecast_sig = self.classifier.forecast_signal(features)
        
        # PatchTST Inference (Deep Layer 7)
        import numpy as np
        patch_result = self.patch_tst.predict(np.array(prices))
        
        self._last_features = features[-1] if features else {}
        
        # STEP 3: Signal Fusion (L4)
        from src.indicators.indicators import atr as compute_atr
        atr_vals = compute_atr(highs, lows, prices)

        final_signals: List[int] = []
        final_decisions: List[Dict] = []

        for i in range(n):
            lgb_c, lgb_p = lgb_predictions[i]
            rl_a, rl_p = rl_predictions[i]

            f = features[i]
            
            # Meta-Arbitration
            final_dir, final_conf, p_scale = self.meta_controller.arbitrate(
                lgb_class=lgb_c, lgb_conf=float(lgb_p),
                rl_action=rl_a, rl_prob=float(rl_p),
                features=f,
                finbert_score=l2_aggregate.get('aggregate_score', 0.0),
                patch_result=patch_result,
                agentic_bias=agentic_bias
            )

            # Adaptive Strategy Selection
            strategy_name = self.adaptive_engine.select_strategy(f, l2_aggregate)
            
            # Risk Gate
            is_safe, risk_reason = self.risk_manager.is_trade_safe(
                prices[i], final_dir, atr_vals[i], account_balance
            )
            
            if not is_safe:
                final_dir = 0
                p_scale = 0.0

            final_signals.append(final_dir)
            final_decisions.append({
                'direction': final_dir, 'confidence': final_conf,
                'scale': p_scale, 'strategy': strategy_name,
                'reason': risk_reason if not is_safe else "AI Confirmed"
            })

        return {
            'signals': final_signals,
            'forecast_signal': forecast_sig,
            'l1_data': {'features': features, 'predictions': lgb_predictions},
            'l2_data': l2_aggregate,
            'final_decisions': final_decisions
        }

    def record_backtest(self, result: 'BacktestResult', asset: str, l1_data: Dict):
        """Log backtest history for L1 retraining."""
        trades = result.trades
        if not trades: return

        for t in trades:
            idx = getattr(t, 'exit_idx', -1)
            if idx != -1 and idx < len(l1_data['features']):
                feats = l1_data['features'][idx]
                label = 1 if t.net_pnl > 0 else 2
                self.classifier.log_trade(
                    features=feats, label=label,
                    entry_price=t.entry_price,
                    exit_price=t.exit_price,
                    bars_held=t.holding_bars
                )

        # Trigger periodic retraining
        if len(self.classifier._trade_log) >= 50:
            self.classifier.retrain_from_log()


class SimpleStrategy:
    """Legacy/Simple strategy placeholder."""
    def __init__(self, config=None):
        pass

    def generate_signals(self, **kwargs):
        return {'signals': [], 'l1_data': {}, 'final_decisions': []}
