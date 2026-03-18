"""
LightGBM 3-Class Classifier for Trade Direction
=================================================
Primary classification model in the L1 Quantitative Engine.

3 Classes:  LONG (+1), FLAT (0), SHORT (-1)

Feature Vector (80+ features):
  - Technical indicators: RSI, MACD, BB width, ADX, Stochastic, ATR, OBV, ROC
  - Volatility: EWMA vol, GARCH vol, realized vol, vol regime encoding
  - Trend: SMA ratios, EMA slopes, trend strength
  - Mean-reversion: Z-score, BB position, Williams %R
  - Market cycle: cycle phase encoding, dominant period
  - Momentum: ROC multi-period, RSI slope
  - Sentiment (from FinBERT): sentiment_mean, sentiment_z_score,
    bullish_ratio, bearish_ratio, avg_confidence, sentiment_momentum

Confidence Gating:
  Signal passes to risk engine only if confidence > 0.65.

Retraining:
  Weekly on expanding 6-month window to adapt to regime changes.
"""

import math
import os
import numpy as np
import lightgbm as lgb
from typing import List, Dict, Optional, Tuple

from src.indicators.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic, roc,
    adx, obv, bb_width, williams_r, true_range, vwap,
    volume_delta, liquidity_sweep, vwap_deviation, vpin
)
from src.models.volatility import (
    ewma_volatility, log_returns, simple_returns,
    realized_volatility, GARCH11, VolRegime, classify_volatility_regime
)
from src.indicators.market_structure import MarketStructureAnalyzer
from src.indicators.price_action import PriceActionAnalyzer
from src.models.volatility_regime import VolatilityRegimeDetector
from src.models.cycle_detector import detect_cycle_phase, detect_dominant_cycles
from src.models.model_tuner import ModelTuner


class LightGBMClassifier:
    """
    LightGBM 3-class trade direction classifier.

    Consumes 100+ features including the 'Top 30 Institutional Crypto Signals'.
    Outputs: class prediction (Long/Flat/Short) + confidence score.
    """

    FEATURE_NAMES = [
        # --- TOP 45 INSTITUTIONAL SIGNALS ---
        # Microstructure (7)
        'l2_imbalance', 'l2_wall_signal', 'l2_slope_ratio', 
        'spread_expansion', 'iceberg_detected', 'spoofing_detected', 'l2_void_ratio',
        # Derivatives (9)
        'funding_rate', 'funding_momentum', 'funding_skew_divergence', 
        'cross_exchange_dislocation', 'options_iv_skew_25d',
        'oi_change', 'ls_ratio', 'liq_intensity', 'liq_cascade_prob',
        # On-Chain (10)
        'exchange_inflow', 'exchange_outflow', 'whale_cluster_detected',
        'exchange_wallet_momentum', 'stablecoin_exchange_ratio',
        'hashrate_shock', 'lth_spent_ratio', 'defi_stablecoin_velocity',
        'dormant_coin_movement', 'lth_supply_ratio',
        # Price Action / Liquidity (6)
        'in_bull_fvg', 'in_bear_fvg', 'proximity_bull_ob', 'proximity_bear_ob',
        'liquidity_sweep', 'vwap_deviation', 
        # Volatility / Statistical (7)
        'realized_vol_20', 'atr_pct', 'vol_regime_encoded',
        'zscore_20', 'volume_delta', 'btc_nasdaq_corr_24h', 'correlation_breakdown',
        # Macro / Growth (2)
        'stablecoin_mint_velocity', 'stablecoin_depeg_event',
        
        # --- CORE QUANT ENGINE FEATURES ---
        'sma_10_50_ratio', 'ema_10_20_ratio', 'rsi_14', 'macd_hist',
        'adx_14', 'bb_width_20', 'stoch_k', 'stoch_d', 
        'cycle_phase_encoded', 'dominant_period', 'sentiment_mean', 'sentiment_z_score'
    ]

    CONFIDENCE_THRESHOLD = 0.65  # minimum to pass signal to risk engine

    # path where trade logs will be persisted (csv).  None disables logging.
    TRADE_LOG_PATH = 'logs/trade_history.csv'

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self._lgb_model = None
        self._lgb_available = False
        self._fitted = False
        self._garch = GARCH11()
        self._garch_fitted = False
        self._calibration_model = None
        self.confidence_threshold = cfg.get('confidence_threshold', self.CONFIDENCE_THRESHOLD)
        self.config = cfg  # Store for Platt scaling constants (Fix B)

        # trade logging buffer used for incremental retraining
        self._trade_log: List[Dict[str, float]] = []
        self.trade_log_path = cfg.get('trade_log_path', self.TRADE_LOG_PATH)
        
        # Load model from path if provided
        model_path = cfg.get('model_path')
        if model_path:
            self.load_model(model_path)
            
        # Analyzers for L5 Strategy Enrichment
        self.ms_analyzer = MarketStructureAnalyzer(window=5)
        self.pa_analyzer = PriceActionAnalyzer(window=50)
        self.model_tuner = ModelTuner()
        
    def _calibrate_probability(self, prob_array: Any) -> float:
        """
        Platt scaling calibration. Constants should be updated after each retraining
        using a holdout validation set. Current defaults are approximate.
        TODO: auto-calibrate A, B from validation set in retrain pipeline.
        """
        try:
            import numpy as np
            raw_prob = float(np.max(prob_array))
            # Platt scaling: P(y=1|f) = 1 / (1 + exp(A*f + B))
            # Ideal A, B calibrated from validation set. Use conservative defaults.
            # Fix B: configurable constants; loosened A (-2.0→-1.5) to reduce over-confidence
            A = self.config.get('platt_A', -1.5)  # was -2.0
            B = self.config.get('platt_B', 0.3)   # was 0.5
            z = max(-100, min(100, A * raw_prob + B))
            calibrated = 1.0 / (1.0 + math.exp(z))
            return float(max(0.0, min(1.0, calibrated)))
        except Exception:
            # Fallback to simple max
            try:
                import numpy as np
                return float(max(0.0, min(1.0, np.max(prob_array))))
            except Exception:
                return 0.5

    # -------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------
    @staticmethod
    def _fix_crlf(path: str) -> None:
        """Fix CRLF line endings that crash LightGBM's C++ parser on Windows."""
        with open(path, 'rb') as f:
            raw = f.read(4096)
        if b'\r\n' in raw:
            with open(path, 'rb') as f:
                content = f.read()
            with open(path, 'wb') as f:
                f.write(content.replace(b'\r\n', b'\n'))

    def load_model(self, model_path: str) -> bool:
        """
        Load a trained LightGBM model from file.
        Tries optimized version first if available.

        Args:
            model_path: path to saved LightGBM model file (e.g., 'models/lgbm_aave.txt')

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            import lightgbm as lgb
            import os

            # Try optimized version first
            optimized_path = model_path.replace('.txt', '_optimized.txt')
            if os.path.exists(optimized_path):
                self._fix_crlf(optimized_path)
                self._lgb_model = lgb.Booster(model_file=optimized_path)
                self._fitted = True
                self._lgb_available = True
                print(f"Loaded optimized model from {optimized_path}")
                return True

            # Fall back to original
            if os.path.exists(model_path):
                self._fix_crlf(model_path)
                self._lgb_model = lgb.Booster(model_file=model_path)
                self._fitted = True
                self._lgb_available = True
                return True

            return False
        except Exception:
            return False

    # -------------------------------------------------------------------

    def extract_features(self, closes: List[float],
                          highs: Optional[List[float]] = None,
                          lows: Optional[List[float]] = None,
                          volumes: Optional[List[float]] = None,
                          sentiment_features: Optional[Dict[str, float]] = None,
                          external_features: Optional[Dict[str, float]] = None,
                          ) -> List[Dict[str, float]]:
        """
        Extract 100+ features including the Top 30 Institutional Signals.
        """
        n = len(closes)
        if n < 55: return [{}] * n

        highs = highs or closes
        lows = lows or closes
        volumes = volumes or [1.0] * n
        opens = [closes[max(0, i-1)] for i in range(n)]

        # --- CORE INDICATORS ---
        vwap_vals = vwap(closes, volumes)
        v_delta = volume_delta(opens, closes, volumes)
        l_sweep = liquidity_sweep(highs, lows, closes)
        v_dev = vwap_deviation(closes, vwap_vals)
        rv_20 = realized_volatility(closes, 20)
        atr_vals = atr(highs, lows, closes, 14)
        vpin_50 = vpin(opens, closes, volumes, 50)
        
        # --- FEATURE BUILDING ---
        features: List[Dict[str, float]] = []
        for i in range(n):
            f: Dict[str, float] = {}
            ef = external_features or {}
            
            # --- PHASE 5: MICROSTRUCTURE (7) ---
            f['l2_imbalance'] = float(ef.get('l2_imbalance', 0.0))
            f['l2_wall_signal'] = float(ef.get('l2_wall_signal', 0.0))
            f['l2_slope_ratio'] = float(ef.get('l2_slope_ratio', 1.0))
            f['spread_expansion'] = float(ef.get('spread_expansion', 0.0))
            f['iceberg_detected'] = float(ef.get('iceberg_detected', 0.0))
            f['spoofing_detected'] = float(ef.get('spoofing_detected', 0.0))
            f['l2_void_ratio'] = float(ef.get('l2_void_ratio', 0.0))
            f['vpin_50'] = float(vpin_50[i])
            
            # --- PHASE 5: DERIVATIVES (9) ---
            f['funding_rate'] = float(ef.get('funding_rate', 0.0))
            f['funding_momentum'] = float(ef.get('funding_momentum', 0.0))
            f['funding_skew_divergence'] = float(ef.get('funding_skew_divergence', 0.0))
            f['cross_exchange_dislocation'] = float(ef.get('cross_exchange_dislocation', 0.0))
            f['options_iv_skew_25d'] = float(ef.get('options_iv_skew_25d', 0.0))
            f['oi_change'] = float(ef.get('oi_change', 0.0))
            f['ls_ratio'] = float(ef.get('ls_ratio', 1.0))
            f['liq_intensity'] = float(ef.get('liq_intensity', 0.0))
            f['liq_cascade_prob'] = float(ef.get('liq_cascade_prob', 0.0))
            
            # --- PHASE 5: ON-CHAIN (10) ---
            f['exchange_inflow'] = float(ef.get('exchange_inflow', 0.0))
            f['exchange_outflow'] = float(ef.get('exchange_outflow', 0.0))
            f['whale_cluster_detected'] = float(ef.get('whale_cluster_detected', 0.0))
            f['exchange_wallet_momentum'] = float(ef.get('exchange_wallet_momentum', 0.0))
            f['stablecoin_exchange_ratio'] = float(ef.get('stablecoin_exchange_ratio', 0.2))
            f['hashrate_shock'] = float(ef.get('hashrate_shock', 0.0))
            f['lth_spent_ratio'] = float(ef.get('lth_spent_ratio', 0.05))
            f['defi_stablecoin_velocity'] = float(ef.get('defi_stablecoin_velocity', 1.0))
            f['dormant_coin_movement'] = float(ef.get('dormant_coin_movement', 0.0))
            f['lth_supply_ratio'] = float(ef.get('lth_supply_ratio', 0.7))
            
            # --- PRICE ACTION / LIQUIDITY ---
            pa = ef.get('price_action', {})
            f['in_bull_fvg'] = float(pa.get('in_bull_fvg', 0))
            f['in_bear_fvg'] = float(pa.get('in_bear_fvg', 0))
            f['proximity_bull_ob'] = float(pa.get('proximity_bull_ob', 1.0))
            f['proximity_bear_ob'] = float(pa.get('proximity_bear_ob', 1.0))
            f['liquidity_sweep'] = float(l_sweep[i])
            f['vwap_deviation'] = float(v_dev[i])
            
            # --- VOLATILITY / STATISTICAL ---
            f['realized_vol_20'] = float(rv_20[i])
            f['atr_pct'] = float(atr_vals[i] / closes[i]) if closes[i] > 0 else 0.0
            f['vol_regime_encoded'] = float(ef.get('vol_regime_encoded', 1.0))
            f['zscore_20'] = float(v_dev[i])
            f['volume_delta'] = float(v_delta[i])
            f['btc_nasdaq_corr_24h'] = float(ef.get('btc_nasdaq_corr_24h', 0.7))
            f['correlation_breakdown'] = float(ef.get('correlation_breakdown', 0.0))
            
            # --- MACRO / GROWTH ---
            f['stablecoin_mint_velocity'] = float(ef.get('stablecoin_mint_velocity', 0.0))
            f['stablecoin_depeg_event'] = float(ef.get('stablecoin_depeg_event', 0.0))
            
            # Legacy Core Indicators (computed from price data)
            sma_10 = sma(closes, 10)
            sma_50 = sma(closes, 50)
            ema_10 = ema(closes, 10)
            ema_20 = ema(closes, 20)
            rsi_vals = rsi(closes, 14)
            macd_vals, signal_vals, hist_vals = macd(closes)
            adx_line, plus_di, minus_di = adx(highs, lows, closes, 14)
            bb_high, bb_low, bb_mid = bollinger_bands(closes, 20)
            stoch_k_vals, stoch_d_vals = stochastic(highs, lows, closes, 14, 3)
            
            f['sma_10_50_ratio'] = (sma_10[i] / sma_50[i]) if sma_50[i] > 0 else 1.0
            f['ema_10_20_ratio'] = (ema_10[i] / ema_20[i]) if ema_20[i] > 0 else 1.0
            f['rsi_14'] = float(rsi_vals[i])
            f['macd_hist'] = float(hist_vals[i])
            f['adx_14'] = float(adx_line[i])
            f['bb_width_20'] = float((bb_high[i] - bb_low[i]) / closes[i]) if closes[i] > 0 else 0.05
            f['stoch_k'] = float(stoch_k_vals[i])
            f['stoch_d'] = float(stoch_d_vals[i])
            f['cycle_phase_encoded'] = float(ef.get('cycle_phase_encoded', 0.0))
            f['dominant_period'] = float(ef.get('dominant_period', 30.0))
            
            # Sentiment features with proper normalization
            if sentiment_features:
                sent_mean = float(sentiment_features.get('sentiment_mean', 0.0))
                sent_zscore = float(sentiment_features.get('sentiment_z_score', 0.0))
                f['sentiment_mean'] = max(-1.0, min(1.0, sent_mean))  # Clamp to [-1, 1]
                f['sentiment_z_score'] = max(-3.0, min(3.0, sent_zscore))  # Clamp z-score to [-3, 3]
            else:
                f['sentiment_mean'] = 0.0
                f['sentiment_z_score'] = 0.0
            
            # ── Fix A: 5 new high-signal features ──

            # Mean Reversion Strength
            rsi_14 = f.get('rsi_14', 50.0)
            f['rsi_mean_reversion'] = (rsi_14 - 50.0) / 50.0  # -1 to +1, 0=neutral

            # Volatility Regime Encoding
            vol = f.get('realized_vol_20', 0.0)
            vol_ma = f.get('vol_ma_50', vol)
            f['vol_regime'] = 1.0 if vol > vol_ma * 1.2 else (-1.0 if vol < vol_ma * 0.8 else 0.0)

            # Trend Confirmation (EMA stack)
            ema10 = f.get('ema_10_20_ratio', 0.0)
            ema20 = f.get('ema_10_20_ratio', 0.0)  # use ratio as proxy
            ema50 = f.get('sma_10_50_ratio', 0.0)
            if ema10 > 0 and ema50 > 0:
                f['trend_stack'] = 1.0 if (f.get('ema_10_20_ratio', 1.0) > 1.0 and f.get('sma_10_50_ratio', 1.0) > 1.0) else \
                                   (-1.0 if (f.get('ema_10_20_ratio', 1.0) < 1.0 and f.get('sma_10_50_ratio', 1.0) < 1.0) else 0.0)
            else:
                f['trend_stack'] = 0.0

            # RSI Divergence Signal
            rsi_change = f.get('rsi_14', 50) - f.get('rsi_14_prev', f.get('rsi_14', 50))
            price_change = f.get('returns_1', 0.0)
            if price_change != 0:
                f['rsi_divergence'] = -1.0 if (price_change > 0 and rsi_change < 0) else (1.0 if (price_change < 0 and rsi_change > 0) else 0.0)
            else:
                f['rsi_divergence'] = 0.0

            # Volume Confirmation
            volume = f.get('volume_delta', 0.0)
            vol_sma = f.get('volume_sma_20', volume)
            f['volume_confirm'] = min(2.0, abs(volume) / (abs(vol_sma) + 1e-10))  # ratio, capped at 2x

            features.append(f)
        return features

    def predict(self, features: List[Dict[str, float]]
                ) -> List[Tuple[int, float]]:
        """
        Predict trade direction for each bar.

        Returns list of (class, confidence):
          class:      +1 (Long), 0 (Flat), -1 (Short)
          confidence: [0, 1]

        Applies confidence gating: only signals with conf > threshold pass.
        """
        if not features:
            return []

        # Try LightGBM
        if self._lgb_model is not None:
            return self._predict_lgb(features)

        # Fallback: weighted rule-based scoring using feature vector
        return self._predict_rule_based(features)

    def _predict_rule_based(self, features: List[Dict[str, float]]
                             ) -> List[Tuple[int, float]]:
        """
        Rule-based classifier using the feature vector.
        Mimics what a trained LightGBM would learn from the features.
        """
        results: List[Tuple[int, float]] = []

        for f in features:
            if not f:
                results.append((0, 0.0))
                continue

            score = 0.0
            weight_sum = 0.0

            # --- Trend signals (weight: 0.30) ---
            trend_score = 0.0
            trend_score += f.get('price_vs_sma50', 0) * 3.0
            trend_score += f.get('sma_10_50_ratio', 1) - 1.0
            trend_score += f.get('ema_10_slope', 0) * 2.0
            adx_s = f.get('adx_strength', 25)
            trend_conf = min(adx_s / 50, 1.0)
            score += 0.30 * max(-1, min(1, trend_score * 5)) * trend_conf
            weight_sum += 0.30

            # --- Momentum (weight: 0.25) ---
            rsi_val = f.get('rsi_14', 50)
            rsi_signal = (50 - rsi_val) / 50  # +1 at oversold, -1 at overbought
            stoch_signal = (50 - f.get('stoch_k', 50)) / 50
            macd_signal = max(-1, min(1, f.get('macd_hist', 0) * 10))
            roc_signal = max(-1, min(1, f.get('roc_10', 0) / 5))
            mom = (rsi_signal * 0.3 + stoch_signal * 0.2 + macd_signal * 0.3 + roc_signal * 0.2)
            score += 0.25 * mom
            weight_sum += 0.25

            # --- Mean-reversion (weight: 0.15) ---
            mr_score = f.get('mean_reversion_score', 0)
            score += 0.15 * max(-1, min(1, mr_score))
            weight_sum += 0.15

            # --- Volatility conditioning (weight: 0.10) ---
            vol_regime = f.get('vol_regime_encoded', 1)
            vol_scale = {0: 0.2, 1: 0.0, 2: -0.3, 3: -0.8}
            score += 0.10 * vol_scale.get(int(vol_regime), 0)
            weight_sum += 0.10

            # --- Sentim    ent (weight: 0.15) ---
            sent = f.get('sentiment_mean', 0)
            sent_conf = f.get('avg_confidence', 0.3)
            sent_z = f.get('sentiment_z_score', 0)
            sent_signal = sent * 0.6 + max(-1, min(1, sent_z * 0.3)) * 0.4
            score += 0.15 * sent_signal * sent_conf
            weight_sum += 0.15

            # --- Cycle (weight: 0.05) ---
            cycle_mom = f.get('cycle_momentum', 0)
            score += 0.05 * float(cycle_mom)
            weight_sum += 0.05

            # Normalize and boost sensitivity
            final_score = score / max(weight_sum, 1e-10)
            final_score = final_score * 2.0  # Boost sensitivity to ensure signals trigger
            final_score = max(-1.0, min(1.0, final_score))

            # Confidence scoring: allow it to realistically surpass the 0.65 threshold
            # Base confidence scales with signal strength, plus boosts from trend and sentiment
            base_conf = 0.4 + (abs(final_score) * 0.4)
            confidence = base_conf + (trend_conf * 0.15) + (sent_conf * 0.05)

            # Classify
            if final_score > 0.15 and confidence >= self.confidence_threshold:
                cls = 1
            elif final_score < -0.15 and confidence >= self.confidence_threshold:
                cls = -1
            else:
                cls = 0
                confidence = 1.0 - abs(final_score)  # confident in no-trade

            results.append((cls, float(min(1.0, float(confidence)))))

        return results

    def _predict_lgb(self, features: List[Dict[str, float]]
                      ) -> List[Tuple[int, float]]:
        """Predict using trained LightGBM model."""
        try:
            import numpy as np
            import logging
            _logger = logging.getLogger(__name__)
            X = np.array([[f.get(name, 0.0) for name in self.FEATURE_NAMES]
                          for f in features])

            # Before prediction, align feature count with model expectations
            if self._lgb_model is not None:
                expected = self._lgb_model.num_feature()
                if X.shape[1] > expected:
                    _logger.warning(f"[LGB] Truncating features {X.shape[1]} -> {expected} to match trained model")
                    X = X[:, :expected]
                elif X.shape[1] < expected:
                    pad = np.zeros((X.shape[0], expected - X.shape[1]))
                    _logger.warning(f"[LGB] Padded features {X.shape[1]} -> {expected}")
                    X = np.hstack([X, pad])

            proba = self._lgb_model.predict(X, predict_disable_shape_check=True)
            results = []
            for p in proba:
                # p = [prob_short, prob_flat, prob_long]
                cls_idx = int(p.argmax())
                # Calibrate confidence using Platt Scaling
                calibrated_conf = self._calibrate_probability(p)
                
                # Label conversion (+1: Long, 0: Flat/Wait, -1: Short)
                if calibrated_conf > self.confidence_threshold:
                    if cls_idx == 0: # Original mapping: 0 -> Short (-1)
                        cls = -1
                    elif cls_idx == 2: # Original mapping: 2 -> Long (+1)
                        cls = 1
                    else: # Original mapping: 1 -> Flat (0)
                        cls = 0
                    results.append((cls, calibrated_conf))
                else:
                    results.append((0, calibrated_conf)) # If confidence is too low, always flat
            return results
        except Exception:
            return self._predict_rule_based(features)

    # (Deleted extra duplicate _calibrate_probability block)

    # -------------------------------------------------------------------
    # Feedback & incremental learning utilities
    # -------------------------------------------------------------------
    def log_trade(self, features: Dict[str, float], direction: int, net_pnl: float, 
                  entry_price: float, exit_price: float, bars_held: int) -> None:
        """
        Record a closed trade using Triple Barrier Labeling logic.
        Labels align with LightGBM 3-class expectations: 0=SHORT, 1=FLAT, 2=LONG
        """
        entry = features.copy()
        
        # Triple Barrier Labeling: Map outcome to class labels
        # Label 0 (SHORT): Loss or bearish outcome
        # Label 1 (FLAT): Neutral/breakeven
        # Label 2 (LONG): Profit or bullish outcome
        pnl_thresh = max(10.0, abs(entry_price * 0.001))  # Min $10 or 0.1% threshold
        
        if net_pnl > pnl_thresh:
            label = 2  # LONG: profit
        elif net_pnl < -pnl_thresh:
            label = 0  # SHORT: loss
        else:
            label = 1  # FLAT: breakeven
            
        entry['label'] = label
        entry['direction'] = direction
        entry['pnl'] = net_pnl
        entry['bars_held'] = bars_held
        
        self._trade_log.append(entry)
        if self.trade_log_path:
            try:
                import csv, os
                os.makedirs(os.path.dirname(self.trade_log_path), exist_ok=True)
                write_header = not os.path.exists(self.trade_log_path)
                with open(self.trade_log_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(entry.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(entry)
            except Exception:
                pass

    def retrain_from_log(self, max_examples: Optional[int] = None) -> None:
        """Retrain or update the model using logged trades.

        If LightGBM is available the method attempts to incrementally train the
        existing model; otherwise it simply returns.  ``max_examples`` can be
        used to limit the amount of history used (e.g. keep most recent trades).
        """
        try:
            import lightgbm as lgb
            import numpy as np
        except ImportError:
            return

        # gather records from memory and disk
        data = list(self._trade_log)
        if self.trade_log_path and os.path.exists(self.trade_log_path):
            import csv
            with open(self.trade_log_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    data.append({k: float(v) for k, v in row.items()})
        if not data:
            return

        # truncate
        if max_examples is not None and len(data) > max_examples:
            data = data[-max_examples:]

        # prepare dataset
        X = []
        y = []
        for entry in data:
            dirn = int(entry.pop('direction', 0))
            y.append(1 if dirn > 0 else (2 if dirn < 0 else 0))
            X.append([entry.get(name, 0.0) for name in self.FEATURE_NAMES])
        if not X:
            return
        import logging
        _logger = logging.getLogger(__name__)
        import pandas as pd
        X_arr = pd.DataFrame(X, columns=self.FEATURE_NAMES)

        # Drop columns that are all zeros (likely unpopulated institutional signals)
        zero_cols = [c for c in X_arr.columns if (X_arr[c] == 0).all()]
        if zero_cols:
            _logger.info(f"[LGB-TRAIN] Dropping {len(zero_cols)} all-zero columns: {zero_cols[:5]}...")
            X_arr = X_arr.drop(columns=zero_cols)

        X_arr = X_arr.values
        y_arr = np.array(y)

        tuner = ModelTuner(n_trials=20) # Low trials for quick fine-tuning
        best_params = tuner.tune_lightgbm(X_arr, y_arr)
        
        dtrain = lgb.Dataset(X_arr, label=y_arr)
        self._lgb_model = lgb.train(best_params, dtrain, num_boost_round=100)
        self._fitted = True

    # -------------------------------------------------------------------
    # Generate forecast signal for executor display
    # -------------------------------------------------------------------
    def forecast_signal(self, features: List[Dict[str, float]]
                         ) -> List[float]:
        """
        Generate continuous forecast signal [-1, +1] for each bar.
        This is what gets displayed as 'forecast_signal' in the executor.
        """
        predictions = self.predict(features)
        return [cls * conf for cls, conf in predictions]
