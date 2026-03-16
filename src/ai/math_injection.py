"""
Math Injection Engine — Pre-Computed Quant Data for LLM Prompts
================================================================
CORE PRINCIPLE: The LLM never computes numbers. Every numerical value in the
prompt is PRE-COMPUTED by our validated quant models. The LLM's job is ONLY
to interpret the data and produce a qualitative decision.

This prevents:
  1. LLM hallucinating fake statistics
  2. LLM making up indicators or signal values
  3. LLM performing incorrect arithmetic
  4. LLM inventing market conditions

The MathInjector computes ALL relevant quant features, formats them into a
structured data block, and embeds it into the prompt with explicit instructions
that these numbers are GROUND TRUTH and must not be modified.

Usage:
    from src.ai.math_injection import MathInjector
    injector = MathInjector()
    quant_block = injector.compute_and_format(prices, highs, lows, volumes, sentiment)
    prompt = f"{quant_block}\\n\\n{analysis_instructions}"
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class MathInjector:
    """
    Computes all quant features and formats them for LLM injection.
    Every number the LLM sees is verified by our models.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._computation_log: List[Dict] = []

    def compute_full_state(self,
                           prices: np.ndarray,
                           highs: np.ndarray,
                           lows: np.ndarray,
                           volumes: np.ndarray,
                           sentiment_score: float = 0.0,
                           asset: str = 'BTCUSDT',
                           account_balance: float = 10000.0,
                           open_positions: Optional[Dict] = None,
                           ) -> Dict[str, Any]:
        """
        Compute ALL quant features from raw data.
        Returns a structured dict with every computed value.
        """
        prices = np.asarray(prices, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        volumes = np.asarray(volumes, dtype=float)
        n = len(prices)

        state = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'asset': asset,
            'current_price': float(prices[-1]) if n > 0 else 0.0,
            'n_bars': n,
        }

        # ── 1. PRICE STATISTICS ──
        if n >= 20:
            state['price_stats'] = {
                'last_price': float(prices[-1]),
                'price_24h_ago': float(prices[-24]) if n >= 24 else float(prices[0]),
                'return_24h_pct': float((prices[-1] / prices[-24] - 1) * 100) if n >= 24 else 0.0,
                'return_7d_pct': float((prices[-1] / prices[-168] - 1) * 100) if n >= 168 else 0.0,
                'high_24h': float(np.max(highs[-24:])) if n >= 24 else float(np.max(highs)),
                'low_24h': float(np.min(lows[-24:])) if n >= 24 else float(np.min(lows)),
                'avg_volume_24h': float(np.mean(volumes[-24:])) if n >= 24 else float(np.mean(volumes)),
                'volume_change_pct': float((volumes[-1] / np.mean(volumes[-20:]) - 1) * 100) if n >= 20 else 0.0,
            }

        # ── 2. TREND INDICATORS ──
        if n >= 50:
            from src.indicators.indicators import sma, ema, rsi, macd, adx, atr, bollinger_bands
            sma_20 = sma(prices.tolist(), 20)
            sma_50 = sma(prices.tolist(), 50)
            ema_10 = ema(prices.tolist(), 10)
            rsi_14 = rsi(prices.tolist(), 14)
            macd_line, macd_sig, macd_hist = macd(prices.tolist())
            adx_val, plus_di, minus_di = adx(highs.tolist(), lows.tolist(), prices.tolist())
            atr_val = atr(highs.tolist(), lows.tolist(), prices.tolist(), 14)
            bb_upper, bb_mid, bb_lower = bollinger_bands(prices.tolist(), 20)

            state['trend'] = {
                'sma_20': float(sma_20[-1]),
                'sma_50': float(sma_50[-1]),
                'price_vs_sma20': 'ABOVE' if prices[-1] > sma_20[-1] else 'BELOW',
                'price_vs_sma50': 'ABOVE' if prices[-1] > sma_50[-1] else 'BELOW',
                'ema_10_slope': float((ema_10[-1] - ema_10[-5]) / (ema_10[-5] + 1e-12) * 100) if n >= 55 else 0.0,
                'golden_cross': sma_20[-1] > sma_50[-1],
                'rsi_14': float(rsi_14[-1]),
                'rsi_zone': 'OVERBOUGHT' if rsi_14[-1] > 70 else 'OVERSOLD' if rsi_14[-1] < 30 else 'NEUTRAL',
                'macd_histogram': float(macd_hist[-1]),
                'macd_signal': 'BULLISH' if macd_hist[-1] > 0 else 'BEARISH',
                'adx': float(adx_val[-1]),
                'trend_strength': 'STRONG' if adx_val[-1] > 25 else 'WEAK',
                'trend_direction': 'UP' if plus_di[-1] > minus_di[-1] else 'DOWN',
                'atr_14': float(atr_val[-1]),
                'atr_pct': float(atr_val[-1] / prices[-1] * 100),
                'bb_position': float((prices[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1] + 1e-12)),
                'bb_width_pct': float((bb_upper[-1] - bb_lower[-1]) / bb_mid[-1] * 100),
            }

        # ── 3. VOLATILITY REGIME ──
        if n >= 30:
            from src.models.volatility import ewma_volatility, classify_volatility_regime
            ewma_vol = ewma_volatility(prices.tolist())
            vol_regimes = classify_volatility_regime(ewma_vol)

            state['volatility'] = {
                'ewma_vol': float(ewma_vol[-1]),
                'regime': str(vol_regimes[-1].name) if vol_regimes else 'MEDIUM',
                'vol_percentile': float(np.mean(np.array(ewma_vol) <= ewma_vol[-1]) * 100),
            }

            # GARCH if enough data
            if n >= 100:
                try:
                    from src.models.volatility import GARCH11
                    g = GARCH11()
                    g.fit(prices.tolist())
                    garch_vol = g.forecast(prices.tolist())
                    state['volatility']['garch_vol'] = float(garch_vol[-1])
                    state['volatility']['garch_vs_ewma'] = 'HIGHER' if garch_vol[-1] > ewma_vol[-1] else 'LOWER'
                except Exception:
                    pass

        # ── 4. HURST EXPONENT ──
        if n >= 100:
            try:
                from src.models.hurst import HurstExponent
                h = HurstExponent()
                hurst_result = h.compute(prices, window=min(200, n))
                state['hurst'] = {
                    'value': float(hurst_result['hurst']),
                    'regime': hurst_result['regime'],
                    'confidence': float(hurst_result.get('r_squared', 0)),
                    'interpretation': (
                        'Series is TRENDING (persistent)' if hurst_result['regime'] == 'trending'
                        else 'Series is MEAN-REVERTING (anti-persistent)' if hurst_result['regime'] == 'mean_reverting'
                        else 'Series is RANDOM WALK (no edge)'
                    ),
                }
            except Exception:
                pass

        # ── 5. HMM REGIME ──
        if n >= 100:
            try:
                from src.models.hmm_regime import HMMRegimeDetector
                hmm = HMMRegimeDetector()
                log_returns = np.diff(np.log(prices))
                realized_vol = np.array([np.std(log_returns[max(0, i-20):i]) for i in range(1, len(log_returns)+1)])
                vol_changes = np.diff(volumes) / (volumes[:-1] + 1e-12)
                # Align lengths
                min_len = min(len(log_returns), len(realized_vol), len(vol_changes))
                hmm.fit(log_returns[-min_len:], realized_vol[-min_len:], vol_changes[-min_len:])
                hmm_result = hmm.predict(log_returns[-min_len:], realized_vol[-min_len:], vol_changes[-min_len:])
                state['hmm_regime'] = {
                    'current_regime': hmm_result['regime'],
                    'crisis_probability': float(hmm_result['crisis_prob']),
                    'regime_stability': float(hmm_result['stability']),
                    'bull_prob': float(hmm_result.get('bull_prob', 0)),
                    'bear_prob': float(hmm_result.get('bear_prob', 0)),
                }
            except Exception as e:
                logger.debug(f"HMM computation failed: {e}")

        # ── 6. ORNSTEIN-UHLENBECK ──
        if n >= 100:
            try:
                from src.models.ou_process import OUProcess
                ou = OUProcess()
                ou_result = ou.fit_and_signal(prices, window=min(252, n))
                state['ou_process'] = {
                    'is_stationary': ou_result.get('ou_is_stationary', False),
                    'half_life_bars': float(ou_result.get('ou_half_life', 999)),
                    'z_score': float(ou_result.get('ou_z_score', 0)),
                    'signal': int(ou_result.get('ou_signal', 0)),
                    'equilibrium': float(ou_result.get('ou_theta', prices[-1])),
                    'mean_reversion_speed': float(ou_result.get('ou_kappa', 0)),
                }
            except Exception:
                pass

        # ── 7. KALMAN FILTER ──
        if n >= 30:
            try:
                from src.models.kalman_filter import KalmanTrendFilter
                kf = KalmanTrendFilter()
                kf_result = kf.filter(prices)
                state['kalman'] = {
                    'filtered_price': float(kf_result['level'][-1]),
                    'slope': float(kf_result['slope'][-1]),
                    'slope_direction': 'UP' if kf_result['slope'][-1] > 0 else 'DOWN',
                    'snr': float(kf_result['snr'][-1]),
                    'signal_clarity': 'HIGH' if kf_result['snr'][-1] > 2 else 'LOW' if kf_result['snr'][-1] < 0.3 else 'MEDIUM',
                    'signal': int(kf_result['signal'][-1]),
                }
            except Exception:
                pass

        # ── 8. MONTE CARLO RISK ──
        if n >= 50:
            try:
                from src.risk.monte_carlo_risk import MonteCarloRisk
                mc = MonteCarloRisk(n_simulations=5000, horizon=24)
                vol_est = float(np.std(np.diff(np.log(prices[-100:]))) if n >= 100 else 0.02)
                regime = state.get('hmm_regime', {}).get('current_regime', 'normal')
                mc_result = mc.simulate(
                    current_price=float(prices[-1]),
                    volatility=vol_est,
                    regime=regime,
                )
                state['monte_carlo_risk'] = {
                    'var_95_pct': float(mc_result['mc_var_95'] * 100),
                    'cvar_95_pct': float(mc_result['mc_cvar_95'] * 100),
                    'var_99_pct': float(mc_result['mc_var_99'] * 100),
                    'risk_score': float(mc_result['mc_risk_score']),
                    'risk_level': 'HIGH' if mc_result['mc_risk_score'] > 0.7 else 'LOW' if mc_result['mc_risk_score'] < 0.3 else 'MEDIUM',
                    'position_scale': float(mc_result['mc_position_scale']),
                    'prob_profit_24h': float(mc_result['mc_prob_profit'] * 100),
                }
            except Exception:
                pass

        # ── 9. EVT TAIL RISK ──
        if n >= 100:
            try:
                from src.risk.evt_risk import EVTRisk
                evt = EVTRisk()
                returns = np.diff(np.log(prices))
                evt_result = evt.fit_and_assess(returns)
                state['evt_tail_risk'] = {
                    'var_99_pct': float(evt_result['evt_var_99'] * 100),
                    'tail_ratio': float(evt_result['evt_tail_ratio']),
                    'tail_type': 'HEAVY' if evt_result['evt_xi'] > 0.1 else 'NORMAL' if evt_result['evt_xi'] > -0.1 else 'THIN',
                    'risk_score': float(evt_result['evt_risk_score']),
                    'position_scale': float(evt_result['evt_position_scale']),
                }
            except Exception:
                pass

        # ── 10. HAWKES EVENT CLUSTERING ──
        if n >= 50:
            try:
                from src.models.hawkes_process import HawkesProcess
                hp = HawkesProcess()
                hp_result = hp.trade_timing_signal(prices, volume=volumes)
                state['hawkes'] = {
                    'intensity': float(hp_result['intensity']),
                    'regime': hp_result['regime'],
                    'trade_allowed': hp_result['trade_allowed'],
                    'interpretation': (
                        'DANGER: Event clustering detected — defer new entries'
                        if not hp_result['trade_allowed']
                        else 'CALM: Low event intensity — favorable for entries'
                        if hp_result['regime'] == 'calm'
                        else 'NORMAL: Proceed with standard caution'
                    ),
                }
            except Exception:
                pass

        # ── 11. ALPHA DECAY ──
        state['alpha_decay'] = {
            'note': 'Alpha decay requires historical signal data. Applied at meta-controller level.',
        }

        # ── 12. SENTIMENT ──
        state['sentiment'] = {
            'finbert_score': float(sentiment_score),
            'sentiment_zone': (
                'BULLISH' if sentiment_score > 0.2
                else 'BEARISH' if sentiment_score < -0.2
                else 'NEUTRAL'
            ),
        }

        # ── 13. ACCOUNT & POSITION STATE ──
        state['account'] = {
            'balance': float(account_balance),
            'open_positions': len(open_positions) if open_positions else 0,
        }

        # Log computation
        self._computation_log.append({
            'time': state['timestamp'],
            'asset': asset,
            'n_features': sum(1 for k in state if isinstance(state[k], dict)),
        })

        return state

    def format_for_prompt(self, state: Dict[str, Any]) -> str:
        """
        Format computed state into a structured text block for LLM injection.
        Uses explicit language that these are GROUND TRUTH values.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("VERIFIED QUANT DATA — PRE-COMPUTED BY VALIDATED MODELS")
        lines.append("IMPORTANT: These numbers are GROUND TRUTH. Do NOT modify,")
        lines.append("recalculate, or override ANY values below. Your role is")
        lines.append("ONLY to INTERPRET this data and make qualitative decisions.")
        lines.append("=" * 70)
        lines.append("")

        # Format each section
        for key, value in state.items():
            if isinstance(value, dict):
                lines.append(f"### {key.upper().replace('_', ' ')}")
                for k, v in value.items():
                    if isinstance(v, float):
                        lines.append(f"  {k}: {v:.6f}")
                    else:
                        lines.append(f"  {k}: {v}")
                lines.append("")
            else:
                lines.append(f"{key}: {value}")

        lines.append("=" * 70)
        lines.append("END VERIFIED DATA — All analysis must reference values above.")
        lines.append("=" * 70)

        return '\n'.join(lines)

    def compute_and_format(self,
                           prices: np.ndarray,
                           highs: np.ndarray,
                           lows: np.ndarray,
                           volumes: np.ndarray,
                           sentiment_score: float = 0.0,
                           asset: str = 'BTCUSDT',
                           account_balance: float = 10000.0,
                           open_positions: Optional[Dict] = None,
                           ) -> str:
        """One-shot: compute all quant features and return formatted prompt block."""
        state = self.compute_full_state(
            prices, highs, lows, volumes, sentiment_score,
            asset, account_balance, open_positions
        )
        return self.format_for_prompt(state)

    def get_feature_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract a compact summary suitable for LightGBM feature injection.
        Returns flat dict of numeric features.
        """
        flat = {}
        for section_key, section in state.items():
            if isinstance(section, dict):
                for k, v in section.items():
                    if isinstance(v, (int, float)):
                        flat[f"{section_key}_{k}"] = float(v)
                    elif isinstance(v, bool):
                        flat[f"{section_key}_{k}"] = 1.0 if v else 0.0
        return flat
