"""
L1 Numerical Models — Signal Generation
=========================================
Z-score, MA crossover, momentum scoring, and composite L1 signal generator.

Mathematical equations:
  Z-score:  z_t = (P_t - μ_w) / σ_w   where μ_w, σ_w are window stats
  MA Cross: signal = 1 if SMA_short crosses above SMA_long (golden cross)
  Momentum: normalized rate of change + trend strength
  Composite: weighted combination of all sub-signals → [-1, +1]
"""

import math
import statistics
from typing import List, Dict, Optional, Tuple

from src.indicators.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr, stochastic, roc, adx, bb_width,
    bulk_indicators
)
from src.models.volatility import (
    ewma_volatility, log_returns, VolRegime, classify_volatility_regime,
    GARCH11
)
from src.models.cycle_detector import (
    detect_dominant_cycles, detect_cycle_phase, adaptive_holding_period
)
from src.models.forecasting import ForecastingEngine


# ---------------------------------------------------------------------------
# Z-Score
# ---------------------------------------------------------------------------
def zscore(series: List[float], window: int) -> List[float]:
    """z_t = (P_t − μ_w) / σ_w"""
    if window <= 0:
        raise ValueError("window must be > 0")
    out: List[float] = []
    for i in range(len(series)):
        if i + 1 < window:
            out.append(float('nan'))
        else:
            window_vals = series[i + 1 - window: i + 1]
            mean = statistics.mean(window_vals)
            stdev = statistics.pstdev(window_vals)
            if stdev == 0:
                out.append(0.0)
            else:
                out.append((series[i] - mean) / stdev)
    return out


# ---------------------------------------------------------------------------
# MA Crossover Signal
# ---------------------------------------------------------------------------
def moving_average_crossover_signal(short_ma: List[float],
                                     long_ma: List[float]) -> List[int]:
    """
    Discrete crossover signal:
      +1  when short MA crosses above long MA (golden cross)
      -1  when short MA crosses below long MA (death cross)
       0  otherwise
    """
    n = min(len(short_ma), len(long_ma))
    out = [0] * n
    prev_diff: Optional[float] = None
    for i in range(n):
        s = short_ma[i]
        l_val = long_ma[i]
        if s != s or l_val != l_val:  # nan check
            out[i] = 0
            prev_diff = None
            continue
        diff = s - l_val
        if prev_diff is None:
            out[i] = 0
        else:
            if prev_diff <= 0 and diff > 0:
                out[i] = 1
            elif prev_diff >= 0 and diff < 0:
                out[i] = -1
            else:
                out[i] = 0
        prev_diff = diff
    return out


# ---------------------------------------------------------------------------
# Multi-Signal L1 Composite Generator
# ---------------------------------------------------------------------------
class L1SignalEngine:
    """
    L1 Quantitative Engine — generates composite trading signals from:
      1. Trend: SMA/EMA crossovers, MACD, ADX
      2. Mean-reversion: Z-score, Bollinger Bands
      3. Momentum: RSI, Stochastic, ROC
      4. Volatility regime: EWMA, GARCH conditioning
      5. Market cycles: FFT-based cycle adaptation

    Each sub-signal is normalized to [-1, +1] and combined with weights.
    Final composite signal: S_L1 = Σ w_i · s_i  where s_i ∈ [-1, +1]
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # Sub-signal weights (must sum to 1.0)
        self.weights = cfg.get('weights', {
            'trend': 0.30,
            'mean_reversion': 0.20,
            'momentum': 0.25,
            'volatility': 0.15,
            'cycle': 0.10,
            # optional forecasting sub-signal (FinGPT/LightGBM/N-Beats/TFT)
            'forecast': 0.0,
        })
        # forecasting engine plugged into L1 layer
        self.forecaster = ForecastingEngine(cfg.get('forecast', {}))
        # Parameters
        self.sma_short = cfg.get('sma_short', 10)
        self.sma_long = cfg.get('sma_long', 50)
        self.z_window = cfg.get('z_window', 20)
        self.rsi_period = cfg.get('rsi_period', 14)
        self.bb_period = cfg.get('bb_period', 20)
        self.roc_period = cfg.get('roc_period', 12)

        self.garch = GARCH11()
        self._garch_fitted = False

    def generate_signals(self, closes: List[float],
                          highs: Optional[List[float]] = None,
                          lows: Optional[List[float]] = None,
                          volumes: Optional[List[float]] = None
                          ) -> Dict:
        """
        Generate all L1 sub-signals and composite.
        Returns dict with all components for downstream use and debugging.
        """
        n = len(closes)
        if n < self.sma_long + 5:
            return self._empty_result(n)

        # Use closes for H/L/V if not provided
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes
        if volumes is None:
            volumes = [1.0] * n

        # ----- 0. BULK INDICATOR SUITE (for ML features) -----
        l1_features = bulk_indicators(closes, highs, lows, volumes)

        # ----- 1. TREND SUB-SIGNAL -----
        short_ma = sma(closes, self.sma_short)
        long_ma = sma(closes, self.sma_long)
        crossover = moving_average_crossover_signal(short_ma, long_ma)
        macd_line, macd_sig, macd_hist = macd(closes)
        adx_line, plus_di, minus_di = adx(highs, lows, closes)

        trend_signal = self._compute_trend_signal(
            closes, short_ma, long_ma, crossover, macd_hist, adx_line, plus_di, minus_di
        )

        # ----- 2. MEAN-REVERSION SUB-SIGNAL -----
        z = zscore(closes, self.z_window)
        bb_upper, bb_mid, bb_lower = bollinger_bands(closes, self.bb_period)
        mr_signal = self._compute_mean_reversion_signal(closes, z, bb_upper, bb_lower)

        # ----- 3. MOMENTUM SUB-SIGNAL -----
        rsi_vals = rsi(closes, self.rsi_period)
        stoch_k, stoch_d = stochastic(highs, lows, closes)
        roc_vals = roc(closes, self.roc_period)
        mom_signal = self._compute_momentum_signal(rsi_vals, stoch_k, roc_vals)

        # ----- 4. VOLATILITY CONDITIONING -----
        ewma_vol = ewma_volatility(closes)
        if not self._garch_fitted and n > 50:
            self.garch.fit(closes)
            self._garch_fitted = True
        garch_vol = self.garch.forecast(closes)
        vol_regimes = classify_volatility_regime(ewma_vol)
        vol_signal = self._compute_volatility_signal(vol_regimes)

        # ----- 5. FORECASTING (optional) -----
        forecast_signal = [0.0] * n
        if hasattr(self, 'forecaster'):
            try:
                forecast_signal = self.forecaster.generate_signal(closes)
            except Exception:
                forecast_signal = [0.0] * n

        # ----- 6. CYCLE ADAPTATION -----
        cycles = detect_dominant_cycles(closes)
        phase = detect_cycle_phase(closes)
        dom_period = cycles[0][0] if cycles else 30
        cycle_signal = self._compute_cycle_signal(phase)

        # ----- 7. KALMAN FILTER TREND (quant upgrade) -----
        kalman_signal = [0.0] * n
        kalman_data = {}
        try:
            from src.models.kalman_filter import KalmanTrendFilter
            kf = KalmanTrendFilter()
            import numpy as _np
            kalman_data = kf.filter(_np.array(closes, dtype=float))
            # Convert slope to [-1, +1] signal
            for i in range(n):
                slope = kalman_data['slope'][i]
                snr = kalman_data['snr'][i]
                # Scale slope by SNR: clear signal → strong signal
                kalman_signal[i] = max(-1.0, min(1.0, slope * 50 * min(snr, 3.0)))
        except Exception:
            pass

        # ----- 8. HURST-WEIGHTED OU MEAN REVERSION (quant upgrade) -----
        ou_signal = [0.0] * n
        hurst_data = {}
        ou_data = {}
        try:
            from src.models.hurst import HurstExponent
            from src.models.ou_process import OUProcess
            import numpy as _np
            arr = _np.array(closes, dtype=float)

            h_est = HurstExponent()
            hurst_data = h_est.compute(arr, window=min(200, n))

            ou = OUProcess()
            ou_data = ou.fit_and_signal(arr, window=min(252, n))

            # Only use OU signal when Hurst confirms mean reversion
            if hurst_data.get('regime') == 'mean_reverting' and ou_data.get('ou_is_stationary'):
                ou_z = ou_data.get('ou_z_score', 0)
                # Last bar gets the OU signal; rest stay 0
                ou_signal[-1] = max(-1.0, min(1.0, -ou_z / 2.0))
        except Exception:
            pass

        # ----- 9. FRACTIONAL DIFFERENCING MOMENTUM (quant upgrade) -----
        fracdiff_signal = [0.0] * n
        fracdiff_data = {}
        try:
            from src.models.fracdiff import FractionalDiff
            import numpy as _np
            fd = FractionalDiff()
            arr = _np.array(closes, dtype=float)
            fracdiff_data = fd.compute_features(arr)
            # Use fracdiff momentum as directional signal
            fd_mom = fracdiff_data.get('fracdiff_momentum', 0)
            fracdiff_signal[-1] = max(-1.0, min(1.0, fd_mom * 10))
        except Exception:
            pass

        # ----- 10. HAWKES TIMING SIGNAL (quant upgrade) -----
        hawkes_signal = [0.0] * n
        hawkes_data = {}
        try:
            from src.models.hawkes_process import HawkesProcess
            import numpy as _np
            hp = HawkesProcess()
            arr = _np.array(closes, dtype=float)
            hawkes_data = hp.trade_timing_signal(arr)
            # Trade-allowed acts as a filter, not a directional signal
            # We use it to dampen signals during clustering
            if not hawkes_data.get('trade_allowed', True):
                hawkes_signal[-1] = -0.5  # Reduce all signals during clustering
        except Exception:
            pass

        # ----- COMPOSITE -----
        composite: List[float] = []
        # Rebalance weights to include new quant signals
        w = self.weights.copy()
        # Allocate weights to new signals (taken from existing)
        w_kalman = 0.05
        w_ou = 0.05
        w_fracdiff = 0.03
        w_trend_adj = max(0.0, w['trend'] - 0.04)
        w_mr_adj = max(0.0, w['mean_reversion'] - 0.04)

        for i in range(n):
            s = (w_trend_adj * trend_signal[i]
                 + w_mr_adj * mr_signal[i]
                 + w['momentum'] * mom_signal[i]
                 + w['volatility'] * vol_signal[i]
                 + w.get('forecast', 0) * forecast_signal[i]
                 + w['cycle'] * cycle_signal[i]
                 + w_kalman * kalman_signal[i]
                 + w_ou * ou_signal[i]
                 + w_fracdiff * fracdiff_signal[i])
            # Clamp to [-1, +1]
            composite.append(max(-1.0, min(1.0, s)))

        # Discretize to trade signals
        trade_signals = self._discretize(composite, vol_regimes)

        return {
            'composite': composite,
            'signals': trade_signals,
            'trend': trend_signal,
            'mean_reversion': mr_signal,
            'momentum': mom_signal,
            'volatility_signal': vol_signal,
            'forecast_signal': forecast_signal,
            'cycle_signal': cycle_signal,
            'kalman_signal': kalman_signal,
            'ou_signal': ou_signal,
            'short_ma': short_ma,
            'long_ma': long_ma,
            'rsi': rsi_vals,
            'macd_hist': macd_hist,
            'zscore': z,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'ewma_vol': ewma_vol,
            'garch_vol': garch_vol,
            'vol_regimes': vol_regimes,
            'cycle_phase': phase,
            'dominant_cycles': cycles,
            'holding_period': adaptive_holding_period(dom_period, phase),
            'l1_features': l1_features,
            'kalman_data': kalman_data,
            'hurst_data': hurst_data,
            'ou_data': ou_data,
            'fracdiff_signal': fracdiff_signal,
            'fracdiff_data': fracdiff_data,
            'hawkes_data': hawkes_data,
        }

    def _compute_trend_signal(self, closes, short_ma, long_ma, crossover,
                               macd_hist, adx_line, plus_di, minus_di) -> List[float]:
        """Combine trend indicators into [-1, +1] signal."""
        n = len(closes)
        out: List[float] = []
        for i in range(n):
            s = 0.0
            cnt = 0

            # MA position (continuous)
            if not math.isnan(short_ma[i]) and not math.isnan(long_ma[i]):
                if long_ma[i] != 0:
                    diff_pct = (short_ma[i] - long_ma[i]) / long_ma[i]
                    s += max(-1.0, min(1.0, diff_pct * 20))  # scale
                    cnt += 1

            # MACD histogram direction
            if i < len(macd_hist) and not math.isnan(macd_hist[i]):
                macd_norm = max(-1.0, min(1.0, macd_hist[i] * 100))
                s += macd_norm
                cnt += 1

            # ADX trend strength * direction
            if i < len(adx_line):
                strength = min(adx_line[i] / 50.0, 1.0)  # normalize to [0,1]
                direction = 1.0 if plus_di[i] > minus_di[i] else -1.0
                s += strength * direction
                cnt += 1

            out.append(s / max(cnt, 1))
        return out

    def _compute_mean_reversion_signal(self, closes, z, bb_upper, bb_lower
                                        ) -> List[float]:
        """Mean-reversion signal: buy oversold, sell overbought."""
        n = len(closes)
        out: List[float] = []
        for i in range(n):
            s = 0.0
            cnt = 0

            # Z-score mean reversion (-z because high z = overbought = sell)
            if not math.isnan(z[i]):
                s += max(-1.0, min(1.0, -z[i] / 2.0))
                cnt += 1

            # Bollinger Band position
            if not math.isnan(bb_upper[i]) and not math.isnan(bb_lower[i]):
                bb_range = bb_upper[i] - bb_lower[i]
                if bb_range > 0:
                    position = (closes[i] - bb_lower[i]) / bb_range  # 0 to 1
                    bb_signal = 1.0 - 2.0 * position  # +1 at lower, -1 at upper
                    s += max(-1.0, min(1.0, bb_signal))
                    cnt += 1

            out.append(s / max(cnt, 1))
        return out

    def _compute_momentum_signal(self, rsi_vals, stoch_k, roc_vals
                                  ) -> List[float]:
        """Momentum signal: RSI + Stochastic + ROC."""
        n = max(len(rsi_vals), len(stoch_k), len(roc_vals))
        out: List[float] = []
        for i in range(n):
            s = 0.0
            cnt = 0

            # RSI: <30 oversold (buy), >70 overbought (sell)
            if i < len(rsi_vals) and not math.isnan(rsi_vals[i]):
                rsi_norm = (50 - rsi_vals[i]) / 50.0  # +1 at 0, -1 at 100
                s += max(-1.0, min(1.0, rsi_norm))
                cnt += 1

            # Stochastic: <20 oversold, >80 overbought
            if i < len(stoch_k) and not math.isnan(stoch_k[i]):
                stoch_norm = (50 - stoch_k[i]) / 50.0
                s += max(-1.0, min(1.0, stoch_norm))
                cnt += 1

            # ROC: positive = bullish momentum
            if i < len(roc_vals) and not math.isnan(roc_vals[i]):
                roc_norm = max(-1.0, min(1.0, roc_vals[i] / 10.0))
                s += roc_norm
                cnt += 1

            out.append(s / max(cnt, 1))
        return out

    def _compute_volatility_signal(self, vol_regimes: List[VolRegime]) -> List[float]:
        """
        Volatility conditioning signal:
        LOW vol  → slightly bullish (breakout expected)
        HIGH vol → reduce exposure
        EXTREME  → strong reduce / cash
        """
        mapping = {
            VolRegime.LOW: 0.3,
            VolRegime.MEDIUM: 0.0,
            VolRegime.HIGH: -0.4,
            VolRegime.EXTREME: -0.9,
        }
        return [mapping.get(r, 0.0) for r in vol_regimes]

    def _compute_cycle_signal(self, phase: str) -> List[float]:
        """Market cycle phase → directional bias."""
        from src.models.cycle_detector import CyclePhase
        mapping = {
            CyclePhase.ACCUMULATION: 0.3,
            CyclePhase.MARKUP: 0.7,
            CyclePhase.DISTRIBUTION: -0.3,
            CyclePhase.MARKDOWN: -0.7,
        }
        val = mapping.get(phase, 0.0)
        # Return constant for all points (phase is global)
        return [val]  # will be broadcasted in composite

    def _discretize(self, composite: List[float],
                     vol_regimes: List[VolRegime]) -> List[int]:
        """
        Convert continuous composite signal to discrete trade signals.
        Threshold adapts to volatility regime.
        """
        thresholds = {
            VolRegime.LOW: 0.15,
            VolRegime.MEDIUM: 0.25,
            VolRegime.HIGH: 0.40,
            VolRegime.EXTREME: 0.60,
        }
        n = len(composite)
        out: List[int] = []
        for i in range(n):
            regime = vol_regimes[i] if i < len(vol_regimes) else VolRegime.MEDIUM
            thresh = thresholds.get(regime, 0.25)
            if composite[i] > thresh:
                out.append(1)
            elif composite[i] < -thresh:
                out.append(-1)
            else:
                out.append(0)
        return out

    def _empty_result(self, n: int) -> Dict:
        zeros = [0.0] * n
        return {
            'composite': zeros,
            'signals': [0] * n,
            'trend': zeros, 'mean_reversion': zeros,
            'momentum': zeros, 'volatility_signal': zeros,
            'forecast_signal': zeros,
            'cycle_signal': zeros,
            'short_ma': zeros, 'long_ma': zeros,
            'rsi': zeros, 'macd_hist': zeros, 'zscore': zeros,
            'bb_upper': zeros, 'bb_lower': zeros,
            'ewma_vol': zeros, 'garch_vol': zeros,
            'vol_regimes': [VolRegime.MEDIUM] * n,
            'cycle_phase': 'accumulation',
            'dominant_cycles': [],
            'holding_period': 5,
        }


