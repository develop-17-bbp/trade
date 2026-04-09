"""
Full Backtest Engine — Replicates Executor Logic
==================================================
Walks through historical bars, applying the exact same signal detection,
entry scoring, risk filters, position management, and trailing SL
as the live trading system.

LLM calls are replaced by entry score threshold (min_entry_score).
Exchange orders are simulated with candle close fills.

When use_ml=True, loads trained ML models and applies the same inference
pipeline as executor.py: LightGBM, LSTM ensemble, Category B risk models
(EVT, Monte Carlo, Hawkes, Temporal Transformer), and ML ensemble voting.
"""

import os
import time
import logging
import numpy as np
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone

from src.backtesting.data_loader import BacktestData, get_context_at_bar
from src.backtesting.signal_generator import compute_tf_signal, compute_indicator_context, compute_entry_score
from src.backtesting.risk_filters import BacktestFilterChain
from src.backtesting.position_manager import (
    BacktestPositionManager, Position, TradeRecord,
    TF_SECONDS,
)
from src.backtesting.metrics import BacktestMetrics

logger = logging.getLogger(__name__)

# ── ML model imports (all optional) ──
try:
    from src.models.lstm_ensemble import LSTMEnsemble
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

try:
    from src.risk.evt_risk import EVTRisk
    EVT_AVAILABLE = True
except Exception:
    EVT_AVAILABLE = False

try:
    from src.risk.monte_carlo_risk import MonteCarloRisk
    MC_AVAILABLE = True
except Exception:
    MC_AVAILABLE = False

try:
    from src.models.hawkes_process import HawkesProcess
    HAWKES_AVAILABLE = True
except Exception:
    HAWKES_AVAILABLE = False

try:
    from src.ai.temporal_transformer import TemporalTransformer
    TFT_AVAILABLE = True
except Exception:
    TFT_AVAILABLE = False


class FullBacktestEngine:
    """Full-fidelity backtester matching executor.py logic."""

    def __init__(self, config: dict = None):
        config = config or {}
        self._config = config  # Store for signal generator

        # Strategy params
        self.ema_period = config.get('ema_period', 8)
        self.min_entry_score = config.get('min_entry_score', 4)
        self.max_entry_score = config.get('max_entry_score', 7)  # Cap high scores (momentum traps)
        self.short_score_penalty = config.get('short_score_penalty', 3)  # Extra score needed for SHORTs
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.risk_per_trade_pct = config.get('risk_per_trade_pct', 2.0)
        self.max_trade_pct = 5.0  # Max 5% of equity per trade

        # Components
        self.filters = BacktestFilterChain(config)
        self.position_mgr = BacktestPositionManager(config)

        # ML models (optional)
        self.use_ml = config.get('use_ml', False)
        self._ml_loaded = False

        # ML model instances
        self._lgbm_raw: Dict[str, Any] = {}   # per-asset LightGBM binary
        self._lstm_per_asset: Dict[str, Any] = {}
        self._evt_risk = None
        self._mc_risk = None
        self._hawkes = None
        self._temporal_transformer = None

        # ML stats tracking
        self.ml_stats = {
            'lgbm_blocks': 0, 'lstm_blocks': 0, 'catb_penalized': 0,
            'ensemble_blocks': 0, 'ml_boosted': 0,
        }

        if self.use_ml:
            self._load_ml_models(config)

        # State
        self.positions: Dict[str, Position] = {}  # asset -> Position
        self.equity = self.initial_capital
        self.cash = self.initial_capital

        # Results
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.signals_generated = 0
        self.entries_attempted = 0

    def _load_ml_models(self, config: dict):
        """Load trained ML models for inference during backtest."""
        import pickle
        assets = [config.get('asset', 'BTC')]  # backtest runs one asset at a time
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), 'models')
        n_features = 50

        # ── LightGBM binary (per-asset) ──
        for asset in assets:
            try:
                import lightgbm as lgb
                model_path = os.path.join(models_dir, f'lgbm_{asset.lower()}_trained.txt')
                if not os.path.exists(model_path):
                    model_path = os.path.join(models_dir, 'lgbm_latest.txt')
                if os.path.exists(model_path):
                    # Fix CRLF
                    with open(model_path, 'rb') as f:
                        raw = f.read(4096)
                    if b'\r\n' in raw:
                        with open(model_path, 'rb') as f:
                            content = f.read()
                        with open(model_path, 'wb') as f:
                            f.write(content.replace(b'\r\n', b'\n'))
                    self._lgbm_raw[asset] = lgb.Booster(model_file=model_path)
                    print(f"  [ML] LightGBM ({asset}) loaded from {os.path.basename(model_path)}")
            except Exception as e:
                print(f"  [ML] LightGBM ({asset}) load failed: {e}")

        # ── LSTM Ensemble (per-asset) ──
        if LSTM_AVAILABLE:
            for asset in assets:
                try:
                    model_dir = os.path.join(models_dir, f'lstm_ensemble_{asset.lower()}')
                    if os.path.exists(model_dir):
                        lstm = LSTMEnsemble(input_dim=n_features, seq_len=30,
                                            num_classes=2, model_dir=model_dir)
                        self._lstm_per_asset[asset] = lstm
                        print(f"  [ML] LSTM Ensemble ({asset}) loaded — {n_features} features, binary SKIP/TRADE")
                except Exception as e:
                    print(f"  [ML] LSTM Ensemble ({asset}) load failed: {e}")

        # ── EVT Tail Risk ──
        if EVT_AVAILABLE:
            try:
                self._evt_risk = EVTRisk(threshold_quantile=0.90, var_level=0.99)
                print(f"  [ML] EVT Tail Risk loaded")
            except Exception as e:
                print(f"  [ML] EVT init failed: {e}")

        # ── Monte Carlo Risk ──
        if MC_AVAILABLE:
            try:
                self._mc_risk = MonteCarloRisk(n_simulations=5000, horizon=24, var_confidence=0.95)
                print(f"  [ML] Monte Carlo Risk loaded")
            except Exception as e:
                print(f"  [ML] Monte Carlo init failed: {e}")

        # ── Hawkes Process ──
        if HAWKES_AVAILABLE:
            try:
                self._hawkes = HawkesProcess()
                print(f"  [ML] Hawkes Process loaded")
            except Exception as e:
                print(f"  [ML] Hawkes init failed: {e}")

        # ── Temporal Transformer ──
        if TFT_AVAILABLE:
            try:
                self._temporal_transformer = TemporalTransformer(d_model=64, n_heads=4, context_len=120)
                print(f"  [ML] Temporal Transformer loaded")
            except Exception as e:
                print(f"  [ML] Temporal Transformer init failed: {e}")

        self._ml_loaded = True
        print(f"  [ML] Model loading complete")

    def run(self, data: BacktestData, verbose: bool = False) -> BacktestMetrics:
        """Run backtest on historical data.

        Args:
            data: BacktestData with multi-timeframe OHLCV
            verbose: print bar-by-bar output

        Returns:
            BacktestMetrics with full analytics
        """
        asset = data.asset
        primary = data.primary
        if not primary or not primary.get('closes'):
            print("  [BACKTEST] ERROR: No primary data")
            return BacktestMetrics([], [], self.initial_capital)

        closes = primary['closes']
        highs = primary['highs']
        lows = primary['lows']
        opens = primary['opens']
        volumes = primary['volumes']
        timestamps = primary['timestamps']
        n_bars = len(closes)

        tf_seconds = TF_SECONDS.get(data.primary_tf, 300)

        print(f"\n  [BACKTEST] Starting: {asset} | {n_bars} bars of {data.primary_tf}")
        print(f"  [BACKTEST] Period: {datetime.fromtimestamp(timestamps[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(timestamps[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        print(f"  [BACKTEST] Capital: ${self.initial_capital:,.0f} | Min score: {self.min_entry_score} | ML: {'ON' if self.use_ml else 'OFF'}")

        start_time = time.time()
        lookback = 100  # Bars of history needed for indicators

        for i in range(lookback, n_bars):
            bar_ts = timestamps[i]

            # Build OHLCV slice up to this bar (no lookahead)
            ohlcv = {
                'opens': opens[max(0, i-lookback):i+1],
                'highs': highs[max(0, i-lookback):i+1],
                'lows': lows[max(0, i-lookback):i+1],
                'closes': closes[max(0, i-lookback):i+1],
                'volumes': volumes[max(0, i-lookback):i+1],
            }

            price = closes[i]

            # === MANAGE EXISTING POSITION ===
            if asset in self.positions:
                pos = self.positions[asset]
                result = self.position_mgr.update_position(pos, i, bar_ts, price, ohlcv)

                if result[0] is None and result[1] is not None:
                    # Position closed
                    trade = result[1]
                    self.trades.append(trade)
                    del self.positions[asset]

                    # Update equity
                    self.equity += trade.pnl_usd
                    self.cash = self.equity

                    is_loss = trade.pnl_pct <= 0
                    self.filters.record_trade_close(asset, bar_ts, is_loss)

                    if verbose:
                        w = "WIN" if trade.pnl_pct > 0 else "LOSS"
                        print(f"  [{i}] CLOSE {trade.direction} {w}: {trade.pnl_pct:+.2f}% | {trade.exit_reason} | {trade.duration_min:.0f}min | SL={trade.max_sl_level}")

                else:
                    self.positions[asset] = result[0]

                self.equity_curve.append(self.equity)
                continue

            # === CHECK FILTERS ===
            passed, reason = self.filters.check_all(asset, bar_ts, ohlcv['closes'])
            if not passed:
                self.equity_curve.append(self.equity)
                continue

            # === COMPUTE SIGNAL (primary TF) ===
            sig = compute_tf_signal(ohlcv, self.ema_period)
            primary_signal = sig['signal']

            if primary_signal == 'NEUTRAL':
                self.equity_curve.append(self.equity)
                continue

            self.signals_generated += 1

            # === MULTI-TF ALIGNMENT (context TFs) ===
            htf_alignment = 0
            for ctx_tf in ['15m', '1h', '4h']:
                ctx_ohlcv = get_context_at_bar(data, ctx_tf, bar_ts, lookback=50)
                if ctx_ohlcv and len(ctx_ohlcv['closes']) >= 20:
                    ctx_sig = compute_tf_signal(ctx_ohlcv, self.ema_period)
                    if ctx_sig['signal'] == primary_signal:
                        htf_alignment += 1
                    elif ctx_sig['ema_direction'] == sig['ema_direction']:
                        htf_alignment += 0.5

            # === COMPUTE INDICATORS + ENTRY SCORE ===
            indicator_ctx = compute_indicator_context(ohlcv)

            # Trendline analysis
            try:
                from src.indicators.trendlines import get_trendline_context
                tl_ctx = get_trendline_context(
                    ohlcv['highs'], ohlcv['lows'], ohlcv['closes'],
                    bar_idx=len(ohlcv['closes']) - 1, timeframe=data.primary_tf
                )
                indicator_ctx.update({
                    'trendline_breakout': tl_ctx.get('trendline_breakout', 0),
                    'trendline_strength': tl_ctx.get('trendline_strength', 0),
                    'trendline_score_adj': tl_ctx.get('trendline_score_adj', 0),
                })
            except Exception:
                pass  # Don't block trades if trendline detection fails

            entry_score, score_reasons = compute_entry_score(
                primary_signal, ohlcv, sig['ema_vals'],
                sig['ema_direction'], sig['price'], indicator_ctx,
                asset=asset, config=self._config
            )

            # HTF alignment bonus
            if htf_alignment >= 2:
                entry_score += 1
                score_reasons.append(f"htf_aligned={htf_alignment:.0f}")

            # === DIRECTION ===
            direction = 'LONG' if primary_signal == 'BUY' else 'SHORT'

            # === ML INFERENCE (mirrors executor.py pipeline) ===
            ml_blocked = False
            if self.use_ml and self._ml_loaded:
                ml_result = self._run_ml_inference(
                    asset, primary_signal, ohlcv, closes[:i+1], highs[:i+1],
                    lows[:i+1], opens[:i+1], volumes[:i+1], price,
                    sig['ema_vals'], sig['atr_vals'],
                )
                if ml_result.get('hard_block'):
                    ml_blocked = True
                else:
                    entry_score += ml_result.get('score_adj', 0)
                    score_reasons.extend(ml_result.get('reasons', []))

            if ml_blocked:
                self.equity_curve.append(self.equity)
                continue

            # === ENTRY GATE (replaces LLM) ===
            effective_min = self.min_entry_score
            if direction == 'SHORT':
                effective_min += self.short_score_penalty
            if entry_score < effective_min or entry_score > self.max_entry_score:
                self.equity_curve.append(self.equity)
                continue

            self.entries_attempted += 1

            # === POSITION SIZING ===
            size_pct = min(self.risk_per_trade_pct, self.max_trade_pct)
            notional = self.equity * (size_pct / 100.0)
            max_trade = min(2000.0, self.equity * 0.05)
            notional = min(notional, max_trade)

            if notional <= 0 or price <= 0:
                self.equity_curve.append(self.equity)
                continue

            qty = notional / price

            # === OPEN POSITION ===
            pos = self.position_mgr.open_position(
                direction=direction,
                price=price,
                ohlcv=ohlcv,
                bar_index=i,
                bar_ts=bar_ts,
                qty=qty,
                entry_score=entry_score,
                timeframe=data.primary_tf,
            )
            self.positions[asset] = pos
            self.filters.record_trade_open(asset, bar_ts)

            if verbose:
                print(f"  [{i}] OPEN {direction} @ ${price:,.2f} | score={entry_score} ({', '.join(score_reasons[:3])}) | SL=${pos.sl:,.2f} | ${notional:,.0f}")

            self.equity_curve.append(self.equity)

            # Progress
            if (i - lookback) % 5000 == 0 and i > lookback:
                pct = (i - lookback) / (n_bars - lookback) * 100
                elapsed = time.time() - start_time
                trades_so_far = len(self.trades)
                wins = sum(1 for t in self.trades if t.pnl_pct > 0)
                wr = wins / trades_so_far if trades_so_far > 0 else 0
                print(f"  [BACKTEST] {pct:.0f}% | {trades_so_far} trades | WR={wr:.1%} | Equity=${self.equity:,.0f} | {elapsed:.0f}s")

        # Close any remaining position at last price
        if asset in self.positions:
            pos = self.positions[asset]
            final_price = closes[-1]
            if pos.direction == 'LONG':
                pnl_pct = ((final_price - pos.entry_price) / pos.entry_price) * 100
            else:
                pnl_pct = ((pos.entry_price - final_price) / pos.entry_price) * 100
            pnl_usd = pnl_pct / 100.0 * pos.entry_price * pos.qty
            self.trades.append(TradeRecord(
                direction=pos.direction, entry_price=pos.entry_price,
                exit_price=final_price, entry_bar=pos.entry_bar,
                exit_bar=n_bars-1, entry_ts=pos.entry_ts,
                exit_ts=timestamps[-1], qty=pos.qty,
                pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                duration_bars=n_bars-1-pos.entry_bar,
                duration_min=((n_bars-1-pos.entry_bar) * tf_seconds) / 60,
                exit_reason="End of backtest",
                max_sl_level=pos.sl_levels[-1],
                entry_score=pos.entry_score,
                sl_levels_hit=list(pos.sl_levels),
            ))
            self.equity += pnl_usd

        elapsed = time.time() - start_time
        print(f"\n  [BACKTEST] Complete in {elapsed:.1f}s | {len(self.trades)} trades | Signals: {self.signals_generated} | Entries: {self.entries_attempted}")

        # Filter stats
        fstats = self.filters.get_stats()
        if any(v > 0 for v in fstats.values()):
            print(f"  [FILTERS] {fstats}")

        # ML stats
        if self.use_ml and any(v > 0 for v in self.ml_stats.values()):
            print(f"  [ML STATS] {self.ml_stats}")

        return BacktestMetrics(self.trades, self.equity_curve, self.initial_capital)

    # ──────────────────────────────────────────────────────────────
    # ML INFERENCE PIPELINE (mirrors executor.py _evaluate_entry)
    # ──────────────────────────────────────────────────────────────
    def _run_ml_inference(self, asset: str, signal: str, ohlcv: dict,
                          closes: list, highs: list, lows: list,
                          opens: list, volumes: list, price: float,
                          ema_vals: list, atr_vals: list) -> dict:
        """Run full ML inference pipeline matching executor.py.

        Returns dict with:
            hard_block: bool — if True, skip entry entirely
            score_adj: int — total score adjustment from ML
            reasons: list[str] — score reason strings
        """
        score_adj = 0
        reasons = []
        ml_context = {}

        # ── LSTM ENSEMBLE PREDICTION (binary SKIP/TRADE) ──
        _lstm = self._lstm_per_asset.get(asset)
        if _lstm and len(closes) >= 80:
            try:
                from src.scripts.train_all_models import compute_strategy_features
                X_seq, _ = compute_strategy_features(
                    closes, highs, lows, opens, volumes,
                    seq_len=30, n_features=50
                )
                if X_seq is not None and len(X_seq) > 0:
                    lstm_pred = _lstm.predict(X_seq[-1])
                    if lstm_pred and lstm_pred.get('confidence', 0) > 0.1:
                        trade_quality = lstm_pred.get('trade_quality', 'UNKNOWN')
                        trade_conf = lstm_pred.get('confidence', 0)
                        ml_context['lstm_signal'] = trade_quality
                        ml_context['lstm_confidence'] = trade_conf

                        if trade_quality == 'TRADE' and trade_conf > 0.55:
                            score_adj += 2
                            reasons.append(f"lstm_TRADE({trade_conf:.0%})")
                        elif trade_quality == 'TRADE' and trade_conf > 0.40:
                            score_adj += 1
                            reasons.append(f"lstm_trade_weak({trade_conf:.0%})")
                        elif trade_quality == 'SKIP' and trade_conf > 0.75:
                            self.ml_stats['lstm_blocks'] += 1
                            return {'hard_block': True, 'score_adj': 0, 'reasons': [f"lstm_HARD_SKIP({trade_conf:.0%})"]}
                        elif trade_quality == 'SKIP' and trade_conf > 0.60:
                            score_adj -= 3
                            reasons.append(f"lstm_SKIP({trade_conf:.0%})")
                        elif trade_quality == 'SKIP' and trade_conf > 0.40:
                            score_adj -= 1
                            reasons.append(f"lstm_skip_weak({trade_conf:.0%})")
            except Exception as e:
                logger.debug(f"LSTM prediction error: {e}")

        # ── LIGHTGBM BINARY GATE ──
        _lgbm = self._lgbm_raw.get(asset)
        if _lgbm and len(closes) >= 55:
            try:
                from src.scripts.train_all_models import compute_strategy_features
                X_seq, _ = compute_strategy_features(
                    closes, highs, lows, opens, volumes,
                    seq_len=1, n_features=50
                )
                if X_seq is not None and len(X_seq) > 0:
                    feat = X_seq[-1].reshape(1, -1)  # All 50 features
                    # Align feature count with trained model
                    expected = _lgbm.num_feature()
                    if feat.shape[1] > expected:
                        feat = feat[:, :expected]
                    elif feat.shape[1] < expected:
                        pad = np.zeros((feat.shape[0], expected - feat.shape[1]))
                        feat = np.hstack([feat, pad])
                    trade_prob = _lgbm.predict(feat, predict_disable_shape_check=True)[0]
                    trade_conf = float(trade_prob)
                    is_trade = trade_conf > 0.5

                    ml_context['lgbm_direction'] = 'TRADE' if is_trade else 'SKIP'
                    ml_context['lgbm_confidence'] = trade_conf if is_trade else (1 - trade_conf)

                    if is_trade and trade_conf > 0.60:
                        score_adj += 2
                        reasons.append(f"lgbm_TRADE({trade_conf:.0%})")
                    elif is_trade and trade_conf > 0.45:
                        score_adj += 1
                        reasons.append(f"lgbm_trade_weak({trade_conf:.0%})")
                    elif not is_trade and (1 - trade_conf) > 0.75:
                        self.ml_stats['lgbm_blocks'] += 1
                        return {'hard_block': True, 'score_adj': 0, 'reasons': [f"lgbm_HARD_SKIP({1-trade_conf:.0%})"]}
                    elif not is_trade and (1 - trade_conf) > 0.60:
                        score_adj -= 3
                        reasons.append(f"lgbm_SKIP({1-trade_conf:.0%})")
                    elif not is_trade and (1 - trade_conf) > 0.45:
                        score_adj -= 1
                        reasons.append(f"lgbm_skip_weak({1-trade_conf:.0%})")
            except Exception as e:
                logger.debug(f"LightGBM prediction error: {e}")

        # ── TEMPORAL TRANSFORMER FORECAST ──
        _tft_bps = 0.0
        _tft_conf = 0.0
        if self._temporal_transformer and len(closes) >= 120:
            try:
                _c = np.array(closes[-120:], dtype=float)
                _h = np.array(highs[-120:], dtype=float)
                _l = np.array(lows[-120:], dtype=float)
                _v = np.array(volumes[-120:], dtype=float)
                _pct = np.diff(_c) / _c[:-1]
                _hpct = (_h[1:] - _c[:-1]) / _c[:-1]
                _lpct = (_l[1:] - _c[:-1]) / _c[:-1]
                _vpct = np.diff(_v) / (_v[:-1] + 1e-12)
                _history = np.column_stack([_pct, _hpct, _lpct, _vpct])[-self._temporal_transformer.context_len:]
                if _history.shape[1] < self._temporal_transformer.d_model:
                    _pad = np.zeros((_history.shape[0], self._temporal_transformer.d_model - _history.shape[1]))
                    _history = np.hstack([_history, _pad])
                forecast = self._temporal_transformer.forecast_return(_history)
                _tft_bps = forecast.get('forecast_return_bps', 0)
                _tft_conf = forecast.get('confidence', 0)
                ml_context['tft_forecast_bps'] = _tft_bps
                ml_context['tft_confidence'] = _tft_conf
            except Exception as e:
                logger.debug(f"TFT error: {e}")

        # ── HAWKES PROCESS: event clustering ──
        _hawkes_intensity = 0.0
        if self._hawkes and len(closes) >= 50:
            try:
                _rets = np.abs(np.diff(np.array(closes[-100:], dtype=float)) / np.array(closes[-100:-1], dtype=float))
                _mean_r = np.mean(_rets)
                _std_r = np.std(_rets)
                _threshold = _mean_r + 2.0 * _std_r
                _event_indices = np.where(_rets > _threshold)[0]
                if len(_event_indices) >= 3:
                    _event_times = _event_indices.astype(float)
                    intensity = self._hawkes.current_intensity(_event_times)
                    _hawkes_intensity = float(intensity)
                    ml_context['hawkes_intensity'] = _hawkes_intensity
            except Exception as e:
                logger.debug(f"Hawkes error: {e}")

        # ── EVT TAIL RISK ──
        _evt_var = 0.0
        _evt_tail_ratio = 1.0
        if self._evt_risk and len(closes) >= 100:
            try:
                _rets = np.diff(np.log(np.array(closes[-500:], dtype=float) + 1e-12))
                if len(_rets) >= 50:
                    evt_result = self._evt_risk.fit_and_assess(_rets)
                    _evt_var = evt_result.get('evt_var_99', 0)
                    _evt_tail_ratio = evt_result.get('tail_ratio', 1.0)
                    ml_context['evt_var_99'] = _evt_var
                    ml_context['evt_tail_ratio'] = _evt_tail_ratio
            except Exception as e:
                logger.debug(f"EVT error: {e}")

        # ── MONTE CARLO RISK ──
        _mc_risk_score = 0.5
        if self._mc_risk and len(closes) >= 50 and price > 0:
            try:
                _rets = np.diff(np.array(closes[-100:], dtype=float)) / np.array(closes[-100:-1], dtype=float)
                _vol = float(np.std(_rets))
                _drift = float(np.mean(_rets))
                mc_result = self._mc_risk.simulate(current_price=price, volatility=_vol, drift=_drift, regime='normal')
                _mc_risk_score = mc_result.get('mc_risk_score', 0.5)
                ml_context['mc_risk_score'] = _mc_risk_score
            except Exception as e:
                logger.debug(f"MC error: {e}")

        # ── CATEGORY B DIRECT SCORE MODIFIERS ──
        # EVT: extreme tail risk
        if _evt_var < -0.08:
            score_adj -= 2
            reasons.append(f"evt_extreme_tail({_evt_var:.3f})")
            self.ml_stats['catb_penalized'] += 1
        elif _evt_tail_ratio > 2.0:
            score_adj -= 1
            reasons.append(f"evt_fat_tail({_evt_tail_ratio:.1f}x)")
            self.ml_stats['catb_penalized'] += 1

        # Monte Carlo: high forward risk
        if _mc_risk_score > 0.8:
            score_adj -= 2
            reasons.append(f"mc_high_risk({_mc_risk_score:.2f})")
            self.ml_stats['catb_penalized'] += 1
        elif _mc_risk_score > 0.6:
            score_adj -= 1
            reasons.append(f"mc_elevated_risk({_mc_risk_score:.2f})")

        # Hawkes: event clustering
        if _hawkes_intensity > 0.8:
            score_adj -= 2
            reasons.append(f"hawkes_clustering({_hawkes_intensity:.2f})")
            self.ml_stats['catb_penalized'] += 1
        elif _hawkes_intensity > 0.5:
            score_adj -= 1
            reasons.append(f"hawkes_elevated({_hawkes_intensity:.2f})")

        # TFT: direction conflict
        if _tft_conf > 0.3:
            _tft_bullish = _tft_bps > 0
            _signal_bullish = signal == 'BUY'
            if _tft_bullish != _signal_bullish and abs(_tft_bps) > 5:
                score_adj -= 1
                reasons.append(f"tft_disagree({_tft_bps:+.0f}bps)")
            elif _tft_bullish == _signal_bullish and abs(_tft_bps) > 10:
                score_adj += 1
                reasons.append(f"tft_confirm({_tft_bps:+.0f}bps)")
                self.ml_stats['ml_boosted'] += 1

        # ── ML ENSEMBLE VOTE: 3+ SKIP = hard block, 2+ = score -3 ──
        ml_skip_votes = []
        if ml_context.get('lgbm_direction') == 'SKIP' and ml_context.get('lgbm_confidence', 0) > 0.55:
            ml_skip_votes.append("LGBM")
        if ml_context.get('lstm_signal') == 'SKIP' and ml_context.get('lstm_confidence', 0) > 0.55:
            ml_skip_votes.append("LSTM")
        if _hawkes_intensity > 0.8:
            ml_skip_votes.append("Hawkes")
        if _mc_risk_score > 0.8:
            ml_skip_votes.append("MC")
        if _evt_var < -0.08:
            ml_skip_votes.append("EVT")
        if _tft_conf > 0.4 and ((_tft_bps > 0) != (signal == 'BUY')) and abs(_tft_bps) > 15:
            ml_skip_votes.append("TFT")

        if len(ml_skip_votes) >= 3:
            self.ml_stats['ensemble_blocks'] += 1
            return {'hard_block': True, 'score_adj': 0, 'reasons': [f"ml_consensus_block({','.join(ml_skip_votes)})"]}
        elif len(ml_skip_votes) >= 2:
            score_adj -= 3
            reasons.append(f"ml_consensus_skip({len(ml_skip_votes)})")

        if score_adj > 0:
            self.ml_stats['ml_boosted'] += 1

        return {'hard_block': False, 'score_adj': score_adj, 'reasons': reasons}
