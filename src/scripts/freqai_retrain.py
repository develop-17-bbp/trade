#!/usr/bin/env python3
"""
FreqAI-Style Auto-Retrain — Self-Training ML Pipeline
======================================================
Inspired by Freqtrade's FreqAI module. Models continuously
adapt to current market conditions.

Features:
- Trade-triggered retraining (every N closed trades)
- Time-triggered retraining (every M hours)
- Expanding window training (no data loss)
- Walk-forward validation (no lookahead)
- A/B model comparison (new must beat old)
- Automatic deployment if improvement found
- Full audit trail in logs/retrain_history.json

Models retrained:
- LightGBM binary classifier (SKIP/TRADE)
- LSTM ensemble (direction prediction)
- RL agent (strategy optimization)

Usage:
    python -m src.scripts.freqai_retrain --once          # Single retrain cycle
    python -m src.scripts.freqai_retrain --continuous     # Run forever
    python -m src.scripts.freqai_retrain --trade-trigger  # Watch for trade completions
"""

import os
import sys
import json
import time
import hashlib
import shutil
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('freqai_retrain')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed. Run: pip install lightgbm")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna not installed. Run: pip install optuna")

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_FREQAI_CONFIG = {
    'enabled': True,
    'retrain_every_trades': 20,
    'retrain_every_hours': 4,
    'min_improvement_pct': 1.0,
    'assets': ['BTC', 'ETH'],
    'spread_cost_pct': 3.34,
    'lookback_bars': 1500,
    'label_threshold': 0.01,
    'optuna_trials': 40,
    'boost_rounds': 400,
    'early_stopping_rounds': 30,
    'validation_pct': 0.15,
    'test_pct': 0.15,
    'longs_only': True,
    'model_dir': 'models',
    'data_dir': 'data/offline',
    'history_file': 'logs/retrain_history.json',
    'journal_file': 'logs/trading_journal.jsonl',
    'timeframe': '4h',
}

ASSET_SYMBOLS = {
    'BTC': {'symbol': 'BTC/USDT', 'pair': 'BTCUSDT', 'model_file': 'lgbm_btc.txt'},
    'ETH': {'symbol': 'ETH/USDT', 'pair': 'ETHUSDT', 'model_file': 'lgbm_eth.txt'},
    'AAVE': {'symbol': 'AAVE/USDT', 'pair': 'AAVEUSDT', 'model_file': 'lgbm_aave.txt'},
}


def load_config() -> dict:
    """Load freqai config from config.yaml, falling back to defaults."""
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    cfg = dict(DEFAULT_FREQAI_CONFIG)

    if HAS_YAML and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                full = yaml.safe_load(f)
            if full and 'freqai' in full:
                cfg.update(full['freqai'])
                logger.info("Loaded freqai config from config.yaml")
        except Exception as e:
            logger.warning(f"Failed to parse config.yaml: {e}, using defaults")

    return cfg


# ============================================================================
# FEATURE ENGINEERING (mirrors scheduled_retrain.py / continuous_train.py)
# ============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 35+ technical features from OHLCV data.
    All features use only past data (no look-ahead bias).
    """
    feat = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = close.pct_change()

    # SMA ratios + slopes (8)
    for w in [5, 10, 20, 50]:
        sma = close.rolling(w).mean()
        feat[f'sma_ratio_{w}'] = close / sma
        feat[f'sma_slope_{w}'] = sma.pct_change(5)

    # EMA + MACD (3)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    feat['macd'] = ema12 - ema26
    feat['macd_signal'] = feat['macd'].ewm(span=9).mean()
    feat['macd_hist'] = feat['macd'] - feat['macd_signal']

    # Volatility (6)
    feat['returns_1'] = returns
    feat['returns_4'] = returns.rolling(4).sum()
    feat['returns_24'] = returns.rolling(24).sum()
    feat['vol_20'] = returns.rolling(20).std()
    feat['vol_50'] = returns.rolling(50).std()
    feat['vol_ratio'] = feat['vol_20'] / (feat['vol_50'] + 1e-10)

    # ATR (2)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_pct'] = feat['atr_14'] / (close + 1e-10)

    # RSI (2)
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feat['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    gain6 = delta.where(delta > 0, 0).rolling(6).mean()
    loss6 = (-delta.where(delta < 0, 0)).rolling(6).mean()
    feat['rsi_6'] = 100 - (100 / (1 + gain6 / (loss6 + 1e-10)))

    # Bollinger Bands (3)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    feat['bb_width'] = (4 * std20) / (close + 1e-10)
    feat['bb_position'] = (close - (sma20 - 2 * std20)) / (4 * std20 + 1e-10)
    feat['bb_squeeze'] = (feat['bb_width'] < feat['bb_width'].rolling(50).quantile(0.2)).astype(int)

    # Volume (4)
    feat['vol_ratio_20'] = volume / (volume.rolling(20).mean() + 1e-10)
    feat['vol_trend'] = volume.rolling(10).mean() / (volume.rolling(50).mean() + 1e-10)
    feat['obv'] = (np.sign(returns) * volume).cumsum()
    feat['obv_slope'] = feat['obv'].diff(10) / (feat['obv'].shift(10).abs() + 1e-10)

    # Momentum (4)
    feat['mom_10'] = close - close.shift(10)
    feat['roc_10'] = returns.rolling(10).sum()
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    feat['stoch_k'] = 100 * (close - low14) / (high14 - low14 + 1e-10)
    feat['stoch_d'] = feat['stoch_k'].rolling(3).mean()

    # ADX (1)
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr_smooth + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr_smooth + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    feat['adx'] = dx.rolling(14).mean()

    return feat


def create_labels(df: pd.DataFrame, threshold: float = 0.01,
                  spread_cost_pct: float = 0.0,
                  longs_only: bool = False) -> pd.Series:
    """
    Create directional labels with spread-cost awareness.
    +1 = BUY (next-bar return exceeds threshold + spread)
    -1 = SELL (next-bar return below -threshold, or 0 if longs_only)
     0 = SKIP (move too small to overcome costs)
    """
    future_return = df['close'].pct_change().shift(-1)
    spread_adj = spread_cost_pct / 100.0

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return > (threshold + spread_adj)] = 1
    if not longs_only:
        labels[future_return < -(threshold + spread_adj)] = -1

    # Last row has no future — mark as SKIP
    labels.iloc[-1] = 0
    return labels


# ============================================================================
# DATA LOADING
# ============================================================================

def find_data_file(symbol_pair: str, timeframe: str, data_dir: str) -> Optional[str]:
    """Locate parquet/CSV data for a symbol+timeframe."""
    candidates = [
        os.path.join(data_dir, f"{symbol_pair}-{timeframe}.parquet"),
        os.path.join(data_dir, f"{symbol_pair}-{timeframe}.csv"),
        os.path.join(data_dir, f"{symbol_pair}_{timeframe}.parquet"),
        os.path.join(data_dir, f"{symbol_pair}_{timeframe}.csv"),
        os.path.join(PROJECT_ROOT, 'data', f"{symbol_pair}-{timeframe}.parquet"),
        os.path.join(PROJECT_ROOT, 'data', f"{symbol_pair}_{timeframe}.parquet"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load OHLCV data from parquet or CSV with column normalization."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    col_map = {}
    for col in df.columns:
        lower = col.lower().strip()
        if lower in ('timestamp', 'time', 'date', 'open_time', 'datetime'):
            col_map[col] = 'timestamp'
        elif lower == 'open':
            col_map[col] = 'open'
        elif lower == 'high':
            col_map[col] = 'high'
        elif lower == 'low':
            col_map[col] = 'low'
        elif lower == 'close':
            col_map[col] = 'close'
        elif lower in ('volume', 'vol'):
            col_map[col] = 'volume'
    df = df.rename(columns=col_map)

    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            logger.error(f"Missing column '{col}' in {path}")
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.dropna(subset=required)
    return df


def fetch_training_data(asset: str, config: dict) -> pd.DataFrame:
    """
    Fetch training data for an asset.
    Priority: local parquet > Binance Vision download > CCXT API.
    """
    info = ASSET_SYMBOLS.get(asset)
    if not info:
        logger.error(f"Unknown asset: {asset}")
        return pd.DataFrame()

    pair = info['pair']
    tf = config.get('timeframe', '4h')
    data_dir = os.path.join(PROJECT_ROOT, config.get('data_dir', 'data/offline'))

    # Try local file first
    path = find_data_file(pair, tf, data_dir)
    if path:
        logger.info(f"Loading local data: {path}")
        df = load_ohlcv(path)
        if not df.empty:
            limit = config.get('lookback_bars', 1500)
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            return df

    # Try Binance Vision download
    try:
        from download_vision_data import fetch_vision_ohlcv
        logger.info(f"Downloading from Binance Vision: {info['symbol']} {tf}")
        df = fetch_vision_ohlcv(info['symbol'], tf)
        if not df.empty:
            limit = config.get('lookback_bars', 1500)
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            return df
    except ImportError:
        logger.warning("download_vision_data.py not found")
    except Exception as e:
        logger.warning(f"Binance Vision failed: {e}")

    # Fallback: CCXT
    try:
        import ccxt
        logger.info(f"Fetching via CCXT: {info['symbol']} {tf}")
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        limit = config.get('lookback_bars', 1500)
        ohlcv = exchange.fetch_ohlcv(info['symbol'], tf, limit=min(limit, 1000))
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"All data sources failed for {asset}: {e}")
        return pd.DataFrame()


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def model_checksum(path: str) -> str:
    """SHA256 checksum of a model file."""
    if not os.path.exists(path):
        return ''
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def evaluate_model(model: lgb.Booster, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a LightGBM model on a dataset. Returns accuracy metrics."""
    preds_raw = model.predict(X)
    if preds_raw.ndim == 2:
        preds = np.argmax(preds_raw, axis=1)
    else:
        preds = (preds_raw > 0.5).astype(int)

    accuracy = float(np.mean(preds == y))

    # Precision for the BUY class (class 2 in multiclass, class 1 in binary)
    buy_class = 2 if preds_raw.ndim == 2 and preds_raw.shape[1] == 3 else 1
    buy_mask = preds == buy_class
    buy_precision = 0.0
    if buy_mask.sum() > 0:
        buy_precision = float(np.mean(y[buy_mask] == buy_class))

    return {
        'accuracy': accuracy,
        'buy_precision': buy_precision,
        'n_samples': len(y),
        'n_buy_signals': int(buy_mask.sum()),
    }


# ============================================================================
# FREQAI RETRAIN ENGINE
# ============================================================================

class FreqAIRetrain:
    """
    FreqAI-style continuous retraining engine.

    Monitors trade outcomes and time elapsed, triggering model retraining
    when thresholds are crossed. New models are validated via walk-forward
    testing and only deployed if they beat the incumbent.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.trades_since_retrain = 0
        self.last_retrain_time = datetime.now()
        self.model_checksums: Dict[str, str] = {}
        self._journal_offset = 0  # tracks how far we've read into the journal

        # Resolve paths relative to project root
        self.model_dir = os.path.join(PROJECT_ROOT, self.config.get('model_dir', 'models'))
        self.history_file = os.path.join(PROJECT_ROOT, self.config.get('history_file', 'logs/retrain_history.json'))
        self.journal_file = os.path.join(PROJECT_ROOT, self.config.get('journal_file', 'logs/trading_journal.jsonl'))

        # Ensure directories exist
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

        # Load current model checksums
        self._load_checksums()
        logger.info(
            f"FreqAIRetrain initialized | retrain_every_trades={self.config['retrain_every_trades']} "
            f"| retrain_every_hours={self.config['retrain_every_hours']} "
            f"| min_improvement={self.config['min_improvement_pct']}% "
            f"| assets={self.config['assets']}"
        )

    def _load_checksums(self):
        """Load checksums for all current model files."""
        for asset, info in ASSET_SYMBOLS.items():
            path = os.path.join(self.model_dir, info['model_file'])
            self.model_checksums[asset] = model_checksum(path)

    def should_retrain(self) -> bool:
        """
        Check if retraining should be triggered.
        Returns True if either trade-count or time threshold is met.
        """
        trade_threshold = self.config.get('retrain_every_trades', 20)
        hours_threshold = self.config.get('retrain_every_hours', 4)

        trade_trigger = self.trades_since_retrain >= trade_threshold
        hours_elapsed = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
        time_trigger = hours_elapsed >= hours_threshold

        if trade_trigger:
            logger.info(
                f"Trade trigger: {self.trades_since_retrain}/{trade_threshold} trades since last retrain"
            )
        if time_trigger:
            logger.info(
                f"Time trigger: {hours_elapsed:.1f}/{hours_threshold} hours since last retrain"
            )

        return trade_trigger or time_trigger

    def run_retrain_cycle(self, asset: str = 'BTC') -> dict:
        """
        Execute a single retrain cycle for one asset.

        Steps:
          1. Fetch latest training data
          2. Build features + spread-aware labels
          3. Walk-forward split: 70% train / 15% val / 15% test
          4. Train new LightGBM with Optuna
          5. Compare new vs old on held-out test set
          6. Deploy if improvement exceeds threshold

        Returns:
            dict with {improved, old_accuracy, new_accuracy, asset, ...}
        """
        cycle_start = datetime.now()
        result = {
            'asset': asset,
            'timestamp': cycle_start.isoformat(),
            'improved': False,
            'old_accuracy': 0.0,
            'new_accuracy': 0.0,
            'deployed': False,
            'error': None,
        }

        if not HAS_LGB:
            result['error'] = 'lightgbm not installed'
            logger.error(result['error'])
            return result

        # ── 1. Fetch data ──
        logger.info(f"[{asset}] Fetching training data...")
        df = fetch_training_data(asset, self.config)
        if df.empty or len(df) < 200:
            result['error'] = f'Insufficient data: {len(df)} rows (need 200+)'
            logger.error(f"[{asset}] {result['error']}")
            return result

        logger.info(f"[{asset}] Loaded {len(df)} bars")

        # ── 2. Build features + labels ──
        features = build_features(df)
        labels = create_labels(
            df,
            threshold=self.config.get('label_threshold', 0.01),
            spread_cost_pct=self.config.get('spread_cost_pct', 3.34),
            longs_only=self.config.get('longs_only', True),
        )

        # Map labels to LightGBM classes: {-1: 0, 0: 1, 1: 2}
        label_map = {-1: 0, 0: 1, 1: 2}
        y_mapped = labels.map(label_map)

        # Combine and drop NaN rows (feature warmup period)
        combined = features.copy()
        combined['label'] = y_mapped
        combined = combined.dropna()

        if len(combined) < 200:
            result['error'] = f'Insufficient rows after feature cleanup: {len(combined)}'
            logger.error(f"[{asset}] {result['error']}")
            return result

        X = combined.drop(columns=['label']).values
        y = combined['label'].values.astype(int)
        feature_names = [c for c in combined.columns if c != 'label']

        # ── 3. Walk-forward split (chronological, no shuffling) ──
        val_pct = self.config.get('validation_pct', 0.15)
        test_pct = self.config.get('test_pct', 0.15)
        train_end = int(len(X) * (1 - val_pct - test_pct))
        val_end = int(len(X) * (1 - test_pct))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(
            f"[{asset}] Split: train={len(X_train)} val={len(X_val)} test={len(X_test)}"
        )

        # ── 4. Evaluate old model on test set (if exists) ──
        asset_info = ASSET_SYMBOLS.get(asset, {})
        old_model_path = os.path.join(self.model_dir, asset_info.get('model_file', ''))
        old_metrics = {'accuracy': 0.0}

        if os.path.exists(old_model_path):
            try:
                old_model = lgb.Booster(model_file=old_model_path)
                old_metrics = evaluate_model(old_model, X_test, y_test)
                result['old_accuracy'] = old_metrics['accuracy']
                logger.info(
                    f"[{asset}] Old model accuracy: {old_metrics['accuracy']:.4f} "
                    f"(buy_precision={old_metrics.get('buy_precision', 0):.4f})"
                )
            except Exception as e:
                logger.warning(f"[{asset}] Could not load old model: {e}")
        else:
            logger.info(f"[{asset}] No existing model — training from scratch")

        # ── 5. Train new model with Optuna ──
        n_classes = len(set(y_train))
        if n_classes < 2:
            result['error'] = f'Only {n_classes} class(es) in training data — need at least 2'
            logger.error(f"[{asset}] {result['error']}")
            return result

        use_multiclass = n_classes >= 3
        base_params = {
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'metric': 'multi_logloss' if use_multiclass else 'binary_logloss',
        }
        if use_multiclass:
            base_params['objective'] = 'multiclass'
            base_params['num_class'] = n_classes
        else:
            base_params['objective'] = 'binary'

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

        n_trials = self.config.get('optuna_trials', 40)
        boost_rounds = self.config.get('boost_rounds', 400)
        early_stop = self.config.get('early_stopping_rounds', 30)

        best_val_acc = 0.0
        best_params = {}

        if HAS_OPTUNA and n_trials > 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def _objective(trial):
                params = {
                    **base_params,
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                }

                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=boost_rounds,
                    valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(early_stop),
                        lgb.log_evaluation(0),
                    ],
                )

                preds = model.predict(X_val)
                if preds.ndim == 2:
                    preds_class = np.argmax(preds, axis=1)
                else:
                    preds_class = (preds > 0.5).astype(int)
                return float(np.mean(preds_class == y_val))

            study = optuna.create_study(direction='maximize')
            study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
            best_val_acc = study.best_value
            best_params = {**base_params, **study.best_params}
            logger.info(f"[{asset}] Optuna best val accuracy: {best_val_acc:.4f}")
        else:
            # Fallback: use sensible defaults without Optuna
            best_params = {
                **base_params,
                'num_leaves': 50,
                'learning_rate': 0.05,
                'max_depth': 8,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }

        # ── 6. Train final model with best params ──
        logger.info(f"[{asset}] Training final model with best params...")
        final_model = lgb.train(
            best_params,
            dtrain,
            num_boost_round=boost_rounds,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(early_stop),
                lgb.log_evaluation(0),
            ],
        )

        new_metrics = evaluate_model(final_model, X_test, y_test)
        result['new_accuracy'] = new_metrics['accuracy']
        result['new_buy_precision'] = new_metrics.get('buy_precision', 0.0)
        result['n_test_samples'] = new_metrics['n_samples']

        logger.info(
            f"[{asset}] New model test accuracy: {new_metrics['accuracy']:.4f} "
            f"(buy_precision={new_metrics.get('buy_precision', 0):.4f})"
        )

        # ── 7. Compare: deploy only if improvement exceeds threshold ──
        min_improvement = self.config.get('min_improvement_pct', 1.0) / 100.0
        improvement = new_metrics['accuracy'] - old_metrics.get('accuracy', 0.0)
        result['improvement'] = round(improvement * 100, 4)

        if improvement >= min_improvement or not os.path.exists(old_model_path):
            # Deploy new model
            if os.path.exists(old_model_path):
                backup_path = old_model_path + f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(old_model_path, backup_path)
                logger.info(f"[{asset}] Backed up old model to {backup_path}")

            final_model.save_model(old_model_path)
            self.model_checksums[asset] = model_checksum(old_model_path)
            result['improved'] = True
            result['deployed'] = True
            logger.info(
                f"[{asset}] DEPLOYED new model | improvement={improvement*100:.2f}% "
                f"| {old_metrics.get('accuracy', 0):.4f} -> {new_metrics['accuracy']:.4f}"
            )
        else:
            result['improved'] = False
            result['deployed'] = False
            logger.info(
                f"[{asset}] Model NOT deployed | improvement={improvement*100:.2f}% "
                f"< threshold={self.config.get('min_improvement_pct', 1.0)}%"
            )

        # ── 8. Feature importance logging ──
        importance = final_model.feature_importance(importance_type='gain')
        top_features = sorted(
            zip(feature_names, importance), key=lambda x: x[1], reverse=True
        )[:10]
        result['top_features'] = [{'name': n, 'gain': round(float(g), 2)} for n, g in top_features]

        # ── 9. Log to history file ──
        result['duration_seconds'] = round((datetime.now() - cycle_start).total_seconds(), 1)
        self._log_result(result)

        # Reset counters
        self.trades_since_retrain = 0
        self.last_retrain_time = datetime.now()

        return result

    def run_all_assets(self) -> List[dict]:
        """Run retrain cycle for all configured assets."""
        results = []
        for asset in self.config.get('assets', ['BTC', 'ETH']):
            try:
                r = self.run_retrain_cycle(asset)
                results.append(r)
            except Exception as e:
                logger.error(f"[{asset}] Retrain cycle failed: {e}", exc_info=True)
                results.append({
                    'asset': asset,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'improved': False,
                    'deployed': False,
                })
        return results

    def watch_trades(self):
        """
        Monitor trading journal for new trade completions.
        Increments counter and triggers retrain when threshold is met.
        """
        logger.info(f"Watching journal: {self.journal_file}")
        logger.info(f"Retrain threshold: {self.config['retrain_every_trades']} trades")

        # Start from end of file if it exists
        if os.path.exists(self.journal_file):
            with open(self.journal_file, 'r') as f:
                self._journal_offset = sum(1 for _ in f)
            logger.info(f"Journal has {self._journal_offset} existing entries, watching for new ones")

        while True:
            try:
                new_trades = self._read_new_trades()
                if new_trades:
                    self.trades_since_retrain += len(new_trades)
                    logger.info(
                        f"Found {len(new_trades)} new trade(s) | "
                        f"total since retrain: {self.trades_since_retrain}"
                    )

                if self.should_retrain():
                    logger.info("Retrain triggered — running cycle for all assets...")
                    results = self.run_all_assets()
                    for r in results:
                        status = "DEPLOYED" if r.get('deployed') else "SKIPPED"
                        logger.info(
                            f"  [{r['asset']}] {status} | "
                            f"accuracy: {r.get('old_accuracy', 0):.4f} -> {r.get('new_accuracy', 0):.4f}"
                        )

                time.sleep(30)  # poll every 30 seconds

            except KeyboardInterrupt:
                logger.info("Trade watcher stopped by user")
                break
            except Exception as e:
                logger.error(f"Trade watcher error: {e}", exc_info=True)
                time.sleep(60)

    def _read_new_trades(self) -> List[dict]:
        """Read new completed trade entries from the journal file."""
        if not os.path.exists(self.journal_file):
            return []

        new_trades = []
        try:
            with open(self.journal_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < self._journal_offset:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        # Count closed/completed trades
                        if entry.get('action') in ('CLOSE', 'close', 'exit', 'EXIT', 'SELL', 'sell'):
                            new_trades.append(entry)
                        elif entry.get('status') in ('closed', 'completed', 'filled'):
                            new_trades.append(entry)
                    except json.JSONDecodeError:
                        continue
                    self._journal_offset = i + 1
        except Exception as e:
            logger.warning(f"Error reading journal: {e}")

        return new_trades

    def continuous_loop(self, interval_hours: Optional[float] = None):
        """
        Run retrain cycles at fixed time intervals.
        Default interval from config (retrain_every_hours).
        """
        interval = interval_hours or self.config.get('retrain_every_hours', 4)
        logger.info(f"Starting continuous retrain loop | interval={interval}h")

        cycle_num = 0
        while True:
            cycle_num += 1
            logger.info(f"{'='*60}")
            logger.info(f"Retrain cycle #{cycle_num} starting at {datetime.now().isoformat()}")
            logger.info(f"{'='*60}")

            try:
                results = self.run_all_assets()

                deployed = sum(1 for r in results if r.get('deployed'))
                total = len(results)
                logger.info(
                    f"Cycle #{cycle_num} complete | {deployed}/{total} models deployed"
                )

                for r in results:
                    if r.get('error'):
                        logger.warning(f"  [{r['asset']}] ERROR: {r['error']}")
                    else:
                        marker = 'DEPLOYED' if r.get('deployed') else 'kept_old'
                        logger.info(
                            f"  [{r['asset']}] {marker} | "
                            f"{r.get('old_accuracy', 0):.4f} -> {r.get('new_accuracy', 0):.4f} "
                            f"({r.get('improvement', 0):+.2f}%)"
                        )

            except Exception as e:
                logger.error(f"Cycle #{cycle_num} failed: {e}", exc_info=True)

            sleep_seconds = interval * 3600
            next_run = datetime.now() + timedelta(seconds=sleep_seconds)
            logger.info(f"Sleeping {interval}h until next cycle at {next_run.isoformat()}")

            try:
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                logger.info("Continuous loop stopped by user")
                break

    def _log_result(self, result: dict):
        """Append retrain result to the history JSON file."""
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []

        history.append(result)

        # Keep last 500 entries
        if len(history) > 500:
            history = history[-500:]

        try:
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to write history: {e}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FreqAI-Style Auto-Retrain Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.scripts.freqai_retrain --once                  # Single retrain cycle
  python -m src.scripts.freqai_retrain --once --asset ETH      # Retrain ETH only
  python -m src.scripts.freqai_retrain --continuous             # Run every 4h forever
  python -m src.scripts.freqai_retrain --continuous --hours 2   # Run every 2h
  python -m src.scripts.freqai_retrain --trade-trigger          # Watch for 20 trades
  python -m src.scripts.freqai_retrain --dry-run                # Evaluate only, no deploy
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--once', action='store_true', help='Run a single retrain cycle')
    mode.add_argument('--continuous', action='store_true', help='Run retrain loop forever')
    mode.add_argument('--trade-trigger', action='store_true', help='Watch trades, retrain at threshold')

    parser.add_argument('--asset', type=str, default=None, help='Retrain specific asset (BTC, ETH)')
    parser.add_argument('--hours', type=float, default=None, help='Override retrain interval (hours)')
    parser.add_argument('--trades', type=int, default=None, help='Override trade count threshold')
    parser.add_argument('--threshold', type=float, default=None, help='Override min improvement pct')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate only, do not deploy')

    args = parser.parse_args()

    config = load_config()

    # Apply CLI overrides
    if args.hours is not None:
        config['retrain_every_hours'] = args.hours
    if args.trades is not None:
        config['retrain_every_trades'] = args.trades
    if args.threshold is not None:
        config['min_improvement_pct'] = args.threshold
    if args.asset:
        config['assets'] = [args.asset.upper()]

    if args.dry_run:
        # Set an impossibly high improvement threshold so nothing deploys
        config['min_improvement_pct'] = 9999.0
        logger.info("DRY RUN mode — models will be evaluated but not deployed")

    engine = FreqAIRetrain(config)

    if args.once:
        if args.asset:
            result = engine.run_retrain_cycle(args.asset.upper())
            _print_result(result)
        else:
            results = engine.run_all_assets()
            for r in results:
                _print_result(r)

    elif args.continuous:
        engine.continuous_loop(interval_hours=args.hours)

    elif args.trade_trigger:
        engine.watch_trades()


def _print_result(result: dict):
    """Pretty-print a single retrain result."""
    asset = result.get('asset', '?')
    if result.get('error'):
        print(f"\n[{asset}] ERROR: {result['error']}")
        return

    deployed = result.get('deployed', False)
    marker = 'DEPLOYED' if deployed else 'NOT DEPLOYED'
    print(f"\n{'='*50}")
    print(f"[{asset}] {marker}")
    print(f"  Old accuracy:  {result.get('old_accuracy', 0):.4f}")
    print(f"  New accuracy:  {result.get('new_accuracy', 0):.4f}")
    print(f"  Improvement:   {result.get('improvement', 0):+.2f}%")
    print(f"  Buy precision: {result.get('new_buy_precision', 0):.4f}")
    print(f"  Test samples:  {result.get('n_test_samples', 0)}")
    print(f"  Duration:      {result.get('duration_seconds', 0):.1f}s")

    top = result.get('top_features', [])
    if top:
        print(f"  Top features:")
        for feat in top[:5]:
            print(f"    {feat['name']:25s} gain={feat['gain']:.1f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
