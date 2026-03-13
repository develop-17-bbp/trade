#!/usr/bin/env python3
"""
Continuous Autonomous Training Loop
=====================================
Self-sustaining recursive engine that:
  1. Shuffles between multiple crypto pairs and timeframes
  2. Downloads historical data from Binance Vision (bypasses API blocks)
  3. Trains LightGBM models with Optuna optimization
  4. Tracks all training results for dashboard visualization
  5. Loops indefinitely with configurable sleep interval

Usage:
    python -m src.scripts.continuous_train
    python -m src.scripts.continuous_train --model-dir models --data-dir data/offline --sleep-seconds 900
    python -m src.scripts.continuous_train --use-kaggle --symbols BTCUSDT ETHUSDT

Schedule as background service:
    nohup python -m src.scripts.continuous_train --sleep-seconds 900 &
"""

import os
import sys
import json
import time
import random
import hashlib
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('continuous_train')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.error("lightgbm not installed. Run: pip install lightgbm")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ============================================================================
# CONFIGURATION
# ============================================================================

ALL_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'AAVEUSDT', 'SOLUSDT', 'BNBUSDT',
    'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'DOTUSDT',
    'LINKUSDT', 'MATICUSDT',
]

ALL_TIMEFRAMES = ['15m', '1h', '4h', '1d']

CORE_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'AAVEUSDT']  # Always train these first

TRAINING_LOG_FILE = 'models/retrain_history.json'
BENCHMARK_LOG_FILE = 'logs/benchmark_history.json'
TRAINING_STATE_FILE = 'logs/training_state.json'

OPTUNA_TRIALS = 25  # Per model (fast enough for continuous loop)
BOOST_ROUNDS = 400
EARLY_STOPPING = 25
LABEL_THRESHOLD = 0.001  # ±0.1% for direction labels
VALIDATION_PCT = 0.15
MIN_ROWS = 2000  # Minimum rows to train a model


# ============================================================================
# DATA LOADING
# ============================================================================

def find_data_file(symbol: str, timeframe: str, data_dir: str) -> Optional[str]:
    """Find parquet/CSV data file for a symbol+timeframe combination."""
    candidates = [
        os.path.join(data_dir, f"{symbol}-{timeframe}.parquet"),
        os.path.join(data_dir, f"{symbol}-{timeframe}.csv"),
        os.path.join(data_dir, f"{symbol}_{timeframe}.parquet"),
        os.path.join(data_dir, f"{symbol}_{timeframe}.csv"),
        # Legacy naming
        os.path.join('data', f"{symbol}-{timeframe}.parquet"),
        os.path.join('data', f"{symbol.replace('USDT','')}_USDT_{timeframe}.csv"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_data(path: str) -> pd.DataFrame:
    """Load OHLCV data from parquet or CSV."""
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # Normalize columns
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


def download_if_missing(symbol: str, timeframe: str, data_dir: str) -> Optional[str]:
    """Download data from Binance Vision if not already present."""
    existing = find_data_file(symbol, timeframe, data_dir)
    if existing:
        return existing

    try:
        from download_vision_data import download_symbol
        path = download_symbol(symbol, timeframe, start_year=2020, output_dir=data_dir)
        return path
    except ImportError:
        logger.warning("download_vision_data.py not found — cannot auto-download")
        return None
    except Exception as e:
        logger.warning(f"Download failed for {symbol}/{timeframe}: {e}")
        return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build features from OHLCV data.
    Returns (feature_df, labels) — ready for LightGBM.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # SMA ratios (8)
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
        (low - close.shift()).abs()
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

    # Bollinger (3)
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

    # Microstructure (3)
    feat['hl_ratio'] = high / (low + 1e-10)
    feat['co_ratio'] = close / (df['open'] + 1e-10)
    feat['body_shadow'] = (close - df['open']).abs() / (high - low + 1e-10)

    # Time (if timestamp available) (2)
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        feat['hour'] = df['timestamp'].dt.hour
        feat['dow'] = df['timestamp'].dt.dayofweek

    # Labels: forward return direction
    future_ret = close.shift(-4) / close - 1
    labels = pd.Series(1, index=df.index)  # Default: FLAT (mapped to 1)
    labels[future_ret > LABEL_THRESHOLD] = 2   # LONG
    labels[future_ret < -LABEL_THRESHOLD] = 0  # SHORT

    # Drop warmup rows
    warmup = 55
    feat = feat.iloc[warmup:]
    labels = labels.iloc[warmup:]

    # Drop rows with NaN
    valid = feat.notna().all(axis=1)
    feat = feat[valid]
    labels = labels[valid]

    return feat, labels


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_single_model(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       use_optuna: bool = True) -> Tuple[lgb.Booster, Dict, float]:
    """Train a single LightGBM model, optionally with Optuna."""

    if use_optuna and HAS_OPTUNA:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'verbosity': -1,
                'num_leaves': trial.suggest_int('num_leaves', 15, 127),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 80),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
            }

            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

            model = lgb.train(
                params, dtrain,
                num_boost_round=BOOST_ROUNDS,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)],
            )

            preds = model.predict(X_val)
            return float(np.mean(np.argmax(preds, axis=1) == y_val))

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        best_params = study.best_params
        val_acc = study.best_value
    else:
        best_params = {
            'num_leaves': 63,
            'learning_rate': 0.02,
            'max_depth': 10,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        val_acc = 0.0

    # Train final model on train+val
    full_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
        **best_params,
    }

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    dtrain = lgb.Dataset(X_full, label=y_full)

    model = lgb.train(full_params, dtrain, num_boost_round=BOOST_ROUNDS)

    return model, best_params, val_acc


def evaluate_model(model: lgb.Booster, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate model on test set."""
    preds = model.predict(X_test)
    pred_class = np.argmax(preds, axis=1)

    accuracy = float(np.mean(pred_class == y_test))

    metrics = {'accuracy': accuracy, 'test_samples': len(y_test)}

    # Per-class
    for cls, name in [(0, 'short'), (1, 'flat'), (2, 'long')]:
        mask = y_test == cls
        if mask.sum() > 0:
            metrics[f'{name}_accuracy'] = float(np.mean(pred_class[mask] == y_test[mask]))
            metrics[f'{name}_count'] = int(mask.sum())

    # High-confidence accuracy
    max_probs = np.max(preds, axis=1)
    hc_mask = max_probs > 0.6
    metrics['high_conf_accuracy'] = float(np.mean(pred_class[hc_mask] == y_test[hc_mask])) \
        if hc_mask.sum() > 10 else accuracy
    metrics['avg_confidence'] = float(np.mean(max_probs))

    return metrics


# ============================================================================
# STATE MANAGEMENT (Dashboard Integration)
# ============================================================================

def update_training_state(symbol: str, timeframe: str, status: str,
                          metrics: Optional[Dict] = None):
    """Update training state file for dashboard consumption."""
    state = {}
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            with open(TRAINING_STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            state = {}

    state['last_updated'] = datetime.utcnow().isoformat()
    state['current_symbol'] = symbol
    state['current_timeframe'] = timeframe
    state['status'] = status

    if metrics:
        state['last_accuracy'] = metrics.get('accuracy', 0)
        state['last_high_conf'] = metrics.get('high_conf_accuracy', 0)

    # Track completed models
    if 'completed_models' not in state:
        state['completed_models'] = {}
    if status == 'COMPLETED' and metrics:
        key = f"{symbol}_{timeframe}"
        state['completed_models'][key] = {
            'accuracy': metrics.get('accuracy', 0),
            'high_conf_accuracy': metrics.get('high_conf_accuracy', 0),
            'trained_at': datetime.utcnow().isoformat(),
            'samples': metrics.get('test_samples', 0),
        }

    state['total_models_trained'] = len(state.get('completed_models', {}))

    os.makedirs(os.path.dirname(TRAINING_STATE_FILE) or '.', exist_ok=True)
    with open(TRAINING_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def log_training_result(symbol: str, timeframe: str, metrics: Dict,
                        params: Dict, rows_used: int, model_path: str):
    """Append training result to retrain history (for dashboard)."""
    history = []
    if os.path.exists(TRAINING_LOG_FILE):
        try:
            with open(TRAINING_LOG_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'asset': symbol.replace('USDT', ''),
        'symbol': symbol,
        'timeframe': timeframe,
        'bars_used': rows_used,
        'model_path': model_path,
        'new_accuracy': metrics.get('accuracy', 0),
        'new_high_conf_accuracy': metrics.get('high_conf_accuracy', 0),
        'avg_confidence': metrics.get('avg_confidence', 0),
        'best_params': params,
        'per_class': {
            'short_acc': metrics.get('short_accuracy', 0),
            'flat_acc': metrics.get('flat_accuracy', 0),
            'long_acc': metrics.get('long_accuracy', 0),
        },
    }
    history.append(entry)

    # Keep last 500 entries
    if len(history) > 500:
        history = history[-500:]

    os.makedirs(os.path.dirname(TRAINING_LOG_FILE) or '.', exist_ok=True)
    with open(TRAINING_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def log_benchmark_snapshot(symbol: str, timeframe: str, metrics: Dict):
    """Add benchmark snapshot for dashboard timeline chart."""
    history = []
    if os.path.exists(BENCHMARK_LOG_FILE):
        try:
            with open(BENCHMARK_LOG_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    snapshot = {
        'timestamp': datetime.utcnow().isoformat(),
        'model_version': f"{symbol}_{timeframe}",
        'results': {
            'lgbm_direction_accuracy': {
                'value': metrics.get('accuracy', 0),
                'samples': metrics.get('test_samples', 0),
            },
            'lgbm_high_conf_accuracy': {
                'value': metrics.get('high_conf_accuracy', 0),
                'samples': metrics.get('test_samples', 0),
            },
        }
    }
    history.append(snapshot)

    # Keep last 200 snapshots
    if len(history) > 200:
        history = history[-200:]

    os.makedirs(os.path.dirname(BENCHMARK_LOG_FILE) or '.', exist_ok=True)
    with open(BENCHMARK_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def train_symbol_timeframe(symbol: str, timeframe: str, data_dir: str,
                           model_dir: str) -> Optional[Dict]:
    """
    Full pipeline for one symbol+timeframe combination:
      1. Find/download data
      2. Build features
      3. Walk-forward split
      4. Train with Optuna
      5. Evaluate on test set
      6. Save model + log results
    """
    if not HAS_LGB:
        return None

    model_name = f"lgbm_{symbol.replace('USDT', '').lower()}_{timeframe}.txt"
    # Core symbols get the standard naming (lgbm_btc.txt, not lgbm_btc_1h.txt)
    if symbol in CORE_SYMBOLS and timeframe == '1h':
        model_name = f"lgbm_{symbol.replace('USDT', '').lower()}.txt"

    model_path = os.path.join(model_dir, model_name)

    logger.info(f"\n{'─'*50}")
    logger.info(f"Training: {symbol} / {timeframe}")
    logger.info(f"{'─'*50}")

    update_training_state(symbol, timeframe, 'IN_PROGRESS')

    # 1. Load data
    data_path = find_data_file(symbol, timeframe, data_dir)
    if not data_path:
        data_path = download_if_missing(symbol, timeframe, data_dir)
    if not data_path:
        logger.warning(f"  No data available for {symbol}/{timeframe}")
        update_training_state(symbol, timeframe, 'NO_DATA')
        return None

    df = load_data(data_path)
    if df.empty or len(df) < MIN_ROWS:
        logger.warning(f"  Insufficient data: {len(df)} rows (need {MIN_ROWS})")
        update_training_state(symbol, timeframe, 'INSUFFICIENT_DATA')
        return None

    logger.info(f"  Data: {len(df)} rows from {data_path}")

    # 2. Build features
    features, labels = build_features(df)
    if len(features) < MIN_ROWS:
        logger.warning(f"  Insufficient clean samples: {len(features)}")
        return None

    X = features.values
    y = labels.values

    # 3. Walk-forward split
    train_end = int(len(X) * (1 - VALIDATION_PCT - 0.10))
    val_end = int(len(X) * (1 - 0.10))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"  Split: train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

    # 4. Train
    logger.info(f"  Training with Optuna ({OPTUNA_TRIALS} trials)...")
    model, best_params, val_acc = train_single_model(X_train, y_train, X_val, y_val)

    # 5. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"  Test accuracy:      {metrics['accuracy']:.4f}")
    logger.info(f"  High-conf accuracy: {metrics['high_conf_accuracy']:.4f}")

    # 6. Save
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(model_path)
    logger.info(f"  Model saved: {model_path}")

    # Also save optimized version for core symbols
    if symbol in CORE_SYMBOLS and timeframe == '1h':
        opt_path = model_path.replace('.txt', '_optimized.txt')
        model.save_model(opt_path)

    # Save feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)
    imp_path = os.path.join(model_dir, 'lgbm_feature_importance.csv')
    imp_df.to_csv(imp_path, index=False)

    # Log results
    log_training_result(symbol, timeframe, metrics, best_params, len(df), model_path)
    log_benchmark_snapshot(symbol, timeframe, metrics)
    update_training_state(symbol, timeframe, 'COMPLETED', metrics)

    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'model_path': model_path,
        'rows': len(df),
        **metrics,
    }


# ============================================================================
# CONTINUOUS LOOP
# ============================================================================

def generate_training_queue(symbols: List[str], timeframes: List[str],
                            prioritize_core: bool = True) -> List[Tuple[str, str]]:
    """
    Generate shuffled training queue.
    Core symbols (BTC, ETH, AAVE) at 1h are always trained first.
    Then shuffled diverse combinations.
    """
    queue = []

    # Priority: core symbols at primary timeframe
    if prioritize_core:
        for sym in CORE_SYMBOLS:
            if sym in symbols:
                queue.append((sym, '1h'))

    # All other combinations (shuffled)
    other = []
    for sym in symbols:
        for tf in timeframes:
            pair = (sym, tf)
            if pair not in queue:
                other.append(pair)

    random.shuffle(other)
    queue.extend(other)

    return queue


def run_continuous(symbols: List[str], timeframes: List[str],
                   data_dir: str, model_dir: str, sleep_seconds: int,
                   use_kaggle: bool = False, max_rounds: int = 0):
    """
    Main continuous training loop.

    Args:
        symbols: List of symbols to train
        timeframes: List of timeframes
        data_dir: Data directory
        model_dir: Model output directory
        sleep_seconds: Sleep between rounds
        use_kaggle: Whether to look for Kaggle datasets
        max_rounds: Max rounds (0 = infinite)
    """
    round_num = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"CONTINUOUS TRAINING ENGINE STARTED")
    logger.info(f"{'='*60}")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframes: {timeframes}")
    logger.info(f"  Data dir: {data_dir}")
    logger.info(f"  Model dir: {model_dir}")
    logger.info(f"  Sleep: {sleep_seconds}s between rounds")
    logger.info(f"  Optuna trials: {OPTUNA_TRIALS}")
    logger.info(f"{'='*60}\n")

    while True:
        round_num += 1
        if max_rounds > 0 and round_num > max_rounds:
            logger.info(f"Reached max rounds ({max_rounds}). Stopping.")
            break

        logger.info(f"\n{'='*60}")
        logger.info(f"ROUND {round_num} — {datetime.utcnow().isoformat()}")
        logger.info(f"{'='*60}")

        queue = generate_training_queue(symbols, timeframes)
        round_results = []

        for symbol, timeframe in queue:
            try:
                result = train_symbol_timeframe(symbol, timeframe, data_dir, model_dir)
                if result:
                    round_results.append(result)
            except KeyboardInterrupt:
                logger.info("\nTraining interrupted by user")
                return
            except Exception as e:
                logger.error(f"Error training {symbol}/{timeframe}: {e}", exc_info=True)
                continue

        # Round summary
        logger.info(f"\n{'─'*50}")
        logger.info(f"ROUND {round_num} SUMMARY: {len(round_results)} models trained")
        for r in round_results:
            logger.info(f"  {r['symbol']}/{r['timeframe']}: "
                        f"acc={r['accuracy']:.4f} | hc={r['high_conf_accuracy']:.4f} | "
                        f"rows={r['rows']}")
        logger.info(f"{'─'*50}")

        if max_rounds > 0 and round_num >= max_rounds:
            break

        logger.info(f"\nSleeping {sleep_seconds}s before next round...")
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("\nTraining stopped by user")
            return


# ============================================================================
# MAIN
# ============================================================================

def _set_optuna_trials(n: int):
    global OPTUNA_TRIALS
    OPTUNA_TRIALS = n


def main():
    parser = argparse.ArgumentParser(description="Continuous autonomous LightGBM training loop")
    parser.add_argument('--symbols', nargs='+', default=CORE_SYMBOLS,
                        help=f'Symbols to train (default: {CORE_SYMBOLS})')
    parser.add_argument('--all-symbols', action='store_true',
                        help='Train all 12 supported symbols')
    parser.add_argument('--timeframes', nargs='+', default=['1h'],
                        help=f'Timeframes (default: [1h])')
    parser.add_argument('--all-timeframes', action='store_true',
                        help='Train all timeframes (15m, 1h, 4h, 1d)')
    parser.add_argument('--data-dir', default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--model-dir', default='models',
                        help='Model output directory (default: models)')
    parser.add_argument('--sleep-seconds', type=int, default=900,
                        help='Sleep between rounds in seconds (default: 900)')
    parser.add_argument('--use-kaggle', action='store_true',
                        help='Look for Kaggle datasets in data/offline')
    parser.add_argument('--max-rounds', type=int, default=0,
                        help='Max training rounds (0=infinite, default: 0)')
    parser.add_argument('--single', action='store_true',
                        help='Run single round then exit')
    parser.add_argument('--trials', type=int, default=25,
                        help='Optuna trials per model (default: 25)')

    args = parser.parse_args()

    _set_optuna_trials(args.trials)

    symbols = ALL_SYMBOLS if args.all_symbols else args.symbols
    timeframes = ALL_TIMEFRAMES if args.all_timeframes else args.timeframes
    max_rounds = 1 if args.single else args.max_rounds

    data_dir = args.data_dir
    if args.use_kaggle:
        data_dir = os.path.join(args.data_dir, 'offline')
        os.makedirs(data_dir, exist_ok=True)

    run_continuous(
        symbols=symbols,
        timeframes=timeframes,
        data_dir=data_dir,
        model_dir=args.model_dir,
        sleep_seconds=args.sleep_seconds,
        use_kaggle=args.use_kaggle,
        max_rounds=max_rounds,
    )


if __name__ == '__main__':
    main()
