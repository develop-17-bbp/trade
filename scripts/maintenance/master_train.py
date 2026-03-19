#!/usr/bin/env python3
"""
Master Training Orchestrator
==============================
Smart infinite training loop that cycles through EVERY unique combination of:
  - Symbol  (12 crypto pairs)
  - Timeframe (15m, 1h, 4h, 1d)
  - Time Period (start year: 2020, 2021, 2022, 2023, 2024)

Logic:
  1. Build full cartesian product of (symbol × timeframe × start_year) = 240 combos
  2. Shuffle and train each one, NEVER repeating within a cycle
  3. After all combos exhausted → reshuffle, bump Optuna seed, start next cycle
  4. Tracks progress in JSON state file (dashboard-compatible)
  5. Downloads data automatically via Binance Vision (no API needed)

Usage:
    python master_train.py                          # Full loop (all combos)
    python master_train.py --quick                  # Core symbols, 1h only, fast
    python master_train.py --symbols BTCUSDT ETHUSDT --timeframes 1h 4h
    python master_train.py --once                   # Single pass then exit
    python master_train.py --trials 10              # Fewer Optuna trials (faster)

Press Ctrl+C to stop gracefully at any time.
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('master_train')

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
    logger.warning("optuna not installed — using default hyperparams")

# ============================================================================
# CONFIGURATION
# ============================================================================

ALL_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'AAVEUSDT', 'SOLUSDT', 'BNBUSDT',
    'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'AVAXUSDT', 'DOTUSDT',
    'LINKUSDT', 'MATICUSDT',
]

ALL_TIMEFRAMES = ['15m', '1h', '4h', '1d']

ALL_START_YEARS = [2020, 2021, 2022, 2023, 2024]

CORE_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'AAVEUSDT']

# State files (dashboard integration)
TRAINING_LOG_FILE = 'models/retrain_history.json'
BENCHMARK_LOG_FILE = 'logs/benchmark_history.json'
TRAINING_STATE_FILE = 'logs/training_state.json'
QUEUE_STATE_FILE = 'logs/master_queue_state.json'

# Training defaults
DEFAULT_OPTUNA_TRIALS = 25
BOOST_ROUNDS = 400
EARLY_STOPPING = 25
LABEL_THRESHOLD = 0.001
VALIDATION_PCT = 0.15
MIN_ROWS = 2000


# ============================================================================
# DATA LOADING (via Binance Vision)
# ============================================================================

def fetch_data(symbol: str, timeframe: str, start_year: int,
               data_dir: str = 'data') -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data. Tries:
      1. Cached parquet for this (symbol, timeframe) pair
      2. Download from Binance Vision with specified start_year
    """
    try:
        from download_vision_data import fetch_vision_ohlcv
        df = fetch_vision_ohlcv(symbol, timeframe, start_year=start_year, data_dir=data_dir)
        if df is not None and not df.empty:
            # Filter to start_year onwards (cached file may have earlier data)
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                cutoff = pd.Timestamp(f'{start_year}-01-01')
                df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
            return df
    except ImportError:
        logger.warning("download_vision_data.py not found")
    except Exception as e:
        logger.warning(f"Vision fetch failed for {symbol}/{timeframe}: {e}")

    # Try loading cached parquet/csv directly
    for ext in ['parquet', 'csv']:
        path = os.path.join(data_dir, f"{symbol}-{timeframe}.{ext}")
        if os.path.exists(path):
            df = pd.read_parquet(path) if ext == 'parquet' else pd.read_csv(path)
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
                cutoff = pd.Timestamp(f'{start_year}-01-01')
                df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
            return df

    return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build 38+ technical features + direction labels from OHLCV."""
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = close.pct_change()

    feat = pd.DataFrame(index=df.index)

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

    # Time features (2)
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        feat['hour'] = df['timestamp'].dt.hour
        feat['dow'] = df['timestamp'].dt.dayofweek

    # Labels: forward 4-bar return direction
    future_ret = close.shift(-4) / close - 1
    labels = pd.Series(1, index=df.index)  # FLAT
    labels[future_ret > LABEL_THRESHOLD] = 2   # LONG
    labels[future_ret < -LABEL_THRESHOLD] = 0  # SHORT

    # Drop warmup
    warmup = 55
    feat = feat.iloc[warmup:]
    labels = labels.iloc[warmup:]

    valid = feat.notna().all(axis=1)
    feat = feat[valid]
    labels = labels[valid]

    return feat, labels


# ============================================================================
# TRAINING ENGINE
# ============================================================================

def train_model(X_train, y_train, X_val, y_val,
                n_trials: int = 25, seed: int = 42):
    """Train LightGBM with optional Optuna HPO."""

    if HAS_OPTUNA and n_trials > 0:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)

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

        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        val_acc = study.best_value
    else:
        best_params = {
            'num_leaves': 63, 'learning_rate': 0.02, 'max_depth': 10,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
        }
        val_acc = 0.0

    # Final model on train+val
    full_params = {'objective': 'multiclass', 'num_class': 3, 'verbosity': -1, **best_params}
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    dtrain = lgb.Dataset(X_full, label=y_full)
    model = lgb.train(full_params, dtrain, num_boost_round=BOOST_ROUNDS)

    return model, best_params, val_acc


def evaluate_model(model, X_test, y_test) -> Dict:
    """Evaluate on held-out test set."""
    preds = model.predict(X_test)
    pred_class = np.argmax(preds, axis=1)
    accuracy = float(np.mean(pred_class == y_test))

    metrics = {'accuracy': accuracy, 'test_samples': len(y_test)}
    for cls, name in [(0, 'short'), (1, 'flat'), (2, 'long')]:
        mask = y_test == cls
        if mask.sum() > 0:
            metrics[f'{name}_accuracy'] = float(np.mean(pred_class[mask] == y_test[mask]))
            metrics[f'{name}_count'] = int(mask.sum())

    max_probs = np.max(preds, axis=1)
    hc_mask = max_probs > 0.6
    metrics['high_conf_accuracy'] = float(np.mean(pred_class[hc_mask] == y_test[hc_mask])) \
        if hc_mask.sum() > 10 else accuracy
    metrics['avg_confidence'] = float(np.mean(max_probs))

    return metrics


# ============================================================================
# QUEUE MANAGEMENT
# ============================================================================

def build_full_queue(symbols: List[str], timeframes: List[str],
                     start_years: List[int]) -> List[Tuple[str, str, int]]:
    """
    Build cartesian product: symbol × timeframe × start_year.
    Core symbols at 1h/2020 go first, rest shuffled.
    """
    priority = []
    for sym in CORE_SYMBOLS:
        if sym in symbols:
            priority.append((sym, '1h', 2020))

    everything = list(itertools.product(symbols, timeframes, start_years))
    others = [combo for combo in everything if combo not in priority]
    random.shuffle(others)

    return priority + others


def load_queue_state() -> Dict:
    """Load queue progress from disk."""
    if os.path.exists(QUEUE_STATE_FILE):
        try:
            with open(QUEUE_STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'completed': [], 'cycle': 1, 'total_trained': 0, 'started_at': datetime.utcnow().isoformat()}


def save_queue_state(state: Dict):
    """Persist queue progress."""
    os.makedirs(os.path.dirname(QUEUE_STATE_FILE) or '.', exist_ok=True)
    with open(QUEUE_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


# ============================================================================
# DASHBOARD STATE (same format as continuous_train.py)
# ============================================================================

def update_training_state(symbol: str, timeframe: str, start_year: int,
                          status: str, metrics: Optional[Dict] = None,
                          queue_pos: int = 0, queue_total: int = 0):
    """Update live training state for dashboard."""
    state = {}
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            with open(TRAINING_STATE_FILE, 'r') as f:
                state = json.load(f)
        except Exception:
            pass

    state['last_updated'] = datetime.utcnow().isoformat()
    state['current_symbol'] = symbol
    state['current_timeframe'] = timeframe
    state['current_start_year'] = start_year
    state['status'] = status
    state['queue_position'] = f"{queue_pos}/{queue_total}"

    if metrics:
        state['last_accuracy'] = metrics.get('accuracy', 0)
        state['last_high_conf'] = metrics.get('high_conf_accuracy', 0)

    if 'completed_models' not in state:
        state['completed_models'] = {}
    if status == 'COMPLETED' and metrics:
        key = f"{symbol}_{timeframe}_{start_year}"
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


def log_training_result(symbol: str, timeframe: str, start_year: int,
                        metrics: Dict, params: Dict, rows: int, model_path: str):
    """Append to retrain_history.json (dashboard timeline)."""
    history = []
    if os.path.exists(TRAINING_LOG_FILE):
        try:
            with open(TRAINING_LOG_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            pass

    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'asset': symbol.replace('USDT', ''),
        'symbol': symbol,
        'timeframe': timeframe,
        'start_year': start_year,
        'bars_used': rows,
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
    if len(history) > 1000:
        history = history[-1000:]

    os.makedirs(os.path.dirname(TRAINING_LOG_FILE) or '.', exist_ok=True)
    with open(TRAINING_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def log_benchmark(symbol: str, timeframe: str, metrics: Dict):
    """Append benchmark snapshot for dashboard chart."""
    history = []
    if os.path.exists(BENCHMARK_LOG_FILE):
        try:
            with open(BENCHMARK_LOG_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            pass

    history.append({
        'timestamp': datetime.utcnow().isoformat(),
        'model_version': f"{symbol}_{timeframe}",
        'results': {
            'lgbm_direction_accuracy': {'value': metrics.get('accuracy', 0), 'samples': metrics.get('test_samples', 0)},
            'lgbm_high_conf_accuracy': {'value': metrics.get('high_conf_accuracy', 0), 'samples': metrics.get('test_samples', 0)},
        }
    })
    if len(history) > 500:
        history = history[-500:]

    os.makedirs(os.path.dirname(BENCHMARK_LOG_FILE) or '.', exist_ok=True)
    with open(BENCHMARK_LOG_FILE, 'w') as f:
        json.dump(history, f, indent=2)


# ============================================================================
# MODEL NAMING
# ============================================================================

def get_model_path(symbol: str, timeframe: str, start_year: int,
                   model_dir: str = 'models') -> str:
    """
    Naming convention:
      Core symbols at 1h/2020 → lgbm_btc.txt (standard)
      Other combos → lgbm_btc_4h.txt, lgbm_btc_4h_2022.txt
    """
    base = symbol.replace('USDT', '').lower()

    if symbol in CORE_SYMBOLS and timeframe == '1h' and start_year == 2020:
        name = f"lgbm_{base}.txt"
    elif start_year == 2020:
        name = f"lgbm_{base}_{timeframe}.txt"
    else:
        name = f"lgbm_{base}_{timeframe}_{start_year}.txt"

    return os.path.join(model_dir, name)


# ============================================================================
# SINGLE COMBO TRAINING PIPELINE
# ============================================================================

def train_one(symbol: str, timeframe: str, start_year: int,
              data_dir: str, model_dir: str, n_trials: int,
              optuna_seed: int, queue_pos: int, queue_total: int) -> Optional[Dict]:
    """Full pipeline for one (symbol, timeframe, start_year) combo."""
    if not HAS_LGB:
        logger.error("LightGBM not installed")
        return None

    model_path = get_model_path(symbol, timeframe, start_year, model_dir)
    combo_str = f"{symbol} / {timeframe} / from {start_year}"

    logger.info(f"\n{'━'*60}")
    logger.info(f"  [{queue_pos}/{queue_total}]  {combo_str}")
    logger.info(f"{'━'*60}")

    update_training_state(symbol, timeframe, start_year, 'DOWNLOADING',
                          queue_pos=queue_pos, queue_total=queue_total)

    # 1. Fetch data
    df = fetch_data(symbol, timeframe, start_year, data_dir)
    if df is None or df.empty or len(df) < MIN_ROWS:
        rows = len(df) if df is not None else 0
        logger.warning(f"  ⏭ Skip — {rows} rows (need {MIN_ROWS})")
        update_training_state(symbol, timeframe, start_year, 'SKIPPED')
        return None

    logger.info(f"  Data: {len(df):,} rows")

    # 2. Build features
    update_training_state(symbol, timeframe, start_year, 'BUILDING_FEATURES',
                          queue_pos=queue_pos, queue_total=queue_total)
    features, labels = build_features(df)
    if len(features) < MIN_ROWS:
        logger.warning(f"  ⏭ Skip — {len(features)} clean samples (need {MIN_ROWS})")
        return None

    X = features.values
    y = labels.values

    # 3. Walk-forward split (75% train / 15% val / 10% test)
    train_end = int(len(X) * 0.75)
    val_end = int(len(X) * 0.90)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"  Split: train={len(X_train):,} | val={len(X_val):,} | test={len(X_test):,}")

    # 4. Train
    update_training_state(symbol, timeframe, start_year, 'TRAINING',
                          queue_pos=queue_pos, queue_total=queue_total)
    logger.info(f"  Optuna: {n_trials} trials (seed={optuna_seed})")

    model, best_params, val_acc = train_model(X_train, y_train, X_val, y_val,
                                               n_trials=n_trials, seed=optuna_seed)

    # 5. Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"  ✓ Accuracy:      {metrics['accuracy']:.4f}  ({metrics['test_samples']} samples)")
    logger.info(f"  ✓ High-conf acc: {metrics['high_conf_accuracy']:.4f}")

    # 6. Save model
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(model_path)
    logger.info(f"  ✓ Saved: {model_path}")

    # Also save optimized copy for core symbols (standard naming)
    if symbol in CORE_SYMBOLS and timeframe == '1h' and start_year == 2020:
        opt_path = model_path.replace('.txt', '_optimized.txt')
        model.save_model(opt_path)

    # Save feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_path = os.path.join(model_dir, f'feature_importance_{symbol.replace("USDT","").lower()}_{timeframe}.csv')
    imp_df.to_csv(imp_path, index=False)

    # 7. Log everything
    log_training_result(symbol, timeframe, start_year, metrics, best_params, len(df), model_path)
    log_benchmark(symbol, timeframe, metrics)
    update_training_state(symbol, timeframe, start_year, 'COMPLETED', metrics,
                          queue_pos=queue_pos, queue_total=queue_total)

    return {
        'symbol': symbol, 'timeframe': timeframe, 'start_year': start_year,
        'model_path': model_path, 'rows': len(df), **metrics,
    }


# ============================================================================
# MAIN LOOP
# ============================================================================

def run_master(symbols: List[str], timeframes: List[str], start_years: List[int],
               data_dir: str, model_dir: str, n_trials: int,
               sleep_seconds: int, run_once: bool = False):
    """
    Master training loop.
    Builds full queue → trains each → when exhausted reshuffles & loops.
    """
    q_state = load_queue_state()
    cycle = q_state.get('cycle', 1)
    total_ever = q_state.get('total_trained', 0)

    total_combos = len(symbols) * len(timeframes) * len(start_years)

    logger.info(f"\n{'='*60}")
    logger.info(f"  MASTER TRAINING ORCHESTRATOR")
    logger.info(f"{'='*60}")
    logger.info(f"  Symbols:     {len(symbols)}  {symbols[:5]}{'...' if len(symbols)>5 else ''}")
    logger.info(f"  Timeframes:  {timeframes}")
    logger.info(f"  Start years: {start_years}")
    logger.info(f"  Total combos per cycle: {total_combos}")
    logger.info(f"  Optuna trials: {n_trials}")
    logger.info(f"  Sleep between cycles: {sleep_seconds}s")
    logger.info(f"  Starting at cycle: {cycle}")
    logger.info(f"  Models trained so far: {total_ever}")
    logger.info(f"{'='*60}\n")

    while True:
        # Build queue for this cycle
        queue = build_full_queue(symbols, timeframes, start_years)

        # Filter out already completed in this cycle
        completed_keys = set(q_state.get('completed', []))
        remaining = [(s, t, y) for s, t, y in queue if f"{s}_{t}_{y}" not in completed_keys]

        if not remaining:
            # Cycle complete — reset
            logger.info(f"\n{'★'*60}")
            logger.info(f"  CYCLE {cycle} COMPLETE — all {total_combos} combos trained!")
            logger.info(f"{'★'*60}\n")

            if run_once:
                logger.info("--once mode: exiting after single cycle")
                break

            cycle += 1
            q_state = {
                'completed': [],
                'cycle': cycle,
                'total_trained': total_ever,
                'started_at': datetime.utcnow().isoformat(),
            }
            save_queue_state(q_state)

            logger.info(f"Starting cycle {cycle} in {sleep_seconds}s...")
            try:
                time.sleep(sleep_seconds)
            except KeyboardInterrupt:
                logger.info("\nStopped by user")
                return
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"  CYCLE {cycle} — {len(remaining)} remaining of {total_combos}")
        logger.info(f"{'='*60}")

        cycle_results = []
        optuna_seed = 42 + (cycle - 1) * 100  # Different seed each cycle

        for i, (symbol, timeframe, start_year) in enumerate(remaining, 1):
            pos = total_combos - len(remaining) + i

            try:
                result = train_one(
                    symbol, timeframe, start_year,
                    data_dir, model_dir, n_trials,
                    optuna_seed=optuna_seed + i,
                    queue_pos=pos, queue_total=total_combos,
                )

                # Mark as completed regardless of result
                key = f"{symbol}_{timeframe}_{start_year}"
                if 'completed' not in q_state:
                    q_state['completed'] = []
                q_state['completed'].append(key)
                total_ever += 1 if result else 0
                q_state['total_trained'] = total_ever
                save_queue_state(q_state)

                if result:
                    cycle_results.append(result)

            except KeyboardInterrupt:
                logger.info("\n\nStopped by user. Progress saved — resume with same command.")
                save_queue_state(q_state)
                return
            except Exception as e:
                logger.error(f"  ✗ Error: {e}", exc_info=True)
                key = f"{symbol}_{timeframe}_{start_year}"
                if 'completed' not in q_state:
                    q_state['completed'] = []
                q_state['completed'].append(key)
                save_queue_state(q_state)
                continue

        # Cycle summary
        if cycle_results:
            logger.info(f"\n{'─'*60}")
            logger.info(f"CYCLE {cycle} RESULTS: {len(cycle_results)} models trained")
            logger.info(f"{'─'*60}")

            # Sort by accuracy descending
            cycle_results.sort(key=lambda r: r.get('accuracy', 0), reverse=True)
            for r in cycle_results[:20]:  # Top 20
                logger.info(f"  {r['symbol']:>10} {r['timeframe']:>4} {r['start_year']}  "
                            f"acc={r['accuracy']:.4f}  hc={r['high_conf_accuracy']:.4f}  "
                            f"rows={r['rows']:>6,}")

            best = cycle_results[0]
            logger.info(f"\n  ★ BEST: {best['symbol']}/{best['timeframe']} from {best['start_year']} "
                        f"→ {best['accuracy']:.4f}")
            logger.info(f"{'─'*60}")

        if run_once:
            break

        # Reset for next cycle
        cycle += 1
        q_state = {
            'completed': [],
            'cycle': cycle,
            'total_trained': total_ever,
            'started_at': datetime.utcnow().isoformat(),
        }
        save_queue_state(q_state)

        logger.info(f"\nCycle complete. Starting cycle {cycle} in {sleep_seconds}s...")
        try:
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            logger.info("\nStopped by user")
            return


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master Training Orchestrator — cycles through all symbol × timeframe × period combos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_train.py                              # Full loop (240 combos)
  python master_train.py --quick                      # Core symbols, 1h, fast
  python master_train.py --once                       # Single cycle then exit
  python master_train.py --symbols BTCUSDT ETHUSDT    # Specific symbols
  python master_train.py --timeframes 1h 4h           # Specific timeframes
  python master_train.py --start-years 2022 2023 2024 # Recent data only
  python master_train.py --trials 10                  # Faster Optuna
  python master_train.py --reset                      # Reset queue, start fresh
        """
    )

    parser.add_argument('--symbols', nargs='+', default=None,
                        help=f'Symbols (default: all 12)')
    parser.add_argument('--timeframes', nargs='+', default=None,
                        help=f'Timeframes (default: {ALL_TIMEFRAMES})')
    parser.add_argument('--start-years', nargs='+', type=int, default=None,
                        help=f'Start years (default: {ALL_START_YEARS})')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: core symbols, 1h, 10 trials')
    parser.add_argument('--once', action='store_true',
                        help='Run single cycle then exit')
    parser.add_argument('--trials', type=int, default=DEFAULT_OPTUNA_TRIALS,
                        help=f'Optuna trials per model (default: {DEFAULT_OPTUNA_TRIALS})')
    parser.add_argument('--data-dir', default='data',
                        help='Data directory (default: data)')
    parser.add_argument('--model-dir', default='models',
                        help='Model output directory (default: models)')
    parser.add_argument('--sleep', type=int, default=300,
                        help='Sleep seconds between cycles (default: 300)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset queue state and start fresh')

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        symbols = CORE_SYMBOLS
        timeframes = ['1h']
        start_years = [2020]
        n_trials = 10
    else:
        symbols = args.symbols or ALL_SYMBOLS
        timeframes = args.timeframes or ALL_TIMEFRAMES
        start_years = args.start_years or ALL_START_YEARS
        n_trials = args.trials

    # Reset queue if requested
    if args.reset and os.path.exists(QUEUE_STATE_FILE):
        os.remove(QUEUE_STATE_FILE)
        logger.info("Queue state reset — starting fresh")

    run_master(
        symbols=symbols,
        timeframes=timeframes,
        start_years=start_years,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_trials=n_trials,
        sleep_seconds=args.sleep,
        run_once=args.once,
    )


if __name__ == '__main__':
    main()
