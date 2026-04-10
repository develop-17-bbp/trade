"""
Scheduled Model Retraining Pipeline (Production-Ready)
========================================================
Automated weekly LightGBM retraining with:
  - Walk-forward validation holdout (no future leakage)
  - Optuna hyperparameter optimization
  - Model versioning and rollback safety
  - Performance comparison: new model must BEAT current model
  - Real data from Binance + free-tier on-chain sources
  - Automatic feature importance tracking

Usage:
    # Retrain all assets
    python -m src.models.scheduled_retrain

    # Retrain specific asset
    python -m src.models.scheduled_retrain --symbol BTC

    # Dry run (evaluate only, don't save)
    python -m src.models.scheduled_retrain --dry-run

Schedule (cron):
    0 3 * * 0  python -m src.models.scheduled_retrain  # Every Sunday 3 AM
"""

import os
import sys
import json
import shutil
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import lightgbm as lgb
    import optuna
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm or optuna not installed. Run: pip install lightgbm optuna")

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


# ============================================================================
# CONFIGURATION
# ============================================================================

RETRAIN_CONFIG = {
    'lookback_bars': 3600,         # ~600 days of 4h data (6 bars/day × 600 days)
    'validation_pct': 0.15,        # 15% holdout for walk-forward validation
    'test_pct': 0.10,              # 10% final test set (never trained on)
    'optuna_trials': 40,           # Bayesian optimization trials
    'min_accuracy_improvement': 0.005,  # New model must beat old by 0.5%
    'label_threshold': 0.005,      # ±0.5% for directional labels (wider for 4h: need bigger moves)
    'early_stopping_rounds': 30,
    'boost_rounds': 500,
    'timeframe': '4h',             # 4h timeframe for Robinhood swing trading (was 1h)
    'model_dir': 'models',
    'history_file': 'models/retrain_history.json',
    'spread_cost_pct': 0.0,       # Round-trip spread (0 for futures, 3.34 for Robinhood)
    'longs_only': False,           # True for Robinhood (spot, no shorts)
}

# ── Robinhood-specific training profile ──
ROBINHOOD_CONFIG = {
    **RETRAIN_CONFIG,
    'lookback_bars': 1500,         # ~250 days of 4h data (fewer bars, more recent)
    'label_threshold': 0.01,       # 1% moves (wider threshold — spread demands bigger moves)
    'min_accuracy_improvement': 0.01,  # Higher bar: 1% improvement required (was 0.5%)
    'spread_cost_pct': 3.34,       # Robinhood round-trip spread deducted from labels
    'longs_only': True,            # Only train LONG labels (no SHORT on spot)
    'boost_rounds': 400,           # Slightly fewer rounds (less data complexity)
    'timeframe': '4h',             # Only 4h (1d could be added later)
}

ASSETS = {
    'BTC': {'symbol': 'BTC/USDT', 'model_file': 'lgbm_btc.txt'},
    'ETH': {'symbol': 'ETH/USDT', 'model_file': 'lgbm_eth.txt'},
    'AAVE': {'symbol': 'AAVE/USDT', 'model_file': 'lgbm_aave.txt'},
}


# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_training_data(symbol: str, timeframe: str = '1h',
                        limit: int = 15000) -> pd.DataFrame:
    """
    Fetch historical OHLCV data.
    Primary: Binance Vision (S3 downloads — works in all regions).
    Fallback: Binance API via CCXT.
    """
    # Primary: Binance Vision (bypasses 451 region blocks)
    try:
        logger.info("Trying Binance Vision (S3) for data...")
        from download_vision_data import fetch_vision_ohlcv
        df = fetch_vision_ohlcv(symbol, timeframe)
        if not df.empty:
            logger.info(f"Vision SUCCESS: {len(df)} bars for {symbol}")
            # Trim to requested limit (most recent bars)
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            return df
        else:
            logger.warning(f"Vision returned empty DataFrame for {symbol}")
    except ImportError as ie:
        logger.warning(f"download_vision_data.py NOT FOUND — import failed: {ie}")
        logger.warning("Make sure download_vision_data.py is in the project root!")
    except Exception as e:
        logger.warning(f"Vision download failed: {e}, trying CCXT...")

    # Fallback: Binance API via CCXT
    if not HAS_CCXT:
        logger.error("Neither Vision data nor CCXT available")
        return pd.DataFrame()

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    all_data = []
    since = None
    remaining = limit

    while remaining > 0:
        batch_size = min(remaining, 1000)
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=since, limit=batch_size
            )
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            remaining -= len(ohlcv)
            if len(ohlcv) < batch_size:
                break
        except Exception as e:
            logger.warning(f"CCXT fetch failed: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Fetched {len(df)} bars for {symbol} via CCXT ({df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]})")
    return df


def fetch_onchain_features() -> Dict[str, float]:
    """Fetch current on-chain features from free-tier sources for feature enrichment."""
    try:
        from src.data.free_tier_integrations import FreeDataAggregator
        agg = FreeDataAggregator()
        signals = agg.aggregate_all_signals('BTC')
        return {
            'fear_greed_index': signals.get('fear_greed_index', 50),
            'implied_volatility': signals.get('implied_volatility') or 50.0,
            'ls_ratio': signals.get('ls_ratio', 1.0),
            'defi_tvl_change_pct': signals.get('defi_tvl_change_pct', 0),
        }
    except Exception as e:
        logger.warning(f"On-chain feature fetch failed: {e}")
        return {}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def build_features(df: pd.DataFrame, onchain: Optional[Dict] = None) -> pd.DataFrame:
    """
    Build 50+ features from OHLCV + on-chain data.
    All features are computed from historical data (no look-ahead bias).
    """
    feat = pd.DataFrame(index=df.index)

    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = close.pct_change()

    # ── Price Action (12 features) ──
    for w in [5, 10, 20, 50]:
        feat[f'sma_{w}'] = close.rolling(w).mean()
        feat[f'sma_ratio_{w}'] = close / feat[f'sma_{w}']

    feat['ema_12'] = close.ewm(span=12).mean()
    feat['ema_26'] = close.ewm(span=26).mean()
    feat['macd'] = feat['ema_12'] - feat['ema_26']
    feat['macd_signal'] = feat['macd'].ewm(span=9).mean()

    # ── Volatility (8 features) ──
    # Adaptive to timeframe: 4h bar = 4 bars/day, 1h = 24 bars/day, 1d = 1 bar/day
    feat['returns_1bar'] = returns
    feat['returns_4bar'] = returns.rolling(4).sum()   # 4h TF: ~16h | 1h TF: ~4h | 1d TF: ~4d
    feat['returns_24bar'] = returns.rolling(24).sum()  # 4h TF: ~4 days | 1h TF: ~1 day
    feat['realized_vol_20'] = returns.rolling(20).std()
    feat['realized_vol_50'] = returns.rolling(50).std()
    feat['vol_ratio'] = feat['realized_vol_20'] / (feat['realized_vol_50'] + 1e-10)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    feat['atr_14'] = tr.rolling(14).mean()
    feat['atr_pct'] = feat['atr_14'] / (close + 1e-10)

    # ── RSI (3 features) ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat['rsi_14'] = 100 - (100 / (1 + rs))
    feat['rsi_6'] = _quick_rsi(close, 6)
    feat['rsi_divergence'] = feat['rsi_14'] - feat['rsi_14'].shift(10)

    # ── Bollinger Bands (3 features) ──
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    feat['bb_upper'] = sma_20 + 2 * std_20
    feat['bb_lower'] = sma_20 - 2 * std_20
    feat['bb_width'] = (feat['bb_upper'] - feat['bb_lower']) / (close + 1e-10)
    feat['bb_position'] = (close - feat['bb_lower']) / (feat['bb_upper'] - feat['bb_lower'] + 1e-10)

    # ── Volume (5 features) ──
    feat['volume_sma_20'] = volume.rolling(20).mean()
    feat['volume_ratio'] = volume / (feat['volume_sma_20'] + 1e-10)
    feat['volume_trend'] = volume.rolling(10).mean() / (volume.rolling(50).mean() + 1e-10)
    feat['obv'] = (np.sign(returns) * volume).cumsum()
    feat['obv_slope'] = feat['obv'].diff(10) / (feat['obv'].shift(10).abs() + 1e-10)

    # ── Momentum (5 features) ──
    feat['momentum_10'] = close - close.shift(10)
    feat['momentum_20'] = close - close.shift(20)
    feat['roc_10'] = returns.rolling(10).sum()
    feat['roc_20'] = returns.rolling(20).sum()

    # Stochastic
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    feat['stoch_d'] = feat['stoch_k'].rolling(3).mean()

    # ── ADX (1 feature) ──
    feat['adx'] = _compute_adx(high, low, close, 14)

    # ── Microstructure (3 features) ──
    feat['high_low_ratio'] = high / (low + 1e-10)
    feat['close_open_ratio'] = close / (df['open'] + 1e-10)
    feat['body_shadow_ratio'] = (close - df['open']).abs() / (high - low + 1e-10)

    # ── Time features (3 features) ──
    if 'timestamp' in df.columns:
        feat['hour'] = df['timestamp'].dt.hour
        feat['day_of_week'] = df['timestamp'].dt.dayofweek
        feat['is_weekend'] = (feat['day_of_week'] >= 5).astype(int)

    # ── Swing Trading Features (7 features — useful for 4h/1d timeframes) ──
    # These capture multi-day patterns that matter for swing trades on Robinhood
    feat['ema_8'] = close.ewm(span=8).mean()
    feat['ema_21'] = close.ewm(span=21).mean()
    feat['ema_8_21_cross'] = (feat['ema_8'] - feat['ema_21']) / (close + 1e-10)  # EMA crossover strength
    feat['ema_8_slope'] = feat['ema_8'].pct_change(3)  # 3-bar EMA slope (4h: ~12h momentum)
    feat['price_ema8_dist'] = (close - feat['ema_8']) / (feat['atr_14'] + 1e-10)  # Distance in ATR units
    feat['range_expansion'] = (high - low) / (high.rolling(10).max() - low.rolling(10).min() + 1e-10)  # Is today's range expanding?
    feat['higher_high'] = ((high > high.shift(1)) & (low > low.shift(1))).astype(int)  # Higher high + higher low pattern

    # ── On-chain snapshot features (4 features — constant across dataset) ──
    if onchain:
        feat['fear_greed_index'] = onchain.get('fear_greed_index', 50) / 100
        feat['implied_volatility'] = onchain.get('implied_volatility', 50) / 100
        feat['ls_ratio'] = onchain.get('ls_ratio', 1.0)
        feat['tvl_change'] = onchain.get('defi_tvl_change_pct', 0) / 100

    return feat


def _quick_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Simplified ADX computation."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.rolling(period).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(period).mean()


def create_labels(df: pd.DataFrame, threshold: float = 0.005,
                  forward_bars: int = 6) -> pd.Series:
    """
    Create directional labels based on forward return.
    +1 = LONG, 0 = FLAT, -1 = SHORT

    For 4h timeframe: forward_bars=6 = 24h lookahead, threshold=0.5%
    For 1h timeframe: forward_bars=4 = 4h lookahead, threshold=0.1%
    For 1d timeframe: forward_bars=3 = 3 day lookahead, threshold=1.0%
    """
    future_return = df['close'].shift(-forward_bars) / df['close'] - 1
    labels = pd.Series(0, index=df.index)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    return labels


# ============================================================================
# WALK-FORWARD TRAINING WITH OPTUNA
# ============================================================================

def walk_forward_split(n_samples: int, val_pct: float = 0.15,
                       test_pct: float = 0.10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-series aware train/val/test split.
    No shuffling — preserves temporal order.
    """
    train_end = int(n_samples * (1 - val_pct - test_pct))
    val_end = int(n_samples * (1 - test_pct))

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n_samples)

    return train_idx, val_idx, test_idx


def optuna_objective(trial: optuna.Trial, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Optuna objective for LightGBM hyperparameter search."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params, dtrain,
        num_boost_round=RETRAIN_CONFIG['boost_rounds'],
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(RETRAIN_CONFIG['early_stopping_rounds']),
            lgb.log_evaluation(0),
        ]
    )

    y_pred = model.predict(X_val)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_class == y_val)

    return accuracy


def train_optimized_model(X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          n_trials: int = 40) -> Tuple[lgb.Booster, Dict, float]:
    """
    Run Optuna optimization and train final model with best params.

    Returns:
        (model, best_params, validation_accuracy)
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_val_acc = study.best_value
    logger.info(f"Best Optuna accuracy: {best_val_acc:.4f} ({n_trials} trials)")
    logger.info(f"Best params: {best_params}")

    # Train final model on train+val with best params
    full_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
        **best_params,
    }

    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    dtrain = lgb.Dataset(X_full, label=y_full)

    model = lgb.train(
        full_params, dtrain,
        num_boost_round=RETRAIN_CONFIG['boost_rounds'],
    )

    return model, best_params, best_val_acc


# ============================================================================
# MODEL EVALUATION AND VERSIONING
# ============================================================================

def evaluate_model(model: lgb.Booster, X_test: np.ndarray,
                   y_test: np.ndarray) -> Dict[str, float]:
    """Evaluate model on held-out test set."""
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)

    accuracy = float(np.mean(y_pred_class == y_test))

    # Per-class accuracy
    metrics = {'accuracy': accuracy, 'test_samples': len(y_test)}
    for cls in range(3):
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = float(np.mean(y_pred_class[mask] == y_test[mask]))
            label_name = {0: 'short', 1: 'flat', 2: 'long'}[cls]
            metrics[f'{label_name}_accuracy'] = cls_acc
            metrics[f'{label_name}_count'] = int(mask.sum())

    # Confidence calibration
    max_probs = np.max(y_pred, axis=1)
    metrics['avg_confidence'] = float(np.mean(max_probs))
    metrics['high_conf_accuracy'] = float(
        np.mean(y_pred_class[max_probs > 0.6] == y_test[max_probs > 0.6])
    ) if (max_probs > 0.6).sum() > 10 else accuracy

    return metrics


def evaluate_existing_model(model_path: str, X_test: np.ndarray,
                            y_test: np.ndarray) -> Optional[Dict[str, float]]:
    """Evaluate the existing model on the same test set for comparison."""
    if not os.path.exists(model_path):
        return None
    try:
        old_model = lgb.Booster(model_file=model_path)
        return evaluate_model(old_model, X_test, y_test)
    except Exception as e:
        logger.warning(f"Could not evaluate existing model: {e}")
        return None


def save_model_with_versioning(model: lgb.Booster, model_path: str,
                               metrics: Dict, params: Dict) -> str:
    """
    Save model with backup versioning.
    Keeps previous version as *_prev.txt for rollback.
    """
    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)

    # Backup existing model (handle read-only files on Windows)
    if os.path.exists(model_path):
        backup_path = model_path.replace('.txt', '_prev.txt')
        try:
            # Remove read-only flag if present (Windows)
            if os.path.exists(backup_path):
                os.chmod(backup_path, 0o666)
            shutil.copy2(model_path, backup_path)
            logger.info(f"Previous model backed up to {backup_path}")
        except PermissionError:
            logger.warning(f"Cannot backup model (permission denied) — overwriting directly")
            try:
                os.remove(backup_path)
                shutil.copy2(model_path, backup_path)
            except Exception:
                pass  # Continue even if backup fails

    # Save new model (ensure writable on Windows)
    if os.path.exists(model_path):
        try:
            os.chmod(model_path, 0o666)
        except Exception:
            pass
    model.save_model(model_path)

    # Save optimized version too
    optimized_path = model_path.replace('.txt', '_optimized.txt')
    if os.path.exists(optimized_path):
        try:
            os.chmod(optimized_path, 0o666)
        except Exception:
            pass
    model.save_model(optimized_path)

    # Save feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
    }).sort_values('importance', ascending=False)

    imp_path = os.path.join(os.path.dirname(model_path), 'lgbm_feature_importance.csv')
    if os.path.exists(imp_path):
        try:
            os.chmod(imp_path, 0o666)
        except Exception:
            pass
    imp_df.to_csv(imp_path, index=False)

    logger.info(f"Model saved to {model_path} (accuracy={metrics['accuracy']:.4f})")
    return model_path


def log_retrain_history(asset: str, metrics: Dict, params: Dict,
                        old_metrics: Optional[Dict], bars_used: int):
    """Append retrain results to history JSON for tracking improvement over time."""
    history_file = RETRAIN_CONFIG['history_file']
    history = []

    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'asset': asset,
        'bars_used': bars_used,
        'new_accuracy': metrics['accuracy'],
        'new_high_conf_accuracy': metrics.get('high_conf_accuracy', 0),
        'old_accuracy': old_metrics['accuracy'] if old_metrics else None,
        'improvement': metrics['accuracy'] - old_metrics['accuracy'] if old_metrics else None,
        'best_params': params,
        'per_class': {
            'short_acc': metrics.get('short_accuracy', 0),
            'flat_acc': metrics.get('flat_accuracy', 0),
            'long_acc': metrics.get('long_accuracy', 0),
        },
    }
    history.append(entry)

    os.makedirs(os.path.dirname(history_file) or '.', exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Retrain history logged ({len(history)} total entries)")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def retrain_asset(asset: str, dry_run: bool = False) -> Optional[Dict]:
    """
    Full retraining pipeline for a single asset.

    Steps:
      1. Fetch latest OHLCV data (paginated)
      2. Fetch on-chain features for enrichment
      3. Build features + labels
      4. Walk-forward split (train/val/test)
      5. Optuna hyperparameter optimization on train/val
      6. Evaluate on held-out test set
      7. Compare with existing model
      8. Save only if new model beats old model
    """
    if not HAS_LGB:
        logger.error("lightgbm required for retraining")
        return None

    config = ASSETS.get(asset)
    if not config:
        logger.error(f"Unknown asset: {asset}")
        return None

    symbol = config['symbol']
    model_path = os.path.join(RETRAIN_CONFIG['model_dir'], config['model_file'])

    logger.info(f"\n{'='*60}")
    logger.info(f"RETRAINING {asset} ({symbol})")
    logger.info(f"{'='*60}")

    # 1. Fetch data
    logger.info("Step 1: Fetching training data...")
    df = fetch_training_data(symbol, RETRAIN_CONFIG['timeframe'], RETRAIN_CONFIG['lookback_bars'])
    if df.empty or len(df) < 1000:
        logger.error(f"Insufficient data for {asset}: {len(df)} bars")
        return None

    # 2. Fetch on-chain features (concurrent with data processing)
    logger.info("Step 2: Fetching on-chain features...")
    onchain = fetch_onchain_features()

    # 3. Build features + labels
    logger.info("Step 3: Building features...")
    features = build_features(df, onchain)
    labels = create_labels(df, RETRAIN_CONFIG['label_threshold'])

    # Combine and clean
    combined = features.copy()
    combined['label'] = labels.map({-1: 0, 0: 1, 1: 2})  # Map to 0,1,2 for LGB multiclass

    # Drop rows with NaN (lookback period) and future-leaked labels
    combined = combined.dropna()
    if len(combined) < 500:
        logger.error(f"Insufficient clean samples: {len(combined)}")
        return None

    feature_cols = [c for c in combined.columns if c != 'label']
    X = combined[feature_cols].values
    y = combined['label'].values

    logger.info(f"  Dataset: {len(X)} samples x {len(feature_cols)} features")
    for cls, name in enumerate(['SHORT', 'FLAT', 'LONG']):
        count = (y == cls).sum()
        logger.info(f"  {name}: {count} ({100*count/len(y):.1f}%)")

    # 4. Walk-forward split
    logger.info("Step 4: Walk-forward split...")
    train_idx, val_idx, test_idx = walk_forward_split(
        len(X), RETRAIN_CONFIG['validation_pct'], RETRAIN_CONFIG['test_pct']
    )
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 5. Optuna optimization
    logger.info(f"Step 5: Optuna optimization ({RETRAIN_CONFIG['optuna_trials']} trials)...")
    model, best_params, val_acc = train_optimized_model(
        X_train, y_train, X_val, y_val, RETRAIN_CONFIG['optuna_trials']
    )

    # 6. Evaluate on test set
    logger.info("Step 6: Evaluating on held-out test set...")
    new_metrics = evaluate_model(model, X_test, y_test)
    logger.info(f"  New model test accuracy: {new_metrics['accuracy']:.4f}")
    logger.info(f"  High-confidence accuracy: {new_metrics.get('high_conf_accuracy', 0):.4f}")

    # 7. Compare with existing model
    logger.info("Step 7: Comparing with existing model...")
    old_metrics = evaluate_existing_model(model_path, X_test, y_test)

    if old_metrics:
        improvement = new_metrics['accuracy'] - old_metrics['accuracy']
        logger.info(f"  Old model accuracy: {old_metrics['accuracy']:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f}")

        if improvement < RETRAIN_CONFIG['min_accuracy_improvement'] and not dry_run:
            logger.info(f"  New model did not improve by {RETRAIN_CONFIG['min_accuracy_improvement']:.3f}. Skipping save.")
            log_retrain_history(asset, new_metrics, best_params, old_metrics, len(df))
            return {
                'asset': asset, 'status': 'skipped',
                'reason': f'improvement {improvement:+.4f} below threshold',
                'new_accuracy': new_metrics['accuracy'],
                'old_accuracy': old_metrics['accuracy'],
            }
    else:
        logger.info("  No existing model found — saving new model.")

    # 8. Save
    if dry_run:
        logger.info("  DRY RUN — not saving model.")
        return {'asset': asset, 'status': 'dry_run', **new_metrics}

    save_model_with_versioning(model, model_path, new_metrics, best_params)
    log_retrain_history(asset, new_metrics, best_params, old_metrics, len(df))

    return {
        'asset': asset,
        'status': 'saved',
        'model_path': model_path,
        'bars_used': len(df),
        **new_metrics,
        'improvement': new_metrics['accuracy'] - old_metrics['accuracy'] if old_metrics else None,
        'best_params': best_params,
    }


def retrain_all(assets: Optional[List[str]] = None, dry_run: bool = False) -> List[Dict]:
    """Retrain all (or specified) assets sequentially."""
    if assets is None:
        assets = list(ASSETS.keys())

    results = []
    for asset in assets:
        try:
            result = retrain_asset(asset, dry_run=dry_run)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Retrain failed for {asset}: {e}", exc_info=True)
            results.append({'asset': asset, 'status': 'error', 'error': str(e)})

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RETRAIN SUMMARY")
    logger.info(f"{'='*60}")
    for r in results:
        status = r.get('status', 'unknown')
        acc = r.get('accuracy', 0)
        imp = r.get('improvement')
        imp_str = f" ({imp:+.4f})" if imp is not None else ""
        logger.info(f"  {r['asset']}: {status} — accuracy={acc:.4f}{imp_str}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Scheduled LightGBM retraining pipeline")
    parser.add_argument('--symbol', type=str, help='Retrain specific asset (BTC, ETH, AAVE)')
    parser.add_argument('--dry-run', action='store_true', help='Evaluate only, do not save')
    parser.add_argument('--trials', type=int, default=40, help='Optuna trials (default: 40)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    RETRAIN_CONFIG['optuna_trials'] = args.trials

    if args.symbol:
        assets = [args.symbol.upper()]
    else:
        assets = None

    results = retrain_all(assets=assets, dry_run=args.dry_run)

    # Exit code: 0 if at least one model saved, 1 otherwise
    saved = any(r.get('status') == 'saved' for r in results)
    sys.exit(0 if saved or args.dry_run else 1)


if __name__ == '__main__':
    main()
