"""
Auto-Retrain Loop for LightGBM Classifier (Layer 1.5)
======================================================
Automated weekly hyperparameter tuning using Optuna.

This script:
1. Fetches the latest 10,000 bars of data for the specified asset
2. Uses Optuna to optimize LightGBM hyperparameters (num_leaves, learning_rate, etc.)
3. Trains a new model with the best parameters
4. Saves the updated model to disk
5. Logs the optimization results

Usage:
    python -m src.models.auto_retrain --symbol AAVE/USDT --model-out models/lgbm_aave.txt

Requirements:
- optuna
- lightgbm
- ccxt for data fetching
"""

import argparse
import time
from typing import List, Dict, Optional
import datetime

import ccxt
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

from src.models.lightgbm_classifier import LightGBMClassifier


def fetch_latest_ohlcv(symbol: str, timeframe: str = '1h', limit: int = 10000) -> pd.DataFrame:
    """
    Fetch the latest OHLCV data.
    Primary: Binance Vision (S3 — works in all regions, bypasses 451).
    Fallback: Binance API via CCXT.

    Args:
        symbol: Trading pair (e.g., 'AAVE/USDT')
        timeframe: Timeframe (e.g., '1h')
        limit: Number of bars to fetch (default 10,000)

    Returns:
        DataFrame with OHLCV data
    """
    # Primary: Binance Vision (no region blocks)
    try:
        print("Trying Binance Vision (S3) for data...")
        from download_vision_data import fetch_vision_ohlcv
        df = fetch_vision_ohlcv(symbol, timeframe)
        if not df.empty:
            if len(df) > limit:
                df = df.tail(limit).reset_index(drop=True)
            print(f"Vision SUCCESS: {len(df)} bars from Binance Vision")
            return df
        else:
            print(f"Vision returned empty for {symbol}")
    except ImportError as ie:
        print(f"[ERROR] download_vision_data.py NOT FOUND: {ie}")
        print("Make sure download_vision_data.py is in the project root!")
    except Exception as e:
        print(f"Vision download failed ({e}), trying CCXT...")

    # Fallback: CCXT
    print("Falling back to CCXT Binance API (may fail with 451 in US)...")
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"[ERROR] CCXT also failed: {e}")
        return pd.DataFrame()


def build_dataset(df: pd.DataFrame) -> (List[Dict[str, float]], List[int]):
    """
    Build feature dataset and labels from OHLCV data.

    Args:
        df: OHLCV DataFrame

    Returns:
        Tuple of (features list, labels list)
    """
    closes = df['close'].tolist()
    highs = df['high'].tolist()
    lows = df['low'].tolist()
    volumes = df['volume'].tolist()

    clf = LightGBMClassifier()
    features = clf.extract_features(closes, highs, lows, volumes)

    # Label by next-bar return
    labels: List[int] = []
    for i in range(len(closes)):
        if i == len(closes) - 1:
            labels.append(0)
        else:
            ret = (closes[i + 1] - closes[i]) / closes[i]
            if ret > 0.001:  # Small threshold to avoid noise
                labels.append(1)
            elif ret < -0.001:
                labels.append(-1)
            else:
                labels.append(0)

    return features, labels


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """
    Optuna objective function for LightGBM hyperparameter optimization.

    Args:
        trial: Optuna trial
        X: Feature matrix
        y: Labels

    Returns:
        Validation accuracy
    """
    # Hyperparameter search space
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
    }

    # Split data for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )

    # Predict and calculate accuracy
    y_pred = model.predict(X_val)
    y_pred_class = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_class == y_val)

    return accuracy


def run_optuna_optimization(features: List[Dict[str, float]], labels: List[int], n_trials: int = 50) -> Dict:
    """
    Run Optuna optimization to find best LightGBM hyperparameters.

    Args:
        features: Feature dictionaries
        labels: Label list
        n_trials: Number of Optuna trials

    Returns:
        Best parameters dictionary
    """
    # Prepare data
    df = pd.DataFrame(features)
    df['label'] = labels
    df = df.dropna()

    if df.empty:
        raise ValueError("No training data after cleaning.")

    X = df.drop(columns=['label']).values
    y = df['label'].map({-1: 0, 0: 1, 1: 2}).values  # Map to multiclass

    # Create Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print(f"Best trial accuracy: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    return study.best_params


def train_with_best_params(features: List[Dict[str, float]], labels: List[int], best_params: Dict, model_out: str):
    """
    Train LightGBM model with optimized hyperparameters and save it.

    Args:
        features: Feature dictionaries
        labels: Label list
        best_params: Best hyperparameters from Optuna
        model_out: Output path for the model
    """
    # Prepare data
    df = pd.DataFrame(features)
    df['label'] = labels
    df = df.dropna()

    X = df.drop(columns=['label']).values
    y = df['label'].map({-1: 0, 0: 1, 1: 2}).values

    # Set up parameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'verbosity': -1,
        **best_params
    }

    dtrain = lgb.Dataset(X, label=y)
    model = lgb.train(params, dtrain, num_boost_round=300)

    # Save model
    model.save_model(model_out)
    print(f"Optimized model saved to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Auto-retrain LightGBM classifier with Optuna optimization")
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., AAVE/USDT)')
    parser.add_argument('--timeframe', default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--limit', type=int, default=10000, help='Number of bars to fetch (default: 10000)')
    parser.add_argument('--model-out', required=True, help='Output path for the trained model')
    parser.add_argument('--n-trials', type=int, default=50, help='Number of Optuna trials (default: 50)')

    args = parser.parse_args()

    print(f"Starting auto-retrain for {args.symbol} at {datetime.datetime.now()}")

    # Fetch latest data
    print("Fetching latest data...")
    df = fetch_latest_ohlcv(args.symbol, args.timeframe, args.limit)
    if df.empty:
        print("No data fetched; exiting.")
        return

    print(f"Fetched {len(df)} bars")

    # Build dataset
    print("Building feature dataset...")
    features, labels = build_dataset(df)

    # Run Optuna optimization
    print("Running Optuna hyperparameter optimization...")
    best_params = run_optuna_optimization(features, labels, args.n_trials)

    # Train final model
    print("Training final model with optimized parameters...")
    train_with_best_params(features, labels, best_params, args.model_out)

    print("Auto-retrain completed successfully!")


if __name__ == '__main__':
    main()