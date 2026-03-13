"""
Historical Backtesting with Real Data Feeds (Production-Ready)
================================================================
Trains and backtests the LightGBM model on real historical data from:
  - Binance:       OHLCV price data (via CCXT, free)
  - DefiLlama:     Historical TVL, stablecoin supply (free API)
  - Blockchain.com: BTC network stats (free API)
  - Alternative.me: Fear & Greed historical (free API)

Walk-forward methodology:
  1. Fetch 6-12 months of hourly data
  2. Split into expanding windows (train on past, test on future)
  3. For each window: train LightGBM, generate signals, run backtest
  4. Aggregate out-of-sample results (no look-ahead bias)

Usage:
    python -m src.trading.historical_backtest --symbol BTC --months 6
    python -m src.trading.historical_backtest --symbol ETH --months 12 --windows 8
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False


# ============================================================================
# DATA COLLECTION — Real Historical Sources
# ============================================================================

def fetch_historical_ohlcv(symbol: str, months: int = 6,
                           timeframe: str = '1h') -> pd.DataFrame:
    """
    Fetch historical OHLCV data.
    Primary: Binance Vision (S3 — works in all regions, bypasses 451).
    Fallback: Binance API via CCXT.
    """
    # Primary: Binance Vision
    try:
        from download_vision_data import fetch_vision_ohlcv
        df = fetch_vision_ohlcv(symbol, timeframe)
        if not df.empty:
            # Filter to requested months
            if 'timestamp' in df.columns:
                cutoff = datetime.utcnow() - timedelta(days=months * 30)
                df = df[df['timestamp'] >= cutoff].reset_index(drop=True)
            if not df.empty:
                logger.info(f"Loaded {len(df)} bars from Binance Vision")
                return df
    except ImportError:
        logger.info("download_vision_data not available, trying CCXT...")
    except Exception as e:
        logger.warning(f"Vision download failed: {e}, trying CCXT...")

    # Fallback: CCXT
    if not HAS_CCXT:
        logger.error("Neither Vision data nor CCXT available")
        return pd.DataFrame()

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    since_dt = datetime.utcnow() - timedelta(days=months * 30)
    since_ms = int(since_dt.timestamp() * 1000)

    all_data = []
    current_since = since_ms

    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not batch:
                break
            all_data.extend(batch)
            current_since = batch[-1][0] + 1
            if len(batch) < 1000:
                break
        except Exception as e:
            logger.warning(f"OHLCV fetch error: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    logger.info(f"Fetched {len(df)} bars for {symbol} "
                f"({df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()})")
    return df


def fetch_historical_fear_greed(days: int = 365) -> pd.DataFrame:
    """Fetch historical Fear & Greed Index from Alternative.me (free, no key)."""
    import requests
    try:
        resp = requests.get(
            f'https://api.alternative.me/fng/?limit={days}',
            timeout=10,
        )
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json().get('data', [])
        records = []
        for entry in data:
            records.append({
                'date': pd.to_datetime(int(entry['timestamp']), unit='s').date(),
                'fear_greed': int(entry['value']),
                'fg_classification': entry['value_classification'],
            })
        df = pd.DataFrame(records)
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} days of Fear & Greed data")
        return df
    except Exception as e:
        logger.warning(f"Fear & Greed fetch failed: {e}")
        return pd.DataFrame()


def fetch_historical_defillama_tvl() -> pd.DataFrame:
    """Fetch historical DeFi TVL from DefiLlama (free, no key)."""
    import requests
    try:
        resp = requests.get('https://api.llama.fi/v2/historicalChainTvl', timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        records = []
        for entry in data:
            records.append({
                'date': pd.to_datetime(int(entry['date']), unit='s').date(),
                'total_tvl': float(entry.get('tvl', 0)),
            })
        df = pd.DataFrame(records)
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} days of DeFi TVL data")
        return df
    except Exception as e:
        logger.warning(f"DefiLlama TVL fetch failed: {e}")
        return pd.DataFrame()


def fetch_historical_stablecoin_mcap() -> pd.DataFrame:
    """Fetch historical stablecoin market cap from DefiLlama (free)."""
    import requests
    try:
        resp = requests.get('https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1', timeout=15)
        if resp.status_code != 200:
            return pd.DataFrame()

        data = resp.json()
        records = []
        for entry in data:
            date = pd.to_datetime(int(entry['date']), unit='s').date()
            total = sum(float(v.get('peggedUSD', 0)) for v in entry.get('totalCirculating', {}).values()) \
                if isinstance(entry.get('totalCirculating'), dict) \
                else float(entry.get('totalCirculating', {}).get('peggedUSD', 0)) \
                if isinstance(entry.get('totalCirculating'), dict) else 0
            # Simpler: just use totalCirculatingUSD if available
            total = float(entry.get('totalCirculatingUSD', {}).get('peggedUSD', total))
            records.append({'date': date, 'stablecoin_mcap': total})

        df = pd.DataFrame(records)
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"Fetched {len(df)} days of stablecoin data")
        return df
    except Exception as e:
        logger.warning(f"Stablecoin mcap fetch failed: {e}")
        return pd.DataFrame()


# ============================================================================
# FEATURE ENGINEERING (Same as scheduled_retrain but with historical on-chain)
# ============================================================================

def build_historical_features(ohlcv: pd.DataFrame, fg: pd.DataFrame,
                              tvl: pd.DataFrame, stables: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from real historical data.
    Merges OHLCV with Fear&Greed, TVL, and stablecoin data by date.
    """
    df = ohlcv.copy()
    df['date'] = df['timestamp'].dt.date

    # Merge Fear & Greed (by date, forward-fill for hourly)
    if not fg.empty:
        df = pd.merge(df, fg, on='date', how='left')
        df['fear_greed'] = df['fear_greed'].fillna(method='ffill').fillna(50)
    else:
        df['fear_greed'] = 50
        df['fg_classification'] = 'Neutral'

    # Merge TVL (by date)
    if not tvl.empty:
        df = pd.merge(df, tvl, on='date', how='left')
        df['total_tvl'] = df['total_tvl'].fillna(method='ffill').fillna(0)
    else:
        df['total_tvl'] = 0

    # Merge stablecoin mcap (by date)
    if not stables.empty:
        df = pd.merge(df, stables, on='date', how='left')
        df['stablecoin_mcap'] = df['stablecoin_mcap'].fillna(method='ffill').fillna(0)
    else:
        df['stablecoin_mcap'] = 0

    # ── Build features ──
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    returns = close.pct_change()

    feat = pd.DataFrame(index=df.index)

    # Price action (8)
    for w in [5, 10, 20, 50]:
        feat[f'sma_ratio_{w}'] = close / close.rolling(w).mean()
    feat['ema_12'] = close.ewm(span=12).mean()
    feat['ema_26'] = close.ewm(span=26).mean()
    feat['macd'] = feat['ema_12'] - feat['ema_26']
    feat['macd_signal'] = feat['macd'].ewm(span=9).mean()

    # Volatility (7)
    feat['returns_1h'] = returns
    feat['returns_4h'] = returns.rolling(4).sum()
    feat['returns_24h'] = returns.rolling(24).sum()
    feat['vol_20'] = returns.rolling(20).std()
    feat['vol_50'] = returns.rolling(50).std()
    feat['vol_ratio'] = feat['vol_20'] / (feat['vol_50'] + 1e-10)

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
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
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    feat['bb_width'] = (4 * std_20) / (close + 1e-10)
    feat['bb_position'] = (close - (sma_20 - 2 * std_20)) / (4 * std_20 + 1e-10)

    # Volume (4)
    feat['volume_ratio'] = volume / (volume.rolling(20).mean() + 1e-10)
    feat['volume_trend'] = volume.rolling(10).mean() / (volume.rolling(50).mean() + 1e-10)
    feat['obv'] = (np.sign(returns) * volume).cumsum()
    feat['obv_slope'] = feat['obv'].diff(10) / (feat['obv'].shift(10).abs() + 1e-10)

    # Momentum (4)
    feat['momentum_10'] = close - close.shift(10)
    feat['roc_10'] = returns.rolling(10).sum()
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    feat['stoch_k'] = 100 * (close - low_14) / (high_14 - low_14 + 1e-10)
    feat['stoch_d'] = feat['stoch_k'].rolling(3).mean()

    # Microstructure (3)
    feat['high_low_ratio'] = high / (low + 1e-10)
    feat['close_open_ratio'] = close / (df['open'] + 1e-10)
    feat['body_shadow_ratio'] = (close - df['open']).abs() / (high - low + 1e-10)

    # Time (2)
    feat['hour'] = df['timestamp'].dt.hour
    feat['day_of_week'] = df['timestamp'].dt.dayofweek

    # ── On-chain features (real historical) ──
    # Fear & Greed (3)
    feat['fear_greed_norm'] = df['fear_greed'] / 100
    feat['fg_momentum'] = df['fear_greed'].diff() / 100
    feat['fg_extreme'] = ((df['fear_greed'] < 20) | (df['fear_greed'] > 80)).astype(int)

    # DeFi TVL (2)
    tvl_series = df['total_tvl']
    feat['tvl_change_pct'] = tvl_series.pct_change()
    feat['tvl_trend'] = tvl_series.rolling(7 * 24).mean() / (tvl_series.rolling(30 * 24).mean() + 1e-10)

    # Stablecoin flows (2)
    stable_series = df['stablecoin_mcap']
    feat['stable_mcap_change'] = stable_series.pct_change()
    feat['stable_btc_ratio'] = stable_series / (close * 21e6 + 1e-10)  # vs BTC total market cap proxy

    # Store metadata columns for backtest
    feat['_close'] = close
    feat['_high'] = high
    feat['_low'] = low
    feat['_volume'] = volume
    feat['_timestamp'] = df['timestamp']

    return feat


def create_labels(closes: pd.Series, threshold: float = 0.001,
                  forward_bars: int = 4) -> pd.Series:
    """Directional labels: +1 LONG, 0 FLAT, -1 SHORT."""
    future_ret = closes.shift(-forward_bars) / closes - 1
    labels = pd.Series(0, index=closes.index)
    labels[future_ret > threshold] = 1
    labels[future_ret < -threshold] = -1
    return labels


# ============================================================================
# WALK-FORWARD BACKTESTING ENGINE
# ============================================================================

@dataclass
class WindowResult:
    """Result from a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    accuracy: float
    high_conf_accuracy: float
    total_trades: int
    win_rate: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_trade_pnl: float


@dataclass
class BacktestReport:
    """Aggregated walk-forward backtest results."""
    symbol: str
    months: int
    total_bars: int
    n_windows: int
    windows: List[WindowResult] = field(default_factory=list)

    # Aggregated out-of-sample metrics
    oos_accuracy: float = 0.0
    oos_high_conf_accuracy: float = 0.0
    oos_total_trades: int = 0
    oos_win_rate: float = 0.0
    oos_total_return_pct: float = 0.0
    oos_avg_sharpe: float = 0.0
    oos_worst_drawdown_pct: float = 0.0
    oos_consistency: float = 0.0  # % of windows with positive return
    oos_avg_daily_return: float = 0.0
    oos_days_to_1pct_target: float = 0.0  # How many days avg to hit 1% target


def walk_forward_backtest(features: pd.DataFrame, n_windows: int = 6,
                          min_train_bars: int = 2000,
                          label_threshold: float = 0.001) -> List[WindowResult]:
    """
    Expanding-window walk-forward backtest.

    For each window:
      1. Train LightGBM on all data up to window start
      2. Generate predictions on the test window
      3. Simulate trading with the predictions
      4. Record out-of-sample performance
    """
    if not HAS_LGB:
        logger.error("lightgbm required for backtesting")
        return []

    # Separate feature columns from metadata
    meta_cols = [c for c in features.columns if c.startswith('_')]
    feat_cols = [c for c in features.columns if not c.startswith('_')]

    closes = features['_close']
    highs = features['_high']
    lows = features['_low']
    timestamps = features['_timestamp']

    # Create labels
    labels = create_labels(closes, label_threshold)
    labels_lgb = labels.map({-1: 0, 0: 1, 1: 2})  # Map for LGB multiclass

    # Clean data
    X_all = features[feat_cols].fillna(0).values
    y_all = labels_lgb.values

    n = len(X_all)
    test_size = (n - min_train_bars) // n_windows
    if test_size < 100:
        logger.error(f"Not enough data for {n_windows} windows (need {min_train_bars + n_windows * 100} bars)")
        return []

    results = []

    for w in range(n_windows):
        train_end = min_train_bars + w * test_size
        test_start = train_end
        test_end = min(test_start + test_size, n)

        if test_end <= test_start:
            break

        X_train = X_all[:train_end]
        y_train = y_all[:train_end]
        X_test = X_all[test_start:test_end]
        y_test = y_all[test_start:test_end]

        test_closes = closes.iloc[test_start:test_end].values
        test_ts = timestamps.iloc[test_start:test_end]

        logger.info(f"  Window {w+1}/{n_windows}: train={train_end} bars, "
                    f"test={test_end-test_start} bars "
                    f"({test_ts.iloc[0].date()} to {test_ts.iloc[-1].date()})")

        # Train LightGBM
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'verbosity': -1,
            'num_leaves': 63,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 10,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
        }

        # Use last 15% of train as early-stopping validation
        val_size = int(len(X_train) * 0.15)
        dtrain = lgb.Dataset(X_train[:-val_size], label=y_train[:-val_size])
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:], reference=dtrain)

        model = lgb.train(
            params, dtrain,
            num_boost_round=400,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
        )

        # Predict
        y_pred_proba = model.predict(X_test)
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        max_probs = np.max(y_pred_proba, axis=1)

        accuracy = float(np.mean(y_pred_class == y_test))

        # High-confidence accuracy
        high_conf_mask = max_probs > 0.6
        high_conf_acc = float(np.mean(y_pred_class[high_conf_mask] == y_test[high_conf_mask])) \
            if high_conf_mask.sum() > 10 else accuracy

        # Map predictions back to signals: 0=SHORT(-1), 1=FLAT(0), 2=LONG(+1)
        signals = y_pred_class.copy().astype(int)
        signals[signals == 0] = -1
        signals[signals == 1] = 0
        signals[signals == 2] = 1

        # Only trade high-confidence signals
        signals[max_probs < 0.6] = 0

        # Simulate trading
        trade_metrics = _simulate_trades(test_closes, signals, max_probs)

        window_result = WindowResult(
            window_id=w + 1,
            train_start=str(timestamps.iloc[0].date()),
            train_end=str(timestamps.iloc[train_end - 1].date()),
            test_start=str(test_ts.iloc[0].date()),
            test_end=str(test_ts.iloc[-1].date()),
            train_samples=train_end,
            test_samples=test_end - test_start,
            accuracy=accuracy,
            high_conf_accuracy=high_conf_acc,
            **trade_metrics,
        )
        results.append(window_result)

        logger.info(f"    Accuracy: {accuracy:.4f} | High-conf: {high_conf_acc:.4f} | "
                    f"Return: {trade_metrics['total_return_pct']:.2f}% | "
                    f"Win rate: {trade_metrics['win_rate']:.2f}")

    return results


def _simulate_trades(closes: np.ndarray, signals: np.ndarray,
                     confidences: np.ndarray,
                     initial_capital: float = 100000.0,
                     fee_pct: float = 0.0004,  # 0.04% maker fee
                     risk_pct: float = 0.01) -> Dict:
    """
    Simple trade simulation for backtest windows.
    Position sizing: risk_pct of capital per trade.
    """
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trades = []
    equity_curve = [capital]

    for i in range(len(closes)):
        price = closes[i]
        signal = signals[i]

        # Close existing position on opposite signal or flat
        if position != 0 and (signal == -np.sign(position) or signal == 0):
            exit_pnl = position * (price - entry_price)
            fee = abs(position * price * fee_pct)
            net_pnl = exit_pnl - fee
            capital += net_pnl
            trades.append({
                'direction': int(np.sign(position)),
                'entry': entry_price,
                'exit': price,
                'pnl': net_pnl,
            })
            position = 0.0

        # Open new position
        if signal != 0 and position == 0:
            size_usd = capital * risk_pct
            position = (size_usd / price) * signal
            entry_price = price
            fee = abs(position * price * fee_pct)
            capital -= fee

        # Track equity
        unrealized = position * (price - entry_price) if position != 0 else 0
        equity_curve.append(capital + unrealized)

    # Close any remaining position
    if position != 0:
        exit_pnl = position * (closes[-1] - entry_price)
        fee = abs(position * closes[-1] * fee_pct)
        capital += exit_pnl - fee
        trades.append({
            'direction': int(np.sign(position)),
            'entry': entry_price,
            'exit': closes[-1],
            'pnl': exit_pnl - fee,
        })

    # Compute metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0

    # Sharpe from equity curve
    eq = np.array(equity_curve)
    daily_returns = np.diff(eq) / eq[:-1]
    sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252 * 24)) \
        if len(daily_returns) > 1 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / (peak + 1e-10) * 100
    max_dd = float(np.max(drawdown))

    avg_pnl = float(np.mean([t['pnl'] for t in trades])) if trades else 0

    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_dd,
        'avg_trade_pnl': avg_pnl,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(symbol: str, months: int, total_bars: int,
                    windows: List[WindowResult]) -> BacktestReport:
    """Aggregate window results into a full report."""
    report = BacktestReport(
        symbol=symbol,
        months=months,
        total_bars=total_bars,
        n_windows=len(windows),
        windows=windows,
    )

    if not windows:
        return report

    # Aggregate metrics
    accs = [w.accuracy for w in windows]
    hc_accs = [w.high_conf_accuracy for w in windows]
    returns = [w.total_return_pct for w in windows]
    sharpes = [w.sharpe_ratio for w in windows]
    dds = [w.max_drawdown_pct for w in windows]

    report.oos_accuracy = float(np.mean(accs))
    report.oos_high_conf_accuracy = float(np.mean(hc_accs))
    report.oos_total_trades = sum(w.total_trades for w in windows)
    report.oos_win_rate = float(np.mean([w.win_rate for w in windows]))
    report.oos_total_return_pct = float(np.sum(returns))  # Cumulative
    report.oos_avg_sharpe = float(np.mean(sharpes))
    report.oos_worst_drawdown_pct = float(np.max(dds))
    report.oos_consistency = float(np.mean([1 if r > 0 else 0 for r in returns]))

    # Days to 1% target estimation
    if report.oos_total_return_pct > 0:
        total_test_bars = sum(w.test_samples for w in windows)
        total_test_days = total_test_bars / 24
        daily_return = report.oos_total_return_pct / total_test_days if total_test_days > 0 else 0
        report.oos_avg_daily_return = daily_return
        report.oos_days_to_1pct_target = 1.0 / daily_return if daily_return > 0 else float('inf')
    else:
        report.oos_avg_daily_return = 0
        report.oos_days_to_1pct_target = float('inf')

    return report


def print_report(report: BacktestReport):
    """Pretty-print the backtest report."""
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD BACKTEST REPORT: {report.symbol}")
    print(f"{'='*70}")
    print(f"  Data: {report.total_bars} bars ({report.months} months)")
    print(f"  Windows: {report.n_windows}")

    print(f"\n{'─'*70}")
    print("  Per-Window Results:")
    print(f"  {'Window':>6} {'Accuracy':>9} {'HighConf':>9} {'Return%':>9} {'WinRate':>8} {'Sharpe':>7} {'MaxDD%':>7}")
    print(f"  {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*7}")
    for w in report.windows:
        print(f"  {w.window_id:>6} {w.accuracy:>9.4f} {w.high_conf_accuracy:>9.4f} "
              f"{w.total_return_pct:>+9.2f} {w.win_rate:>8.2f} {w.sharpe_ratio:>7.2f} {w.max_drawdown_pct:>7.2f}")

    print(f"\n{'─'*70}")
    print("  Aggregated Out-of-Sample Results:")
    print(f"    Accuracy:            {report.oos_accuracy:.4f}")
    print(f"    High-Conf Accuracy:  {report.oos_high_conf_accuracy:.4f}")
    print(f"    Total Trades:        {report.oos_total_trades}")
    print(f"    Win Rate:            {report.oos_win_rate:.2%}")
    print(f"    Total Return:        {report.oos_total_return_pct:+.2f}%")
    print(f"    Avg Sharpe Ratio:    {report.oos_avg_sharpe:.2f}")
    print(f"    Worst Drawdown:      {report.oos_worst_drawdown_pct:.2f}%")
    print(f"    Consistency:         {report.oos_consistency:.0%} windows profitable")

    print(f"\n  1% Daily Target Analysis:")
    print(f"    Avg Daily Return:    {report.oos_avg_daily_return:.4f}%")
    if report.oos_days_to_1pct_target < float('inf'):
        print(f"    Days to 1% Target:   {report.oos_days_to_1pct_target:.1f} days")
        if report.oos_avg_daily_return >= 1.0:
            print(f"    Status:              TARGET MET (avg {report.oos_avg_daily_return:.2f}%/day)")
        elif report.oos_avg_daily_return >= 0.5:
            print(f"    Status:              ON TRACK (need {1.0/report.oos_avg_daily_return:.1f}x improvement)")
        else:
            print(f"    Status:              NEEDS IMPROVEMENT")
    else:
        print(f"    Status:              NEGATIVE RETURNS — needs work")

    print(f"\n{'='*70}\n")


def save_report(report: BacktestReport, output_dir: str = 'logs'):
    """Save report as JSON for dashboard consumption."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'backtest_{report.symbol.lower()}_{datetime.utcnow().strftime("%Y%m%d")}.json')

    data = {
        'symbol': report.symbol,
        'months': report.months,
        'total_bars': report.total_bars,
        'n_windows': report.n_windows,
        'oos_accuracy': report.oos_accuracy,
        'oos_high_conf_accuracy': report.oos_high_conf_accuracy,
        'oos_total_trades': report.oos_total_trades,
        'oos_win_rate': report.oos_win_rate,
        'oos_total_return_pct': report.oos_total_return_pct,
        'oos_avg_sharpe': report.oos_avg_sharpe,
        'oos_worst_drawdown_pct': report.oos_worst_drawdown_pct,
        'oos_consistency': report.oos_consistency,
        'oos_avg_daily_return': report.oos_avg_daily_return,
        'oos_days_to_1pct_target': report.oos_days_to_1pct_target,
        'generated_at': datetime.utcnow().isoformat(),
        'windows': [asdict(w) for w in report.windows],
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    logger.info(f"Report saved to {filepath}")
    return filepath


# ============================================================================
# MAIN
# ============================================================================

def run_historical_backtest(symbol: str = 'BTC/USDT', months: int = 6,
                            n_windows: int = 6) -> BacktestReport:
    """
    Full historical backtesting pipeline:
      1. Fetch real data from multiple sources
      2. Build features from historical on-chain + price data
      3. Walk-forward train/test
      4. Generate report
    """
    logger.info(f"Starting historical backtest for {symbol} ({months} months, {n_windows} windows)")

    # 1. Fetch all data concurrently
    logger.info("Fetching historical data from real sources...")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix='hist') as pool:
        f_ohlcv = pool.submit(fetch_historical_ohlcv, symbol, months)
        f_fg = pool.submit(fetch_historical_fear_greed, months * 30)
        f_tvl = pool.submit(fetch_historical_defillama_tvl)
        f_stables = pool.submit(fetch_historical_stablecoin_mcap)

        ohlcv = f_ohlcv.result(timeout=120)
        fg = f_fg.result(timeout=30)
        tvl = f_tvl.result(timeout=30)
        stables = f_stables.result(timeout=30)

    if ohlcv.empty:
        logger.error("No OHLCV data — cannot backtest")
        return BacktestReport(symbol=symbol, months=months, total_bars=0, n_windows=0)

    # 2. Build features
    logger.info("Building features from real historical data...")
    features = build_historical_features(ohlcv, fg, tvl, stables)
    features = features.dropna(subset=[c for c in features.columns if not c.startswith('_')])

    logger.info(f"Feature matrix: {len(features)} samples x "
                f"{len([c for c in features.columns if not c.startswith('_')])} features")

    # 3. Walk-forward backtest
    logger.info(f"Running {n_windows}-window walk-forward backtest...")
    windows = walk_forward_backtest(features, n_windows=n_windows)

    # 4. Generate report
    report = generate_report(symbol, months, len(ohlcv), windows)
    print_report(report)
    save_report(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Historical backtest with real data feeds")
    parser.add_argument('--symbol', default='BTC', help='Asset to backtest (BTC, ETH, AAVE)')
    parser.add_argument('--months', type=int, default=6, help='Months of historical data')
    parser.add_argument('--windows', type=int, default=6, help='Walk-forward windows')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    symbol_map = {'BTC': 'BTC/USDT', 'ETH': 'ETH/USDT', 'AAVE': 'AAVE/USDT'}
    symbol = symbol_map.get(args.symbol.upper(), f'{args.symbol.upper()}/USDT')

    run_historical_backtest(symbol, args.months, args.windows)


if __name__ == '__main__':
    main()
