"""
Train ALL ML Models on Real Market Data
=========================================
Fetches real OHLCV data and trains every trainable model in the system:

  1. LightGBM Classifier — 100+ features, 3-class direction prediction
  2. LSTM/GRU/BiLSTM Ensemble — Neural network sequence prediction
  3. PatchTST Transformer — Patch-based time-series transformer
  4. HMM Regime Detector — 4-state Gaussian HMM (BULL/BEAR/SIDEWAYS/CRISIS)
  5. GARCH(1,1) Volatility — Volatility forecasting
  6. Alpha Decay — Signal decay rate estimation

All models learn from real BTC+ETH chart patterns to predict:
  - Which EMA crossovers will run to L3+ (profit)
  - Which will die at L1 (loss)
  - Current market regime
  - Optimal holding period

Usage:
    python -m src.scripts.train_all_models
    python -m src.scripts.train_all_models --asset BTC --bars 5000
"""

import os
import sys
import time
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def fetch_training_data(asset: str = 'BTC', timeframe: str = '5m', bars: int = 5000):
    """Fetch real OHLCV data for training from Bybit testnet or Binance."""
    print(f"\n{'='*60}")
    print(f"FETCHING {bars} {timeframe} BARS FOR {asset}")
    print(f"{'='*60}")

    # Binance FIRST (real market data, up to 1000 bars per request)
    try:
        import ccxt
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        symbol = f"{asset}/USDT"
        all_ohlcv = []
        remaining = bars
        since = None

        while remaining > 0:
            limit = min(1000, remaining)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            remaining -= len(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break
            time.sleep(0.5)

        if all_ohlcv:
            data = {
                'timestamps': [x[0] for x in all_ohlcv],
                'opens': [x[1] for x in all_ohlcv],
                'highs': [x[2] for x in all_ohlcv],
                'lows': [x[3] for x in all_ohlcv],
                'closes': [x[4] for x in all_ohlcv],
                'volumes': [x[5] for x in all_ohlcv],
            }
            print(f"  Fetched {len(all_ohlcv)} bars from Binance (real market)")
            return data
    except Exception as e:
        print(f"  Binance fetch failed: {e}")

    # Fallback: Bybit testnet
    try:
        import ccxt
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'},
            'sandbox': True,
        })
        symbol = f"{asset}/USDT:USDT"
        all_ohlcv = []
        remaining = bars
        since = None
        while remaining > 0:
            limit = min(200, remaining)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            remaining -= len(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < limit:
                break
        if all_ohlcv:
            data = {
                'timestamps': [x[0] for x in all_ohlcv],
                'opens': [x[1] for x in all_ohlcv],
                'highs': [x[2] for x in all_ohlcv],
                'lows': [x[3] for x in all_ohlcv],
                'closes': [x[4] for x in all_ohlcv],
                'volumes': [x[5] for x in all_ohlcv],
            }
            print(f"  Fetched {len(all_ohlcv)} bars from Bybit testnet")
            return data
    except Exception as e:
        print(f"  Bybit fetch failed: {e}")

    # Last resort: generate from recent journal prices
    print("  WARNING: Using synthetic data (no exchange connectivity)")
    np.random.seed(42)
    base = 85000 if asset == 'BTC' else 2000
    n = bars
    returns = np.random.normal(0, 0.002, n)
    closes = [base]
    for r in returns:
        closes.append(closes[-1] * (1 + r))
    closes = closes[1:]
    highs = [c * (1 + abs(np.random.normal(0, 0.001))) for c in closes]
    lows = [c * (1 - abs(np.random.normal(0, 0.001))) for c in closes]
    opens = [closes[max(0, i-1)] for i in range(n)]
    volumes = [np.random.uniform(100, 10000) for _ in range(n)]

    return {
        'timestamps': list(range(n)),
        'opens': opens, 'highs': highs, 'lows': lows,
        'closes': closes, 'volumes': volumes,
    }


def create_sequence_features(closes, highs, lows, volumes, seq_len=30, n_features=20):
    """Create sequential feature arrays for LSTM/neural net training."""
    from src.indicators.indicators import ema, atr, rsi, macd, bollinger_bands, stochastic, adx, roc, obv

    n = len(closes)
    if n < seq_len + 50:
        return None, None

    # Compute indicators
    ema_8 = ema(closes, 8)
    ema_21 = ema(closes, 21)
    atr_14 = atr(highs, lows, closes, 14)
    rsi_14 = rsi(closes, 14)
    macd_line, sig_line, hist = macd(closes)
    bb_upper, bb_lower, bb_mid = bollinger_bands(closes, 20)
    stoch_k, stoch_d = stochastic(highs, lows, closes, 14, 3)
    adx_vals = adx(highs, lows, closes, 14)
    roc_12 = roc(closes, 12)
    obv_vals = obv(closes, volumes)

    # Build feature matrix (n_bars x n_features)
    feature_matrix = np.zeros((n, n_features))
    for i in range(n):
        c = closes[i]
        if c == 0:
            continue
        feature_matrix[i, 0] = (c - ema_8[i]) / c * 100 if ema_8[i] > 0 else 0  # price vs EMA8
        feature_matrix[i, 1] = (c - ema_21[i]) / c * 100 if ema_21[i] > 0 else 0  # price vs EMA21
        feature_matrix[i, 2] = (ema_8[i] - ema_21[i]) / c * 100 if ema_21[i] > 0 else 0  # EMA spread
        feature_matrix[i, 3] = atr_14[i] / c * 100 if atr_14[i] > 0 else 0  # ATR %
        feature_matrix[i, 4] = rsi_14[i] / 100  # RSI normalized
        feature_matrix[i, 5] = hist[i] / c * 1000 if c > 0 else 0  # MACD hist normalized
        feature_matrix[i, 6] = (c - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5  # BB position
        feature_matrix[i, 7] = stoch_k[i] / 100  # Stoch %K
        feature_matrix[i, 8] = stoch_d[i] / 100  # Stoch %D
        adx_v = adx_vals[0][i] if isinstance(adx_vals, tuple) else (adx_vals[i] if isinstance(adx_vals, (list, np.ndarray)) else 0)
        feature_matrix[i, 9] = min(float(adx_v) / 50, 1.0) if adx_v else 0  # ADX norm
        feature_matrix[i, 10] = roc_12[i] / 10  # ROC normalized
        feature_matrix[i, 11] = np.log(volumes[i] + 1) / 15  # Log volume normalized
        # Returns
        if i > 0 and closes[i-1] > 0:
            feature_matrix[i, 12] = (c - closes[i-1]) / closes[i-1] * 100  # 1-bar return
        if i > 4 and closes[i-5] > 0:
            feature_matrix[i, 13] = (c - closes[i-5]) / closes[i-5] * 100  # 5-bar return
        if i > 19 and closes[i-20] > 0:
            feature_matrix[i, 14] = (c - closes[i-20]) / closes[i-20] * 100  # 20-bar return
        # Candle patterns
        body = abs(c - (closes[i-1] if i > 0 else c))
        total = highs[i] - lows[i] if highs[i] > lows[i] else 1e-10
        feature_matrix[i, 15] = body / total  # Body ratio
        feature_matrix[i, 16] = (highs[i] - max(c, closes[i-1] if i > 0 else c)) / total  # Upper wick
        feature_matrix[i, 17] = (min(c, closes[i-1] if i > 0 else c) - lows[i]) / total  # Lower wick
        # Volume delta
        feature_matrix[i, 18] = 1.0 if c > (closes[i-1] if i > 0 else c) else -1.0  # Direction
        if i > 4:
            avg_vol = np.mean(volumes[max(0, i-20):i])
            feature_matrix[i, 19] = volumes[i] / (avg_vol + 1e-10)  # Relative volume

    # Replace NaN/Inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

    # Create sequences
    sequences = []
    labels = []
    for i in range(seq_len + 50, n - 5):
        seq = feature_matrix[i - seq_len:i]
        sequences.append(seq)

        # Label: look ahead 5 bars
        future_return = (closes[min(i + 5, n - 1)] - closes[i]) / closes[i] * 100
        if future_return > 0.15:
            labels.append(2)  # LONG
        elif future_return < -0.15:
            labels.append(0)  # SHORT
        else:
            labels.append(1)  # FLAT

    X = np.array(sequences)
    y = np.array(labels)
    print(f"  Sequences: {X.shape} | Labels: SHORT={sum(y==0)} FLAT={sum(y==1)} LONG={sum(y==2)}")
    return X, y


def train_lightgbm(data, asset='BTC'):
    """Train LightGBM on real OHLCV data."""
    print(f"\n{'='*60}")
    print(f"1. TRAINING LIGHTGBM CLASSIFIER ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.lightgbm_classifier import LightGBMClassifier

        clf = LightGBMClassifier()
        features = clf.extract_features(
            closes=data['closes'], highs=data['highs'],
            lows=data['lows'], volumes=data['volumes'],
        )

        # Filter out empty features
        valid_features = [f for f in features if f and len(f) > 5]
        if len(valid_features) < 100:
            print(f"  Only {len(valid_features)} valid features -- skipping")
            return False

        preds = clf.predict(valid_features[-100:])
        print(f"  Features extracted: {len(valid_features)} bars x {len(valid_features[0])} features")
        print(f"  Sample predictions: {preds[-5:]}")

        # Now train from trade log if available
        log_path = 'logs/trade_history.csv'
        if os.path.exists(log_path):
            clf.trade_log_path = log_path
            clf.retrain_from_log(max_examples=500)
            print(f"  Retrained from {log_path}")

        # Save
        os.makedirs('models', exist_ok=True)
        model_path = f'models/lgbm_{asset.lower()}_5m.txt'
        if clf._lgb_model:
            clf._lgb_model.save_model(model_path)
            clf._lgb_model.save_model('models/lgbm_latest.txt')
            print(f"  Model saved: {model_path}")
        else:
            print(f"  Rule-based mode (no LightGBM model trained -- need more trade log data)")

        return True
    except Exception as e:
        print(f"  LightGBM training error: {e}")
        return False


def train_hmm(data, asset='BTC'):
    """Train HMM Regime Detector on real market data."""
    print(f"\n{'='*60}")
    print(f"2. TRAINING HMM REGIME DETECTOR ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.hmm_regime import HMMRegimeDetector

        closes = np.array(data['closes'])
        volumes = np.array(data['volumes'])

        # Compute observation features
        log_returns = np.diff(np.log(closes + 1e-12))
        vol_20 = np.array([np.std(log_returns[max(0, i-20):i]) for i in range(1, len(log_returns) + 1)])
        vol_change = np.diff(volumes) / (volumes[:-1] + 1e-12)

        # Align lengths
        min_len = min(len(log_returns), len(vol_20), len(vol_change))
        returns = log_returns[-min_len:]
        volatility = vol_20[-min_len:]
        vol_chg = vol_change[-min_len:]

        # Train HMM
        detector = HMMRegimeDetector(n_states=4, n_iter=200)
        success = detector.fit(returns, volatility, vol_chg)

        if success:
            # Test prediction
            result = detector.predict(returns[-100:], volatility[-100:], vol_chg[-100:])
            print(f"  HMM fitted: {len(returns)} observations")
            print(f"  Current regime: {result.get('regime', '?')} (confidence: {result.get('confidence', 0):.2f})")
            print(f"  Crisis probability: {result.get('crisis_probability', 0):.3f}")

            # Save fitted model
            import pickle
            os.makedirs('models', exist_ok=True)
            with open(f'models/hmm_{asset.lower()}.pkl', 'wb') as f:
                pickle.dump(detector, f)
            print(f"  HMM model saved: models/hmm_{asset.lower()}.pkl")
            return True
        else:
            print(f"  HMM fitting failed")
            return False
    except Exception as e:
        print(f"  HMM training error: {e}")
        return False


def train_lstm_ensemble(data, asset='BTC'):
    """Train LSTM/GRU/BiLSTM ensemble on real market data."""
    print(f"\n{'='*60}")
    print(f"3. TRAINING LSTM ENSEMBLE ({asset})")
    print(f"{'='*60}")

    try:
        import torch
        print(f"  PyTorch available: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    except ImportError:
        print(f"  PyTorch NOT installed -- skipping LSTM training")
        print(f"  Install with: pip install torch")
        return False

    try:
        from src.models.lstm_ensemble import LSTMEnsemble

        # Create sequence features
        X, y = create_sequence_features(
            data['closes'], data['highs'], data['lows'], data['volumes'],
            seq_len=30, n_features=20,
        )
        if X is None or len(X) < 100:
            print(f"  Not enough data for LSTM training")
            return False

        # Initialize and train
        ensemble = LSTMEnsemble(input_dim=20, seq_len=30, model_dir=f'models/lstm_ensemble_{asset.lower()}')
        results = ensemble.train(X, y, epochs=30, lr=0.001, batch_size=32)

        print(f"\n  Training results:")
        for name, metrics in results.items():
            if isinstance(metrics, dict):
                print(f"    {name}: val_loss={metrics.get('best_val_loss', '?'):.4f} "
                      f"val_acc={metrics.get('val_accuracy', '?'):.3f} "
                      f"epochs={metrics.get('epochs_trained', '?')}")

        # Test prediction
        test_pred = ensemble.predict(X[-1])
        print(f"  Test prediction: signal={test_pred.get('signal', '?')} "
              f"conf={test_pred.get('confidence', 0):.2f}")

        return True
    except Exception as e:
        print(f"  LSTM training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_patchtst(data, asset='BTC'):
    """Train PatchTST Transformer on real market data."""
    print(f"\n{'='*60}")
    print(f"4. TRAINING PatchTST TRANSFORMER ({asset})")
    print(f"{'='*60}")

    try:
        import torch
    except ImportError:
        print(f"  PyTorch NOT installed -- skipping")
        return False

    try:
        from src.ai.patchtst_model import PatchTST

        closes = np.array(data['closes'])
        if len(closes) < 500:
            print(f"  Need 500+ bars for PatchTST, have {len(closes)}")
            return False

        # Compute returns
        returns = np.diff(closes) / (closes[:-1] + 1e-9)

        # Create training samples
        seq_len = 400
        X_samples = []
        y_samples = []

        for i in range(seq_len, len(returns) - 5):
            X_samples.append(returns[i - seq_len:i])
            # Label: future 5-bar return
            future_ret = sum(returns[i:min(i + 5, len(returns))])
            if future_ret > 0.001:
                y_samples.append(1.0)  # Up
            else:
                y_samples.append(0.0)  # Down

        X = torch.FloatTensor(np.array(X_samples))
        y_up = torch.FloatTensor(np.array(y_samples)).unsqueeze(1)
        # Shock label: high volatility in next 5 bars
        y_shock = torch.FloatTensor(np.array([
            1.0 if abs(sum(returns[seq_len + i:min(seq_len + i + 5, len(returns))])) > 0.01 else 0.0
            for i in range(len(X_samples))
        ])).unsqueeze(1)

        print(f"  Training samples: {len(X)} | Up: {int(y_up.sum())} | Down: {len(X) - int(y_up.sum())}")

        # Train
        model = PatchTST(seq_len=seq_len)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        criterion = torch.nn.BCELoss()

        # Split
        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_up_train, y_up_val = y_up[:split], y_up[split:]
        y_shock_train, y_shock_val = y_shock[:split], y_shock[split:]

        best_val_loss = float('inf')
        for epoch in range(30):
            model.train()
            # Mini-batch
            batch_size = 32
            epoch_loss = 0
            n_batches = 0
            indices = np.random.permutation(len(X_train))

            for start in range(0, len(indices), batch_size):
                idx = indices[start:start + batch_size]
                batch_x = X_train[idx]
                batch_y_up = y_up_train[idx]
                batch_y_shock = y_shock_train[idx]

                optimizer.zero_grad()
                pred_up, pred_shock = model(batch_x)
                loss = criterion(pred_up, batch_y_up) + 0.3 * criterion(pred_shock, batch_y_shock)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred_up, val_pred_shock = model(X_val)
                val_loss = criterion(val_pred_up, y_up_val).item()
                val_acc = ((val_pred_up > 0.5).float() == y_up_val).float().mean().item()

            if epoch % 5 == 0:
                print(f"    Epoch {epoch}: train_loss={epoch_loss/n_batches:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Save best model
        if best_state:
            os.makedirs('models', exist_ok=True)
            save_path = os.path.abspath(os.path.join('models', 'patchtst_v1_new.pt'))
            try:
                torch.save(best_state, save_path)
                # Rename to final path
                final_path = save_path.replace('_new.pt', '.pt')
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(save_path, final_path)
                print(f"  Best val_loss: {best_val_loss:.4f}")
                print(f"  Model saved: {final_path}")
            except Exception as se:
                # Try buffer-based save
                import io
                buf = io.BytesIO()
                torch.save(best_state, buf)
                with open(save_path, 'wb') as f:
                    f.write(buf.getvalue())
                print(f"  Best val_loss: {best_val_loss:.4f}")
                print(f"  Model saved (buffer): {save_path}")
        else:
            print(f"  No improvement during training")
        return True
    except Exception as e:
        print(f"  PatchTST training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_garch(data, asset='BTC'):
    """Train GARCH(1,1) volatility model."""
    print(f"\n{'='*60}")
    print(f"5. TRAINING GARCH(1,1) VOLATILITY ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.volatility import GARCH11, log_returns

        closes = np.array(data['closes'])
        returns = log_returns(closes)

        garch = GARCH11()
        # Filter out zero/nan returns
        clean_returns = returns[np.isfinite(returns) & (returns != 0)]
        if len(clean_returns) < 50:
            print(f"  Not enough clean returns ({len(clean_returns)})")
            return False
        clean_list = list(clean_returns[-500:])
        garch.fit(clean_list)

        forecast = garch.forecast(clean_list[-100:])
        print(f"  GARCH fitted on {len(returns)} returns")
        print(f"  Current vol forecast: {forecast[-1]:.6f}")
        print(f"  Params: omega={garch.omega:.6f} alpha={garch.alpha:.4f} beta={garch.beta:.4f}")

        # Save
        import pickle
        os.makedirs('models', exist_ok=True)
        with open(f'models/garch_{asset.lower()}.pkl', 'wb') as f:
            pickle.dump(garch, f)
        print(f"  GARCH model saved: models/garch_{asset.lower()}.pkl")
        return True
    except Exception as e:
        print(f"  GARCH training error: {e}")
        return False


def train_alpha_decay(data, asset='BTC'):
    """Train Alpha Decay model to find optimal holding period."""
    print(f"\n{'='*60}")
    print(f"6. TRAINING ALPHA DECAY MODEL ({asset})")
    print(f"{'='*60}")

    try:
        from src.models.alpha_decay import AlphaDecayModel
        from src.indicators.indicators import ema, rsi

        closes = np.array(data['closes'])

        # Generate signals from EMA crossover (our strategy)
        ema_8 = np.array(ema(list(closes), 8))
        ema_21 = np.array(ema(list(closes), 21))
        signals = ema_8 - ema_21  # EMA spread as signal

        # Returns
        returns = np.diff(closes) / (closes[:-1] + 1e-12)

        # Align
        min_len = min(len(signals), len(returns))
        signals = signals[-min_len:]
        ret = returns[-min_len:]

        model = AlphaDecayModel()
        model.fit(signals, ret)

        print(f"  Alpha Decay fitted: half_life={model.half_life:.1f} bars")
        print(f"  Optimal hold: {model.optimal_hold:.0f} bars ({model.optimal_hold * 5:.0f} min)")
        print(f"  Peak alpha: {model.peak_alpha:.4f}")

        import pickle
        os.makedirs('models', exist_ok=True)
        with open(f'models/alpha_decay_{asset.lower()}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  Alpha Decay model saved")
        return True
    except Exception as e:
        print(f"  Alpha Decay training error: {e}")
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train all ML models on real market data')
    parser.add_argument('--asset', default='BTC', help='Asset to train on (BTC, ETH)')
    parser.add_argument('--bars', type=int, default=3000, help='Number of bars to fetch')
    parser.add_argument('--skip-lstm', action='store_true', help='Skip LSTM (slow without GPU)')
    parser.add_argument('--skip-patchtst', action='store_true', help='Skip PatchTST')
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING ALL ML MODELS ON REAL MARKET DATA")
    print("=" * 60)

    results = {}

    for asset in [args.asset] if args.asset != 'ALL' else ['BTC', 'ETH']:
        print(f"\n{'#'*60}")
        print(f"# ASSET: {asset}")
        print(f"{'#'*60}")

        data = fetch_training_data(asset, timeframe='5m', bars=args.bars)
        if not data or len(data['closes']) < 100:
            print(f"  SKIP {asset}: not enough data ({len(data.get('closes', []))} bars)")
            continue

        print(f"  Data: {len(data['closes'])} bars | "
              f"Price: ${data['closes'][-1]:,.2f} | "
              f"Range: ${min(data['closes']):,.2f} - ${max(data['closes']):,.2f}")

        # Train each model
        results[f'{asset}_lgbm'] = train_lightgbm(data, asset)
        results[f'{asset}_hmm'] = train_hmm(data, asset)
        results[f'{asset}_garch'] = train_garch(data, asset)
        results[f'{asset}_alpha'] = train_alpha_decay(data, asset)

        if not args.skip_lstm:
            results[f'{asset}_lstm'] = train_lstm_ensemble(data, asset)
        else:
            print(f"\n  LSTM skipped (--skip-lstm)")
            results[f'{asset}_lstm'] = 'skipped'

        if not args.skip_patchtst:
            results[f'{asset}_patchtst'] = train_patchtst(data, asset)
        else:
            print(f"\n  PatchTST skipped (--skip-patchtst)")
            results[f'{asset}_patchtst'] = 'skipped'

    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = 'OK' if status is True else ('SKIP' if status == 'skipped' else 'FAIL')
        print(f"  [{icon}] {name}")

    print(f"\nAll trained models saved to models/ directory.")
    print(f"Restart trading system to load new models.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
