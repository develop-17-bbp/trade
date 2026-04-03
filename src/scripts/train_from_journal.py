"""
Train ML Models from Trading Journal — Learn YOUR Patterns
============================================================
Analyzes your actual trade history to learn:
  1. Which EMA crossovers become L3+ runners (profit) vs L1 deaths (loss)
  2. Which hours, assets, confidence levels are profitable
  3. Optimal thresholds for entry_score, confidence, bear veto

Then trains:
  - LightGBM classifier to predict trade quality BEFORE entry
  - Saves optimized config thresholds to config_optimized.yaml
  - Generates a report of findings

Usage:
    python -m src.scripts.train_from_journal
    python -m src.scripts.train_from_journal --journal logs/trading_journal.jsonl
"""

import json
import os
import sys
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def load_journal(path: str = 'logs/trading_journal.jsonl') -> List[Dict]:
    """Load and filter journal trades."""
    trades = []
    with open(path) as f:
        for line in f:
            try:
                t = json.loads(line.strip())
                trades.append(t)
            except Exception:
                pass
    return trades


def classify_trade(t: Dict) -> str:
    """Classify trade outcome for training labels."""
    pnl = float(t.get('pnl_usd', 0) or 0)
    sl_prog = t.get('sl_progression', '') or 'L1'
    n_levels = len(sl_prog.split('->'))

    if pnl == 0:
        return 'OPEN'  # Skip
    if n_levels >= 3 and pnl > 0:
        return 'RUNNER'  # L3+ winner — we WANT these
    elif n_levels <= 1 and pnl < 0:
        return 'L1_DEATH'  # Instant reversal — we want to BLOCK these
    elif pnl > 0:
        return 'SMALL_WIN'
    else:
        return 'SMALL_LOSS'


def extract_journal_features(t: Dict) -> Dict[str, float]:
    """Extract features from a journal trade for ML training."""
    features = {}

    # Basic trade info
    features['confidence'] = float(t.get('confidence', 0) or 0)
    features['duration_min'] = float(t.get('duration_minutes', 0) or 0)
    features['is_long'] = 1.0 if 'long' in (t.get('action', '') or '').lower() else 0.0
    features['is_btc'] = 1.0 if t.get('asset', '') == 'BTC' else 0.0

    # Time features
    ts = t.get('timestamp', '')
    if 'T' in ts:
        try:
            hour = int(ts.split('T')[1][:2])
            features['hour'] = hour
            features['is_asian_session'] = 1.0 if 0 <= hour < 8 else 0.0
            features['is_london_session'] = 1.0 if 8 <= hour < 16 else 0.0
            features['is_ny_session'] = 1.0 if 16 <= hour < 24 else 0.0
            # Profitable hours from our analysis: 06-13
            features['is_profitable_hour'] = 1.0 if 6 <= hour <= 13 else 0.0
            # Dangerous hours: 19-04
            features['is_dangerous_hour'] = 1.0 if hour >= 19 or hour <= 4 else 0.0
        except Exception:
            features['hour'] = 12
            features['is_asian_session'] = 0.0
            features['is_london_session'] = 1.0
            features['is_ny_session'] = 0.0
            features['is_profitable_hour'] = 0.0
            features['is_dangerous_hour'] = 0.0

    # Price features
    entry = float(t.get('entry_price', 0) or 0)
    exit_p = float(t.get('exit_price', 0) or 0)
    if entry > 0 and exit_p > 0:
        features['price_move_pct'] = (exit_p - entry) / entry * 100
    else:
        features['price_move_pct'] = 0.0

    # SL level progression
    sl_prog = t.get('sl_progression', '') or 'L1'
    features['n_sl_levels'] = float(len(sl_prog.split('->')))

    # Exit reason encoding
    reason = (t.get('exit_reason', '') or '').lower()
    features['exit_hard_stop'] = 1.0 if 'hard' in reason else 0.0
    features['exit_ema_reversal'] = 1.0 if 'ema' in reason or 'reversal' in reason or 'e1' in reason else 0.0
    features['exit_sl'] = 1.0 if 'sl' in reason else 0.0
    features['exit_time'] = 1.0 if 'time' in reason else 0.0

    # Extra fields from newer trades
    extra = t.get('extra', {}) or {}
    features['risk_score'] = float(extra.get('risk_score', 0) or 0)
    features['bear_risk'] = float(extra.get('bear_risk', 0) or 0)
    features['hurst'] = float(extra.get('hurst', 0.5) or 0.5)

    # Exchange
    features['is_bybit'] = 1.0 if t.get('exchange', '') == 'bybit' else 0.0

    return features


def compute_optimal_thresholds(trades: List[Dict]) -> Dict[str, Any]:
    """Grid search for optimal thresholds using journal data."""
    # Filter to real trades
    real = [t for t in trades
            if float(t.get('pnl_usd', 0) or 0) != 0
            and float(t.get('confidence', 0) or 0) > 0
            and abs(float(t.get('pnl_usd', 0))) < 5000]  # Remove outlier artifacts

    if not real:
        print("No real trades found!")
        return {}

    results = {}

    # 1. Optimal confidence threshold
    print("\n=== CONFIDENCE THRESHOLD OPTIMIZATION ===")
    best_conf = 0.60
    best_conf_pf = 0.0
    for conf_thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        filtered = [t for t in real if float(t.get('confidence', 0)) >= conf_thresh]
        if len(filtered) < 10:
            continue
        wins = sum(1 for t in filtered if float(t['pnl_usd']) > 0)
        losses = sum(1 for t in filtered if float(t['pnl_usd']) < 0)
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        win_sum = sum(float(t['pnl_usd']) for t in filtered if float(t['pnl_usd']) > 0)
        loss_sum = abs(sum(float(t['pnl_usd']) for t in filtered if float(t['pnl_usd']) < 0))
        pf = win_sum / loss_sum if loss_sum > 0 else 99
        net = sum(float(t['pnl_usd']) for t in filtered)
        print(f"  conf >= {conf_thresh:.2f}: {len(filtered):3d} trades | WR: {wr:.0f}% | PF: {pf:.2f} | Net: ${net:,.2f}")
        if pf > best_conf_pf and len(filtered) >= 10:
            best_conf_pf = pf
            best_conf = conf_thresh
    results['optimal_confidence'] = best_conf
    print(f"  -> BEST: {best_conf:.2f} (PF: {best_conf_pf:.2f})")

    # 2. Hour-of-day filter
    print("\n=== HOUR FILTER OPTIMIZATION ===")
    hour_pnl = defaultdict(float)
    hour_count = defaultdict(int)
    for t in real:
        ts = t.get('timestamp', '')
        if 'T' in ts:
            try:
                hour = int(ts.split('T')[1][:2])
                hour_pnl[hour] += float(t['pnl_usd'])
                hour_count[hour] += 1
            except Exception:
                pass

    profitable_hours = [h for h in range(24) if hour_pnl.get(h, 0) > 0]
    dangerous_hours = [h for h in range(24) if hour_pnl.get(h, 0) < -100 and hour_count.get(h, 0) >= 3]
    results['profitable_hours'] = profitable_hours
    results['dangerous_hours'] = dangerous_hours
    print(f"  Profitable hours: {profitable_hours}")
    print(f"  Dangerous hours (>$100 loss): {dangerous_hours}")

    # What if we ONLY traded profitable hours?
    hour_filtered = [t for t in real if any(
        'T' in t.get('timestamp', '') and int(t['timestamp'].split('T')[1][:2]) in profitable_hours
        for _ in [1]  # dummy loop for condition
    )]
    if hour_filtered:
        net = sum(float(t['pnl_usd']) for t in hour_filtered)
        wr = sum(1 for t in hour_filtered if float(t['pnl_usd']) > 0) / len(hour_filtered) * 100
        print(f"  If ONLY profitable hours: {len(hour_filtered)} trades | WR: {wr:.0f}% | Net: ${net:,.2f}")

    # 3. L1 death rate by direction
    print("\n=== DIRECTION ANALYSIS ===")
    for direction in ['long', 'short']:
        dt = [t for t in real if direction in (t.get('action', '') or '').lower()]
        if not dt:
            continue
        losses = [t for t in dt if float(t['pnl_usd']) < 0]
        l1_deaths = [t for t in losses if len((t.get('sl_progression', 'L1') or 'L1').split('->')) <= 1]
        wr = sum(1 for t in dt if float(t['pnl_usd']) > 0) / len(dt) * 100
        l1_rate = len(l1_deaths) / len(losses) * 100 if losses else 0
        net = sum(float(t['pnl_usd']) for t in dt)
        print(f"  {direction.upper()}: {len(dt)} trades | WR: {wr:.0f}% | L1 death rate: {l1_rate:.0f}% | Net: ${net:,.2f}")

    return results


def train_lgbm_from_journal(trades: List[Dict]) -> bool:
    """Train LightGBM classifier using journal trade outcomes."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not available — skipping model training")
        return False

    # Filter to real closed trades with meaningful data
    real = [t for t in trades
            if float(t.get('pnl_usd', 0) or 0) != 0
            and float(t.get('confidence', 0) or 0) > 0
            and abs(float(t.get('pnl_usd', 0))) < 5000]

    if len(real) < 30:
        print(f"Only {len(real)} trades — need at least 30 for training")
        return False

    # Extract features and labels
    X_data = []
    y_data = []
    feature_names = None

    for t in real:
        features = extract_journal_features(t)
        if not feature_names:
            feature_names = sorted(features.keys())

        label_str = classify_trade(t)
        # 3-class: 0=L1_DEATH/SMALL_LOSS, 1=SMALL_WIN, 2=RUNNER
        if label_str == 'RUNNER':
            label = 2
        elif label_str in ('SMALL_WIN',):
            label = 1
        else:  # L1_DEATH, SMALL_LOSS
            label = 0

        X_data.append([features[k] for k in feature_names])
        y_data.append(label)

    X = np.array(X_data)
    y = np.array(y_data)

    print(f"\n=== TRAINING LGBM ===")
    print(f"Samples: {len(X)} | Features: {len(feature_names)}")
    print(f"Labels: LOSS={sum(y==0)} | SMALL_WIN={sum(y==1)} | RUNNER={sum(y==2)}")

    # Time-series split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Handle class imbalance
    class_counts = np.bincount(y_train, minlength=3)
    total = len(y_train)
    class_weights = {i: total / (3 * max(1, c)) for i, c in enumerate(class_counts)}
    sample_weights = np.array([class_weights[label] for label in y_train])

    # Train
    dtrain = lgb.Dataset(X_train, label=y_train, weight=sample_weights, feature_name=feature_names)
    dtest = lgb.Dataset(X_test, label=y_test, feature_name=feature_names, reference=dtrain)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 15,  # Small to prevent overfitting on 300 trades
        'learning_rate': 0.05,
        'min_data_in_leaf': 5,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
    }

    callbacks = [lgb.log_evaluation(period=50)]
    model = lgb.train(
        params, dtrain,
        num_boost_round=200,
        valid_sets=[dtest],
        callbacks=callbacks,
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_cls = np.argmax(y_pred, axis=1)
    accuracy = np.mean(y_pred_cls == y_test)

    # Per-class accuracy
    for cls, name in [(0, 'LOSS'), (1, 'SMALL_WIN'), (2, 'RUNNER')]:
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = np.mean(y_pred_cls[mask] == cls)
            print(f"  {name}: {mask.sum()} samples | accuracy: {cls_acc:.0%}")

    print(f"  Overall accuracy: {accuracy:.0%}")

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    print(f"\n=== TOP FEATURES (what predicts success) ===")
    for name, imp in feat_imp[:10]:
        print(f"  {name}: {imp:.0f}")

    # Simulate: what if we BLOCKED trades the model predicts as class 0 (LOSS)?
    print(f"\n=== SIMULATION: Block model-predicted losses ===")
    sim_trades = real[split_idx:]  # Test set trades
    blocked = 0
    allowed_pnl = 0
    blocked_pnl = 0
    for i, t in enumerate(sim_trades):
        pred_class = y_pred_cls[i] if i < len(y_pred_cls) else 1
        pnl = float(t['pnl_usd'])
        if pred_class == 0:  # Model says LOSS
            blocked += 1
            blocked_pnl += pnl
        else:
            allowed_pnl += pnl

    total_pnl = sum(float(t['pnl_usd']) for t in sim_trades)
    print(f"  Without filter: {len(sim_trades)} trades | Net: ${total_pnl:,.2f}")
    print(f"  With filter: {len(sim_trades)-blocked} trades | Net: ${allowed_pnl:,.2f}")
    print(f"  Blocked: {blocked} trades | Their P&L: ${blocked_pnl:,.2f}")
    print(f"  Improvement: ${allowed_pnl - total_pnl:+,.2f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/lgbm_journal.txt'
    model.save_model(model_path)
    print(f"\n  Model saved to {model_path}")

    # Also save as lgbm_latest.txt for executor to auto-load
    latest_path = 'models/lgbm_latest.txt'
    model.save_model(latest_path)
    print(f"  Also saved to {latest_path} (auto-loaded by executor)")

    return True


def generate_pattern_report(trades: List[Dict]) -> str:
    """Generate a detailed pattern report for the LLM prompt system."""
    real = [t for t in trades
            if float(t.get('pnl_usd', 0) or 0) != 0
            and float(t.get('confidence', 0) or 0) > 0
            and abs(float(t.get('pnl_usd', 0))) < 5000]

    runners = [t for t in real if classify_trade(t) == 'RUNNER']
    l1_deaths = [t for t in real if classify_trade(t) == 'L1_DEATH']

    report = []
    report.append("=" * 60)
    report.append("JOURNAL PATTERN ANALYSIS REPORT")
    report.append("=" * 60)
    report.append(f"Total real trades: {len(real)}")
    report.append(f"L3+ Runners (GOOD): {len(runners)} ({len(runners)/len(real)*100:.0f}%)")
    report.append(f"L1 Deaths (BAD): {len(l1_deaths)} ({len(l1_deaths)/len(real)*100:.0f}%)")

    # Runner patterns
    report.append("\n--- RUNNER PATTERNS (what to REPEAT) ---")
    runner_hours = defaultdict(int)
    for t in runners:
        ts = t.get('timestamp', '')
        if 'T' in ts:
            try:
                hour = int(ts.split('T')[1][:2])
                runner_hours[hour] += 1
            except Exception:
                pass
    top_runner_hours = sorted(runner_hours.items(), key=lambda x: x[1], reverse=True)[:5]
    report.append(f"Best hours for runners: {[f'{h}:00 ({c})' for h, c in top_runner_hours]}")

    runner_dur = [float(t.get('duration_minutes', 0) or 0) for t in runners]
    if runner_dur:
        report.append(f"Runner avg duration: {sum(runner_dur)/len(runner_dur):.0f}min")

    runner_exits = defaultdict(int)
    for t in runners:
        r = (t.get('exit_reason', '') or '').lower()
        if 'ema' in r or 'e1' in r:
            runner_exits['EMA Reversal'] += 1
        elif 'sl' in r:
            runner_exits['Trailing SL'] += 1
        elif 'roi' in r:
            runner_exits['ROI Table'] += 1
        else:
            runner_exits['Other'] += 1
    report.append(f"Runner exit types: {dict(runner_exits)}")

    # L1 death patterns
    report.append("\n--- L1 DEATH PATTERNS (what to AVOID) ---")
    death_hours = defaultdict(int)
    for t in l1_deaths:
        ts = t.get('timestamp', '')
        if 'T' in ts:
            try:
                hour = int(ts.split('T')[1][:2])
                death_hours[hour] += 1
            except Exception:
                pass
    top_death_hours = sorted(death_hours.items(), key=lambda x: x[1], reverse=True)[:5]
    report.append(f"Worst hours for L1 deaths: {[f'{h}:00 ({c})' for h, c in top_death_hours]}")

    return "\n".join(report)


def save_optimized_config(thresholds: Dict, path: str = 'config_optimized.yaml'):
    """Save optimized thresholds to a config file."""
    import yaml

    config = {
        'adaptive': {
            'min_entry_score': 5,  # Higher gate to block weak setups
        },
        'ai': {
            'llm_trade_conf_threshold': thresholds.get('optimal_confidence', 0.65),
            'bear_veto_threshold': 7,
            'bear_reduce_threshold': 5,
        },
        'risk': {
            'risk_per_trade_pct': 0.3,  # Smaller positions = smaller L1 losses
            'daily_loss_limit_pct': 2.0,
        },
        'filters': {
            'profitable_hours': thresholds.get('profitable_hours', list(range(6, 14))),
            'dangerous_hours': thresholds.get('dangerous_hours', list(range(19, 24)) + list(range(0, 5))),
            'block_dangerous_hours': True,
            'reduce_size_off_hours': 0.5,  # 50% size outside profitable hours
        },
    }

    try:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\nOptimized config saved to {path}")
    except ImportError:
        # No yaml module, save as JSON
        json_path = path.replace('.yaml', '.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\nOptimized config saved to {json_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train ML models from trading journal')
    parser.add_argument('--journal', default='logs/trading_journal.jsonl', help='Path to journal file')
    parser.add_argument('--no-train', action='store_true', help='Skip LightGBM training')
    args = parser.parse_args()

    print("=" * 60)
    print("TRAINING ML MODELS FROM YOUR TRADING JOURNAL")
    print("=" * 60)

    # Load journal
    trades = load_journal(args.journal)
    print(f"Loaded {len(trades)} journal entries")

    # Compute optimal thresholds
    thresholds = compute_optimal_thresholds(trades)

    # Train LightGBM
    if not args.no_train:
        train_lgbm_from_journal(trades)

    # Generate report
    report = generate_pattern_report(trades)
    print(f"\n{report}")

    # Save optimized config
    save_optimized_config(thresholds)

    print("\n" + "=" * 60)
    print("DONE — Restart the trading system to use new model + thresholds")
    print("=" * 60)


if __name__ == '__main__':
    main()
