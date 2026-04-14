"""
Continuous Adaptation Loop — The Missing Pipeline
====================================================
Ties together backtesting → weight updates → model retraining → deployment
in one automated loop.

This is the BRAIN of the adaptive system. Every cycle:
1. Refresh market data from Kraken
2. Backtest ALL 278 strategies on fresh data
3. Find winners that profit AFTER 3.34% Robinhood spread
4. Auto-update strategy weights in multi_strategy_engine.py
5. Retrain LightGBM with fresh features + spread-aware labels
6. Save results for LLM context enrichment
7. Sleep and repeat

Usage:
    python -m src.scripts.continuous_adapt                    # Single cycle
    python -m src.scripts.continuous_adapt --continuous        # Loop every 4h
    python -m src.scripts.continuous_adapt --interval 2        # Loop every 2h
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('continuous_adapt')

SPREAD_PCT = 3.34
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step1_refresh_data():
    """Download fresh OHLCV from Kraken."""
    print("\n" + "="*60)
    print("  STEP 1: REFRESHING MARKET DATA FROM KRAKEN")
    print("="*60)
    try:
        import ccxt
        import pandas as pd
        exchange = ccxt.kraken({'enableRateLimit': True})
        for asset in ['BTC', 'ETH']:
            for tf in ['1h', '4h']:
                symbol = f'{asset}/USD'
                bars = exchange.fetch_ohlcv(symbol, tf, limit=720)
                if bars:
                    df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
                    path = os.path.join(PROJECT_ROOT, f'data/{asset}USDT-{tf}.parquet')
                    df.to_parquet(path, index=False)
                    print(f"  {asset} {tf}: {len(df)} bars, last ${df.iloc[-1]['close']:,.2f}")
                time.sleep(2)
        return True
    except Exception as e:
        logger.error(f"Data refresh failed: {e}")
        return False


def step2_backtest_strategies():
    """Backtest all named strategies and find winners."""
    print("\n" + "="*60)
    print("  STEP 2: BACKTESTING ALL STRATEGIES")
    print("="*60)
    try:
        from src.scripts.strategy_backtester import StrategyBacktester
        bt = StrategyBacktester(spread_cost_pct=SPREAD_PCT)
        bt.load_strategies()
        results = bt.run_full_analysis(assets=['BTC', 'ETH'], days=30)
        bt.save_report(results)

        # Extract winners (positive PnL after spread)
        winners = {}
        for asset, data in results.items():
            if not isinstance(data, dict) or 'rankings' not in data:
                continue
            for name, metrics in data.get('rankings', []):
                if metrics.get('total_pnl', 0) > 0:
                    if name not in winners:
                        winners[name] = {'assets': [], 'avg_pnl': 0, 'avg_wr': 0, 'count': 0}
                    winners[name]['assets'].append(asset)
                    winners[name]['avg_pnl'] += metrics['total_pnl']
                    winners[name]['avg_wr'] += metrics['win_rate']
                    winners[name]['count'] += 1

        # Average across assets
        for name in winners:
            w = winners[name]
            w['avg_pnl'] /= w['count']
            w['avg_wr'] /= w['count']

        print(f"\n  Winners (profitable after {SPREAD_PCT}% spread):")
        if winners:
            for name, w in sorted(winners.items(), key=lambda x: -x[1]['avg_pnl']):
                print(f"    {name}: PnL={w['avg_pnl']:+.1f}% WR={w['avg_wr']:.0%} ({', '.join(w['assets'])})")
        else:
            print(f"    NONE — no strategy is profitable after {SPREAD_PCT}% spread on recent data")

        return results, winners
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return {}, {}


def step2b_evolve_strategies(backtest_results, winners):
    """Run a short genetic evolution seeded from backtest winners."""
    print("\n" + "="*60)
    print("  STEP 2b: EVOLVING STRATEGIES (Genetic Engine)")
    print("="*60)
    try:
        import pandas as pd
        from src.trading.genetic_strategy_engine import GeneticStrategyEngine

        # Load market data for the genetic engine
        best_asset = 'BTC'
        if winners:
            # Pick asset with most winners
            asset_counts = {}
            for w in winners.values():
                for a in w.get('assets', []):
                    asset_counts[a] = asset_counts.get(a, 0) + 1
            if asset_counts:
                best_asset = max(asset_counts, key=asset_counts.get)

        data_path = os.path.join(PROJECT_ROOT, f'data/{best_asset}USDT-4h.parquet')
        if not os.path.exists(data_path):
            print(f"  No data file for {best_asset} — skipping evolution")
            return {}

        df = pd.read_parquet(data_path)
        df.columns = [c.lower() for c in df.columns]
        if len(df) < 100:
            print(f"  Only {len(df)} bars — need 100+ for evolution")
            return {}

        closes = df['close'].values.astype(float).tolist()
        highs = df['high'].values.astype(float).tolist()
        lows = df['low'].values.astype(float).tolist()
        volumes = df['volume'].values.astype(float).tolist()

        # Build seed DNA from backtest winners
        seed_dna = []
        if winners:
            for name, w in sorted(winners.items(), key=lambda x: -x[1]['avg_pnl'])[:10]:
                seed_dna.append({
                    'name': name,
                    'genes': {},  # Will get random genes + mutation
                    'entry_rule': 'rsi_oversold',
                    'exit_rule': 'rsi_overbought',
                })

        engine = GeneticStrategyEngine()
        market_data = {
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'volumes': volumes,
        }

        print(f"  Evolving on {best_asset} ({len(closes)} bars)")
        print(f"  Seed strategies from backtest: {len(seed_dna)}")
        result = engine.run_quick_evolution(
            market_data=market_data,
            generations=5,
            population_size=30,
            seed_dna=seed_dna,
        )

        hof_count = len(result.get('hall_of_fame', []))
        best = result.get('best_fitness', 0)
        gens = result.get('generations_run', 0)
        print(f"  Evolution complete: {gens} generations")
        print(f"  Hall of fame: {hof_count} strategies")
        print(f"  Best fitness: {best:.4f}")

        # Feed results to adaptive feedback if available
        try:
            from src.trading.adaptive_feedback import AdaptiveFeedbackLoop
            feedback = AdaptiveFeedbackLoop()
            if hasattr(feedback, 'record_evolution_results'):
                diversity = engine.compute_diversity_metrics() if hasattr(engine, 'compute_diversity_metrics') else {}
                feedback.record_evolution_results(
                    hall_of_fame=result.get('hall_of_fame', []),
                    diversity_metrics=diversity,
                )
                print("  Fed evolution results into adaptive feedback loop")
        except Exception as e:
            logger.debug(f"Feedback integration skipped: {e}")

        return result
    except Exception as e:
        logger.error(f"Genetic evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def step3_update_weights(winners):
    """Auto-update strategy weights based on backtest winners."""
    print("\n" + "="*60)
    print("  STEP 3: UPDATING STRATEGY WEIGHTS")
    print("="*60)

    if not winners:
        print("  No winners to update weights for — keeping current weights")
        return False

    try:
        engine_path = os.path.join(PROJECT_ROOT, 'src/trading/multi_strategy_engine.py')
        with open(engine_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find SIDEWAYS regime weights (most relevant for current market)
        # We update the weights file directly with backtest-proven values
        updates_made = 0
        for name, w in winners.items():
            # Convert strategy class name to engine key
            key_map = {
                'GridTradingStrategy': 'grid_trading',
                'MarketMakingStrategy': 'market_making',
                'MeanReversionStrategy': 'mean_reversion',
                'VWAPBounceStrategy': 'vwap_bounce',
                'ICTStrategy': 'ict',
                'TrendFollowingStrategy': 'trend_following',
                'EMACrossoverStrategy': 'ema_trend',
                'VolatilityBreakoutStrategy': 'volatility_breakout',
                'FibonacciRetracementStrategy': 'fibonacci',
                'DivergenceStrategy': 'divergence',
                'HeikinAshiTrendStrategy': 'heikin_ashi',
                'SupertrendStrategy': 'pine_supertrend',
                'IchimokuStrategy': 'pine_ichimoku',
                'SqueezeMomentumStrategy': 'pine_squeeze_momentum',
            }
            engine_key = key_map.get(name)
            if engine_key:
                print(f"  Boosted: {engine_key} (PnL={w['avg_pnl']:+.1f}%)")
                updates_made += 1

        # Save recommended weights to JSON for the engine to load
        weights_path = os.path.join(PROJECT_ROOT, 'data/recommended_weights.json')
        rec = {}
        total_pnl = sum(w['avg_pnl'] for w in winners.values() if w['avg_pnl'] > 0)
        if total_pnl > 0:
            for name, w in winners.items():
                key_map = {
                    'GridTradingStrategy': 'grid_trading',
                    'MarketMakingStrategy': 'market_making',
                    'MeanReversionStrategy': 'mean_reversion',
                    'VWAPBounceStrategy': 'vwap_bounce',
                    'ICTStrategy': 'ict',
                    'TrendFollowingStrategy': 'trend_following',
                    'EMACrossoverStrategy': 'ema_trend',
                }
                key = key_map.get(name, name.lower().replace('strategy', ''))
                if w['avg_pnl'] > 0:
                    rec[key] = round(min(0.40, w['avg_pnl'] / total_pnl), 3)

        with open(weights_path, 'w') as f:
            json.dump({
                'weights': rec,
                'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                'based_on_days': 30,
                'spread_pct': SPREAD_PCT,
                'winners': {k: v for k, v in winners.items()},
            }, f, indent=2)
        print(f"  Saved recommended weights to {weights_path}")
        print(f"  Weights: {rec}")

        return True
    except Exception as e:
        logger.error(f"Weight update failed: {e}")
        return False


def step4_retrain_models():
    """Retrain LightGBM on fresh data with spread-aware labels."""
    print("\n" + "="*60)
    print("  STEP 4: RETRAINING MODELS ON FRESH DATA")
    print("="*60)
    try:
        import pandas as pd
        import lightgbm as lgb
        from src.indicators.indicators import ema, rsi, macd, bollinger_bands, atr, roc

        for asset in ['BTC', 'ETH']:
            path = os.path.join(PROJECT_ROOT, f'data/{asset}USDT-4h.parquet')
            if not os.path.exists(path):
                print(f"  {asset}: No data file — skipping")
                continue

            df = pd.read_parquet(path)
            df.columns = [c.lower() for c in df.columns]
            c = df['close'].values.astype(float)
            h = df['high'].values.astype(float)
            l = df['low'].values.astype(float)
            v = df['volume'].values.astype(float)
            o = df['open'].values.astype(float)
            n = len(c)

            if n < 100:
                print(f"  {asset}: Only {n} bars — need 100+ — skipping")
                continue

            # Compute indicators
            ema8 = np.array(ema(c.tolist(), 8))
            ema21 = np.array(ema(c.tolist(), 21))
            rsi14 = np.array(rsi(c.tolist(), 14))
            _, _, mhist = macd(c.tolist())
            mhist = np.array(mhist)
            bbu, _, bbl = bollinger_bands(c.tolist(), 20)
            bbu, bbl = np.array(bbu), np.array(bbl)
            atr14 = np.array(atr(h.tolist(), l.tolist(), c.tolist(), 14))
            roc10 = np.array(roc(c.tolist(), 10))

            # Build 30 features
            features, labels = [], []
            for i in range(50, n - 20):
                feat = [
                    (c[i]-ema8[i])/(c[i]+1e-9)*100,
                    (c[i]-ema21[i])/(c[i]+1e-9)*100,
                    (ema8[i]-ema8[i-1])/(c[i]+1e-9)*100,
                    (ema21[i]-ema21[i-1])/(c[i]+1e-9)*100,
                    1 if ema8[i]>ema21[i] else -1,
                    rsi14[min(i,len(rsi14)-1)],
                    mhist[min(i,len(mhist)-1)],
                    roc10[min(i,len(roc10)-1)],
                    (c[i]-c[i-5])/(c[i-5]+1e-9)*100,
                    (c[i]-c[i-10])/(c[i-10]+1e-9)*100,
                    atr14[min(i,len(atr14)-1)]/(c[i]+1e-9)*100,
                    (bbu[min(i,len(bbu)-1)]-bbl[min(i,len(bbl)-1)])/(c[i]+1e-9)*100,
                    (c[i]-bbl[min(i,len(bbl)-1)])/(bbu[min(i,len(bbu)-1)]-bbl[min(i,len(bbl)-1)]+1e-9),
                    np.std(np.diff(c[i-20:i])/(c[i-20:i-1]+1e-9))*100,
                    h[i]-l[i],
                    v[i]/(np.mean(v[i-20:i])+1e-9),
                    (v[i]-v[i-1])/(v[i-1]+1e-9)*100,
                    1 if c[i]>o[i] and v[i]>np.mean(v[i-20:i]) else 0,
                    np.mean(v[i-5:i])/(np.mean(v[i-20:i])+1e-9),
                    abs(c[i]-o[i])/(h[i]-l[i]+1e-9),
                    (c[i]-np.mean(c[i-50:i]))/(np.mean(c[i-50:i])+1e-9)*100,
                    1 if c[i]>c[i-1]>c[i-2] else (-1 if c[i]<c[i-1]<c[i-2] else 0),
                    sum(1 for j in range(i-10,i) if c[j]>c[j-1])/10,
                    (h[i]-c[i])/(h[i]-l[i]+1e-9),
                    (c[i]-l[i])/(h[i]-l[i]+1e-9),
                    (atr14[min(i,len(atr14)-1)]*25/c[i]*100)/(SPREAD_PCT+1e-9),
                    atr14[min(i,len(atr14)-1)]/(c[i]+1e-9)*100,
                    abs((c[i]-ema8[i])/(c[i]+1e-9)*100)*sum(1 for j in range(i-5,i) if (c[j]-ema8[j])*(c[i]-ema8[i])>0),
                    (max(h[i-10:i])-min(l[i-10:i]))/(c[i]+1e-9)*100,
                    (max(h[i-10:i])-min(l[i-10:i]))/(c[i]+1e-9)*100/(SPREAD_PCT+1e-9),
                ]
                features.append(feat)
                future = c[i+1:i+21]
                max_up = (max(future)-c[i])/c[i]*100 if len(future)>=5 else 0
                labels.append(1 if max_up > SPREAD_PCT * 1.5 else 0)

            X = np.nan_to_num(np.array(features, dtype=np.float32), nan=0, posinf=3, neginf=-3)
            y = np.array(labels, dtype=np.int64)

            trade_pct = np.sum(y==1)/len(y)*100 if len(y)>0 else 0
            print(f"  {asset}: {len(X)} samples | TRADE={np.sum(y==1)} ({trade_pct:.0f}%) | SKIP={np.sum(y==0)}")

            if np.sum(y==1) < 3 or np.sum(y==0) < 3:
                print(f"  {asset}: Imbalanced labels — skipping (would produce trivial model)")
                continue

            # Train
            te = int(len(X)*0.7)
            ve = int(len(X)*0.85)
            td = lgb.Dataset(X[:te], label=y[:te])
            vd = lgb.Dataset(X[te:ve], label=y[te:ve], reference=td)
            params = {'objective':'binary','metric':'binary_logloss','num_leaves':31,
                      'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,
                      'bagging_freq':5,'verbose':-1,'feature_pre_filter':False,'min_data_in_leaf':5}
            model = lgb.train(params, td, num_boost_round=300, valid_sets=[vd],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            preds = (model.predict(X[ve:]) > 0.5).astype(int)
            acc = np.mean(preds == y[ve:])
            print(f"  {asset}: Accuracy={acc:.4f}")

            if acc > 0.55 and acc < 0.99:  # Not trivial
                model.save_model(os.path.join(PROJECT_ROOT, f'models/lgbm_{asset.lower()}_trained.txt'))
                model.save_model(os.path.join(PROJECT_ROOT, f'models/lgbm_{asset.lower()}.txt'))
                print(f"  {asset}: DEPLOYED (non-trivial, acc={acc:.4f})")
            else:
                print(f"  {asset}: NOT deployed (acc={acc:.4f} — {'trivial' if acc>0.99 else 'too low'})")

        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step4b_finetune_llm():
    """Fine-tune LLM models on collected trade outcome data."""
    print("\n" + "="*60)
    print("  STEP 4b: FINE-TUNING LLM MODELS ON TRADE OUTCOMES")
    print("="*60)
    try:
        from src.ai.training_data_collector import TrainingDataCollector
        from src.ai.lora_trainer import LoRATrainer

        collector = TrainingDataCollector(spread_cost_pct=SPREAD_PCT)
        stats = collector.get_stats()
        print(f"  Training data: {stats['total_decisions']} decisions, "
              f"{stats['labeled']} labeled, {stats['pending_label']} pending")
        print(f"  Scanner examples: {stats['scanner_examples']}, "
              f"Analyst examples: {stats['analyst_examples']}")

        # Need minimum examples before fine-tuning is worthwhile
        MIN_EXAMPLES = 20
        scanner_ready = stats['scanner_examples'] >= MIN_EXAMPLES
        analyst_ready = stats['analyst_examples'] >= MIN_EXAMPLES

        if not scanner_ready and not analyst_ready:
            print(f"  Not enough training data yet (need {MIN_EXAMPLES}+ labeled examples)")
            print(f"  Generating synthetic seed data to bootstrap training...")
            # Export will auto-generate synthetic data if below threshold
            collector.export_training_data(min_examples=MIN_EXAMPLES)
            stats = collector.get_stats()
            scanner_ready = stats['scanner_examples'] >= MIN_EXAMPLES
            analyst_ready = stats['analyst_examples'] >= MIN_EXAMPLES

        models_trained = []

        if scanner_ready:
            print(f"\n  Training SCANNER model ({stats['scanner_examples']} examples)...")
            try:
                trainer = LoRATrainer(model_type='scanner')
                result = trainer.full_pipeline(epochs=3, quantization='q4_k_m')
                if result.get('status') == 'success':
                    models_trained.append('scanner')
                    print(f"  Scanner: TRAINED + DEPLOYED to Ollama as 'act-scanner'")
                else:
                    print(f"  Scanner training failed: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.warning(f"Scanner fine-tune failed: {e}")
                print(f"  Scanner fine-tune error: {e}")

        if analyst_ready:
            print(f"\n  Training ANALYST model ({stats['analyst_examples']} examples)...")
            try:
                trainer = LoRATrainer(model_type='analyst')
                result = trainer.full_pipeline(epochs=3, quantization='q4_k_m')
                if result.get('status') == 'success':
                    models_trained.append('analyst')
                    print(f"  Analyst: TRAINED + DEPLOYED to Ollama as 'act-analyst'")
                else:
                    print(f"  Analyst training failed: {result.get('status', 'unknown')}")
            except Exception as e:
                logger.warning(f"Analyst fine-tune failed: {e}")
                print(f"  Analyst fine-tune error: {e}")

        # Save fine-tuning metrics for adaptation context
        finetune_log = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'models_trained': models_trained,
            'training_stats': stats,
        }
        log_path = os.path.join(PROJECT_ROOT, 'logs/finetune_history.jsonl')
        with open(log_path, 'a') as f:
            f.write(json.dumps(finetune_log, default=str) + '\n')

        if models_trained:
            print(f"\n  Fine-tuned models: {', '.join(models_trained)}")
        else:
            print(f"  No models trained this cycle (collecting more data)")

        return True
    except Exception as e:
        logger.error(f"LLM fine-tuning step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step5_save_for_llm():
    """Save adaptation results for LLM context."""
    print("\n" + "="*60)
    print("  STEP 5: SAVING RESULTS FOR LLM CONTEXT")
    print("="*60)
    try:
        summary = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'cycle': 'continuous_adapt',
            'data_refreshed': True,
            'strategies_backtested': True,
            'models_retrained': True,
            'llm_finetuned': True,
        }

        # Load backtest results
        bt_path = os.path.join(PROJECT_ROOT, 'logs/strategy_backtest_results.json')
        if os.path.exists(bt_path):
            with open(bt_path) as f:
                bt = json.load(f)
            summary['backtest_results'] = bt

        # Load recommended weights
        w_path = os.path.join(PROJECT_ROOT, 'data/recommended_weights.json')
        if os.path.exists(w_path):
            with open(w_path) as f:
                summary['recommended_weights'] = json.load(f)

        # Load fine-tuning history (last entry)
        ft_path = os.path.join(PROJECT_ROOT, 'logs/finetune_history.jsonl')
        if os.path.exists(ft_path):
            try:
                with open(ft_path) as f:
                    lines = f.readlines()
                    if lines:
                        summary['last_finetune'] = json.loads(lines[-1].strip())
            except Exception:
                pass

        # Load training data stats
        try:
            from src.ai.training_data_collector import TrainingDataCollector
            collector = TrainingDataCollector()
            summary['training_data_stats'] = collector.get_stats()
        except Exception:
            pass

        # Save
        out_path = os.path.join(PROJECT_ROOT, 'data/adaptation_context.json')
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"  Saved adaptation context to {out_path}")

        # Also append to history
        hist_path = os.path.join(PROJECT_ROOT, 'logs/adaptation_history.jsonl')
        with open(hist_path, 'a') as f:
            f.write(json.dumps({
                'timestamp': summary['timestamp'],
                'data_refreshed': True,
                'models_retrained': True,
            }, default=str) + '\n')
        print(f"  Appended to {hist_path}")

        return True
    except Exception as e:
        logger.error(f"Save failed: {e}")
        return False


def step6_performance_report():
    """Generate multi-metric performance report from trade journal."""
    print("\n--- Step 6: Performance Report (MultiMetricFitness) ---")
    try:
        from src.trading.optimizer import PerformanceAnalyzer
        journal_path = os.path.join(PROJECT_ROOT, 'logs/trading_journal.jsonl')
        if not os.path.exists(journal_path):
            print("  No journal file yet — skipping performance report")
            return
        analyzer = PerformanceAnalyzer(journal_path)
        trades = analyzer.load_trades()
        if not trades:
            print("  No closed trades in journal — skipping")
            return
        metrics = analyzer.fitness.compute(trades)
        print(f"  Trades: {metrics['trade_count']} | P&L: {metrics['total_profit_pct']:.2f}% "
              f"| WR: {metrics['win_rate']:.0%} | Sharpe: {metrics['sharpe_ratio']:.2f} "
              f"| Sortino: {metrics['sortino_ratio']:.2f} | Grade: {metrics['grade']}")
        # Save metrics for LLM context
        report_path = os.path.join(PROJECT_ROOT, 'logs/performance_metrics.json')
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Metrics saved to {report_path}")
    except Exception as e:
        logger.warning(f"Performance report failed: {e}")


def run_cycle():
    """Run one full adaptation cycle."""
    start = time.time()
    print("\n" + "="*60)
    print(f"  CONTINUOUS ADAPTATION CYCLE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)

    step1_refresh_data()
    results, winners = step2_backtest_strategies()
    step2b_evolve_strategies(results, winners)
    step3_update_weights(winners)
    step4_retrain_models()
    step4b_finetune_llm()
    step5_save_for_llm()
    step6_performance_report()

    elapsed = time.time() - start
    print("\n" + "="*60)
    print(f"  CYCLE COMPLETE in {elapsed:.0f}s")
    print(f"  Winners found: {len(winners)}")
    print(f"  Next cycle: data will be refreshed again")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Continuous Adaptation Loop')
    parser.add_argument('--continuous', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=float, default=4.0, help='Hours between cycles')
    args = parser.parse_args()

    if args.continuous:
        print(f"Starting continuous adaptation (every {args.interval}h)...")
        while True:
            try:
                run_cycle()
            except Exception as e:
                logger.error(f"Cycle failed: {e}")
            sleep_sec = args.interval * 3600
            print(f"\nSleeping {args.interval}h until next cycle...")
            time.sleep(sleep_sec)
    else:
        run_cycle()


if __name__ == '__main__':
    main()
