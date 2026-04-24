"""
ACT Autonomous Improvement Loop — Self-Learning Trading Entity
================================================================
The brain of ACT's autonomy. Runs continuously and orchestrates:

  1. MONITOR  — Track live trade outcomes, win rate, drawdown, regime
  2. LEARN    — Feed outcomes into adaptive feedback + genetic evolution
  3. RETRAIN  — Auto-retrain ML models when performance degrades
  4. EVOLVE   — Evolve strategy populations via genetic engine
  5. ADAPT    — Adjust risk parameters, strategy weights, agent weights
  6. HEAL     — Restart crashed processes, rotate logs, check connectivity

The loop never stops. It gets smarter after every trade.
No human intervention required unless a fix fails twice.

Usage:
    python -m src.scripts.autonomous_loop                  # Start the loop
    python -m src.scripts.autonomous_loop --interval 1     # Check every 1 hour
    python -m src.scripts.autonomous_loop --dry-run        # Monitor only
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logging.handlers import RotatingFileHandler
_AUTONOMY_LOG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'logs', 'autonomous_loop.log',
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AUTONOMY] %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Rotate at 50 MB, keep 3 backups — prevents disk-full failure
        # when long-running loops produce multi-GB logs.
        RotatingFileHandler(_AUTONOMY_LOG, maxBytes=50 * 1024 * 1024, backupCount=3),
    ],
)
logger = logging.getLogger('autonomous_loop')

PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ═══════════════════════════════════════════════════════════════
# Thresholds — the system uses these to decide when to act
# ═══════════════════════════════════════════════════════════════

THRESHOLDS = {
    'win_rate_retrain': 0.40,       # Retrain if rolling WR drops below
    'win_rate_evolve': 0.45,        # Run genetic evolution if WR below
    'max_drawdown_retrain': 0.08,   # Retrain if drawdown exceeds 8%
    'max_drawdown_halt': 0.15,      # Halt trading if drawdown exceeds 15%
    'min_trades_for_action': 10,    # Need at least 10 trades before acting
    'evolve_every_hours': 6,        # Run evolution at minimum every 6h
    'retrain_every_hours': 12,      # Run full retrain at minimum every 12h
    'health_check_every_min': 30,   # Health check every 30 minutes
    'stagnation_trades': 20,        # If no improvement in 20 trades, force evolve
}


class AutonomousLoop:
    """
    The autonomous brain of ACT. Monitors, learns, and improves.

    State is persisted to data/autonomous_state.json so it survives restarts.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self._state_file = PROJECT_ROOT / 'data' / 'autonomous_state.json'
        self._state = self._load_state()
        self._cycle_count = self._state.get('cycle_count', 0)

    # ══════════════════════════════════════════════════════════════
    # 1. MONITOR — Read current system performance
    # ══════════════════════════════════════════════════════════════

    def monitor(self) -> Dict[str, Any]:
        """Read live performance metrics from the trading journal."""
        logger.info("--- MONITOR: Reading system performance ---")
        metrics = {
            'total_trades': 0, 'rolling_win_rate': 0.5, 'rolling_pnl': 0,
            'max_drawdown': 0, 'recent_trades': [],
            'regime': 'UNKNOWN', 'stagnation': False,
        }

        journal_path = PROJECT_ROOT / 'logs' / 'trading_journal.jsonl'
        if not journal_path.exists():
            logger.info("  No journal yet — waiting for first trade")
            return metrics

        trades = []
        try:
            with open(journal_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            trades.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"  Journal read error: {e}")
            return metrics

        if not trades:
            return metrics

        metrics['total_trades'] = len(trades)

        # Rolling window (last 50 closed trades)
        closed = [t for t in trades if t.get('pnl_pct') is not None]
        recent = closed[-50:]
        metrics['recent_trades'] = recent[-10:]  # Last 10 for context

        if recent:
            wins = sum(1 for t in recent if t.get('pnl_pct', 0) > 0)
            metrics['rolling_win_rate'] = round(wins / len(recent), 4)
            metrics['rolling_pnl'] = round(sum(t.get('pnl_pct', 0) for t in recent), 4)

            # Max drawdown
            import numpy as np
            pnls = [t.get('pnl_pct', 0) for t in closed]
            if pnls:
                cum = np.cumsum(pnls)
                peak = np.maximum.accumulate(cum)
                dd = peak - cum
                metrics['max_drawdown'] = round(float(np.max(dd)) if len(dd) > 0 else 0, 4)

        # Detect stagnation (no improvement in last N trades)
        prev_best_wr = self._state.get('best_rolling_wr', 0)
        if metrics['rolling_win_rate'] <= prev_best_wr and metrics['total_trades'] - self._state.get('best_wr_at_trade', 0) > THRESHOLDS['stagnation_trades']:
            metrics['stagnation'] = True

        if metrics['rolling_win_rate'] > prev_best_wr:
            self._state['best_rolling_wr'] = metrics['rolling_win_rate']
            self._state['best_wr_at_trade'] = metrics['total_trades']

        # Current regime
        try:
            adaptive_path = PROJECT_ROOT / 'data' / 'adaptive_state.json'
            if adaptive_path.exists():
                with open(adaptive_path) as f:
                    adaptive = json.load(f)
                regimes = adaptive.get('regime_profitability', {})
                if regimes:
                    metrics['regime'] = max(regimes, key=lambda r: regimes[r].get('trades', 0))
        except Exception:
            pass

        logger.info(
            f"  Trades: {metrics['total_trades']} | "
            f"WR: {metrics['rolling_win_rate']:.0%} | "
            f"PnL: {metrics['rolling_pnl']:+.2f}% | "
            f"DD: {metrics['max_drawdown']:.1%} | "
            f"Stagnant: {metrics['stagnation']}"
        )
        return metrics

    # ══════════════════════════════════════════════════════════════
    # 2. LEARN — Feed outcomes into adaptive systems
    # ══════════════════════════════════════════════════════════════

    def learn(self, metrics: Dict) -> Dict:
        """Feed trade outcomes into adaptive feedback and self-evolving overlay."""
        logger.info("--- LEARN: Updating adaptive systems ---")
        result = {'feedback_updated': False, 'overlay_updated': False}

        try:
            from src.trading.adaptive_feedback import AdaptiveFeedbackLoop
            feedback = AdaptiveFeedbackLoop()
            context = feedback.get_adaptive_context('BTC', metrics.get('regime', 'UNKNOWN'))
            result['feedback_updated'] = True
            result['confidence_mult'] = context.get('confidence_multiplier', 1.0)
            result['size_mult'] = context.get('size_multiplier', 1.0)
            result['blacklisted'] = context.get('strategy_blacklist', [])
            logger.info(
                f"  Feedback: conf={result['confidence_mult']:.2f} "
                f"size={result['size_mult']:.2f} "
                f"blacklisted={len(result['blacklisted'])}"
            )
        except Exception as e:
            logger.warning(f"  Feedback update failed: {e}")

        try:
            from src.trading.self_evolving_overlay import SelfEvolvingOverlay
            overlay = SelfEvolvingOverlay()
            overrides = overlay.get_overrides()
            result['overlay_updated'] = True
            result['risk_overrides'] = overrides.get('risk', {})
            result['indicator_source'] = overrides.get('meta', {}).get('indicator_source', 'defaults')
            logger.info(
                f"  Overlay: risk_size={overrides.get('risk', {}).get('position_size_mult', 1):.2f} "
                f"indicators={result['indicator_source']}"
            )
        except Exception as e:
            logger.warning(f"  Overlay update failed: {e}")

        # C18 — learning-mesh step: DSR reward + credit reassignment +
        # safety quarantine + co-evolution publication. This is the
        # glue that makes credit_assigner + reward + safety +
        # coevolution actually influence the next cycle, rather than
        # sitting as orphaned modules.
        mesh = self._mesh_step(metrics)
        result['mesh'] = mesh

        return result

    # ══════════════════════════════════════════════════════════════
    # Learning-mesh step (C18)
    # ══════════════════════════════════════════════════════════════

    def _mesh_step(self, metrics: Dict) -> Dict[str, Any]:
        """Process recent closed trades through the learning mesh.

        Order of operations:
          1. DSR update for each component using recent pnl returns.
          2. credit_assigner.record() + assign() → normalized weights.
          3. safety.QuarantineManager observes each component's DSR
             to detect pathological drift.
          4. coevolution publishes top-K fitness hints back to signal
             bus so RL + GA cross-pollinate.

        Never raises — any failure collapses to a "mesh disabled" marker.
        """
        mesh: Dict[str, Any] = {'enabled': False}
        recent = metrics.get('recent_trades') or []
        if not recent:
            mesh['reason'] = 'no recent trades'
            return mesh

        try:
            from src.learning.reward import get_tracker
            tracker = get_tracker()
        except Exception as e:
            mesh['reason'] = f'reward import failed: {e}'
            return mesh

        # Maintain a monotonic watermark so a trade isn't fed twice.
        processed_ids = set(self._state.get('mesh_processed_ids', []))
        new_ids: List[str] = []
        dsr_updates: Dict[str, float] = {}

        try:
            from src.learning.credit_assigner import CreditAssigner, TradeRow
        except Exception:
            CreditAssigner = None
            TradeRow = None
        credit_recorded = 0
        assigner = None
        if CreditAssigner is not None and TradeRow is not None:
            assigner = getattr(self, '_credit_assigner', None)
            if assigner is None:
                assigner = CreditAssigner()
                self._credit_assigner = assigner

        for t in recent:
            tid = str(t.get('trade_id') or t.get('id') or t.get('ts') or '')
            if tid and tid in processed_ids:
                continue
            pnl_pct = float(t.get('pnl_pct') or 0.0) / 100.0
            asset = (t.get('asset') or t.get('symbol') or 'BTC').upper()
            components = t.get('component_actions') or {}

            # Portfolio-level DSR (per-asset).
            portfolio_dsr = tracker.update('portfolio', pnl_pct, asset=asset)
            dsr_updates[f"portfolio:{asset}"] = portfolio_dsr

            for comp, action in components.items():
                # Scale per-component "contribution" by its signed action
                # so a component that voted FLAT doesn't get credit/blame
                # for this trade's outcome.
                try:
                    contribution = pnl_pct * float(action)
                except Exception:
                    contribution = 0.0
                dsr_val = tracker.update(comp, contribution, asset=asset)
                dsr_updates[f"{comp}:{asset}"] = dsr_val

            if assigner is not None and components:
                try:
                    assigner.record(TradeRow(
                        component_actions={k: float(v) for k, v in components.items()
                                           if isinstance(v, (int, float))},
                        realized_pnl=pnl_pct,
                    ))
                    credit_recorded += 1
                except Exception:
                    pass
            if tid:
                new_ids.append(tid)

        # Bound the watermark so state.json stays small.
        processed_ids |= set(new_ids)
        if len(processed_ids) > 500:
            processed_ids = set(list(processed_ids)[-500:])
        self._state['mesh_processed_ids'] = sorted(processed_ids)

        weights: Dict[str, float] = {}
        if assigner is not None:
            try:
                weights = assigner.assign()
            except Exception:
                weights = {}

        # Safety — feed each component's DSR into the quarantine manager.
        quarantined: List[str] = []
        try:
            from src.learning.safety import QuarantineManager
            qm = getattr(self, '_quarantine_mgr', None)
            if qm is None:
                qm = QuarantineManager()
                self._quarantine_mgr = qm
            for stream, dsr_val in dsr_updates.items():
                comp, _, _asset = stream.partition(':')
                if not comp or comp == 'portfolio':
                    continue
                qm.should_accept(comp, 'dsr', dsr_val)
            quarantined = [k for k, q in qm.quarantined_learners().items() if q]
        except Exception:
            pass

        # Coevolution publication — top-K DNA if the strategy repo is up.
        try:
            from src.trading.strategy_repository import get_repo
            from src.learning.coevolution import apply_rl_warm_starts_publish
            repo = get_repo()
            top_k = repo.search(status='champion', limit=5) or []
            serializable = []
            for rec in top_k:
                serializable.append({
                    'name': getattr(rec, 'strategy_id', 'unknown'),
                    'fitness': float(getattr(rec, 'live_sharpe', 0.5) or 0.5),
                    'genes': getattr(rec, 'dna', {}) or {},
                })
            if serializable:
                apply_rl_warm_starts_publish(
                    serializable,
                    regime=str(metrics.get('regime', 'UNKNOWN')),
                    model_version=os.getenv('ACT_MODEL_VERSION', 'dev'),
                )
        except Exception:
            pass

        mesh.update({
            'enabled': True,
            'dsr_streams': len(dsr_updates),
            'credit_recorded': credit_recorded,
            'weights': {k: round(v, 3) for k, v in weights.items()},
            'quarantined': quarantined,
            'portfolio_dsr': dsr_updates.get('portfolio:BTC', 0.0),
        })
        logger.info(
            f"  Mesh: DSR_streams={mesh['dsr_streams']} "
            f"credit_rows={mesh['credit_recorded']} "
            f"quarantined={quarantined or 'none'}"
        )
        return mesh

    # ══════════════════════════════════════════════════════════════
    # 3. RETRAIN — Auto-retrain ML models when performance degrades
    # ══════════════════════════════════════════════════════════════

    def retrain(self, metrics: Dict) -> Dict:
        """Auto-retrain models if performance drops below thresholds."""
        logger.info("--- RETRAIN: Checking if models need retraining ---")
        result = {'needed': False, 'executed': False, 'reason': None}

        wr = metrics.get('rolling_win_rate', 0.5)
        dd = metrics.get('max_drawdown', 0)
        total = metrics.get('total_trades', 0)
        last_retrain = self._state.get('last_retrain_time', 0)
        hours_since = (time.time() - last_retrain) / 3600

        reasons = []
        if total >= THRESHOLDS['min_trades_for_action']:
            if wr < THRESHOLDS['win_rate_retrain']:
                reasons.append(f"WR {wr:.0%} < {THRESHOLDS['win_rate_retrain']:.0%}")
            if dd > THRESHOLDS['max_drawdown_retrain']:
                reasons.append(f"DD {dd:.1%} > {THRESHOLDS['max_drawdown_retrain']:.1%}")
        if hours_since > THRESHOLDS['retrain_every_hours']:
            reasons.append(f"Scheduled ({hours_since:.0f}h since last)")

        if not reasons:
            logger.info("  No retrain needed")
            return result

        result['needed'] = True
        result['reason'] = '; '.join(reasons)
        logger.info(f"  RETRAIN TRIGGERED: {result['reason']}")

        if self.dry_run:
            return result

        try:
            from src.scripts.continuous_adapt import run_cycle
            run_cycle()
            result['executed'] = True
            self._state['last_retrain_time'] = time.time()
            logger.info("  Retrain cycle COMPLETE")
        except Exception as e:
            logger.error(f"  Retrain failed: {e}")
            result['error'] = str(e)

        return result

    # ══════════════════════════════════════════════════════════════
    # 4. EVOLVE — Run genetic strategy evolution
    # ══════════════════════════════════════════════════════════════

    def evolve(self, metrics: Dict) -> Dict:
        """Evolve strategy populations when performance stagnates."""
        logger.info("--- EVOLVE: Checking if evolution needed ---")
        result = {'needed': False, 'executed': False, 'hall_of_fame_count': 0}

        wr = metrics.get('rolling_win_rate', 0.5)
        stagnant = metrics.get('stagnation', False)
        last_evolve = self._state.get('last_evolve_time', 0)
        hours_since = (time.time() - last_evolve) / 3600

        if wr < THRESHOLDS['win_rate_evolve'] or stagnant or hours_since > THRESHOLDS['evolve_every_hours']:
            result['needed'] = True
            reason = []
            if wr < THRESHOLDS['win_rate_evolve']:
                reason.append(f"WR {wr:.0%} < {THRESHOLDS['win_rate_evolve']:.0%}")
            if stagnant:
                reason.append("stagnation detected")
            if hours_since > THRESHOLDS['evolve_every_hours']:
                reason.append(f"scheduled ({hours_since:.0f}h)")
            result['reason'] = '; '.join(reason)
            logger.info(f"  EVOLUTION TRIGGERED: {result['reason']}")
        else:
            logger.info("  No evolution needed")
            return result

        if self.dry_run:
            return result

        try:
            import pandas as pd
            from src.trading.genetic_strategy_engine import GeneticStrategyEngine

            # Load real market data
            for asset in ['BTC', 'ETH']:
                data_path = PROJECT_ROOT / f'data/{asset}USDT-4h.parquet'
                if not data_path.exists():
                    continue

                df = pd.read_parquet(data_path)
                df.columns = [c.lower() for c in df.columns]
                if len(df) < 100:
                    continue

                engine = GeneticStrategyEngine()
                evo_result = engine.run_quick_evolution(
                    market_data={
                        'closes': df['close'].values.astype(float).tolist(),
                        'highs': df['high'].values.astype(float).tolist(),
                        'lows': df['low'].values.astype(float).tolist(),
                        'volumes': df['volume'].values.astype(float).tolist(),
                    },
                    generations=5,
                    population_size=30,
                )

                hof = evo_result.get('hall_of_fame', [])
                result['hall_of_fame_count'] += len(hof)
                logger.info(f"  {asset}: {len(hof)} evolved strategies, best fitness={evo_result.get('best_fitness', 0):.2f}")

                # Feed into adaptive feedback
                try:
                    from src.trading.adaptive_feedback import AdaptiveFeedbackLoop
                    fb = AdaptiveFeedbackLoop()
                    if hasattr(fb, 'record_evolution_results'):
                        diversity = engine.compute_diversity_metrics()
                        fb.record_evolution_results(hof, diversity)
                except Exception:
                    pass

                break  # One asset per cycle to keep it fast

            result['executed'] = True
            self._state['last_evolve_time'] = time.time()
            logger.info(f"  Evolution COMPLETE — {result['hall_of_fame_count']} strategies in hall of fame")
        except Exception as e:
            logger.error(f"  Evolution failed: {e}")
            result['error'] = str(e)

        return result

    # ══════════════════════════════════════════════════════════════
    # 5. ADAPT — Adjust live parameters based on learning
    # ══════════════════════════════════════════════════════════════

    def adapt(self, metrics: Dict, learn_result: Dict) -> Dict:
        """Push adaptive adjustments to the running system."""
        logger.info("--- ADAPT: Pushing adjustments to live system ---")
        result = {'adjustments': []}

        # Emergency halt if drawdown too high
        dd = metrics.get('max_drawdown', 0)
        if dd > THRESHOLDS['max_drawdown_halt']:
            logger.critical(f"  EMERGENCY: Drawdown {dd:.1%} exceeds halt threshold!")
            result['adjustments'].append('EMERGENCY_HALT_RECOMMENDED')
            result['emergency'] = True
            # Write halt signal
            if not self.dry_run:
                halt_path = PROJECT_ROOT / 'data' / 'halt_signal.json'
                with open(halt_path, 'w') as f:
                    json.dump({
                        'halt': True, 'reason': f'Drawdown {dd:.1%}',
                        'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    }, f, indent=2)
            return result

        # Write adaptation context for the executor to consume
        context = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'cycle': self._cycle_count,
            'metrics': {
                'rolling_win_rate': metrics.get('rolling_win_rate', 0.5),
                'rolling_pnl': metrics.get('rolling_pnl', 0),
                'max_drawdown': dd,
                'total_trades': metrics.get('total_trades', 0),
                'regime': metrics.get('regime', 'UNKNOWN'),
            },
            'adjustments': {
                'confidence_multiplier': learn_result.get('confidence_mult', 1.0),
                'size_multiplier': learn_result.get('size_mult', 1.0),
                'strategy_blacklist': learn_result.get('blacklisted', []),
                'risk_overrides': learn_result.get('risk_overrides', {}),
                'indicator_source': learn_result.get('indicator_source', 'defaults'),
            },
            'evolution': {
                'last_evolve': self._state.get('last_evolve_time', 0),
                'last_retrain': self._state.get('last_retrain_time', 0),
            },
        }

        if not self.dry_run:
            ctx_path = PROJECT_ROOT / 'data' / 'adaptation_context.json'
            with open(ctx_path, 'w') as f:
                json.dump(context, f, indent=2, default=str)
            result['adjustments'].append('adaptation_context_written')

        logger.info(f"  Pushed {len(result['adjustments'])} adjustments")
        return result

    # ══════════════════════════════════════════════════════════════
    # 6. HEAL — System health maintenance
    # ══════════════════════════════════════════════════════════════

    def heal(self) -> Dict:
        """Run health checks and fix issues."""
        logger.info("--- HEAL: System health maintenance ---")
        result = {'status': 'GREEN'}

        last_health = self._state.get('last_health_check', 0)
        minutes_since = (time.time() - last_health) / 60

        if minutes_since < THRESHOLDS['health_check_every_min']:
            logger.info(f"  Skipping (last check {minutes_since:.0f}m ago)")
            return result

        try:
            from src.scripts.daily_ops import DailyOps
            ops = DailyOps(dry_run=self.dry_run)

            # Only run lightweight checks (not full retrain)
            ops.step1_check_processes()
            ops.step4_check_tunnels()
            ops.step5_disk_and_logs()

            self._state['last_health_check'] = time.time()
            result['status'] = ops.report.get('overall_status', 'GREEN')

            if ops.report.get('human_review_needed'):
                result['human_review'] = ops.report['human_review_needed']
                logger.warning(f"  Issues needing review: {len(result['human_review'])}")
            else:
                logger.info("  All systems healthy")
        except Exception as e:
            logger.error(f"  Health check failed: {e}")
            result['status'] = 'YELLOW'
            result['error'] = str(e)

        return result

    # ══════════════════════════════════════════════════════════════
    # Main Cycle
    # ══════════════════════════════════════════════════════════════

    def run_cycle(self) -> Dict:
        """Execute one full autonomous improvement cycle."""
        self._cycle_count += 1
        start = time.time()

        logger.info("=" * 60)
        logger.info(f"  AUTONOMOUS CYCLE #{self._cycle_count}")
        logger.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        cycle_report = {'cycle': self._cycle_count, 'timestamp': datetime.now(tz=timezone.utc).isoformat()}

        # 1. Monitor
        metrics = self.monitor()
        cycle_report['metrics'] = metrics

        # 2. Learn
        learn_result = self.learn(metrics)
        cycle_report['learn'] = learn_result

        # 3. Retrain (if needed)
        retrain_result = self.retrain(metrics)
        cycle_report['retrain'] = retrain_result

        # 4. Evolve (if needed)
        evolve_result = self.evolve(metrics)
        cycle_report['evolve'] = evolve_result

        # 5. Adapt
        adapt_result = self.adapt(metrics, learn_result)
        cycle_report['adapt'] = adapt_result

        # 6. Heal
        heal_result = self.heal()
        cycle_report['heal'] = heal_result

        # Save state
        elapsed = time.time() - start
        self._state['cycle_count'] = self._cycle_count
        self._state['last_cycle_time'] = time.time()
        self._state['last_cycle_duration'] = elapsed
        self._save_state()

        # Append to cycle history
        history_path = PROJECT_ROOT / 'logs' / 'autonomous_cycles.jsonl'
        try:
            with open(history_path, 'a') as f:
                summary = {
                    'cycle': self._cycle_count,
                    'timestamp': cycle_report['timestamp'],
                    'duration_s': round(elapsed, 1),
                    'win_rate': metrics.get('rolling_win_rate', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'retrained': retrain_result.get('executed', False),
                    'evolved': evolve_result.get('executed', False),
                    'heal_status': heal_result.get('status', 'UNKNOWN'),
                }
                f.write(json.dumps(summary, default=str) + '\n')
        except Exception:
            pass

        logger.info("")
        logger.info(f"  Cycle #{self._cycle_count} complete in {elapsed:.1f}s")
        logger.info(f"  WR: {metrics.get('rolling_win_rate', 0):.0%} | "
                    f"PnL: {metrics.get('rolling_pnl', 0):+.2f}% | "
                    f"DD: {metrics.get('max_drawdown', 0):.1%}")
        logger.info("=" * 60)

        return cycle_report

    # ══════════════════════════════════════════════════════════════
    # State Persistence
    # ══════════════════════════════════════════════════════════════

    def _load_state(self) -> Dict:
        try:
            if self._state_file.exists():
                with open(self._state_file) as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            'cycle_count': 0,
            'last_retrain_time': 0,
            'last_evolve_time': 0,
            'last_health_check': 0,
            'best_rolling_wr': 0,
            'best_wr_at_trade': 0,
        }

    def _save_state(self):
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"State save failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='ACT Autonomous Improvement Loop')
    parser.add_argument('--interval', type=float, default=2.0, help='Hours between cycles')
    parser.add_argument('--dry-run', action='store_true', help='Monitor only, no actions')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    args = parser.parse_args()

    loop = AutonomousLoop(dry_run=args.dry_run)

    if args.once:
        report = loop.run_cycle()
        print(json.dumps(report, indent=2, default=str))
        return

    logger.info(f"ACT Autonomous Loop STARTING (every {args.interval}h)")
    logger.info("The system will now self-learn, self-train, self-adapt, and improve.")
    logger.info("No human intervention needed unless flagged.")

    while True:
        try:
            loop.run_cycle()
        except KeyboardInterrupt:
            logger.info("Shutting down autonomous loop (user interrupt)")
            break
        except Exception as e:
            logger.error(f"Cycle failed: {e}")
            traceback.print_exc()

        sleep_sec = args.interval * 3600
        logger.info(f"Next cycle in {args.interval}h...")
        time.sleep(sleep_sec)


if __name__ == '__main__':
    main()
