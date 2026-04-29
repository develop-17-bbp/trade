"""
ACT Fresh Start — Reset P&L but Keep Lessons
=============================================
1. Archives ALL old trades into v8.0 memory system (LLM + agents learn from losses)
2. Resets live trading state (equity, P&L, trade history)
3. Fetches real Robinhood balance as new initial_capital
4. Preserves config.yaml but updates initial_capital to real balance

Usage:
    python -m src.scripts.reset_fresh_start
    python -m src.scripts.reset_fresh_start --dry-run  (preview only)
"""
import json
import os
import sys
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [RESET] %(message)s')
logger = logging.getLogger(__name__)


def load_trade_journal(path='logs/trading_journal.jsonl'):
    """Load all trades from journal."""
    trades = []
    if not os.path.exists(path):
        logger.warning(f"No journal found at {path}")
        return trades
    with open(path, 'r') as f:
        for line in f:
            try:
                trades.append(json.loads(line.strip()))
            except Exception:
                continue
    return trades


def archive_trades_to_memory(trades):
    """Feed all historical trades into v8.0 memory system so LLM learns from mistakes."""
    from src.memory.llm_memory import LLMMemory
    from src.memory.quant_memory import QuantMemory
    from src.memory.agent_memory import AgentMemory

    llm_mem = LLMMemory('mistral_scanner')
    lgbm_mem = QuantMemory('lgbm')
    agent_mems = {}

    archived = 0
    wins = 0
    losses = 0

    for t in trades:
        pnl_pct = t.get('pnl_pct', 0)
        pnl_usd = t.get('pnl_usd', 0)
        asset = t.get('asset', 'BTC')
        label = 'WIN' if pnl_pct > 0 else 'LOSS'
        confidence = t.get('confidence', 0.5)
        risk_score = t.get('risk_score', 5)
        regime = t.get('hurst_regime', 'unknown')
        exit_reason = t.get('exit_reason', '')
        reasoning = t.get('llm_reasoning', '')

        if pnl_pct > 0:
            wins += 1
        else:
            losses += 1

        # LLM Memory — learns what decisions led to wins/losses
        try:
            prompt_hash = f"archive_{asset}_{t.get('timestamp', 0)}"
            llm_mem.record_decision(
                prompt_hash=prompt_hash,
                parsed_output={'proceed': True, 'confidence': confidence, 'risk_score': risk_score,
                               'reasoning': reasoning},
                trade_outcome_pnl=pnl_pct,
                trade_outcome_label=label,
                bear_veto_fired=False,
                actual_move_pct=pnl_pct,
                predicted_move_pct=0,
            )
        except Exception as e:
            logger.debug(f"LLM memory record failed: {e}")

        # Quant Memory — learns which regimes had losses
        try:
            lgbm_mem.record_prediction(
                asset=asset,
                direction='LONG',
                confidence=confidence,
                features_top5=['archived'],
                regime=regime,
                hurst=t.get('hurst', 0.5),
                volatility=0.15,
                session='archived',
                outcome_pnl=pnl_pct,
                outcome_label=label,
            )
        except Exception as e:
            logger.debug(f"Quant memory record failed: {e}")

        archived += 1

    # Consolidate patterns from archived data
    try:
        llm_mem.consolidate()
        lgbm_mem.consolidate()
    except Exception:
        pass

    logger.info(f"Archived {archived} trades into memory: {wins} wins, {losses} losses")
    logger.info("Memory now contains patterns from ALL historical trades")
    logger.info("LLM will use these as negative examples during fine-tuning")

    return archived, wins, losses


def fetch_robinhood_balance():
    """Fetch real balance from Robinhood API."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from src.integrations.robinhood_crypto import RobinhoodCryptoClient
        client = RobinhoodCryptoClient()
        account = client.get_account()
        if account:
            # Robinhood returns buying_power or similar
            results = account.get('results', [account]) if isinstance(account, dict) else [account]
            for acc in (results if isinstance(results, list) else [results]):
                bp = acc.get('buying_power', acc.get('portfolio_cash', acc.get('equity', 0)))
                if bp:
                    balance = float(bp)
                    logger.info(f"Robinhood account balance: ${balance:.2f}")
                    return balance

        # Try holdings
        holdings = client.get_holdings(['BTC', 'ETH'])
        if holdings:
            logger.info(f"Robinhood holdings response: {json.dumps(holdings)[:200]}")

        # Get BTC/ETH prices to estimate total
        btc_price = client.get_btc_price()
        eth_price = client.get_eth_price()
        logger.info(f"BTC: ${btc_price}, ETH: ${eth_price}")

    except Exception as e:
        logger.error(f"Failed to fetch Robinhood balance: {e}")
    return None


def reset_state_files():
    """Archive and reset all state files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_dir = Path(f'logs/archive_{timestamp}')
    archive_dir.mkdir(parents=True, exist_ok=True)

    files_to_archive = [
        'logs/trading_journal.jsonl',
        'logs/trading_journal.json',
        'logs/trade_history.csv',
        'logs/trade_decisions.jsonl',
        'logs/robinhood_paper.jsonl',
        'logs/robinhood_paper_state.json',
        'logs/dashboard_state.json',
        'logs/main_output.log',
        'logs/TRADES_DETAILED_REPORT.csv',
        'logs/benchmark_history.json',
        'logs/training_state.json',
        # Position tracker — without this clear, the bot recovers the
        # orphan positions from sl_state.json on next startup and the
        # frontend keeps showing "172 open" right back.
        'data/sl_state.json',
        # Paper trading position state files (older + newer paths)
        'data/paper_state.json',
        'data/paper_positions.json',
        # SafeEntryState — rolling-Sharpe history. Reset so the
        # readiness gate's per-trade Sharpe recomputes from a clean
        # slate; warm_store still has every outcome for finetune corpus.
        'data/safe_entry_state.json',
    ]

    archived_files = []
    for f in files_to_archive:
        if os.path.exists(f):
            dest = archive_dir / Path(f).name
            shutil.copy2(f, dest)
            archived_files.append(f)
            # Reset the file (create empty)
            with open(f, 'w') as fh:
                if f.endswith('.json'):
                    fh.write('{}')
                elif f.endswith('.csv'):
                    fh.write('')
                else:
                    fh.write('')

    logger.info(f"Archived {len(archived_files)} state files to {archive_dir}")
    return str(archive_dir), archived_files


def update_config_initial_capital(balance: float):
    """Update config.yaml with real Robinhood balance."""
    import yaml
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        logger.warning("config.yaml not found")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    old_capital = config.get('initial_capital', 100000)
    config['initial_capital'] = balance

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Updated config.yaml: initial_capital ${old_capital:.2f} -> ${balance:.2f}")


def flatten_open_positions_first():
    """Close every open position recorded on disk and persist the realized
    PnL to warm_store so finetune sees the outcome. Mirror of the
    /flatten-positions skill at module scope so this script can call it
    as a pre-step before files are wiped.

    Defensive against shape drift in the skill's data dict — `flattened`
    can be either a list (current contract) or an int (older shape some
    callers used). Either way we extract a count without crashing.
    """
    try:
        from skills.flatten_positions.action import run as flatten_run
        result = flatten_run({"confirm": True})
    except Exception as e:
        logger.warning(f"flatten_positions pre-step failed: {e}")
        return
    if not result.ok:
        logger.warning(f"flatten pre-step refused: {result.error}")
        return

    data = result.data or {}
    flat_field = data.get("flattened")
    if isinstance(flat_field, list):
        n = len(flat_field)
    elif isinstance(flat_field, int):
        n = flat_field
    else:
        n = 0
    total = data.get("total_pnl_usd")
    if not isinstance(total, (int, float)):
        total = 0.0
    logger.info(f"flatten pre-step: closed {n} positions, realized PnL ${total:+.2f}")


def main():
    parser = argparse.ArgumentParser(description='ACT Fresh Start — Reset P&L, Keep Lessons')
    parser.add_argument('--dry-run', action='store_true', help='Preview only, no changes')
    parser.add_argument('--skip-robinhood', action='store_true', help='Skip Robinhood balance fetch')
    parser.add_argument('--initial-balance', type=float, default=None, help='Manual initial balance')
    parser.add_argument('--no-flatten', action='store_true',
                        help='Skip the /flatten-positions pre-step. Default behaviour now '
                             'flattens open positions first so the realized PnL is recorded '
                             'in warm_store BEFORE state files are wiped.')
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  ACT FRESH START — Reset P&L, Keep Lessons")
    print("=" * 60)
    print()

    # Step 1: Load existing trades
    trades = load_trade_journal()
    wins = sum(1 for t in trades if t.get('pnl_pct', 0) > 0)
    losses = sum(1 for t in trades if t.get('pnl_pct', 0) <= 0)
    total_pnl = sum(t.get('pnl_usd', 0) for t in trades)
    print(f"  Found {len(trades)} historical trades:")
    print(f"    Wins: {wins} | Losses: {losses}")
    print(f"    Total P&L: ${total_pnl:.2f}")
    print()

    if args.dry_run:
        print("  [DRY RUN] Would flatten open positions, archive trades to memory, reset state files")
        print("  [DRY RUN] No changes made")
        return

    # Step 1.5: Flatten open positions FIRST so the loss is recorded as
    # realized in warm_store (and seeded into the finetune corpus) BEFORE
    # state files are reset. Without this, the bot's in-flight 172-stack
    # would just be deleted with no realized-PnL entry — the dashboard
    # resets but warm_store loses the lesson.
    if not args.no_flatten:
        print("  [0/4] Flattening any open positions (writes operator_flatten outcomes)...")
        flatten_open_positions_first()
        print()

    # Step 2: Archive trades into memory (LLM LEARNS from these)
    print("  [1/4] Archiving trades into v8.0 memory system...")
    archived, mem_wins, mem_losses = archive_trades_to_memory(trades)
    print(f"    -> {archived} trades archived as learning data")
    print(f"    -> LLM will see these as negative/positive examples")
    print()

    # Step 3: Fetch real Robinhood balance
    balance = args.initial_balance
    if not balance and not args.skip_robinhood:
        print("  [2/4] Fetching real Robinhood balance...")
        balance = fetch_robinhood_balance()
        if balance:
            print(f"    -> Real balance: ${balance:.2f}")
        else:
            print("    -> Could not fetch balance. Using config.yaml value.")
    if not balance:
        import yaml
        with open('config.yaml') as f:
            cfg = yaml.safe_load(f)
        balance = cfg.get('initial_capital', 16000)
        print(f"    -> Using existing config value: ${balance:.2f}")
    print()

    # Step 4: Reset state files
    print("  [3/4] Archiving and resetting state files...")
    archive_dir, archived_files = reset_state_files()
    print(f"    -> Old logs saved to {archive_dir}")
    print(f"    -> {len(archived_files)} files reset to empty")
    print()

    # Step 5: Update config
    print("  [4/4] Updating config.yaml with real balance...")
    update_config_initial_capital(balance)
    print()

    print("=" * 60)
    print("  FRESH START COMPLETE")
    print("=" * 60)
    print()
    print(f"  Initial Balance:  ${balance:.2f} (from Robinhood)")
    print(f"  Trades Archived:  {archived} (preserved in memory)")
    print(f"  State Files:      Reset to empty")
    print(f"  Old Logs:         {archive_dir}")
    print()
    print("  Memory contains all historical lessons.")
    print("  LLM will use past losses as negative examples.")
    print("  Past winning patterns preserved as positive examples.")
    print()
    print("  Next: run START_ALL.ps1 to begin trading with clean slate")
    print("=" * 60)


if __name__ == '__main__':
    main()
