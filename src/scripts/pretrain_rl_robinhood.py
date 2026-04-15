"""
RL Pre-Training for Robinhood (Spread-Aware, Multi-Timeframe)
=============================================================
Simulates the EMA(8) strategy on SPOT historical data (4h + 1d) WITH
Robinhood's 1.69% round-trip spread cost. Feeds completed trades into the
RL agent so it learns which setups are profitable AFTER spread.

Uses SPOT market data (not futures) to match Robinhood's actual pricing.
Trains on both 4h and 1d timeframes since those are the only viable TFs
for Robinhood's wide spread.

This gives the RL agent ~300-800 simulated trades worth of experience,
equivalent to months of real-time learning, in under 60 seconds.

Usage:
    python -m src.scripts.pretrain_rl_robinhood
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import io

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.ai.reinforcement_learning import EMAStrategyRL, EMATradeState


# -- CONFIG --
SPREAD_COST_PCT = 1.69          # Robinhood round-trip spread
EMA_PERIOD = 8
ASSETS = ['BTC', 'ETH']
HARD_STOP_PCT = -5.0            # Same as config
ATR_TP_MULT = 10.0              # Same as config
ATR_SL_MULT = 3.0               # Same as config
MIN_ENTRY_SCORE = 8             # Sniper mode minimum
MIN_CONFLUENCE = 4              # Sniper mode minimum

# Multi-timeframe config: (timeframe, lookback_bars, timeframe_rank)
TIMEFRAMES = [
    ('4h', 4500, 3),    # ~750 days of 4h data
    ('1d', 1500, 4),    # ~1500 days (~4 years) of daily data
]

# Data sources to try (in order) - all SPOT markets to match Robinhood
DATA_SOURCES = [
    {
        'name': 'Binance Spot',
        'exchange': 'binance',
        'options': {'defaultType': 'spot'},
    },
    {
        'name': 'Kraken Spot',
        'exchange': 'kraken',
        'options': {},
    },
    {
        'name': 'Coinbase Spot',
        'exchange': 'coinbasepro',
        'options': {},
    },
]

NUM_TRAINING_PASSES = 4  # More passes for deeper learning


def fetch_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """
    Fetch historical SPOT OHLCV data from multiple exchanges.
    Tries Binance -> Kraken -> Coinbase until one works.
    Uses SPOT (not futures) to match Robinhood pricing.
    """
    try:
        import ccxt
    except ImportError:
        print("  ERROR: ccxt not installed. Run: pip install ccxt")
        return pd.DataFrame()

    for source in DATA_SOURCES:
        try:
            print(f"    Trying {source['name']}...")
            exchange_cls = getattr(ccxt, source['exchange'])
            exchange = exchange_cls({
                'enableRateLimit': True,
                'options': source['options'],
            })

            all_data = []
            since = None
            remaining = limit
            retries = 0

            while remaining > 0:
                batch_size = min(remaining, 1000)
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=batch_size)
                except Exception as e:
                    retries += 1
                    if retries > 3:
                        raise
                    time.sleep(2)
                    continue

                if not ohlcv:
                    break
                all_data.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                remaining -= len(ohlcv)
                if len(ohlcv) < batch_size:
                    break
                time.sleep(0.5)  # Rate limit

            if not all_data or len(all_data) < 100:
                print(f"    {source['name']}: Only got {len(all_data)} bars, trying next...")
                continue

            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            print(f"    {source['name']}: Got {len(df)} bars")
            return df

        except Exception as e:
            print(f"    {source['name']} failed: {e}")
            continue

    print("  All data sources failed!")
    return pd.DataFrame()


def compute_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA array."""
    ema = np.zeros_like(closes, dtype=float)
    ema[0] = closes[0]
    mult = 2.0 / (period + 1)
    for i in range(1, len(closes)):
        ema[i] = closes[i] * mult + ema[i - 1] * (1 - mult)
    return ema


def compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR array."""
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    tr[0] = highs[0] - lows[0]

    atr = np.zeros(n)
    atr[:period] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


def compute_entry_score(ema_vals: np.ndarray, closes: np.ndarray, volumes: np.ndarray,
                        price: float, i: int, direction: str) -> int:
    """Simplified entry score computation (mirrors executor logic)."""
    score = 0

    if i < 5:
        return 0

    # 1. EMA slope strength (0-3)
    slope = abs(ema_vals[i] - ema_vals[i-3]) / ema_vals[i-3] * 100 if ema_vals[i-3] > 0 else 0
    if slope > 0.3:
        score += 3
    elif slope > 0.1:
        score += 2
    elif slope > 0.03:
        score += 1

    # 2. Consecutive EMA direction (0-3)
    consec = 0
    for j in range(i-1, max(0, i-12), -1):
        if j > 0:
            if direction == 'LONG' and ema_vals[j] > ema_vals[j-1]:
                consec += 1
            elif direction == 'SHORT' and ema_vals[j] < ema_vals[j-1]:
                consec += 1
            else:
                break
    if consec >= 5:
        score += 3
    elif consec >= 3:
        score += 2
    elif consec >= 2:
        score += 1

    # 3. Price vs EMA separation (0-2)
    sep = abs(price - ema_vals[i]) / ema_vals[i] * 100 if ema_vals[i] > 0 else 0
    if sep > 0.5:
        score += 2
    elif sep > 0.2:
        score += 1

    # 4. Candle momentum (0-2)
    if i >= 3:
        if direction == 'LONG' and closes[i] > closes[i-1] > closes[i-2]:
            score += 2
        elif direction == 'SHORT' and closes[i] < closes[i-1] < closes[i-2]:
            score += 2

    # 5. Volume confirmation (0-1)
    if i >= 5:
        vol_recent = np.mean(volumes[i-2:i+1])
        vol_prior = np.mean(volumes[i-5:i-2]) if np.mean(volumes[i-5:i-2]) > 0 else vol_recent
        if vol_recent / vol_prior > 1.2:
            score += 1

    return score


def compute_confluence(ema_vals, ema_direction, closes, volumes, atr_vals,
                       i, direction, entry_score, move_to_spread, ema_slope,
                       timeframe_rank):
    """Count independent signals agreeing with the trade."""
    confluence = 0

    # 1. EMA direction matches trade
    if ema_direction[i] == ('RISING' if direction == 'LONG' else 'FALLING'):
        confluence += 1

    # 2. Good entry score
    if entry_score >= 6:
        confluence += 1

    # 3. Volume confirms
    if i >= 20:
        vol_ratio = volumes[i] / np.mean(volumes[i-20:i]) if np.mean(volumes[i-20:i]) > 0 else 1.0
        if vol_ratio > 1.2:
            confluence += 1

    # 4. Expected move clears spread
    if move_to_spread > 2.0:
        confluence += 1

    # 5. Strong slope
    if abs(ema_slope) > 0.1:
        confluence += 1

    # 6. 20-bar trend alignment
    if i >= 20:
        long_trend = closes[i] > closes[i-20]
        if (direction == 'LONG' and long_trend) or (direction == 'SHORT' and not long_trend):
            confluence += 1

    # 7. Higher timeframe bonus (1d gets extra confluence for being inherently stronger)
    if timeframe_rank >= 4:
        confluence += 1

    # 8. Price making higher highs / lower lows (trend structure)
    if i >= 10:
        if direction == 'LONG':
            recent_high = max(closes[i-5:i+1])
            prior_high = max(closes[i-10:i-5])
            if recent_high > prior_high:
                confluence += 1
        else:
            recent_low = min(closes[i-5:i+1])
            prior_low = min(closes[i-10:i-5])
            if recent_low < prior_low:
                confluence += 1

    return confluence


def simulate_trades(df: pd.DataFrame, asset: str, timeframe_rank: int) -> list:
    """
    Simulate EMA(8) strategy on historical SPOT data.
    Returns list of trade dicts with all info needed for RL training.
    """
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    volumes = df['volume'].values
    n = len(closes)

    ema_vals = compute_ema(closes, EMA_PERIOD)
    atr_vals = compute_atr(highs, lows, closes, 14)

    # Also compute EMA(21) for cross signals
    ema_21 = compute_ema(closes, 21)

    trades = []
    in_trade = False
    entry_price = 0.0
    entry_idx = 0
    direction = ''
    peak_price = 0.0
    sl_price = 0.0

    # Track EMA direction
    ema_direction = ['FLAT'] * n
    for i in range(1, n):
        if ema_vals[i] > ema_vals[i-1]:
            ema_direction[i] = 'RISING'
        elif ema_vals[i] < ema_vals[i-1]:
            ema_direction[i] = 'FALLING'
        else:
            ema_direction[i] = ema_direction[i-1]

    # Max hold: 7 days for 4h (42 bars), 14 days for 1d (14 bars)
    max_hold = 42 if timeframe_rank <= 3 else 14

    # Detect EMA new lines and simulate trades
    for i in range(EMA_PERIOD + 5, n - 10):  # Leave room for forward bars
        atr = atr_vals[i]
        price = closes[i]

        if in_trade:
            # -- MANAGE POSITION --
            hold_bars = i - entry_idx

            if direction == 'LONG':
                peak_price = max(peak_price, highs[i])
                pnl_pct = (price - entry_price) / entry_price * 100
            else:
                peak_price = min(peak_price, lows[i])
                pnl_pct = (entry_price - price) / entry_price * 100

            exit_reason = None

            # Hard stop
            if pnl_pct <= HARD_STOP_PCT:
                exit_reason = 'hard_stop'

            # SL hit (ATR-based)
            elif direction == 'LONG' and price <= sl_price:
                exit_reason = 'sl'
            elif direction == 'SHORT' and price >= sl_price:
                exit_reason = 'sl'

            # EMA flip exit (only in profit)
            elif hold_bars >= 3 and pnl_pct > 0:
                if direction == 'LONG' and ema_direction[i] == 'FALLING' and ema_direction[i-1] == 'FALLING':
                    exit_reason = 'ema_exit'
                elif direction == 'SHORT' and ema_direction[i] == 'RISING' and ema_direction[i-1] == 'RISING':
                    exit_reason = 'ema_exit'

            # Time exit
            elif hold_bars >= max_hold:
                exit_reason = 'time'

            # Ratchet profit lock
            elif pnl_pct >= 3.0 and hold_bars >= 6:
                if direction == 'LONG':
                    trail = peak_price - (peak_price - entry_price) * 0.5
                    if price <= trail:
                        exit_reason = 'ratchet'
                else:
                    trail = peak_price + (entry_price - peak_price) * 0.5
                    if price >= trail:
                        exit_reason = 'ratchet'

            if exit_reason:
                # -- COMPUTE RL STATE AT ENTRY --
                e_i = entry_idx
                expected_move_pct = (atr_vals[e_i] * ATR_TP_MULT / closes[e_i]) * 100
                move_to_spread = expected_move_pct / SPREAD_COST_PCT if SPREAD_COST_PCT > 0 else 10

                # ATR percentile at entry
                if e_i >= 100:
                    sorted_atr = sorted(atr_vals[e_i-100:e_i])
                    atr_pctile = sum(1 for v in sorted_atr if v <= atr_vals[e_i]) / len(sorted_atr)
                else:
                    atr_pctile = 0.5

                # Volume ratio at entry
                if e_i >= 20:
                    vol_ratio = volumes[e_i] / np.mean(volumes[e_i-20:e_i]) if np.mean(volumes[e_i-20:e_i]) > 0 else 1.0
                else:
                    vol_ratio = 1.0

                # EMA slope at entry
                ema_slope = (ema_vals[e_i] - ema_vals[e_i-2]) / ema_vals[e_i-2] * 100 if ema_vals[e_i-2] > 0 else 0

                # Entry score
                entry_score = compute_entry_score(ema_vals, closes, volumes, closes[e_i], e_i, direction)

                # Confluence
                confluence = compute_confluence(
                    ema_vals, ema_direction, closes, volumes, atr_vals,
                    e_i, direction, entry_score, move_to_spread, ema_slope,
                    timeframe_rank
                )

                state = EMATradeState(
                    ema_slope=ema_slope,
                    ema_slope_bars=3,
                    price_ema_distance_atr=(closes[e_i] - ema_vals[e_i]) / atr_vals[e_i] if atr_vals[e_i] > 0 else 0,
                    ema_acceleration=0.0,
                    trend_bars_since_flip=5,
                    trend_consistency=0.7,
                    higher_tf_alignment=0.5 if confluence >= 3 else 0.0,
                    atr_percentile=atr_pctile,
                    volume_ratio=vol_ratio,
                    spread_atr_ratio=SPREAD_COST_PCT / (atr_vals[e_i] / closes[e_i] * 100) if atr_vals[e_i] > 0 else 1.0,
                    recent_win_rate=0.5,
                    daily_pnl_pct=0.0,
                    open_positions=0,
                    consecutive_losses=0,
                    hour_of_day=12,
                    day_of_week=3,
                    spread_cost_pct=SPREAD_COST_PCT,
                    expected_move_pct=expected_move_pct,
                    move_to_spread_ratio=move_to_spread,
                    is_spot=True,
                    confluence_count=confluence,
                    entry_score=entry_score,
                    timeframe_rank=timeframe_rank,
                )

                trade = {
                    'state': state,
                    'pnl_pct': pnl_pct,
                    'exit_type': exit_reason,
                    'hold_bars': hold_bars,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_score': entry_score,
                    'confluence': confluence,
                    'move_to_spread': move_to_spread,
                    'net_pnl': pnl_pct - SPREAD_COST_PCT,
                    'timeframe_rank': timeframe_rank,
                }
                trades.append(trade)
                in_trade = False

        else:
            # -- DETECT ENTRY SIGNAL --
            # EMA new line: direction flipped after 2+ bars in prior direction
            if i >= 3:
                if ema_direction[i] == 'RISING' and ema_direction[i-1] == 'RISING':
                    prior_falling = sum(1 for j in range(max(0, i-6), i-1) if ema_direction[j] == 'FALLING')
                    if prior_falling >= 2:
                        direction = 'LONG'
                        entry_price = closes[i]
                        entry_idx = i
                        peak_price = highs[i]
                        sl_price = entry_price - atr * ATR_SL_MULT
                        in_trade = True

                elif ema_direction[i] == 'FALLING' and ema_direction[i-1] == 'FALLING':
                    prior_rising = sum(1 for j in range(max(0, i-6), i-1) if ema_direction[j] == 'RISING')
                    if prior_rising >= 2:
                        direction = 'SHORT'
                        entry_price = closes[i]
                        entry_idx = i
                        peak_price = lows[i]
                        sl_price = entry_price + atr * ATR_SL_MULT
                        in_trade = True

    return trades


def pretrain_rl(asset: str, trades: list, rl: EMAStrategyRL):
    """Feed simulated trades into RL agent for learning."""
    wins = 0
    losses = 0
    total_net_pnl = 0.0
    skipped_by_rl = 0
    entered_by_rl = 0

    for trade in trades:
        state = trade['state']

        # Let RL decide on this trade
        decision = rl.decide(state)
        action_idx = decision.action_idx

        if decision.enter_trade:
            entered_by_rl += 1
            trade_result = {
                'pnl_pct': trade['pnl_pct'],
                'exit_type': trade['exit_type'],
                'hold_bars': trade['hold_bars'],
                'was_skipped': False,
                'spread_cost_pct': SPREAD_COST_PCT,
                'is_spot': True,
            }
        else:
            skipped_by_rl += 1
            trade_result = {
                'pnl_pct': trade['pnl_pct'],
                'exit_type': trade['exit_type'],
                'hold_bars': trade['hold_bars'],
                'was_skipped': True,
                'would_have_pnl': trade['pnl_pct'],
                'spread_cost_pct': SPREAD_COST_PCT,
                'is_spot': True,
            }

        rl.record_trade_result(state, action_idx, trade_result)

        net = trade['pnl_pct'] - SPREAD_COST_PCT
        total_net_pnl += net
        if net > 0:
            wins += 1
        else:
            losses += 1

    return {
        'total_trades': len(trades),
        'wins_after_spread': wins,
        'losses_after_spread': losses,
        'wr_after_spread': wins / len(trades) * 100 if trades else 0,
        'total_net_pnl': total_net_pnl,
        'entered_by_rl': entered_by_rl,
        'skipped_by_rl': skipped_by_rl,
    }


def main():
    print("=" * 60)
    print("  RL PRE-TRAINING FOR ROBINHOOD (Spread-Aware)")
    print(f"  Spread: {SPREAD_COST_PCT}% round-trip | SPOT market data")
    print(f"  Assets: {ASSETS}")
    print(f"  Timeframes: {[tf for tf, _, _ in TIMEFRAMES]}")
    print(f"  Training passes: {NUM_TRAINING_PASSES}")
    print("=" * 60)

    for asset in ASSETS:
        symbol = f"{asset}/USDT"
        model_path = f"models/rl_ema_{asset.lower()}.json"

        print(f"\n{'=' * 50}")
        print(f"  [{asset}] MULTI-TIMEFRAME PRE-TRAINING")
        print(f"{'=' * 50}")

        all_trades = []

        for timeframe, lookback, tf_rank in TIMEFRAMES:
            tf_label = timeframe.upper()
            print(f"\n  [{asset}] [{tf_label}] Fetching {lookback} bars of SPOT data...")

            df = fetch_data(symbol, timeframe, lookback)
            if df.empty or len(df) < 100:
                print(f"  [{asset}] [{tf_label}] Not enough data -- skipping this timeframe")
                continue

            date_range = f"{df['timestamp'].iloc[0].strftime('%Y-%m-%d')} to {df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}"
            print(f"  [{asset}] [{tf_label}] Got {len(df)} bars ({date_range})")

            # Simulate trades
            print(f"  [{asset}] [{tf_label}] Simulating EMA(8) strategy with {SPREAD_COST_PCT}% spread...")
            trades = simulate_trades(df, asset, tf_rank)

            if not trades:
                print(f"  [{asset}] [{tf_label}] No trades simulated -- skipping")
                continue

            # Stats before RL
            net_pnls = [t['pnl_pct'] - SPREAD_COST_PCT for t in trades]
            wins = sum(1 for p in net_pnls if p > 0)
            raw_wins = sum(1 for t in trades if t['pnl_pct'] > 0)
            avg_win = np.mean([p for p in net_pnls if p > 0]) if wins > 0 else 0
            avg_loss = np.mean([p for p in net_pnls if p <= 0]) if (len(net_pnls) - wins) > 0 else 0

            print(f"  [{asset}] [{tf_label}] {len(trades)} trades simulated:")
            print(f"    Raw WR (no spread): {raw_wins}/{len(trades)} ({raw_wins/len(trades)*100:.0f}%)")
            print(f"    After {SPREAD_COST_PCT}% spread: {wins}/{len(trades)} ({wins/len(trades)*100:.0f}%)")
            print(f"    Avg win: +{avg_win:.2f}% | Avg loss: {avg_loss:.2f}%")
            print(f"    Total net P&L: {sum(net_pnls):.1f}%")

            # Breakdown by exit type
            exit_types = {}
            for t in trades:
                et = t['exit_type']
                if et not in exit_types:
                    exit_types[et] = {'count': 0, 'total_pnl': 0.0}
                exit_types[et]['count'] += 1
                exit_types[et]['total_pnl'] += t['net_pnl']
            print(f"    Exit breakdown:")
            for et, stats in sorted(exit_types.items(), key=lambda x: x[1]['count'], reverse=True):
                avg = stats['total_pnl'] / stats['count']
                print(f"      {et:12s}: {stats['count']:4d} trades, avg net: {avg:+.2f}%")

            all_trades.extend(trades)

        if not all_trades:
            print(f"\n  [{asset}] No trades from any timeframe -- skipping asset")
            continue

        print(f"\n  [{asset}] TOTAL: {len(all_trades)} trades across all timeframes")

        # TF breakdown
        for timeframe, _, tf_rank in TIMEFRAMES:
            tf_trades = [t for t in all_trades if t['timeframe_rank'] == tf_rank]
            if tf_trades:
                tf_wins = sum(1 for t in tf_trades if t['net_pnl'] > 0)
                print(f"    {timeframe}: {len(tf_trades)} trades, {tf_wins}/{len(tf_trades)} win ({tf_wins/len(tf_trades)*100:.0f}%)")

        # Pre-train RL
        print(f"\n  [{asset}] Pre-training RL agent ({NUM_TRAINING_PASSES} passes, {len(all_trades)} trades each)...")
        rl = EMAStrategyRL({'rl_model_path': model_path})

        for pass_num in range(NUM_TRAINING_PASSES):
            # Shuffle order each pass (keeps state->result pairs intact)
            indices = np.random.permutation(len(all_trades))
            shuffled = [all_trades[i] for i in indices]

            result = pretrain_rl(asset, shuffled, rl)
            print(f"    Pass {pass_num + 1}/{NUM_TRAINING_PASSES}: "
                  f"entered={result['entered_by_rl']} skipped={result['skipped_by_rl']} | "
                  f"Q-table: {len(rl.q_table)} states | eps={rl.epsilon:.3f}")

        # Save model
        rl._save_model()
        print(f"\n  [{asset}] Model saved to {model_path}")

        # Print learned insights
        insights = rl.get_strategy_insights()
        print(f"\n  [{asset}] RL LEARNED:")
        print(f"    Q-table: {insights['q_table_size']} states explored")
        print(f"    Skip rate: {insights['skip_rate']:.1%}")
        print(f"    Learned skip states: {insights['learned_skip_states']}")
        print(f"    Epsilon: {insights['epsilon']:.3f}")
        print(f"    Action performance:")
        for label, perf in sorted(insights['action_performance'].items(), key=lambda x: x[1]['avg_pnl'], reverse=True):
            if perf['count'] > 0:
                print(f"      {label:22s}: {perf['count']:4d} trades, avg P&L: {perf['avg_pnl']:+.3f}%")

    print(f"\n{'=' * 60}")
    print("  PRE-TRAINING COMPLETE!")
    print("  RL agents now have historical knowledge of:")
    print("    - Robinhood's 1.69% round-trip spread impact")
    print("    - Which setups are profitable AFTER spread on SPOT")
    print("    - Multi-timeframe patterns (4h + 1d)")
    print("  They will continue learning online from live trades.")
    print("=" * 60)


if __name__ == '__main__':
    main()
