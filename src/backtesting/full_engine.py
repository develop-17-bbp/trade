"""
Full Backtest Engine — Replicates Executor Logic
==================================================
Walks through historical bars, applying the exact same signal detection,
entry scoring, risk filters, position management, and trailing SL
as the live trading system.

LLM calls are replaced by entry score threshold (min_entry_score).
Exchange orders are simulated with candle close fills.
"""

import time
from typing import Optional, Dict, List
from datetime import datetime, timezone

from src.backtesting.data_loader import BacktestData, get_context_at_bar
from src.backtesting.signal_generator import compute_tf_signal, compute_indicator_context, compute_entry_score
from src.backtesting.risk_filters import BacktestFilterChain
from src.backtesting.position_manager import (
    BacktestPositionManager, Position, TradeRecord,
    TF_SECONDS,
)
from src.backtesting.metrics import BacktestMetrics


class FullBacktestEngine:
    """Full-fidelity backtester matching executor.py logic."""

    def __init__(self, config: dict = None):
        config = config or {}

        # Strategy params
        self.ema_period = config.get('ema_period', 8)
        self.min_entry_score = config.get('min_entry_score', 4)
        self.max_entry_score = config.get('max_entry_score', 99)  # Cap high scores (momentum traps)
        self.short_score_penalty = config.get('short_score_penalty', 0)  # Extra score needed for SHORTs
        self.initial_capital = config.get('initial_capital', 100000.0)
        self.risk_per_trade_pct = config.get('risk_per_trade_pct', 2.0)
        self.max_trade_pct = 5.0  # Max 5% of equity per trade

        # Components
        self.filters = BacktestFilterChain(config)
        self.position_mgr = BacktestPositionManager(config)

        # ML models (optional)
        self.use_ml = config.get('use_ml', False)
        self._ml_loaded = False

        # State
        self.positions: Dict[str, Position] = {}  # asset -> Position
        self.equity = self.initial_capital
        self.cash = self.initial_capital

        # Results
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = []
        self.signals_generated = 0
        self.entries_attempted = 0

    def run(self, data: BacktestData, verbose: bool = False) -> BacktestMetrics:
        """Run backtest on historical data.

        Args:
            data: BacktestData with multi-timeframe OHLCV
            verbose: print bar-by-bar output

        Returns:
            BacktestMetrics with full analytics
        """
        asset = data.asset
        primary = data.primary
        if not primary or not primary.get('closes'):
            print("  [BACKTEST] ERROR: No primary data")
            return BacktestMetrics([], [], self.initial_capital)

        closes = primary['closes']
        highs = primary['highs']
        lows = primary['lows']
        opens = primary['opens']
        volumes = primary['volumes']
        timestamps = primary['timestamps']
        n_bars = len(closes)

        tf_seconds = TF_SECONDS.get(data.primary_tf, 300)

        print(f"\n  [BACKTEST] Starting: {asset} | {n_bars} bars of {data.primary_tf}")
        print(f"  [BACKTEST] Period: {datetime.fromtimestamp(timestamps[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(timestamps[-1]/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        print(f"  [BACKTEST] Capital: ${self.initial_capital:,.0f} | Min score: {self.min_entry_score} | ML: {'ON' if self.use_ml else 'OFF'}")

        start_time = time.time()
        lookback = 100  # Bars of history needed for indicators

        for i in range(lookback, n_bars):
            bar_ts = timestamps[i]

            # Build OHLCV slice up to this bar (no lookahead)
            ohlcv = {
                'opens': opens[max(0, i-lookback):i+1],
                'highs': highs[max(0, i-lookback):i+1],
                'lows': lows[max(0, i-lookback):i+1],
                'closes': closes[max(0, i-lookback):i+1],
                'volumes': volumes[max(0, i-lookback):i+1],
            }

            price = closes[i]

            # === MANAGE EXISTING POSITION ===
            if asset in self.positions:
                pos = self.positions[asset]
                result = self.position_mgr.update_position(pos, i, bar_ts, price, ohlcv)

                if result[0] is None and result[1] is not None:
                    # Position closed
                    trade = result[1]
                    self.trades.append(trade)
                    del self.positions[asset]

                    # Update equity
                    self.equity += trade.pnl_usd
                    self.cash = self.equity

                    is_loss = trade.pnl_pct <= 0
                    self.filters.record_trade_close(asset, bar_ts, is_loss)

                    if verbose:
                        w = "WIN" if trade.pnl_pct > 0 else "LOSS"
                        print(f"  [{i}] CLOSE {trade.direction} {w}: {trade.pnl_pct:+.2f}% | {trade.exit_reason} | {trade.duration_min:.0f}min | SL={trade.max_sl_level}")

                else:
                    self.positions[asset] = result[0]

                self.equity_curve.append(self.equity)
                continue

            # === CHECK FILTERS ===
            passed, reason = self.filters.check_all(asset, bar_ts, ohlcv['closes'])
            if not passed:
                self.equity_curve.append(self.equity)
                continue

            # === COMPUTE SIGNAL (primary TF) ===
            sig = compute_tf_signal(ohlcv, self.ema_period)
            primary_signal = sig['signal']

            if primary_signal == 'NEUTRAL':
                self.equity_curve.append(self.equity)
                continue

            self.signals_generated += 1

            # === MULTI-TF ALIGNMENT (context TFs) ===
            htf_alignment = 0
            for ctx_tf in ['15m', '1h', '4h']:
                ctx_ohlcv = get_context_at_bar(data, ctx_tf, bar_ts, lookback=50)
                if ctx_ohlcv and len(ctx_ohlcv['closes']) >= 20:
                    ctx_sig = compute_tf_signal(ctx_ohlcv, self.ema_period)
                    if ctx_sig['signal'] == primary_signal:
                        htf_alignment += 1
                    elif ctx_sig['ema_direction'] == sig['ema_direction']:
                        htf_alignment += 0.5

            # === COMPUTE INDICATORS + ENTRY SCORE ===
            indicator_ctx = compute_indicator_context(ohlcv)

            # Trendline analysis
            try:
                from src.indicators.trendlines import get_trendline_context
                tl_ctx = get_trendline_context(
                    ohlcv['highs'], ohlcv['lows'], ohlcv['closes'],
                    bar_idx=len(ohlcv['closes']) - 1, timeframe=data.primary_tf
                )
                indicator_ctx.update({
                    'trendline_breakout': tl_ctx.get('trendline_breakout', 0),
                    'trendline_strength': tl_ctx.get('trendline_strength', 0),
                    'trendline_score_adj': tl_ctx.get('trendline_score_adj', 0),
                })
            except Exception:
                pass  # Don't block trades if trendline detection fails

            entry_score, score_reasons = compute_entry_score(
                primary_signal, ohlcv, sig['ema_vals'],
                sig['ema_direction'], sig['price'], indicator_ctx
            )

            # HTF alignment bonus
            if htf_alignment >= 2:
                entry_score += 1
                score_reasons.append(f"htf_aligned={htf_alignment:.0f}")

            # === DIRECTION ===
            direction = 'LONG' if primary_signal == 'BUY' else 'SHORT'

            # === ENTRY GATE (replaces LLM) ===
            effective_min = self.min_entry_score
            if direction == 'SHORT':
                effective_min += self.short_score_penalty
            if entry_score < effective_min or entry_score > self.max_entry_score:
                self.equity_curve.append(self.equity)
                continue

            self.entries_attempted += 1

            # === POSITION SIZING ===
            size_pct = min(self.risk_per_trade_pct, self.max_trade_pct)
            notional = self.equity * (size_pct / 100.0)
            max_trade = min(2000.0, self.equity * 0.05)
            notional = min(notional, max_trade)

            if notional <= 0 or price <= 0:
                self.equity_curve.append(self.equity)
                continue

            qty = notional / price

            # === OPEN POSITION ===
            pos = self.position_mgr.open_position(
                direction=direction,
                price=price,
                ohlcv=ohlcv,
                bar_index=i,
                bar_ts=bar_ts,
                qty=qty,
                entry_score=entry_score,
                timeframe=data.primary_tf,
            )
            self.positions[asset] = pos
            self.filters.record_trade_open(asset, bar_ts)

            if verbose:
                print(f"  [{i}] OPEN {direction} @ ${price:,.2f} | score={entry_score} ({', '.join(score_reasons[:3])}) | SL=${pos.sl:,.2f} | ${notional:,.0f}")

            self.equity_curve.append(self.equity)

            # Progress
            if (i - lookback) % 5000 == 0 and i > lookback:
                pct = (i - lookback) / (n_bars - lookback) * 100
                elapsed = time.time() - start_time
                trades_so_far = len(self.trades)
                wins = sum(1 for t in self.trades if t.pnl_pct > 0)
                wr = wins / trades_so_far if trades_so_far > 0 else 0
                print(f"  [BACKTEST] {pct:.0f}% | {trades_so_far} trades | WR={wr:.1%} | Equity=${self.equity:,.0f} | {elapsed:.0f}s")

        # Close any remaining position at last price
        if asset in self.positions:
            pos = self.positions[asset]
            final_price = closes[-1]
            if pos.direction == 'LONG':
                pnl_pct = ((final_price - pos.entry_price) / pos.entry_price) * 100
            else:
                pnl_pct = ((pos.entry_price - final_price) / pos.entry_price) * 100
            pnl_usd = pnl_pct / 100.0 * pos.entry_price * pos.qty
            self.trades.append(TradeRecord(
                direction=pos.direction, entry_price=pos.entry_price,
                exit_price=final_price, entry_bar=pos.entry_bar,
                exit_bar=n_bars-1, entry_ts=pos.entry_ts,
                exit_ts=timestamps[-1], qty=pos.qty,
                pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                duration_bars=n_bars-1-pos.entry_bar,
                duration_min=((n_bars-1-pos.entry_bar) * tf_seconds) / 60,
                exit_reason="End of backtest",
                max_sl_level=pos.sl_levels[-1],
                entry_score=pos.entry_score,
                sl_levels_hit=list(pos.sl_levels),
            ))
            self.equity += pnl_usd

        elapsed = time.time() - start_time
        print(f"\n  [BACKTEST] Complete in {elapsed:.1f}s | {len(self.trades)} trades | Signals: {self.signals_generated} | Entries: {self.entries_attempted}")

        # Filter stats
        fstats = self.filters.get_stats()
        if any(v > 0 for v in fstats.values()):
            print(f"  [FILTERS] {fstats}")

        return BacktestMetrics(self.trades, self.equity_curve, self.initial_capital)
