"""
Robinhood Paper Trading Fetcher
=================================
Real-time quotes from Robinhood Crypto API + paper position tracking.

Phase 1: Read-only — no real orders placed.
  - Fetches live bid/ask from Robinhood (real account, real spreads)
  - Records entry/exit signals from our strategy engine
  - Tracks paper positions with real market prices
  - Calculates theoretical PnL including Robinhood spreads
  - Logs everything to robinhood_paper.jsonl for analysis

Usage:
    fetcher = RobinhoodPaperFetcher()
    fetcher.record_entry("BTC", "LONG", price=72000, score=5, ...)
    fetcher.update_positions()  # call every poll cycle
    fetcher.record_exit("BTC", reason="SL L2 hit")
    print(fetcher.report())
"""

import os
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class PaperPosition:
    """A simulated position tracked against real Robinhood prices."""
    asset: str
    direction: str              # LONG or SHORT
    entry_price: float          # mid price at signal time
    entry_bid: float            # real Robinhood bid at entry
    entry_ask: float            # real Robinhood ask at entry
    entry_spread_pct: float     # spread at entry
    entry_time: str             # ISO timestamp
    quantity: float             # paper quantity (based on position sizing)
    score: int                  # strategy entry score
    ml_confidence: float = 0.0
    llm_confidence: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    peak_price: float = 0.0    # best price seen (for trailing)
    current_price: float = 0.0
    current_pnl_pct: float = 0.0
    current_pnl_usd: float = 0.0
    status: str = "open"        # open, closed_win, closed_loss, closed_sl, closed_tp
    exit_price: float = 0.0
    exit_time: str = ""
    exit_reason: str = ""
    final_pnl_pct: float = 0.0
    final_pnl_usd: float = 0.0
    bars_held: int = 0


class RobinhoodPaperFetcher:
    """
    Complete paper trading system using Robinhood real-time quotes.

    Hooks into the trading executor's signal pipeline:
    1. When executor generates a LONG/SHORT signal → record_entry()
    2. Every poll cycle → update_positions() with fresh Robinhood prices
    3. When executor closes a position → record_exit()
    4. Generates reports comparing paper PnL vs real market
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._client = None
        self._lock = threading.RLock()  # RLock: reentrant — record_entry calls record_exit
        # Spread cost for P&L deduction
        self._spread_cost_pct = 0.0
        for ex in self.config.get('exchanges', []):
            if ex.get('name', '').lower() == 'robinhood':
                self._spread_cost_pct = ex.get('round_trip_spread_pct', 3.34)
                break

        # Paper state — keyed by unique trade ID to support multiple concurrent positions per asset
        self._next_trade_id: int = 0
        self.positions: Dict[str, PaperPosition] = {}  # trade_id -> open position
        self.closed_trades: List[PaperPosition] = []
        self.initial_capital = self.config.get('initial_capital', 100000.0)
        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital

        # Logs
        self._log_dir = os.path.join(PROJECT_ROOT, 'logs')
        self._trade_log = os.path.join(self._log_dir, 'robinhood_paper.jsonl')
        self._position_log = os.path.join(self._log_dir, 'robinhood_paper_positions.jsonl')
        os.makedirs(self._log_dir, exist_ok=True)

        # Stats
        self.stats = {
            'total_signals': 0,
            'entries': 0,
            'exits': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl_usd': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'price_snapshots': 0,
        }

        # Init Robinhood client
        self._init_client()

    def _init_client(self):
        """Initialize Robinhood API client."""
        try:
            from src.integrations.robinhood_crypto import RobinhoodCryptoClient
            self._client = RobinhoodCryptoClient()
            if self._client.authenticated:
                acct = self._client.get_account()
                if acct and 'error' not in acct:
                    print(f"  [PAPER] Robinhood connected | Account: {acct.get('account_number')} | Buying Power: ${acct.get('buying_power')}")
                    # Use real buying power as reference
                    real_bp = float(acct.get('buying_power', 0))
                    if real_bp > 0:
                        self.initial_capital = real_bp
                        self.equity = real_bp
                        self.peak_equity = real_bp
                else:
                    logger.warning(f"[PAPER] Robinhood auth error: {acct}")
                    self._client = None
            else:
                logger.info("[PAPER] Robinhood not configured — paper tracking with CCXT prices only")
                self._client = None
        except Exception as e:
            logger.warning(f"[PAPER] Robinhood init failed: {e}")
            self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.authenticated

    def get_live_price(self, asset: str) -> Dict[str, float]:
        """
        Get real-time bid/ask/mid from Robinhood.

        Returns:
            {'bid': float, 'ask': float, 'mid': float, 'spread_pct': float, 'timestamp': str}
        """
        if not self._client:
            return {}

        symbol = f"{asset}-USD"
        data = self._client.get_best_price(symbol)
        if data and "results" in data and data["results"]:
            r = data["results"][0]
            bid = float(r.get("bid_inclusive_of_sell_spread", 0))
            ask = float(r.get("ask_inclusive_of_buy_spread", 0))
            mid = float(r.get("price", (bid + ask) / 2 if bid and ask else 0))
            spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 0
            self.stats['price_snapshots'] += 1
            return {
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread_pct': spread_pct,
                'buy_spread': r.get('buy_spread'),
                'sell_spread': r.get('sell_spread'),
                'timestamp': r.get('timestamp', ''),
            }
        return {}

    def get_account_snapshot(self) -> Dict:
        """Get real Robinhood account state."""
        if not self._client:
            return {}
        acct = self._client.get_account()
        holdings = self._client.get_holdings()
        return {
            'account': acct,
            'holdings': holdings.get('results', []) if holdings else [],
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
        }

    # ── Paper Trade Tracking ──

    def record_entry(self, asset: str, direction: str, price: float,
                     score: int, quantity: float = 0.0,
                     sl_price: float = 0.0, tp_price: float = 0.0,
                     ml_confidence: float = 0.0, llm_confidence: float = 0.0,
                     size_pct: float = 0.0, reasoning: str = "",
                     **extra) -> Optional[PaperPosition]:
        """
        Record a paper trade entry with real Robinhood price snapshot.

        Called by executor when it would normally place an order.
        """
        with self._lock:
            # Use the executor's fill price directly — avoids redundant Robinhood API call
            # The executor already fetched the order book and determined the fill price
            fill_price = price
            bid = price
            ask = price
            mid = price
            spread_pct = 0.0

            # Default quantity from position sizing
            if quantity <= 0 and size_pct > 0:
                trade_usd = self.equity * (size_pct / 100.0)
                quantity = trade_usd / fill_price if fill_price > 0 else 0

            pos = PaperPosition(
                asset=asset,
                direction=direction,
                entry_price=fill_price,
                entry_bid=bid,
                entry_ask=ask,
                entry_spread_pct=spread_pct,
                entry_time=datetime.now(tz=timezone.utc).isoformat(),
                quantity=quantity,
                score=score,
                ml_confidence=ml_confidence,
                llm_confidence=llm_confidence,
                sl_price=sl_price,
                tp_price=tp_price,
                peak_price=fill_price,
                current_price=fill_price,
            )

            # Unique trade ID so multiple positions on same asset don't overwrite
            self._next_trade_id += 1
            trade_id = f"{asset}_{self._next_trade_id}"
            self.positions[trade_id] = pos
            self.stats['entries'] += 1
            self.stats['total_signals'] += 1

            # Log
            entry_log = {
                'event': 'ENTRY',
                'timestamp': pos.entry_time,
                'asset': asset,
                'trade_id': trade_id,
                'direction': direction,
                'fill_price': fill_price,
                'rh_bid': bid,
                'rh_ask': ask,
                'rh_mid': mid,
                'spread_pct': round(spread_pct, 4),
                'quantity': quantity,
                'score': score,
                'sl': sl_price,
                'tp': tp_price,
                'ml_confidence': ml_confidence,
                'llm_confidence': llm_confidence,
                'size_pct': size_pct,
                'reasoning': reasoning[:200] if reasoning else '',
                'equity': round(self.equity, 2),
            }
            self._append_log(self._trade_log, entry_log)

            print(f"  [PAPER] ENTRY {direction} {asset} @ ${fill_price:,.2f} "
                  f"(bid=${bid:,.2f} ask=${ask:,.2f} spread={spread_pct:.2f}%) "
                  f"qty={quantity:.6f} score={score} [id={trade_id}]")

            return pos

    def record_exit(self, asset: str, reason: str = "",
                    exit_price_override: float = None) -> Optional[PaperPosition]:
        """
        Record a paper trade exit with real Robinhood price snapshot.

        Called by executor when it would normally close a position.
        """
        with self._lock:
            # Find position by asset name (returns first matching open position for this asset)
            # Supports both old-style "BTC" keys and new-style "BTC_1" trade IDs
            trade_key = None
            pos = None
            if asset in self.positions:
                # Legacy key match
                trade_key = asset
                pos = self.positions[asset]
            else:
                # Search by asset field in new trade_id format
                for key, p in self.positions.items():
                    if p.asset == asset:
                        trade_key = key
                        pos = p
                        break

            if pos is None:
                return None

            # Use executor's price — avoids redundant Robinhood API call
            bid = pos.current_price
            ask = pos.current_price

            if exit_price_override:
                fill_price = exit_price_override
            else:
                fill_price = pos.current_price

            # Calculate PnL (with spread deduction for honest reporting)
            if pos.direction == "LONG":
                pnl_pct = ((fill_price - pos.entry_price) / pos.entry_price) * 100
            else:
                pnl_pct = ((pos.entry_price - fill_price) / pos.entry_price) * 100

            # Deduct round-trip spread cost for realistic P&L
            if self._spread_cost_pct > 0:
                pnl_pct -= self._spread_cost_pct

            pnl_usd = pos.quantity * pos.entry_price * (pnl_pct / 100)

            # Update position
            pos.exit_price = fill_price
            pos.exit_time = datetime.now(tz=timezone.utc).isoformat()
            pos.exit_reason = reason
            pos.final_pnl_pct = pnl_pct
            pos.final_pnl_usd = pnl_usd
            pos.status = "closed_win" if pnl_pct > 0 else "closed_loss"

            # Update equity
            self.equity += pnl_usd
            if self.equity > self.peak_equity:
                self.peak_equity = self.equity

            # Stats
            self.stats['exits'] += 1
            if pnl_pct > 0:
                self.stats['wins'] += 1
                self.stats['largest_win'] = max(self.stats['largest_win'], pnl_usd)
            else:
                self.stats['losses'] += 1
                self.stats['largest_loss'] = min(self.stats['largest_loss'], pnl_usd)
            self.stats['total_pnl_usd'] += pnl_usd

            # Move to closed
            self.closed_trades.append(pos)
            del self.positions[trade_key]

            # Log (fixed: was referencing undefined 'rh_price' — now uses local 'bid'/'ask')
            exit_log = {
                'event': 'EXIT',
                'timestamp': pos.exit_time,
                'asset': asset,
                'trade_id': trade_key,
                'direction': pos.direction,
                'entry_price': pos.entry_price,
                'exit_price': fill_price,
                'rh_bid': bid,
                'rh_ask': ask,
                'pnl_pct': round(pnl_pct, 4),
                'pnl_usd': round(pnl_usd, 2),
                'reason': reason,
                'equity': round(self.equity, 2),
                'bars_held': pos.bars_held,
                'entry_spread_pct': pos.entry_spread_pct,
            }
            self._append_log(self._trade_log, exit_log)

            tag = "WIN" if pnl_pct > 0 else "LOSS"
            print(f"  [PAPER] EXIT {pos.direction} {asset} @ ${fill_price:,.2f} "
                  f"| {tag} {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | {reason} "
                  f"| equity=${self.equity:,.2f} [id={trade_key}]")

            return pos

    def update_positions(self, live_prices: dict = None):
        """
        Update all open paper positions with current prices.

        Args:
            live_prices: Dict of {asset: price} from executor's OHLCV feed.
                         If None, skips update (no Robinhood API call needed —
                         the executor already manages SL/TP via _manage_position).
        """
        with self._lock:
            for trade_id, pos in list(self.positions.items()):
                asset = pos.asset
                mid = 0
                if live_prices and asset in live_prices:
                    mid = live_prices[asset]
                else:
                    continue  # No price available, skip

                if mid <= 0:
                    continue

                pos.current_price = mid
                pos.bars_held += 1

                # Track peak
                if pos.direction == "LONG" and mid > pos.peak_price:
                    pos.peak_price = mid
                elif pos.direction == "SHORT" and mid < pos.peak_price:
                    pos.peak_price = mid

                # Calculate unrealized PnL
                if pos.direction == "LONG":
                    pos.current_pnl_pct = ((mid - pos.entry_price) / pos.entry_price) * 100
                else:
                    pos.current_pnl_pct = ((pos.entry_price - mid) / pos.entry_price) * 100

                pos.current_pnl_usd = pos.quantity * pos.entry_price * (pos.current_pnl_pct / 100)

    def log_signal(self, asset: str, direction: str, score: int,
                   action_taken: str = "SKIP", reason: str = "",
                   ml_confidence: float = 0.0, llm_confidence: float = 0.0):
        """
        Log any signal (even ones that don't result in entries).
        Useful for analyzing filter effectiveness.
        """
        rh_price = self.get_live_price(asset)
        entry = {
            'event': 'SIGNAL',
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'asset': asset,
            'direction': direction,
            'score': score,
            'action': action_taken,
            'reason': reason[:200] if reason else '',
            'ml_confidence': ml_confidence,
            'llm_confidence': llm_confidence,
            'rh_mid': rh_price.get('mid', 0),
            'rh_spread_pct': rh_price.get('spread_pct', 0),
        }
        self.stats['total_signals'] += 1
        self._append_log(self._trade_log, entry)

    # ── Reporting ──

    def report(self) -> str:
        """Generate paper trading performance report."""
        total = len(self.closed_trades)
        if total == 0:
            open_str = ""
            for trade_id, pos in self.positions.items():
                open_str += f"\n    {pos.direction} {pos.asset} @ ${pos.entry_price:,.2f} | P&L: {pos.current_pnl_pct:+.2f}% (${pos.current_pnl_usd:+,.2f}) [{trade_id}]"
            return (
                f"\n{'='*60}\n"
                f"  ROBINHOOD PAPER TRADING REPORT\n"
                f"{'='*60}\n"
                f"  Status: {'CONNECTED' if self.connected else 'OFFLINE'}\n"
                f"  Signals logged: {self.stats['total_signals']}\n"
                f"  Open positions: {len(self.positions)}{open_str}\n"
                f"  Closed trades: 0\n"
                f"  Equity: ${self.equity:,.2f}\n"
                f"{'='*60}"
            )

        wins = self.stats['wins']
        losses = self.stats['losses']
        wr = (wins / total * 100) if total > 0 else 0

        # Profit factor
        gross_profit = sum(t.final_pnl_usd for t in self.closed_trades if t.final_pnl_usd > 0)
        gross_loss = abs(sum(t.final_pnl_usd for t in self.closed_trades if t.final_pnl_usd < 0))
        pf = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        # Drawdown
        dd_pct = ((self.peak_equity - self.equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0

        # Average trade
        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0

        # Spread impact
        total_spread_cost = sum(
            t.entry_spread_pct * t.quantity * t.entry_price / 100
            for t in self.closed_trades
        )

        # Per-asset breakdown
        asset_pnl = {}
        for t in self.closed_trades:
            if t.asset not in asset_pnl:
                asset_pnl[t.asset] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
            asset_pnl[t.asset]['trades'] += 1
            if t.final_pnl_usd > 0:
                asset_pnl[t.asset]['wins'] += 1
            asset_pnl[t.asset]['pnl'] += t.final_pnl_usd

        # Open positions
        open_str = ""
        for trade_id, pos in self.positions.items():
            open_str += f"\n    {pos.direction} {pos.asset} @ ${pos.entry_price:,.2f} | P&L: {pos.current_pnl_pct:+.2f}% (${pos.current_pnl_usd:+,.2f}) | {pos.bars_held} bars [{trade_id}]"

        # Asset breakdown
        asset_str = ""
        for a, d in asset_pnl.items():
            a_wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            asset_str += f"\n    {a}: {d['trades']} trades | WR={a_wr:.0f}% | PnL=${d['pnl']:+,.2f}"

        return (
            f"\n{'='*60}\n"
            f"  ROBINHOOD PAPER TRADING REPORT\n"
            f"{'='*60}\n"
            f"  Status: {'CONNECTED' if self.connected else 'OFFLINE'}\n"
            f"  Equity: ${self.equity:,.2f} (started ${self.initial_capital:,.2f})\n"
            f"  Total PnL: ${self.stats['total_pnl_usd']:+,.2f}\n"
            f"  Drawdown: {dd_pct:.1f}% from peak ${self.peak_equity:,.2f}\n"
            f"\n  Trades: {total} ({wins}W / {losses}L)\n"
            f"  Win Rate: {wr:.1f}%\n"
            f"  Profit Factor: {pf:.2f}\n"
            f"  Avg Win: ${avg_win:,.2f} | Avg Loss: ${avg_loss:,.2f}\n"
            f"  Largest Win: ${self.stats['largest_win']:+,.2f}\n"
            f"  Largest Loss: ${self.stats['largest_loss']:+,.2f}\n"
            f"  Spread Cost: ${total_spread_cost:,.2f}\n"
            f"\n  Per-Asset:{asset_str}\n"
            f"\n  Open Positions ({len(self.positions)}):{open_str or ' none'}\n"
            f"  Signals Logged: {self.stats['total_signals']}\n"
            f"  Price Snapshots: {self.stats['price_snapshots']}\n"
            f"{'='*60}"
        )

    def positions_status(self) -> str:
        """Quick one-line status for main loop output."""
        if not self.positions:
            return "[PAPER] No open positions"
        parts = []
        for trade_id, pos in self.positions.items():
            parts.append(f"{pos.direction[0]}{pos.asset}:{pos.current_pnl_pct:+.1f}%")
        total_unrealized = sum(p.current_pnl_usd for p in self.positions.values())
        closed_pnl = self.stats['total_pnl_usd']
        return f"[PAPER] {' | '.join(parts)} ({len(self.positions)} open) | unrealized=${total_unrealized:+,.0f} realized=${closed_pnl:+,.0f}"

    # ── Persistence ──

    def save_state(self):
        """Save current positions and closed trades to disk."""
        state = {
            'timestamp': datetime.now(tz=timezone.utc).isoformat(),
            'equity': self.equity,
            'peak_equity': self.peak_equity,
            'initial_capital': self.initial_capital,
            'stats': self.stats,
            'next_trade_id': self._next_trade_id,
            'positions': {tid: vars(p) for tid, p in self.positions.items()},
            'closed_count': len(self.closed_trades),
        }
        path = os.path.join(self._log_dir, 'robinhood_paper_state.json')
        try:
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"[PAPER] State save failed: {e}")

    def load_state(self):
        """Load previous paper trading state from disk."""
        path = os.path.join(self._log_dir, 'robinhood_paper_state.json')
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            self.equity = state.get('equity', self.initial_capital)
            self.peak_equity = state.get('peak_equity', self.equity)
            self.stats = state.get('stats', self.stats)
            self._next_trade_id = state.get('next_trade_id', 0)
            # Positions are reconstructed from log on restart
            loaded_pos = state.get('positions', {})
            for trade_id, pdata in loaded_pos.items():
                self.positions[trade_id] = PaperPosition(**pdata)
                # Update next_trade_id to be higher than any loaded ID
                try:
                    num = int(trade_id.split('_')[-1])
                    self._next_trade_id = max(self._next_trade_id, num)
                except (ValueError, IndexError):
                    pass
            if loaded_pos:
                print(f"  [PAPER] Restored {len(loaded_pos)} open positions from state")
        except Exception as e:
            logger.warning(f"[PAPER] State load failed: {e}")

    def _append_log(self, path: str, data: dict):
        """Append a JSON line to a log file."""
        try:
            with open(path, 'a') as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.warning(f"[PAPER] Log write failed: {e}")
