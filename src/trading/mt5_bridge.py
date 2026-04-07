"""
MetaTrader 5 Bridge — Mirror & Execute Trades
================================================
Two modes:
  1. MIRROR mode: Trades execute on primary exchange (Bybit/Delta),
     and are mirrored to MT5 for visual tracking on charts.
  2. EXECUTE mode: Trades execute directly through MT5 broker.

Supports:
  - Crypto CFDs (BTCUSD, ETHUSD) on MT5 brokers
  - Forex pairs if you want to extend the system
  - Real-time position sync
  - Visual trade markers on MT5 charts
  - P&L tracking in MT5 terminal

Setup:
  1. Install MetaTrader 5 terminal (download from broker)
  2. Enable Algo Trading in MT5: Tools → Options → Expert Advisors → Allow
  3. pip install MetaTrader5
  4. Add mt5 config to config.yaml

config.yaml example:
  mt5:
    enabled: true
    mode: mirror          # 'mirror' or 'execute'
    terminal_path: "C:/Program Files/MetaTrader 5/terminal64.exe"
    login: 12345678
    password: "your_password"
    server: "BrokerName-Server"
    symbol_map:
      BTC: "BTCUSD"      # Your broker's BTC symbol
      ETH: "ETHUSD"      # Your broker's ETH symbol
    lot_size:
      BTC: 0.01           # Min lot for BTC CFD
      ETH: 0.1            # Min lot for ETH CFD
    magic_number: 888888  # Unique ID for our EA trades
"""

import os
import time
import logging
import json
from typing import Optional, Dict, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import MT5 — graceful fallback if not installed
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None


# ═══════════════════════════════════════════════════════════════════
# MT5 Symbol Mapping
# ═══════════════════════════════════════════════════════════════════

# Common broker symbol names for crypto CFDs
DEFAULT_SYMBOL_MAP = {
    'BTC': ['BTCUSD', 'BTCUSD.m', 'BTCUSDm', 'Bitcoin', 'BTCUSDT', '#BTCUSD', 'BTC/USD', 'BTC'],
    'ETH': ['ETHUSD', 'ETHUSD.m', 'ETHUSDm', 'Ethereum', 'ETHUSDT', '#ETHUSD', 'ETH/USD', 'ETH'],
    'SOL': ['SOLUSD', 'SOLUSD.m', '#SOLUSD', 'SOL'],
    'XRP': ['XRPUSD', 'XRPUSD.m', '#XRPUSD', 'XRP'],
}

# MT5 timeframe mapping
TF_MAP = {
    '1m': mt5.TIMEFRAME_M1 if mt5 else 1,
    '5m': mt5.TIMEFRAME_M5 if mt5 else 5,
    '15m': mt5.TIMEFRAME_M15 if mt5 else 15,
    '1h': mt5.TIMEFRAME_H1 if mt5 else 60,
    '4h': mt5.TIMEFRAME_H4 if mt5 else 240,
    '1d': mt5.TIMEFRAME_D1 if mt5 else 1440,
}


class MT5Bridge:
    """
    Bridge between our trading system and MetaTrader 5.

    Supports two modes:
    - MIRROR: Replicate trades for visualization (primary exchange still executes)
    - EXECUTE: Route trades directly through MT5 broker
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.mt5_config = self.config.get('mt5', {})
        self.enabled = self.mt5_config.get('enabled', False)
        self.mode = self.mt5_config.get('mode', 'mirror')  # 'mirror' or 'execute'
        self.magic = self.mt5_config.get('magic_number', 888888)
        self.connected = False
        self.positions: Dict[str, dict] = {}  # Our tracked positions on MT5
        self._symbol_cache: Dict[str, str] = {}  # asset -> MT5 symbol

        # Symbol mapping from config or defaults
        self._symbol_map = self.mt5_config.get('symbol_map', {})
        self._lot_sizes = self.mt5_config.get('lot_size', {'BTC': 0.01, 'ETH': 0.1})

        # Trade log for P&L tracking
        self._trade_log: List[dict] = []
        self._log_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'logs', 'mt5_trades.jsonl'
        )

        if self.enabled:
            self._connect()

    # ─── Connection Management ───────────────────────────────────

    def _connect(self) -> bool:
        """Initialize MT5 terminal connection."""
        if not MT5_AVAILABLE:
            print("[MT5] MetaTrader5 package not installed. Run: pip install MetaTrader5")
            self.enabled = False
            return False

        terminal_path = self.mt5_config.get('terminal_path', None)
        login = self.mt5_config.get('login', None)
        password = self.mt5_config.get('password', '')
        server = self.mt5_config.get('server', '')

        # Initialize MT5
        init_kwargs = {}
        if terminal_path:
            init_kwargs['path'] = terminal_path
        if login:
            init_kwargs['login'] = int(login)
        if password:
            init_kwargs['password'] = str(password)
        if server:
            init_kwargs['server'] = str(server)

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            print(f"[MT5] Failed to initialize: {error}")
            print("[MT5] Make sure MT5 terminal is installed and Algo Trading is enabled")
            print("[MT5]   Tools -> Options -> Expert Advisors -> Allow Algorithmic Trading")
            self.enabled = False
            return False

        # Get account info
        account = mt5.account_info()
        if account:
            print(f"[MT5] Connected to {account.server}")
            print(f"[MT5]   Account: {account.login} ({account.name})")
            print(f"[MT5]   Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")
            print(f"[MT5]   Leverage: 1:{account.leverage}")
            print(f"[MT5]   Mode: {self.mode.upper()}")
            self.connected = True
        else:
            print(f"[MT5] Connected but no account info — check credentials")
            self.connected = True

        # Resolve symbols
        self._resolve_symbols()
        return True

    def _resolve_symbols(self):
        """Find the correct MT5 symbol names for our assets."""
        if not self.connected:
            return

        for asset, configured_symbol in self._symbol_map.items():
            # If user specified exact symbol in config, verify it exists
            info = mt5.symbol_info(configured_symbol)
            if info:
                self._symbol_cache[asset] = configured_symbol
                if not info.visible:
                    mt5.symbol_select(configured_symbol, True)
                print(f"[MT5]   {asset} -> {configured_symbol} (configured)")
                continue

        # Auto-discover symbols not in config
        for asset, candidates in DEFAULT_SYMBOL_MAP.items():
            if asset in self._symbol_cache:
                continue
            for candidate in candidates:
                info = mt5.symbol_info(candidate)
                if info:
                    self._symbol_cache[asset] = candidate
                    if not info.visible:
                        mt5.symbol_select(candidate, True)
                    print(f"[MT5]   {asset} -> {candidate} (auto-detected)")
                    break

        if not self._symbol_cache:
            print("[MT5] WARNING: No crypto symbols found on this broker")
            print("[MT5]   Available symbols with 'BTC' or 'ETH':")
            all_symbols = mt5.symbols_get()
            if all_symbols:
                for s in all_symbols:
                    if 'BTC' in s.name.upper() or 'ETH' in s.name.upper():
                        print(f"[MT5]     {s.name} (spread={s.spread})")

    def get_mt5_symbol(self, asset: str) -> Optional[str]:
        """Get MT5 symbol name for an asset."""
        return self._symbol_cache.get(asset.upper())

    def _get_filling_mode(self, symbol: str) -> int:
        """Detect the correct filling mode for a symbol (broker-specific)."""
        info = mt5.symbol_info(symbol)
        if not info:
            return mt5.ORDER_FILLING_IOC
        # Check bitmask: bit0=FOK, bit1=IOC, bit2=RETURN
        if info.filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        elif info.filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def is_available(self, asset: str) -> bool:
        """Check if we can trade this asset on MT5."""
        return self.connected and self.enabled and asset.upper() in self._symbol_cache

    # ─── Trade Execution (EXECUTE mode) ──────────────────────────

    def open_position(self, asset: str, direction: str, price: float,
                      qty: float, sl: float, comment: str = "") -> dict:
        """
        Open a position on MT5.

        Args:
            asset: 'BTC' or 'ETH'
            direction: 'LONG' or 'SHORT'
            price: Target entry price
            qty: Position size in asset units
            sl: Stop loss price
            comment: Trade comment/reason

        Returns:
            dict with 'status', 'order_id', 'fill_price', 'message'
        """
        if not self.is_available(asset):
            return {'status': 'error', 'message': f'MT5 not available for {asset}'}

        symbol = self._symbol_cache[asset]
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return {'status': 'error', 'message': f'Symbol info unavailable: {symbol}'}

        # Calculate lot size
        lot = self._calculate_lots(asset, qty, symbol_info)

        # Order type
        if direction == 'LONG':
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid

        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "deviation": 20,  # Max slippage in points
            "magic": self.magic,
            "comment": f"EMA8_{direction}_{comment[:20]}" if comment else f"EMA8_{direction}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }

        # Send order
        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            return {'status': 'error', 'message': f'Order send failed: {error}'}

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            fill_price = result.price
            self.positions[asset] = {
                'ticket': result.order,
                'deal': result.deal,
                'direction': direction,
                'entry_price': fill_price,
                'lot': lot,
                'sl': sl,
                'symbol': symbol,
                'open_time': time.time(),
                'comment': comment,
            }

            # Log trade
            self._log_trade('OPEN', asset, direction, fill_price, lot, sl, comment)

            print(f"  [MT5:{asset}] OPENED {direction} {lot} lots @ ${fill_price:,.2f} | SL: ${sl:,.2f}")
            return {
                'status': 'success',
                'order_id': str(result.order),
                'deal_id': str(result.deal),
                'fill_price': fill_price,
            }
        else:
            msg = f"Retcode {result.retcode}: {result.comment}"
            print(f"  [MT5:{asset}] ORDER REJECTED: {msg}")
            return {'status': 'error', 'message': msg}

    def close_position(self, asset: str, price: float, reason: str = "") -> dict:
        """Close a position on MT5."""
        if asset not in self.positions:
            # Try to find by scanning MT5 positions
            return self._close_by_symbol(asset, reason)

        pos = self.positions[asset]
        symbol = pos['symbol']
        ticket = pos.get('ticket', 0)
        direction = pos['direction']

        # Close = opposite order
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {'status': 'error', 'message': f'No tick data for {symbol}'}

        if direction == 'LONG':
            close_type = mt5.ORDER_TYPE_SELL
            close_price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            close_price = tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos['lot'],
            "type": close_type,
            "price": close_price,
            "deviation": 20,
            "magic": self.magic,
            "comment": f"CLOSE_{reason[:20]}" if reason else "CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }

        # If we have the position ticket, use it
        if ticket:
            request["position"] = ticket

        result = mt5.order_send(request)

        if result is None:
            error = mt5.last_error()
            return {'status': 'error', 'message': f'Close failed: {error}'}

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            fill_price = result.price
            entry = pos['entry_price']

            if direction == 'LONG':
                pnl_pct = ((fill_price - entry) / entry) * 100
            else:
                pnl_pct = ((entry - fill_price) / entry) * 100

            self._log_trade('CLOSE', asset, direction, fill_price, pos['lot'], 0,
                          f"{reason} | PnL: {pnl_pct:+.2f}%")

            print(f"  [MT5:{asset}] CLOSED {direction} @ ${fill_price:,.2f} | PnL: {pnl_pct:+.2f}% | {reason}")
            del self.positions[asset]
            return {
                'status': 'success',
                'fill_price': fill_price,
                'pnl_pct': pnl_pct,
            }
        else:
            msg = f"Close retcode {result.retcode}: {result.comment}"
            print(f"  [MT5:{asset}] CLOSE FAILED: {msg}")
            return {'status': 'error', 'message': msg}

    def update_sl(self, asset: str, new_sl: float) -> bool:
        """Update stop loss for an existing MT5 position."""
        if asset not in self.positions:
            return False

        pos = self.positions[asset]
        symbol = pos['symbol']
        ticket = pos.get('ticket', 0)

        if not ticket:
            return False

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": 0.0,  # No TP — we use EMA exit
            "magic": self.magic,
        }

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            pos['sl'] = new_sl
            return True
        return False

    def _close_by_symbol(self, asset: str, reason: str) -> dict:
        """Close position by scanning MT5 open positions."""
        symbol = self.get_mt5_symbol(asset)
        if not symbol:
            return {'status': 'error', 'message': f'No MT5 symbol for {asset}'}

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return {'status': 'error', 'message': f'No open positions for {symbol}'}

        results = []
        for pos in positions:
            if pos.magic == self.magic:
                tick = mt5.symbol_info_tick(symbol)
                if pos.type == mt5.ORDER_TYPE_BUY:
                    close_type = mt5.ORDER_TYPE_SELL
                    close_price = tick.bid
                else:
                    close_type = mt5.ORDER_TYPE_BUY
                    close_price = tick.ask

                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": pos.volume,
                    "type": close_type,
                    "position": pos.ticket,
                    "price": close_price,
                    "deviation": 20,
                    "magic": self.magic,
                    "comment": f"CLOSE_{reason[:20]}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": self._get_filling_mode(symbol),
                }
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    results.append(result)

        if results:
            return {'status': 'success', 'closed': len(results)}
        return {'status': 'error', 'message': 'Failed to close positions'}

    # ─── Mirror Mode (replicate trades from primary exchange) ────

    def mirror_open(self, asset: str, direction: str, price: float,
                    qty: float, sl: float, entry_score: int = 0,
                    confidence: float = 0.5) -> bool:
        """
        Mirror an entry from the primary exchange to MT5.
        Called by executor after successful entry on Bybit/Delta.
        """
        if not self.is_available(asset):
            return False

        if self.mode != 'mirror':
            return False

        comment = f"S{entry_score}_C{confidence:.0%}"
        result = self.open_position(asset, direction, price, qty, sl, comment)
        return result.get('status') == 'success'

    def mirror_close(self, asset: str, price: float, reason: str = "") -> bool:
        """Mirror a close from the primary exchange to MT5."""
        if not self.is_available(asset):
            return False
        if self.mode != 'mirror':
            return False

        result = self.close_position(asset, price, reason)
        return result.get('status') == 'success'

    def mirror_sl_update(self, asset: str, new_sl: float) -> bool:
        """Mirror SL ratchet update to MT5."""
        if not self.is_available(asset):
            return False
        if self.mode != 'mirror':
            return False
        return self.update_sl(asset, new_sl)

    # ─── Data Fetching (for MT5-native data) ─────────────────────

    def fetch_ohlcv(self, asset: str, timeframe: str = '5m',
                    count: int = 500) -> Optional[List[dict]]:
        """Fetch OHLCV candles from MT5 terminal."""
        if not self.connected:
            return None

        symbol = self.get_mt5_symbol(asset)
        if not symbol:
            return None

        tf = TF_MAP.get(timeframe, mt5.TIMEFRAME_M5 if mt5 else 5)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            return None

        candles = []
        for r in rates:
            candles.append({
                'timestamp': int(r['time']) * 1000,
                'open': float(r['open']),
                'high': float(r['high']),
                'low': float(r['low']),
                'close': float(r['close']),
                'volume': float(r['tick_volume']),
            })
        return candles

    def get_current_price(self, asset: str) -> Optional[float]:
        """Get current price from MT5."""
        if not self.connected:
            return None
        symbol = self.get_mt5_symbol(asset)
        if not symbol:
            return None
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            return (tick.bid + tick.ask) / 2.0
        return None

    # ─── Account & Position Info ─────────────────────────────────

    def get_account_info(self) -> dict:
        """Get MT5 account information."""
        if not self.connected:
            return {}
        account = mt5.account_info()
        if not account:
            return {}
        return {
            'login': account.login,
            'server': account.server,
            'balance': account.balance,
            'equity': account.equity,
            'margin': account.margin,
            'free_margin': account.margin_free,
            'profit': account.profit,
            'leverage': account.leverage,
        }

    def get_open_positions(self) -> List[dict]:
        """Get all open positions from MT5 placed by our system."""
        if not self.connected:
            return []
        positions = mt5.positions_get()
        if not positions:
            return []

        our_positions = []
        for pos in positions:
            if pos.magic == self.magic:
                our_positions.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'direction': 'LONG' if pos.type == 0 else 'SHORT',
                    'volume': pos.volume,
                    'entry_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'swap': pos.swap,
                    'time': pos.time,
                    'comment': pos.comment,
                })
        return our_positions

    def get_trade_history(self, days: int = 30) -> List[dict]:
        """Get closed trade history from MT5."""
        if not self.connected:
            return []

        from datetime import datetime, timedelta
        date_from = datetime.now() - timedelta(days=days)
        date_to = datetime.now()

        deals = mt5.history_deals_get(date_from, date_to)
        if not deals:
            return []

        trades = []
        for deal in deals:
            if deal.magic == self.magic:
                trades.append({
                    'ticket': deal.ticket,
                    'order': deal.order,
                    'symbol': deal.symbol,
                    'type': 'BUY' if deal.type == 0 else 'SELL',
                    'volume': deal.volume,
                    'price': deal.price,
                    'profit': deal.profit,
                    'commission': deal.commission,
                    'swap': deal.swap,
                    'time': deal.time,
                    'comment': deal.comment,
                })
        return trades

    def print_status(self):
        """Print current MT5 status summary."""
        if not self.connected:
            print("[MT5] Not connected")
            return

        account = self.get_account_info()
        positions = self.get_open_positions()

        print("\n" + "=" * 55)
        print("  MT5 STATUS")
        print("=" * 55)
        print(f"  Account: {account.get('login', '?')} @ {account.get('server', '?')}")
        print(f"  Balance: ${account.get('balance', 0):,.2f}")
        print(f"  Equity:  ${account.get('equity', 0):,.2f}")
        print(f"  P&L:     ${account.get('profit', 0):+,.2f}")
        print(f"  Mode:    {self.mode.upper()}")

        if positions:
            print(f"\n  Open Positions ({len(positions)}):")
            for pos in positions:
                pnl_str = f"${pos['profit']:+,.2f}"
                print(f"    {pos['symbol']} {pos['direction']} {pos['volume']} lots "
                      f"@ ${pos['entry_price']:,.2f} -> ${pos['current_price']:,.2f} "
                      f"[{pnl_str}] SL=${pos['sl']:,.2f}")
        else:
            print("\n  No open positions")
        print("=" * 55)

    # ─── Helpers ─────────────────────────────────────────────────

    def _calculate_lots(self, asset: str, qty: float,
                        symbol_info) -> float:
        """Convert asset quantity to MT5 lot size."""
        # Get broker's lot parameters
        lot_min = symbol_info.volume_min
        lot_max = symbol_info.volume_max
        lot_step = symbol_info.volume_step
        contract_size = symbol_info.trade_contract_size

        # Convert qty to lots
        if contract_size > 0:
            lots = qty / contract_size
        else:
            # Fallback to configured lot sizes
            lots = self._lot_sizes.get(asset, 0.01)

        # Round to lot step
        lots = max(lot_min, min(lot_max, lots))
        lots = round(lots / lot_step) * lot_step
        lots = round(lots, 8)  # Avoid floating point noise

        return lots

    def _log_trade(self, action: str, asset: str, direction: str,
                   price: float, lot: float, sl: float, comment: str):
        """Log trade to JSONL file."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'asset': asset,
            'direction': direction,
            'price': price,
            'lot': lot,
            'sl': sl,
            'comment': comment,
            'mode': self.mode,
        }
        self._trade_log.append(entry)

        try:
            os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
            with open(self._log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.debug(f"[MT5] Log write failed: {e}")

    def shutdown(self):
        """Clean shutdown of MT5 connection."""
        if self.connected and MT5_AVAILABLE:
            # Close all our positions if in mirror mode
            if self.mode == 'mirror' and self.positions:
                print(f"[MT5] Closing {len(self.positions)} mirrored positions...")
                for asset in list(self.positions.keys()):
                    self.close_position(asset, 0, "SHUTDOWN")

            mt5.shutdown()
            self.connected = False
            print("[MT5] Disconnected")

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Convenience: Quick status check
# ═══════════════════════════════════════════════════════════════════

def check_mt5_setup():
    """Quick diagnostic: check if MT5 is properly set up."""
    print("=" * 55)
    print("  MT5 SETUP CHECK")
    print("=" * 55)

    if not MT5_AVAILABLE:
        print("  [FAIL] MetaTrader5 package not installed")
        print("         Run: pip install MetaTrader5")
        return False

    print("  [OK] MetaTrader5 package installed")

    # Try to connect
    if not mt5.initialize():
        error = mt5.last_error()
        print(f"  [FAIL] Cannot connect to MT5 terminal: {error}")
        print("         1. Install MetaTrader 5 from your broker")
        print("         2. Open MT5 and log in")
        print("         3. Enable: Tools -> Options -> Expert Advisors -> Allow Algo Trading")
        return False

    print("  [OK] MT5 terminal connected")

    account = mt5.account_info()
    if account:
        print(f"  [OK] Account: {account.login} @ {account.server}")
        print(f"       Balance: ${account.balance:,.2f}")
        print(f"       Leverage: 1:{account.leverage}")

    # Check for crypto symbols
    print("\n  Available crypto symbols:")
    all_symbols = mt5.symbols_get()
    crypto_found = False
    if all_symbols:
        for s in all_symbols:
            name = s.name.upper()
            if any(c in name for c in ['BTC', 'ETH', 'CRYPTO', 'BITCOIN', 'ETHEREUM']):
                spread_pts = s.spread
                print(f"    {s.name} | spread={spread_pts} | min_lot={s.volume_min} | contract={s.trade_contract_size}")
                crypto_found = True

    if not crypto_found:
        print("    No crypto symbols found! Your broker may not offer crypto CFDs.")
        print("    Consider brokers like: Exness, RoboForex, IC Markets, FP Markets")

    mt5.shutdown()
    print("=" * 55)
    return True


if __name__ == '__main__':
    check_mt5_setup()
