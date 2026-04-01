"""
Enhanced Price Data Fetcher
============================
Multi-timeframe OHLCV data from CCXT (Binance and other exchanges).
Supports:
  - Historical OHLCV candle fetching
  - Multi-timeframe data (1m, 5m, 15m, 1h, 4h, 1d)
  - Ticker (current price) snapshot
  - Data normalization and validation
  - Binance Testnet for paper trading with fake money
  - Alpaca Paper Trading (crypto + stocks)
  - LiveCoinWatch fast price data
  - Order placement (testnet / live)
"""

import os
import time
import threading
import logging
import json
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# LiveCoinWatch Fast Market Data
# ═══════════════════════════════════════════════════════════════════

class LiveCoinWatchFetcher:
    """
    Fast crypto price data from LiveCoinWatch API.
    Lower latency than CCXT for simple price/ticker lookups.
    Used as primary price source with CCXT as fallback.
    """

    BASE_URL = "https://api.livecoinwatch.com"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('LIVECOINWATCH_API_KEY', '')
        self.available = bool(self.api_key)
        self._session = None
        if self.available:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update({
                    'content-type': 'application/json',
                    'x-api-key': self.api_key,
                })
                logger.info(f"[LCW] LiveCoinWatch initialized (API key: ...{self.api_key[-6:]})")
            except ImportError:
                self.available = False

    def _coin_code(self, symbol: str) -> str:
        """Convert trading pair to LCW coin code. 'ETH/USD' → 'ETH', 'BTC' → 'BTC'."""
        return symbol.split('/')[0].upper().strip()

    def fetch_price(self, symbol: str) -> Optional[float]:
        """Fetch current price for a symbol. Very fast (~100ms)."""
        if not self.available or not self._session:
            return None
        try:
            resp = self._session.post(
                f"{self.BASE_URL}/coins/single",
                json={"currency": "USD", "code": self._coin_code(symbol), "meta": False},
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get('rate', 0))
        except Exception as e:
            logger.debug(f"[LCW] Price fetch failed for {symbol}: {e}")
        return None

    def fetch_multi_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch prices for multiple symbols in one call."""
        if not self.available or not self._session:
            return {}
        try:
            codes = [self._coin_code(s) for s in symbols]
            resp = self._session.post(
                f"{self.BASE_URL}/coins/list",
                json={
                    "currency": "USD",
                    "sort": "rank",
                    "order": "ascending",
                    "offset": 0,
                    "limit": 50,
                    "meta": False,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                result = {}
                for coin in data:
                    code = coin.get('code', '').upper()
                    if code in codes:
                        result[code] = float(coin.get('rate', 0))
                return result
        except Exception as e:
            logger.debug(f"[LCW] Multi-price fetch failed: {e}")
        return {}

    def fetch_coin_history(self, symbol: str, period_days: int = 7) -> List[Dict]:
        """Fetch historical price data (daily)."""
        if not self.available or not self._session:
            return []
        try:
            import datetime
            end = int(datetime.datetime.now().timestamp() * 1000)
            start = end - (period_days * 86400 * 1000)
            resp = self._session.post(
                f"{self.BASE_URL}/coins/single/history",
                json={
                    "currency": "USD",
                    "code": self._coin_code(symbol),
                    "start": start,
                    "end": end,
                    "meta": False,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get('history', [])
        except Exception as e:
            logger.debug(f"[LCW] History fetch failed for {symbol}: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
# Alpaca Paper Trading Client
# ═══════════════════════════════════════════════════════════════════

class AlpacaClient:
    """
    Alpaca paper trading client for crypto.
    Uses Alpaca's REST API directly (no SDK dependency needed).
    Paper trading URL: https://paper-api.alpaca.markets
    """

    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(self, api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 paper: bool = True):
        self.api_key = api_key or os.environ.get('APCA_API_KEY_ID', '')
        self.api_secret = api_secret or os.environ.get('APCA_API_SECRET_KEY', '')
        self.paper = paper
        self.base_url = self.PAPER_URL if paper else self.LIVE_URL
        self.available = bool(self.api_key and self.api_secret)
        self._session = None

        if self.available:
            try:
                import requests
                self._session = requests.Session()
                self._session.headers.update({
                    'APCA-API-KEY-ID': self.api_key,
                    'APCA-API-SECRET-KEY': self.api_secret,
                    'Content-Type': 'application/json',
                })
                # Verify connection
                resp = self._session.get(f"{self.base_url}/v2/account", timeout=10)
                if resp.status_code == 200:
                    acct = resp.json()
                    equity = float(acct.get('equity', 0))
                    cash = float(acct.get('cash', 0))
                    logger.info(f"[ALPACA] Connected to {'PAPER' if paper else 'LIVE'} trading")
                    logger.info(f"[ALPACA] Account equity: ${equity:,.2f} | Cash: ${cash:,.2f}")
                else:
                    logger.warning(f"[ALPACA] Auth failed: {resp.status_code} - {resp.text[:200]}")
                    self.available = False
            except Exception as e:
                logger.warning(f"[ALPACA] Connection failed: {e}")
                self.available = False

    def get_account(self) -> Dict:
        """Get account info (equity, cash, buying power)."""
        if not self.available:
            return {'error': 'Not connected'}
        try:
            resp = self._session.get(f"{self.base_url}/v2/account", timeout=10)
            if resp.status_code == 200:
                acct = resp.json()
                return {
                    'equity': float(acct.get('equity', 0)),
                    'cash': float(acct.get('cash', 0)),
                    'buying_power': float(acct.get('buying_power', 0)),
                    'portfolio_value': float(acct.get('portfolio_value', 0)),
                    'status': acct.get('status', 'UNKNOWN'),
                    'crypto_status': acct.get('crypto_status', 'UNKNOWN'),
                }
            return {'error': f'HTTP {resp.status_code}'}
        except Exception as e:
            return {'error': str(e)}

    def get_positions(self) -> List[Dict]:
        """Get all open positions."""
        if not self.available:
            return []
        try:
            resp = self._session.get(f"{self.base_url}/v2/positions", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return []

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'market', limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_percent: Optional[float] = None,
                    time_in_force: str = 'gtc') -> Dict:
        """
        Place any order type on Alpaca:
          - market: immediate fill at best price
          - limit: fill at limit_price or better
          - stop: triggers market order when stop_price is hit
          - stop_limit: triggers limit order when stop_price hit, fills at limit_price
          - trailing_stop: trails by trail_percent, triggers when price reverses by that %
        """
        if not self.available:
            return {'status': 'error', 'message': 'Not connected'}

        order_data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side.lower(),
            'type': order_type.lower(),
            'time_in_force': time_in_force,
        }

        # Add price fields based on order type
        if order_type in ('limit', 'stop_limit') and limit_price:
            order_data['limit_price'] = str(round(limit_price, 2))
        if order_type in ('stop', 'stop_limit') and stop_price:
            order_data['stop_price'] = str(round(stop_price, 2))
        if order_type == 'trailing_stop' and trail_percent:
            order_data['trail_percent'] = str(round(trail_percent, 2))

        try:
            resp = self._session.post(
                f"{self.base_url}/v2/orders",
                json=order_data,
                timeout=15,
            )
            if resp.status_code in (200, 201):
                order = resp.json()
                return {
                    'status': 'success',
                    'order_id': order.get('id'),
                    'client_order_id': order.get('client_order_id'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'qty': order.get('qty'),
                    'filled_qty': order.get('filled_qty'),
                    'filled_avg_price': order.get('filled_avg_price'),
                    'status_detail': order.get('status'),
                    'testnet': self.paper,
                }
            else:
                error_detail = resp.json() if resp.headers.get('content-type', '').startswith('application/json') else resp.text
                return {
                    'status': 'error',
                    'message': str(error_detail)[:500],
                    'http_code': resp.status_code,
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_order(self, order_id: str) -> Dict:
        """Check order status."""
        if not self.available:
            return {'status': 'error'}
        try:
            resp = self._session.get(f"{self.base_url}/v2/orders/{order_id}", timeout=10)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return {'status': 'error'}

    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order."""
        if not self.available:
            return {'status': 'error'}
        try:
            resp = self._session.delete(f"{self.base_url}/v2/orders/{order_id}", timeout=10)
            return {'status': 'cancelled' if resp.status_code in (200, 204) else 'error'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders."""
        if not self.available:
            return {'status': 'error'}
        try:
            resp = self._session.delete(f"{self.base_url}/v2/orders", timeout=10)
            return {'status': 'success', 'cancelled': resp.status_code in (200, 207)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def close_all_positions(self) -> Dict:
        """Close all open positions (fresh start)."""
        if not self.available:
            return {'status': 'error'}
        try:
            resp = self._session.delete(f"{self.base_url}/v2/positions", timeout=15)
            return {'status': 'success', 'closed': resp.status_code in (200, 207)}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def fetch_crypto_price(self, symbol: str) -> Optional[float]:
        """Fetch latest crypto price from Alpaca data API."""
        try:
            # Convert BTC/USDT → BTC/USD for Alpaca
            coin = symbol.split('/')[0].upper()
            alpaca_sym = f"{coin}/USD"
            resp = self._session.get(
                f"{self.DATA_URL}/v1beta3/crypto/us/latest/trades",
                params={'symbols': alpaca_sym},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                trades = data.get('trades', {})
                trade = trades.get(alpaca_sym, {})
                return float(trade.get('p', 0)) if trade else None
        except Exception as e:
            logger.debug(f"[ALPACA] Price fetch failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Bybit Testnet Client (Futures — supports LONG + SHORT)
# ═══════════════════════════════════════════════════════════════════

class BybitClient:
    """
    Bybit testnet/live client via CCXT.
    Uses USDT-margined linear perpetual futures for full LONG + SHORT.
    """

    def __init__(self, api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = True):
        self.api_key = api_key or os.environ.get('BYBIT_TESTNET_KEY', '')
        self.api_secret = api_secret or os.environ.get('BYBIT_TESTNET_SECRET', '')
        self.testnet = testnet
        self.available = False
        self.exchange = None

        if not (self.api_key and self.api_secret):
            logger.warning("[BYBIT] No API keys — set BYBIT_TESTNET_KEY / BYBIT_TESTNET_SECRET")
            return

        try:
            import ccxt
            self.exchange = ccxt.bybit({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': testnet,
                'options': {
                    'defaultType': 'linear',
                    'recvWindow': 60000,
                    'adjustForTimeDifference': True,
                },
                'enableRateLimit': True,
            })
            # Sync clock with Bybit server to prevent timestamp rejections
            self.exchange.load_time_difference()
            if hasattr(self.exchange, 'options'):
                self.exchange.options['recvWindow'] = 60000
            logger.info(f"[BYBIT] Time offset: {getattr(self.exchange, 'timeOffset', 0)}ms")

            # Verify auth
            balance = self.exchange.fetch_balance()
            usdt = float(balance.get('USDT', {}).get('total', 0) or 0)
            btc = float(balance.get('BTC', {}).get('total', 0) or 0)
            logger.info(f"[BYBIT] Connected to {'TESTNET' if testnet else 'LIVE'}")
            logger.info(f"[BYBIT] USDT: {usdt:,.2f} | BTC: {btc:,.6f}")
            self.available = True
        except Exception as e:
            logger.warning(f"[BYBIT] Connection failed: {e}")
            self.available = False

    def get_account(self) -> Dict:
        if not self.available:
            return {'error': 'Not connected'}
        try:
            balance = self.exchange.fetch_balance()

            # Extract REAL equity from Bybit's raw API response
            # fetch_balance()['info'] contains the raw Bybit V5 response with:
            #   totalEquity = wallet balance + unrealized PnL (REAL account value)
            #   totalWalletBalance = deposited funds only (ignores open positions)
            # We MUST use totalEquity for sizing — wallet balance hides -$31K losses
            info = balance.get('info', {})
            raw_list = info.get('result', {}).get('list', [{}])
            raw_acct = raw_list[0] if raw_list else {}

            total_equity = float(raw_acct.get('totalEquity', 0) or 0)
            total_wallet = float(raw_acct.get('totalWalletBalance', 0) or 0)
            unrealized_pnl = float(raw_acct.get('totalPerpUPL', 0) or 0)
            available_balance = float(raw_acct.get('totalAvailableBalance', 0) or 0)

            # Fallback: if raw API fields are missing, use ccxt balance
            if total_equity <= 0:
                usdt = balance.get('USDT', {})
                btc = balance.get('BTC', {})
                usdt_total = float(usdt.get('total', 0) or 0)
                usdt_free = float(usdt.get('free', 0) or 0)
                btc_total = float(btc.get('total', 0) or 0)
                try:
                    btc_price = float(self.exchange.fetch_ticker('BTC/USDT:USDT').get('last', 0))
                except Exception:
                    btc_price = 0
                total_equity = usdt_total + btc_total * btc_price
                available_balance = usdt_free
                total_wallet = total_equity

            return {
                'equity': total_equity,           # Real equity (wallet + unrealized PnL)
                'wallet_balance': total_wallet,    # Wallet only (deposits - withdrawals)
                'unrealized_pnl': unrealized_pnl,  # Open position PnL
                'cash': available_balance,          # Available to trade
                'buying_power': available_balance * 2,
                'portfolio_value': total_equity,
                'status': 'ACTIVE',
            }
        except Exception as e:
            return {'error': str(e)}

    def fetch_order_book(self, symbol: str, limit: int = 25) -> Dict:
        """Fetch L2 order book from Bybit. Returns {bids: [[price, qty], ...], asks: [[price, qty], ...]}."""
        if not self.available:
            return {'bids': [], 'asks': []}
        try:
            bybit_sym = self._convert_symbol(symbol)
            ob = self.exchange.fetch_order_book(bybit_sym, limit=limit)
            return {
                'bids': ob.get('bids', []),
                'asks': ob.get('asks', []),
            }
        except Exception as e:
            logger.debug(f"[BYBIT] fetch_order_book error: {e}")
            return {'bids': [], 'asks': []}

    def get_positions(self) -> List[Dict]:
        if not self.available:
            return []
        try:
            positions = self.exchange.fetch_positions()
            result = []
            for p in positions:
                contracts = float(p.get('contracts', 0) or 0)
                if contracts > 0:
                    result.append({
                        'symbol': p.get('symbol', ''),
                        'side': p.get('side', ''),
                        'qty': str(contracts),
                        'avg_entry_price': str(p.get('entryPrice', 0)),
                        'current_price': str(p.get('markPrice', 0)),
                        'unrealized_pl': str(p.get('unrealizedPnl', 0)),
                        'market_value': str(contracts * float(p.get('markPrice', 0) or 0)),
                    })
            return result
        except Exception as e:
            logger.debug(f"[BYBIT] get_positions error: {e}")
            return []

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'market', limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_percent: Optional[float] = None,
                    time_in_force: str = 'gtc',
                    reduce_only: bool = False) -> Dict:
        if not self.available:
            return {'status': 'error', 'message': 'Not connected'}

        try:
            # Convert symbol: BTC/USD -> BTC/USDT:USDT, ETH/USDT -> ETH/USDT:USDT
            bybit_sym = self._convert_symbol(symbol)
            params = {}
            if reduce_only:
                params['reduceOnly'] = True

            if order_type == 'limit' and limit_price:
                order = self.exchange.create_order(bybit_sym, 'limit', side, qty, limit_price, params)
            elif order_type in ('stop', 'stop_limit') and stop_price:
                params['stopPrice'] = stop_price
                if order_type == 'stop_limit' and limit_price:
                    order = self.exchange.create_order(bybit_sym, 'limit', side, qty, limit_price, params)
                else:
                    order = self.exchange.create_order(bybit_sym, 'market', side, qty, None, params)
            else:
                # Pure market order — NO price parameter (Bybit rejects market+price as IOC)
                order = self.exchange.create_order(bybit_sym, 'market', side, qty, None, params)

            return {
                'status': 'success',
                'order_id': order.get('id'),
                'client_order_id': order.get('clientOrderId'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'type': order.get('type'),
                'qty': str(qty),
                'filled_qty': str(order.get('filled', 0)),
                'filled_avg_price': order.get('average') or order.get('price'),
                'status_detail': order.get('status'),
                'testnet': self.testnet,
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)[:500]}

    def cancel_all_orders(self) -> Dict:
        if not self.available:
            return {'status': 'error'}
        try:
            self.exchange.cancel_all_orders('BTC/USDT:USDT')
            self.exchange.cancel_all_orders('ETH/USDT:USDT')
            return {'status': 'success', 'cancelled': True}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def close_all_positions(self) -> Dict:
        if not self.available:
            return {'status': 'error'}
        try:
            positions = self.exchange.fetch_positions()
            for p in positions:
                contracts = float(p.get('contracts', 0) or 0)
                if contracts > 0:
                    close_side = 'sell' if p.get('side') == 'long' else 'buy'
                    self.exchange.create_order(p['symbol'], 'market', close_side, contracts, None, {'reduceOnly': True})
            return {'status': 'success', 'closed': True}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def fetch_crypto_price(self, symbol: str) -> Optional[float]:
        try:
            bybit_sym = self._convert_symbol(symbol)
            ticker = self.exchange.fetch_ticker(bybit_sym)
            return float(ticker.get('last', 0)) if ticker else None
        except Exception:
            return None

    def _convert_symbol(self, symbol: str) -> str:
        """Convert generic symbols to Bybit linear perpetual format.
        Target: BTC/USDT:USDT (ccxt format for Bybit linear perp)
        """
        s = symbol.upper().strip()
        # Already correct format
        if '/USDT:USDT' in s:
            return s
        # BTC/USD -> BTC/USDT:USDT
        if s.endswith('/USD'):
            return s.replace('/USD', '/USDT:USDT')
        # BTC/USDT -> BTC/USDT:USDT
        if s.endswith('/USDT') and ':' not in s:
            return f"{s}:USDT"
        # BTCUSDT -> BTC/USDT:USDT
        if ':' not in s and '/' not in s:
            for base in ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'XRP', 'AVAX', 'LINK', 'DOT']:
                if s.startswith(base):
                    return f"{base}/USDT:USDT"
        # BTC -> BTC/USDT:USDT
        if len(s) <= 5 and '/' not in s:
            return f"{s}/USDT:USDT"
        return s


class DeltaClient:
    """
    Delta Exchange testnet/live client via CCXT.
    Uses correct testnet URL: cdn-ind.testnet.deltaex.org
    (ccxt default URL is wrong for Delta testnet)
    """

    def __init__(self, api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 testnet: bool = True):
        self.api_key = api_key or os.environ.get('DELTA_API_KEY', '')
        self.api_secret = api_secret or os.environ.get('DELTA_API_SECRET', '')
        self.testnet = testnet
        self.available = False
        self.exchange = None

        if not (self.api_key and self.api_secret):
            logger.warning("[DELTA] No API keys — set DELTA_API_KEY / DELTA_API_SECRET")
            return

        try:
            import ccxt
            self.exchange = ccxt.delta({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'sandbox': testnet,
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
            })
            # Override testnet URL — ccxt default is wrong for Delta India demo
            if testnet:
                self.exchange.urls['api'] = {
                    'public': 'https://cdn-ind.testnet.deltaex.org',
                    'private': 'https://cdn-ind.testnet.deltaex.org',
                }

            # Verify auth
            balance = self.exchange.fetch_balance()
            usd = float(balance.get('USD', {}).get('total', 0) or 0)
            logger.info(f"[DELTA] Connected to {'TESTNET' if testnet else 'LIVE'}")
            logger.info(f"[DELTA] USD: {usd:,.2f}")
            self.available = True
        except Exception as e:
            logger.warning(f"[DELTA] Connection failed: {e}")
            self.available = False

    def get_account(self) -> Dict:
        if not self.available:
            return {'error': 'Not connected'}
        try:
            balance = self.exchange.fetch_balance()
            usd = balance.get('USD', {})
            usd_total = float(usd.get('total', 0) or 0)
            usd_free = float(usd.get('free', 0) or 0)
            return {
                'equity': usd_total,
                'cash': usd_free,
                'buying_power': usd_free,
                'portfolio_value': usd_total,
                'status': 'ACTIVE',
            }
        except Exception as e:
            return {'error': str(e)}

    def get_positions(self) -> List[Dict]:
        if not self.available:
            return []
        try:
            positions = self.exchange.fetch_positions()
            result = []
            for p in positions:
                contracts = float(p.get('contracts', 0) or 0)
                if contracts > 0:
                    result.append({
                        'symbol': p.get('symbol', ''),
                        'side': p.get('side', ''),
                        'qty': str(contracts),
                        'avg_entry_price': str(p.get('entryPrice', 0)),
                        'current_price': str(p.get('markPrice', 0)),
                        'unrealized_pl': str(p.get('unrealizedPnl', 0)),
                        'market_value': str(contracts * float(p.get('markPrice', 0) or 0)),
                    })
            return result
        except Exception as e:
            logger.debug(f"[DELTA] get_positions error: {e}")
            return []

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = 'market', limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_percent: Optional[float] = None,
                    time_in_force: str = 'gtc',
                    reduce_only: bool = False) -> Dict:
        if not self.available:
            return {'status': 'error', 'message': 'Not connected'}
        try:
            delta_sym = self._convert_symbol(symbol)
            params = {'reduceOnly': True} if reduce_only else {}
            if order_type == 'limit' and limit_price:
                order = self.exchange.create_order(delta_sym, 'limit', side, qty, limit_price, params)
            else:
                order = self.exchange.create_order(delta_sym, 'market', side, qty, None, params)

            return {
                'status': 'success',
                'order_id': str(order.get('id', '')),
                'symbol': order.get('symbol', ''),
                'side': order.get('side', ''),
                'type': order.get('type', ''),
                'amount': float(order.get('amount', 0) or 0),
                'filled_qty': float(order.get('filled', 0) or 0),
                'filled_avg_price': order.get('average') or order.get('price'),
                'testnet': self.testnet,
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'testnet': self.testnet}

    def fetch_order_book(self, symbol: str, limit: int = 25) -> Dict:
        if not self.available:
            return {'bids': [], 'asks': []}
        try:
            delta_sym = self._convert_symbol(symbol)
            ob = self.exchange.fetch_order_book(delta_sym, limit=limit)
            return {'bids': ob.get('bids', []), 'asks': ob.get('asks', [])}
        except Exception as e:
            logger.debug(f"[DELTA] fetch_order_book error: {e}")
            return {'bids': [], 'asks': []}

    def _convert_symbol(self, s: str) -> str:
        # BTC -> BTCUSD, BTC/USDT -> BTCUSD, BTC/USDT:USDT -> BTCUSD
        s = s.replace(':USDT', '').replace('/USDT', '').replace('/USD', '')
        if len(s) <= 5 and '/' not in s:
            return f"{s}USD"
        return s


class PriceFetcher:
    """
    Multi-exchange price data fetcher.
    Supports: Alpaca (paper/live), CCXT exchanges, and LiveCoinWatch for fast data.

    Primary: Alpaca paper trading for order execution.
    Data: LiveCoinWatch for fast prices, CCXT as fallback for OHLCV.
    """

    VALID_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']

    def __init__(self, exchange_name: str = "alpaca",
                 testnet: bool = True,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 max_requests_per_second: int = 10):
        self.testnet = testnet
        self.exchange_name = exchange_name.lower()
        self._authenticated = False

        # Exchange API rate limiter (application-level)
        self._rate_lock = threading.Lock()
        self._rate_timestamps: List[float] = []
        self._max_rps = max_requests_per_second

        # Slippage tracker
        self._slippage_history: List[Dict] = []

        # ── Exchange clients ──
        self.alpaca: Optional[AlpacaClient] = None
        self.bybit: Optional[BybitClient] = None
        self.delta: Optional[DeltaClient] = None

        # ── LiveCoinWatch (fast price data) ──
        self.lcw: Optional[LiveCoinWatchFetcher] = None
        lcw_key = os.environ.get('LIVECOINWATCH_API_KEY', '')
        if lcw_key:
            self.lcw = LiveCoinWatchFetcher(api_key=lcw_key)

        # ── CCXT exchange (fallback for OHLCV data) ──
        self.exchange = None

        # ── Initialize based on exchange_name ──
        if 'delta' in self.exchange_name:
            self._init_delta(api_key, api_secret, testnet)
        elif 'bybit' in self.exchange_name:
            self._init_bybit(api_key, api_secret, testnet)
        elif 'alpaca' in self.exchange_name:
            self._init_alpaca(api_key, api_secret, testnet)
        else:
            self._init_ccxt(exchange_name, testnet, api_key, api_secret)

    def _init_bybit(self, api_key, api_secret, testnet):
        """Initialize Bybit as primary exchange (futures — LONG + SHORT)."""
        self.bybit = BybitClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        self._authenticated = self.bybit.available
        self._available = self.bybit.available

        if self.bybit.available:
            print(f"  [BYBIT] {'TESTNET' if testnet else 'LIVE'} trading connected")
            acct = self.bybit.get_account()
            if 'equity' in acct:
                print(f"  [BYBIT] Equity: ${acct['equity']:,.2f} | USDT: ${acct.get('cash', 0):,.2f}")
            # Use Bybit's own exchange for OHLCV data too
            self.exchange = self.bybit.exchange
            print(f"  [DATA] Bybit CCXT initialized for OHLCV data")
        else:
            print(f"  [BYBIT] Not connected - set BYBIT_TESTNET_KEY / BYBIT_TESTNET_SECRET")
            self._init_ccxt_readonly()

    def _init_delta(self, api_key, api_secret, testnet):
        """Initialize Delta Exchange as primary exchange (futures — LONG + SHORT)."""
        self.delta = DeltaClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
        )
        self._authenticated = self.delta.available
        self._available = self.delta.available

        if self.delta.available:
            print(f"  [DELTA] {'TESTNET' if testnet else 'LIVE'} trading connected")
            acct = self.delta.get_account()
            if 'equity' in acct:
                print(f"  [DELTA] Equity: ${acct['equity']:,.2f} | Cash: ${acct.get('cash', 0):,.2f}")
            self.exchange = self.delta.exchange
            print(f"  [DATA] Delta CCXT initialized for OHLCV data")
        else:
            print(f"  [DELTA] Not connected - set DELTA_API_KEY / DELTA_API_SECRET")
            self._init_ccxt_readonly()

    def _init_alpaca(self, api_key, api_secret, paper):
        """Initialize Alpaca as primary exchange."""
        self.alpaca = AlpacaClient(
            api_key=api_key,
            api_secret=api_secret,
            paper=paper,
        )
        self._authenticated = self.alpaca.available
        self._available = self.alpaca.available

        if self.alpaca.available:
            print(f"  [ALPACA] {'PAPER' if paper else 'LIVE'} trading connected")
            acct = self.alpaca.get_account()
            if 'equity' in acct:
                print(f"  [ALPACA] Equity: ${acct['equity']:,.2f} | Cash: ${acct['cash']:,.2f}")
            self._authenticated = True
            self._available = True
        else:
            print(f"  [ALPACA] Not connected — set APCA_API_KEY_ID / APCA_API_SECRET_KEY env vars")

        # Also init a CCXT exchange for OHLCV data (read-only, no auth needed)
        self._init_ccxt_readonly()

        # System is available if either Alpaca or CCXT works
        if not self._available and self.exchange is not None:
            self._available = True
            print(f"  [INFO] Running with CCXT data only (no order execution without Alpaca)")

    def _init_ccxt_readonly(self):
        """Initialize a CCXT exchange for read-only OHLCV data."""
        try:
            import ccxt
            # Use Kraken (no auth needed for public data) as OHLCV source
            self.exchange = ccxt.kraken({'enableRateLimit': True})
            print(f"  [DATA] Kraken CCXT initialized for OHLCV data (read-only)")
        except Exception as e:
            logger.debug(f"CCXT fallback init failed: {e}")
            self.exchange = None

    def _init_ccxt(self, exchange_name, testnet, api_key, api_secret):
        """Initialize CCXT exchange (legacy/fallback)."""
        try:
            import ccxt
            exchange_config: Dict = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                },
            }
            key = api_key or os.environ.get('BINANCE_TESTNET_KEY' if testnet else 'BINANCE_API_KEY', '')
            secret = api_secret or os.environ.get('BINANCE_TESTNET_SECRET' if testnet else 'BINANCE_API_SECRET', '')
            if key and secret:
                exchange_config['apiKey'] = key
                exchange_config['secret'] = secret
                self._authenticated = True

            self.exchange = getattr(ccxt, exchange_name)(exchange_config)
            try:
                self.exchange.load_time_difference()
            except Exception:
                pass

            if testnet:
                self.exchange.set_sandbox_mode(True)
                print(f"  [TESTNET] Connected to {exchange_name.upper()} Testnet (sandbox mode)")

            self._available = True
        except Exception as e:
            print(f"[WARN] CCXT {exchange_name} not available ({e}).")
            self.exchange = None
            self._available = bool(self.alpaca and self.alpaca.available)

    def _throttle(self):
        """Application-level rate limiter. Blocks if requests exceed max_rps."""
        with self._rate_lock:
            now = time.time()
            # Remove timestamps older than 1 second
            self._rate_timestamps = [t for t in self._rate_timestamps if now - t < 1.0]
            if len(self._rate_timestamps) >= self._max_rps:
                sleep_time = 1.0 - (now - self._rate_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self._rate_timestamps.append(time.time())

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_authenticated(self) -> bool:
        return self._available and self._authenticated

    def get_ticker(self, symbol: str) -> Dict:
        """Get current ticker (price snapshot) for a symbol."""
        if not self._available:
            return {'last': 0.0, 'bid': 0.0, 'ask': 0.0}
        try:
            self._throttle()
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"Warning: fetch_ticker failed for {symbol}: {e}")
            return {'last': 0.0, 'bid': 0.0, 'ask': 0.0}

    def fetch_latest_price(self, symbol: str) -> Optional[float]:
        """Fetch current price. Priority: Bybit → LiveCoinWatch → Alpaca → CCXT."""
        # 1. Bybit (if connected — fastest for futures)
        if self.bybit and self.bybit.available:
            price = self.bybit.fetch_crypto_price(symbol)
            if price and price > 0:
                return price

        # 2. LiveCoinWatch (fast, ~100ms)
        if self.lcw and self.lcw.available:
            price = self.lcw.fetch_price(symbol)
            if price and price > 0:
                return price

        # 3. Alpaca data API
        if self.alpaca and self.alpaca.available:
            price = self.alpaca.fetch_crypto_price(symbol)
            if price and price > 0:
                return price

        # 4. CCXT fallback
        ticker = self.get_ticker(symbol)
        return float(ticker['last']) if ticker.get('last') else None

    def fetch_order_book(self, symbol: str, limit: int = 25) -> Dict:
        """Fetch L2 order book. Routes to Bybit or CCXT."""
        if self.bybit and self.bybit.available:
            return self.bybit.fetch_order_book(symbol, limit=limit)
        if self.exchange and hasattr(self.exchange, 'fetch_order_book'):
            try:
                ob = self.exchange.fetch_order_book(symbol, limit=limit)
                return {'bids': ob.get('bids', []), 'asks': ob.get('asks', [])}
            except Exception:
                pass
        return {'bids': [], 'asks': []}

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d',
                     limit: int = 100) -> List[List[float]]:
        if not self._available:
            raise RuntimeError("Exchange not available")
        if not hasattr(self.exchange, 'fetch_ohlcv'):
            raise RuntimeError('Exchange does not support OHLCV')
        self._throttle()
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_derivatives_data(self, symbol: str, current_price: float = 0.0) -> Dict[str, float]:
        """Fetch funding rates, OI, and advanced derivatives signals."""
        if not self._available or 'binance' not in self.exchange_name.lower():
            return {}
        try:
            perp_symbol = symbol if ':' in symbol else f"{symbol}:USDT"
            fr_data = self.exchange.fetch_funding_rate(perp_symbol) or {}
            oi_data = self.exchange.fetch_open_interest(perp_symbol) or {}
            
            oi = float(oi_data.get('open_interest', oi_data.get('openInterestAmount', 0.0)))
            funding = float(fr_data.get('fundingRate', 0.0))
            
            # 1. Funding Rate Momentum (Institutional Signal 12)
            prev_fr = getattr(self, f"_prev_fr_{symbol}", funding)
            fr_momentum = funding - prev_fr
            setattr(self, f"_prev_fr_{symbol}", funding)
            
            # 2. Perpetual Funding Skew Curve (Institutional Signal 4)
            # Compare funding across exchanges via public Binance futures API
            fr_skew = abs(funding * 0.2)  # Default: 20% of funding as skew

            # 3. Cross-Exchange Price Dislocation (Institutional Signal 5)
            price_dislocation = 0.0

            try:
                import ccxt as _ccxt
                # Reuse cached exchange instances to avoid recreating per call
                if not hasattr(self, '_alt_exchanges'):
                    self._alt_exchanges = {}
                for alt_ex_name in ['bybit', 'okx']:
                    try:
                        if alt_ex_name not in self._alt_exchanges:
                            self._alt_exchanges[alt_ex_name] = getattr(_ccxt, alt_ex_name)({'enableRateLimit': True})
                        alt_ex = self._alt_exchanges[alt_ex_name]

                        # Funding skew
                        alt_fr = alt_ex.fetch_funding_rate(perp_symbol)
                        if alt_fr and alt_fr.get('fundingRate') is not None:
                            fr_skew = abs(funding - float(alt_fr['fundingRate']))

                        # Price dislocation
                        spot_symbol = symbol.split(':')[0] if ':' in symbol else symbol
                        alt_ticker = alt_ex.fetch_ticker(spot_symbol)
                        if alt_ticker and alt_ticker.get('last'):
                            alt_price = float(alt_ticker['last'])
                            price_dislocation = max(price_dislocation, abs(current_price - alt_price))
                        break  # Success on first exchange is sufficient
                    except Exception as e:
                        logger.debug(f"Alt exchange {alt_ex_name} unavailable: {e}")
                        continue
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"Cross-exchange check failed: {e}")
            
            # OI Divergence
            prev_oi = getattr(self, f"_prev_oi_{symbol}", oi)
            oi_change = (oi - prev_oi) / (prev_oi + 1e-10)
            setattr(self, f"_prev_oi_{symbol}", oi)
            
            return {
                'funding_rate': funding,
                'funding_momentum': fr_momentum,
                'funding_skew_divergence': fr_skew,
                'cross_exchange_dislocation': price_dislocation,
                'open_interest': oi,
                'oi_change': float(oi_change)
            }
        except Exception as e:
            logger.debug(f"Derivatives fetch failed for {symbol}: {e}")
            return {}

    def fetch_multi_timeframe(self, symbol: str,
                                timeframes: Optional[List[str]] = None,
                                limit: int = 100) -> Dict[str, List[List[float]]]:
        """
        Fetch OHLCV data across multiple timeframes.
        Returns dict of {timeframe: ohlcv_data}.
        """
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']

        result: Dict[str, List[List[float]]] = {}
        for tf in timeframes:
            try:
                data = self.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
                result[tf] = data
            except Exception as e:
                print(f"Warning: Failed to fetch {tf} data for {symbol}: {e}")
                result[tf] = []
        return result

    # ------------------------------------------------------------------
    # Order execution (testnet or live)
    # ------------------------------------------------------------------

    def get_slippage_stats(self) -> Dict:
        """Get slippage statistics from recent trades."""
        if not self._slippage_history:
            return {'avg_slippage_pct': 0.0, 'max_slippage_pct': 0.0, 'trade_count': 0}
        recent = self._slippage_history[-100:]  # Last 100 trades
        slippages = [s['slippage_pct'] for s in recent]
        return {
            'avg_slippage_pct': sum(slippages) / len(slippages),
            'max_slippage_pct': max(slippages),
            'min_slippage_pct': min(slippages),
            'trade_count': len(recent),
            'high_slippage_count': sum(1 for s in slippages if s > 0.5),
        }

    def get_balance(self) -> Dict:
        """Fetch account balances. Routes to Alpaca or CCXT."""
        if not self.is_authenticated:
            return {'error': 'Not authenticated. Set API key/secret.', 'read_only': True}

        # ── Alpaca balance ──
        if self.alpaca and self.alpaca.available:
            acct = self.alpaca.get_account()
            if 'error' not in acct:
                cash = acct.get('cash', 0)
                equity = acct.get('equity', 0)
                return {
                    'total': {'USD': equity},
                    'free': {'USD': cash},
                    'USDT': cash,  # Treat USD as USDT equivalent
                    'USD': cash,
                    'BTC': 0.0,
                    'ETH': 0.0,
                    'equity': equity,
                    'buying_power': acct.get('buying_power', 0),
                    'exchange': 'alpaca',
                }
            return acct  # error dict

        # ── CCXT balance (fallback) ──
        try:
            self._throttle()
            balance = self.exchange.fetch_balance()
            total = balance.get('total', {})
            free = balance.get('free', {})
            return {
                'total': {k: v for k, v in total.items() if v and v > 0},
                'free': {k: v for k, v in free.items() if v and v > 0},
                'USDT': free.get('USDT', 0.0),
                'BTC': free.get('BTC', 0.0),
                'ETH': free.get('ETH', 0.0),
            }
        except Exception as e:
            error_msg = str(e)
            if '2008' in error_msg or 'Invalid Api-Key' in error_msg or 'Unauthorized' in error_msg:
                return {
                    'error': f'Invalid or expired API credentials. {error_msg}',
                    'invalid_credentials': True,
                    'read_only': True,
                }
            return {'error': str(e)}

    def total_portfolio_value_usd(self, asset_symbols: List[str]) -> Optional[float]:
        """
        Spot portfolio NAV in USDT: USDT balance + sum(holding * USDT price) for each symbol.
        Uses *total* balances from the exchange. Pass configured trading assets plus BNB (fees).
        """
        balance = self.get_balance()
        if 'error' in balance:
            return None
        total_bal = balance.get('total', {})
        if not total_bal:
            return None
        usdt = float(total_bal.get('USDT', 0) or 0)
        nav = usdt
        for sym in asset_symbols:
            if not sym or sym == 'USDT':
                continue
            amt = float(total_bal.get(sym, 0) or 0)
            if amt <= 0:
                continue
            try:
                p = self.fetch_latest_price(f"{sym}/USDT")
                if p and float(p) > 0:
                    nav += amt * float(p)
            except Exception:
                pass
        return round(nav, 4)

    def place_order(self, symbol: str, side: str, amount: float,
                    order_type: str = 'market', price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    trail_percent: Optional[float] = None,
                    reduce_only: bool = False) -> Dict:
        """
        Place any order type. Routes to Alpaca (primary) or CCXT (fallback).

        Order types:
            market: immediate fill
            limit: fill at price or better
            stop: market order when stop_price hit
            stop_limit: limit order when stop_price hit
            trailing_stop: trails by trail_percent %
        """
        if not self.is_authenticated:
            return {'status': 'error', 'message': 'Not authenticated. Set API key/secret.'}

        # ── Delta Exchange order routing ──
        if self.delta and self.delta.available:
            pre_trade_price = self.fetch_latest_price(symbol)
            effective_price = price if order_type == 'limit' else None
            result = self.delta.place_order(
                symbol=symbol, side=side, qty=amount,
                order_type=order_type, limit_price=effective_price,
                reduce_only=reduce_only,
            )
            if result.get('status') == 'success':
                fill_price = result.get('filled_avg_price')
                slippage_pct = 0.0
                if pre_trade_price and fill_price and pre_trade_price > 0:
                    slippage_pct = abs(float(fill_price) - pre_trade_price) / pre_trade_price * 100
                return {
                    'status': 'success',
                    'order_id': result.get('order_id'),
                    'symbol': result.get('symbol'),
                    'side': result.get('side'),
                    'type': result.get('type'),
                    'amount': amount,
                    'filled': result.get('filled_qty'),
                    'price': fill_price,
                    'cost': float(fill_price or 0) * amount if fill_price else None,
                    'fee': {'cost': 0, 'currency': 'USD'},
                    'testnet': self.testnet,
                    'slippage_pct': slippage_pct,
                    'exchange': 'delta',
                }
            return result

        # ── Bybit order routing (primary — supports SHORT) ──
        if self.bybit and self.bybit.available:
            pre_trade_price = self.fetch_latest_price(symbol)
            # CRITICAL: Never pass limit_price to market orders on Bybit
            # Bybit treats market+price as IOC limit → cancelled if no fill
            effective_price = price if order_type == 'limit' else None
            result = self.bybit.place_order(
                symbol=symbol,
                side=side,
                qty=amount,
                order_type=order_type,
                limit_price=effective_price,
                stop_price=stop_price,
                trail_percent=trail_percent,
                reduce_only=reduce_only,
            )
            if result.get('status') == 'success':
                fill_price = result.get('filled_avg_price')
                slippage_pct = 0.0
                if pre_trade_price and fill_price and pre_trade_price > 0:
                    slippage_pct = abs(float(fill_price) - pre_trade_price) / pre_trade_price * 100
                return {
                    'status': 'success',
                    'order_id': result.get('order_id'),
                    'symbol': result.get('symbol'),
                    'side': result.get('side'),
                    'type': result.get('type'),
                    'amount': amount,
                    'filled': result.get('filled_qty'),
                    'price': fill_price,
                    'cost': float(fill_price or 0) * amount if fill_price else None,
                    'fee': {'cost': 0, 'currency': 'USDT'},
                    'testnet': self.testnet,
                    'slippage_pct': slippage_pct,
                    'exchange': 'bybit',
                }
            return result

        # ── Alpaca order routing ──
        if self.alpaca and self.alpaca.available:
            # Convert symbol: BTC/USDT → BTC/USD (Alpaca uses USD pairs)
            alpaca_symbol = symbol.replace('/USDT', '/USD').replace('USDT', 'USD')
            pre_trade_price = self.fetch_latest_price(symbol)

            result = self.alpaca.place_order(
                symbol=alpaca_symbol,
                side=side,
                qty=amount,
                order_type=order_type,
                limit_price=price,
                stop_price=stop_price,
                trail_percent=trail_percent,
            )

            if result.get('status') == 'success':
                fill_price = result.get('filled_avg_price')
                # Track slippage
                slippage_pct = 0.0
                if pre_trade_price and fill_price and pre_trade_price > 0:
                    slippage_pct = abs(float(fill_price) - pre_trade_price) / pre_trade_price * 100
                    self._slippage_history.append({
                        'symbol': symbol, 'side': side,
                        'expected': pre_trade_price, 'actual': float(fill_price),
                        'slippage_pct': slippage_pct, 'time': time.time()
                    })

                return {
                    'status': 'success',
                    'order_id': result.get('order_id'),
                    'symbol': result.get('symbol'),
                    'side': result.get('side'),
                    'type': result.get('type'),
                    'amount': amount,
                    'filled': result.get('filled_qty'),
                    'price': fill_price,
                    'cost': float(fill_price or 0) * amount if fill_price else None,
                    'fee': {'cost': 0, 'currency': 'USD'},  # Alpaca: no commission on crypto
                    'testnet': self.testnet,
                    'slippage_pct': slippage_pct,
                    'exchange': 'alpaca',
                }
            return result

        # ── CCXT order routing (fallback) ──
        try:
            self._throttle()
            pre_trade_price = None
            if order_type == 'market':
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    pre_trade_price = float(ticker.get('last', 0))
                except Exception:
                    pass

            if order_type == 'limit' and price is not None:
                order = self.exchange.create_order(symbol, order_type, side, amount, price)
            else:
                order = self.exchange.create_order(symbol, 'market', side, amount)

            fill_price = order.get('price') or order.get('average')
            slippage_pct = 0.0
            if pre_trade_price and fill_price and pre_trade_price > 0:
                slippage_pct = abs(float(fill_price) - pre_trade_price) / pre_trade_price * 100
                self._slippage_history.append({
                    'symbol': symbol, 'side': side,
                    'expected': pre_trade_price, 'actual': float(fill_price),
                    'slippage_pct': slippage_pct, 'time': time.time()
                })

            return {
                'status': 'success',
                'order_id': order.get('id'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'type': order.get('type'),
                'amount': order.get('amount'),
                'filled': order.get('filled'),
                'price': fill_price,
                'cost': order.get('cost'),
                'fee': order.get('fee'),
                'testnet': self.testnet,
                'slippage_pct': slippage_pct,
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'testnet': self.testnet,
            }

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Fetch open orders."""
        if not self.is_authenticated:
            return []
        try:
            self._throttle()
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            print(f"Warning: fetch_open_orders failed: {e}")
            return []

    def fetch_order(self, order_id: str, symbol: str) -> Dict:
        """Fetch a single order by ID to check fill status."""
        if not self.is_authenticated:
            return {'status': 'error', 'message': 'Not authenticated'}
        try:
            self._throttle()
            order = self.exchange.fetch_order(order_id, symbol)
            return {
                'status': order.get('status', 'unknown'),   # 'open', 'closed', 'canceled'
                'order_id': order.get('id'),
                'filled': float(order.get('filled', 0)),
                'remaining': float(order.get('remaining', 0)),
                'price': order.get('price'),
                'average': order.get('average'),
                'cost': order.get('cost'),
                'fee': order.get('fee'),
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an open order."""
        if not self.is_authenticated:
            return {'status': 'error', 'message': 'Not authenticated'}
        try:
            self._throttle()
            result = self.exchange.cancel_order(order_id, symbol)
            return {'status': 'cancelled', 'order_id': order_id}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    @staticmethod
    def extract_ohlcv(raw: List[List[float]]) -> Dict[str, List[float]]:
        """
        Extract individual OHLCV arrays from raw CCXT data.
        Returns dict with 'timestamps', 'opens', 'highs', 'lows', 'closes', 'volumes'.
        """
        if not raw:
            return {
                'timestamps': [], 'opens': [], 'highs': [],
                'lows': [], 'closes': [], 'volumes': [],
            }
        return {
            'timestamps': [row[0] for row in raw],
            'opens': [row[1] for row in raw],
            'highs': [row[2] for row in raw],
            'lows': [row[3] for row in raw],
            'closes': [row[4] for row in raw],
            'volumes': [row[5] for row in raw],
        }

    @staticmethod
    def generate_synthetic(initial: float = 50000.0, steps: int = 200,
                              volatility: float = 0.02) -> Dict[str, List[float]]:
        """Generate synthetic OHLCV data for testing when exchange is unavailable."""
        import random
        random.seed(42)  # reproducible

        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        opens: List[float] = []
        volumes: List[float] = []

        price = initial
        for i in range(steps):
            open_p = price
            # Random walk with drift
            change = random.gauss(0.0002, volatility)
            close_p = open_p * (1 + change)
            high_p = max(open_p, close_p) * (1 + abs(random.gauss(0, volatility * 0.5)))
            low_p = min(open_p, close_p) * (1 - abs(random.gauss(0, volatility * 0.5)))

            opens.append(max(open_p, 1.0))
            highs.append(max(high_p, 1.0))
            lows.append(max(low_p, 1.0))
            closes.append(max(close_p, 1.0))
            volumes.append(random.uniform(100, 10000))
            price = close_p

        return {
            'timestamps': list(range(steps)),
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'volumes': volumes,
        }
