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
  - Order placement (testnet / live)
"""

import os
import time
from typing import List, Dict, Optional, Tuple


class PriceFetcher:
    """
    CCXT-based real-time price data fetcher.
    Supports multiple exchanges, timeframes, and testnet mode.

    Testnet mode connects to Binance's sandbox (testnet.binance.vision)
    where you trade with free test BTC/ETH against a live order book.
    """

    VALID_TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d', '1w', '1M']

    def __init__(self, exchange_name: str = "binance",
                 testnet: bool = False,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        self.testnet = testnet
        self.exchange_name = exchange_name
        self._authenticated = False

        try:
            import ccxt

            # Build exchange config
            exchange_config: Dict = {
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            }

            # Add API credentials if provided
            key = api_key or os.environ.get('BINANCE_TESTNET_KEY' if testnet else 'BINANCE_API_KEY')
            secret = api_secret or os.environ.get('BINANCE_TESTNET_SECRET' if testnet else 'BINANCE_API_SECRET')

            if key and secret:
                exchange_config['apiKey'] = key
                exchange_config['secret'] = secret
                self._authenticated = True

            self.exchange = getattr(ccxt, exchange_name)(exchange_config)

            # Enable testnet sandbox
            if testnet:
                self.exchange.set_sandbox_mode(True)
                print(f"  [TESTNET] Connected to {exchange_name.upper()} Testnet (sandbox mode)")
                if self._authenticated:
                    print(f"  [TESTNET] API key authenticated — order execution enabled")
                else:
                    print(f"  [TESTNET] No API key — read-only mode (set BINANCE_TESTNET_KEY/SECRET)")

            self._available = True
        except Exception as e:
            print(f"[FATAL] CCXT not available ({e}). Real-time data required.")
            self.exchange = None
            self._available = False

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
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            print(f"Warning: fetch_ticker failed for {symbol}: {e}")
            return {'last': 0.0, 'bid': 0.0, 'ask': 0.0}

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1d',
                     limit: int = 100) -> List[List[float]]:
        if not self._available:
            raise RuntimeError("Exchange not available")
        if not hasattr(self.exchange, 'fetch_ohlcv'):
            raise RuntimeError('Exchange does not support OHLCV')
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def fetch_derivatives_data(self, symbol: str) -> Dict[str, float]:
        """Fetch funding rates and open interest (Binance only)."""
        if not self._available or 'binance' not in self.exchange_name.lower():
            return {}
        try:
            # CCXT fetch_funding_rate for perpetuals
            fr = self.exchange.fetch_funding_rate(symbol)
            oi_data = self.exchange.fetch_open_interest(symbol)
            return {
                'funding_rate': float(fr.get('fundingRate', 0.0)),
                'open_interest': float(oi_data.get('openInterestAmount', 0.0)),
                'oi_change': 0.0 # logic to track change would go here
            }
        except Exception:
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

    def get_balance(self) -> Dict:
        """Fetch account balances. Requires API key authentication."""
        if not self.is_authenticated:
            return {'error': 'Not authenticated. Set API key/secret.'}
        try:
            balance = self.exchange.fetch_balance()
            # Extract relevant info
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
            return {'error': str(e)}

    def place_order(self, symbol: str, side: str, amount: float,
                    order_type: str = 'market', price: Optional[float] = None) -> Dict:
        """
        Place an order on the exchange (testnet or live).

        Args:
            symbol: Trading pair (e.g. 'BTC/USDT')
            side: 'buy' or 'sell'
            amount: Quantity to trade
            order_type: 'market' or 'limit'
            price: Required for limit orders

        Returns:
            Order result dict with status, id, filled amount, etc.
        """
        if not self.is_authenticated:
            return {'status': 'error', 'message': 'Not authenticated. Set API key/secret.'}

        try:
            if order_type == 'limit' and price is not None:
                order = self.exchange.create_order(symbol, order_type, side, amount, price)
            else:
                order = self.exchange.create_order(symbol, 'market', side, amount)

            return {
                'status': 'success',
                'order_id': order.get('id'),
                'symbol': order.get('symbol'),
                'side': order.get('side'),
                'type': order.get('type'),
                'amount': order.get('amount'),
                'filled': order.get('filled'),
                'price': order.get('price') or order.get('average'),
                'cost': order.get('cost'),
                'fee': order.get('fee'),
                'testnet': self.testnet,
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
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            print(f"Warning: fetch_open_orders failed: {e}")
            return []

    def cancel_order(self, order_id: str, symbol: str) -> Dict:
        """Cancel an open order."""
        if not self.is_authenticated:
            return {'status': 'error', 'message': 'Not authenticated'}
        try:
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
