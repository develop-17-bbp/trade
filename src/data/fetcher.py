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
import threading
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


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
                 api_secret: Optional[str] = None,
                 max_requests_per_second: int = 10):
        self.testnet = testnet
        self.exchange_name = exchange_name
        self._authenticated = False

        # Exchange API rate limiter (application-level)
        self._rate_lock = threading.Lock()
        self._rate_timestamps: List[float] = []
        self._max_rps = max_requests_per_second

        # Slippage tracker
        self._slippage_history: List[Dict] = []

        try:
            import ccxt

            # Build exchange config
            exchange_config: Dict = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,
                },
            }

            # Add API credentials if provided
            key = api_key or os.environ.get('BINANCE_TESTNET_KEY' if testnet else 'BINANCE_API_KEY')
            secret = api_secret or os.environ.get('BINANCE_TESTNET_SECRET' if testnet else 'BINANCE_API_SECRET')

            if key and secret:
                exchange_config['apiKey'] = key
                exchange_config['secret'] = secret
                self._authenticated = True

            self.exchange = getattr(ccxt, exchange_name)(exchange_config)
            
            # Force CCXT to calculate server time offset
            try:
                self.exchange.load_time_difference()
            except Exception as e:
                print(f"  [WARN] Failed to sync exchange time: {e}")

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
        """Fetch the current price of a symbol."""
        ticker = self.get_ticker(symbol)
        return float(ticker['last']) if ticker.get('last') else None

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
        """Fetch account balances. Requires API key authentication."""
        if not self.is_authenticated:
            return {'error': 'Not authenticated. Set API key/secret.', 'read_only': True}
        try:
            self._throttle()
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
            error_msg = str(e)
            # Check if this is an auth error
            if '2008' in error_msg or 'Invalid Api-Key' in error_msg or 'Unauthorized' in error_msg:
                return {
                    'error': f'Invalid or expired API credentials. {error_msg}',
                    'invalid_credentials': True,
                    'read_only': True,
                }
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
            self._throttle()
            # Capture pre-trade price for slippage calculation
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

            # Track slippage for market orders
            slippage_pct = 0.0
            if pre_trade_price and fill_price and pre_trade_price > 0:
                slippage_pct = abs(float(fill_price) - pre_trade_price) / pre_trade_price * 100
                self._slippage_history.append({
                    'symbol': symbol, 'side': side,
                    'expected': pre_trade_price, 'actual': float(fill_price),
                    'slippage_pct': slippage_pct, 'time': time.time()
                })
                if slippage_pct > 0.5:
                    logger.warning(f"[SLIPPAGE] High slippage on {symbol}: {slippage_pct:.3f}% "
                                   f"(expected={pre_trade_price}, filled={fill_price})")

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
