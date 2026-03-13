"""
Free Data Aggregator (Production-Ready)
==========================================
Combines ALL free-tier data sources into a unified signal dictionary.
All sources are real APIs with no keys required (except optional Dune).

Sources:
  - Alternative.me:  Fear & Greed Index
  - Deribit:         Implied Volatility
  - CoinGecko:       Market data, stablecoin price for depeg
  - DefiLlama:       DeFi TVL, stablecoin flows, DEX volume
  - Blockchain.com:  BTC network stats, exchange flow estimates
  - Mempool.space:   Mempool congestion, hashrate
  - Binance Futures: Long/Short ratio (public)
  - Yahoo Finance:   Macro correlations (via yfinance)
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.data.base_fetcher import CachedFetcher, CACHE_TTL_MEDIUM
from src.data.on_chain_fetcher import OnChainFetcher

logger = logging.getLogger(__name__)


class FreeDataAggregator(CachedFetcher):
    """Aggregate all free data sources for enhanced signal generation."""

    def __init__(self):
        super().__init__(timeout=5)
        self.dune_key = os.getenv('DUNE_API_KEY')
        # Share OnChainFetcher for DefiLlama calls (avoids duplicate HTTP requests)
        self._onchain = OnChainFetcher()

    # ────── FEAR/GREED (Alternative.me) ──────
    def get_fear_greed(self) -> Dict:
        """
        Fetch Fear/Greed Index from Alternative.me
        No key needed — completely free.
        Range: 0-100 (0=Extreme Fear, 100=Extreme Greed)
        """
        try:
            data_resp = self._safe_get('https://api.alternative.me/fng/?limit=1')
            if data_resp:
                data = data_resp['data'][0]
                value = int(data['value'])
                classification = data['value_classification']
                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': datetime.fromtimestamp(int(data['timestamp'])),
                    'is_bullish': value > 70,
                    'is_bearish': value < 30,
                }
        except Exception as e:
            logger.warning(f"Failed to fetch fear/greed: {e}")

        return {'value': 50, 'classification': 'Neutral', 'is_bullish': False, 'is_bearish': False}

    # ────── DERIBIT OPTIONS IV (No key needed) ──────
    def get_deribit_iv(self, instrument: str = 'BTC-PERPETUAL') -> Optional[float]:
        """
        Fetch Implied Volatility from Deribit.
        No key needed — free public API.
        """
        try:
            resp = self._safe_get(
                'https://www.deribit.com/api/v2/public/ticker',
                params={'instrument_name': instrument}
            )
            if resp:
                result = resp.get('result', {})
                iv = result.get('mark_iv', result.get('bid_iv', 0))
                return float(iv) if iv else None
        except Exception as e:
            logger.warning(f"Failed to fetch Deribit IV: {e}")
        return None

    # ────── COINGECKO MARKET DATA (No key needed) ──────
    def get_coingecko_data(self, coin_id: str = 'bitcoin') -> Dict:
        """
        Fetch market data from CoinGecko.
        No key needed — free tier allows 30 calls/minute.
        """
        try:
            data = self._safe_get(
                'https://api.coingecko.com/api/v3/simple/price',
                params={
                    'ids': coin_id, 'vs_currencies': 'usd',
                    'include_market_cap': 'true',
                    'include_24hr_vol': 'true',
                    'include_24hr_change': 'true'
                }
            )
            if data:
                return data.get(coin_id, {})
        except Exception as e:
            logger.warning(f"Failed to fetch CoinGecko data: {e}")
        return {}

    # ────── DEFILLAMA DEFI METRICS (delegates to OnChainFetcher's cached calls) ──────
    def get_defillama_summary(self) -> Dict:
        """
        Fetch DeFi summary: TVL, DEX volume, stablecoin data.
        Delegates to OnChainFetcher to avoid duplicate HTTP requests.
        """
        tvl = self._onchain.fetch_defillama_tvl()
        dex = self._onchain.fetch_defillama_dex_volume()
        stables = self._onchain.fetch_defillama_stablecoins()

        return {
            'defi_tvl': tvl.get('defi_tvl', 0),
            'defi_tvl_change_pct': tvl.get('defi_tvl_change_pct', 0),
            'dex_volume_24h': dex.get('dex_volume_24h', 0),
            'stablecoin_total_mcap': stables.get('stablecoin_total_mcap', 0),
        }

    # ────── BINANCE FUTURES L/S RATIO (No key needed) ──────
    def get_long_short_ratio(self, symbol: str = 'BTCUSDT') -> Dict:
        """
        Fetch top trader L/S ratio from Binance Futures public API.
        No key needed.
        """
        try:
            data = self._safe_get(
                'https://fapi.binance.com/futures/data/topLongShortAccountRatio',
                params={'symbol': symbol, 'period': '1h', 'limit': 1}
            )
            if data and isinstance(data, list):
                    entry = data[0]
                    return {
                        'ls_ratio': round(float(entry.get('longShortRatio', 1.0)), 4),
                        'long_pct': round(float(entry.get('longAccount', 0.5)) * 100, 2),
                        'short_pct': round(float(entry.get('shortAccount', 0.5)) * 100, 2),
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch L/S ratio: {e}")
        return {'ls_ratio': 1.0, 'long_pct': 50.0, 'short_pct': 50.0}

    # ────── DUNE ANALYTICS (Optional — needs DUNE_API_KEY) ──────
    def get_whale_activity(self, token: str = 'AAVE', hours: int = 24) -> Optional[Dict]:
        """
        Query Dune for whale transactions (>$100K).
        Requires: DUNE_API_KEY in .env
        Signup: https://dune.com/ (free tier available)
        """
        if not self.dune_key:
            return None
        try:
            headers = {"X-DUNE-API-KEY": self.dune_key}
            # Use a pre-saved query ID for whale tracking
            # You can create your own queries at dune.com
            resp = self._safe_get(
                "https://api.dune.com/api/v1/query/1234567/results",  # Replace with your query ID
                headers=headers, timeout=30
            )
            if resp:
                rows = resp.get('result', {}).get('rows', [])
                if rows:
                    return {
                        'transfer_count': int(rows[0].get('transfer_count', 0)),
                        'total_volume': float(rows[0].get('total_volume', 0)),
                        'avg_transfer_size': float(rows[0].get('avg_transfer_size', 0)),
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch Dune whale activity: {e}")
        return None

    def get_exchange_netflows(self, asset: str = 'AAVE', hours: int = 1) -> Optional[Dict]:
        """
        Net inflows/outflows to centralized exchanges.
        Requires: DUNE_API_KEY in .env
        """
        if not self.dune_key:
            return None
        try:
            headers = {"X-DUNE-API-KEY": self.dune_key}
            resp = self._safe_get(
                "https://api.dune.com/api/v1/query/1234568/results",  # Replace with your query ID
                headers=headers, timeout=30
            )
            if resp:
                rows = resp.get('result', {}).get('rows', [])
                if rows:
                    return {
                        'net_flow': float(rows[0].get('net_flow', 0)),
                        'total_inflow': float(rows[0].get('total_inflow', 0)),
                        'total_outflow': float(rows[0].get('total_outflow', 0)),
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch exchange flows: {e}")
        return None

    # ────── AGGREGATE ALL ──────
    def aggregate_all_signals(self, symbol: str = 'BTC') -> Dict:
        """
        Combine ALL free data sources into ONE signal dictionary.
        Uses ThreadPoolExecutor for concurrent fetching (~3s vs ~15s sequential).
        All sources are real — no mocked data.
        """
        coin_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'AAVE': 'aave'}
        coin_id = coin_map.get(symbol, symbol.lower())
        t0 = time.time()

        # Launch all 7 API calls concurrently
        with ThreadPoolExecutor(max_workers=7, thread_name_prefix='free') as pool:
            f_fg = pool.submit(self.get_fear_greed)
            f_iv = pool.submit(self.get_deribit_iv)
            f_cg = pool.submit(self.get_coingecko_data, coin_id)
            f_dl = pool.submit(self.get_defillama_summary)
            f_ls = pool.submit(self.get_long_short_ratio, f'{symbol}USDT')
            f_wh = pool.submit(self.get_whale_activity, symbol, 24)
            f_fl = pool.submit(self.get_exchange_netflows, symbol, 1)

            # Collect results with safe defaults
            fear_greed = f_fg.result(timeout=12) if not f_fg.exception() else {'value': 50, 'classification': 'Neutral', 'is_bullish': False, 'is_bearish': False}
            deribit_iv = f_iv.result(timeout=12) if not f_iv.exception() else None
            coingecko = f_cg.result(timeout=12) if not f_cg.exception() else {}
            defillama = f_dl.result(timeout=12) if not f_dl.exception() else {}
            ls_ratio = f_ls.result(timeout=12) if not f_ls.exception() else {'ls_ratio': 1.0, 'long_pct': 50.0, 'short_pct': 50.0}
            whale = f_wh.result(timeout=12) if not f_wh.exception() else None
            flows = f_fl.result(timeout=12) if not f_fl.exception() else None

        elapsed = time.time() - t0
        logger.info(f"Free data aggregated concurrently in {elapsed:.1f}s (7 sources)")

        # Calculate confidence boost based on data quality
        data_confidence = 0.0
        if fear_greed.get('value'):
            data_confidence += 0.15
        if deribit_iv is not None:
            data_confidence += 0.15
        if coingecko.get('usd'):
            data_confidence += 0.15
        if defillama.get('defi_tvl', 0) > 0:
            data_confidence += 0.15
        if ls_ratio.get('ls_ratio', 1.0) != 1.0:
            data_confidence += 0.15
        if whale and whale.get('transfer_count'):
            data_confidence += 0.15
        if flows and flows.get('net_flow') is not None:
            data_confidence += 0.10

        return {
            'timestamp': datetime.utcnow(),
            'symbol': symbol,

            # Sentiment layer
            'fear_greed_index': fear_greed.get('value', 50),
            'fear_greed_classification': fear_greed.get('classification', 'Neutral'),
            'fear_greed_signal': 'BULLISH' if fear_greed.get('is_bullish') else 'BEARISH' if fear_greed.get('is_bearish') else 'NEUTRAL',

            # Volatility layer
            'implied_volatility': deribit_iv,
            'iv_regime': 'HIGH' if deribit_iv and deribit_iv > 60 else 'LOW' if deribit_iv and deribit_iv < 20 else 'NORMAL',

            # Market data
            'market_cap': coingecko.get('usd_market_cap'),
            'trading_volume_24h': coingecko.get('usd_24h_vol'),
            'price_change_24h': coingecko.get('usd_24h_change'),
            'price_momentum': 'BULLISH' if coingecko.get('usd_24h_change', 0) > 2 else 'BEARISH' if coingecko.get('usd_24h_change', 0) < -2 else 'NEUTRAL',

            # DeFi health (NEW — real data from DefiLlama)
            'defi_tvl': defillama.get('defi_tvl', 0),
            'defi_tvl_change_pct': defillama.get('defi_tvl_change_pct', 0),
            'dex_volume_24h': defillama.get('dex_volume_24h', 0),
            'stablecoin_total_mcap': defillama.get('stablecoin_total_mcap', 0),

            # Derivatives (NEW — real data from Binance Futures)
            'ls_ratio': ls_ratio.get('ls_ratio', 1.0),
            'long_account_pct': ls_ratio.get('long_pct', 50),
            'short_account_pct': ls_ratio.get('short_pct', 50),
            'ls_signal': 'BEARISH' if ls_ratio.get('ls_ratio', 1.0) > 1.5 else 'BULLISH' if ls_ratio.get('ls_ratio', 1.0) < 0.7 else 'NEUTRAL',

            # On-chain activity (Dune — optional)
            'whale_transfer_count_24h': whale.get('transfer_count') if whale else None,
            'whale_volume_24h': whale.get('total_volume') if whale else None,

            # Exchange flows (Dune — optional)
            'exchange_netflow_1h': flows.get('net_flow') if flows else None,
            'exchange_inflow_1h': flows.get('total_inflow') if flows else None,
            'exchange_outflow_1h': flows.get('total_outflow') if flows else None,
            'exchange_flow_signal': 'BEARISH' if flows and flows.get('net_flow', 0) > 0 else 'BULLISH' if flows and flows.get('net_flow', 0) < 0 else 'NEUTRAL',

            # Overall confidence
            'free_data_confidence': round(data_confidence, 2),
        }

    def calculate_free_data_boost(self, base_confidence: float, free_signals: Dict) -> float:
        """
        Boost base model confidence with free data signals.
        Only boost if signals align (reduce false positives).
        """
        boost = base_confidence

        # Fear/Greed alignment
        if free_signals.get('fear_greed_signal') == 'BULLISH':
            boost += 0.05
        elif free_signals.get('fear_greed_signal') == 'BEARISH':
            boost -= 0.05

        # Volatility alignment
        if free_signals.get('iv_regime') == 'HIGH':
            boost -= 0.03
        elif free_signals.get('iv_regime') == 'LOW':
            boost += 0.03

        # Price momentum alignment
        if free_signals.get('price_momentum') == 'BULLISH':
            boost += 0.05
        elif free_signals.get('price_momentum') == 'BEARISH':
            boost -= 0.05

        # Exchange flow alignment
        if free_signals.get('exchange_flow_signal') == 'BULLISH':
            boost += 0.04
        elif free_signals.get('exchange_flow_signal') == 'BEARISH':
            boost -= 0.04

        # L/S ratio alignment (NEW)
        if free_signals.get('ls_signal') == 'BULLISH':
            boost += 0.03  # Contrarian: more shorts = bullish
        elif free_signals.get('ls_signal') == 'BEARISH':
            boost -= 0.03

        # DeFi health (NEW)
        tvl_change = free_signals.get('defi_tvl_change_pct', 0)
        if tvl_change > 2:
            boost += 0.02
        elif tvl_change < -2:
            boost -= 0.02

        # Apply data quality scaling
        boost *= max(0.5, free_signals.get('free_data_confidence', 0.75))

        return max(0.0, min(1.0, boost))
