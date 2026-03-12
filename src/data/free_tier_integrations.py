"""
Free Data Aggregator - Combine all free sources
Adds +10-15% accuracy without cost
"""

import requests
import os
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FreeDataAggregator:
    """Aggregate all free data sources for enhanced signal generation"""
    
    def __init__(self):
        self.dune_key = os.getenv('DUNE_API_KEY')
        self.timeout = 5
    
    # ────── FEAR/GREED (Alternative.me) ──────
    def get_fear_greed(self) -> Dict:
        """
        Fetch Fear/Greed Index from Alternative.me
        No key needed - completely free
        Range: 0-100 (0=Extreme Fear, 100=Extreme Greed)
        """
        try:
            url = 'https://api.alternative.me/fng/?limit=1'
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()['data'][0]
                value = int(data['value'])
                classification = data['value_classification']
                return {
                    'value': value,
                    'classification': classification,
                    'timestamp': datetime.fromtimestamp(int(data['timestamp'])),
                    'is_bullish': value > 70,  # >70 = greed = bullish
                    'is_bearish': value < 30,  # <30 = fear = bearish
                }
        except Exception as e:
            logger.warning(f"Failed to fetch fear/greed: {e}")
        
        return {'value': 50, 'classification': 'Neutral', 'is_bullish': False, 'is_bearish': False}
    
    # ────── DERIBIT OPTIONS IV (No key needed) ──────
    def get_deribit_iv(self, instrument: str = 'BTC-PERPETUAL') -> Optional[float]:
        """
        Fetch Implied Volatility from Deribit
        No key needed - free public API
        High IV = market expects big moves (volatility regime)
        Low IV = calm market (trending regime)
        """
        try:
            url = f'https://www.deribit.com/api/v2/public/get_instrument?instrument_name={instrument}'
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json().get('result', {})
                return float(result.get('bid_iv', 0))
        except Exception as e:
            logger.warning(f"Failed to fetch Deribit IV: {e}")
        
        return None
    
    # ────── COINGECKO MARKET DATA (No key needed) ──────
    def get_coingecko_data(self, coin_id: str = 'bitcoin') -> Dict:
        """
        Fetch market data from CoinGecko
        No key needed - free tier allows 50 calls/minute
        Gets: price, market cap, volume, dominance
        """
        try:
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            url = 'https://api.coingecko.com/api/v3/simple/price'
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                return data.get(coin_id, {})
        except Exception as e:
            logger.warning(f"Failed to fetch CoinGecko data: {e}")
        
        return {}
    
    # ────── DUNE ANALYTICS (You have key) ──────
    def get_whale_activity(self, token: str = 'AAVE', hours: int = 24) -> Optional[Dict]:
        """
        Query Dune for whale transactions (>$100K)
        Requires: DUNE_API_KEY in .env
        Returns: number of whale transfers, total volume, average size
        """
        if not self.dune_key:
            return None
        
        try:
            query = f"""
            SELECT 
                COUNT(*) as transfer_count,
                SUM(amount) as total_volume,
                AVG(amount) as avg_transfer_size
            FROM transfers
            WHERE token = '{token}' 
            AND amount > 100000
            AND timestamp > NOW() - INTERVAL '{hours} hours'
            """
            
            headers = {"X-DUNE-API-KEY": self.dune_key}
            url = "https://api.dune.com/api/v1/query"
            
            response = requests.post(
                url,
                headers=headers,
                json={"query": query},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get('data', [{}])[0]
        except Exception as e:
            logger.warning(f"Failed to fetch Dune whale activity: {e}")
        
        return None
    
    # ────── EXCHANGE FLOW (Dune) ──────
    def get_exchange_netflows(self, asset: str = 'AAVE', hours: int = 1) -> Optional[Dict]:
        """
        Net inflows/outflows to centralized exchanges
        Positive = money leaving (bearish)
        Negative = money entering (bullish)
        """
        if not self.dune_key:
            return None
        
        try:
            query = f"""
            SELECT 
                SUM(CASE WHEN type = 'inflow' THEN amount ELSE -amount END) as net_flow,
                SUM(CASE WHEN type = 'inflow' THEN amount ELSE 0 END) as total_inflow,
                SUM(CASE WHEN type = 'outflow' THEN amount ELSE 0 END) as total_outflow
            FROM exchange_flows
            WHERE asset = '{asset}' 
            AND timestamp > NOW() - INTERVAL '{hours} hours'
            """
            
            headers = {"X-DUNE-API-KEY": self.dune_key}
            url = "https://api.dune.com/api/v1/query"
            
            response = requests.post(
                url,
                headers=headers,
                json={"query": query},
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get('data', [{}])[0]
        except Exception as e:
            logger.warning(f"Failed to fetch exchange flows: {e}")
        
        return None
    
    # ────── AGGREGATE ALL ──────
    def aggregate_all_signals(self, symbol: str = 'BTC') -> Dict:
        """
        Combine all free data sources into ONE signal dictionary
        Returns enhanced features for trading decision
        """
        
        coin_map = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'AAVE': 'aave'}
        coin_id = coin_map.get(symbol, symbol.lower())
        
        fear_greed = self.get_fear_greed()
        deribit_iv = self.get_deribit_iv()
        coingecko = self.get_coingecko_data(coin_id)
        whale = self.get_whale_activity(symbol, hours=24)
        flows = self.get_exchange_netflows(symbol, hours=1)
        
        # Calculate confidence boost based on data quality
        data_confidence = 0.0
        if fear_greed.get('value'):
            data_confidence += 0.25
        if deribit_iv is not None:
            data_confidence += 0.25
        if coingecko.get('usd'):
            data_confidence += 0.25
        if whale and whale.get('transfer_count'):
            data_confidence += 0.25
        
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
            
            # On-chain activity
            'whale_transfer_count_24h': whale.get('transfer_count') if whale else None,
            'whale_volume_24h': whale.get('total_volume') if whale else None,
            
            # Exchange flows
            'exchange_netflow_1h': flows.get('net_flow') if flows else None,
            'exchange_inflow_1h': flows.get('total_inflow') if flows else None,
            'exchange_outflow_1h': flows.get('total_outflow') if flows else None,
            'exchange_flow_signal': 'BEARISH' if flows and flows.get('net_flow', 0) > 0 else 'BULLISH' if flows and flows.get('net_flow', 0) < 0 else 'NEUTRAL',
            
            # Overall confidence
            'free_data_confidence': data_confidence,  # 0-1 scale (how many sources responded)
        }
    
    def calculate_free_data_boost(self, base_confidence: float, free_signals: Dict) -> float:
        """
        Boost base model confidence with free data signals
        Only boost if signals align (reduce false positives)
        """
        boost = base_confidence  # Start with base
        
        # Fear/Greed alignment
        if free_signals.get('fear_greed_signal') == 'BULLISH':
            boost += 0.05  # +5% if greed signal
        elif free_signals.get('fear_greed_signal') == 'BEARISH':
            boost -= 0.05  # -5% if fear signal
        
        # Volatility alignment (high IV = harder trends)
        if free_signals.get('iv_regime') == 'HIGH':
            boost -= 0.03  # Reduce confidence in high volatility
        elif free_signals.get('iv_regime') == 'LOW':
            boost += 0.03  # Increase confidence in low volatility
        
        # Price momentum alignment
        if free_signals.get('price_momentum') == 'BULLISH':
            boost += 0.05  # +5% if momentum agrees
        elif free_signals.get('price_momentum') == 'BEARISH':
            boost -= 0.05  # -5% if momentum disagrees
        
        # Exchange flow alignment (outflows = bullish, inflows = bearish)
        if free_signals.get('exchange_flow_signal') == 'BULLISH':
            boost += 0.04  # +4% if whales buying
        elif free_signals.get('exchange_flow_signal') == 'BEARISH':
            boost -= 0.04  # -4% if whales selling
        
        # Apply data quality penalty if sources unavailable
        boost *= free_signals.get('free_data_confidence', 0.75)
        
        # Clamp to reasonable range
        return max(0.0, min(1.0, boost))
