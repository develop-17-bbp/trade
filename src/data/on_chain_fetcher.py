import random
from typing import Dict, Any, List
from datetime import datetime

class OnChainFetcher:
    """
    Layer 4: On-Chain Tracking Component.
    Monitors whale movements, exchange flows, and network health.
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        # Common on-chain thresholds
        self.WHALE_THRESHOLD = {
            "BTC": 100,  # 100 BTC
            "ETH": 1000, # 1000 ETH
            "AAVE": 5000 # 5000 AAVE
        }

    def fetch_whale_flows(self, asset: str) -> Dict[str, Any]:
        """
        Detects large transfers (> threshold) from/to exchanges.
        Returns: { 'large_transfers_count': int, 'net_exchange_flow': float, 'sentiment': 'BULLISH'|'BEARISH' }
        """
        # Simulation: High volatility usually correlates with high whale movement
        # In real world: Call WhaleAlert API
        count = random.randint(0, 15)
        flow = random.uniform(-500, 500) # Positive = inflow to exchange (Bearish), Negative = outflow (Bullish)
        
        sentiment = "NEUTRAL"
        if flow < -200: sentiment = "BULLISH" # Whales moving to cold storage
        elif flow > 200: sentiment = "BEARISH" # Whales moving to sell
        
        return {
            "large_transfers_count": count,
            "net_exchange_flow": round(flow, 2),
            "whale_sentiment": sentiment,
            "top_move_usd": round(random.uniform(1e6, 50e6), 2)
        }

    def fetch_network_stats(self, asset: str) -> Dict[str, Any]:
        """
        Active Addresses, Hashrate (BTC), Transaction Counts.
        """
        # Growth simulation
        active_addresses = random.randint(500000, 1000000)
        daily_growth = random.uniform(-0.05, 0.05)
        
        return {
            "active_addresses": active_addresses,
            "address_growth_pct": round(daily_growth * 100, 2),
            "network_utilization_pct": round(random.uniform(40, 95), 2)
        }

    def get_market_context(self, asset: str) -> Dict[str, Any]:
        """Aggregates all on-chain metrics for the Strategist."""
        whale_data = self.fetch_whale_flows(asset)
        network_data = self.fetch_network_stats(asset)
        
        # Combined insight
        bias = 0.0
        if whale_data['whale_sentiment'] == "BULLISH": bias += 0.1
        if whale_data['whale_sentiment'] == "BEARISH": bias -= 0.1
        if network_data['address_growth_pct'] > 2: bias += 0.05
        
        return {
            "whale_metrics": whale_data,
            "network_metrics": network_data,
            "onchain_combined_bias": round(bias, 2),
            "timestamp": datetime.now().isoformat()
        }

# For backward compatibility with the existing functional call
def fetch_metrics(symbol: str) -> Dict[str, Any]:
    asset = symbol.split('/')[0] if '/' in symbol else symbol
    fetcher = OnChainFetcher()
    return fetcher.get_market_context(asset)
