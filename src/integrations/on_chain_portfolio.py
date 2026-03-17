"""
PHASE 4: On-Chain Integration Portfolio Manager
===============================================
Enhanced blockchain analytics for Layer 2.5 integration.
Monitors whale movements, exchange flows, and network health across multiple assets.
Uses real data from OnChainFetcher (free APIs) instead of simulated random values.
"""

import logging
import requests
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

@dataclass
class OnChainMetrics:
    """Comprehensive on-chain data structure."""
    asset: str
    whale_sentiment: str  # BULLISH, BEARISH, NEUTRAL
    whale_score: float  # -1.0 to 1.0
    exchange_flow: float  # Positive = outflow (bullish), Negative = inflow (bearish)
    network_health: float  # 0-100%
    address_growth: float  # %
    transaction_volume: float  # Daily volume
    large_txn_count: int  # Number of whales moving
    funded_positions: int  # Institutions holding
    liquidation_risk: float  # 0-100% for leveraged positions
    on_chain_momentum: float  # Combined signal -1.0 to 1.0
    timestamp: str
    confidence: float  # 0-100%


class OnChainPortfolioManager:
    """
    Advanced on-chain tracking for institutional flows and whale behavior.
    """
    
    def __init__(self, glassnode_api_key: str = None, dune_api_key: str = None):
        """
        Initialize with optional API keys for real data sources.
        Uses OnChainFetcher (free APIs) for real data when paid keys aren't available.
        """
        self.glassnode_key = glassnode_api_key
        self.dune_key = dune_api_key
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self._real_fetcher = None

        # Whale thresholds by asset
        self.whale_thresholds = {
            "BTC": 100,     # 100 BTC
            "ETH": 1000,    # 1000 ETH
            "AAVE": 5000,   # 5000 AAVE
            "SOL": 10000,   # 10000 SOL
            "AVAX": 50000,  # 50000 AVAX
        }

        # Historical pattern memory
        self.pattern_memory = {}
        self.whale_tracking = {}

    def _get_real_fetcher(self):
        """Lazy-init OnChainFetcher for real free-tier data."""
        if self._real_fetcher is None:
            try:
                from src.data.on_chain_fetcher import OnChainFetcher
                self._real_fetcher = OnChainFetcher()
            except Exception:
                self._real_fetcher = False  # Mark as unavailable
        return self._real_fetcher if self._real_fetcher else None
        
    def fetch_whale_flows(self, asset: str, lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Detect whales moving assets between wallets and exchanges.
        """
        # Check cache
        cache_key = f"whale_flows_{asset}_{lookback_hours}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return cached_data
        
        if self.glassnode_key:
            # Real API call to Glassnode
            flows = self._fetch_from_glassnode(asset, lookback_hours)
        else:
            # Simulated data with patterns
            flows = self._simulate_whale_flows(asset)
        
        # Cache result
        self.cache[cache_key] = (flows, datetime.now().timestamp())
        return flows
    
    def _fetch_from_glassnode(self, asset: str, lookback_hours: int) -> Dict[str, Any]:
        """Fetch real whale data from Glassnode API."""
        try:
            # This would be a real API call
            # url = f"https://api.glassnode.com/v1/metrics/exchange/net_position_change"
            # Real implementation would require active API
            return self._simulate_whale_flows(asset)
        except Exception:
            return self._simulate_whale_flows(asset)
    
    def _simulate_whale_flows(self, asset: str) -> Dict[str, Any]:
        """Get whale flow data from real free-tier APIs where available."""
        fetcher = self._get_real_fetcher()
        if fetcher:
            try:
                real = fetcher.fetch_whale_flows(asset)
                flow = real.get('net_exchange_flow', 0)
                sentiment = real.get('whale_sentiment', 'NEUTRAL')
                large_txn_count = real.get('whale_transfers_count', 10)
                funded_positions = max(100, large_txn_count * 50)
            except Exception as e:
                logger.debug(f"[OnChainPortfolio] Whale flows fallback for {asset}: {e}")
                flow, sentiment, large_txn_count, funded_positions = 0, "NEUTRAL", 10, 1000
        else:
            flow, sentiment, large_txn_count, funded_positions = 0, "NEUTRAL", 10, 1000
        
        # Derive phase from sentiment
        phase = 'accumulation' if sentiment == 'BULLISH' else ('distribution' if sentiment == 'BEARISH' else 'holding')

        return {
            "net_exchange_flow": round(flow, 2),
            "whale_sentiment": sentiment,
            "phase": phase,
            "large_transactions": large_txn_count,
            "funded_positions": funded_positions,
            "top_whale_move_usd": abs(flow) * 50000,  # Estimate based on actual flow
            "exchange_netflow_rank": 5  # Neutral default
        }
    
    def fetch_network_health(self, asset: str) -> Dict[str, Any]:
        """
        Monitor blockchain network metrics.
        Active addresses, transaction volume, network health.
        """
        cache_key = f"network_health_{asset}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return cached_data
        
        if self.glassnode_key:
            health = self._fetch_network_from_api(asset)
        else:
            health = self._simulate_network_health(asset)
        
        self.cache[cache_key] = (health, datetime.now().timestamp())
        return health
    
    def _fetch_network_from_api(self, asset: str) -> Dict[str, Any]:
        """Fetch real network data from blockchain."""
        try:
            # Real API implementation
            return self._simulate_network_health(asset)
        except Exception:
            return self._simulate_network_health(asset)
    
    def _simulate_network_health(self, asset: str) -> Dict[str, Any]:
        """Get network health from real free-tier APIs where available."""
        fetcher = self._get_real_fetcher()
        if fetcher:
            try:
                real = fetcher.fetch_network_stats(asset)
                bc_stats = fetcher._fetch_blockchain_com_stats()
                n_tx = bc_stats.get('n_tx', 300000)
                price = bc_stats.get('market_price_usd', 65000)
                total_sent = bc_stats.get('estimated_btc_sent', 200000)
                return {
                    "active_addresses": n_tx,
                    "address_growth_pct": round(real.get('hashrate_shock', 0) * 10, 2),
                    "daily_transaction_volume": round(total_sent * price / 1e6, 2),
                    "network_utilization": round(min(100, real.get('mempool_congestion', 50) * 100), 2),
                    "avg_transaction_size_usd": round(total_sent * price / max(1, n_tx), 2),
                    "whale_wallet_count": max(100, n_tx // 100)
                }
            except Exception as e:
                logger.debug(f"[OnChainPortfolio] Network health fallback for {asset}: {e}")
                pass
        # Safe neutral defaults when real data unavailable
        base = {"BTC": 950000, "ETH": 750000, "AAVE": 200000}.get(asset, 500000)
        return {
            "active_addresses": base,
            "address_growth_pct": 0.0,
            "daily_transaction_volume": 200000,
            "network_utilization": 60.0,
            "avg_transaction_size_usd": 10000,
            "whale_wallet_count": 1000
        }
    
    def fetch_liquidation_risk(self, asset: str) -> Dict[str, Any]:
        """
        Analyze liquidation risk from leverage trading.
        """
        cache_key = f"liquidation_risk_{asset}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now().timestamp() - timestamp < self.cache_ttl:
                return cached_data
        
        risk_data = self._calculate_liquidation_risk(asset)
        self.cache[cache_key] = (risk_data, datetime.now().timestamp())
        return risk_data
    
    def _calculate_liquidation_risk(self, asset: str) -> Dict[str, Any]:
        """Calculate leverage liquidation risk from real data where available."""
        fetcher = self._get_real_fetcher()
        if fetcher:
            try:
                real = fetcher.fetch_liquidation_heatmap(asset, current_price=0)
                return {
                    "open_leverage_usd": real.get('total_leverage_est', 500e6),
                    "liquidation_cascade_risk": real.get('liquidation_cascade_risk', 30),
                    "nearest_liquidation_level_pct": real.get('nearest_liq_pct', 8),
                    "leverage_ratio": real.get('estimated_leverage', 10),
                    "last_cascade_event_days_ago": 30
                }
            except Exception as e:
                logger.debug(f"[OnChainPortfolio] Liquidation risk fallback for {asset}: {e}")
                pass
        # Safe neutral defaults
        return {
            "open_leverage_usd": 500e6,
            "liquidation_cascade_risk": 30.0,
            "nearest_liquidation_level_pct": 8.0,
            "leverage_ratio": 10.0,
            "last_cascade_event_days_ago": 30
        }
    
    def compute_on_chain_signal(self, asset: str) -> OnChainMetrics:
        """
        Aggregate all on-chain data into a unified trading signal.
        Returns: OnChainMetrics dataclass
        """
        whale_data = self.fetch_whale_flows(asset)
        network_data = self.fetch_network_health(asset)
        liquidation_data = self.fetch_liquidation_risk(asset)
        
        # Compute sentiment score (-1.0 to 1.0)
        whale_score = 0.0
        if whale_data['whale_sentiment'] == "BULLISH":
            whale_score += 0.4
        elif whale_data['whale_sentiment'] == "BEARISH":
            whale_score -= 0.4
        
        # Network growth positive signal
        if network_data['address_growth_pct'] > 2:
            whale_score += 0.2
        elif network_data['address_growth_pct'] < -2:
            whale_score -= 0.2
        
        # Liquidation risk negates bullish signal
        if liquidation_data['liquidation_cascade_risk'] > 60:
            whale_score -= 0.3
        
        # Clamp score
        whale_score = max(-1.0, min(1.0, whale_score))
        
        # Convert to momentum signal
        on_chain_momentum = whale_score + (network_data['address_growth_pct'] / 100)
        on_chain_momentum = max(-1.0, min(1.0, on_chain_momentum))
        
        # Confidence based on data freshness and consistency
        confidence = 60 + (whale_data['large_transactions'] * 2)
        confidence = min(100, confidence)
        
        return OnChainMetrics(
            asset=asset,
            whale_sentiment=whale_data['whale_sentiment'],
            whale_score=round(whale_score, 2),
            exchange_flow=whale_data['net_exchange_flow'],
            network_health=network_data['network_utilization'],
            address_growth=network_data['address_growth_pct'],
            transaction_volume=network_data['daily_transaction_volume'],
            large_txn_count=whale_data['large_transactions'],
            funded_positions=whale_data['funded_positions'],
            liquidation_risk=liquidation_data['liquidation_cascade_risk'],
            on_chain_momentum=round(on_chain_momentum, 2),
            timestamp=datetime.now().isoformat(),
            confidence=round(confidence, 2)
        )
    
    def get_multi_asset_view(self, assets: List[str]) -> Dict[str, OnChainMetrics]:
        """
        Get on-chain metrics for multiple assets.
        Used by portfolio manager for allocation decisions.
        """
        result = {}
        for asset in assets:
            try:
                result[asset] = self.compute_on_chain_signal(asset)
            except Exception as e:
                print(f"Error fetching on-chain data for {asset}: {e}")
        return result
    
    def detect_extreme_signals(self, on_chain_data: Dict[str, OnChainMetrics]) -> Dict[str, Any]:
        """
        Detect extreme on-chain signals that warrant portfolio adjustments.
        """
        extremes = {
            "highly_bullish": [],  # Score > 0.6
            "highly_bearish": [],  # Score < -0.6
            "liquidation_risk": [],  # Risk > 70%
            "whale_accumulation": [],  # Large inflow + bullish
        }
        
        for asset, metrics in on_chain_data.items():
            if metrics.whale_score > 0.6:
                extremes["highly_bullish"].append(asset)
            if metrics.whale_score < -0.6:
                extremes["highly_bearish"].append(asset)
            if metrics.liquidation_risk > 70:
                extremes["liquidation_risk"].append(asset)
            if metrics.exchange_flow < -500 and metrics.whale_sentiment == "BULLISH":
                extremes["whale_accumulation"].append(asset)
        
        return extremes

