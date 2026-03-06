import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SlippageModel:
    """
    Advanced Slippage Modeling for institutional order impact estimation.
    Used for pre-trade planning and execution quality assessment.
    """
    def __init__(self, volatility_scale: float = 0.5):
        self.vol_scale = volatility_scale

    def estimate_price_impact(self, order_size: float, 
                             adv_24h: float, 
                             volatility_daily: float = 0.02) -> float:
        """
        Uses a Square-Root Impact Model (Standard institutional model).
        Impact = K * Volatility * sqrt( OrderSize / TotalVolume )
        
        Args:
            order_size: Proposed order quantity in base currency.
            adv_24h: Average Daily Volume (last 24h).
            volatility_daily: Asset daily standard deviation of returns.
        """
        if adv_24h <= 0: return 0.05 # Conservative fallback
        
        participation = order_size / adv_24h
        
        # Institutional 'I-Star' model
        # impact_bps = K * (std_dev_daily) * sqrt(phi)
        # where phi = order_size / ADV
        
        # Typical K values range from 0.1 to 0.7 depending on venue
        k_impact = 0.4 
        
        impact_bps = k_impact * volatility_daily * np.sqrt(participation)
        
        # Linear spread component (half spread)
        spread_cost = 0.0001 # 1 bps assume
        
        expected_slippage = spread_cost + impact_bps
        
        return expected_slippage

    def get_max_position_size_for_slippage(self, target_slippage_bps: float, 
                                          adv_24h: float, 
                                          volatility_daily: float = 0.02) -> float:
        """
        Inverse calculation: What's the max order size for a given slippage tolerance?
        """
        # bps to decimal: 10 bps = 0.001
        limit = target_slippage_bps / 10000.0
        
        # impact = K * Vol * sqrt( participation )
        # participation = (impact / (K * Vol)) ^ 2
        
        k_impact = 0.4
        
        inner = limit / (k_impact * volatility_daily)
        max_participation = inner**2
        
        return max_participation * adv_24h
