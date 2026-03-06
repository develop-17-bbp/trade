import numpy as np
from typing import Dict, Any, List

class LiquidityEstimator:
    """
    Estimates market liquidity and expected slippage for large orders.
    Essential for institutional-grade order splitting.
    """
    def __init__(self, impact_coefficient: float = 0.1):
        self.impact_coeff = impact_coefficient

    def estimate_liquidity(self, order_book: Dict[str, List[List[float]]], 
                          recent_volume: float = 100.0) -> Dict[str, Any]:
        """
        Analyze order book depth and spread.
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if not bids or not asks:
            return {"max_safe_size": 0, "expected_slippage": 1.0}

        spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        
        # Calculate depth (e.g., within 1% of mid-price)
        mid_price = (bids[0][0] + asks[0][0]) / 2
        bid_depth_1pct = sum(qty for price, qty in bids if price > mid_price * 0.99)
        ask_depth_1pct = sum(qty for price, qty in asks if price < mid_price * 1.01)
        
        # Institutional heuristic: Max safe size is 10% of 1% depth
        max_safe_size = min(bid_depth_1pct, ask_depth_1pct) * 0.1
        
        return {
            "mid_price": mid_price,
            "spread": spread,
            "bid_depth_1pct": bid_depth_1pct,
            "ask_depth_1pct": ask_depth_1pct,
            "max_safe_size": max_safe_size,
            "liquidity_score": min(1.0, (bid_depth_1pct + ask_depth_1pct) / (recent_volume * 10))
        }

    def calculate_expected_slippage(self, order_size: float, side: str, 
                                   order_book: Dict[str, List[List[float]]]) -> float:
        """
        Calculates theoretical slippage based on walking the book.
        """
        levels = order_book.get('asks' if side == 'buy' else 'bids', [])
        if not levels: return 0.05
        
        remaining = order_size
        total_cost = 0.0
        
        for price, qty in levels:
            fill = min(remaining, qty)
            total_cost += fill * price
            remaining -= fill
            if remaining <= 0: break
            
        if remaining > 0:
            # Not enough liquidity even in full book
            avg_price = (total_cost / (order_size - remaining)) if (order_size - remaining) > 0 else levels[-1][0]
            avg_price *= 1.1 # Penalize for incomplete fill
        else:
            avg_price = total_cost / order_size
            
        mid = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        slippage = abs(avg_price - mid) / mid
        return slippage
