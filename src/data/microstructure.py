"""
PHASE 5: Order Book Microstructure Analysis
============================================
Analyzes L2 Order Book depth, imbalance, and pressure.
Essential for institutional scalping and execution timing.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

class MicrostructureAnalyzer:
    """
    Analyzes Order Book depth and bid/ask pressure.
    Detects liquidity walls and imbalances.
    """
    def __init__(self, depth: int = 20):
        self.depth = depth

    def detect_liquidity_voids(self, levels: np.ndarray) -> float:
        """
        Identify areas with very little liquidity (Institutional Signal 9).
        Price moves extremely fast through these zones.
        """
        if len(levels) < 10: return 0.0
        
        # Calculate volume density per price step
        vols = levels[:, 1]
        median_vol = np.median(vols)
        
        # A 'void' is a sequence of levels where vol < 10% of median
        voids = np.where(vols < (median_vol * 0.1))[0]
        return float(len(voids) / len(levels)) # Void Ratio

    def analyze_order_book(self, order_book: Dict[str, List[List[float]]]) -> Dict[str, float]:
        """
        Calculates imbalance, pressure, and VOIDS from L2 data.
        """
        bids = np.array(order_book.get('bids', []))
        asks = np.array(order_book.get('asks', []))
        
        if len(bids) < 5 or len(asks) < 5:
            return {
                "l2_imbalance": 0.0, "l2_slope_ratio": 1.0, "spread_expansion": 0.0,
                "iceberg_detected": 0.0, "spoofing_detected": 0.0, "l2_void_ratio": 0.0
            }

        # 1. Bid/Ask Imbalance
        bid_vol = np.sum(bids[:self.depth, 1])
        ask_vol = np.sum(asks[:self.depth, 1])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-10)
        
        # 2. Liquidity Voids (Signal 9)
        bid_voids = self.detect_liquidity_voids(bids)
        ask_voids = self.detect_liquidity_voids(asks)
        
        # 3. Order Book Slope
        bid_slope = (np.sum(bids[:5, 1])) / (abs(bids[0, 0] - bids[4, 0]) + 1e-10)
        ask_slope = (np.sum(asks[:5, 1])) / (abs(asks[0, 0] - asks[4, 0]) + 1e-10)
        slope_ratio = bid_slope / (ask_slope + 1e-10)
        
        # 4. Spread Delta
        best_bid, best_ask = bids[0, 0], asks[0, 0]
        current_spread = (best_ask - best_bid) / best_bid
        spread_expansion = 1.0 if current_spread > 0.001 else 0.0 
        
        # 5. Liquidity Walls
        avg_bid_vol, avg_ask_vol = np.mean(bids[:, 1]), np.mean(asks[:, 1])
        bid_wall = 1.0 if np.any(bids[:10, 1] > avg_bid_vol * 3.5) else 0.0
        ask_wall = -1.0 if np.any(asks[:10, 1] > avg_ask_vol * 3.5) else 0.0
        
        return {
            "l2_imbalance": float(imbalance),
            "l2_slope_ratio": float(slope_ratio),
            "spread_expansion": float(spread_expansion),
            "l2_void_ratio": float(bid_voids + ask_voids),
            "l2_wall_signal": float(bid_wall + ask_wall),
            "bid_depth_usd": float(bid_vol * best_bid),
            "ask_depth_usd": float(ask_vol * best_ask)
        }

    def detect_liquidity_regime(self, bid_depth: float, ask_depth: float) -> str:
        """
        Detects current sessions and liquidity levels.
        """
        import time
        from datetime import datetime
        now = datetime.utcnow()
        hour = now.hour
        is_weekend = now.weekday() >= 5
        
        session = "OTHER"
        if 0 <= hour < 8: session = "ASIA"
        elif 8 <= hour < 16: session = "LONDON"
        elif 16 <= hour <= 23: session = "US"
        if is_weekend: session = "WEEKEND"
        
        total_depth = bid_depth + ask_depth
        regime = "NORMAL"
        if is_weekend or total_depth < 1e6: regime = "LOW"
        elif total_depth > 5e6: regime = "HIGH"
        
        return f"{session}_{regime}"

    def estimate_slippage(self, order_book: Dict[str, List[List[float]]], quantity: float, side: str) -> float:
        """
        Estimates expected slippage by walking the order book.
        """
        levels = order_book.get('bids' if side == 'sell' else 'asks', [])
        if not levels or quantity <= 0:
            return 0.0
            
        initial_price = levels[0][0]
        filled = 0.0
        weighted_price = 0.0
        
        for price, size in levels:
            needed = quantity - filled
            can_fill = min(needed, size)
            weighted_price += can_fill * price
            filled += can_fill
            if filled >= quantity:
                break
                
        if filled < quantity:
            return 1.0 # 100% slippage penalty
            
        avg_price = weighted_price / quantity
        slippage = abs(avg_price - initial_price) / (initial_price + 1e-10)
        return float(slippage)
