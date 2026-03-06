"""
PHASE 5: Market Structure Analysis
==================================
Detects HH, HL, LH, LL to identify structural trends.
Identifies BOS (Break of Structure) and CHoCH (Change of Character).
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

@dataclass
class Pivot:
    index: int
    price: float
    type: str  # 'HH', 'HL', 'LH', 'LL'

class MarketStructureAnalyzer:
    """
    Analyzes price pivots to determine market structure and trend flips.
    Essential for avoiding 'falling knife' scenarios.
    """
    def __init__(self, window: int = 5):
        self.window = window

    def find_pivots(self, highs: List[float], lows: List[float]) -> List[Pivot]:
        """Identify swing highs and swing lows."""
        pivots = []
        n = len(highs)
        
        for i in range(self.window, n - self.window):
            # Swing High
            if highs[i] == max(highs[i - self.window : i + self.window + 1]):
                # Determine if HH or LH
                pve_highs = [p for p in pivots if p.type in ['HH', 'LH']]
                p_type = 'HH'
                if pve_highs and highs[i] <= pve_highs[-1].price:
                    p_type = 'LH'
                pivots.append(Pivot(i, highs[i], p_type))
            
            # Swing Low
            if lows[i] == min(lows[i - self.window : i + self.window + 1]):
                # Determine if HL or LL
                pve_lows = [p for p in pivots if p.type in ['HL', 'LL']]
                p_type = 'HL'
                if pve_lows and lows[i] <= pve_lows[-1].price:
                    p_type = 'LL'
                pivots.append(Pivot(i, lows[i], p_type))
                
        return pivots

    def detect_structure_breaks(self, pivots: List[Pivot], current_price: float) -> Dict[str, bool]:
        """
        Detect BOS and CHoCH.
        BOS: Price breaks past a previous HH (in uptrend) or LL (in downtrend).
        CHoCH: Price breaks past the last HL in an uptrend (first sign of reversal).
        """
        if not pivots:
            return {"BOS": False, "CHoCH": False}
            
        last_hh = next((p for p in reversed(pivots) if p.type == 'HH'), None)
        last_ll = next((p for p in reversed(pivots) if p.type == 'LL'), None)
        last_hl = next((p for p in reversed(pivots) if p.type == 'HL'), None)
        last_lh = next((p for p in reversed(pivots) if p.type == 'LH'), None)
        
        bos = False
        choch = False
        
        # Bullish BOS: Price > last HH
        if last_hh and current_price > last_hh.price:
            bos = True
        # Bearish BOS: Price < last LL
        if last_ll and current_price < last_ll.price:
            bos = True
            
        # Bullish CHoCH: Price > last LH (after a downtrend)
        if last_lh and current_price > last_lh.price:
            choch = True
        # Bearish CHoCH: Price < last HL (after an uptrend)
        if last_hl and current_price < last_hl.price:
            choch = True
            
        return {"bos": bos, "choch": choch}

    def get_market_structure_features(self, highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, float]:
        """Generate features for ML consumption."""
        pivots = self.find_pivots(highs, lows)
        breaks = self.detect_structure_breaks(pivots, closes[-1])
        
        # Determine current trend based on last 2 pivots
        trend = 0 # Neutral
        if len(pivots) >= 2:
            last = pivots[-1]
            if last.type in ['HH', 'HL']:
                trend = 1
            elif last.type in ['LL', 'LH']:
                trend = -1
                
        return {
            "ms_trend": float(trend),
            "ms_bos": 1.0 if breaks["bos"] else 0.0,
            "ms_choch": 1.0 if breaks["choch"] else 0.0,
            "last_pivot_type": 1.0 if pivots and pivots[-1].type in ['HH', 'HL'] else -1.0
        }
