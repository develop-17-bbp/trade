"""
PHASE 5: Price Action Indicators
==================================
Detects Order Blocks, Fair Value Gaps (FVG), and Support/Resistance zones.
These zones define high-probability areas for trading on historical liquidity.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

class PriceActionAnalyzer:
    """
    Analyzes institutional liquidity zones (Order Blocks, FVG)
    to identify rejection and balance areas.
    """
    def __init__(self, window: int = 50):
        self.window = window

    def get_fvg(self, highs: List[float], lows: List[float]) -> List[Dict[str, float]]:
        """
        Fair Value Gap Detection.
        Occurs when there's an imbalance in a 3-candle sequence.
        Bullish FVG: low[i] > high[i-2]
        Bearish FVG: high[i] < low[i-2]
        """
        fvgs = []
        n = len(highs)
        for i in range(2, n):
            # Bullish FVG
            if lows[i] > highs[i - 2]:
                gap_size = lows[i] - highs[i - 2]
                fvgs.append({
                    "type": "bullish",
                    "top": lows[i],
                    "bottom": highs[i - 2],
                    "size": gap_size,
                    "index": i - 1
                })
            # Bearish FVG
            elif highs[i] < lows[i - 2]:
                gap_size = lows[i - 2] - highs[i]
                fvgs.append({
                    "type": "bearish",
                    "top": lows[i - 2],
                    "bottom": highs[i],
                    "size": gap_size,
                    "index": i - 1
                })
        return fvgs

    def get_order_blocks(self, opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> List[Dict[str, float]]:
        """
        Institutional Order Block Detection.
        Bullish OB: Last down candle before a strong move up (BOS).
        Bearish OB: Last up candle before a strong move down (BOS).
        """
        obs = []
        n = len(closes)
        for i in range(1, n - 2):
            # Check for strong expansion (imbalance)
            is_strong_up = (closes[i + 1] > closes[i]) and (closes[i+2] > closes[i+1]*1.005)
            is_strong_down = (closes[i + 1] < closes[i]) and (closes[i+2] < closes[i+1]*0.995)
            
            # Possible Bullish OB: last down candle
            if is_strong_up and closes[i] < opens[i]:
                obs.append({
                    "type": "bullish",
                    "top": highs[i],
                    "bottom": lows[i],
                    "volume": volumes[i],
                    "index": i
                })
            # Possible Bearish OB: last up candle
            elif is_strong_down and closes[i] > opens[i]:
                obs.append({
                    "type": "bearish",
                    "top": highs[i],
                    "bottom": lows[i],
                    "volume": volumes[i],
                    "index": i
                })
        return obs

    def get_support_resistance(self, highs: List[float], lows: List[float], closes: List[float]) -> Tuple[List[float], List[float]]:
        """Identify key Support and Resistance levels based on historical rejections."""
        # Simple local peaks/troughs
        resistances = []
        supports = []
        n = len(closes)
        lookback = 10
        
        for i in range(lookback, n - lookback):
            if highs[i] == max(highs[i-lookback : i+lookback]):
                resistances.append(highs[i])
            if lows[i] == min(lows[i-lookback : i+lookback]):
                supports.append(lows[i])
                
        # Consolidate nearby levels (simple grouping)
        def _consolidate(items: List[float], threshold=0.01):
            if not items: return []
            items.sort()
            groups = [[items[0]]]
            for x in items[1:]:
                if (x / groups[-1][-1]) - 1 < threshold:
                    groups[-1].append(x)
                else:
                    groups.append([x])
            return [sum(g) / len(g) for g in groups]

        return _consolidate(supports), _consolidate(resistances)

    def get_price_action_features(self, opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict[str, float]:
        """Generate ML features from price action zones."""
        fvgs = self.get_fvg(highs, lows)
        obs = self.get_order_blocks(opens, highs, lows, closes, volumes)
        supports, resistances = self.get_support_resistance(highs, lows, closes)
        
        cur_p = closes[-1]
        
        # Proximity to nearest OB
        nearest_bull_ob_dist = 1.0
        for ob in reversed(obs):
            if ob['type'] == 'bullish' and cur_p > ob['top']:
                nearest_bull_ob_dist = (cur_p / ob['top']) - 1
                break
                
        nearest_bear_ob_dist = 1.0
        for ob in reversed(obs):
            if ob['type'] == 'bearish' and cur_p < ob['bottom']:
                nearest_bear_ob_dist = 1 - (cur_p / ob['bottom'])
                break
                
        # Is price currently in an FVG?
        in_bull_fvg = 0.0
        for fvg in reversed(fvgs):
            if fvg['type'] == 'bullish' and fvg['bottom'] <= cur_p <= fvg['top']:
                in_bull_fvg = 1.0
                break
                
        return {
            "proximity_bull_ob": nearest_bull_ob_dist,
            "proximity_bear_ob": nearest_bear_ob_dist,
            "in_bull_fvg": in_bull_fvg,
            "in_bear_fvg": 1.0 if any(f['type'] == 'bearish' and f['bottom'] <= cur_p <= f['top'] for fvg in reversed(fvgs)) else 0.0,
            "dist_to_support": min([(cur_p / s - 1) for s in supports if cur_p > s] or [1.0]),
            "dist_to_resistance": min([(r / cur_p - 1) for r in resistances if r > cur_p] or [1.0]),
        }
