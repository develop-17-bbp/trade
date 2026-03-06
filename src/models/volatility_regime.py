"""
PHASE 5: Volatility Regime Detection
==================================
Identifies Market Regimes based on volatility profiles.
Essential for switching between Trending and Mean Reversion strategies.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from enum import Enum

class VolatilityRegime(Enum):
    LOW_VOL_RANGE = "LOW_VOL_RANGE"
    TREND_EXPANSION = "TREND_EXPANSION"
    HIGH_VOL_PANIC = "HIGH_VOL_PANIC"
    NORMAL = "NORMAL"

class VolatilityRegimeDetector:
    """
    Detects Volatility Regimes using ATR, and clustering of returns (realized vol).
    """
    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def detect_regime(self, closes: List[float], highs: List[float], lows: List[float]) -> Dict[str, Any]:
        """
        Categorizes current market into one of the VolatilityRegime states.
        """
        if len(closes) < 50:
            return {"regime": VolatilityRegime.NORMAL, "realized_vol": 0.05}
            
        # 1. Realized Volatility (Standard Deviation of returns)
        returns = np.diff(np.log(closes))
        realized_vol = float(np.std(returns) * np.sqrt(365 * 24 * 12)) # Annualized for 5m bars
        
        # 2. Average True Range (ATR) normalization
        atr = self._calculate_atr(highs, lows, closes, 14)
        atr_pct = atr[-1] / closes[-1] if closes[-1] > 0 else 0.0
        
        # Historical context for ATR (z-score)
        avg_atr_pct = np.mean([a / c for a, c in zip(atr, closes) if c > 0])
        std_atr_pct = np.std([a / c for a, c in zip(atr, closes) if c > 0]) + 1e-10
        atr_z = (atr_pct - avg_atr_pct) / std_atr_pct
        
        # 3. Volatility Clustering (Clusters of high returns)
        cluster_score = np.mean(np.abs(returns[-5:])) / (np.mean(np.abs(returns[-50:])) + 1e-10)
        
        regime = VolatilityRegime.NORMAL
        
        if atr_z < -1.0 and cluster_score < 0.8:
            regime = VolatilityRegime.LOW_VOL_RANGE
        elif (atr_z > 0.5 or cluster_score > 1.5) and abs(np.mean(returns[-5:])) > 0.001:
            regime = VolatilityRegime.TREND_EXPANSION
        
        if atr_z > 2.5 or cluster_score > 3.0:
            regime = VolatilityRegime.HIGH_VOL_PANIC
            
        # Encoding for ML processing (Signal 70)
        vol_map = {
            VolatilityRegime.LOW_VOL_RANGE.value: 0,
            VolatilityRegime.NORMAL.value: 1,
            VolatilityRegime.TREND_EXPANSION.value: 2,
            VolatilityRegime.HIGH_VOL_PANIC.value: 3
        }
        
        return {
            "vol_regime": regime.value,
            "vol_regime_encoded": float(vol_map.get(regime.value, 1)),
            "realized_vol_annual": realized_vol,
            "atr_pct": atr_pct,
            "vol_cluster_score": cluster_score,
            "atr_z_score": atr_z
        }

    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> List[float]:
        """Calculates Average True Range."""
        tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) 
              for i in range(1, len(closes))]
        tr = [tr[0]] + tr # Pad first value
        atr = []
        val = np.mean(tr[:n])
        for i in range(len(tr)):
            if i < n:
                atr.append(val)
            else:
                val = (val * (n-1) + tr[i]) / n
                atr.append(val)
        return atr
