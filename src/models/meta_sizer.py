"""
Meta-RL position sizing agent
==============================
A second RL-based agent that learns optimal position size rather than direction.
Currently returns fixed fraction of 1.0 (full size).
"""

from typing import List, Dict, Tuple


class MetaSizer:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def size(self, features: Dict[str, float], win_prob: float = 0.5, win_loss_ratio: float = 2.0) -> float:
        """
        Return a size multiplier between 0.0 and 1.0 using Kelly Criterion.
        Kelly % = (p * b - q) / b
        p = win_prob, q = loss_prob, b = win_loss_ratio
        """
        p = win_prob
        q = 1.0 - p
        b = win_loss_ratio
        
        if b <= 0: return 0.1
        
        kf = (p * b - q) / b
        
        # Fractional Kelly (safer)
        fractional_kelly = kf * 0.5
        
        return float(max(0.1, min(1.0, float(fractional_kelly))))
