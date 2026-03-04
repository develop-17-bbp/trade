"""
Cross-Asset Correlation Monitor
===============================
Tracks relationships between BTC, ETH, and other assets to identify:
1. Laggard trades (Coin A hasn't followed Coin B yet)
2. Herd immunity (Diversification vetoes when all coins are correlated 1.0)
"""
import numpy as np
from typing import Dict, List, Optional

class CorrelationMonitor:
    def __init__(self, window: int = 20):
        self.window = window
        self.price_history: Dict[str, List[float]] = {}

    def update(self, asset: str, price: float):
        if asset not in self.price_history:
            self.price_history[asset] = []
        self.price_history[asset].append(price)
        if len(self.price_history[asset]) > self.window:
            self.price_history[asset].pop(0)

    def get_correlation(self, asset_a: str, asset_b: str) -> float:
        """Returns Pearson correlation between two assets."""
        if asset_a not in self.price_history or asset_b not in self.price_history:
            return 0.0
        
        hist_a = self.price_history[asset_a]
        hist_b = self.price_history[asset_b]
        
        # Ensure equal length
        min_len = min(len(hist_a), len(hist_b))
        if min_len < 5: return 0.0
        
        a = np.array(hist_a[-min_len:])
        b = np.array(hist_b[-min_len:])
        
        if np.std(a) == 0 or np.std(b) == 0:
            return 1.0
            
        return float(np.corrcoef(a, b)[0, 1])

    def get_market_beta(self) -> float:
        """Similarity between BTC and ETH (proxy for market herd behavior)."""
        return self.get_correlation('BTC', 'ETH')
