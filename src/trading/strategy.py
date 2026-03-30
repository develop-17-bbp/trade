"""
Strategy — EMA Crossover via Adaptive Engine
=============================================
Thin wrapper around AdaptiveEngine which holds the EMA crossover strategy.
All other layers (LightGBM, FinBERT, MetaController, etc.) removed.
"""

from typing import Dict, Optional
from src.trading.adaptive_engine import AdaptiveEngine


class HybridStrategy:
    """
    Simplified strategy: only holds AdaptiveEngine for EMA crossover access.
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        self.adaptive_engine = AdaptiveEngine(cfg.get('adaptive', {}))

    def get_ema_strategy(self):
        """Get the EMA crossover strategy instance."""
        return self.adaptive_engine.strategies.get('ema_crossover')


class SimpleStrategy:
    """Legacy placeholder."""
    def __init__(self, config=None):
        pass
