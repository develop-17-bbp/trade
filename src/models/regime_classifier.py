"""
Regime classifier
=================
Predicts market regime (bull/chop/bear/volatility) from feature vectors.
Currently a stub returning 'neutral' for every input.
"""

from typing import List, Dict


class RegimeClassifier:
    def __init__(self, config: Dict = None):
        self.config = config or {}

    def predict(self, features: List[Dict[str, float]]) -> List[str]:
        """Return regime label for each feature dict."""
        return ["neutral" for _ in features]
