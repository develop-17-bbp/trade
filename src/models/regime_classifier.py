"""
Regime Classifier — HMM-Powered Market Regime Detection
=========================================================
Predicts market regime (bull/bear/sideways/crisis) using a Hidden Markov Model
fitted on [returns, volatility, volume_changes]. Falls back to heuristic if
hmmlearn is not installed.

Replaces the original stub that returned 'neutral' for every input.
"""

import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from src.models.hmm_regime import HMMRegimeDetector
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


class RegimeClassifier:
    """
    HMM-backed regime classifier.
    Falls back to volatility-based heuristic when HMM is unavailable or unfitted.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.hmm: Optional[HMMRegimeDetector] = None
        self._fitted = False

        if HAS_HMM:
            self.hmm = HMMRegimeDetector(
                n_states=self.config.get('n_states', 4),
                n_iter=self.config.get('n_iter', 100),
            )

    def fit(self, returns: np.ndarray, volatility: np.ndarray,
            volume_changes: np.ndarray) -> bool:
        """Fit HMM on historical data. Call periodically (e.g., weekly)."""
        if self.hmm is not None:
            self._fitted = self.hmm.fit(returns, volatility, volume_changes)
            return self._fitted
        return False

    def fit_from_ohlcv(self, df) -> bool:
        """Convenience: fit from OHLCV DataFrame."""
        if self.hmm is None:
            return False
        returns, vol, vol_chg = HMMRegimeDetector.prepare_from_ohlcv(df)
        return self.fit(returns, vol, vol_chg)

    def predict(self, features: List[Dict[str, float]]) -> List[str]:
        """
        Return regime label for each feature dict.
        Uses HMM if fitted, otherwise falls back to volatility heuristic.
        """
        results = []
        for feat in features:
            results.append(self._predict_single(feat))
        return results

    def predict_detailed(self, returns: np.ndarray, volatility: np.ndarray,
                         volume_changes: np.ndarray) -> Dict:
        """
        Full HMM prediction with probabilities.
        Returns: {regime, regime_id, probs, crisis_prob, stability, ...}
        """
        if self.hmm is not None and self._fitted:
            return self.hmm.predict(returns, volatility, volume_changes)

        # Fallback
        vol = float(np.mean(volatility[-20:])) if len(volatility) >= 20 else 0.02
        return self._heuristic_detailed(vol)

    def _predict_single(self, features: Dict[str, float]) -> str:
        """Predict regime from a single feature dict."""
        vol = features.get('ewma_vol', features.get('volatility', 0.02))
        trend = features.get('trend_strength', features.get('ema_10_slope', 0.0))
        returns_24 = features.get('returns_24', 0.0)

        # Use HMM regime if available in features (pre-computed)
        hmm_regime = features.get('hmm_regime', None)
        if hmm_regime is not None:
            return str(hmm_regime)

        # Heuristic fallback
        if vol > 0.06:
            return 'crisis'
        elif vol > 0.04 and returns_24 < -0.03:
            return 'bear'
        elif abs(trend) > 0.5 and returns_24 > 0.02:
            return 'bull'
        elif vol < 0.015:
            return 'sideways'
        else:
            return 'sideways'

    def _heuristic_detailed(self, vol: float) -> Dict:
        """Heuristic regime detection when HMM is unavailable."""
        if vol > 0.06:
            regime, rid = 'crisis', 3
            probs = [0.05, 0.15, 0.05, 0.75]
        elif vol > 0.035:
            regime, rid = 'bear', 1
            probs = [0.1, 0.6, 0.15, 0.15]
        elif vol < 0.015:
            regime, rid = 'sideways', 2
            probs = [0.15, 0.1, 0.65, 0.1]
        else:
            regime, rid = 'sideways', 2
            probs = [0.25, 0.25, 0.35, 0.15]

        return {
            'regime': regime,
            'regime_id': rid,
            'probs': probs,
            'bull_prob': probs[0],
            'bear_prob': probs[1],
            'sideways_prob': probs[2],
            'crisis_prob': probs[3],
            'stability': max(probs),
            'duration': 0,
            'transition_matrix': [[0.25]*4]*4,
        }
