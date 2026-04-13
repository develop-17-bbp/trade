"""
Hidden Markov Model Regime Detector
=====================================
Fits a Gaussian HMM to market observables to identify latent regimes:

  State 0: BULL      — positive mean return, moderate volatility
  State 1: BEAR      — negative mean return, moderate volatility
  State 2: SIDEWAYS  — near-zero return, low volatility
  State 3: CRISIS    — extreme negative return, high volatility

Observation vector (3D Gaussian emission):
    [log_return, realized_volatility, volume_change]

State labeling is done by sorting HMM-fitted means:
    - Highest vol state → CRISIS
    - Among non-crisis: highest mean return → BULL
    - Most negative mean return → BEAR
    - Remaining → SIDEWAYS

Usage:
    from src.models.hmm_regime import HMMRegimeDetector
    detector = HMMRegimeDetector()
    detector.fit(returns, volatility, volume_changes)
    result = detector.predict(returns, volatility, volume_changes)
    # result = {'regime': 'bull', 'regime_id': 0, 'probs': [0.85, 0.05, 0.08, 0.02], ...}
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    logger.warning("hmmlearn not installed. Run: pip install hmmlearn")


# Regime labels
BULL = 0
BEAR = 1
SIDEWAYS = 2
CRISIS = 3
REGIME_NAMES = {BULL: 'bull', BEAR: 'bear', SIDEWAYS: 'sideways', CRISIS: 'crisis'}


class HMMRegimeDetector:
    """
    4-state Gaussian HMM for market regime detection.
    """

    def __init__(self, n_states: int = 4, n_iter: int = 100,
                 covariance_type: str = 'diag', random_state: int = 42):
        """
        Args:
            n_states: Number of hidden states (default 4: bull/bear/sideways/crisis)
            n_iter: EM iterations for fitting
            covariance_type: 'full', 'diag', 'spherical', 'tied'
            random_state: Random seed for reproducibility
        """
        self.n_states = n_states
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state

        self.model: Optional[GaussianHMM] = None
        self.state_map: Dict[int, int] = {}  # HMM state → regime label
        self.is_fitted = False

    def _build_observations(self, returns: np.ndarray,
                            volatility: np.ndarray,
                            volume_changes: np.ndarray) -> np.ndarray:
        """Stack observation features into (n_samples, 3) matrix."""
        n = min(len(returns), len(volatility), len(volume_changes))
        obs = np.column_stack([
            returns[:n],
            volatility[:n],
            volume_changes[:n],
        ])
        # Remove NaN/Inf rows
        valid = np.all(np.isfinite(obs), axis=1)
        obs = obs[valid]
        # Add tiny jitter to prevent degenerate (zero-variance) covariance matrices
        obs += np.random.default_rng(42).normal(0, 1e-7, obs.shape)
        return obs

    def _label_states(self):
        """
        Map HMM hidden states to semantic regime labels.
        Strategy:
            1. Highest volatility mean → CRISIS
            2. Among rest: highest return mean → BULL
            3. Most negative return mean → BEAR
            4. Remaining → SIDEWAYS
        """
        if self.model is None:
            return

        means = self.model.means_  # (n_states, 3) — [return, vol, vol_change]
        n = self.n_states

        state_ids = list(range(n))
        self.state_map = {}

        # 1. CRISIS = highest volatility
        vol_means = [means[i, 1] for i in state_ids]
        crisis_id = state_ids[int(np.argmax(vol_means))]
        self.state_map[crisis_id] = CRISIS
        remaining = [s for s in state_ids if s != crisis_id]

        if len(remaining) >= 2:
            # 2. BULL = highest return mean
            ret_means = {s: means[s, 0] for s in remaining}
            bull_id = max(ret_means, key=ret_means.get)
            self.state_map[bull_id] = BULL
            remaining.remove(bull_id)

            # 3. BEAR = most negative return mean
            ret_means = {s: means[s, 0] for s in remaining}
            bear_id = min(ret_means, key=ret_means.get)
            self.state_map[bear_id] = BEAR
            remaining.remove(bear_id)

        # 4. SIDEWAYS = remaining
        for s in remaining:
            self.state_map[s] = SIDEWAYS

    def fit(self, returns: np.ndarray, volatility: np.ndarray,
            volume_changes: np.ndarray) -> bool:
        """
        Fit HMM to historical data.

        Args:
            returns: Log returns series
            volatility: Realized volatility series (e.g., rolling 20-bar std)
            volume_changes: Volume pct changes

        Returns:
            True if fitting succeeded
        """
        if not HAS_HMM:
            logger.warning("hmmlearn not installed — cannot fit HMM (optional)")
            return False

        obs = self._build_observations(returns, volatility, volume_changes)
        if len(obs) < 100:
            logger.warning(f"Too few observations ({len(obs)}) for HMM fitting")
            return False

        try:
            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                verbose=False,
            )
            self.model.fit(obs)
            self._label_states()
            self.is_fitted = True

            logger.info(f"HMM fitted on {len(obs)} observations, "
                        f"{self.n_states} states, "
                        f"score={self.model.score(obs):.2f}")
            return True

        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            self.is_fitted = False
            return False

    def predict(self, returns: np.ndarray, volatility: np.ndarray,
                volume_changes: np.ndarray) -> Dict:
        """
        Predict current regime from recent observations.

        Returns:
            Dict with: regime, regime_id, probs (all 4), crisis_prob,
                       stability, duration, transition_matrix
        """
        if not self.is_fitted or self.model is None:
            return self._default_result()

        obs = self._build_observations(returns, volatility, volume_changes)
        if len(obs) < 2:
            return self._default_result()

        try:
            # Get state sequence and posterior probabilities
            log_prob, state_seq = self.model.decode(obs, algorithm='viterbi')
            posteriors = self.model.predict_proba(obs)

            # Current state (last observation)
            current_hmm_state = state_seq[-1]
            current_regime_id = self.state_map.get(current_hmm_state, SIDEWAYS)
            current_probs = posteriors[-1]

            # Map probabilities to regime order
            regime_probs = np.zeros(4)
            for hmm_state, regime_id in self.state_map.items():
                if hmm_state < len(current_probs):
                    regime_probs[regime_id] += current_probs[hmm_state]

            # Stability = max posterior probability
            stability = float(np.max(regime_probs))

            # Duration = bars since last regime change
            duration = 1
            for i in range(len(state_seq) - 2, -1, -1):
                if self.state_map.get(state_seq[i], SIDEWAYS) == current_regime_id:
                    duration += 1
                else:
                    break

            # Transition matrix (mapped to regime labels)
            transmat = np.zeros((4, 4))
            hmm_trans = self.model.transmat_
            for from_hmm, from_regime in self.state_map.items():
                for to_hmm, to_regime in self.state_map.items():
                    if from_hmm < hmm_trans.shape[0] and to_hmm < hmm_trans.shape[1]:
                        transmat[from_regime, to_regime] += hmm_trans[from_hmm, to_hmm]

            return {
                'regime': REGIME_NAMES[current_regime_id],
                'regime_id': current_regime_id,
                'probs': regime_probs.tolist(),
                'bull_prob': float(regime_probs[BULL]),
                'bear_prob': float(regime_probs[BEAR]),
                'sideways_prob': float(regime_probs[SIDEWAYS]),
                'crisis_prob': float(regime_probs[CRISIS]),
                'stability': stability,
                'duration': duration,
                'transition_matrix': transmat.tolist(),
            }

        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return self._default_result()

    def _default_result(self) -> Dict:
        """Return neutral/sideways result when HMM isn't available."""
        return {
            'regime': 'sideways',
            'regime_id': SIDEWAYS,
            'probs': [0.25, 0.25, 0.25, 0.25],
            'bull_prob': 0.25,
            'bear_prob': 0.25,
            'sideways_prob': 0.25,
            'crisis_prob': 0.25,
            'stability': 0.25,
            'duration': 0,
            'transition_matrix': [[0.25]*4]*4,
        }

    # ══════════════════════════════════════════════════════════════
    # REGIME TRANSITION PREDICTION (inspired by Markov Engine post)
    # ══════════════════════════════════════════════════════════════

    def predict_transition(self, returns: np.ndarray, volatility: np.ndarray,
                           volume_changes: np.ndarray, horizon_bars: int = 6) -> Dict:
        """
        Predict WHERE the regime is likely to transition next.

        Instead of just "we're in SIDEWAYS", answers:
        "SIDEWAYS is 35% likely to transition to BULL in next 6 bars"

        Uses the HMM transition matrix raised to the power of horizon_bars
        to compute multi-step transition probabilities.

        Args:
            returns, volatility, volume_changes: Recent observations
            horizon_bars: How many bars ahead to predict (default 6 = ~24h on 4h TF)

        Returns:
            Dict with: current_regime, next_probable_regime, transition_probs,
                       regime_stability, regime_age, is_regime_about_to_change,
                       recommended_strategy_bias
        """
        current = self.predict(returns, volatility, volume_changes)
        regime = current['regime']
        regime_id = current['regime_id']
        stability = current['stability']
        duration = current['duration']
        transmat = np.array(current['transition_matrix'])

        # Multi-step transition: P(state at t+h | state at t) = T^h
        try:
            multi_step = np.linalg.matrix_power(transmat, horizon_bars)
            # Probabilities of being in each regime after horizon_bars
            future_probs = multi_step[regime_id]
        except Exception:
            future_probs = [0.25, 0.25, 0.25, 0.25]

        # Most likely next regime (excluding staying in same)
        future_copy = list(future_probs)
        stay_prob = future_copy[regime_id]
        future_copy[regime_id] = 0  # Exclude current
        next_regime_id = int(np.argmax(future_copy))
        change_prob = 1.0 - stay_prob

        # Hysteresis: regime is "about to change" if:
        # 1. Stay probability < 60% (weakening)
        # 2. OR duration > 20 bars (regime exhaustion — markets don't stay in one state forever)
        # 3. AND stability < 0.7 (not strongly in current regime)
        is_about_to_change = (
            (stay_prob < 0.60) or
            (duration > 20 and stability < 0.70) or
            (change_prob > 0.45)
        )

        # Persistence score: how "sticky" is the current regime?
        # High persistence = stay in current strategy allocation
        # Low persistence = prepare to shift
        persistence = stay_prob * min(1.0, stability * 1.2)

        # Strategy bias recommendation based on transition prediction
        if regime == 'sideways' and future_probs[BULL] > future_probs[BEAR]:
            strategy_bias = "PREPARE_LONG"
            bias_reason = f"SIDEWAYS→BULL transition {future_probs[BULL]:.0%} likely"
        elif regime == 'sideways' and future_probs[BEAR] > future_probs[BULL]:
            strategy_bias = "PREPARE_DEFENSIVE"
            bias_reason = f"SIDEWAYS→BEAR transition {future_probs[BEAR]:.0%} likely"
        elif regime == 'bull' and future_probs[BEAR] + future_probs[CRISIS] > 0.30:
            strategy_bias = "TIGHTEN_STOPS"
            bias_reason = f"BULL→BEAR/CRISIS risk {future_probs[BEAR]+future_probs[CRISIS]:.0%}"
        elif regime == 'bear' and future_probs[BULL] > 0.25:
            strategy_bias = "WATCH_REVERSAL"
            bias_reason = f"BEAR→BULL transition {future_probs[BULL]:.0%} possible"
        elif regime == 'crisis':
            strategy_bias = "CAPITAL_PRESERVATION"
            bias_reason = "CRISIS regime — protect capital"
        else:
            strategy_bias = "HOLD_CURRENT"
            bias_reason = f"Regime stable ({stay_prob:.0%} persistence)"

        return {
            'current_regime': regime,
            'current_stability': round(stability, 3),
            'regime_age_bars': duration,
            'persistence_score': round(persistence, 3),

            # Transition prediction
            'stay_probability': round(stay_prob, 3),
            'change_probability': round(change_prob, 3),
            'next_probable_regime': REGIME_NAMES[next_regime_id],
            'next_regime_probability': round(future_copy[next_regime_id], 3),
            'is_about_to_change': is_about_to_change,
            'horizon_bars': horizon_bars,

            # Full future probability distribution
            'future_probs': {
                'bull': round(future_probs[BULL], 3),
                'bear': round(future_probs[BEAR], 3),
                'sideways': round(future_probs[SIDEWAYS], 3),
                'crisis': round(future_probs[CRISIS], 3),
            },

            # Strategy recommendation
            'strategy_bias': strategy_bias,
            'bias_reason': bias_reason,

            # Raw transition matrix (single step)
            'transition_matrix': transmat.tolist(),
        }

    def regime_encoded(self, regime_id: int) -> int:
        """Encode regime as integer for ML features (same as regime_id)."""
        return regime_id

    @staticmethod
    def prepare_from_ohlcv(df) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare HMM inputs from OHLCV DataFrame.

        Args:
            df: DataFrame with 'close' and 'volume' columns

        Returns:
            (returns, volatility, volume_changes) arrays
        """
        close = df['close'].values.astype(float)
        volume = df['volume'].values.astype(float)

        returns = np.diff(np.log(close + 1e-12))

        # 20-bar rolling volatility
        vol = np.zeros(len(returns))
        for i in range(20, len(returns)):
            vol[i] = np.std(returns[i-20:i])
        vol[:20] = vol[20] if len(vol) > 20 else 0.01

        # Volume pct changes
        vol_changes = np.diff(volume) / (volume[:-1] + 1e-12)

        # Align lengths
        n = min(len(returns), len(vol), len(vol_changes))
        return returns[:n], vol[:n], vol_changes[:n]
