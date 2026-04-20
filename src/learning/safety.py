"""Learning-mesh safety guardrails — Phase 4.5c (§7).

Four layers of defense, in order of how often they fire:

  (1) Learning delta caps — reject an update that moves a learner too far
      from its prior checkpoint in one cycle. Prevents runaway drift on a
      single outlier trade.

      RL policy:   KL(π_new || π_old)  ≤ 0.05
      LoRA:        cosine(adapter_new, adapter_old)  ≥ 0.92
      Genetic:     σ(mutation_vector)  ≤  2·rolling_σ
      Calibrator:  max |curve_new - curve_old|  ≤ 0.08

  (2) Z-score filter on cross-learner signals — a consumed signal passing
      through > 3σ of its rolling 100-sample distribution is dropped
      (Plan §7.2). Keeps a broken producer from poisoning consumers.

  (3) Quarantine manager — a learner whose signals deviate > 3σ for >5
      consecutive outcomes is quarantined: marked `quarantined=True`,
      its signals stop being consumed by others, and it keeps training
      in isolation until signals return to within 2σ.

  (4) Authority compliance gate — trades that violated authority rules
      at decision time produce zero positive reward for non-veto
      components, regardless of PnL (§7.3).

Everything here is pure-Python + numpy — no external deps. Safe to import
from anywhere in the learning package.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)


# ── 1. Delta caps ──────────────────────────────────────────────────────

KL_RL_MAX = 0.05
COSINE_LORA_MIN = 0.92
GENETIC_SIGMA_MAX_MULT = 2.0
CALIBRATOR_SHIFT_MAX = 0.08


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Discrete KL(p || q) over a shared action space. p/q must sum to 1."""
    s = 0.0
    for k, pk in p.items():
        qk = max(1e-12, q.get(k, 1e-12))
        if pk > 0:
            s += pk * math.log(pk / qk)
    return s


def cosine_similarity(a, b) -> float:
    """Cosine between two vectors. Returns 1.0 on degenerate input."""
    try:
        import numpy as np
        av = np.asarray(a, dtype=float).ravel()
        bv = np.asarray(b, dtype=float).ravel()
        na, nb = float(np.linalg.norm(av)), float(np.linalg.norm(bv))
        if na == 0 or nb == 0:
            return 1.0
        return float(np.dot(av, bv) / (na * nb))
    except Exception:
        # Fallback pure-Python
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 1.0
        return dot / (na * nb)


def check_rl_update(new_policy: Dict[str, float], old_policy: Dict[str, float]) -> bool:
    """True if the update is within cap, False if it should be rejected."""
    return kl_divergence(new_policy, old_policy) <= KL_RL_MAX


def check_lora_update(new_adapter, old_adapter) -> bool:
    return cosine_similarity(new_adapter, old_adapter) >= COSINE_LORA_MIN


def check_calibrator_update(new_curve: list, old_curve: list) -> bool:
    if len(new_curve) != len(old_curve):
        return False
    return max(abs(n - o) for n, o in zip(new_curve, old_curve)) <= CALIBRATOR_SHIFT_MAX


def check_genetic_update(mutation_vec, rolling_sigma: float) -> bool:
    try:
        import numpy as np
        sigma = float(np.std(np.asarray(mutation_vec, dtype=float)))
    except Exception:
        sigma = 0.0
    return sigma <= max(rolling_sigma, 1e-9) * GENETIC_SIGMA_MAX_MULT


# ── 2. Z-score filter + 3. quarantine manager ──────────────────────────

Z_SCORE_REJECT = 3.0
Z_SCORE_ACCEPT = 2.0   # below this → out of quarantine
CONSECUTIVE_TO_QUARANTINE = 5
ROLLING_WINDOW = 100


@dataclass
class _LearnerState:
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=ROLLING_WINDOW))
    consecutive_breaches: int = 0
    quarantined: bool = False


class QuarantineManager:
    """Tracks rolling z-scores per (learner, signal) pair and quarantines.

    Not a singleton by design — the executor makes one and passes it to
    consumers. Tests can spin disposables without state leakage.
    """

    def __init__(self) -> None:
        self._state: Dict[str, _LearnerState] = {}
        self._lock = threading.Lock()

    def _key(self, learner: str, signal: str) -> str:
        return f"{learner}:{signal}"

    def observe(self, learner: str, signal: str, value: float) -> float:
        """Record a sample, return its z-score vs the rolling distribution.

        Returns 0.0 for cold-start (fewer than 10 samples — can't estimate σ).
        """
        k = self._key(learner, signal)
        with self._lock:
            st = self._state.setdefault(k, _LearnerState())
            st.samples.append(float(value))
            if len(st.samples) < 10:
                return 0.0
            mean = sum(st.samples) / len(st.samples)
            var = sum((x - mean) ** 2 for x in st.samples) / len(st.samples)
            std = math.sqrt(var) if var > 0 else 1e-9
            return (float(value) - mean) / std

    def should_accept(self, learner: str, signal: str, value: float) -> bool:
        """True iff the caller should consume this signal."""
        z = self.observe(learner, signal, value)
        az = abs(z)
        k = self._key(learner, signal)
        with self._lock:
            st = self._state.setdefault(k, _LearnerState())
            if az >= Z_SCORE_REJECT:
                st.consecutive_breaches += 1
                if st.consecutive_breaches >= CONSECUTIVE_TO_QUARANTINE:
                    self._quarantine_locked(learner, signal)
                return False
            if az <= Z_SCORE_ACCEPT:
                st.consecutive_breaches = 0
                if st.quarantined:
                    self._unquarantine_locked(learner, signal)
            return not st.quarantined

    def is_quarantined(self, learner: str) -> bool:
        with self._lock:
            return any(s.quarantined for k, s in self._state.items() if k.startswith(f"{learner}:"))

    def quarantined_learners(self) -> Dict[str, bool]:
        with self._lock:
            out: Dict[str, bool] = {}
            for k, s in self._state.items():
                lname = k.split(":", 1)[0]
                out[lname] = out.get(lname, False) or s.quarantined
            return out

    def _quarantine_locked(self, learner: str, signal: str) -> None:
        st = self._state[self._key(learner, signal)]
        if not st.quarantined:
            st.quarantined = True
            logger.warning("[SAFETY] quarantined %s:%s after %d breaches",
                           learner, signal, st.consecutive_breaches)
            self._emit_state(learner, True)

    def _unquarantine_locked(self, learner: str, signal: str) -> None:
        st = self._state[self._key(learner, signal)]
        if st.quarantined:
            st.quarantined = False
            logger.info("[SAFETY] released quarantine for %s:%s", learner, signal)
            self._emit_state(learner, False)

    @staticmethod
    def _emit_state(learner: str, quarantined: bool) -> None:
        try:
            from src.orchestration.metrics import record_quarantine_state
            record_quarantine_state(learner=learner, quarantined=quarantined)
        except Exception:
            pass


# ── 4. Authority compliance gate ───────────────────────────────────────

def gate_credit_by_authority(
    credit: Dict[str, float],
    authority_violations: list,
    veto_components: tuple = ("authority_guardian", "loss_prevention"),
) -> Dict[str, float]:
    """If authority rules were violated at decision time, non-veto components
    get ZERO credit — even if the trade happened to profit (§7.3 anti-luck).
    """
    if not authority_violations:
        return dict(credit)
    veto_total = sum(credit.get(k, 0.0) for k in veto_components)
    if veto_total <= 0:
        # Degenerate case — return uniform across veto list so downstream
        # math still sees a valid distribution.
        return {k: 1.0 / max(1, len(veto_components)) for k in veto_components}
    return {k: credit.get(k, 0.0) / veto_total for k in veto_components}
