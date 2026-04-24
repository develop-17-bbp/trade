"""Credit assigner — Phase 4.5a (Learning Mesh §3).

Turns a realized PnL into per-component credit weights so the four learners
(L1 RL, L7 LoRA, L9 genetic, Confidence Calibrator + Authority Guardian)
each get a reward proportional to their actual contribution, not a flat
share of every trade's outcome.

Algorithm — rolling regression (Plan §3.2 method c):
    For each component k, let x_k ∈ {-1, 0, +1} (short / flat / long vote,
    scaled by confidence). Over the last N=500 closed trades:

        realized_pnl ≈ Σ_k  β_k · x_k + ε

    Ridge-fit the β vector, normalize |β_k| / Σ|β| → credit weight. Clip
    any individual weight to [0.02, 0.60] so one dominant component
    can't monopolize credit during a strong regime.

Cold start (Plan §3.4):
    First 100 trades — not enough data. Use a Dirichlet prior weighted
    toward `authority_guardian` (alpha=5.0) with everyone else at alpha=1.0.
    Between trades 100-500, anneal linearly from prior → learned.

Soft dep: if scikit-learn isn't installed we fall back to a uniform split.
That shouldn't happen in practice — it's a hard requirement in Phase 4.5a.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sklearn.linear_model import Ridge
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

_COLD_CUTOFF = 100
_WARM_CUTOFF = 500
_MIN_WEIGHT = 0.02
_MAX_WEIGHT = 0.60

# Default Authority-dominant Dirichlet prior.
_DEFAULT_PRIOR_ALPHAS: Dict[str, float] = {
    "authority_guardian": 5.0,
    "loss_prevention": 3.0,
    "l1_rl": 1.0,
    "l7_lora": 1.0,
    "l9_genetic": 1.0,
    "confidence_calibrator": 1.0,
}


@dataclass
class TradeRow:
    """One closed-trade record used by the regression."""

    component_actions: Dict[str, float]   # {component -> signed confidence in [-1, 1]}
    realized_pnl: float


@dataclass
class CreditAssigner:
    """Rolling-500 credit model. Call `assign()` after each closed trade."""

    prior_alphas: Dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_PRIOR_ALPHAS))
    window: int = _WARM_CUTOFF
    _history: List[TradeRow] = field(default_factory=list, init=False)
    _last_weights: Dict[str, float] = field(default_factory=dict, init=False)
    _last_r2: float = field(default=0.0, init=False)

    def record(self, row: TradeRow) -> None:
        """Remember a trade. Drops the oldest if we exceed `window`."""
        self._history.append(row)
        if len(self._history) > self.window:
            self._history = self._history[-self.window :]

    # ── Core API ──────────────────────────────────────────────────────

    def assign(self) -> Dict[str, float]:
        """Return the current credit_allocation dict for the Experience envelope.

        Blends prior ↔ learned based on history length. Always returns a
        properly-normalized distribution that sums to 1.0. Applies a small
        DSR-derived bonus per component on top of the ridge fit so a
        component running hot on risk-adjusted returns gets rewarded
        even if raw PnL regression hasn't converged yet (C18a).
        """
        learned = self._learned_weights()
        prior = self._prior_weights()
        if len(self._history) <= _COLD_CUTOFF:
            out = prior
        elif len(self._history) >= _WARM_CUTOFF:
            out = learned
        else:
            frac = (len(self._history) - _COLD_CUTOFF) / (_WARM_CUTOFF - _COLD_CUTOFF)
            out = self._blend(prior, learned, frac)
        out = self._apply_dsr_bonus(out)
        out = self._clip_and_renormalize(out)
        self._last_weights = out
        self._emit_metrics()
        return out

    @staticmethod
    def _apply_dsr_bonus(w: Dict[str, float]) -> Dict[str, float]:
        """Additive DSR bonus per component. Best-effort — missing tracker
        collapses to a no-op."""
        try:
            from src.learning.reward import dsr_to_credit_bonus, get_tracker
            tracker = get_tracker()
        except Exception:
            return w
        out = {}
        for k, v in w.items():
            try:
                bonus = dsr_to_credit_bonus(tracker.get(k))
            except Exception:
                bonus = 0.0
            out[k] = max(0.0, v + bonus)
        return out

    def weights(self) -> Dict[str, float]:
        """Most recent credit_allocation (or an empty dict if never assigned)."""
        return dict(self._last_weights)

    def last_r2(self) -> float:
        return self._last_r2

    # ── Internals ─────────────────────────────────────────────────────

    def _prior_weights(self) -> Dict[str, float]:
        total = sum(self.prior_alphas.values()) or 1.0
        return {k: v / total for k, v in self.prior_alphas.items()}

    def _learned_weights(self) -> Dict[str, float]:
        if not _HAS_SKLEARN or not self._history:
            return self._prior_weights()

        components = sorted({k for r in self._history for k in r.component_actions})
        if not components:
            return self._prior_weights()

        X = np.array([
            [r.component_actions.get(k, 0.0) for k in components]
            for r in self._history
        ], dtype=float)
        y = np.array([r.realized_pnl for r in self._history], dtype=float)

        try:
            model = Ridge(alpha=1.0, fit_intercept=True)
            model.fit(X, y)
            self._last_r2 = float(model.score(X, y))
            beta = np.abs(model.coef_)
            s = beta.sum()
            if s <= 1e-9:
                return self._prior_weights()
            return {k: float(b / s) for k, b in zip(components, beta)}
        except Exception as e:
            logger.warning("credit regression failed: %s — falling back to prior", e)
            return self._prior_weights()

    @staticmethod
    def _blend(a: Dict[str, float], b: Dict[str, float], frac_b: float) -> Dict[str, float]:
        keys = set(a) | set(b)
        return {k: (1 - frac_b) * a.get(k, 0.0) + frac_b * b.get(k, 0.0) for k in keys}

    @staticmethod
    def _clip_and_renormalize(w: Dict[str, float]) -> Dict[str, float]:
        if not w:
            return w
        clipped = {k: min(_MAX_WEIGHT, max(_MIN_WEIGHT, v)) for k, v in w.items()}
        s = sum(clipped.values()) or 1.0
        return {k: v / s for k, v in clipped.items()}

    def _emit_metrics(self) -> None:
        try:
            from src.orchestration.metrics import record_credit_allocation, record_credit_r2
            for k, v in self._last_weights.items():
                record_credit_allocation(component=k, weight=v)
            record_credit_r2(self._last_r2)
        except Exception:
            pass
