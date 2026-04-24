"""Reward shaping for the learning mesh — C18a.

Raw PnL is a noisy reward signal: a single outsized win can distort a
rolling accuracy score, and raw returns don't punish volatility. ACT's
learners (credit_assigner, thompson_bandit, coevolution → RL warm-starts)
want a signal that tracks **risk-adjusted performance**, not raw PnL.

This module implements the Moody & Saffell (1998) differential Sharpe
ratio (DSR) — an online, incremental form of the Sharpe ratio that
updates with each trade and can be used as a direct reward.

    A_t  = A_{t-1} + η · (R_t - A_{t-1})           # EMA of returns
    B_t  = B_{t-1} + η · (R_t² - B_{t-1})          # EMA of squared returns
    σ²_t = B_t - A_t²                                # rolling variance
    DSR_t = ( B_{t-1}·(R_t - A_{t-1}) - ½·A_{t-1}·(R_t² - B_{t-1}) )
            / (B_{t-1} - A_{t-1}²)^{3/2}

With η ≈ 1/N the EMA has effective window N. η=0.01 → ≈100-trade window;
η=0.05 → ≈20-trade window. DSR is unitless and bounded (practically
within roughly [-5, +5] for realistic return streams), which makes it
a clean reward signal for bandits and RL alike.

Design:
  * **Pure-Python + math** — no numpy, no sklearn. Safe to import from
    any learning module.
  * **Thread-safe** — one lock per `DSRState` instance so concurrent
    outcome-update threads can't corrupt the EMA.
  * **Namespaced state** — `DSRTracker` keeps a DSR per (component,
    asset) key so the brain sees {"scanner:BTC": 0.42, ...} rather
    than one blurred scalar.

Not in scope:
  * Wiring into autonomous_loop — that's C18.
  * PPO/A2C-style value/advantage — ACT runs LoRA fine-tune rather than
    deep-RL policy gradients; DSR is the reward signal the bandit +
    credit_assigner consume, not a drop-in for V(s).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Default EMA rate. 0.01 ≈ 100-trade effective window — responsive enough
# to see regime shifts, slow enough not to flip on one trade.
DEFAULT_ETA = 0.01

# Minimum variance before DSR is meaningful. Below this we return 0.0 so
# the first handful of trades don't emit infinite-DSR garbage.
_MIN_VAR_FOR_DSR = 1e-9


@dataclass
class DSRState:
    """Per-stream DSR state (A = EMA(R), B = EMA(R²))."""

    eta: float = DEFAULT_ETA
    A: float = 0.0
    B: float = 0.0
    n: int = 0
    last_dsr: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, return_t: float) -> float:
        """Process one realized return; return the DSR for this step.

        `return_t` is a fractional return (e.g. pnl_pct / 100 or just
        pnl_pct — either is fine as long as the scale is consistent).
        Bounded defensively: gigantic returns from a data bug would
        destabilize A/B.
        """
        r = _clip(return_t, -10.0, 10.0)
        with self._lock:
            prev_A = self.A
            prev_B = self.B
            denom = prev_B - prev_A * prev_A

            # Update EMAs first so future calls see fresh state.
            self.A = prev_A + self.eta * (r - prev_A)
            self.B = prev_B + self.eta * (r * r - prev_B)
            self.n += 1

            # Not enough spread yet to compute a meaningful DSR.
            if denom < _MIN_VAR_FOR_DSR or self.n < 3:
                self.last_dsr = 0.0
                return 0.0

            num = prev_B * (r - prev_A) - 0.5 * prev_A * (r * r - prev_B)
            try:
                dsr = num / math.pow(denom, 1.5)
            except (ValueError, ZeroDivisionError):
                dsr = 0.0
            # Clip to a sane band — tails from near-zero variance are
            # the only way DSR explodes, and downstream consumers
            # (thompson bonus, credit weights) don't need > 5σ.
            dsr = _clip(dsr, -5.0, 5.0)
            self.last_dsr = dsr
            return dsr

    def current(self) -> float:
        with self._lock:
            return self.last_dsr

    def snapshot(self) -> Dict[str, float]:
        """Debug/telemetry view — never mutate from the return value."""
        with self._lock:
            var = max(0.0, self.B - self.A * self.A)
            return {
                "eta": self.eta,
                "A": self.A,
                "B": self.B,
                "var": var,
                "n": float(self.n),
                "last_dsr": self.last_dsr,
            }

    def reset(self) -> None:
        """Wipe state — used by tests and after a regime/model switch."""
        with self._lock:
            self.A = 0.0
            self.B = 0.0
            self.n = 0
            self.last_dsr = 0.0


class DSRTracker:
    """Named-stream tracker — keeps one DSR state per (component, key).

    Example:
        tracker = DSRTracker(eta=0.02)
        tracker.update("scanner", 0.008, asset="BTC")     # +0.8% trade
        tracker.update("l9_genetic", -0.003, asset="ETH") # -0.3% trade
        tracker.get("scanner", asset="BTC")               # -> DSR
    """

    def __init__(self, eta: float = DEFAULT_ETA) -> None:
        self.eta = eta
        self._states: Dict[Tuple[str, str], DSRState] = {}
        self._lock = threading.Lock()

    def _key(self, component: str, asset: Optional[str]) -> Tuple[str, str]:
        return (component or "_all_", (asset or "_all_").upper())

    def state(self, component: str, asset: Optional[str] = None) -> DSRState:
        k = self._key(component, asset)
        with self._lock:
            st = self._states.get(k)
            if st is None:
                st = DSRState(eta=self.eta)
                self._states[k] = st
            return st

    def update(self, component: str, return_t: float, asset: Optional[str] = None) -> float:
        return self.state(component, asset).update(return_t)

    def get(self, component: str, asset: Optional[str] = None) -> float:
        return self.state(component, asset).current()

    def snapshot_all(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            keys = list(self._states.items())
        return {f"{c}:{a}": st.snapshot() for (c, a), st in keys}

    def reset(self) -> None:
        with self._lock:
            for st in self._states.values():
                st.reset()


# ── Module-level singleton ─────────────────────────────────────────────
#
# Most callers don't want to thread a tracker through every module. A
# singleton keeps the reward signal coherent across the autonomous loop,
# credit_assigner, thompson_bandit, and brain-to-body controller.

_TRACKER_LOCK = threading.Lock()
_TRACKER: Optional[DSRTracker] = None


def get_tracker(eta: Optional[float] = None) -> DSRTracker:
    global _TRACKER
    with _TRACKER_LOCK:
        if _TRACKER is None:
            _TRACKER = DSRTracker(eta=eta if eta is not None else DEFAULT_ETA)
        elif eta is not None and abs(_TRACKER.eta - eta) > 1e-9:
            # Caller is asking for a different eta — honor it, but keep
            # the existing per-stream history. Individual DSRState eta is
            # rechecked lazily when update() fires.
            _TRACKER.eta = eta
        return _TRACKER


def reset_singleton() -> None:
    """Test helper — drop the singleton so the next call builds a fresh one."""
    global _TRACKER
    with _TRACKER_LOCK:
        _TRACKER = None


# ── Helpers ────────────────────────────────────────────────────────────

def _clip(x: float, lo: float, hi: float) -> float:
    if x != x:  # NaN guard
        return 0.0
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def dsr_to_credit_bonus(dsr: float, cap: float = 0.15) -> float:
    """Map a DSR → bonus weight in [-cap, +cap] for credit_assigner.

    Components with sustained positive DSR get a small additive bump to
    their ridge-fit credit; sustained negative DSR gets a penalty. The
    cap (default 15%) keeps one hot streak from dominating the ridge
    weights permanently.
    """
    if dsr != dsr:  # NaN
        return 0.0
    bonus = math.tanh(dsr) * cap
    return _clip(bonus, -cap, cap)


def dsr_to_bandit_bonus(dsr: float, cap: float = 2.0) -> float:
    """Map a DSR → pseudo-alpha increment for thompson_bandit.

    A strategy running hot on risk-adjusted terms gets its Beta α bumped
    by a small amount (max +cap) so the next sample is more likely to
    pick it — even before its raw WR catches up. Negative DSR adds to β
    with the same cap so a broken strategy drifts down faster than raw
    WR alone would.
    """
    if dsr != dsr:
        return 0.0
    return _clip(math.tanh(dsr) * cap, -cap, cap)
