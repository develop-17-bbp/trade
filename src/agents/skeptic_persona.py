"""Adversarial skeptic persona (FinMem-style character design).

Reads the consensus decision *before* TradePlan submission and
generates a structured contrarian argument. The skeptic's veto
weight (0.0-1.0) is configurable; when the weighted veto crosses a
threshold, the analyst is informed and the plan is downgraded
(SKIP, smaller size, or held back).

Why: research (FinMem) shows performance lift from a deliberate
"contrarian voice" stress-testing high-conviction plans before
submission. Specifically catches over-confident plans where the
crowd of agreeing agents missed a structural risk.

Anti-noise design:
  * Skeptic ONLY speaks when consensus confidence > 0.7 (skipping
    already-uncertain plans avoids piling noise on noise)
  * Output is one structured paragraph (not free-form rant), capped
    at 400 chars
  * Veto weight is tunable via env (default 0.0 = no effect)
  * SKIP-rate impact is monitored so the operator can dial weight up
    gradually (0.0 → 0.1 → 0.2 → 0.3) over a soak window

Activation:
  ACT_SKEPTIC unset / "0"  → module dormant
  ACT_SKEPTIC = "1"        → skeptic generates argument; veto weight
                              from ACT_SKEPTIC_WEIGHT (default 0.0)
  ACT_SKEPTIC_WEIGHT       → "0.0" to "1.0"; default "0.0" so even
                              when enabled the skeptic only ADVISES
                              until the operator dials weight up
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SHADOW_LOG_PATH = "logs/skeptic_shadow.jsonl"
DEFAULT_MIN_CONSENSUS_CONFIDENCE = 0.7
DEFAULT_VETO_THRESHOLD = 0.6


@dataclass
class SkepticVerdict:
    """One structured contrarian argument."""
    asset: str
    consensus_direction: str
    consensus_confidence: float
    skeptic_argument: str = ""        # capped at 400 chars
    counter_signals: List[str] = field(default_factory=list)
    veto_strength: float = 0.0        # 0.0 (none) - 1.0 (full)
    veto_applied: bool = False        # True when weight × strength > threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "consensus_direction": self.consensus_direction,
            "consensus_confidence": round(float(self.consensus_confidence), 3),
            "skeptic_argument": self.skeptic_argument[:400],
            "counter_signals": self.counter_signals[:10],
            "veto_strength": round(float(self.veto_strength), 3),
            "veto_applied": bool(self.veto_applied),
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_SKEPTIC") or "").strip().lower()
    return val in ("1", "true", "on", "shadow")


def is_authoritative() -> bool:
    """The skeptic only affects TradePlan output when ACT_SKEPTIC=1
    AND ACT_SKEPTIC_WEIGHT > 0.0. Shadow mode logs only."""
    val = (os.environ.get("ACT_SKEPTIC") or "").strip().lower()
    if val not in ("1", "true", "on"):
        return False
    try:
        weight = float(os.environ.get("ACT_SKEPTIC_WEIGHT") or "0.0")
        return weight > 0.0
    except Exception:
        return False


def get_veto_weight() -> float:
    """Veto weight from env, clamped [0.0, 1.0]. Default 0.0."""
    try:
        w = float(os.environ.get("ACT_SKEPTIC_WEIGHT") or "0.0")
        return max(0.0, min(1.0, w))
    except Exception:
        return 0.0


def _gather_counter_signals(asset: str, consensus_direction: str,
                            tick_snap: Optional[Dict[str, Any]]) -> List[str]:
    """Pull signals from tick_state that contradict the consensus.
    Returns short labels, not raw values, to avoid prompt bloat."""
    if not tick_snap:
        return []
    counters: List[str] = []
    cd = (consensus_direction or "").upper()
    long_consensus = cd in ("LONG", "BUY")

    # Macro bias
    macro = tick_snap.get("macro_bias")
    if isinstance(macro, (int, float)):
        if long_consensus and macro < -0.3:
            counters.append(f"macro_bias_bearish={macro:+.2f}")
        elif (not long_consensus) and macro > 0.3:
            counters.append(f"macro_bias_bullish={macro:+.2f}")

    # Hurst regime mean-reverting against trend consensus
    hurst_v = tick_snap.get("hurst_value", 0.5)
    if isinstance(hurst_v, (int, float)) and float(hurst_v) < 0.45:
        counters.append(f"hurst_mean_reverting={hurst_v:.2f}")

    # HMM CRISIS regardless of direction
    regime = str(tick_snap.get("regime", "")).upper()
    if regime == "CRISIS":
        counters.append("hmm_regime=CRISIS")

    # ML ensemble disagreement
    ml_meta_dir = tick_snap.get("ml_meta_dir")
    if isinstance(ml_meta_dir, (int, float)):
        if long_consensus and float(ml_meta_dir) < 0:
            counters.append(f"ml_meta_dir={int(ml_meta_dir)}")
        elif (not long_consensus) and float(ml_meta_dir) > 0:
            counters.append(f"ml_meta_dir={int(ml_meta_dir)}")

    # VPIN toxic flow
    if tick_snap.get("vpin_toxic"):
        counters.append("vpin_toxic_flow")

    # Spread vs expected move (if expected move would barely clear spread)
    spread = float(tick_snap.get("spread_pct", 0.0) or 0.0)
    if spread > 1.5 and consensus_direction in ("LONG", "BUY", "SHORT", "SELL"):
        counters.append(f"high_spread={spread:.2f}%")

    # Already deeply concentrated on same direction
    n_open = int(tick_snap.get("open_positions_same_asset", 0) or 0)
    if n_open >= 3 and long_consensus:
        counters.append(f"already_long_{n_open}x")

    # Recent exits losing on same asset
    exits = tick_snap.get("recent_exits") or []
    if isinstance(exits, list):
        recent_losers = sum(
            1 for e in exits[:3]
            if isinstance(e, dict) and float(e.get("pnl_net_pct", 0)) < 0
        )
        if recent_losers >= 2:
            counters.append(f"{recent_losers}_recent_losers")

    return counters[:10]


def evaluate(asset: str, consensus_direction: str, consensus_confidence: float,
             tick_snap: Optional[Dict[str, Any]] = None) -> Optional[SkepticVerdict]:
    """Generate a contrarian verdict OR return None when skeptic should
    stay quiet. Skeptic stays quiet when:
      * module disabled
      * consensus confidence < threshold (already uncertain plans)
      * consensus direction is SKIP/FLAT (no plan to argue against)
    """
    if not is_enabled():
        return None
    if str(consensus_direction).upper() in ("SKIP", "FLAT", "NONE", ""):
        return None
    if consensus_confidence < DEFAULT_MIN_CONSENSUS_CONFIDENCE:
        return None

    counters = _gather_counter_signals(asset, consensus_direction, tick_snap)
    veto_strength = min(1.0, len(counters) * 0.15)

    # Build a structured argument (deterministic — no LLM call needed
    # for this skeptic; we synthesize from tick_state contradictions).
    if counters:
        argument = (
            f"Consensus is {consensus_direction} @ conf={consensus_confidence:.2f}, "
            f"but {len(counters)} structural counter-signals are present: "
            + ", ".join(counters)
            + ". Recommend size reduction or thesis re-validation."
        )
    else:
        argument = (
            f"No structural counter-signals found. {consensus_direction} "
            f"plan at conf={consensus_confidence:.2f} appears defensible "
            "from the contrarian angle."
        )

    weight = get_veto_weight()
    veto_applied = (
        is_authoritative() and (weight * veto_strength) >= DEFAULT_VETO_THRESHOLD
    )

    return SkepticVerdict(
        asset=asset,
        consensus_direction=str(consensus_direction),
        consensus_confidence=float(consensus_confidence),
        skeptic_argument=argument[:400],
        counter_signals=counters,
        veto_strength=veto_strength,
        veto_applied=veto_applied,
    )


def format_for_brain(verdict: Optional[SkepticVerdict]) -> str:
    """Render the skeptic's verdict as a single evidence line for the
    analyst's prompt. Empty string when skeptic stayed quiet."""
    if verdict is None or not verdict.skeptic_argument:
        return ""
    veto_marker = " [VETO]" if verdict.veto_applied else ""
    return (
        f"SKEPTIC{veto_marker}: {verdict.skeptic_argument} "
        f"(strength={verdict.veto_strength:.2f}, "
        f"weight={get_veto_weight():.2f})"
    )


def log_shadow(verdict: SkepticVerdict, plan_proceeded: bool) -> None:
    """Append shadow-mode comparison: did the plan proceed despite the
    skeptic's argument? Tells the operator whether dialing weight up
    would have flipped outcomes. Never raises."""
    try:
        path = Path(SHADOW_LOG_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": time.time(),
            "verdict": verdict.to_dict(),
            "plan_proceeded": bool(plan_proceeded),
            "veto_weight": get_veto_weight(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception as e:
        logger.debug("skeptic shadow log failed: %s", e)
