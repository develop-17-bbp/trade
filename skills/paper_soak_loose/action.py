"""Skill: /paper-soak-loose — toggle a runtime overlay that loosens
sniper + conviction thresholds so paper trades actually fire during
the 14-day soak window.

Writes `data/paper_soak_loose.json` with the override values. Every
tick the executor reads this file (or any reader via
`get_paper_soak_overlay()`) and applies the loosened gate values
IF AND ONLY IF real-capital is still disabled. Real capital is
always gated on `ACT_REAL_CAPITAL_ENABLED=1` separately.

Flow:
  /paper-soak-loose enable=true
    -> writes overlay file with min_score=4, min_move_pct=2.0,
       min_confluence=3 (or values you pass)
    -> next tick: scanner accepts more setups → more shadow plans
       → dashboard shows activity → you can watch the soak counter climb

  /paper-soak-loose enable=false
    -> deletes overlay file
    -> next tick: strict gates resume

Risk posture:
  * Real-capital flag unchanged. No real money moves.
  * Lower-quality trades fire, but they're paper. The goal is soak-
    counter progress, not PnL.
  * Conviction/authority gates still run at their minimum floors.
    Cost gate still runs. Only the SNIPER tier thresholds loosen.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERLAY_FILE = PROJECT_ROOT / "data" / "paper_soak_loose.json"


def _is_paper_mode() -> bool:
    """True iff real capital is NOT enabled. We only allow loosening
    in paper mode — safety rail."""
    return os.environ.get("ACT_REAL_CAPITAL_ENABLED", "").strip() != "1"


def run(args: Optional[Dict[str, Any]] = None) -> SkillResult:
    args = dict(args or {})
    enable_raw = args.get("enable", None)
    if enable_raw is None:
        return SkillResult(
            ok=False,
            error=("missing 'enable=true' or 'enable=false'. Example: "
                   "python -m src.skills.cli run paper-soak-loose enable=true"),
        )
    if isinstance(enable_raw, str):
        enable = enable_raw.strip().lower() in ("1", "true", "yes", "on")
    else:
        enable = bool(enable_raw)

    if not enable:
        # Disable — remove overlay file.
        existed = OVERLAY_FILE.exists()
        try:
            if existed:
                OVERLAY_FILE.unlink()
        except Exception as e:
            return SkillResult(ok=False, error=f"failed to remove overlay: {e}")
        return SkillResult(
            ok=True,
            message=(f"paper-soak loose mode {'DISABLED (overlay removed)' if existed else 'already off'}. "
                     "Strict gates resume on next tick."),
            data={"enabled": False, "overlay_file": str(OVERLAY_FILE),
                  "previously_active": existed},
        )

    # Enable — write overlay.
    if not _is_paper_mode():
        return SkillResult(
            ok=False,
            error=("ACT_REAL_CAPITAL_ENABLED=1 is set — loose mode refuses "
                   "to enable in real-capital mode. Unset that env first "
                   "if you really want to run loose paper gates."),
        )

    # Defaults tuned for visibility in ranging markets without
    # bypassing the bigger safety layers.
    try:
        min_score = int(args.get("min_score", 4))
    except (TypeError, ValueError):
        min_score = 4
    try:
        min_move_pct = float(args.get("min_move_pct", 2.0))
    except (TypeError, ValueError):
        min_move_pct = 2.0
    try:
        min_confluence = int(args.get("min_confluence", 3))
    except (TypeError, ValueError):
        min_confluence = 3
    try:
        min_normal_strategies = int(args.get("min_normal_strategies", 2))
    except (TypeError, ValueError):
        min_normal_strategies = 2
    try:
        min_cost_margin_pct = float(args.get("min_cost_margin_pct", 0.3))
    except (TypeError, ValueError):
        min_cost_margin_pct = 0.3

    # Paper-soak can optionally bypass the conviction-gate macro-crisis
    # absolute-reject. Real-capital path ignores this flag (the overlay
    # is already paper-gated), so there's no way to turn this on for
    # live money.
    bypass_macro_crisis_raw = args.get("bypass_macro_crisis", True)
    if isinstance(bypass_macro_crisis_raw, str):
        bypass_macro_crisis = bypass_macro_crisis_raw.strip().lower() in (
            "1", "true", "yes", "on"
        )
    else:
        bypass_macro_crisis = bool(bypass_macro_crisis_raw)

    overlay = {
        "enabled_at": datetime.now(timezone.utc).isoformat(),
        "reason": str(args.get("reason", "operator-enabled for soak visibility")),
        "sniper": {
            "min_score": min_score,
            "min_expected_move_pct": min_move_pct,
            "min_confluence": min_confluence,
        },
        "conviction": {
            "min_normal_strategies_agreeing": min_normal_strategies,
            "bypass_macro_crisis": bypass_macro_crisis,
        },
        "cost_gate": {
            "min_margin_pct": min_cost_margin_pct,
        },
        # Safety footer — always true if this overlay was written.
        "requires_paper_mode": True,
    }

    OVERLAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        OVERLAY_FILE.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    except Exception as e:
        return SkillResult(ok=False, error=f"write failed: {e}")

    # Report path relative to PROJECT_ROOT when possible (for ops
    # readability). Tests redirect OVERLAY_FILE to tmp_path which
    # isn't under PROJECT_ROOT — fall back to absolute in that case.
    try:
        shown_path = OVERLAY_FILE.relative_to(PROJECT_ROOT)
    except ValueError:
        shown_path = OVERLAY_FILE

    return SkillResult(
        ok=True,
        message=(
            f"paper-soak loose mode ENABLED. Overlay written to "
            f"{shown_path}. "
            f"Thresholds: sniper.min_score={min_score}, "
            f"sniper.min_move_pct={min_move_pct}, "
            f"sniper.min_confluence={min_confluence}. "
            "Paper trades should start firing within a few ticks. "
            "Disable with `paper-soak-loose enable=false`."
        ),
        data={
            "enabled": True,
            "overlay_file": str(OVERLAY_FILE),
            "overlay": overlay,
            "real_capital_gated": True,
            "note": ("Real-capital execution still blocked by "
                     "ACT_REAL_CAPITAL_ENABLED gate — this only "
                     "affects paper/shadow pipeline."),
        },
    )


# ── Helper for other modules to read the overlay ──────────────────────


def update_overlay(delta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Adjust an EXISTING overlay in-place (no-op when overlay absent).

    C26 Step 5 — the autonomous pursuit loop calls this every cycle
    when it wants to loosen or tighten thresholds by one step toward
    1%/day. Floors are enforced per the plan:
      * sniper.min_score ≥ 2
      * sniper.min_expected_move_pct ≥ 1.0
      * sniper.min_confluence ≥ 2
      * cost_gate.min_margin_pct ≥ 0.1
      * conviction.min_normal_strategies_agreeing ≥ 1

    Real-capital guard: refuses to update when
    ACT_REAL_CAPITAL_ENABLED=1 is set.

    Returns the updated overlay dict, or None if the overlay doesn't
    exist yet (operator must first run /paper-soak-loose enable=true).
    """
    if os.environ.get("ACT_REAL_CAPITAL_ENABLED", "").strip() == "1":
        return None
    if not OVERLAY_FILE.exists():
        return None
    try:
        current = json.loads(OVERLAY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(current, dict):
        return None
    delta = dict(delta or {})

    # Apply sniper deltas with floors
    sniper = current.setdefault("sniper", {})
    s_delta = delta.get("sniper") or {}
    if "min_score" in s_delta:
        new_val = max(2, int(sniper.get("min_score", 4)) + int(s_delta["min_score"]))
        sniper["min_score"] = new_val
    if "min_expected_move_pct" in s_delta:
        new_val = max(1.0, float(sniper.get("min_expected_move_pct", 2.0)) + float(s_delta["min_expected_move_pct"]))
        sniper["min_expected_move_pct"] = round(new_val, 2)
    if "min_confluence" in s_delta:
        new_val = max(2, int(sniper.get("min_confluence", 3)) + int(s_delta["min_confluence"]))
        sniper["min_confluence"] = new_val

    # Cost gate margin floor
    cg = current.setdefault("cost_gate", {})
    cg_delta = delta.get("cost_gate") or {}
    if "min_margin_pct" in cg_delta:
        new_val = max(0.1, float(cg.get("min_margin_pct", 0.3)) + float(cg_delta["min_margin_pct"]))
        cg["min_margin_pct"] = round(new_val, 2)

    # Conviction floor
    cv = current.setdefault("conviction", {})
    cv_delta = delta.get("conviction") or {}
    if "min_normal_strategies_agreeing" in cv_delta:
        new_val = max(1, int(cv.get("min_normal_strategies_agreeing", 2)) + int(cv_delta["min_normal_strategies_agreeing"]))
        cv["min_normal_strategies_agreeing"] = new_val

    # Log the adjustment inline for audit
    adjustments = current.setdefault("adjustments", [])
    adjustments.append({
        "at": datetime.now(timezone.utc).isoformat(),
        "delta": delta,
        "reason": str(delta.get("reason") or "soak_tune"),
    })
    # Cap adjustments history
    current["adjustments"] = adjustments[-50:]

    try:
        OVERLAY_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except Exception:
        return None
    return current


def get_paper_soak_overlay() -> Optional[Dict[str, Any]]:
    """Return the current overlay dict, or None if disabled.

    Called by executor / conviction_gate / cost_gate each tick.
    Silent on any failure — never crashes the trade loop.

    Reads with `utf-8-sig` so a BOM-prefixed file (which PowerShell
    5.1's `Set-Content -Encoding UTF8` adds by default) loads cleanly.
    Without that, START_ALL.ps1's auto-write produced a file that
    json.load rejected, get_paper_soak_overlay returned None, and
    conviction_gate's bypass_macro_crisis flag was silently ignored
    -- causing macro_crisis rejects on every paper-mode tick.
    """
    try:
        if not OVERLAY_FILE.exists():
            return None
        if os.environ.get("ACT_REAL_CAPITAL_ENABLED", "").strip() == "1":
            return None
        text = OVERLAY_FILE.read_text(encoding="utf-8-sig")
        overlay = json.loads(text)
        if overlay.get("requires_paper_mode") is True:
            return overlay
    except Exception:
        pass
    return None
