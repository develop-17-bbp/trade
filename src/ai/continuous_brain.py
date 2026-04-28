"""Continuous-thinking brain daemon.

The brain reasons every ~10s on tick cadence. Between ticks the GPU
is mostly idle (after the analyst call returns, before the next tick
fires). This daemon uses that idle time to:

  1. Pre-compute scenario plans — "what would I do if BTC drops 1%
     in the next 30s? climbs 1%? regime flips to CRISIS?" Cached in
     tick_state for the next tick to read.
  2. Run periodic self-checks — "am I in a regime my recent
     critiques say to avoid?", "is the gap_to_1pct closing or
     widening on my current trajectory?"

When the next tick fires and conditions match a pre-computed
scenario, the brain has its answer ready instead of computing fresh.

Anti-noise design:
  * Default OFF (ACT_CONTINUOUS_BRAIN=1 to enable)
  * Cache invalidates after 30s — no stale scenarios
  * Cache caps at 5 scenarios per asset (no unbounded growth)
  * Pre-computed scenarios are HINTS in tick_state, not authoritative
    decisions — the next tick's analyst still reasons from scratch
    but with a cached suggestion to consider
  * Doesn't make actual LLM calls in the daemon; uses fast rule-based
    scenario evaluation. (LLM calls are expensive and would defeat
    the "use idle compute" purpose if invoked too often.)

Activation:
  ACT_CONTINUOUS_BRAIN unset / "0"  → daemon dormant
  ACT_CONTINUOUS_BRAIN = "1"        → daemon runs scenario pre-compute
                                       on configured interval
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 5.0   # how often to refresh scenarios
DEFAULT_CACHE_TTL_S = 30.0
DEFAULT_MAX_SCENARIOS_PER_ASSET = 5


@dataclass
class Scenario:
    name: str                  # "drop_1pct" | "climb_1pct" | "regime_crisis" etc.
    trigger_condition: str     # human-readable
    suggested_action: str      # "EXIT_ALL" | "ADD_50%" | "HOLD" | "PARTIAL_25%"
    rationale: str = ""
    computed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trigger_condition": self.trigger_condition,
            "suggested_action": self.suggested_action,
            "rationale": self.rationale[:200],
            "computed_at": self.computed_at,
            "age_s": round(time.time() - self.computed_at, 1),
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_CONTINUOUS_BRAIN") or "").strip().lower()
    return val in ("1", "true", "on")


class ContinuousBrain:
    """Daemon that pre-computes scenarios per asset during quiet ticks."""

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        interval_s: float = DEFAULT_INTERVAL_S,
        cache_ttl_s: float = DEFAULT_CACHE_TTL_S,
    ) -> None:
        self.assets = list(assets or ["BTC", "ETH"])
        self.interval_s = max(2.0, interval_s)
        self.cache_ttl_s = cache_ttl_s
        self._cache: Dict[str, List[Scenario]] = {}
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._n_refreshes = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not is_enabled():
            logger.info("[CONTINUOUS-BRAIN] disabled (set "
                        "ACT_CONTINUOUS_BRAIN=1 to enable)")
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, name="ContinuousBrain", daemon=True,
        )
        self._thread.start()
        logger.info("[CONTINUOUS-BRAIN] daemon started "
                    "(interval=%.1fs, cache_ttl=%.1fs)",
                    self.interval_s, self.cache_ttl_s)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)

    def get_cache(self, asset: str) -> List[Dict[str, Any]]:
        """Return current pre-computed scenarios for an asset, with
        stale entries pruned. Used by the brain on next tick to read
        what the daemon already thought through."""
        now = time.time()
        scenarios = self._cache.get(asset, [])
        fresh = [s for s in scenarios if now - s.computed_at <= self.cache_ttl_s]
        self._cache[asset] = fresh
        return [s.to_dict() for s in fresh]

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": is_enabled(),
            "n_refreshes": self._n_refreshes,
            "scenarios_cached_per_asset": {
                a: len(self.get_cache(a)) for a in self.assets
            },
        }

    # ── Internals ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                for asset in self.assets:
                    self._refresh_asset(asset)
                self._n_refreshes += 1
            except Exception as e:
                logger.debug("[CONTINUOUS-BRAIN] loop error: %s", e)
            self._stop_evt.wait(self.interval_s)

    def _refresh_asset(self, asset: str) -> None:
        """Compute scenarios from current tick_state and cache them.

        Rule-based, no LLM calls — fast enough to run every 5s without
        consuming the analyst budget. The brain on next tick reads the
        cache via continuous_brain.get_cache(asset) and uses scenarios
        as HINTS, not authoritative decisions.
        """
        try:
            from src.ai import tick_state as _ts
        except Exception:
            return
        snap = _ts.get(asset) or {}
        if not snap:
            return

        scenarios: List[Scenario] = []
        now = time.time()

        # Scenario 1: BTC drops 1% — what would the body's auto-ratchet do?
        ratchet_label = str(snap.get("ratchet_label", "NONE"))
        ratchet_pnl = float(snap.get("ratchet_current_pnl_pct", 0.0))
        n_open = int(snap.get("open_positions_same_asset", 0))
        if n_open > 0:
            if ratchet_pnl < 0.5:
                action = "EXIT_AT_RISK_LIMIT"
                rat = ("ratchet not yet at BREAKEVEN; -1% would push "
                       "average position into hard-stop territory")
            elif ratchet_label == "BREAKEVEN":
                action = "HOLD_RATCHET_PROTECTS"
                rat = "BREAKEVEN ratchet active; SL = entry"
            else:
                action = "PARTIAL_50%"
                rat = ("ratchet locked partial profit; -1% takes back "
                       "more than the lock — partial exit reasonable")
            scenarios.append(Scenario(
                name="drop_1pct", trigger_condition="if BTC drops 1% in 30s",
                suggested_action=action, rationale=rat, computed_at=now,
            ))

        # Scenario 2: regime flip to CRISIS
        regime = str(snap.get("regime", "")).upper()
        if regime != "CRISIS":
            scenarios.append(Scenario(
                name="regime_crisis",
                trigger_condition="if HMM flips to CRISIS",
                suggested_action="CLOSE_ALL_PRESERVE_CAPITAL",
                rationale=("CRISIS regime requires capital preservation "
                           "over alpha-seeking; close all opens immediately"),
                computed_at=now,
            ))

        # Scenario 3: gap_to_1pct trajectory
        gap = float(snap.get("gap_to_1pct", 0.0))
        today_total = float(snap.get("today_pct_total", 0.0))
        n_trades = int(snap.get("today_trades", 0))
        if gap > 0.5 and n_trades < 3:
            scenarios.append(Scenario(
                name="gap_widening",
                trigger_condition=f"current today_pct={today_total:+.2f}%, gap_to_1pct={gap:+.2f}%, only {n_trades} trades",
                suggested_action="HUNT_HIGHER_CONVICTION_SETUPS",
                rationale=("under-trading the goal; bias toward sniper-tier "
                           "confluence to make up the gap before EOD"),
                computed_at=now,
            ))

        # Scenario 4: news catalyst proximity (if last_catalyst timestamp recent)
        last_cat_ts = float(snap.get("last_catalyst_ts", 0.0))
        if last_cat_ts > 0 and now - last_cat_ts < 300:
            scenarios.append(Scenario(
                name="recent_catalyst",
                trigger_condition=f"catalyst {now-last_cat_ts:.0f}s ago: {snap.get('last_catalyst', '?')}",
                suggested_action="MONITOR_FOR_CONTINUATION",
                rationale=("recent catalyst — direction-confirming follow-through "
                           "on next 1-3 candles is a tradeable signal"),
                computed_at=now,
            ))

        # Scenario 5: factor synthesis bias state — single source of truth
        # from factor_synthesis module. Same view as catalyst listener +
        # agentic loop.
        if "factor_bias_score" in snap:
            bias = float(snap.get("factor_bias_score", 0.0))
            regime = str(snap.get("factor_regime", "?"))
            action = str(snap.get("factor_action", "skip"))
            if bias > 0.5:
                scenarios.append(Scenario(
                    name="factor_synthesis_strong_long",
                    trigger_condition=f"bias_score=+{bias:.2f} regime={regime}",
                    suggested_action="SUBMIT_LONG_IF_GATES_PASS",
                    rationale=(f"6-factor synthesis says STRONG LONG ({bias:+.2f}). "
                               "Macro + cross-asset + order-flow aligned bullish. "
                               "Verify slippage + sizing then submit_trade_plan."),
                    computed_at=now,
                ))
            elif bias < -0.5:
                scenarios.append(Scenario(
                    name="factor_synthesis_strong_short",
                    trigger_condition=f"bias_score={bias:+.2f} regime={regime}",
                    suggested_action="NO_NEW_LONGS_CONSIDER_THESIS_BROKEN_CLOSE",
                    rationale=(f"6-factor synthesis says STRONG SHORT ({bias:+.2f}). "
                               "Robinhood is longs-only — skip all new longs. If "
                               "ACT_HOLD_UNTIL_PROFIT not set, consider thesis_broken "
                               "close on positions whose macro thesis is invalidated."),
                    computed_at=now,
                ))

        # Cap and write
        scenarios = scenarios[:DEFAULT_MAX_SCENARIOS_PER_ASSET]
        self._cache[asset] = scenarios

        # Surface a summary line in tick_state so brain sees it next tick
        try:
            summary = " | ".join(
                f"{s.name}->{s.suggested_action}" for s in scenarios
            )
            _ts.update(asset, continuous_brain_scenarios=summary[:300])
        except Exception:
            pass


# Singleton accessor
_singleton: Optional[ContinuousBrain] = None


def get_brain() -> ContinuousBrain:
    global _singleton
    if _singleton is None:
        _singleton = ContinuousBrain()
    return _singleton
