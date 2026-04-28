"""Event-driven catalyst preemption daemon.

The brain reasons every ~10s on tick cadence. A news flash at second
3 isn't acted on until second 13 — and on news-driven moves the
first 30s often holds 30% of the move. This daemon watches a small
set of high-impact triggers and fires an out-of-cycle brain call so
response time drops from <10s to <2s.

Triggers (any one fires):
  * news_risk_score > 0.8           — high-impact news classified
  * regime flip detected             — HMM regime changes
  * drawdown spike > 2% in 60s       — fast position bleed
  * macro event in <5min             — FOMC/CPI imminent

Anti-noise design:
  * Cooldown per asset (default 30s) — can't preempt the same asset
    repeatedly in the same window
  * Hysteresis on news_risk — must cross 0.8 from below 0.6 (avoids
    flapping at the threshold)
  * Default OFF — set ACT_CATALYST_LISTENER=1 to enable
  * Logs every preemption with the trigger reason for audit

Default OFF; existing tick-driven brain calls continue normally.
When enabled, daemon runs in background and only TRIGGERS the same
agentic_bridge.compile_agentic_plan path the tick loop uses — no
new decision logic, just a faster firing condition.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LOG_PATH = "logs/catalyst_preemptions.jsonl"
DEFAULT_COOLDOWN_S = 30.0
DEFAULT_NEWS_HIGH_THRESHOLD = 0.8
DEFAULT_NEWS_LOW_THRESHOLD = 0.6  # hysteresis
DEFAULT_DRAWDOWN_TRIGGER_PCT = 2.0
DEFAULT_DRAWDOWN_WINDOW_S = 60.0
# Factor-synthesis triggers (single source of truth from tick_state)
DEFAULT_BIAS_FLIP_THRESHOLD = 0.6   # |new - prev| crossing > this fires
DEFAULT_BIAS_HIGH_ABS = 0.5         # extreme bias triggers preemption


@dataclass
class CatalystEvent:
    asset: str
    trigger_type: str             # "news" | "regime_flip" | "drawdown" | "macro_event"
    trigger_value: float
    threshold: float
    ts: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "trigger_type": self.trigger_type,
            "trigger_value": round(float(self.trigger_value), 4),
            "threshold": round(float(self.threshold), 4),
            "ts": self.ts,
            "extra": self.extra,
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_CATALYST_LISTENER") or "").strip().lower()
    return val in ("1", "true", "on", "shadow")


def is_authoritative() -> bool:
    """Authoritative = actually triggers a brain call. Shadow = logs only."""
    val = (os.environ.get("ACT_CATALYST_LISTENER") or "").strip().lower()
    return val in ("1", "true", "on")


class CatalystListener:
    """Background daemon. Polls tick_state for trigger conditions and
    invokes a callback when one fires (default: out-of-cycle brain
    plan compile).

    Caller controls:
      * `assets`       — list to monitor (default ['BTC', 'ETH'])
      * `callback`     — invoked with CatalystEvent; default = None
                         (just logs)
      * `poll_interval_s` — how often to check (default 1.0s)
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        callback: Optional[Callable[[CatalystEvent], None]] = None,
        poll_interval_s: float = 1.0,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> None:
        self.assets = list(assets or ["BTC", "ETH"])
        self.callback = callback
        self.poll_interval_s = max(0.5, poll_interval_s)
        self.cooldown_s = cooldown_s
        self._last_preempt_at: Dict[str, float] = {}
        self._last_news_risk: Dict[str, float] = {}    # for hysteresis
        self._last_regime: Dict[str, str] = {}
        self._last_factor_regime: Dict[str, str] = {}    # macro regime flip detection
        self._last_bias_score: Dict[str, float] = {}     # bias score flip detection
        self._equity_window: List[tuple] = []           # (ts, equity)
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._n_preemptions = 0

    def start(self) -> None:
        """Spawn the daemon thread. Idempotent."""
        if self._thread and self._thread.is_alive():
            return
        if not is_enabled():
            logger.info("[CATALYST] disabled (set ACT_CATALYST_LISTENER=1 to enable)")
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._loop, name="CatalystListener", daemon=True,
        )
        self._thread.start()
        logger.info("[CATALYST] listener started "
                    "(cooldown=%.1fs, mode=%s)",
                    self.cooldown_s,
                    "AUTHORITATIVE" if is_authoritative() else "SHADOW")

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)

    def stats(self) -> Dict[str, Any]:
        return {
            "enabled": is_enabled(),
            "authoritative": is_authoritative(),
            "n_preemptions": self._n_preemptions,
            "last_preempt_per_asset": {
                a: round(time.time() - t, 1)
                for a, t in self._last_preempt_at.items()
            },
        }

    # ── Internals ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                for asset in self.assets:
                    self._check_asset(asset)
            except Exception as e:
                logger.debug("[CATALYST] loop error: %s", e)
            self._stop_evt.wait(self.poll_interval_s)

    def _check_asset(self, asset: str) -> None:
        try:
            from src.ai import tick_state as _ts
        except Exception:
            return
        snap = _ts.get(asset) or {}
        if not snap:
            return
        now = time.time()

        # Cooldown check
        last = self._last_preempt_at.get(asset, 0.0)
        if now - last < self.cooldown_s:
            return

        # Trigger 1: regime flip
        regime = str(snap.get("regime", "")).upper()
        prev_regime = self._last_regime.get(asset, "")
        if regime and prev_regime and regime != prev_regime:
            self._fire(CatalystEvent(
                asset=asset, trigger_type="regime_flip",
                trigger_value=1.0, threshold=1.0, ts=now,
                extra={"from": prev_regime, "to": regime},
            ))
        self._last_regime[asset] = regime

        # Trigger 2: news risk score with hysteresis
        news_risk = float(snap.get("news_risk_score", 0.0) or 0.0)
        prev_news = self._last_news_risk.get(asset, 0.0)
        if (news_risk >= DEFAULT_NEWS_HIGH_THRESHOLD
                and prev_news < DEFAULT_NEWS_LOW_THRESHOLD):
            self._fire(CatalystEvent(
                asset=asset, trigger_type="news",
                trigger_value=news_risk,
                threshold=DEFAULT_NEWS_HIGH_THRESHOLD, ts=now,
            ))
        self._last_news_risk[asset] = news_risk

        # Trigger 3: factor synthesis macro regime flip (single source of
        # truth from tick_state — populated by factor_synthesis module).
        factor_regime = str(snap.get("factor_regime", ""))
        prev_factor_regime = self._last_factor_regime.get(asset, "")
        if (factor_regime and prev_factor_regime
                and factor_regime != prev_factor_regime
                and factor_regime != "unavailable"):
            self._fire(CatalystEvent(
                asset=asset, trigger_type="factor_regime_flip",
                trigger_value=1.0, threshold=1.0, ts=now,
                extra={"from": prev_factor_regime, "to": factor_regime},
            ))
        if factor_regime:
            self._last_factor_regime[asset] = factor_regime

        # Trigger 4: factor bias score flip (large swing)
        bias = float(snap.get("factor_bias_score", 0.0) or 0.0)
        prev_bias = self._last_bias_score.get(asset, 0.0)
        if abs(bias - prev_bias) >= DEFAULT_BIAS_FLIP_THRESHOLD:
            self._fire(CatalystEvent(
                asset=asset, trigger_type="bias_score_flip",
                trigger_value=bias - prev_bias,
                threshold=DEFAULT_BIAS_FLIP_THRESHOLD, ts=now,
                extra={"prev_bias": prev_bias, "new_bias": bias},
            ))
        if "factor_bias_score" in snap:
            self._last_bias_score[asset] = bias

        # Trigger 5: drawdown spike (>2% in 60s)
        equity = float(snap.get("equity_usd", 0.0) or 0.0)
        if equity > 0:
            self._equity_window.append((now, equity))
            self._equity_window = [
                (t, e) for t, e in self._equity_window
                if now - t <= DEFAULT_DRAWDOWN_WINDOW_S
            ]
            if len(self._equity_window) >= 2:
                peak = max(e for _, e in self._equity_window)
                dd_pct = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
                if dd_pct >= DEFAULT_DRAWDOWN_TRIGGER_PCT:
                    self._fire(CatalystEvent(
                        asset=asset, trigger_type="drawdown",
                        trigger_value=dd_pct,
                        threshold=DEFAULT_DRAWDOWN_TRIGGER_PCT,
                        ts=now,
                        extra={"peak": peak, "current": equity,
                               "window_s": DEFAULT_DRAWDOWN_WINDOW_S},
                    ))

    def _fire(self, event: CatalystEvent) -> None:
        self._last_preempt_at[event.asset] = event.ts
        self._n_preemptions += 1
        # Always log
        try:
            path = Path(LOG_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as e:
            logger.debug("[CATALYST] log write failed: %s", e)
        logger.info("[CATALYST:%s] PREEMPT trigger=%s value=%.3f threshold=%.3f",
                    event.asset, event.trigger_type,
                    event.trigger_value, event.threshold)
        # Surface the catalyst into tick_state so the brain reads it
        # on the (now triggered) preemptive call.
        try:
            from src.ai import tick_state as _ts
            _ts.update(event.asset,
                       last_catalyst=f"{event.trigger_type}={event.trigger_value:.2f}",
                       last_catalyst_ts=event.ts)
        except Exception:
            pass
        # Authoritative mode: invoke the callback (typically a brain call)
        if is_authoritative() and self.callback is not None:
            try:
                self.callback(event)
            except Exception as e:
                logger.warning("[CATALYST] callback failed: %s", e)


# Module-level singleton accessor for executor integration
_singleton: Optional[CatalystListener] = None


def get_listener() -> CatalystListener:
    global _singleton
    if _singleton is None:
        _singleton = CatalystListener()
    return _singleton
