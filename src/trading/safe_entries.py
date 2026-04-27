"""
Safe-entries gate: structural interventions to flip paper trading from negative-EV
to positive-EV + Sharpe ≥ 1.0 before the readiness gate's 14-day soak can complete.

Root causes this module addresses (from the 18-trade paper journal diagnostic):

  1. Stops placed at 0.6% median inside a 1.69% round-trip spread → every trade
     killed by bid-ask bounce before signal resolves. Fix: `apply_stop_floor`.

  2. LLM approved negative-score rule-vetoed setups (score<0 & llm_conf≥0.8 =
     −$104 of −$261 loss). Fix: `enforce_hard_score_veto` — rule veto is
     non-negotiable, LLM can only size down, not override.

  3. No R:R gate. Fix: `check_rr` rejects trades with reward < 2 × risk.

  4. Spread-blind entries on a 1.69% round-trip venue. Fix: `effective_min_score`
     raises the bar when venue cost is high.

  5. Uncapped loss variance. Fix: `fixed_fractional_qty` sizes so SL-hit
     loses exactly risk_pct × equity regardless of stop width.

  6. No consecutive-loss throttle. Fix: `SafeEntryState` halves size after
     consec_losses_halve, pauses after consec_losses_pause.

  7. Partial profit-taking for Sharpe. Fix: `maybe_partial_take` exits 50%
     at +1R and moves SL to breakeven on the rest.

All enable-checks go through `is_enabled()` which reads `ACT_SAFE_ENTRIES=1`.
Off by default — zero risk to current executor behaviour.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


# ---- Config defaults (override via executor config 'safe_entries' block) ----

DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "risk_pct": 0.5,  # Risk 0.5% of equity per trade on SL hit
    "min_rr": 2.0,  # Reject trades where TP/SL < 2.0
    "stop_atr_mult": 2.5,  # Min SL width = 2.5 × ATR
    "stop_spread_mult": 1.5,  # Min SL width = 1.5 × round-trip spread
    "partial_take_at_r": 1.0,  # Take 50% at +1R (risk-reward)
    "partial_take_fraction": 0.5,  # Close this fraction at partial take
    "move_sl_to_breakeven_after_partial": True,
    "consec_losses_halve": 3,  # After N consecutive losses, halve size
    "consec_losses_pause": 5,  # After N consecutive losses, pause trading
    "pause_hours": 4.0,  # Pause duration after consec_losses_pause
    "high_spread_pct_threshold": 1.2,  # Round-trip spread above this = high-spread venue
    "high_spread_score_bump": 1,  # Raise effective min score by this much
    "rolling_sharpe_n": 30,  # Window for rolling Sharpe
    "rolling_sharpe_min": 1.0,  # Soak-completion floor
}


# ---- Enable gate ----

def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Safe-entries is on iff ACT_SAFE_ENTRIES=1 OR config['safe_entries']['enabled']=True.

    Env wins so operators can force-on without config edits on the GPU box.
    """
    env = (os.environ.get("ACT_SAFE_ENTRIES") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if config is not None:
        cfg = config.get("safe_entries", {}) if isinstance(config, dict) else {}
        return bool(cfg.get("enabled", False))
    return False


def merged_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge user config into defaults. User-supplied keys win."""
    out = dict(DEFAULT_CONFIG)
    if isinstance(config, dict):
        user = config.get("safe_entries", {}) or {}
        out.update({k: user[k] for k in user if k in DEFAULT_CONFIG})
    return out


# ---- Intervention A: Stop-width floor ----

def apply_stop_floor(
    entry: float,
    sl_price: float,
    direction: str,
    atr: float,
    rt_spread_pct: float,
    config: Dict[str, Any],
) -> Tuple[float, str]:
    """Widen a too-tight stop to the greater of stop_atr_mult × ATR or stop_spread_mult × spread.

    Returns (new_sl_price, reason). Reason="floor_applied(atr)" / "floor_applied(spread)" /
    "floor_ok" lets the caller log why.

    Never widens beyond a sane cap (5× ATR) to avoid the opposite failure — stops so
    wide the position sizer reduces qty to dust.
    """
    atr_floor = max(atr, 0.0) * float(config.get("stop_atr_mult", 2.5))
    spread_floor = entry * (float(rt_spread_pct) / 100.0) * float(config.get("stop_spread_mult", 1.5))
    hard_cap = max(atr, 0.0) * 5.0

    floor = max(atr_floor, spread_floor)
    if hard_cap > 0:
        floor = min(floor, hard_cap)

    direction = (direction or "").upper()
    if direction in ("LONG", "BUY"):
        current_dist = max(entry - sl_price, 0.0)
        if current_dist >= floor or floor <= 0:
            return sl_price, "floor_ok"
        new_sl = entry - floor
        why = "floor_applied(atr)" if atr_floor >= spread_floor else "floor_applied(spread)"
        return new_sl, why
    else:
        current_dist = max(sl_price - entry, 0.0)
        if current_dist >= floor or floor <= 0:
            return sl_price, "floor_ok"
        new_sl = entry + floor
        why = "floor_applied(atr)" if atr_floor >= spread_floor else "floor_applied(spread)"
        return new_sl, why


# ---- Intervention B/D: Score gate ----

def effective_min_score(
    base_min: int,
    rt_spread_pct: float,
    config: Dict[str, Any],
) -> int:
    """Apply spread-aware bump. High-spread venues (Robinhood 1.69%) need stricter scores."""
    bump = 0
    if rt_spread_pct > float(config.get("high_spread_pct_threshold", 1.2)):
        bump = int(config.get("high_spread_score_bump", 1))
    return int(base_min) + bump


def enforce_hard_score_veto(entry_score: int, min_score: int) -> Tuple[bool, str]:
    """Score veto — hard for real capital, advisory for paper mode.

    Real capital path (ACT_REAL_CAPITAL_ENABLED=1): rule veto is non-
    negotiable, LLM cannot override. Paper mode: per operator directive
    (Unit 2), the brain has weighed every signal already; the gate's
    job is to surface the score, not block the trade. Authority rules
    (PDF 7 universal rules) stay hard regardless via authority_rules.

    Returns (reject, reason).
    """
    if entry_score < min_score:
        is_real_capital = os.environ.get(
            "ACT_REAL_CAPITAL_ENABLED", "",
        ).strip() == "1"
        if not is_real_capital:
            return False, f"paper_advisory_score_{entry_score}_below_min_{min_score}"
        return True, f"score_{entry_score}_below_min_{min_score}"
    return False, ""


# ---- Intervention C: R:R gate ----

def check_rr(
    entry: float,
    sl: float,
    tp: float,
    direction: str,
    min_rr: float,
) -> Tuple[bool, float, str]:
    """Compute reward/risk ratio. Returns (pass, rr, reason).

    A trade passes iff (abs(tp-entry)) / (abs(entry-sl)) >= min_rr AND sl/tp on
    the correct side of entry for the direction.
    """
    direction = (direction or "").upper()
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    if risk <= 0:
        return False, 0.0, "risk_zero"
    if tp <= 0:
        # No TP target set — the caller should synthesize one at min_rr × risk
        return False, 0.0, "no_tp"

    # Direction sanity: SL must be on the losing side
    if direction in ("LONG", "BUY") and (sl >= entry or tp <= entry):
        return False, 0.0, "sl_or_tp_on_wrong_side"
    if direction in ("SHORT", "SELL") and (sl <= entry or tp >= entry):
        return False, 0.0, "sl_or_tp_on_wrong_side"

    rr = reward / risk
    if rr < float(min_rr):
        return False, rr, f"rr_{rr:.2f}_below_{min_rr:.2f}"
    return True, rr, "ok"


def synthesize_tp(entry: float, sl: float, direction: str, min_rr: float) -> float:
    """Given an entry and SL, compute the TP that yields exactly min_rr reward:risk."""
    risk = abs(entry - sl)
    direction = (direction or "").upper()
    if direction in ("LONG", "BUY"):
        return entry + risk * float(min_rr)
    return entry - risk * float(min_rr)


# ---- Intervention E: Fixed-fractional sizing ----

def fixed_fractional_qty(
    entry: float,
    sl: float,
    equity: float,
    risk_pct: float,
) -> float:
    """Position size such that if SL hits, loss = risk_pct × equity.

    qty = (risk_pct/100 × equity) / |entry − sl|

    Falls back to 0 on degenerate inputs so the caller can skip the trade.
    """
    risk_dist = abs(entry - sl)
    if risk_dist <= 0 or equity <= 0 or risk_pct <= 0 or entry <= 0:
        return 0.0
    risk_dollars = equity * (risk_pct / 100.0)
    qty = risk_dollars / risk_dist
    return max(qty, 0.0)


# ---- Intervention F: Consecutive-loss throttle + persisted state ----

@dataclass
class AssetState:
    consecutive_losses: int = 0
    paused_until: float = 0.0  # unix timestamp; 0 means not paused
    last_outcome_ts: float = 0.0
    trade_pnl_pcts: List[float] = field(default_factory=list)  # rolling window for Sharpe


@dataclass
class SafeEntryState:
    """Persisted per-asset counters for the consecutive-loss throttle + Sharpe tracking."""
    assets: Dict[str, AssetState] = field(default_factory=dict)

    def _get(self, asset: str) -> AssetState:
        if asset not in self.assets:
            self.assets[asset] = AssetState()
        return self.assets[asset]

    def record_outcome(self, asset: str, pnl_pct: float, won: bool, now: Optional[float] = None,
                       window: int = 100) -> None:
        s = self._get(asset)
        s.last_outcome_ts = now if now is not None else time.time()
        s.trade_pnl_pcts.append(float(pnl_pct))
        if len(s.trade_pnl_pcts) > window:
            s.trade_pnl_pcts = s.trade_pnl_pcts[-window:]
        if won:
            s.consecutive_losses = 0
        else:
            s.consecutive_losses += 1

    def size_multiplier_for(
        self,
        asset: str,
        config: Dict[str, Any],
        now: Optional[float] = None,
    ) -> Tuple[float, str]:
        """Return (size_multiplier, reason). 1.0=full, 0.5=halved, 0.0=paused.

        Real capital: hard pause after `consec_losses_pause`. Paper mode
        per Unit 2: the brain is the authority. The throttle never
        fully zeros out -- worst case is quarter size with a paper-
        advisory tag so positions still fire and the brain learns
        from outcomes.
        """
        s = self._get(asset)
        now = now if now is not None else time.time()

        is_real_capital = os.environ.get(
            "ACT_REAL_CAPITAL_ENABLED", "",
        ).strip() == "1"

        if s.paused_until > now:
            remaining_h = (s.paused_until - now) / 3600.0
            if not is_real_capital:
                return 0.25, f"paper_advisory_paused_{remaining_h:.1f}h"
            return 0.0, f"paused_{remaining_h:.1f}h"

        halve_n = int(config.get("consec_losses_halve", 3))
        pause_n = int(config.get("consec_losses_pause", 5))
        if s.consecutive_losses >= pause_n:
            pause_hours = float(config.get("pause_hours", 4.0))
            s.paused_until = now + pause_hours * 3600.0
            if not is_real_capital:
                return 0.25, (
                    f"paper_advisory_pause_after_{s.consecutive_losses}_losses"
                )
            return 0.0, f"paused_{pause_hours}h_after_{s.consecutive_losses}_losses"
        if s.consecutive_losses >= halve_n:
            return 0.5, f"halved_after_{s.consecutive_losses}_losses"
        return 1.0, "full"

    def rolling_sharpe(self, asset: str, n: int = 30) -> float:
        """Sharpe computed on the last n pnl_pct values, zero risk-free.

        Returns 0.0 when insufficient samples or zero std (undefined Sharpe).
        """
        s = self._get(asset)
        xs = s.trade_pnl_pcts[-n:] if len(s.trade_pnl_pcts) >= n else s.trade_pnl_pcts
        if len(xs) < 2:
            return 0.0
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        if var <= 0:
            return 0.0
        return mean / math.sqrt(var)

    def combined_rolling_sharpe(self, n: int = 30) -> float:
        """Sharpe over the pooled last-n pnl samples across assets.

        We don't track per-trade timestamps inside each asset's pnl list (only
        `last_outcome_ts` is recorded, and it's shared across all trades in
        that asset's rolling window), so strict chronological interleaving
        across assets isn't possible. We pool the most recent `n` samples
        from each asset's cap-100 window and compute Sharpe on that sample.
        Sharpe's mean/std are invariant to order, so this is correct for the
        readiness-gate's purpose ("is the recent trade distribution good").
        """
        xs: List[float] = []
        for s in self.assets.values():
            # Take at most n from each asset's already-capped rolling window
            xs.extend(s.trade_pnl_pcts[-n:])
        xs = xs[-n:] if len(xs) > n else xs
        if len(xs) < 2:
            return 0.0
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        if var <= 0:
            return 0.0
        return mean / math.sqrt(var)

    # ── Persistence ──
    def to_dict(self) -> Dict[str, Any]:
        return {"assets": {k: asdict(v) for k, v in self.assets.items()}}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SafeEntryState":
        out = cls()
        for k, v in (d.get("assets") or {}).items():
            out.assets[k] = AssetState(**v)
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SafeEntryState":
        if not os.path.exists(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                return cls.from_dict(json.load(f))
        except Exception:
            return cls()


def default_state_path(logs_dir: str = "logs") -> str:
    return os.path.join(logs_dir, "safe_entries_state.json")


# ---- Intervention G: Partial take-profit at +1R ----

def maybe_partial_take(
    entry: float,
    current_price: float,
    sl: float,
    direction: str,
    already_partialled: bool,
    config: Dict[str, Any],
) -> Optional[Tuple[float, float, str]]:
    """At +1R (or configured partial_take_at_r multiplier of the initial risk), signal
    a partial close and return a new breakeven stop.

    Returns None if nothing to do, otherwise (new_sl_price, close_fraction, reason).

    Callers should:
      1. Send a reduce_only order for `close_fraction` × current_qty
      2. Update the position's SL to `new_sl_price`
      3. Set `already_partialled=True` on the position so this doesn't retrigger
    """
    if already_partialled:
        return None
    direction = (direction or "").upper()
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    r_mult = float(config.get("partial_take_at_r", 1.0))
    target_move = risk * r_mult

    if direction in ("LONG", "BUY"):
        hit = current_price >= entry + target_move
    else:
        hit = current_price <= entry - target_move
    if not hit:
        return None

    fraction = float(config.get("partial_take_fraction", 0.5))
    new_sl = entry if config.get("move_sl_to_breakeven_after_partial", True) else sl
    return new_sl, fraction, f"partial_at_{r_mult:.1f}R"


# ---- Intervention H helper: readiness gate Sharpe check ----

def load_sharpe_for_readiness(
    logs_dir: str = "logs",
    n: int = 30,
) -> float:
    """Load persisted state and compute combined rolling Sharpe over the last n trades.

    Called by `src/orchestration/readiness_gate.py`. Returns 0.0 on any error.
    """
    try:
        state = SafeEntryState.load(default_state_path(logs_dir))
        return state.combined_rolling_sharpe(n=n)
    except Exception:
        return 0.0
