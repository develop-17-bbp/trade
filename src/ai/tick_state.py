"""Per-asset latest-tick signal snapshot, shared in-memory between
executor and context_builders.

Operator directive (2026-04-27): "make sure LLMs have every context"
-- the executor computes 38+ subsystem signals every tick (multi-
strategy consensus, 242-strategy universe vote, ML ensemble, sniper
advisory, pattern score, conviction state, VPIN, hurst, kalman,
GARCH, etc) but most of them never reach the brain's prompt.

This module is the bridge: executor writes the latest tick state
on every `_process_asset` call, and `context_builders.build_evidence_document`
reads it and adds a `TICK_SNAPSHOT` section to the brain's prompt.

Thread-safe (one lock per asset). The state is in-memory only; if
the bot restarts the section will be empty for the first tick after
boot but populates on tick 2.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

# {asset: {field: value, ..., 'updated_at': float}}
_state: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def update(asset: str, **fields: Any) -> None:
    """Merge `fields` into the asset's tick snapshot. Adds updated_at
    automatically. Pass any subsystem signal here.
    """
    if not asset:
        return
    asset = asset.upper()
    with _lock:
        cur = _state.setdefault(asset, {})
        cur.update(fields)
        cur["updated_at"] = time.time()


def get(asset: str) -> Dict[str, Any]:
    """Return a shallow copy of the asset's snapshot. Empty dict if
    the executor hasn't written for this asset yet."""
    if not asset:
        return {}
    asset = asset.upper()
    with _lock:
        return dict(_state.get(asset) or {})


def age_s(asset: str) -> Optional[float]:
    """Seconds since the snapshot was last updated. None if never written."""
    snap = get(asset)
    ts = snap.get("updated_at")
    if not isinstance(ts, (int, float)):
        return None
    return max(0.0, time.time() - float(ts))


def format_for_brain(asset: str, max_age_s: float = 300.0) -> str:
    """Format the snapshot as a compact <=600-char block the brain
    can read. Returns empty string when stale or empty so the brain
    sees it as 'no recent tick' rather than misleading old data."""
    snap = get(asset)
    if not snap:
        return ""
    age = age_s(asset)
    if age is None or age > max_age_s:
        return ""

    lines = []
    # PORTFOLIO: existing exposure FIRST so the brain doesn't keep
    # stacking on the same asset when it's already long N times.
    if "open_positions_same_asset" in snap:
        _n = snap.get('open_positions_same_asset', 0)
        _other = snap.get('open_positions_other_assets', 0)
        _exp = snap.get('exposure_pct', 0.0)
        _avg = snap.get('avg_unrealized_pct', 0.0)
        _age = snap.get('oldest_position_min', 0.0)
        _eq = snap.get('equity_usd', 0.0)
        lines.append(
            f"PORTFOLIO: same_asset_open={_n} other_assets_open={_other} "
            f"exposure={_exp:.1f}% avg_unrealized={_avg:+.2f}% "
            f"oldest={_age:.0f}min equity=${_eq:,.0f} "
            "(if same_asset_open >=3, prefer HOLD/EXIT over new ENTRY)"
        )
    # COST: round-trip spread the brain MUST clear before any plan
    # makes economic sense (Robinhood is ~1.69%, Bybit ~0.055%).
    if "spread_pct" in snap:
        lines.append(
            f"COST: round_trip_spread={snap.get('spread_pct', 0):.2f}% "
            f"min_profitable_move={snap.get('min_profitable_move_pct', 0):.2f}% "
            "(every plan must beat this — sub-spread trades are -EV)"
        )
    # Price + EMA
    if "price" in snap:
        lines.append(
            f"price=${snap.get('price', 0):,.2f} "
            f"ema_5m=${snap.get('ema_5m', 0):,.2f} "
            f"ema_dir={snap.get('ema_direction', '?')} "
            f"atr=${snap.get('atr', 0):,.2f}"
        )
    # TF signals
    tf = snap.get("tf_signals")
    if tf:
        lines.append(f"tf_signals={tf}")
    # Multi-strategy + universe consensus
    if "multi_consensus_score" in snap:
        lines.append(
            f"multi_strategy={snap.get('multi_consensus', '?')} "
            f"score={snap.get('multi_consensus_score', 0):+.3f}"
        )
    if "universe_consensus" in snap:
        lines.append(
            f"universe={snap.get('universe_buy', 0)}up/"
            f"{snap.get('universe_sell', 0)}down/"
            f"{snap.get('universe_flat', 0)}flat "
            f"consensus={snap.get('universe_consensus', '?')} "
            f"conf={snap.get('universe_confidence', 0):.2f}"
        )
    # Conviction
    if "conviction_tier" in snap:
        lines.append(
            f"conviction={snap.get('conviction_tier', '?')} "
            f"size_mult={snap.get('conviction_size_mult', 0):.1f} "
            f"reasons={snap.get('conviction_reasons', '')[:120]}"
        )
    # Sniper
    if "sniper_confluence" in snap:
        lines.append(
            f"sniper={snap.get('sniper_status', '?')} "
            f"confluence={snap.get('sniper_confluence', 0)}/"
            f"{snap.get('sniper_min_confluence', 0)} "
            f"reasons={snap.get('sniper_reasons', '')[:120]}"
        )
    # Pattern
    if "pattern_score" in snap:
        lines.append(
            f"pattern={snap.get('pattern_label', '?')} "
            f"score={snap.get('pattern_score', 0)}/10 "
            f"factors={snap.get('pattern_factors', '')[:120]}"
        )
    # Macro bias / crisis
    if "macro_bias" in snap:
        lines.append(
            f"macro_bias={snap.get('macro_bias', 0):+.2f} "
            f"crisis={snap.get('macro_crisis', False)}"
        )
    # ML ensemble
    if "ml_lgbm_conf" in snap or "ml_meta_prob" in snap:
        lines.append(
            f"ml_lgbm={snap.get('ml_lgbm_conf', 0):.2f} "
            f"ml_meta={snap.get('ml_meta_prob', 0):.2f} "
            f"ml_lstm={snap.get('ml_lstm_pred', '?')} "
            f"ml_patchtst={snap.get('ml_patchtst_pred', '?')} "
            f"ml_rl={snap.get('ml_rl_action', '?')}"
        )
    # Hurst / Kalman / GARCH
    if "hurst_regime" in snap:
        lines.append(
            f"hurst={snap.get('hurst_regime', '?')} "
            f"value={snap.get('hurst_value', 0):.2f} "
            f"kalman_slope={snap.get('kalman_slope', 0):+.4f} "
            f"garch_vol_pct={snap.get('garch_vol_pct', 0):.2f}"
        )
    # VPIN / microstructure
    if "vpin" in snap or "ob_imbalance" in snap:
        lines.append(
            f"vpin={snap.get('vpin', 0):.2f} "
            f"toxic_flow={snap.get('vpin_toxic', False)} "
            f"ob_imb={snap.get('ob_imbalance', 0):+.2f} "
            f"spread_pct={snap.get('spread_pct', 0):.3f}%"
        )
    # Price action
    if "fvg_zones" in snap or "order_block" in snap:
        lines.append(
            f"price_action={snap.get('price_action_summary', '')[:150]}"
        )
    # Trade-quality + position-limits
    if "trade_quality" in snap:
        lines.append(
            f"trade_quality={snap.get('trade_quality', 0)}/10 "
            f"risk_score={snap.get('risk_score', 0)}/10 "
            f"max_size_pct={snap.get('max_size_pct', 0):.1f}%"
        )
    # Genetic / agents
    if "genetic_vote" in snap:
        lines.append(
            f"genetic_vote={snap.get('genetic_vote', '?')} "
            f"({snap.get('genetic_count', 0)} strategies)"
        )
    if "agents_consensus" in snap:
        lines.append(
            f"agents={snap.get('agents_consensus', '?')} "
            f"votes={snap.get('agents_votes', '')[:120]}"
        )
    # Adaptive / accuracy
    if "adaptive_size_mult" in snap:
        lines.append(
            f"adaptive_size_mult={snap.get('adaptive_size_mult', 1.0):.2f} "
            f"recent_wr={snap.get('adaptive_recent_wr', 0):.2f}"
        )
    if "accuracy_weights" in snap:
        lines.append(
            f"accuracy_weights={snap.get('accuracy_weights', '')[:150]}"
        )
    # Self-evolving overlay
    if "evolved_ema_short" in snap:
        lines.append(
            f"evolved_ema={snap.get('evolved_ema_short', 0)}/"
            f"{snap.get('evolved_ema_long', 0)} "
            f"evolved_rsi={snap.get('evolved_rsi_period', 0)}"
        )
    # Range / regime
    if "range_pct_10c" in snap:
        lines.append(
            f"range_10candles={snap.get('range_pct_10c', 0):.2f}% "
            f"regime={snap.get('regime', '?')}"
        )

    if not lines:
        return ""
    return "\n".join(lines)
