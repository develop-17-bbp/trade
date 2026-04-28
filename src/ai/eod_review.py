"""End-of-day review — daily learning distillation.

Operator: "the brain should learn ... should only think about achieving
goal."

Without an explicit EOD review, every day starts fresh. Recent
critiques drift out of the 5-trace window. The brain re-discovers
yesterday's lessons mid-tick today. This module distills each day's
experience into a compact memo that:

  * Lists today's trades + outcomes
  * Identifies 3 best + 3 worst decisions
  * Surfaces patterns ("strong_long bucket WR was 30% today" /
    "I traded against macro headwind 4 times")
  * Writes adjustments for tomorrow ("raise sniper threshold,
    weight macro factor higher")

Output: Markdown file at memory/eod_<YYYY-MM-DD>.md plus a structured
summary surfaced via tick_state.eod_review_yesterday for the first
50 ticks of the next day.

Run as a skill (operator-triggered) OR via scheduler at end-of-day
UTC. Pure read-only over warm_store. No new schema.

Anti-overfit:
  * Aggregates only — doesn't store per-trade tool calls
  * Adjustments are SUGGESTIONS, not rules — brain reads as guidance
  * Skipped if <3 trades closed today (insufficient data for meaningful
    distillation)
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EOD_DIR = "memory/eod"
MIN_TRADES_FOR_REVIEW = 3
EQUITY_FALLBACK_USD = 16000.0


@dataclass
class EODReview:
    date_utc: str
    n_trades_closed: int
    win_rate: float
    realized_pnl_pct: float
    realized_pnl_usd: float
    best_trades: List[Dict[str, Any]] = field(default_factory=list)
    worst_trades: List[Dict[str, Any]] = field(default_factory=list)
    accuracy_calibration: str = "neutral"
    factor_correlations: Dict[str, float] = field(default_factory=dict)
    patterns_observed: List[str] = field(default_factory=list)
    adjustments_for_tomorrow: List[str] = field(default_factory=list)
    skipped_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date_utc": self.date_utc,
            "n_trades_closed": int(self.n_trades_closed),
            "win_rate": round(float(self.win_rate), 3),
            "realized_pnl_pct": round(float(self.realized_pnl_pct), 3),
            "realized_pnl_usd": round(float(self.realized_pnl_usd), 2),
            "best_trades": self.best_trades[:3],
            "worst_trades": self.worst_trades[:3],
            "accuracy_calibration": self.accuracy_calibration,
            "factor_correlations": self.factor_correlations,
            "patterns_observed": self.patterns_observed[:10],
            "adjustments_for_tomorrow": self.adjustments_for_tomorrow[:10],
            "skipped_reason": self.skipped_reason,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# EOD Review — {self.date_utc}",
            "",
            f"**Trades closed:** {self.n_trades_closed}  ",
            f"**Win rate:** {self.win_rate:.1%}  ",
            f"**Realized PnL:** {self.realized_pnl_pct:+.2f}%  (${self.realized_pnl_usd:+,.2f})  ",
            f"**Calibration label:** {self.accuracy_calibration}  ",
            "",
        ]
        if self.skipped_reason:
            lines.append(f"_Skipped review: {self.skipped_reason}_")
            return "\n".join(lines)

        lines.append("## Best 3 trades")
        if self.best_trades:
            for t in self.best_trades[:3]:
                lines.append(
                    f"- {t.get('asset', '?')} {t.get('direction', '?')} "
                    f"+{t.get('pnl_pct', 0):.2f}% — "
                    f"{t.get('thesis', '')[:120]}"
                )
        else:
            lines.append("- (none)")
        lines.append("")

        lines.append("## Worst 3 trades")
        if self.worst_trades:
            for t in self.worst_trades[:3]:
                lines.append(
                    f"- {t.get('asset', '?')} {t.get('direction', '?')} "
                    f"{t.get('pnl_pct', 0):+.2f}% — "
                    f"{t.get('thesis', '')[:120]}"
                )
        else:
            lines.append("- (none)")
        lines.append("")

        if self.patterns_observed:
            lines.append("## Patterns observed today")
            for p in self.patterns_observed:
                lines.append(f"- {p}")
            lines.append("")

        if self.adjustments_for_tomorrow:
            lines.append("## Adjustments for tomorrow")
            for a in self.adjustments_for_tomorrow:
                lines.append(f"- {a}")
            lines.append("")

        if self.factor_correlations:
            lines.append("## Factor outcome correlations")
            for k, v in self.factor_correlations.items():
                lines.append(f"- {k}: {v:+.2f}")
            lines.append("")

        return "\n".join(lines)


def _read_today_trades(date_utc: str) -> List[Dict[str, Any]]:
    """Pull today's closed trades from warm_store (exit_time on date_utc)."""
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        target_date = datetime.fromisoformat(date_utc).replace(tzinfo=timezone.utc)
        start_ns = int(target_date.timestamp() * 1e9)
        end_ns = int((target_date + timedelta(days=1)).timestamp() * 1e9)
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT ts_ns, symbol, plan_json, self_critique, "
                "component_signals "
                "FROM decisions "
                "WHERE ts_ns >= ? AND ts_ns < ? "
                "AND self_critique != '{}' AND self_critique IS NOT NULL "
                "ORDER BY ts_ns DESC",
                (start_ns, end_ns),
            ).fetchall()
        finally:
            conn.close()
        out = []
        for ts_ns, sym, plan_raw, crit_raw, comp_raw in rows:
            try:
                plan = json.loads(plan_raw or "{}")
                crit = json.loads(crit_raw or "{}")
                comp = json.loads(comp_raw or "{}")
            except Exception:
                continue
            pnl = crit.get("realized_pnl_pct")
            if pnl is None:
                continue
            out.append({
                "ts_ns": int(ts_ns),
                "asset": sym,
                "direction": plan.get("direction", "?"),
                "tier": plan.get("entry_tier", ""),
                "thesis": str(plan.get("thesis", ""))[:300],
                "pnl_pct": float(pnl),
                "regime": comp.get("regime", "unknown"),
                "miss_reason": str(crit.get("miss_reasons", ""))[:120],
                "lessons": str(crit.get("lessons", ""))[:200],
            })
        return out
    except Exception as e:
        logger.debug("eod_review read failed: %s", e)
        return []


def _detect_patterns(trades: List[Dict[str, Any]]) -> List[str]:
    """Surface the day's notable patterns. Pure heuristic."""
    patterns: List[str] = []
    if not trades:
        return patterns

    # 1. Direction skew
    n_long = sum(1 for t in trades if t["direction"] == "LONG")
    n_short = sum(1 for t in trades if t["direction"] == "SHORT")
    if n_long + n_short > 0:
        long_wr = sum(1 for t in trades if t["direction"] == "LONG" and t["pnl_pct"] > 0) / max(1, n_long) if n_long else 0
        if n_long >= 3 and long_wr < 0.4:
            patterns.append(f"LONG WR low today: {long_wr:.0%} on {n_long} longs — possible macro headwind")

    # 2. Tier skew
    sniper_trades = [t for t in trades if t["tier"] == "sniper"]
    if len(sniper_trades) >= 2:
        sniper_wr = sum(1 for t in sniper_trades if t["pnl_pct"] > 0) / len(sniper_trades)
        if sniper_wr < 0.5:
            patterns.append(f"sniper-tier WR={sniper_wr:.0%} on {len(sniper_trades)} trades — confluence criteria over-confident today")

    # 3. Regime skew
    regime_groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in trades:
        regime_groups.setdefault(t["regime"], []).append(t)
    for rkey, group in regime_groups.items():
        if len(group) >= 3:
            wr = sum(1 for t in group if t["pnl_pct"] > 0) / len(group)
            patterns.append(f"regime={rkey}: {len(group)} trades WR={wr:.0%}")

    # 4. Miss reason concentration
    miss_counter: Dict[str, int] = {}
    for t in trades:
        m = t.get("miss_reason", "")
        if m:
            miss_counter[m[:60]] = miss_counter.get(m[:60], 0) + 1
    top_miss = sorted(miss_counter.items(), key=lambda kv: kv[1], reverse=True)[:1]
    if top_miss and top_miss[0][1] >= 2:
        patterns.append(f"top miss reason: '{top_miss[0][0]}' ({top_miss[0][1]}x)")

    return patterns[:10]


def _adjustments_from_patterns(patterns: List[str], calib: str) -> List[str]:
    """Translate observed patterns + calibration label into concrete
    adjustments for tomorrow. Suggestions only."""
    adj: List[str] = []
    if calib == "over_confident":
        adj.append("DOWNWEIGHT strong_long bias — your high-conviction calls underperformed")
    if calib == "under_confident":
        adj.append("ACCEPT moderate-bias setups — you skipped winners")
    for p in patterns:
        if "macro headwind" in p.lower():
            adj.append("Weight macro_overlay HIGHER in synthesis tomorrow; SKIP setups against macro")
        if "sniper-tier" in p.lower() and "over-confident" in p.lower():
            adj.append("Raise sniper min_confluence by 1 OR require sniper score >= 7 instead of >= 6")
        if "miss reason" in p.lower():
            adj.append(f"Watch for repeating: {p}")
    if not adj:
        adj.append("No clear pattern flagged — continue current playbook")
    return adj[:10]


def compute_eod_review(date_utc: Optional[str] = None) -> EODReview:
    """Generate today's review (or specified date if past). Returns
    EODReview struct. Caller writes to memory/eod/<date>.md."""
    if date_utc is None:
        date_utc = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    trades = _read_today_trades(date_utc)
    n = len(trades)

    if n < MIN_TRADES_FOR_REVIEW:
        return EODReview(
            date_utc=date_utc,
            n_trades_closed=n,
            win_rate=0.0,
            realized_pnl_pct=0.0,
            realized_pnl_usd=0.0,
            skipped_reason=f"only {n} trades — need >= {MIN_TRADES_FOR_REVIEW} for meaningful review",
        )

    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    wr = wins / n
    realized_pct = sum(t["pnl_pct"] for t in trades)
    try:
        from src.ai import tick_state as _ts
        snap = _ts.get("BTC") or {}
        equity_usd = float(snap.get("equity_usd") or EQUITY_FALLBACK_USD)
    except Exception:
        equity_usd = EQUITY_FALLBACK_USD
    realized_usd = realized_pct * 0.01 * equity_usd

    sorted_by_pnl = sorted(trades, key=lambda t: t["pnl_pct"], reverse=True)
    best = sorted_by_pnl[:3]
    worst = sorted_by_pnl[-3:][::-1]

    # Get calibration label
    try:
        from src.ai.prediction_accuracy import compute_accuracy
        acc = compute_accuracy(lookback_days=7)
        calib = acc.calibration_label
    except Exception:
        calib = "neutral"

    patterns = _detect_patterns(trades)
    adjustments = _adjustments_from_patterns(patterns, calib)

    return EODReview(
        date_utc=date_utc,
        n_trades_closed=n,
        win_rate=wr,
        realized_pnl_pct=realized_pct,
        realized_pnl_usd=realized_usd,
        best_trades=best,
        worst_trades=worst,
        accuracy_calibration=calib,
        patterns_observed=patterns,
        adjustments_for_tomorrow=adjustments,
    )


def write_eod_review(review: EODReview) -> str:
    """Persist review to memory/eod/<date>.md. Returns the path."""
    try:
        d = Path(EOD_DIR)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"eod_{review.date_utc}.md"
        path.write_text(review.to_markdown(), encoding="utf-8")
        return str(path)
    except Exception as e:
        logger.debug("eod_review write failed: %s", e)
        return ""


_YESTERDAY_CACHE: "tuple[str, str] | None" = None


def get_yesterday_summary() -> str:
    """Return a compact one-paragraph summary of yesterday's review
    for tick_state injection. Empty string when no review exists yet.

    Cached per-day — file changes once at UTC rollover; per-tick
    disk reads were pure waste."""
    global _YESTERDAY_CACHE
    try:
        yesterday = (datetime.now(tz=timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        if _YESTERDAY_CACHE and _YESTERDAY_CACHE[0] == yesterday:
            return _YESTERDAY_CACHE[1]
        path = Path(EOD_DIR) / f"eod_{yesterday}.md"
        if not path.exists():
            _YESTERDAY_CACHE = (yesterday, "")
            return ""
        text = path.read_text(encoding="utf-8")[:800]
        line = text.replace("\n", " ").replace("##", "—")
        out = f"YESTERDAY_REVIEW: {line[:400]}"
        _YESTERDAY_CACHE = (yesterday, out)
        return out
    except Exception:
        return ""


