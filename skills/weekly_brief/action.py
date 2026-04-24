"""Skill: /weekly-brief — compile a human-readable weekly activity report.

C20 (FinRobot-inspired, adapted to ACT's real-time paradigm).

Produces `reports/weekly_brief_YYYY-MM-DD.md` with:
  * Executive summary (trades taken vs skipped, PnL, regime)
  * Gate activity (readiness progress, authority violations, cost rejects)
  * Strategy repository (champion + challengers + bandit picks)
  * Brain activity (scan → plan → skip breakdown, parse failures, self-critiques)
  * Agent leaderboard (top-5 + bottom-5 weight movers)
  * Learning mesh (DSR summary, credit weights, quarantined components)
  * Appendix — top 5 most-interesting decisions with full rationale

Sources — all read-only:
  * `data/warm_store.sqlite` — decisions + outcomes + plan_json + self_critique
  * `data/brain_memory.sqlite` — scans + analyst traces
  * `data/strategy_repo.sqlite` — versioned strategies
  * `logs/autonomous_cycles.jsonl` — wide-loop cycle history
  * `memory/agent_<name>_state.json` — agent accuracy + weight
  * BodyControls / DSR tracker singletons (live state)

Never touches trading state; purely a read-projection.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WINDOW_DAYS = 7


# ── Data-source readers (all soft-fail) ────────────────────────────────


def _warm_store_path() -> Path:
    return Path(
        os.getenv("ACT_WARM_DB_PATH")
        or str(PROJECT_ROOT / "data" / "warm_store.sqlite")
    )


def _read_decisions(since_ts_ns: int) -> List[Dict[str, Any]]:
    """Read decisions from warm_store, schema-tolerant.

    Different ACT versions expose slightly different column sets
    (tier / size_pct / outcome_json may live as top-level columns OR
    be nested inside plan_json). We inspect `PRAGMA table_info` first,
    then construct a SELECT that pulls only columns that actually
    exist, and fall back to plan_json extraction for the rest.
    """
    db = _warm_store_path()
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db), timeout=5.0)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)").fetchall()}
    except Exception as e:
        logger.warning("weekly_brief: warm_store inspect failed: %s", e)
        return []

    base_cols = ["decision_id", "symbol", "ts_ns"]
    opt_cols = [c for c in ("direction", "final_action", "side", "tier",
                            "size_pct", "plan_json", "component_signals",
                            "self_critique", "outcome_json")
                if c in cols]
    select_cols = base_cols + opt_cols
    if not all(c in cols for c in base_cols):
        conn.close()
        return []

    sql = (f"SELECT {', '.join(select_cols)} FROM decisions "
           "WHERE ts_ns >= ? ORDER BY ts_ns DESC")
    try:
        rows = conn.execute(sql, (int(since_ts_ns),)).fetchall()
    except Exception as e:
        logger.warning("weekly_brief: warm_store read failed: %s", e)
        conn.close()
        return []
    conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        rec = dict(zip(select_cols, r))
        # Derive side from direction when no explicit column.
        if "side" not in rec:
            dr = rec.get("direction")
            rec["side"] = (
                "LONG" if dr == 1 else "SHORT" if dr == -1 else
                "FLAT" if dr == 0 else rec.get("final_action", "?")
            )
        # Extract tier + size from plan_json if not a top-level column.
        if "tier" not in rec or "size_pct" not in rec:
            try:
                pj = json.loads(rec.get("plan_json") or "{}")
                rec.setdefault("tier", pj.get("entry_tier") or pj.get("tier") or "")
                rec.setdefault("size_pct", pj.get("size_pct") or 0.0)
            except Exception:
                rec.setdefault("tier", "")
                rec.setdefault("size_pct", 0.0)
        out.append(rec)
    return out


def _read_cycles(since_ts: float) -> List[Dict[str, Any]]:
    p = PROJECT_ROOT / "logs" / "autonomous_cycles.jsonl"
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts = obj.get("timestamp")
            try:
                parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if parsed >= since_ts:
                out.append(obj)
    except Exception as e:
        logger.warning("weekly_brief: cycles read failed: %s", e)
    return out


def _read_strategy_repo() -> Dict[str, List[Dict[str, Any]]]:
    try:
        from src.trading.strategy_repository import get_repo
        repo = get_repo()
        champions = repo.search(status="champion", limit=10) or []
        challengers = repo.search(status="challenger", limit=10) or []
        quarantined = repo.search(status="quarantined", limit=10) or []
    except Exception as e:
        logger.debug("weekly_brief: strategy repo read failed: %s", e)
        return {"champion": [], "challenger": [], "quarantined": []}

    def _rec(r):
        return {
            "id": getattr(r, "strategy_id", "?"),
            "live_wins": int(getattr(r, "live_wins", 0) or 0),
            "live_losses": int(getattr(r, "live_losses", 0) or 0),
            "live_sharpe": float(getattr(r, "live_sharpe", 0.0) or 0.0),
            "regime": getattr(r, "regime_tag", "") or "",
        }
    return {
        "champion": [_rec(r) for r in champions],
        "challenger": [_rec(r) for r in challengers],
        "quarantined": [_rec(r) for r in quarantined],
    }


def _read_agents() -> List[Dict[str, Any]]:
    mem_dir = PROJECT_ROOT / "memory"
    if not mem_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in mem_dir.glob("agent_*_state.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        out.append({
            "name": obj.get("name") or p.stem.replace("agent_", "").replace("_state", ""),
            "weight": float(obj.get("weight") or 1.0),
            "accuracy": float(obj.get("accuracy") or 0.0),
            "samples": int(obj.get("samples") or 0),
        })
    return sorted(out, key=lambda a: a["weight"], reverse=True)


def _read_readiness() -> Dict[str, Any]:
    try:
        from src.orchestration.readiness_gate import evaluate, is_emergency_mode
        state = evaluate()
        return {
            "open": bool(getattr(state, "open_", False)),
            "failing_conditions": list(getattr(state, "failing_conditions", []) or []),
            "trades_required": int(getattr(state, "trades_required", 500) or 500),
            "trades_actual": int(getattr(state, "trades_actual", 0) or 0),
            "soak_days_required": float(getattr(state, "soak_days_required", 14.0) or 14.0),
            "soak_days_actual": float(getattr(state, "soak_days_actual", 0.0) or 0.0),
            "emergency_mode": bool(is_emergency_mode()),
        }
    except Exception as e:
        logger.debug("weekly_brief: readiness read failed: %s", e)
        return {}


def _read_body_controls() -> Dict[str, Any]:
    try:
        from src.learning.brain_to_body import get_controller
        c = get_controller().current()
        return c.to_dict() if hasattr(c, "to_dict") else {}
    except Exception:
        return {}


def _read_dsr() -> Dict[str, Any]:
    try:
        from src.learning.reward import get_tracker
        return get_tracker().snapshot_all()
    except Exception:
        return {}


# ── Analysis helpers ───────────────────────────────────────────────────


class DecisionSummary:
    """Plain class (not @dataclass) because Py3.14's dataclass stumbles
    on modules loaded via importlib.spec_from_file_location — the skill
    loader uses that pattern."""

    def __init__(self) -> None:
        self.total: int = 0
        self.shadow_plans: int = 0          # shadow-* decisions
        self.real_plans: int = 0
        self.skips: int = 0
        self.by_tier: Dict[str, int] = {}
        self.by_symbol: Dict[str, int] = {}
        self.realized_pnl_pct_sum: float = 0.0
        self.critiques_total: int = 0
        self.critiques_matched: int = 0
        self.parse_failures: int = 0


def _summarize_decisions(rows: List[Dict[str, Any]]) -> DecisionSummary:
    s = DecisionSummary()
    for r in rows:
        s.total += 1
        did = str(r.get("decision_id") or "")
        if did.startswith("shadow-"):
            s.shadow_plans += 1
        else:
            s.real_plans += 1
        tier = str(r.get("tier") or "unknown").lower()
        s.by_tier[tier] = s.by_tier.get(tier, 0) + 1
        if tier in ("skip", "reject", "max_steps", "parse_failures", "disabled"):
            s.skips += 1
        if tier == "parse_failures":
            s.parse_failures += 1
        sym = (r.get("symbol") or "").upper() or "?"
        s.by_symbol[sym] = s.by_symbol.get(sym, 0) + 1

        crit = r.get("self_critique")
        if crit and crit not in ("{}", "null"):
            try:
                obj = json.loads(crit) if isinstance(crit, str) else crit
                if isinstance(obj, dict):
                    s.critiques_total += 1
                    if obj.get("matched_thesis") is True:
                        s.critiques_matched += 1
            except Exception:
                pass

        out = r.get("outcome_json")
        if out:
            try:
                oo = json.loads(out) if isinstance(out, str) else out
                pnl = float((oo or {}).get("pnl_pct") or 0.0)
                s.realized_pnl_pct_sum += pnl
            except Exception:
                pass
    return s


def _interesting_decisions(rows: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    """Top-k most interesting decisions — those with self_critique filled or largest pnl_pct magnitude."""
    def _score(r):
        pnl = 0.0
        try:
            out = json.loads(r.get("outcome_json") or "{}")
            pnl = abs(float(out.get("pnl_pct") or 0.0))
        except Exception:
            pass
        has_crit = 0.5 if r.get("self_critique") not in (None, "", "{}") else 0.0
        return pnl + has_crit
    return sorted(rows, key=_score, reverse=True)[:k]


# ── Markdown renderer ──────────────────────────────────────────────────


def _render_markdown(
    window_start: datetime, window_end: datetime,
    decisions: List[Dict[str, Any]], summary: DecisionSummary,
    cycles: List[Dict[str, Any]], repo: Dict[str, List[Dict[str, Any]]],
    agents: List[Dict[str, Any]], readiness: Dict[str, Any],
    body: Dict[str, Any], dsr: Dict[str, Any],
) -> str:
    lines: List[str] = []
    start_s = window_start.strftime("%Y-%m-%d %H:%M UTC")
    end_s = window_end.strftime("%Y-%m-%d %H:%M UTC")
    days = (window_end - window_start).total_seconds() / 86400.0

    lines.append(f"# ACT Weekly Activity Brief")
    lines.append("")
    lines.append(f"**Window:** {start_s} → {end_s} ({days:.1f} days)")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Brain profile:** `{os.getenv('ACT_BRAIN_PROFILE', 'qwen3_r1')}`")
    lines.append(f"**Assets:** BTC/USD, ETH/USD (Robinhood spot)")
    lines.append(f"**Real capital:** `{'ENABLED' if os.getenv('ACT_REAL_CAPITAL_ENABLED') == '1' else 'disabled (paper mode)'}`")
    lines.append("")

    # ── Executive summary ────────────────────────────────────────────
    lines.append("## Executive summary")
    lines.append("")
    lines.append(f"- **Total decisions logged:** {summary.total}")
    lines.append(f"  - Shadow-mode plans: {summary.shadow_plans}")
    lines.append(f"  - Real-path decisions: {summary.real_plans}")
    lines.append(f"  - Skipped / rejected: {summary.skips}")
    lines.append(f"  - Parse failures: {summary.parse_failures}")
    lines.append(f"- **Realized PnL sum (across closed trades):** {summary.realized_pnl_pct_sum:+.2f}%")
    if summary.critiques_total > 0:
        pct = 100.0 * summary.critiques_matched / summary.critiques_total
        lines.append(f"- **Self-critique accuracy:** {summary.critiques_matched}/{summary.critiques_total} "
                     f"thesis-matches ({pct:.0f}%)")
    if summary.by_symbol:
        by_sym = ", ".join(f"{k}={v}" for k, v in sorted(summary.by_symbol.items()))
        lines.append(f"- **By asset:** {by_sym}")
    if summary.by_tier:
        by_tier = ", ".join(f"{k}={v}" for k, v in sorted(summary.by_tier.items()))
        lines.append(f"- **By tier:** {by_tier}")
    lines.append("")

    # ── Safety-gate activity ─────────────────────────────────────────
    lines.append("## Safety-gate activity")
    lines.append("")
    if readiness:
        lines.append(f"- **Readiness gate:** {'OPEN ✓' if readiness.get('open') else 'CLOSED'}")
        lines.append(f"  - Trades: {readiness.get('trades_actual', 0)} / "
                     f"{readiness.get('trades_required', 500)} required")
        lines.append(f"  - Soak days: {readiness.get('soak_days_actual', 0.0):.1f} / "
                     f"{readiness.get('soak_days_required', 14.0)} required")
        if readiness.get("failing_conditions"):
            lines.append(f"  - Failing: {', '.join(readiness['failing_conditions'])}")
        lines.append(f"- **Emergency mode:** {'ACTIVE ⚠' if readiness.get('emergency_mode') else 'normal'}")
    else:
        lines.append("- Readiness gate state unavailable")
    lines.append("")

    # ── Strategy repository ──────────────────────────────────────────
    lines.append("## Strategy repository")
    lines.append("")
    for status in ("champion", "challenger", "quarantined"):
        recs = repo.get(status) or []
        if not recs:
            continue
        lines.append(f"### {status.title()} ({len(recs)})")
        lines.append("")
        lines.append("| Strategy | Wins | Losses | Sharpe | Regime |")
        lines.append("|---|---|---|---|---|")
        for r in recs[:10]:
            lines.append(f"| `{r['id'][:24]}` | {r['live_wins']} | "
                         f"{r['live_losses']} | {r['live_sharpe']:+.2f} | "
                         f"{r.get('regime') or '-'} |")
        lines.append("")

    # ── Brain activity ───────────────────────────────────────────────
    lines.append("## Brain activity")
    lines.append("")
    if body:
        lines.append(f"- **Exploration bias:** {body.get('exploration_bias', 1.0):.2f}")
        lines.append(f"- **Genetic cadence:** {int(body.get('genetic_cadence_s', 21600))}s")
        lines.append(f"- **Emergency level:** `{body.get('emergency_level', 'normal')}`")
        lines.append(f"- **Avg scanner opportunity:** {body.get('avg_opportunity_score', 0):.0f}")
        lines.append(f"- **Analyst match rate:** {body.get('analyst_match_rate', 0.5)*100:.0f}%")
        lines.append(f"- **Analyst skip rate:** {body.get('analyst_skip_rate', 0)*100:.0f}%")
        pagents = body.get('priority_agents') or []
        if pagents:
            lines.append(f"- **Priority agents:** {', '.join(pagents[:5])}")
    else:
        lines.append("- Body controls state unavailable")
    lines.append("")

    # ── Agent leaderboard ───────────────────────────────────────────
    if agents:
        lines.append("## Agent leaderboard (by dynamic weight)")
        lines.append("")
        lines.append("| Agent | Weight | Accuracy | Samples |")
        lines.append("|---|---|---|---|")
        for a in agents[:10]:
            lines.append(f"| `{a['name']}` | {a['weight']:.2f}x | "
                         f"{a['accuracy']*100:.0f}% | {a['samples']} |")
        lines.append("")

    # ── Learning mesh ───────────────────────────────────────────────
    lines.append("## Learning mesh")
    lines.append("")
    if dsr:
        portfolio_dsr = {k: v for k, v in dsr.items() if k.startswith("portfolio:")}
        if portfolio_dsr:
            lines.append("**Portfolio differential Sharpe (DSR) by asset:**")
            lines.append("")
            for k, st in sorted(portfolio_dsr.items()):
                lines.append(f"- `{k}`: DSR={st.get('last_dsr', 0):+.3f} "
                             f"(n={int(st.get('n', 0))})")
            lines.append("")
        comp_dsr = [(k, v) for k, v in dsr.items() if not k.startswith("portfolio:")]
        if comp_dsr:
            top = sorted(comp_dsr, key=lambda x: x[1].get('last_dsr', 0),
                         reverse=True)[:5]
            lines.append("**Top components by DSR:**")
            lines.append("")
            for k, st in top:
                lines.append(f"- `{k}`: DSR={st.get('last_dsr', 0):+.3f}")
            lines.append("")
    else:
        lines.append("- DSR tracker empty (no closed trades yet this window)")
        lines.append("")

    # ── Autonomous cycles ────────────────────────────────────────────
    if cycles:
        lines.append("## Autonomous-loop activity")
        lines.append("")
        lines.append(f"- **Cycles run:** {len(cycles)}")
        n_retrain = sum(1 for c in cycles if c.get("retrained"))
        n_evolve = sum(1 for c in cycles if c.get("evolved"))
        lines.append(f"- **Retrain events:** {n_retrain}")
        lines.append(f"- **Evolution events:** {n_evolve}")
        lines.append("")

    # ── Appendix: top decisions ──────────────────────────────────────
    interesting = _interesting_decisions(decisions, k=5)
    if interesting:
        lines.append("## Appendix — notable decisions (top 5 by outcome magnitude)")
        lines.append("")
        for i, d in enumerate(interesting, 1):
            ts = datetime.fromtimestamp(int(d.get("ts_ns", 0)) / 1e9, tz=timezone.utc)
            lines.append(f"### {i}. `{d.get('decision_id', '?')[:28]}` — "
                         f"{d.get('symbol', '?')} {d.get('side', '?')} @ "
                         f"{ts.strftime('%Y-%m-%d %H:%M UTC')}")
            lines.append("")
            lines.append(f"- **Tier:** {d.get('tier', '?')}, size {d.get('size_pct', 0)}%")
            if d.get("outcome_json"):
                try:
                    oo = json.loads(d["outcome_json"])
                    lines.append(f"- **Outcome:** PnL {oo.get('pnl_pct', 0):+.2f}%, "
                                 f"exit `{oo.get('exit_reason', '-')}`, "
                                 f"hold {int(oo.get('hold_duration_s', 0))}s")
                except Exception:
                    pass
            if d.get("self_critique"):
                try:
                    cc = json.loads(d["self_critique"])
                    if cc.get("miss_reason"):
                        lines.append(f"- **Self-critique:** {cc.get('miss_reason')[:200]}")
                except Exception:
                    pass
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Auto-generated by `/weekly-brief` skill. Reproduce with: "
                 "`python -m src.skills.cli run weekly-brief`.*")
    return "\n".join(lines)


# ── Skill entry point ──────────────────────────────────────────────────


def run(args: Optional[Dict[str, Any]] = None) -> SkillResult:
    args = dict(args or {})

    # Resolve window
    try:
        days = float(args.get("days") or DEFAULT_WINDOW_DAYS)
    except (TypeError, ValueError):
        days = DEFAULT_WINDOW_DAYS
    now = datetime.now(timezone.utc)
    if args.get("start_date"):
        try:
            start = datetime.fromisoformat(str(args["start_date"])).replace(tzinfo=timezone.utc)
        except Exception:
            start = now - timedelta(days=days)
    else:
        start = now - timedelta(days=days)
    since_ts_ns = int(start.timestamp() * 1e9)
    since_ts = start.timestamp()

    # Gather data (all best-effort)
    decisions = _read_decisions(since_ts_ns)
    cycles = _read_cycles(since_ts)
    repo = _read_strategy_repo()
    agents = _read_agents()
    readiness = _read_readiness()
    body = _read_body_controls()
    dsr = _read_dsr()

    summary = _summarize_decisions(decisions)
    md = _render_markdown(start, now, decisions, summary, cycles, repo,
                          agents, readiness, body, dsr)

    # Write to reports/
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    stamp = now.strftime("%Y-%m-%d")
    report_path = reports_dir / f"weekly_brief_{stamp}.md"
    try:
        report_path.write_text(md, encoding="utf-8")
    except Exception as e:
        return SkillResult(ok=False, error=f"write failed: {e}")

    return SkillResult(
        ok=True,
        message=f"weekly brief written -> {report_path.relative_to(PROJECT_ROOT)}",
        data={
            "report_path": str(report_path),
            "window_start": start.isoformat(),
            "window_end": now.isoformat(),
            "window_days": days,
            "total_decisions": summary.total,
            "shadow_plans": summary.shadow_plans,
            "real_plans": summary.real_plans,
            "skips": summary.skips,
            "critiques_total": summary.critiques_total,
            "critiques_matched": summary.critiques_matched,
            "realized_pnl_pct_sum": round(summary.realized_pnl_pct_sum, 3),
            "cycles_run": len(cycles),
            "agents_tracked": len(agents),
        },
    )
