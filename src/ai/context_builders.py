"""
Shared analyst-context builders — used by agentic_bridge._fetch_scan_context
AND polymarket_analyst._build_context_block.

Both call-sites assembled the same 4 sources (scanner / news /
fear-greed / graph) independently. When polymarket_hunt scans 20
markets that's 80 context-subqueries per hunt — the same data
re-fetched 20 times. This module centralizes the builders so:

  1. Callers that want the same block for the same asset share the
     result within one tick (MAX_AGE_S TTL cache).
  2. New consumers that need the same context don't re-invent the
     function.

All builders are best-effort — missing data collapses to empty
string, never raises.

C19 (WebCryptoAgent-inspired): in addition to the free-form blob that
the analyst consumes today, we also expose a structured `EvidenceDocument`
that labels each section, tags it with a confidence score + age, and
can be re-emitted to JSON for audit + fine-tune training signal.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Structured Evidence Document (C19) ─────────────────────────────────


@dataclass
class EvidenceSection:
    """One labeled source in the analyst's evidence bundle."""
    name: str                       # "SCANNER_REPORT" / "NEWS" / "FEAR_GREED" / etc.
    content: str                    # human-readable body (already truncated)
    confidence: float = 0.5         # 0.0 - 1.0, source-self-reported
    age_s: float = 0.0              # freshness, seconds
    source: str = ""                # module / url for audit
    # FS-ReasoningAgent kind tag — "factual" | "subjective" | "mixed"
    # | "technical". Used by the analyst's regime-aware weighting and
    # by post-mortem analysis to attribute wins/losses to evidence
    # types (arXiv:2410.12464 Oct 2024).
    kind: str = "mixed"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "age_s": round(self.age_s, 1),
            "source": self.source,
            "kind": self.kind,
        }

    def to_prompt_block(self) -> str:
        """Formatted block the analyst prompt consumes."""
        tags = []
        if self.kind and self.kind != "mixed":
            tags.append(f"kind={self.kind}")
        if self.confidence < 1.0:
            tags.append(f"conf={self.confidence:.2f}")
        if self.age_s > 0:
            tags.append(f"age={int(self.age_s)}s")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"## {self.name}{tag_str}\n{self.content}".rstrip()


# Global budget for the assembled evidence document. Each individual
# section already self-truncates (NEWS=400, FEAR_GREED=120, TRACES=300,
# AGENT_VOTES=600, CRITIQUES=500, GRAPH=400, scanner rationale=200,
# BODY_CONTROLS=200, TICK_SNAPSHOT=variable) but there was no GLOBAL
# cap — if every section filled its budget AND TICK_SNAPSHOT was large,
# the total could blow past the LLM provider's user-prompt allotment
# and force a hard truncation that drops the actual task description.
# 4000 chars (~1000 tokens) leaves comfortable headroom inside a 16K
# num_ctx so the analyst's task / tool-call prompt isn't squeezed.
DEFAULT_EVIDENCE_DOC_BUDGET_CHARS = 4000
_TRUNC_SUFFIX = "\n[…evidence truncated to fit context budget…]"


@dataclass
class EvidenceDocument:
    """Structured bundle of all sources the analyst sees for an asset.

    The analyst still reads the flattened string (.to_prompt()); the
    structured form exists for audit, fine-tune training-data
    extraction, and post-hoc self-critique ("which section did I
    weight most heavily?").
    """
    asset: str
    compiled_at: float = field(default_factory=time.time)
    sections: List[EvidenceSection] = field(default_factory=list)

    def add(self, section: Optional[EvidenceSection]) -> None:
        """Append a section if it has non-empty content; ignore otherwise."""
        if section and section.content.strip():
            self.sections.append(section)

    def to_prompt(self, max_chars: int = DEFAULT_EVIDENCE_DOC_BUDGET_CHARS) -> str:
        """Flatten to the analyst-facing blob, capped at `max_chars` total.

        Sections are emitted in insertion order. Once the running total
        would exceed the budget, the offending section is hard-truncated
        and subsequent sections are dropped — better to lose a few
        late-priority blocks than to let one bloated section silently
        crowd out the entire task description downstream. Pass max_chars=0
        to disable the cap (audit / debug only — not recommended for
        live tick paths).
        """
        if max_chars <= 0:
            return "\n\n".join(s.to_prompt_block() for s in self.sections)

        out: List[str] = []
        used = 0
        sep_chars = 2  # "\n\n" between sections
        for s in self.sections:
            block = s.to_prompt_block()
            if not block:
                continue
            cost = (sep_chars if out else 0) + len(block)
            if used + cost <= max_chars:
                out.append(block)
                used += cost
                continue
            # Section overflow: truncate the head of this section to fit
            # remaining budget (after suffix). If even the suffix wouldn't
            # fit, just stop appending entirely.
            remaining = max_chars - used - (sep_chars if out else 0) - len(_TRUNC_SUFFIX)
            if remaining > 80:
                truncated = block[:remaining].rstrip() + _TRUNC_SUFFIX
                out.append(truncated)
            break
        return "\n\n".join(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "compiled_at": self.compiled_at,
            "section_count": len(self.sections),
            "sections": [s.to_dict() for s in self.sections],
        }

    def section(self, name: str) -> Optional[EvidenceSection]:
        for s in self.sections:
            if s.name == name:
                return s
        return None


# TTL for the per-asset cached context block. One shadow tick is
# 60-180s; 30s is long enough to share across the tick's compile +
# polymarket hunt, short enough that stale data doesn't reach the
# analyst.
DEFAULT_CONTEXT_TTL_S = 30.0


_CACHE: Dict[str, tuple] = {}   # asset.upper() -> (expires_ts, block)
_CACHE_LOCK = threading.Lock()


def _scanner_block(asset: str) -> str:
    try:
        from src.ai.brain_memory import get_scan_for_analyst
        r = get_scan_for_analyst(asset)
    except Exception:
        return ""
    if r is None:
        return ""
    return (
        f"## SCANNER REPORT ({int(r.age_s())}s old)\n"
        f"- opportunity_score: {r.opportunity_score:.0f}\n"
        f"- proposed_direction: {r.proposed_direction}\n"
        f"- signals: {', '.join(r.top_signals[:5]) or 'none'}\n"
        f"- rationale: {r.rationale[:200]}"
    )


_TRACES_BUDGET_CHARS = 300


def _traces_block(asset: str, limit: int = 5, max_age_s: float = 900.0) -> str:
    """Render the analyst's last `limit` decisions so it sees its own
    chain-of-reasoning across ticks. Capped at 300 chars so the seed
    block stays compact.

    Format:
        Recent thoughts (last N ticks):
        [HH:MM] LONG, thesis=tf_aligned, verdict=plan
        [HH:MM] SKIP, reason=parse_failure
    """
    try:
        from src.ai.brain_memory import read_recent_analyst_traces
        traces = read_recent_analyst_traces(
            asset, limit=limit, max_age_s=max_age_s,
        ) or []
    except Exception:
        return ""
    if not traces:
        return ""

    import time as _time
    lines: List[str] = [f"Recent thoughts (last {len(traces)} ticks):"]
    used = len(lines[0])
    for t in traces:
        try:
            hhmm = _time.strftime("%H:%M", _time.localtime(float(t.ts)))
        except Exception:
            hhmm = "--:--"
        direction = (t.direction or "").upper()
        if direction in ("SKIP", "FLAT", ""):
            reason = (t.thesis or t.verdict or "-").strip().splitlines()[0][:60]
            line = f"[{hhmm}] {direction or 'SKIP'}, reason={reason}"
        else:
            thesis = (t.thesis or "-").strip().splitlines()[0][:60]
            line = f"[{hhmm}] {direction}, thesis={thesis}, verdict={t.verdict or '-'}"
        if used + 1 + len(line) > _TRACES_BUDGET_CHARS and len(lines) > 1:
            break
        lines.append(line)
        used += 1 + len(line)
    return "\n".join(lines)


def _news_block(asset: str, hours: int = 12) -> str:
    try:
        from src.ai.web_context import get_news_digest
        d = get_news_digest(asset, hours=hours)
    except Exception:
        return ""
    if not d or not d.summary or d.summary == "unavailable":
        return ""
    return f"## NEWS\n{d.summary[:400]}"


def _fear_greed_block() -> str:
    try:
        from src.ai.web_context import get_fear_greed_digest
        fg = get_fear_greed_digest()
    except Exception:
        return ""
    if not fg or not fg.summary or fg.summary == "unavailable":
        return ""
    return f"## FEAR/GREED\n{fg.summary[:120]}"


def _graph_block(asset: str) -> str:
    try:
        from src.ai.graph_rag import query_digest
        g = query_digest(asset, since_s=3600, max_chars=400)
    except Exception:
        return ""
    if not g or g.startswith("[graph disabled") or g.startswith("[graph unavailable"):
        return ""
    return g


_AGENT_VOTES_BUDGET_CHARS = 600
_AGENT_VOTES_TOP_K = 6


def _agent_votes_block(asset: str) -> str:
    """Render the most-recent per-agent vote rationale.

    Phase D.4 wiring repair: the orchestrator computes 13 individual
    AgentVotes per cycle, but until now only the aggregate consensus
    landed in the analyst's TICK_SNAPSHOT. The analyst had to spend
    ReAct turns calling each `ask_<agent>` tool to recover the rationale
    that already existed. This block surfaces the top-K by confidence
    inline so the analyst can reason over them in step 1.
    """
    try:
        from src.agents.orchestrator import latest_votes, latest_votes_age_s
        votes = latest_votes(asset)
        age = latest_votes_age_s(asset)
    except Exception:
        return ""
    if not votes:
        return ""

    # Sort by confidence desc, then by agent name (stable tiebreak).
    items = []
    for name, vote in votes.items():
        try:
            conf = float(getattr(vote, "confidence", 0.0) or 0.0)
            direction = int(getattr(vote, "direction", 0) or 0)
            reasoning = (getattr(vote, "reasoning", "") or "").strip().splitlines()
            reason = reasoning[0][:80] if reasoning else ""
            items.append((conf, name, direction, reason))
        except Exception:
            continue
    if not items:
        return ""
    items.sort(key=lambda t: (-t[0], t[1]))
    items = items[:_AGENT_VOTES_TOP_K]

    age_str = f"{int(age)}s old" if age is not None else "age=?"
    lines = [f"Top-{len(items)} agent votes ({age_str}):"]
    used = len(lines[0])
    for conf, name, direction, reason in items:
        d_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(direction, "?")
        line = f"- {name}: {d_label} (conf={conf:.2f}) {reason}"
        if used + 1 + len(line) > _AGENT_VOTES_BUDGET_CHARS and len(lines) > 1:
            break
        lines.append(line)
        used += 1 + len(line)
    return "\n".join(lines)


_CRITIQUES_BUDGET_CHARS = 500


def _recent_critiques_block(asset: str, limit: int = 5, max_age_s: float = 86400.0) -> str:
    """Render the last `limit` post-trade self-critiques.

    Phase D.4 wiring repair: `trade_verifier.verify_and_persist` writes
    SelfCritique to `warm_store.decisions.self_critique`, but the
    analyst's seed context only reads `analyst_traces` from brain_memory
    — so the critique it asked the verifier to produce never made it
    back. CLAUDE.md §2 explicitly promises this loop closes; this block
    is what closes it.
    """
    try:
        from src.orchestration.warm_store import get_store
        import time as _time
        store = get_store()
        # Reads are lock-free; ensure pending writes are flushed first
        # so we see fresh critiques.
        store.flush()
        conn = store._get_conn()
        rows = conn.execute(
            """
            SELECT ts_ns, direction, final_action, self_critique
              FROM decisions
             WHERE symbol = ?
               AND self_critique IS NOT NULL AND self_critique != '{}' AND self_critique != ''
             ORDER BY ts_ns DESC
             LIMIT ?
            """,
            (asset, limit),
        ).fetchall()
    except Exception:
        return ""
    if not rows:
        return ""

    import json as _json
    import time as _time

    cutoff_ts_s = _time.time() - max_age_s
    lines: List[str] = [f"Recent critiques (last {len(rows)}):"]
    used = len(lines[0])

    for row in rows:
        ts_ns, direction, final_action, critique_json = row
        if not ts_ns:
            continue
        ts_s = ts_ns / 1e9
        if ts_s < cutoff_ts_s:
            continue
        try:
            sc = _json.loads(critique_json) if isinstance(critique_json, str) else (critique_json or {})
        except Exception:
            sc = {}
        if not isinstance(sc, dict):
            continue
        try:
            hhmm = _time.strftime("%H:%M", _time.localtime(ts_s))
        except Exception:
            hhmm = "--:--"
        verdict = str(sc.get("verdict") or sc.get("alignment") or "?").strip()[:24]
        ev_pred = sc.get("ev_predicted")
        ev_real = sc.get("ev_realized") or sc.get("pnl_pct")
        delta = sc.get("ev_delta")
        d_label = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(int(direction or 0), "?")
        nugget = (str(sc.get("lesson") or sc.get("note") or "").strip().splitlines() or [""])[0][:60]
        line = (
            f"[{hhmm}] {d_label} {final_action or '?'}: verdict={verdict} "
            f"pred={ev_pred} real={ev_real} Δ={delta} {nugget}"
        )
        if used + 1 + len(line) > _CRITIQUES_BUDGET_CHARS and len(lines) > 1:
            break
        lines.append(line)
        used += 1 + len(line)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


_OPEN_POSITIONS_BUDGET_CHARS = 1200


def _open_positions_block() -> str:
    """Render every open position across every active venue (alpaca-stocks,
    alpaca-crypto, alpaca-options, robinhood paper-sim). Asset-class-tagged
    so the analyst sees stocks alongside crypto on every tick — this is
    the cross-asset awareness operator demanded 2026-04-30.

    Each position line carries the data the LLM needs to decide HOLD /
    EXIT / ADD without making a follow-up tool call:
      * venue + asset_class
      * symbol, side, qty
      * entry_price → current_price (PnL % net + USD)
      * age_min, sl_price, tp_price
      * thesis (200 chars) so it remembers WHY the position was opened

    Failures per-venue are caught + logged; one bad fetcher doesn't
    blank out the whole block. Returns empty string when zero positions
    exist (no point bloating the prompt with an empty header).
    """
    if os.environ.get("ACT_DISABLE_OPEN_POSITIONS_BLOCK", "").strip() == "1":
        return ""
    rows: List[str] = []

    # Alpaca: stocks + crypto + options share the same /v2/positions
    # endpoint. asset_class field distinguishes 'us_equity' / 'crypto'
    # / 'us_option'. side is 'long' / 'short'. unrealized_plpc is a
    # decimal (0.012 = +1.2%).
    try:
        from src.data.fetcher import AlpacaClient
        ac = AlpacaClient(paper=True)
        if ac.available:
            for p in (ac.get_positions() or []):
                try:
                    sym = str(p.get("symbol") or "")
                    if not sym:
                        continue
                    asset_cls = str(p.get("asset_class") or "us_equity").upper()
                    if asset_cls == "US_EQUITY":
                        cls_tag = "STOCK alpaca"
                    elif asset_cls == "CRYPTO":
                        cls_tag = "CRYPTO alpaca"
                    elif asset_cls == "US_OPTION":
                        cls_tag = "OPTION alpaca"
                    else:
                        cls_tag = f"{asset_cls} alpaca"
                    side = str(p.get("side") or "long").upper()
                    qty = float(p.get("qty") or 0)
                    entry = float(p.get("avg_entry_price") or 0)
                    cur = float(p.get("current_price") or 0)
                    pnl_pct = float(p.get("unrealized_plpc") or 0) * 100.0
                    pnl_usd = float(p.get("unrealized_pl") or 0)
                    rows.append(
                        f"[{cls_tag}] {sym} {side} {qty}u @ ${entry:,.2f} → cur ${cur:,.2f} "
                        f"({pnl_pct:+.2f}% / ${pnl_usd:+.2f})"
                    )
                except Exception as e:
                    logger.debug("alpaca pos parse error: %s", e)
                    continue
    except Exception as e:
        logger.debug("alpaca positions fetch failed: %s", e)

    # Robinhood paper-sim: positions live in RobinhoodPaperFetcher.positions
    # dict, keyed by trade_id. Now that RH fetcher load_state on init
    # (commit 4d41fd2), we can get a fresh fetcher and read the persisted
    # state file. Each PaperPosition has entry_price, current_pnl_pct_net,
    # quantity, age_min, sl_price, tp_price, thesis.
    try:
        from src.data.robinhood_fetcher import RobinhoodPaperFetcher
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        rh = RobinhoodPaperFetcher(config=cfg)
        for tid, pos in (rh.positions or {}).items():
            try:
                sym = getattr(pos, "asset", "?")
                direction = getattr(pos, "direction", "LONG")
                qty = getattr(pos, "quantity", 0)
                entry = float(getattr(pos, "entry_price", 0) or 0)
                pnl_pct = float(getattr(pos, "current_pnl_pct_net", 0) or 0)
                age_min = float(getattr(pos, "age_min", 0) or 0)
                sl = float(getattr(pos, "sl_price", 0) or 0)
                tp = float(getattr(pos, "tp_price", 0) or 0)
                thesis = (str(getattr(pos, "thesis", "") or "")[:120]).replace("\n", " ")
                # RH paper-sim doesn't track current_price separately;
                # reconstruct from entry × (1 + pnl_pct/100)
                cur = entry * (1.0 + pnl_pct / 100.0)
                rows.append(
                    f"[CRYPTO robinhood] {sym} {direction} {qty} @ ${entry:,.2f} → cur ${cur:,.2f} "
                    f"({pnl_pct:+.2f}%) age={age_min:.0f}m sl=${sl:,.2f} tp=${tp:,.2f}"
                )
                if thesis:
                    rows.append(f"    thesis: {thesis!r}")
            except Exception as e:
                logger.debug("rh pos parse error: %s", e)
                continue
    except Exception as e:
        logger.debug("rh positions fetch failed: %s", e)

    if not rows:
        return ""   # don't bloat the prompt with an empty section

    # Trim to budget
    out: List[str] = [f"## OPEN_POSITIONS ({len([r for r in rows if not r.startswith('    thesis')])} positions)"]
    used = len(out[0])
    for r in rows:
        cost = 1 + len(r)
        if used + cost > _OPEN_POSITIONS_BUDGET_CHARS:
            out.append("    [...truncated; call query_open_positions_detail for the rest]")
            break
        out.append(r)
        used += cost
    return "\n".join(out)


_CAPITAL_STATE_BUDGET_CHARS = 600


def _capital_state_block() -> str:
    """Aggregate equity / cash / buying_power across every active venue
    on this box — alpaca paper (stocks + crypto + options) and robinhood
    paper-sim. The analyst's per-tick "where am I, how much can I deploy"
    snapshot. Operator directive 2026-04-30: every analysis must see this.

    Failures per-venue collapse to that venue being absent in the
    aggregate — we don't fail-block on a transient API error. Empty
    string when zero venues respond (rare; usually means env keys
    missing OR all networks down).
    """
    if os.environ.get("ACT_DISABLE_CAPITAL_STATE_BLOCK", "").strip() == "1":
        return ""
    lines: List[str] = []
    total_cash = 0.0
    total_equity = 0.0
    venue_count = 0

    try:
        from src.data.fetcher import AlpacaClient
        ac = AlpacaClient(paper=True)
        if ac.available:
            acct = ac.get_account() or {}
            equity = float(acct.get("equity") or 0)
            cash = float(acct.get("cash") or 0)
            bp = float(acct.get("buying_power") or 0)
            status = acct.get("status", "?")
            if equity > 0:
                lines.append(
                    f"[ALPACA paper]   equity=${equity:,.2f} cash=${cash:,.2f} "
                    f"buying_power=${bp:,.2f} status={status}"
                )
                total_cash += cash
                total_equity += equity
                venue_count += 1
    except Exception as e:
        logger.debug("alpaca account fetch failed: %s", e)

    try:
        from src.data.robinhood_fetcher import RobinhoodPaperFetcher
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        rh = RobinhoodPaperFetcher(config=cfg)
        rh_eq = float(getattr(rh, "equity", 0) or 0)
        rh_init = float(getattr(rh, "initial_capital", 0) or 0)
        if rh_eq > 0:
            ret_pct = ((rh_eq - rh_init) / rh_init * 100.0) if rh_init > 0 else 0.0
            lines.append(
                f"[ROBINHOOD sim]  equity=${rh_eq:,.2f} (start ${rh_init:,.2f}, "
                f"{ret_pct:+.2f}%) [no leverage; spot-long only]"
            )
            total_cash += rh_eq
            total_equity += rh_eq
            venue_count += 1
    except Exception as e:
        logger.debug("rh account fetch failed: %s", e)

    if not lines:
        return ""
    header = f"## CAPITAL_STATE ({venue_count} venue{'s' if venue_count != 1 else ''} active)"
    summary = (
        f"\nTOTAL deployable cash: ${total_cash:,.2f}  "
        f"combined equity: ${total_equity:,.2f}"
    )
    body = "\n".join(lines)
    return (header + "\n" + body + summary)[:_CAPITAL_STATE_BUDGET_CHARS]


def _goal_aware_pnl_block() -> str:
    """Surface today's PnL vs the 1%/day target so the LLM sizes new
    entries against residual gap, not as if every trade starts from zero.

    Operator directive 2026-04-30: 1%/day is the non-negotiable target.
    Research finding (LLMQuant 2026): 'use AI to think more clearly
    about risk, limits, and failure - not to think for you'. Concretely
    that means feeding the LLM a goal-aware risk budget every tick:

        TODAY: 14:32 ET (4.2h until close)
        REALIZED: +$23.50 (+0.21%)  UNREALIZED: -$5.00 (-0.04%)
        TODAY TOTAL: +0.17%
        TARGET: +1.00% / day
        GAP: +0.83% remaining
        RECOMMENDED RISK BUDGET: 0.5-2x normal size (gap is large but achievable)

    The LLM will then preferentially take HIGHER-conviction setups when
    the gap is large vs trail-stop-tighter when comfortably ahead.
    """
    if os.environ.get("ACT_DISABLE_GOAL_PNL_BLOCK", "").strip() == "1":
        return ""
    try:
        import datetime as _dt
        now_utc = _dt.datetime.now(_dt.timezone.utc)
        # Today UTC start
        day_start_ns = int(_dt.datetime(
            now_utc.year, now_utc.month, now_utc.day,
            tzinfo=_dt.timezone.utc,
        ).timestamp() * 1e9)
    except Exception:
        return ""

    realized_today = 0.0
    unrealized_now = 0.0
    initial_capital = 0.0
    current_equity = 0.0

    # Aggregate across venues
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        store.flush()
        conn = store._get_conn()
        rows = conn.execute(
            "SELECT plan_json FROM decisions "
            "WHERE ts_ns >= ? AND final_action IN ('CLOSE','EXIT','LONG','SHORT') "
            "AND decision_id NOT LIKE 'shadow-%'",
            (day_start_ns,),
        ).fetchall()
        for (raw,) in rows:
            if not raw:
                continue
            try:
                import json as _j
                d = _j.loads(raw) if isinstance(raw, str) else raw
                pnl = float(d.get("realized_pnl_usd") or d.get("pnl_usd") or 0)
                realized_today += pnl
            except Exception:
                continue
    except Exception:
        pass

    try:
        from src.data.fetcher import AlpacaClient
        ac = AlpacaClient(paper=True)
        if ac.available:
            acct = ac.get_account() or {}
            current_equity += float(acct.get("equity") or 0)
            try:
                _last = float(acct.get("last_equity") or 0)
                if _last > 0:
                    realized_today += (current_equity - _last)
            except Exception:
                pass
    except Exception:
        pass

    try:
        from src.data.robinhood_fetcher import RobinhoodPaperFetcher
        import yaml
        from pathlib import Path
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        rh = RobinhoodPaperFetcher(config=cfg)
        rh_eq = float(getattr(rh, "equity", 0) or 0)
        rh_init = float(getattr(rh, "initial_capital", 0) or 0)
        if rh_eq > 0:
            current_equity += rh_eq
            initial_capital += rh_init
            for p in (getattr(rh, "positions", {}) or {}).values():
                try:
                    unrealized_now += float(getattr(p, "unrealized_pnl_usd", 0) or 0)
                except Exception:
                    pass
    except Exception:
        pass

    if current_equity <= 0:
        return ""

    today_pct = (realized_today / current_equity * 100.0) if current_equity > 0 else 0.0
    unrealized_pct = (unrealized_now / current_equity * 100.0) if current_equity > 0 else 0.0
    total_pct = today_pct + unrealized_pct
    target_pct = 1.0
    gap_pct = target_pct - total_pct

    # Hours until US-equity close (16:00 ET = 20:00 UTC) for stock-side risk pacing.
    try:
        et_close = now_utc.replace(hour=20, minute=0, second=0, microsecond=0)
        if now_utc > et_close:
            hours_to_close = 24 - (now_utc.hour - 20)
        else:
            hours_to_close = (et_close - now_utc).total_seconds() / 3600.0
    except Exception:
        hours_to_close = 6.0

    # Risk-budget guidance keyed on residual gap.
    if gap_pct <= 0:
        budget = "0.3-0.5x normal size (target hit; preserve gains, no chasing)"
    elif gap_pct < 0.3:
        budget = "0.5-1.0x normal size (gap small; precision over volume)"
    elif gap_pct < 0.7:
        budget = "1.0-1.5x normal size (gap moderate; take quality setups)"
    else:
        budget = "1.0-2.0x normal size (gap large but achievable; lean into conviction)"

    lines = [
        f"TODAY: {now_utc.strftime('%H:%M UTC')} (~{hours_to_close:.1f}h to NYSE close)",
        f"REALIZED: ${realized_today:+,.2f} ({today_pct:+.2f}%)  "
        f"UNREALIZED: ${unrealized_now:+,.2f} ({unrealized_pct:+.2f}%)",
        f"TODAY TOTAL: {total_pct:+.2f}%",
        f"TARGET: +{target_pct:.2f}% / day",
        f"GAP TO TARGET: {gap_pct:+.2f}%",
        f"RECOMMENDED RISK BUDGET: {budget}",
    ]
    return "\n".join(lines)


def _strategy_performance_block() -> str:
    """Surface dynamic Bayesian agent weights + recent accuracy + the
    strategy_repository's current champion + top challengers so the LLM
    knows WHICH inputs to trust on this tick - not all signals equally.

    Operator directive 2026-04-30 ('evolve' principle): the LLM should
    learn to lean on agents/strategies that have been profitable
    recently and discount or ignore ones that have been losing. Without
    this block, agent_votes shows direction+confidence but not weight,
    so the LLM treats trend_momentum (acc=72%, w=2.3) and a quarantined
    component the same. This closes that gap.
    """
    lines: list = []

    # Section 1: per-agent weight + accuracy from the live orchestrator.
    try:
        from src.agents.orchestrator import _ORCHESTRATOR_SINGLETON
        orch = _ORCHESTRATOR_SINGLETON
    except Exception:
        orch = None
    if orch is None:
        # Fall back to JSON state files written by save_state().
        try:
            import os as _os
            import glob as _glob
            import json as _json
            mem_dir = _os.path.join(
                _os.path.dirname(_os.path.dirname(
                    _os.path.dirname(_os.path.abspath(__file__))
                )), "memory",
            )
            files = _glob.glob(_os.path.join(mem_dir, "agent_*_state.json"))
            agents_data: list = []
            for f in files:
                try:
                    with open(f) as fh:
                        s = _json.load(fh)
                    name = s.get("name") or _os.path.basename(f)[6:-11]
                    weight = float(s.get("weight") or 1.0)
                    hist = s.get("accuracy_history") or []
                    acc = (sum(hist) / len(hist)) if hist else 0.5
                    n = int(s.get("total_calls") or 0)
                    agents_data.append((name, weight, acc, n))
                except Exception:
                    continue
            agents_data.sort(key=lambda t: -t[1])
            if agents_data:
                lines.append("AGENT WEIGHTS (Bayesian, dynamic; persisted):")
                for name, w, acc, n in agents_data[:10]:
                    tag = ("STRONG" if w >= 1.5 else
                           "GOOD" if w >= 1.0 else
                           "WEAK" if w >= 0.5 else
                           "DISTRUST")
                    lines.append(
                        f"- {name[:24]:<24} w={w:.2f} acc={acc*100:.0f}% n={n}  {tag}"
                    )
        except Exception:
            pass
    else:
        try:
            agents = getattr(orch, "agents", {}) or {}
            agents_data = []
            for name, agent in agents.items():
                try:
                    w = float(agent.get_weight()) if hasattr(agent, "get_weight") else 1.0
                    acc = float(agent.get_accuracy()) if hasattr(agent, "get_accuracy") else 0.5
                    n = int(getattr(agent, "_total_calls", 0))
                    agents_data.append((name, w, acc, n))
                except Exception:
                    continue
            agents_data.sort(key=lambda t: -t[1])
            if agents_data:
                lines.append("AGENT WEIGHTS (Bayesian, dynamic; live):")
                for name, w, acc, n in agents_data[:10]:
                    tag = ("STRONG" if w >= 1.5 else
                           "GOOD" if w >= 1.0 else
                           "WEAK" if w >= 0.5 else
                           "DISTRUST")
                    lines.append(
                        f"- {name[:24]:<24} w={w:.2f} acc={acc*100:.0f}% n={n}  {tag}"
                    )
        except Exception:
            pass

    # Section 2: top genetic strategies (champion + 3 challengers).
    try:
        from src.trading.strategy_repository import StrategyRepository
        repo = StrategyRepository()
        champ = repo.get_champion() if hasattr(repo, "get_champion") else None
        top = []
        if hasattr(repo, "list_strategies"):
            try:
                top = repo.list_strategies(status="champion") or []
                top += repo.list_strategies(status="challenger") or []
                top.sort(key=lambda s: -float(getattr(s, "sharpe", 0.0) or 0.0))
                top = top[:4]
            except Exception:
                top = []
        if top:
            lines.append("\nGENETIC HALL-OF-FAME (top 4 by Sharpe, recent window):")
            for s in top:
                name = getattr(s, "name", "?")
                sharpe = float(getattr(s, "sharpe", 0.0) or 0.0)
                wr = float(getattr(s, "win_rate", 0.0) or 0.0)
                status = getattr(s, "status", "?")
                lines.append(
                    f"- {name[:30]:<30} Sharpe={sharpe:.2f} win={wr*100:.0f}% [{status}]"
                )
    except Exception:
        pass

    # Section 3: genetic-audit modules (P0 walk-forward + P2 MAP-Elites)
    # Surfaces the OOS-validated winners + niche-diverse winners so the
    # brain can discount in-sample fitness and pick regime-fit strategies.
    try:
        import json as _json
        import os as _os
        ctx_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.dirname(
                _os.path.abspath(__file__)))),
            "data", "adaptation_context.json",
        )
        if _os.path.exists(ctx_path):
            ctx = _json.loads(open(ctx_path).read() or "{}")
            audit = ctx.get("genetic_audit") or {}
            wf = audit.get("walk_forward")
            best_promotable = (wf or {}).get("best_promotable")
            if best_promotable:
                lines.append(
                    "\nWALK-FORWARD OOS WINNER (passes Deflated-Sharpe + overfit gate):"
                )
                lines.append(
                    f"- {str(best_promotable.get('dna_name', '?'))[:30]:<30} "
                    f"test_sharpe={best_promotable.get('test_sharpe', 0):.2f} "
                    f"DSR={best_promotable.get('deflated_sharpe', 0):.2f} "
                    f"p_pos={best_promotable.get('p_true_sharpe_positive', 0):.2f} "
                    f"trades={best_promotable.get('test_trades', 0)}"
                )
            elif wf and wf.get("best_oos"):
                bo = wf["best_oos"]
                lines.append(
                    "\nWALK-FORWARD OOS BEST (none promotable — discount in-sample fitness):"
                )
                lines.append(
                    f"- {str(bo.get('dna_name', '?'))[:30]:<30} "
                    f"test_sharpe={bo.get('test_sharpe', 0):.2f} "
                    f"overfit_indicator={bo.get('overfit_indicator', 0):.2f}"
                )
            me = audit.get("map_elites")
            diverse = (me or {}).get("diverse_top_5") or []
            if diverse:
                summary = me.get("summary", {}) if me else {}
                lines.append(
                    f"\nMAP-ELITES NICHE WINNERS (coverage "
                    f"{summary.get('coverage_pct', 0):.0f}% of 300 cells):"
                )
                for entry in diverse[:3]:
                    fam = entry.get("entry_family", "?")
                    lines.append(
                        f"- [{fam:<8}] {str(entry.get('dna_name', '?'))[:24]:<24} "
                        f"fitness={entry.get('fitness', 0):.2f} "
                        f"win={int(entry.get('win_rate', 0)*100)}% "
                        f"sharpe={entry.get('sharpe', 0):.2f}"
                    )
            drift = audit.get("drift_signal")
            if drift and drift.get("drift_detected"):
                lines.append(
                    f"\nREGIME DRIFT DETECTED — triggers: "
                    f"{', '.join(drift.get('triggers', []))[:120]}. "
                    f"Treat older strategies skeptically."
                )
    except Exception:
        pass

    if not lines:
        return ""
    lines.append(
        "\nGUIDANCE: lean toward STRONG-weight agents (w>=1.5) and "
        "champion strategies. Discount WEAK/DISTRUST agents. When the "
        "champion strategy and STRONG agents agree, conviction is "
        "higher than either alone."
    )
    return "\n".join(lines)


def _body_controls_block() -> str:
    try:
        from src.learning.brain_to_body import get_controller
        c = get_controller().current()
    except Exception:
        return ""
    return (
        f"## BODY CONTROLS\n"
        f"- emergency_level: {c.emergency_level}\n"
        f"- exploration_bias: {c.exploration_bias:.2f}\n"
        f"- priority_agents (ask these first): "
        f"{', '.join(c.priority_agents[:5]) or '(default)'}\n"
        f"- reason: {c.reason[:200]}"
    )


def build_evidence_document(
    asset: str,
    *,
    include_scanner: bool = True,
    include_traces: bool = True,
    include_news: bool = True,
    include_fear_greed: bool = True,
    include_graph: bool = True,
    include_body_controls: bool = True,
    include_agent_votes: bool = True,
    include_recent_critiques: bool = True,
    include_open_positions: bool = True,
    include_capital_state: bool = True,
    include_strategy_performance: bool = True,
) -> EvidenceDocument:
    """Structured assemble of the analyst's evidence bundle (C19).

    Returns an EvidenceDocument with labeled, confidence-scored,
    age-tagged sections. Callers that want the legacy flat-string
    form can call `.to_prompt()` on the result.
    """
    asset_key = asset.upper()
    doc = EvidenceDocument(asset=asset_key)

    if include_scanner:
        c = _scanner_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="SCANNER_REPORT", content=c,
                confidence=0.7, source="brain_memory", kind="mixed",
            ))
    if include_traces:
        # Cross-tick chain-of-reasoning: analyst sees its own last 5
        # thoughts within a 15-min window so it can hold/continue
        # instead of restarting fresh every tick.
        c = _traces_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="RECENT_ANALYST_TRACES", content=c,
                confidence=0.6, source="brain_memory", kind="technical",
            ))
    if include_news:
        c = _news_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="NEWS", content=c,
                confidence=0.5, source="web_context.news",
                # News blends fact (regulation/tech) + subjectivity
                # (sentiment angles). Default to mixed; specific
                # callers that fetch FILTERED-factual or
                # FILTERED-subjective news streams can override.
                kind="mixed",
            ))
    if include_fear_greed:
        c = _fear_greed_block()
        if c:
            doc.add(EvidenceSection(
                name="FEAR_GREED", content=c,
                confidence=0.6, source="web_context.fear_greed",
                kind="subjective",   # crowd-emotion gauge
            ))
    if include_graph:
        c = _graph_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="KNOWLEDGE_GRAPH", content=c,
                confidence=0.65, source="graph_rag.query_digest",
                kind="mixed",
            ))
    if include_body_controls:
        c = _body_controls_block()
        if c:
            doc.add(EvidenceSection(
                name="BODY_CONTROLS", content=c,
                confidence=0.8, source="brain_to_body.controller",
                kind="technical",    # quant-derived control signals
            ))
    if include_agent_votes:
        # Phase D.4 wiring: individual orchestrator-agent votes (top-K
        # by confidence) so the analyst sees rationale, not just
        # aggregate consensus.
        c = _agent_votes_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="AGENT_VOTES", content=c,
                confidence=0.75, source="orchestrator.latest_votes",
                kind="mixed",
            ))
    if include_strategy_performance:
        # Operator directive 2026-04-30 (the 'evolve' principle): LLM
        # must SEE which agents/strategies have been profitable recently
        # so it leans on winners and discounts losers, instead of
        # treating every signal as equal weight. Surfaces dynamic
        # Bayesian agent weights + recent accuracy + champion strategy
        # + top genetic challengers.
        c = _strategy_performance_block()
        if c:
            doc.add(EvidenceSection(
                name="STRATEGY_PERFORMANCE", content=c,
                confidence=0.85, source="agents.weights+strategy_repo",
                kind="meta",
            ))
    # Goal-aware PnL: research finding (LLMQuant 2026) - LLMs need
    # explicit risk budget keyed on residual gap to target. Without
    # this they size every entry as if starting from zero. Surfaces
    # today_pct vs 1%/day target + recommended size multiplier.
    c = _goal_aware_pnl_block()
    if c:
        doc.add(EvidenceSection(
            name="GOAL_AWARE_PNL", content=c,
            confidence=0.95, source="warm_store+venue_accounts",
            kind="meta",
        ))
    if include_recent_critiques:
        # Phase D.4 wiring: post-trade self-critiques from trade_verifier
        # so the analyst calibrates against its own past predictive errors.
        c = _recent_critiques_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="RECENT_CRITIQUES", content=c,
                confidence=0.7, source="warm_store.self_critique",
                kind="mixed",
            ))
    if include_open_positions:
        # Operator directive 2026-04-30: every analysis must see open
        # positions across ALL venues (stocks + crypto + options).
        # Cross-asset visibility — when reasoning about NVDA the analyst
        # also sees the open BTC position; when on BTC it sees NVDA.
        # Closes the "LLM doesn't manage existing portfolio" gap.
        c = _open_positions_block()
        if c:
            doc.add(EvidenceSection(
                name="OPEN_POSITIONS", content=c,
                confidence=0.9, source="alpaca+robinhood positions",
                kind="technical",
            ))
    if include_capital_state:
        # Same operator directive: per-tick "where am I, how much can I
        # deploy" snapshot aggregated across every active venue.
        c = _capital_state_block()
        if c:
            doc.add(EvidenceSection(
                name="CAPITAL_STATE", content=c,
                confidence=0.95, source="alpaca+robinhood accounts",
                kind="technical",
            ))

    # TICK_SNAPSHOT — every subsystem signal the executor just computed.
    # Brings the "11 brain blind spots" into the LLM's prompt:
    # multi-strategy, 242-universe, conviction tier, sniper confluence,
    # pattern score, macro bias, ML ensemble (LGBM/LSTM/PatchTST/RL),
    # hurst/kalman/GARCH, VPIN, microstructure, price-action, trade-
    # quality, genetic vote, agents consensus, evolved overlay params.
    # Operator directive 2026-04-27: "make sure LLMs have every context."
    try:
        from src.ai.tick_state import format_for_brain as _format_tick
        c = _format_tick(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="TICK_SNAPSHOT", content=c,
                confidence=0.9, source="executor.tick_state",
                kind="technical",
            ))
    except Exception:
        pass
    return doc


def build_analyst_context(
    asset: str,
    *,
    include_scanner: bool = True,
    include_traces: bool = True,
    include_news: bool = True,
    include_fear_greed: bool = True,
    include_graph: bool = True,
    include_body_controls: bool = True,
    include_agent_votes: bool = True,
    include_recent_critiques: bool = True,
    include_open_positions: bool = True,
    include_capital_state: bool = True,
    include_strategy_performance: bool = True,
    ttl_s: float = DEFAULT_CONTEXT_TTL_S,
) -> str:
    """Assemble the standard analyst seed-context block for `asset`.

    Cached per-asset for `ttl_s` seconds so stacked callers within one
    tick (agentic_bridge + polymarket_analyst scanning multiple markets)
    share the work. Pass ttl_s=0 to bypass the cache.

    Thin wrapper over build_evidence_document — delegates to the
    structured builder and flattens to the analyst-prompt string.
    """
    asset_key = asset.upper()
    now = time.time()
    mask = (
        int(include_scanner), int(include_traces), int(include_news),
        int(include_fear_greed), int(include_graph), int(include_body_controls),
        int(include_agent_votes), int(include_recent_critiques),
        int(include_open_positions), int(include_capital_state),
        int(include_strategy_performance),
    )
    cache_key = f"{asset_key}:{mask}"
    if ttl_s > 0:
        with _CACHE_LOCK:
            hit = _CACHE.get(cache_key)
            if hit and hit[0] > now:
                return hit[1]

    doc = build_evidence_document(
        asset_key,
        include_scanner=include_scanner,
        include_traces=include_traces,
        include_news=include_news,
        include_fear_greed=include_fear_greed,
        include_graph=include_graph,
        include_body_controls=include_body_controls,
        include_agent_votes=include_agent_votes,
        include_recent_critiques=include_recent_critiques,
        include_open_positions=include_open_positions,
        include_capital_state=include_capital_state,
        include_strategy_performance=include_strategy_performance,
    )
    block = doc.to_prompt()
    if ttl_s > 0:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (now + ttl_s, block)
    return block


def clear_cache() -> None:
    """Test helper — empty the TTL cache."""
    with _CACHE_LOCK:
        _CACHE.clear()
