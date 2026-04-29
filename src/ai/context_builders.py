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

    def to_prompt(self) -> str:
        """Flatten to the analyst-facing blob."""
        return "\n\n".join(s.to_prompt_block() for s in self.sections)

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
