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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "confidence": round(self.confidence, 3),
            "age_s": round(self.age_s, 1),
            "source": self.source,
        }

    def to_prompt_block(self) -> str:
        """Formatted block the analyst prompt consumes."""
        tags = []
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


def _traces_block(asset: str, limit: int = 3) -> str:
    try:
        from src.ai.brain_memory import get_recent_analyst_traces
        traces = get_recent_analyst_traces(asset, limit=limit) or []
    except Exception:
        return ""
    if not traces:
        return ""
    bullets = [
        f"- {t.direction}/{t.tier} size={t.size_pct}% verdict={t.verdict or '-'}"
        for t in traces
    ]
    return "## RECENT ANALYST DECISIONS\n" + "\n".join(bullets)


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
                confidence=0.7, source="brain_memory",
            ))
    if include_traces:
        c = _traces_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="RECENT_ANALYST_DECISIONS", content=c,
                confidence=0.6, source="brain_memory",
            ))
    if include_news:
        c = _news_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="NEWS", content=c,
                confidence=0.5, source="web_context.news",
            ))
    if include_fear_greed:
        c = _fear_greed_block()
        if c:
            doc.add(EvidenceSection(
                name="FEAR_GREED", content=c,
                confidence=0.6, source="web_context.fear_greed",
            ))
    if include_graph:
        c = _graph_block(asset_key)
        if c:
            doc.add(EvidenceSection(
                name="KNOWLEDGE_GRAPH", content=c,
                confidence=0.65, source="graph_rag.query_digest",
            ))
    if include_body_controls:
        c = _body_controls_block()
        if c:
            doc.add(EvidenceSection(
                name="BODY_CONTROLS", content=c,
                confidence=0.8, source="brain_to_body.controller",
            ))
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
