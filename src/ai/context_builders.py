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
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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
    """
    asset_key = asset.upper()
    now = time.time()
    # Key the cache by asset + the include-mask so a terse request doesn't
    # reuse a fat-cached value and vice versa.
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

    parts: List[str] = []
    if include_scanner:
        b = _scanner_block(asset_key)
        if b:
            parts.append(b)
    if include_traces:
        b = _traces_block(asset_key)
        if b:
            parts.append(b)
    if include_news:
        b = _news_block(asset_key)
        if b:
            parts.append(b)
    if include_fear_greed:
        b = _fear_greed_block()
        if b:
            parts.append(b)
    if include_graph:
        b = _graph_block(asset_key)
        if b:
            parts.append(b)
    if include_body_controls:
        b = _body_controls_block()
        if b:
            parts.append(b)

    block = "\n\n".join(parts)
    if ttl_s > 0:
        with _CACHE_LOCK:
            _CACHE[cache_key] = (now + ttl_s, block)
    return block


def clear_cache() -> None:
    """Test helper — empty the TTL cache."""
    with _CACHE_LOCK:
        _CACHE.clear()
