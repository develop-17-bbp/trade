"""
Web context builder — summary-first composition over ACT's existing fetchers.

Design principle (per operator direction this session):
  The less noise in the main LLM context, the longer and more effectively
  the agentic trade loop can work. Every tool here returns a COMPACT digest
  (<= ~500 chars), not raw payload. Raw fetcher outputs stay in their own
  fetcher caches / warm_store; only the digest flows into the parent's
  context window.

Reuses existing infrastructure — does not fetch anything directly:

  * src/data/news_fetcher.py::NewsFetcher.fetch_all       → headline list
  * src/data/polymarket_fetcher.py::PolymarketFetcher     → prob. divergences
  * src/data/on_chain_fetcher.py                          → whale flows, liq
  * src/data/institutional_fetcher.py::
        get_all_institutional(asset)                      → cross-exchange
  * src/data/economic_intelligence.py::
        get_llm_context_block()                           → 12-layer summary
  * src/agents/sentiment_decoder_agent.py                 → fused sentiment
  * src/trading/macro_bias.py::compute_macro_bias         → signed tilt

Each public function here returns a `WebDigest` — a small dataclass with
a `summary` field the LLM reads and a `raw_ref` field the auditor can use
post-hoc to re-fetch the full payload if needed. The LLM only sees
`summary`; `raw_ref` is logged to warm_store alongside the decision.

Fallbacks: every import is guarded. If a fetcher is missing or errors,
the digest comes back with `summary="unavailable"` instead of raising.
A tool outage never blocks a trade cycle.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Per-tool hard timeout in seconds. A web fetcher that takes longer than
# this returns "unavailable" rather than blocking the agentic loop.
DEFAULT_TIMEOUT_S = float(os.getenv("ACT_WEB_TOOL_TIMEOUT_S", "5.0"))

# Process-wide TTL cache: key -> (expires_ts, digest). Avoids re-hitting
# external services when the LLM asks the same question twice in one cycle.
_CACHE: Dict[str, tuple] = {}
_DEFAULT_TTL_S = float(os.getenv("ACT_WEB_CACHE_TTL_S", "60"))


@dataclass
class WebDigest:
    """One tool's compact output — fits in a chat message comfortably."""
    source: str
    summary: str                         # <= ~500 chars
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5              # how much the LLM should weight this
    raw_ref: Optional[str] = None        # pointer to raw payload (log id etc.)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "summary": self.summary[:800],
            "tags": list(self.tags),
            "confidence": round(self.confidence, 2),
            "raw_ref": self.raw_ref,
        }


def _cached(key: str, fn: Callable[[], WebDigest], ttl_s: float = _DEFAULT_TTL_S) -> WebDigest:
    now = time.time()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1]
    try:
        digest = _with_timeout(fn, DEFAULT_TIMEOUT_S)
    except Exception as e:
        logger.debug("web_context %s failed: %s", key, e)
        digest = WebDigest(source=key.split(":", 1)[0], summary="unavailable", confidence=0.0)
    _CACHE[key] = (now + ttl_s, digest)
    return digest


def _with_timeout(fn: Callable[[], WebDigest], timeout_s: float) -> WebDigest:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except _Timeout:
            future.cancel()
            raise TimeoutError(f"web tool exceeded {timeout_s}s")


# ── Individual tool wrappers ────────────────────────────────────────────


def get_news_digest(asset: str, hours: int = 6, limit: int = 30) -> WebDigest:
    """Top-N headlines scored by recency + impact. Summary = 3-line digest."""
    key = f"news:{asset}:{hours}"

    def _fn() -> WebDigest:
        try:
            from src.data.news_fetcher import NewsFetcher
            fetcher = NewsFetcher()
            items = fetcher.fetch_all(query=asset.lower(), limit=limit) or []
        except Exception as e:
            return WebDigest(source="news", summary=f"fetcher unavailable: {e}", confidence=0.0)

        cutoff = time.time() - hours * 3600
        recent = [it for it in items if getattr(it, "timestamp", 0) >= cutoff]
        if not recent:
            return WebDigest(source="news", summary=f"no headlines in last {hours}h", confidence=0.2)

        # Count event types and pick top-3 titles for the digest.
        type_counts: Dict[str, int] = {}
        for it in recent:
            t = getattr(it, "event_type", "general") or "general"
            type_counts[t] = type_counts.get(t, 0) + 1
        top3 = [getattr(it, "title", "") for it in recent[:3]]
        type_line = ", ".join(f"{t}={n}" for t, n in sorted(type_counts.items(), key=lambda kv: -kv[1])[:5])
        summary = (
            f"{len(recent)} {asset} headlines in {hours}h. "
            f"Event mix: {type_line}. "
            f"Top: {' | '.join(t[:80] for t in top3)}"
        )
        return WebDigest(
            source="news", summary=summary,
            tags=[t for t in type_counts.keys()],
            confidence=min(1.0, 0.3 + 0.05 * len(recent)),
        )

    return _cached(key, _fn)


def get_sentiment_digest(asset: str) -> WebDigest:
    """Fused sentiment — reuses sentiment_decoder_agent's aggregation."""
    key = f"sentiment:{asset}"

    def _fn() -> WebDigest:
        try:
            from src.agents.sentiment_decoder_agent import SentimentDecoderAgent
            agent = SentimentDecoderAgent()
            # SentimentDecoderAgent.analyze returns an AgentVote-like object.
            vote = agent.analyze({"asset": asset}, {})
        except Exception as e:
            return WebDigest(source="sentiment", summary=f"unavailable: {e}", confidence=0.0)

        direction = getattr(vote, "direction", 0) or 0
        conf = float(getattr(vote, "confidence", 0.5) or 0.5)
        rationale = getattr(vote, "rationale", "") or getattr(vote, "reason", "")
        label = {1: "bullish", -1: "bearish", 0: "neutral"}.get(direction, "neutral")
        return WebDigest(
            source="sentiment",
            summary=f"fused sentiment: {label} (conf {conf:.2f}). {rationale[:200]}",
            tags=[label],
            confidence=conf,
        )

    return _cached(key, _fn)


def get_polymarket_digest() -> WebDigest:
    """Prediction-market divergences — signals implied-probability mismatches."""
    key = "polymarket:divergences"

    def _fn() -> WebDigest:
        try:
            from src.data.polymarket_fetcher import PolymarketFetcher
            fetcher = PolymarketFetcher()
            summary_dict = fetcher.get_summary_for_dashboard()
        except Exception as e:
            return WebDigest(source="polymarket", summary=f"unavailable: {e}", confidence=0.0)

        if not summary_dict:
            return WebDigest(source="polymarket", summary="no markets", confidence=0.1)
        # Compact a dict of markets into one line of counts + top-3.
        n = len(summary_dict) if isinstance(summary_dict, (dict, list)) else 0
        top = ""
        if isinstance(summary_dict, dict):
            items = list(summary_dict.items())[:3]
            top = " | ".join(f"{k}={v}" for k, v in items)
        return WebDigest(
            source="polymarket",
            summary=f"{n} markets tracked. {top}",
            confidence=0.5,
        )

    return _cached(key, _fn)


def get_institutional_digest(asset: str) -> WebDigest:
    """Cross-exchange prices, stablecoin flows, L/S ratio — one-call aggregate."""
    key = f"institutional:{asset}"

    def _fn() -> WebDigest:
        try:
            from src.data.institutional_fetcher import InstitutionalFetcher
            fetcher = InstitutionalFetcher()
            data = fetcher.get_all_institutional(asset) or {}
        except Exception as e:
            return WebDigest(source="institutional", summary=f"unavailable: {e}", confidence=0.0)

        if not data:
            return WebDigest(source="institutional", summary="no data", confidence=0.1)
        # Pick the top-5 highest-magnitude signals for the line.
        numeric = [(k, v) for k, v in data.items() if isinstance(v, (int, float))]
        numeric.sort(key=lambda kv: abs(kv[1]), reverse=True)
        line = ", ".join(f"{k}={v:.3f}" for k, v in numeric[:5])
        return WebDigest(
            source="institutional",
            summary=f"{asset} institutional: {line}",
            confidence=0.6,
        )

    return _cached(key, _fn)


def get_macro_digest() -> WebDigest:
    """12-layer economic intelligence — delegated to existing context builder."""
    key = "macro:ei_block"

    def _fn() -> WebDigest:
        try:
            from src.data.economic_intelligence import EconomicIntelligence
            ei = EconomicIntelligence()
            block = ei.get_llm_context_block()
        except Exception as e:
            return WebDigest(source="macro", summary=f"unavailable: {e}", confidence=0.0)
        block = (block or "").strip()
        if not block:
            return WebDigest(source="macro", summary="no macro data", confidence=0.1)
        # Already LLM-ready; trim to soft cap so it doesn't dominate context.
        return WebDigest(
            source="macro",
            summary=block if len(block) <= 800 else block[:800] + "...",
            confidence=0.7,
        )

    return _cached(key, _fn, ttl_s=300.0)  # macro changes slowly — 5min cache


def get_fear_greed_digest() -> WebDigest:
    """Fear & Greed index — piggybacks on sentiment_decoder which already reads it."""
    key = "fear_greed:latest"

    def _fn() -> WebDigest:
        # Try pulling from economic_intelligence (layer may cache it) or
        # fall back to the sentiment decoder's ext_feats dict.
        try:
            from src.data.economic_intelligence import EconomicIntelligence
            ei = EconomicIntelligence()
            data = ei.fetch_all_now() or {}
            fg = None
            for _name, payload in data.items():
                if isinstance(payload, dict) and "fear_greed" in payload:
                    fg = payload["fear_greed"]
                    break
                if isinstance(payload, dict) and "fear_greed_index" in payload:
                    fg = payload["fear_greed_index"]
                    break
            if fg is None:
                return WebDigest(source="fear_greed", summary="no index", confidence=0.1)
            val = float(fg) if isinstance(fg, (int, float)) else 50.0
            label = (
                "extreme fear" if val < 25 else
                "fear" if val < 45 else
                "neutral" if val < 55 else
                "greed" if val < 75 else
                "extreme greed"
            )
            return WebDigest(
                source="fear_greed",
                summary=f"F&G {val:.0f} ({label})",
                tags=[label.replace(" ", "_")],
                confidence=0.8,
            )
        except Exception as e:
            return WebDigest(source="fear_greed", summary=f"unavailable: {e}", confidence=0.0)

    return _cached(key, _fn, ttl_s=3600.0)  # index updates hourly at most


# ── Parallel bundle fetch ───────────────────────────────────────────────


def fetch_bundle(asset: str, include: Optional[List[str]] = None) -> Dict[str, WebDigest]:
    """Fan out every Tier-1 tool in parallel, collect digests, return dict.

    Default include = all tools. Passing a subset keeps the context even
    leaner — e.g. on a low-conviction tick the LLM only needs `macro` +
    `sentiment` + `fear_greed`.
    """
    tools = {
        "news": lambda: get_news_digest(asset),
        "sentiment": lambda: get_sentiment_digest(asset),
        "polymarket": lambda: get_polymarket_digest(),
        "institutional": lambda: get_institutional_digest(asset),
        "macro": lambda: get_macro_digest(),
        "fear_greed": lambda: get_fear_greed_digest(),
    }
    if include is not None:
        tools = {k: v for k, v in tools.items() if k in include}
    out: Dict[str, WebDigest] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(tools))) as pool:
        futures = {name: pool.submit(fn) for name, fn in tools.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result(timeout=DEFAULT_TIMEOUT_S + 1.0)
            except Exception as e:
                logger.debug("fetch_bundle %s failed: %s", name, e)
                out[name] = WebDigest(source=name, summary="unavailable", confidence=0.0)
    return out


def bundle_to_prompt_block(bundle: Dict[str, WebDigest]) -> str:
    """Format a fetched bundle as a single LLM-ready prompt block."""
    lines = ["## WEB CONTEXT DIGESTS"]
    for name, d in bundle.items():
        lines.append(f"- [{name}] (conf {d.confidence:.2f}): {d.summary}")
    return "\n".join(lines)


def clear_cache() -> None:
    """Test helper + ops knob — drop the TTL cache."""
    _CACHE.clear()
