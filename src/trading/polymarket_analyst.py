"""
Polymarket probability estimation via the Analyst brain.

Before this module, `skills/polymarket_hunt/action.py::_estimate_probability`
just returned the market-implied `yes_price`. That's zero edge — the
bot can never beat a market that already reflects its own view. The
only way a shadow plan becomes a winning plan is if the Analyst's
estimate MATERIALLY DIFFERS from the market's implied price.

This module gives the Analyst the full ACT context stack + the market
question and asks for a grounded probability estimate. The resulting
edge (estimated - market_implied) is what `polymarket_conviction.py`
tiers into sniper/normal/reject.

Grounding inputs:
  * Market question + yes_price + end_ts + volume_24h.
  * Current asset (BTC / ETH) state via brain_memory scanner report.
  * Recent news headlines via web_context.get_news_digest.
  * Fear & greed index.
  * Knowledge-graph digest for the relevant asset.

Output is a PolymarketProbabilityEstimate with:
  * estimated_yes_probability  — [0, 1]
  * confidence                 — [0, 1]  (LLM's self-reported)
  * rationale                  — ≤ 400 chars
  * edge                       — estimated - implied (signed)
  * source_model               — which model made the call
  * fallback_used              — bool

Safety:
  * Never raises. If the LLM is unreachable or its output can't be
    parsed, returns an estimate == implied (zero edge, conviction
    gate auto-rejects).
  * Output clamped to (0.01, 0.99) to avoid degenerate probabilities
    that would break EV math in the conviction gate.
  * Bounded prompt — market question truncated, context digests
    truncated, total prompt stays <2 KB.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


MIN_PROB = 0.01
MAX_PROB = 0.99


@dataclass
class PolymarketProbabilityEstimate:
    """Structured output from the analyst's estimation pass."""
    market_id: str
    question: str
    implied_yes_probability: float     # = market's yes_price
    estimated_yes_probability: float   # analyst's view
    edge: float                        # signed: positive if we think YES mispriced low
    confidence: float                  # analyst's self-reported [0, 1]
    rationale: str
    source_model: str = ""
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_id": self.market_id,
            "question": self.question[:200],
            "implied_yes_probability": round(self.implied_yes_probability, 4),
            "estimated_yes_probability": round(self.estimated_yes_probability, 4),
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale[:400],
            "source_model": self.source_model,
            "fallback_used": bool(self.fallback_used),
        }


# ── Context builders ───────────────────────────────────────────────────


def _build_context_block(asset: str) -> str:
    """Pull a compact context block from ACT's live streams for the
    analyst's prompt. Every source is optional; missing data just
    produces a thinner (but still valid) prompt."""
    parts = []
    try:
        from src.ai.brain_memory import get_scan_for_analyst
        scan = get_scan_for_analyst(asset, max_age_s=3600.0)
        if scan is not None:
            parts.append(
                f"## SCANNER ({int(scan.age_s())}s old)\n"
                f"opportunity_score={scan.opportunity_score:.0f}  "
                f"direction={scan.proposed_direction}\n"
                f"signals: {', '.join(scan.top_signals[:5]) or 'none'}\n"
                f"rationale: {scan.rationale[:200]}"
            )
    except Exception:
        pass

    try:
        from src.ai.web_context import get_news_digest
        news = get_news_digest(asset, hours=12)
        if news and news.summary and news.summary != "unavailable":
            parts.append(f"## NEWS\n{news.summary[:400]}")
    except Exception:
        pass

    try:
        from src.ai.web_context import get_fear_greed_digest
        fg = get_fear_greed_digest()
        if fg and fg.summary and fg.summary != "unavailable":
            parts.append(f"## FEAR/GREED\n{fg.summary[:120]}")
    except Exception:
        pass

    try:
        from src.ai.graph_rag import query_digest
        g = query_digest(asset, since_s=3600, max_chars=400)
        if g and not g.startswith("[graph disabled") and not g.startswith("[graph unavailable"):
            parts.append(g)
    except Exception:
        pass

    return "\n\n".join(parts) if parts else f"## CONTEXT\n(no fresh data for {asset})"


# ── Parser ─────────────────────────────────────────────────────────────


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_estimate(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first balanced JSON object from LLM output. Same
    greedy-then-shrink trick the agentic_trade_loop uses."""
    if not text:
        return None
    # Strip ```json ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    m = _JSON_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    for end in range(len(blob), 0, -1):
        try:
            return json.loads(blob[:end])
        except Exception:
            continue
    return None


def _clamp_prob(p: float) -> float:
    try:
        p = float(p)
    except Exception:
        return 0.5
    return max(MIN_PROB, min(MAX_PROB, p))


# ── Public API ─────────────────────────────────────────────────────────


def estimate_probability(
    market: Dict[str, Any],
    *,
    asset_hint: str = "BTC",
) -> PolymarketProbabilityEstimate:
    """Invoke the Analyst brain to estimate the true YES probability.

    Returns a zero-edge fallback if the LLM is unreachable / returns
    unparseable output — `polymarket_conviction.evaluate` will then
    auto-reject the market (no edge, no trade).
    """
    market_id = str(market.get("market_id") or market.get("id") or "?")
    question = str(market.get("question") or "")
    try:
        implied_yes = float(market.get("yes_price") or 0.5)
    except Exception:
        implied_yes = 0.5
    implied_yes = _clamp_prob(implied_yes)

    zero_edge = PolymarketProbabilityEstimate(
        market_id=market_id, question=question,
        implied_yes_probability=implied_yes,
        estimated_yes_probability=implied_yes,
        edge=0.0, confidence=0.0,
        rationale="estimator fell back to market-implied (LLM unavailable)",
        fallback_used=True,
    )

    if not question:
        return zero_edge

    # Heuristic: pull asset from the question if BTC/ETH mentioned, else use hint.
    q_lower = question.lower()
    if "bitcoin" in q_lower or "btc" in q_lower:
        asset = "BTC"
    elif "ethereum" in q_lower or "eth" in q_lower:
        asset = "ETH"
    else:
        asset = asset_hint.upper()

    context_block = _build_context_block(asset)

    system_prompt = (
        "You are ACT's probability estimator for Polymarket YES/NO "
        "markets. Given a market question + ACT's live context + the "
        "market's implied YES price, return your best estimate of the "
        "TRUE YES probability. If you have no edge, return the implied "
        "price verbatim — do NOT fabricate an edge. Be conservative: "
        "only deviate from the market when the context materially "
        "supports a different view. Your output MUST be ONE JSON "
        "object and nothing else."
    )

    user_prompt = (
        f"## MARKET\n"
        f"id: {market_id}\n"
        f"question: {question[:400]}\n"
        f"implied_yes_price: {implied_yes:.3f}\n"
        f"volume_24h_usd: {market.get('volume_24h', 0)}\n"
        f"end_ts: {market.get('end_ts', 0)}\n\n"
        f"{context_block}\n\n"
        f"## OUTPUT SCHEMA\n"
        f'{{"estimated_yes_probability": 0..1, "confidence": 0..1, '
        f'"rationale": "<=200 chars"}}'
    )

    try:
        from src.ai.dual_brain import analyze
        resp = analyze(user_prompt, extra_system=system_prompt,
                      allow_fallback=True)
    except Exception as e:
        logger.debug("polymarket_analyst: analyze() failed: %s", e)
        return zero_edge

    if resp is None or not resp.ok or not resp.text:
        return zero_edge

    parsed = _parse_estimate(resp.text)
    if not isinstance(parsed, dict):
        return zero_edge

    try:
        est = _clamp_prob(parsed.get("estimated_yes_probability"))
        conf = _clamp_prob(parsed.get("confidence", 0.5))
        rationale = str(parsed.get("rationale", ""))[:400] or \
            "(no rationale provided)"
    except Exception:
        return zero_edge

    return PolymarketProbabilityEstimate(
        market_id=market_id, question=question,
        implied_yes_probability=implied_yes,
        estimated_yes_probability=est,
        edge=round(est - implied_yes, 4),
        confidence=conf,
        rationale=rationale,
        source_model=getattr(resp, "model", "") or "",
        fallback_used=bool(getattr(resp, "fallback_used", False)),
    )
