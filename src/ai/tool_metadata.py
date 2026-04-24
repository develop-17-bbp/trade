"""FinToolBench-inspired 3-axis tool classification.

Reference: arXiv:2603.08262 "FinToolBench: Evaluating LLM Agents for
Real-World Financial Tool Use" (Mar 2026). They classify 760 financial
tools across three dimensions to help LLM agents pick the right tool
and to audit domain alignment.

ACT adopts the same classification for its 30+ tools so:
  1. The analyst can filter tools by timeliness (don't call a daily-
     aggregated tool when you need a tick-level price).
  2. The audit trail can show tool-to-intent match for every decision.
  3. The compliance reviewer can see at a glance which tools are
     crypto-native vs equity-borrowed vs experimental.

Axes:
  * timeliness    — how fresh is the data? realtime / minute / hour /
                    daily / static
  * intent_type   — what does the tool DO? data-fetch / analysis /
                    action / query / diagnostic
  * regulatory    — does this tool's output have regulatory weight?
                    crypto_native / equity_borrowed / experimental /
                    internal

Missing entries fall back to 'unknown' — no crash, just no metadata
for that tool. Callers should treat 'unknown' as 'proceed with extra
caution' rather than 'equivalent to classified.'
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


Timeliness = Literal["realtime", "minute", "hour", "daily", "static", "unknown"]
IntentType = Literal["data_fetch", "analysis", "action", "query",
                     "diagnostic", "unknown"]
RegulatoryDomain = Literal["crypto_native", "equity_borrowed",
                           "experimental", "internal", "unknown"]


class ToolClassification:
    """Plain class (not @dataclass) so it survives Py3.14's dataclass
    anonymous-module-loading quirk seen in the skills system."""

    def __init__(
        self,
        timeliness: Timeliness = "unknown",
        intent_type: IntentType = "unknown",
        regulatory: RegulatoryDomain = "unknown",
        notes: str = "",
    ) -> None:
        self.timeliness = timeliness
        self.intent_type = intent_type
        self.regulatory = regulatory
        self.notes = notes

    def to_dict(self) -> Dict[str, str]:
        return {
            "timeliness": self.timeliness,
            "intent_type": self.intent_type,
            "regulatory": self.regulatory,
            "notes": self.notes,
        }

    def is_fully_classified(self) -> bool:
        return "unknown" not in (self.timeliness, self.intent_type, self.regulatory)


# ── Registry of per-tool classifications ───────────────────────────────
#
# Keyed by tool name (the string the analyst's tool-use loop sees).
# Add entries when new tools are registered in trade_tools.py or an
# MCP server is wired.

TOOL_CLASSIFICATIONS: Dict[str, ToolClassification] = {
    # ── In-process data ────────────────────────────────────────────
    "get_recent_bars": ToolClassification(
        timeliness="minute", intent_type="data_fetch",
        regulatory="crypto_native", notes="OHLCV bars from cached price fetcher",
    ),
    "get_support_resistance": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="crypto_native",
    ),
    "get_orderbook_imbalance": ToolClassification(
        timeliness="realtime", intent_type="data_fetch",
        regulatory="crypto_native",
    ),
    "get_macro_bias": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="equity_borrowed",
        notes="DXY/VIX/CPI synthesis — equity-native data adapted for crypto",
    ),
    "get_regime": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="crypto_native",
    ),
    "query_recent_trades": ToolClassification(
        timeliness="realtime", intent_type="query",
        regulatory="internal",
    ),
    "get_readiness_state": ToolClassification(
        timeliness="realtime", intent_type="diagnostic",
        regulatory="internal",
    ),
    "estimate_impact": ToolClassification(
        timeliness="minute", intent_type="analysis",
        regulatory="experimental",
        notes="Monte-Carlo over last 100 outcomes",
    ),
    "search_strategy_repo": ToolClassification(
        timeliness="daily", intent_type="query",
        regulatory="internal",
    ),
    "submit_trade_plan": ToolClassification(
        timeliness="realtime", intent_type="action",
        regulatory="crypto_native",
        notes="THE ONLY action tool; authority-gated",
    ),

    # ── Quantitative math tools ───────────────────────────────────
    "fit_ou_process": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="equity_borrowed",
        notes="OU mean-reversion — classical quant, applied to crypto",
    ),
    "hurst_exponent": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="equity_borrowed",
    ),
    "kalman_trend": ToolClassification(
        timeliness="minute", intent_type="analysis",
        regulatory="equity_borrowed",
    ),
    "hmm_regime": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="experimental",
    ),
    "hawkes_clustering": ToolClassification(
        timeliness="hour", intent_type="analysis",
        regulatory="experimental",
    ),
    "test_cointegration": ToolClassification(
        timeliness="daily", intent_type="analysis",
        regulatory="equity_borrowed",
    ),

    # ── Web / news tools ──────────────────────────────────────────
    "get_web_context": ToolClassification(
        timeliness="minute", intent_type="data_fetch",
        regulatory="crypto_native",
    ),
    "get_news_digest": ToolClassification(
        timeliness="minute", intent_type="data_fetch",
        regulatory="crypto_native",
    ),
    "get_fear_greed": ToolClassification(
        timeliness="hour", intent_type="data_fetch",
        regulatory="crypto_native",
    ),

    # ── Knowledge graph ───────────────────────────────────────────
    "query_knowledge_graph": ToolClassification(
        timeliness="realtime", intent_type="query",
        regulatory="internal",
        notes="Real-time graph over live data streams",
    ),

    # ── Agent sub-queries ─────────────────────────────────────────
    "ask_risk_guardian": ToolClassification(
        timeliness="realtime", intent_type="analysis",
        regulatory="internal",
    ),
    "ask_loss_prevention": ToolClassification(
        timeliness="realtime", intent_type="analysis",
        regulatory="internal",
    ),
    "ask_trend_momentum": ToolClassification(
        timeliness="realtime", intent_type="analysis",
        regulatory="internal",
    ),
    "ask_mean_reversion": ToolClassification(
        timeliness="realtime", intent_type="analysis",
        regulatory="internal",
    ),
    "ask_sentiment_decoder": ToolClassification(
        timeliness="realtime", intent_type="analysis",
        regulatory="internal",
    ),
}


def classify(tool_name: str) -> ToolClassification:
    """Look up a tool's classification. Unknown tools return an 'unknown'
    classification rather than raising — callers should treat it as a
    proceed-with-caution signal, not a crash."""
    hit = TOOL_CLASSIFICATIONS.get(tool_name)
    if hit is not None:
        return hit
    logger.debug("tool_metadata: no classification for %r", tool_name)
    return ToolClassification()


def filter_tools_by_intent(intent: IntentType) -> Dict[str, ToolClassification]:
    """Return all classified tools with the requested intent type."""
    return {k: v for k, v in TOOL_CLASSIFICATIONS.items()
            if v.intent_type == intent}


def filter_tools_by_timeliness(required_freshness: Timeliness) -> Dict[str, ToolClassification]:
    """Return tools fresh enough to satisfy `required_freshness`.

    Ordering: realtime > minute > hour > daily > static. A request for
    'hour' gets realtime + minute + hour tools; 'static' gets everything.
    """
    ranks = {"realtime": 4, "minute": 3, "hour": 2, "daily": 1,
             "static": 0, "unknown": 0}
    want = ranks.get(required_freshness, 0)
    return {k: v for k, v in TOOL_CLASSIFICATIONS.items()
            if ranks.get(v.timeliness, 0) >= want}


def audit_coverage() -> Dict[str, Any]:
    """Diagnostic — how many tools are fully classified?"""
    total = len(TOOL_CLASSIFICATIONS)
    full = sum(1 for v in TOOL_CLASSIFICATIONS.values()
               if v.is_fully_classified())
    return {
        "total_tools": total,
        "fully_classified": full,
        "coverage_pct": round(100.0 * full / max(total, 1), 1),
        "by_regulatory": _count_by("regulatory"),
        "by_timeliness": _count_by("timeliness"),
        "by_intent": _count_by("intent_type"),
    }


def _count_by(attr: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in TOOL_CLASSIFICATIONS.values():
        k = getattr(v, attr, "unknown")
        out[k] = out.get(k, 0) + 1
    return out
