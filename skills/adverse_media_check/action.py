"""Skill: /adverse-media-check — AML-style negative-news screening.

Takes an entity (exchange, protocol, stablecoin, oracle address, person)
and screens recent headlines for critical/high-severity adverse events.

Typical invocation:
    python -m src.skills.cli run adverse-media-check entity=Binance hours=48

Output: structured SkillResult with
  * `adverse_found` boolean — any critical/high hits
  * `worst_item` — the headline + classification + source
  * `all_adverse` — full list of adverse headlines within the window
  * `recommendation` — PROCEED / CAUTION / BLOCK

Never blocks a trade directly — this is a DIAGNOSTIC skill. The
operator or a pre-trade hook can read its output and veto if
`recommendation` is BLOCK.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.skills.registry import SkillResult

logger = logging.getLogger(__name__)


def _fetch_recent_headlines(entity: str, hours: float) -> List[Dict[str, Any]]:
    """Pull headlines mentioning the entity from the last `hours`."""
    try:
        from src.data.news_fetcher import NewsFetcher
        nf = NewsFetcher()
        items = nf.fetch_all(query=entity.lower(), limit=200) or []
    except Exception as e:
        logger.debug("adverse_media: news_fetcher failed: %s", e)
        return []

    import time
    cutoff = time.time() - hours * 3600.0
    out: List[Dict[str, Any]] = []
    lower_entity = entity.lower()
    for it in items:
        title = getattr(it, "title", "") or ""
        ts = float(getattr(it, "timestamp", 0.0) or 0.0)
        if ts < cutoff:
            continue
        # Secondary filter — the aggregator may return broad matches;
        # we require the entity name to actually appear in the title.
        if lower_entity not in title.lower():
            continue
        out.append({
            "title": title,
            "source": getattr(it, "source", "unknown"),
            "timestamp": ts,
            "age_hours": (time.time() - ts) / 3600.0,
            "url": getattr(it, "url", ""),
        })
    return out


def _severity_at_or_above(severity: str, minimum: str) -> bool:
    rank = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}
    return rank.get(severity, 0) >= rank.get(minimum, 0)


def run(args: Optional[Dict[str, Any]] = None) -> SkillResult:
    args = dict(args or {})
    entity = str(args.get("entity") or "").strip()
    if not entity:
        return SkillResult(
            ok=False,
            error="missing required arg: entity=<name> (e.g. entity=Binance)",
        )

    try:
        hours = float(args.get("hours") or 48)
    except (TypeError, ValueError):
        hours = 48
    min_severity = str(args.get("min_severity") or "high").lower()
    if min_severity not in ("critical", "high", "medium", "low"):
        min_severity = "high"

    try:
        from src.ai.news_risk_classifier import (
            classify_risk_event, summarize_risk_distribution,
        )
    except Exception as e:
        return SkillResult(ok=False, error=f"classifier import failed: {e}")

    headlines = _fetch_recent_headlines(entity, hours)
    if not headlines:
        return SkillResult(
            ok=True,
            message=(f"no headlines found for entity={entity!r} in last "
                     f"{hours:.0f}h — either entity is quiet or news "
                     "sources unreachable"),
            data={
                "entity": entity, "window_hours": hours,
                "min_severity": min_severity,
                "adverse_found": False,
                "recommendation": "PROCEED",
                "total_headlines": 0,
                "adverse_count": 0,
                "all_adverse": [],
                "worst_item": None,
            },
        )

    # Classify each
    classified = []
    for h in headlines:
        c = classify_risk_event(h["title"])
        classified.append({**h, **c.to_dict()})

    adverse = [c for c in classified
               if _severity_at_or_above(c["severity"], min_severity)]
    adverse.sort(key=lambda x: x.get("severity_rank", 0), reverse=True)

    worst = adverse[0] if adverse else None
    dist = summarize_risk_distribution([h["title"] for h in headlines])

    # Recommendation logic
    if dist.get("any_critical"):
        recommendation = "BLOCK"
    elif dist.get("any_high") and len(adverse) >= 2:
        recommendation = "BLOCK"
    elif dist.get("any_high") or dist.get("by_severity", {}).get("medium", 0) >= 3:
        recommendation = "CAUTION"
    else:
        recommendation = "PROCEED"

    msg = (f"{entity}: {len(adverse)} adverse in {len(headlines)} "
           f"headlines over last {hours:.0f}h → {recommendation}")

    return SkillResult(
        ok=True,
        message=msg,
        data={
            "entity": entity,
            "window_hours": hours,
            "min_severity": min_severity,
            "total_headlines": len(headlines),
            "adverse_count": len(adverse),
            "adverse_found": len(adverse) > 0,
            "distribution": dist,
            "worst_item": worst,
            "all_adverse": adverse[:10],   # cap at 10 for message size
            "recommendation": recommendation,
        },
    )
