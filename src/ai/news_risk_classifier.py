"""News risk-event classifier.

Reference: arXiv:2508.10927 — "Modeling and Detecting Company Risks
from News" (Aug 2025). They propose a taxonomy of adverse risk events
extracted from financial news headlines, with severity grading.

ACT's existing `src/data/news_fetcher.py::EVENT_KEYWORDS` covers
general category tags (regulatory / hack / etf / macro / exchange /
adoption). This module adds:

  1. **Risk-specific sub-categories** — sanctions, bankruptcy, court
     rulings, depeg events, forks, liquidations, etc.
  2. **Severity grading** — critical / high / medium / low based on
     keyword strength and co-occurrence.
  3. **Structured output** — every classification carries both the
     event_type, severity, and the matched keywords so audit can trace
     why a headline was flagged.

Not a replacement for NewsFetcher's event_type. This runs on top:
  * NewsFetcher's event_type says "what domain is this?"
  * This module says "how bad is this?"

Kept pure-Python + regex — no extra deps. Good enough for most
financial adverse signals; LLM-based entity extraction is the next
tier up (deferred).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


Severity = str  # "critical" | "high" | "medium" | "low" | "none"


# Risk-event patterns — ordered by severity (critical checked first).
# Each entry: (event_type, severity, regex pattern, notes)
#
# Patterns use word boundaries to avoid false positives (e.g. "hack"
# inside "hackathon"). Case-insensitive at evaluation time.

RISK_PATTERNS: List[Tuple[str, Severity, str, str]] = [
    # ── CRITICAL — direct, quantified losses or enforcement ────────
    ("hack",       "critical", r"\b(?:hack(?:ed)?|breach(?:ed)?|exploit(?:ed)?|drained|stolen|siphon(?:ed)?)\b.*\b(?:million|billion|m\b|b\b|\$[\d,]+)\b",
     "Hack / exploit / theft with dollar figure"),
    ("bankruptcy", "critical", r"\b(?:bankrupt(?:cy)?|chapter\s*11|liquidat(?:ion|ed)|insolven(?:t|cy)|receiver(?:ship)?)\b",
     "Formal bankruptcy / insolvency filings"),
    ("sanctions",  "critical", r"\b(?:ofac|sanction(?:ed|s)|treasury\s+designat|asset\s+freeze|seiz(?:ed|ure))\b",
     "OFAC / treasury sanctions / asset seizure"),
    ("depeg",      "critical", r"\b(?:depeg(?:ging|ged)?|lost\s+(?:its?\s+)?peg|broke\s+(?:its?|the)?\s*peg|stablecoin\s+collapse)\b",
     "Stablecoin depeg events"),
    ("indictment", "critical", r"\b(?:indict(?:ed|ment)|arrest(?:ed)?|charges?\s+filed|criminal\s+charge)\b",
     "Criminal indictment or arrest"),

    # ── HIGH — significant adverse but not yet quantified ─────────
    ("hack",       "high",    r"\b(?:hack(?:ed)?|breach(?:ed)?|exploit(?:ed)?|vulnerability|zero[-\s]?day)\b",
     "Unquantified hack / breach / exploit"),
    ("regulation", "high",    r"\b(?:sec\s+(?:sues|charges)|cease\s+and\s+desist|enforcement\s+action|subpoena)\b",
     "Regulatory enforcement action"),
    ("ban",        "high",    r"\b(?:ban(?:ned|s)?|prohibit(?:ed|ion)|outlaw(?:ed)?)\b.*\b(?:crypto|btc|eth|bitcoin|ethereum|stablecoin)\b",
     "Government ban on crypto"),
    ("lawsuit",    "high",    r"\b(?:lawsuit|sued|sues|class\s+action|litigation)\b",
     "Civil lawsuit"),
    ("liquidation","high",    r"\b(?:mass\s+liquidation|cascad(?:e|ing)\s+liquidation|long\s+squeeze|short\s+squeeze)\b",
     "Market-wide liquidation cascade"),
    ("fork_contentious", "high", r"\b(?:contentious\s+fork|chain\s+split|hard\s+fork\s+drama|community\s+split)\b",
     "Contentious hard fork"),

    # ── MEDIUM — warning signals, not immediate ───────────────────
    ("regulation", "medium",  r"\b(?:sec|cftc|finra|fincen|regulat(?:ion|or|ory)|compliance|kyc|aml)\b",
     "Regulatory mentions"),
    ("outage",     "medium",  r"\b(?:outage|downtime|maintenance|offline|unreachable)\b",
     "Exchange / infrastructure outage"),
    ("delisting",  "medium",  r"\b(?:delist(?:ed|ing)?|removed\s+from|pair\s+removal)\b",
     "Token delisting"),
    ("rug_pull",   "medium",  r"\b(?:rug\s*pull(?:ed)?|exit\s+scam|dev\s+abandoned)\b",
     "Rug-pull / exit scam"),
    ("court",      "medium",  r"\b(?:court\s+rul(?:ing|ed)|judge\s+rul(?:ing|ed)|appeal(?:ed|s)|verdict)\b",
     "Court ruling"),

    # ── LOW — ambient context ────────────────────────────────────
    ("partnership_loss", "low", r"\b(?:partnership\s+(?:dissolved|ended|terminat)|cut\s+ties|severed\s+relations)\b",
     "Partnership dissolution"),
    ("leadership_change", "low", r"\b(?:ceo\s+(?:resigns?|steps?\s+down|fired)|exec(?:utive)?\s+departure)\b",
     "Leadership change"),
]

# Pre-compile.
_COMPILED: List[Tuple[str, Severity, re.Pattern, str]] = [
    (t, s, re.compile(p, re.IGNORECASE), n)
    for t, s, p, n in RISK_PATTERNS
]

SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}


class RiskClassification:
    """Plain class (Py3.14 dataclass anonymous-load quirk)."""

    def __init__(
        self,
        event_type: str = "none",
        severity: Severity = "none",
        matched_keywords: Optional[List[str]] = None,
        notes: str = "",
    ) -> None:
        self.event_type = event_type
        self.severity = severity
        self.matched_keywords = list(matched_keywords or [])
        self.notes = notes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "severity_rank": SEVERITY_RANK.get(self.severity, 0),
            "matched_keywords": list(self.matched_keywords),
            "notes": self.notes,
        }


def classify_risk_event(headline: str) -> RiskClassification:
    """Classify a single headline. Returns highest-severity match.

    Pattern ordering (critical first) means the first match wins — the
    patterns below it would be a less-severe label for the same event.
    """
    if not headline:
        return RiskClassification()
    text = str(headline)
    for event_type, severity, pat, notes in _COMPILED:
        m = pat.search(text)
        if m:
            return RiskClassification(
                event_type=event_type,
                severity=severity,
                matched_keywords=[m.group(0)],
                notes=notes,
            )
    return RiskClassification()


def summarize_risk_distribution(headlines: List[str]) -> Dict[str, Any]:
    """Aggregate classification counts + highest-severity sample."""
    by_severity: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0,
                                    "low": 0, "none": 0}
    by_event: Dict[str, int] = {}
    worst: Optional[Dict[str, Any]] = None
    worst_rank = -1

    for h in headlines or []:
        c = classify_risk_event(h)
        by_severity[c.severity] = by_severity.get(c.severity, 0) + 1
        by_event[c.event_type] = by_event.get(c.event_type, 0) + 1
        rank = SEVERITY_RANK.get(c.severity, 0)
        if rank > worst_rank:
            worst_rank = rank
            worst = {"headline": h, **c.to_dict()}

    return {
        "total_classified": len(headlines or []),
        "by_severity": by_severity,
        "by_event_type": by_event,
        "worst_item": worst,
        "any_critical": by_severity.get("critical", 0) > 0,
        "any_high": by_severity.get("high", 0) > 0,
    }
