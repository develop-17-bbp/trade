"""Privacy audit — who sees what across the agentic workflow.

Reference: arXiv:2603.04902 — "AgentSCOPE: Evaluating Contextual
Privacy Across Agentic Workflows" (Mar 2026). They observe that in
multi-agent LLM systems, sensitive fields leak across agents via
shared context / memory / tool outputs — often unintentionally.

ACT's risk: the scanner brain's ScanReport is read by the analyst
brain AND the agentic_bridge AND the polymarket_analyst. If the
scanner's prompt accidentally echoes operator-secret data in its
output, three downstream consumers see it.

This module provides a **data-flow trace helper**: given a set of
LLM prompts + outputs + memory writes across a tick, produce a
report of which fields propagate to which consumer. Used by the
`/agent-post-mortem` skill and by periodic compliance audits.

Not a firewall — this is *diagnostic*. Actual leakage prevention is
the `output_scrubber` module's job. This one answers "what COULD
have leaked if the scrubber missed something?"
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Canonical downstream-consumer catalog. When a new tick starts, we
# know which subsystems will read the scanner's output, which will
# read the analyst's TradePlan, etc. Keeping the map here makes
# audits predictable.

CONSUMERS_BY_PRODUCER: Dict[str, List[str]] = {
    "scanner_brain": [
        "analyst_brain",
        "brain_memory_writer",
        "agentic_bridge",
        "polymarket_analyst",
        "body_controller",
        "context_builders_cache",
    ],
    "analyst_brain": [
        "trade_verifier",
        "warm_store_writer",
        "credit_assigner",
        "thompson_bandit",
        "orchestrator",
        "executor",
    ],
    "agent_debate": [
        "analyst_brain",
        "warm_store_writer",
        "decision_auditor",
    ],
    "tool_invocation": [
        "analyst_brain",
        "warm_store_writer",
    ],
}


SENSITIVE_FIELD_PATTERNS: Set[str] = {
    # Prompt-level leakage risks
    "DASHBOARD_API_KEY", "NEWSAPI_KEY", "CRYPTOPANIC_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "robinhood_account_id", "rh_signing_key", "wallet_private_key",
    # Operator-identifying fields
    "operator_email", "operator_phone",
}


class PrivacyFlow:
    """One producer -> consumer edge with the sensitive-field tags
    observed in the payload."""

    def __init__(
        self,
        producer: str,
        consumer: str,
        sensitive_tags: Optional[List[str]] = None,
        payload_bytes: int = 0,
    ) -> None:
        self.producer = producer
        self.consumer = consumer
        self.sensitive_tags = list(sensitive_tags or [])
        self.payload_bytes = payload_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "producer": self.producer,
            "consumer": self.consumer,
            "sensitive_tags": list(self.sensitive_tags),
            "payload_bytes": self.payload_bytes,
            "risk": self.risk_level(),
        }

    def risk_level(self) -> str:
        if not self.sensitive_tags:
            return "none"
        if len(self.sensitive_tags) >= 3:
            return "high"
        return "medium"


def scan_payload_for_sensitive_tags(payload: Any) -> List[str]:
    """Walk a payload for keys/values that match sensitive patterns.

    Returns the list of sensitive-tag names that appeared. Does NOT
    return the values themselves (that would defeat the audit).
    """
    found: Set[str] = set()
    _walk_for_tags(payload, found)
    return sorted(found)


def _walk_for_tags(node: Any, found: Set[str]) -> None:
    if node is None:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k in SENSITIVE_FIELD_PATTERNS:
                found.add(k)
            _walk_for_tags(v, found)
        return
    if isinstance(node, list):
        for x in node:
            _walk_for_tags(x, found)
        return
    if isinstance(node, str):
        for pat in SENSITIVE_FIELD_PATTERNS:
            if pat in node:
                found.add(pat)


def audit_flow(
    producer: str, payload: Any,
    consumers: Optional[List[str]] = None,
) -> List[PrivacyFlow]:
    """For one producer's payload, produce per-consumer flow records.

    `consumers=None` uses the default CONSUMERS_BY_PRODUCER mapping.
    """
    if consumers is None:
        consumers = CONSUMERS_BY_PRODUCER.get(producer, [])
    tags = scan_payload_for_sensitive_tags(payload)
    try:
        size = len(str(payload))
    except Exception:
        size = 0
    return [PrivacyFlow(producer, c, tags, size) for c in consumers]


def summarize_audit(flows: List[PrivacyFlow]) -> Dict[str, Any]:
    """Roll up a batch of flows into a compact diagnostic."""
    total = len(flows)
    by_risk: Dict[str, int] = {"none": 0, "medium": 0, "high": 0}
    all_tags: Set[str] = set()
    at_risk: List[Dict[str, Any]] = []
    for f in flows:
        r = f.risk_level()
        by_risk[r] = by_risk.get(r, 0) + 1
        for t in f.sensitive_tags:
            all_tags.add(t)
        if r != "none":
            at_risk.append(f.to_dict())
    return {
        "total_flows": total,
        "by_risk_level": by_risk,
        "all_sensitive_tags": sorted(all_tags),
        "any_high_risk": by_risk.get("high", 0) > 0,
        "at_risk_flows": at_risk[:20],     # cap for message size
    }
