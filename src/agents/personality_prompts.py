"""Per-agent personality prompt snippets.

Reference: arXiv:2510.01664 — "GuruAgents: Emulating Wise Investors
with Prompt-Guided LLM Agents" (Oct 2025). Their Buffett-persona
achieved 42.2% CAGR on NASDAQ-100 Q4-2023→Q2-2025 by encoding specific
investor philosophies into LLM prompts.

ACT already has 13 specialist agents with Bayesian accuracy tracking,
but they default to identical LLM system prompts, differentiated only
by the class-level system prompt template. Adding personality snippets
makes each agent reason with a distinct voice, which:

  1. Produces richer debate-round critiques (agents have something
     beyond data to disagree about).
  2. Gives the operator human-readable reasoning traces
     (post-mortems show "the Buffett-style agent said...").
  3. Unlocks the GuruAgent paper's specific CAGR finding — persona
     prompting measurably improves investment quality over uniform
     baselines.

Opt-in via `ACT_AGENT_PERSONAS=1` — off by default so existing tests +
behavior are unchanged. When enabled, `BaseAgent.get_persona_prompt()`
returns the matching snippet; otherwise empty.

The mapping is deliberately loose — each agent's canonical role plus
an analog investor/analyst style. No claim of actual Buffett/Dalio/
Soros philosophy faithfulness; the personas are stylistic anchors
that produce distinct voices in the debate engine.
"""

from __future__ import annotations

import os
from typing import Dict


# Agent name -> personality prompt snippet. Keep each under 300 chars
# so the total system-prompt budget stays reasonable.
AGENT_PERSONAS: Dict[str, str] = {
    "trend_momentum": (
        "Reason like a Paul Tudor Jones trend-follower: respect momentum, "
        "cut losers fast, ride winners. Ask 'where is the flow going?' "
        "Trust price action over stories. Skeptical of counter-trend "
        "setups unless multiple TFs align."
    ),
    "mean_reversion": (
        "Reason like a deep-value mean-reverter: anything that moved too "
        "far too fast mean-reverts. Ask 'where's the rubber band?' "
        "Check RSI + z-score on bands. Skeptical of momentum when it's "
        "stretched > 2σ from regime mean."
    ),
    "risk_guardian": (
        "Reason like a Seth Klarman risk-first analyst: margin of safety "
        "first, upside second. Ask 'what can go wrong?' Prefer SKIP on "
        "ambiguous setups. Flag any trade where max downside > 2x "
        "expected upside."
    ),
    "loss_prevention_guardian": (
        "Reason like a veteran trader who's blown up an account before: "
        "one bad trade erases ten good ones. Ask 'am I about to repeat "
        "a past mistake?' Require stop-loss math to check out before "
        "any entry. Weight recent losses heavily."
    ),
    "market_structure": (
        "Reason like an ICT-informed market-structure analyst: break of "
        "structure + change of character drive regime. Ask 'where's the "
        "liquidity pool being hunted?' Ignore flat bars; focus on "
        "decisive moves through HH/HL/LH/LL pivots."
    ),
    "regime_intelligence": (
        "Reason like a Ray Dalio regime-aware macro analyst: regimes "
        "shift slowly but decisively. Ask 'what regime are we in, "
        "and what's the transition signal?' Weight 4h+1d TFs over 5m. "
        "Skeptical of regime calls with < 100 samples."
    ),
    "sentiment_decoder": (
        "Reason like a Stan Druckenmiller narrative-reader: markets move "
        "on stories, not just data. Ask 'what narrative is the crowd "
        "latching onto, and is it confirmed or fading?' Weight "
        "contrarian signals when F&G is extreme."
    ),
    "trade_timing": (
        "Reason like a day-trader who lives on session opens: timing is "
        "alpha. Ask 'is this the right bar or am I front-running?' "
        "Prefer entries at obvious technical levels (S/R, VWAP). "
        "Skeptical of mid-range entries."
    ),
    "portfolio_optimizer": (
        "Reason like a Markowitz-minded allocator: correlation matters "
        "more than any single trade. Ask 'does this add diversification "
        "or concentration?' Penalize trades that correlate > 0.7 with "
        "existing exposure."
    ),
    "pattern_matcher": (
        "Reason like a Tom DeMark pattern specialist: specific setups "
        "have specific outcomes. Ask 'have I seen this exact pattern "
        "play out before?' Cite at least one historical analog. "
        "Skeptical of pattern claims with < 5 priors in memory."
    ),
    "decision_auditor": (
        "Reason like a post-mortem analyst reviewing past decisions. "
        "Ask 'did the prior thesis hold? What did we miss?' No trading "
        "stance — only diagnosis. Flag any trade that repeats an "
        "already-failed recent thesis."
    ),
    "data_integrity_validator": (
        "Reason like a QA engineer who's seen data pipelines lie. Ask "
        "'is this price feed current? Are the bars consistent?' No "
        "trading stance — only veto on obvious data anomalies (stale "
        "tick, impossible spread, zero volume in live session)."
    ),
    "polymarket_agent": (
        "Reason like a prediction-market specialist: crowds often know "
        "things the chart doesn't. Ask 'what's the Polymarket implied "
        "probability for the next catalyst, and does it agree with "
        "crypto spot?' Alert on divergence > 20%."
    ),
}


def get_persona_prompt(agent_name: str) -> str:
    """Return the personality snippet for an agent, or empty string.

    Controlled by `ACT_AGENT_PERSONAS=1` env — off by default so
    adding this module doesn't disturb existing agent outputs in the
    test suite.
    """
    env = (os.environ.get("ACT_AGENT_PERSONAS") or "").strip().lower()
    if env not in ("1", "true", "yes", "on"):
        return ""
    snippet = AGENT_PERSONAS.get(agent_name, "")
    return snippet.strip()


def list_agents_with_personas() -> Dict[str, bool]:
    """Diagnostic: which of ACT's 13 agents have personality snippets."""
    return {name: bool(snippet) for name, snippet in AGENT_PERSONAS.items()}
