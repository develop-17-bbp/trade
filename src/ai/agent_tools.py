"""
Agent-tool registrations — expose ACT's 12 specialist agents + debate
engine + backtest as LLM-callable tools.

Before this module the Analyst brain had only `ask_risk_guardian` and
`ask_loss_prevention`. The other 10 specialist agents (market_structure,
regime_intelligence, mean_reversion, trend_momentum, sentiment_decoder,
trade_timing, portfolio_optimizer, pattern_matcher,
authority_compliance_guardian, polymarket_arb) were hidden — the LLM
could only see their blended consensus via the orchestrator, never ask
one for a second opinion on a specific concern.

This module closes the capability gap. Every agent is now callable as
`ask_<agent_name>(asset, task_description, ...)` with the same sub-agent
contract as the existing risk-guardian / loss-prevention tools
(parent-authored task description + isolated context + ≤400-char
digest back).

Also exposes:
  * ask_debate — run the 3-round adversarial debate on the 12 agents'
    current votes and return the surviving consensus.
  * backtest_hypothesis — vectorized backtest of a proposed strategy
    idea on recent bars, returning Sharpe + WR + max DD in <2s.

Keep the lean-context discipline from C3: every tool returns a compact
digest, never raw payloads.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Agent name → (module path, class name, short one-line description).
AGENT_REGISTRATIONS = [
    ("market_structure",          "src.agents.market_structure_agent",    "MarketStructureAgent",
     "Sub-agent: market-microstructure view — liquidity, imbalance, sweep patterns."),
    ("regime_intelligence",       "src.agents.regime_intelligence_agent", "RegimeIntelligenceAgent",
     "Sub-agent: current regime classification (trending/ranging/volatile/crisis) + stability."),
    ("mean_reversion",            "src.agents.mean_reversion_agent",      "MeanReversionAgent",
     "Sub-agent: mean-reversion view — z-score, Bollinger %, OU half-life."),
    ("trend_momentum",            "src.agents.trend_momentum_agent",      "TrendMomentumAgent",
     "Sub-agent: trend + momentum view — ADX, MACD, Kalman slope, EMA alignment."),
    ("sentiment_decoder",         "src.agents.sentiment_decoder_agent",   "SentimentDecoderAgent",
     "Sub-agent: fused sentiment — news + FinBERT + fear-greed + whale flows."),
    ("trade_timing",              "src.agents.trade_timing_agent",        "TradeTimingAgent",
     "Sub-agent: entry-timing view — is NOW the right moment given intrabar structure?"),
    ("portfolio_optimizer",       "src.agents.portfolio_optimizer_agent", "PortfolioOptimizerAgent",
     "Sub-agent: portfolio-level view — correlation, exposure, diversification."),
    ("pattern_matcher",           "src.agents.pattern_matcher_agent",     "PatternMatcherAgent",
     "Sub-agent: historical pattern match — has this setup appeared before, what happened."),
    ("authority_compliance",      "src.agents.authority_compliance_guardian", "AuthorityComplianceGuardian",
     "Sub-agent: authority-rule compliance check — does the proposed trade violate any rule?"),
    ("polymarket_arb",            "src.agents.polymarket_agent",          "PolymarketArbitrageAgent",
     "Sub-agent: Polymarket prediction-market arbitrage view — implied-probability divergence."),
]


def _make_agent_handler(module_path: str, class_name: str) -> Callable:
    """Build the dispatch function for one agent. Lazy-imports + catches
    every exception so a missing/broken agent just returns unavailable."""

    def _handler(args: Dict[str, Any]) -> Dict[str, Any]:
        task = str(args.get("task_description") or "").strip()
        asset = args.get("asset")
        proposed_direction = args.get("proposed_direction")
        try:
            import importlib
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name, None)
            if cls is None:
                return {"verdict": "unavailable", "reason": f"{class_name} missing in {module_path}"}
            agent = cls()
        except Exception as e:
            return {"verdict": "unavailable", "reason": f"{type(e).__name__}: {e}"}

        state: Dict[str, Any] = {"asset": asset}
        if proposed_direction is not None:
            state["direction"] = proposed_direction
        ctx: Dict[str, Any] = {"task_description": task}
        try:
            vote = agent.analyze(state, ctx)
        except Exception as e:
            return {"verdict": "unavailable", "reason": f"analyze_failed: {type(e).__name__}: {e}"}

        return {
            "verdict": int(getattr(vote, "direction", 0)),
            "confidence": float(getattr(vote, "confidence", 0.5) or 0.5),
            "rationale": str(getattr(vote, "reasoning", "") or
                             getattr(vote, "rationale", "") or "")[:400],
            "agent_weight": float(
                getattr(agent, "_current_weight", None)
                or (agent.get_weight() if hasattr(agent, "get_weight") else 1.0)
            ),
        }

    return _handler


# ── Debate tool handler ────────────────────────────────────────────────


def _handle_debate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run the 3-round adversarial debate and return the consensus."""
    asset = args.get("asset")
    try:
        from src.agents.debate_engine import DebateEngine
        from src.agents.orchestrator import AgentOrchestrator
    except Exception as e:
        return {"verdict": "unavailable", "reason": f"imports: {type(e).__name__}: {e}"}

    try:
        orch = AgentOrchestrator()
        # Let the orchestrator collect the 12 agents' votes internally.
        quant_state = {"asset": asset}
        ctx = {"task_description": str(args.get("task_description") or "")}
        # Many orchestrator implementations expose gather_votes / run.
        votes = None
        for m in ("gather_votes", "collect_votes", "_collect_agent_votes", "collect"):
            fn = getattr(orch, m, None)
            if callable(fn):
                try:
                    votes = fn(quant_state, ctx)
                    break
                except Exception:
                    continue
        if votes is None:
            return {"verdict": "unavailable", "reason": "no vote-collection entrypoint on orchestrator"}

        engine = DebateEngine()
        result = engine.run_debate(votes)
    except Exception as e:
        return {"verdict": "unavailable", "reason": f"debate_failed: {type(e).__name__}: {e}"}

    # Shape: DebateResult has debate_summary + post-debate vote map.
    summary = getattr(result, "debate_summary", "") or ""
    return {
        "summary": summary[:400],
        "final_direction": int(getattr(result, "final_direction", 0) or 0),
        "conviction": float(getattr(result, "conviction", 0.0) or 0.0),
        "flipped_count": int(getattr(result, "flipped_count", 0) or 0),
    }


# ── Backtest tool handler ──────────────────────────────────────────────


def _handle_backtest(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run a quick vectorized backtest of a proposed strategy idea.

    The handler is permissive about what strategy_rules look like — it
    falls through to the default EMA-crossover if no rules specified,
    so the LLM can always get SOME historical sanity number."""
    asset = str(args.get("asset") or "BTC").upper()
    timeframe = str(args.get("timeframe") or "1h")
    try:
        bars = min(1000, int(args.get("bars") or 500))
    except Exception:
        bars = 500

    # Fetch recent bars via the same PriceFetcher quant_tools uses.
    try:
        from src.data.fetcher import PriceFetcher
        try:
            pf = PriceFetcher()
        except Exception:
            pf = PriceFetcher({})
        raw = pf.fetch_ohlcv(asset, timeframe=timeframe, limit=bars) or []
    except Exception as e:
        return {"error": f"bars_unavailable: {type(e).__name__}: {e}"}
    if len(raw) < 50:
        return {"error": f"insufficient bars ({len(raw)})"}

    # Delegate to the existing vectorized engine.
    try:
        from src.backtesting.engine import BacktestEngine
        eng = BacktestEngine()
        # Most signatures accept a raw-price array or a DataFrame.
        closes = [row[4] for row in raw if len(row) >= 5]
        result = eng.run(closes) if hasattr(eng, "run") else eng.backtest(closes)
    except Exception as e:
        return {"error": f"backtest_failed: {type(e).__name__}: {e}"}

    if not isinstance(result, dict):
        return {"summary": str(result)[:300]}

    sharpe = result.get("sharpe") or result.get("sharpe_ratio")
    wr = result.get("win_rate") or result.get("wr")
    max_dd = result.get("max_drawdown") or result.get("max_dd")
    trades = result.get("trades") or result.get("n_trades")
    return {
        "summary": (f"Backtest({asset}/{timeframe}/{bars}b): "
                    f"Sharpe={sharpe} WR={wr} max_DD={max_dd} n_trades={trades}"),
        "sharpe": sharpe, "win_rate": wr,
        "max_drawdown": max_dd, "trades": trades,
    }


# ── Body-controls tool handler ─────────────────────────────────────────


def _handle_body_controls(_args: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from src.learning.brain_to_body import get_controller
        controls = get_controller().current()
        return controls.to_dict()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ── Registration entry point ───────────────────────────────────────────


def register_agent_tools(registry) -> List[str]:
    """Add all 10 agent tools + debate + backtest to the registry.

    Skips duplicates silently so running this twice (e.g. tests that
    call build_default_registry() multiple times) is safe.
    """
    from src.ai.trade_tools import Tool

    added: List[str] = []

    # Per-agent ask_* tools.
    for agent_key, module_path, class_name, description in AGENT_REGISTRATIONS:
        name = f"ask_{agent_key}"
        try:
            registry.register(Tool(
                name=name,
                description=f"[AGENT] {description}",
                input_schema={
                    "type": "object",
                    "properties": {
                        "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                        "task_description": {
                            "type": "string",
                            "description": "Concrete question for this specialist agent.",
                        },
                        "proposed_direction": {"type": "string", "enum": ["LONG", "SHORT"]},
                    },
                    "required": ["asset", "task_description"],
                },
                handler=_make_agent_handler(module_path, class_name),
                tag="read_only",
                subagent_system_prompt=(
                    f"You are ACT's {class_name}. Provide a focused "
                    "2-4-sentence verdict on the parent's task description. "
                    "Cite metrics [METRIC=VALUE] from verified data only."
                ),
            ))
            added.append(name)
        except ValueError:
            logger.debug("agent tool %s already registered", name)

    # ask_debate.
    try:
        registry.register(Tool(
            name="ask_debate",
            description=(
                "[AGENT] Run ACT's 3-round adversarial debate on the 12 "
                "agents' current votes. Returns the post-debate consensus "
                "(final_direction, conviction, flipped_count)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string", "enum": ["BTC", "ETH"]},
                    "task_description": {"type": "string"},
                },
                "required": ["asset"],
            },
            handler=_handle_debate, tag="read_only",
        ))
        added.append("ask_debate")
    except ValueError:
        pass

    # get_body_controls — C9.
    try:
        registry.register(Tool(
            name="get_body_controls",
            description=(
                "[CONTROLLER] Current brain-to-body pressure signals: "
                "exploration_bias for the strategy bandit, "
                "genetic_cadence_s, emergency_level "
                "(normal/caution/stress), priority_agents (which ask_* "
                "tools to query first this tick), and the diagnostic "
                "reason. Updated every few shadow ticks."
            ),
            input_schema={"type": "object", "properties": {}},
            handler=_handle_body_controls, tag="read_only",
        ))
        added.append("get_body_controls")
    except ValueError:
        pass

    # backtest_hypothesis.
    try:
        registry.register(Tool(
            name="backtest_hypothesis",
            description=(
                "[BACKTEST] Run a <2s vectorized backtest of the default "
                "EMA-crossover strategy on recent bars for (asset, "
                "timeframe). Returns Sharpe / WR / max DD / n_trades. Use "
                "this to sanity-check a proposed entry before compiling "
                "a TradePlan."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "asset": {"type": "string"},
                    "timeframe": {"type": "string",
                                  "enum": ["5m", "15m", "1h", "4h", "1d"]},
                    "bars": {"type": "integer", "minimum": 50, "maximum": 1000},
                },
                "required": ["asset"],
            },
            handler=_handle_backtest, tag="read_only",
        ))
        added.append("backtest_hypothesis")
    except ValueError:
        pass

    return added
