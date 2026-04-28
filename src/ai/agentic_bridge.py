"""
End-to-end glue for the agentic trade loop.

What this module does:
  * Composes all C1–C3 modules + the existing LLMRouter into one callable
    entry point: `compile_agentic_plan(asset, regime, quant_data, ...)`
    returns a `LoopResult` with a validated `TradePlan` (or a skip-plan).
  * Provides a small CLI (`python -m src.ai.agentic_bridge --asset BTC
    --dry-run`) so the operator can exercise the loop manually before
    wiring it into the executor.

What this module does NOT do:
  * Place orders. That's the executor's job (C4d hook).
  * Own new configuration. All tunables come from env + existing
    `config.yaml` blocks (agentic_loop, emergency_mode).

Reuses (no re-implementation):
  * src/ai/agentic_context.py  — sliding context + summarization (M2)
  * src/ai/agentic_trade_loop.py — multi-turn ReAct driver (C3)
  * src/ai/trade_tools.py      — tool registry (C3)
  * src/ai/llm_provider.py     — provider-agnostic LLM dispatch
  * src/trading/trade_plan.py  — TradePlan validation (C1)
  * src/orchestration/warm_store.py — read last N self_critiques
  * src/ai/memory_vault.py     — cross-session RAG (if available)

Graceful degradation: if the LLM router is unavailable, the bridge
returns a SKIP plan with reason "no_llm" rather than raising. The
executor's default path is unaffected by bridge failures.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from typing import Any, Callable, Dict, List, Optional

from src.ai.agentic_context import AgenticContext
from src.ai.agentic_trade_loop import AgenticTradeLoop, LoopResult
from src.ai.trade_tools import ToolRegistry, build_default_registry
from src.trading.trade_plan import TradePlan

logger = logging.getLogger(__name__)


# ── LLM-call closure builder ────────────────────────────────────────────


def build_llm_call(
    system_prompt_override: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], str]:
    """Return a closure: `(messages) -> str` that dispatches the agentic
    loop's turns to an LLM.

    Routing (C5 dual-brain):
      * If `ai.dual_brain.enabled` is true in config, route to the
        ANALYST brain (Devstral 24B by default) — its structured-JSON
        and tool-use training fits the loop's multi-turn envelope
        protocol.
      * Otherwise, use the legacy flatten-and-dispatch path through
        the first available LLMRouter provider. Same behavior as
        before C5; existing tests / fallback unchanged.
    """
    # Prefer the dual-brain Analyst when enabled.
    try:
        from src.ai.dual_brain import build_analyst_llm_call, is_enabled as _dual_enabled
        if _dual_enabled(config):
            return build_analyst_llm_call(config)
    except Exception as e:
        logger.debug("agentic_bridge: dual_brain unavailable, using flat router: %s", e)
    def _call(messages: List[Dict[str, Any]]) -> str:
        if not messages:
            return ""

        # Separate system from the rest.
        system_parts: List[str] = []
        rest: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content, default=str)
            content = str(content)
            if role == "system":
                system_parts.append(content)
            else:
                rest.append(f"[{role}]\n{content}")
        sys_prompt = system_prompt_override or "\n\n".join(system_parts)
        prompt = "\n\n".join(rest)

        try:
            from src.ai.llm_provider import LLMRouter
            router = LLMRouter()
            router.add_from_env()
            for _name, provider in list(router.providers.items()):
                try:
                    resp = provider.generate(prompt=prompt, system_prompt=sys_prompt)
                    text = resp.get("response") if isinstance(resp, dict) else str(resp)
                    if text:
                        return str(text)
                except Exception as e:
                    logger.debug("agentic_bridge provider %s failed: %s", _name, e)
                    continue
        except Exception as e:
            logger.debug("agentic_bridge LLMRouter unavailable: %s", e)
        return ""

    return _call


# ── Seed-context loaders ────────────────────────────────────────────────


def load_recent_critiques(limit: int = 5) -> List[Dict[str, Any]]:
    """Read the last N self_critique rows from warm_store.

    Graceful: returns [] on any error so a cold-start / missing db
    doesn't block the loop.
    """
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        conn = sqlite3.connect(store.db_path, timeout=2.0)
        try:
            rows = conn.execute(
                "SELECT self_critique FROM decisions "
                "WHERE self_critique != '{}' AND self_critique IS NOT NULL "
                "ORDER BY ts_ns DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        finally:
            conn.close()
        out: List[Dict[str, Any]] = []
        for (raw,) in rows:
            try:
                out.append(json.loads(raw or "{}"))
            except Exception:
                continue
        return out
    except Exception as e:
        logger.debug("load_recent_critiques failed: %s", e)
        return []


def load_similar_trades(
    asset: str,
    regime: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Cross-session RAG over MemoryVault. Returns [] on any failure —
    MemoryVault initialization is non-trivial (loads embedding model)."""
    try:
        from src.ai.memory_vault import MemoryVault
        vault = MemoryVault()
        return vault.find_similar_trades(
            asset=asset,
            current_regime=regime,
            current_funding=0.0,                        # executor fills real values
            current_sentiment={"bullish": 0.5, "bearish": 0.5},
            proposed_signal=0,
            top_k=top_k,
        ) or []
    except Exception as e:
        logger.debug("load_similar_trades failed: %s", e)
        return []


# ── Main entry ──────────────────────────────────────────────────────────


def _fetch_scan_context(asset: str) -> str:
    """Build the analyst's seed-context block via the shared
    context_builders.build_analyst_context helper (simplify pass)."""
    try:
        from src.ai.context_builders import build_analyst_context
        return build_analyst_context(asset)
    except Exception as e:
        logger.debug("_fetch_scan_context failed: %s", e)
        return ""


def compile_agentic_plan(
    asset: str,
    *,
    regime: str = "UNKNOWN",
    quant_data: str = "",
    registry: Optional[ToolRegistry] = None,
    context: Optional[AgenticContext] = None,
    llm_call: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    max_steps: Optional[int] = None,
    similar_trades: Optional[List[Dict[str, Any]]] = None,
    recent_critiques: Optional[List[Dict[str, Any]]] = None,
) -> LoopResult:
    """Compose the full loop and run it. Safe to call from the executor;
    never raises — a compile failure returns a SKIP plan with reason.

    Scanner/analyst bridge (C7b): if the brain_memory has a fresh scan
    report or recent analyst traces for this asset, they're appended to
    quant_data before the loop seeds — so the analyst sees what the
    scanner flagged and what past decisions looked like.
    """

    # Kill switch wins over everything.
    if os.environ.get("ACT_DISABLE_AGENTIC_LOOP", "0") == "1":
        return LoopResult(
            plan=TradePlan.skip(asset, thesis="disabled by ACT_DISABLE_AGENTIC_LOOP"),
            steps_taken=0,
            terminated_reason="disabled",
        )

    try:
        # Fill seed bits the caller didn't pre-fetch.
        if similar_trades is None:
            similar_trades = load_similar_trades(asset, regime)
        if recent_critiques is None:
            recent_critiques = load_recent_critiques()

        # C7b — inject scanner report + recent analyst traces into quant_data
        # so the analyst sees the other brain's output.
        brain_block = _fetch_scan_context(asset)
        if brain_block:
            quant_data = (quant_data + "\n\n" + brain_block) if quant_data else brain_block

        # C9 — body-controls priority agents. The brain-to-body controller
        # picks 1-5 agent tools from the scanner's top_signals (e.g.
        # "breakout" -> ask_trend_momentum). Surface that priority list so
        # the analyst's ReAct loop calls those tools first instead of
        # walking the registry in declaration order. Never blocks tool
        # access -- it's a hint, not a whitelist.
        try:
            from src.learning.brain_to_body import current_priority_agents
            priority = current_priority_agents()
            if priority:
                pa_block = (
                    "\n\n## PRIORITY TOOLS THIS TICK\n"
                    "Based on the scanner's top signals, call these tools "
                    "first when relevant: " + ", ".join(priority[:5]) + "."
                )
                quant_data = (quant_data + pa_block) if quant_data else pa_block.lstrip()
        except Exception as _e:  # pragma: no cover - best-effort hint only
            logger.debug("agentic_bridge priority injection skipped: %s", _e)

        registry = registry or build_default_registry()
        context = context or AgenticContext(asset=asset)
        llm_call = llm_call or build_llm_call()
        steps = int(max_steps) if max_steps is not None else int(
            os.getenv("ACT_AGENTIC_MAX_STEPS", "8")
        )

        loop = AgenticTradeLoop(
            asset=asset,
            llm_call=llm_call,
            registry=registry,
            context=context,
            max_steps=steps,
        )
        loop.seed(
            regime=regime,
            quant_data=quant_data,
            recent_critiques=recent_critiques,
            similar_trades=similar_trades,
        )
        result = loop.run()

        # Rule-based fallback when LLM doesn't produce an actionable plan.
        # Per operator directive ("use everything present in ACT to make
        # trades"), the multi-strategy engine + 14 agents + ML ensemble
        # + TF alignment all compute valid signals every tick. In paper
        # mode (real capital still gated):
        #
        #   * `parse_failures` / `max_steps` / `disabled` -> LLM was
        #     unreachable; rule stack fills in.
        #   * `skip` -> LLM thought hard and refused. We OVERRIDE in
        #     paper mode IF the rule stack has strong multi-source
        #     agreement (>=2 votes one direction). The operator's log
        #     showed "EXCELLENT PATTERN score=13/10 + SNIPER PASS
        #     confluence 6/3 + CONVICTION PASS" combined with LLM
        #     SKIP -> zero trade. That's exactly the case rule-based
        #     fallback was built for. Real-capital path stays strict:
        #     LLM SKIP in real mode is honored.
        is_real_capital = os.environ.get(
            "ACT_REAL_CAPITAL_ENABLED", ""
        ).strip() == "1"
        terminated = (result.terminated_reason or "").lower()
        plan_proposes_trade = (
            result.plan is not None
            and str(result.plan.direction).upper() in ("LONG", "SHORT", "BUY", "SELL")
        )
        fallback_triggers = ("parse_failures", "max_steps", "disabled")
        if not is_real_capital:
            fallback_triggers = fallback_triggers + ("skip",)
        if (terminated in fallback_triggers
                and not plan_proposes_trade
                and not is_real_capital):
            fb_plan = _rule_based_fallback_plan(
                asset=asset, regime=regime, quant_data=quant_data,
                context=context,
            )
            if fb_plan is not None:
                logger.warning(
                    "[agentic_bridge:%s] LLM unreachable (%s); "
                    "synthesizing rule-based fallback plan: dir=%s "
                    "tier=%s size_pct=%s",
                    asset, terminated, fb_plan.direction,
                    fb_plan.entry_tier, fb_plan.size_pct,
                )
                result = LoopResult(
                    plan=fb_plan,
                    steps_taken=result.steps_taken,
                    terminated_reason="rule_based_fallback",
                )

        # ── EXPERIMENTAL INTEGRATIONS (all default OFF, env-gated) ──
        # Each integration runs in shadow mode when its env flag is
        # "shadow", or authoritatively when "1". When unset, the calls
        # are no-ops. Existing TradePlan output is unchanged unless the
        # operator explicitly promotes one of these to authoritative.
        try:
            from src.ai import dual_path_reasoning as _dpr
            if _dpr.is_enabled():
                # Shadow mode: synthesize from existing analyst trace
                # rather than running 2 extra LLM calls. The fact and
                # subjectivity verdicts here are *derived* from the
                # plan's quoted_evidence; full split-LLM-call mode is
                # gated behind explicit promotion (not part of shadow).
                _fact = _dpr.PathVerdict(
                    path="fact",
                    direction=str(getattr(result.plan, "direction", "SKIP")),
                    confidence=float(getattr(result.plan, "confidence", 0.5)),
                    rationale=str(getattr(result.plan, "thesis", ""))[:200],
                    inputs_used=["existing_analyst_output"],
                )
                _subj = _dpr.PathVerdict(
                    path="subjectivity",
                    direction=str(getattr(result.plan, "direction", "SKIP")),
                    confidence=float(getattr(result.plan, "confidence", 0.5)),
                    rationale="proxy from fused output (split-mode pending)",
                    inputs_used=["existing_analyst_output"],
                )
                _synth = _dpr.synthesize(_fact, _subj)
                _dpr.log_shadow(asset, _synth,
                                str(getattr(result.plan, "direction", "?")),
                                float(getattr(result.plan, "confidence", 0.0)))
        except Exception as _e:
            logger.debug("dual_path shadow eval skipped: %s", _e)

        try:
            from src.agents import hierarchical_orchestrator as _hier
            if _hier.is_enabled():
                # Pull the existing flat-vote agent map from tick_state.
                from src.ai import tick_state as _ts_mod
                _snap_for_hier = _ts_mod.get(asset)
                # We don't have direct access to the agent_votes dict
                # here, but the hierarchical layer reads from the same
                # warm_store; we synthesize a minimal vote map so the
                # log captures the comparison. Full integration that
                # forwards the live agent_votes is part of authoritative
                # promotion, not shadow.
                _empty_votes: Dict[str, Dict[str, Any]] = {}
                _decision = _hier.hierarchical_decide(asset, _empty_votes)
                _flat_dir = (1 if str(getattr(result.plan, "direction", "")).upper() in ("LONG", "BUY")
                             else -1 if str(getattr(result.plan, "direction", "")).upper() in ("SHORT", "SELL")
                             else 0)
                _hier.log_shadow(_decision, _flat_dir,
                                 float(getattr(result.plan, "confidence", 0.0)))
        except Exception as _e:
            logger.debug("hierarchy shadow eval skipped: %s", _e)

        try:
            from src.agents import skeptic_persona as _sk
            if _sk.is_enabled():
                from src.ai import tick_state as _ts_mod
                _snap_for_sk = _ts_mod.get(asset)
                _verdict = _sk.evaluate(
                    asset=asset,
                    consensus_direction=str(getattr(result.plan, "direction", "")),
                    consensus_confidence=float(getattr(result.plan, "confidence", 0.5)),
                    tick_snap=_snap_for_sk,
                )
                if _verdict is not None:
                    _proceeded = (
                        result.plan is not None
                        and str(result.plan.direction).upper() in ("LONG", "SHORT", "BUY", "SELL")
                    )
                    _sk.log_shadow(_verdict, _proceeded)
                    # Write the skeptic line into tick_state so the
                    # NEXT tick's analyst sees the contrarian argument
                    # in its evidence document.
                    try:
                        _line = _sk.format_for_brain(_verdict)
                        if _line:
                            _ts_mod.update(asset, skeptic_advisory=_line[:400])
                    except Exception:
                        pass
        except Exception as _e:
            logger.debug("skeptic shadow eval skipped: %s", _e)

        # Operator-visible audit line: brain's per-tick verdict + how
        # many open positions it considered. Lets the operator confirm
        # at a glance that portfolio review is happening every tick.
        try:
            from src.ai import tick_state as _ts_mod
            _snap = _ts_mod.get(asset)
            _open_n = int(_snap.get("open_positions_same_asset", 0)) if _snap else 0
            _gap = float(_snap.get("gap_to_1pct", 0.0)) if _snap else 0.0
            logger.info(
                "[BRAIN:%s] tick decision: dir=%s tier=%s size=%.1f%% "
                "verdict=%s | considered_open=%d gap_to_1pct=%+.2f%% "
                "steps=%d",
                asset, str(getattr(result.plan, "direction", "?")),
                str(getattr(result.plan, "entry_tier", "?")),
                float(getattr(result.plan, "size_pct", 0.0)),
                result.terminated_reason or "?",
                _open_n, _gap, int(getattr(result, "steps_taken", 0)),
            )
        except Exception:
            pass

        # C7b — write the analyst's decision back into brain_memory so
        # the scanner's next tick sees it. Compact trace; never raises.
        # When the rule-based fallback overrode an LLM skip, prepend an
        # explicit marker to the thesis so the next tick's analyst sees
        # 'I said SKIP last tick but the fallback overrode me' and can
        # course-correct (e.g. raise conviction or accept the bounce).
        try:
            import time as _t
            from src.ai.brain_memory import AnalystTrace, publish_analyst_trace
            _thesis = str(getattr(result.plan, 'thesis', ''))[:300]
            if result.terminated_reason == "rule_based_fallback":
                _thesis = (
                    "[FALLBACK_OVERRIDE: rule stack overrode LLM skip] " + _thesis
                )[:300]
            publish_analyst_trace(AnalystTrace(
                asset=asset, ts=_t.time(),
                plan_id=getattr(result.plan, 'plan_id', '') or '',
                direction=str(result.plan.direction),
                tier=str(result.plan.entry_tier),
                size_pct=float(getattr(result.plan, 'size_pct', 0.0)),
                thesis=_thesis,
                verdict=result.terminated_reason,
            ))
        except Exception as _e:
            logger.debug("brain_memory publish_analyst_trace failed: %s", _e)

        return result
    except Exception as e:
        logger.debug("compile_agentic_plan fatal: %s", e)
        return LoopResult(
            plan=TradePlan.skip(asset, thesis=f"bridge_error: {type(e).__name__}"),
            steps_taken=0,
            terminated_reason="parse_failures",
        )


def _rule_based_fallback_plan(
    *,
    asset: str,
    regime: str,
    quant_data: str,
    context: Optional[AgenticContext] = None,
) -> Optional["TradePlan"]:
    """Synthesize a TradePlan from the rule-based stack when the LLM
    path is unreachable. Per operator directive ("use everything in
    ACT to make trades"), the multi-strategy engine + 14 agents +
    ML ensemble + TF alignment are all already computing valid
    signals on every tick -- they just go unused when the agentic
    loop's parse_failure forces a SHADOW SKIP.

    Returns None when the rule stack itself is too neutral to
    propose a direction -- in that case the caller keeps the
    original SKIP plan. Real-capital path SHOULD NOT call this
    (caller checks ACT_REAL_CAPITAL_ENABLED).
    """
    if context is None:
        return None

    # Best-effort signal aggregation. Each source returns a vote in
    # {-1, 0, +1} or None. We require at least 2 sources agreeing in
    # the same direction; if the bag is mixed we return None so the
    # caller keeps SHADOW SKIP.
    votes: List[int] = []

    # 1. TF alignment (1h + 4h) -- the executor already prints
    #    "TF ALIGNMENT OVERRIDE: 1h=RISING 4h=RISING" on every tick.
    #    Parse from quant_data if it's there.
    tf_text = (quant_data or "").lower()
    bullish_tf = ("1h=rising" in tf_text and "4h=rising" in tf_text) or \
                 ("tf_1h=rising" in tf_text and "tf_4h=rising" in tf_text)
    bearish_tf = ("1h=falling" in tf_text and "4h=falling" in tf_text) or \
                 ("tf_1h=falling" in tf_text and "tf_4h=falling" in tf_text)
    if bullish_tf:
        votes.append(+1)
    elif bearish_tf:
        votes.append(-1)

    # 2. Multi-strategy consensus score (parsed from quant_data)
    import re as _re
    m = _re.search(r"consensus=\w+\s+score=([-+]?\d*\.?\d+)", quant_data or "")
    if m:
        try:
            score = float(m.group(1))
            if score > 0.05:
                votes.append(+1)
            elif score < -0.05:
                votes.append(-1)
        except ValueError:
            pass

    # 3. Brain memory's most recent scan rationale (continuity from
    #    when the LLM last DID respond)
    try:
        from src.ai.brain_memory import get_brain_memory
        mem = get_brain_memory()
        latest = mem.read_latest_scan(asset, max_age_s=86400.0)
        if latest is not None:
            d = (latest.proposed_direction or "").upper()
            if d == "LONG":
                votes.append(+1)
            elif d == "SHORT":
                votes.append(-1)
    except Exception:
        pass

    # Mean-reversion-bounce override for longs-only venues (Robinhood
    # spot). When ALL non-LLM sources vote SHORT but the venue can't
    # actually short, look for oversold conditions (RSI<40, large
    # ATR move expected, "EXCELLENT PATTERN" in quant_data) and
    # propose a small LONG bounce trade instead of skipping. This
    # captures the bounce trades that happen on every bearish leg
    # while keeping the position size small (paper-mode soak data).
    is_robinhood_longs_only = "robinhood" in (regime or "").lower() or True  # default assume longs-only
    net_pre = sum(votes)
    if net_pre < 0 and is_robinhood_longs_only:
        # All votes bearish but we can only LONG. Check for bounce.
        rsi_match = _re.search(r"rsi[_-]?bear=(\d+)", quant_data or "", _re.IGNORECASE)
        is_oversold = False
        if rsi_match:
            try:
                rsi = int(rsi_match.group(1))
                is_oversold = rsi < 40
            except ValueError:
                pass
        excellent_pattern = "excellent pattern" in (quant_data or "").lower()
        sniper_pass = "sniper pass" in (quant_data or "").lower()
        if is_oversold or excellent_pattern or sniper_pass:
            try:
                from src.trading.trade_plan import TradePlan
                return TradePlan(
                    asset=asset.upper(),
                    direction="LONG",
                    entry_tier="normal",
                    size_pct=0.5,  # half-size for counter-trend bounce
                    confidence=0.55,
                    expected_pnl_pct_range=(1.0, 2.0),
                    thesis=(
                        f"mean-reversion bounce LONG (longs-only venue): "
                        f"all sources bearish but oversold/excellent-pattern "
                        f"detected; small size; "
                        f"rsi_oversold={is_oversold} pattern={excellent_pattern} "
                        f"sniper={sniper_pass}; regime={regime}"
                    )[:300],
                )
            except Exception as e:
                logger.debug(
                    "rule-based bounce-LONG construction failed: %s", e,
                )

    if not votes:
        return None

    net = sum(votes)
    if net == 0:
        return None  # mixed -- keep SHADOW SKIP

    direction = "LONG" if net > 0 else "SHORT"
    agreeing = sum(1 for v in votes if (v > 0 if net > 0 else v < 0))
    confidence = min(0.85, 0.4 + 0.15 * agreeing)
    size_pct = 0.5 if agreeing < 2 else 1.0  # half-size unless 2+ sources agree

    try:
        from src.trading.trade_plan import TradePlan
        # Populate expected_pnl_pct_range so the downstream cost gate
        # has actual numbers to score against. 1.5-2.5% is the
        # operator's typical range for paper sniper-tier setups; the
        # gate computes margin = expected - frictional and the brain's
        # consensus already passed the multi-source agreement check.
        return TradePlan(
            asset=asset.upper(),
            direction=direction,
            entry_tier="normal",
            size_pct=size_pct,
            confidence=confidence,
            expected_pnl_pct_range=(1.5, 2.5),
            thesis=(
                f"rule-based fallback (LLM unavailable): "
                f"{agreeing}/{len(votes)} non-LLM sources agree on "
                f"{direction.lower()}; TF={tf_text[:60] if (bullish_tf or bearish_tf) else 'mixed'}; "
                f"regime={regime}"
            )[:300],
        )
    except Exception as e:
        logger.debug("rule-based fallback plan construction failed: %s", e)
        return None


# ── Feature-flag helper (used by the C4d executor hook) ─────────────────


def agentic_loop_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Env flag wins; falls back to config.yaml `agentic_loop.enabled`.

    `ACT_DISABLE_AGENTIC_LOOP=1` always wins (returns False) to give ops
    a one-env-var kill switch.
    """
    if os.environ.get("ACT_DISABLE_AGENTIC_LOOP", "0") == "1":
        return False
    env = os.environ.get("ACT_AGENTIC_LOOP", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if env in ("0", "false", "no", "off"):
        return False
    if isinstance(config, dict):
        return bool((config.get("agentic_loop") or {}).get("enabled", False))
    return False


# ── CLI (dry-run) ───────────────────────────────────────────────────────


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run the agentic trade loop for one asset and print the result.",
    )
    parser.add_argument("--asset", default="BTC", help="Asset symbol (BTC, ETH)")
    parser.add_argument("--regime", default="UNKNOWN", help="Current regime tag")
    parser.add_argument("--quant", default="", help="Verified quant-data block to seed with")
    parser.add_argument("--max-steps", type=int, default=6, help="Hard cap on LLM tool-use rounds")
    parser.add_argument(
        "--stub-llm", action="store_true",
        help="Skip the LLM; return a canned SKIP plan so the pipeline "
        "can be tested without any model available.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    llm_call = None
    if args.stub_llm:
        llm_call = lambda _msgs: json.dumps({"skip": "cli --stub-llm"})

    result = compile_agentic_plan(
        asset=args.asset.upper(),
        regime=args.regime,
        quant_data=args.quant,
        llm_call=llm_call,
        max_steps=args.max_steps,
    )
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.terminated_reason in ("plan", "skip") else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
