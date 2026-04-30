"""
Dual-brain LLM router — scanner (right-brain) + analyst (left-brain).

Default profile (2026-04 "qwen3_r1", aligned with 2026-April Ollama
best-practice rankings for trading workloads):

  * **Scanner = Qwen 3 32B** (right brain). Best speed-to-intelligence
    ratio for macro/news/sentiment summarization at tick cadence. Runs
    every scheduler tick.

  * **Analyst = DeepSeek-R1 32B distill** (left brain). Chain-of-thought
    reasoning — "king of reasoning" for evaluating complex spot trading
    signals and compiling TradePlans under Pydantic validation. Runs
    on-demand when the scanner flags a setup.

Both run locally via Ollama on the RTX 5090 at Q4_K_M quantization
(~20GB qwen3 + ~19GB deepseek-r1 = ~39GB raw; Ollama swaps the analyst
out when idle so the scanner stays resident at tick cadence and the
full pair fits within VRAM headroom).

Design constraints:
  * Zero new hard deps — composes over the existing
    src/ai/llm_provider.py::LLMRouter.
  * Cross-fallback: if the primary brain errors, try the other with a
    small adapter prompt. Trading never blocks on one model being
    unavailable.
  * Role-tagged system prompts — scanner and analyst get task-shaped
    guidance so they stay in their lane. No cross-pollination.
  * Env overrides win over config: ACT_SCANNER_MODEL,
    ACT_ANALYST_MODEL, ACT_DISABLE_DUAL_BRAIN.

Not in scope here (handled elsewhere):
  * The actual multi-turn loop — src/ai/agentic_trade_loop.py still owns
    iteration + tool dispatch.
  * Prompt construction — src/ai/prompt_constraints.py + authority_rules
    still own the "absolute rules" system prompt that both brains see.
  * Post-close verification — trade_verifier.py does its own LLM call
    independent of this router.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# DeepSeek-R1 and other reasoning-trained models emit <think>...</think>
# blocks before their final answer. For the scanner's tick-cadence output
# we want the answer only — the CoT trace can be 1000+ tokens and would
# bloat brain_memory + the analyst's seed context.
_THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
# Also handle unclosed <think> (model ran out of budget mid-trace): in
# that case nothing after <think> is useful — drop everything from the
# opener to the end, keeping whatever came before.
_UNCLOSED_THINK_RE = re.compile(r"<think>.*$", re.IGNORECASE | re.DOTALL)


def strip_reasoning_tags(text: str) -> str:
    """Remove <think>...</think> reasoning traces from model output.

    Safe no-op on models that don't emit the tag (plain JSON / prose).
    Preserves the final answer so the tool-use loop's JSON parser sees
    the complete envelope. Trimmed to a single trailing newline.
    """
    if not text or "<think>" not in text.lower():
        return text
    stripped = _THINK_RE.sub("", text)
    stripped = _UNCLOSED_THINK_RE.sub("", stripped)
    return stripped.strip()


# Brain names — keep these stable; other modules check against them.
SCANNER = "scanner"
ANALYST = "analyst"
VALID_BRAINS = (SCANNER, ANALYST)

DISABLE_ENV = "ACT_DISABLE_DUAL_BRAIN"


# ── Profiles (C5d) ──────────────────────────────────────────────────────
#
# Three named brain profiles. Operator can A/B-test them via
# ACT_BRAIN_PROFILE=<name> at runtime, or pin via config.yaml
# `ai.dual_brain.profile: <name>`. All three fit on RTX 5090 32 GB.
#
#   dense_r1      — deepseek-r1:32b + deepseek-r1:7b (default, most
#                   consistent, reasoning-depth both sides). Safest
#                   for paper soak; fewer variance sources to debug.
#   moe_agentic   — qwen3-coder:30b + qwen2.5-coder:7b (MoE analyst
#                   + dense worker). Best tool-use speed, 2026
#                   agentic gold standard; less raw reasoning depth.
#   hybrid        — qwen3-coder:30b (MoE analyst, ~3B active) +
#                   deepseek-r1:7b (dense reasoning scanner).
#                   Author's recommendation post-soak: fast agentic
#                   tool-use where it matters, reasoning consistency
#                   at tick cadence.
#
# Operators can still override per-role via ACT_SCANNER_MODEL /
# ACT_ANALYST_MODEL (those win over profile selection).

BRAIN_PROFILES: Dict[str, Dict[str, Any]] = {
    # 2026-04 DEFAULT — scanner=qwen3:32b + analyst=deepseek-r1:32b.
    # Matches the April-2026 community consensus (Qwen3 for broad
    # context-processing / macro summarization, DeepSeek-R1 for CoT
    # reasoning over trade decisions). Verified against operator's
    # `ollama list` 2026-04-23.
    "qwen3_r1": {
        "scanner_model": "qwen3:32b",
        "analyst_model": "deepseek-r1:32b",
        "scanner_temperature": 0.2,
        "analyst_temperature": 0.4,
        "description": "Qwen3 32B scanner + DeepSeek-R1 32B analyst. 2026-04 recommended pair.",
    },
    "dense_r1": {
        "scanner_model": "deepseek-r1:7b",
        "analyst_model": "deepseek-r1:32b",
        "scanner_temperature": 0.3,
        "analyst_temperature": 0.4,
        "description": "Both DeepSeek-R1 distills (dense). Lightweight; fallback when Qwen3 swap cost hurts tick cadence.",
    },
    "moe_agentic": {
        "scanner_model": "qwen2.5-coder:7b",
        "analyst_model": "qwen3-coder:30b",     # MoE 30B-A3B (~3B active)
        "scanner_temperature": 0.2,
        "analyst_temperature": 0.3,
        "description": "MoE analyst + dense coder worker. Best for strict JSON/tool-use; slightly weaker raw reasoning than qwen3_r1.",
    },
    # Devstral (agentic-coding) scanner + Qwen3-Coder analyst — the
    # "bot-building" pair per 2026-04 guidance: devstral's agentic
    # tool-calling specialization on the front, qwen3-coder's MoE
    # precision on TradePlan JSON output at the back.
    "devstral_qwen3coder": {
        "scanner_model": "devstral:24b",
        "analyst_model": "qwen3-coder:30b",
        "scanner_temperature": 0.3,
        "analyst_temperature": 0.2,
        "description": "Devstral 24B agentic scanner + Qwen3-Coder 30B analyst (strict JSON / tool-use pair).",
    },
    # Local-only 8GB profile — analyst=qwen3:8b (best 8B for strict JSON
    # / instruction-following per April-2026 benchmarks), scanner=
    # qwen2.5-coder:7b (fast, structured output for scan reports).
    # Total ~5.5GB VRAM, fits comfortably on a 4060 (8GB). Use when
    # OLLAMA_REMOTE_URL is unset and a 30B analyst can't fit locally.
    # Operator directive 2026-04-30: parse_failure rate on
    # qwen2.5-coder:7b-as-analyst was killing every Alpaca tick;
    # qwen3:8b drops parse_failures dramatically. Operator follow-up
    # required: `ollama pull qwen3:8b` on the 4060 box.
    "local_8gb": {
        "scanner_model": "qwen2.5-coder:7b",
        "analyst_model": "qwen3:8b",
        "scanner_temperature": 0.3,
        "analyst_temperature": 0.2,
        "description": "qwen3:8b analyst + qwen2.5-coder:7b scanner. ~5.5GB VRAM total. Use when only 8-10GB available locally and no remote analyst.",
    },
}

# Default profile: moe_agentic. Operator deprecated deepseek-r1 family
# 2026-04-30 — too aggressive a default for boxes that don't have it
# pre-pulled, and the MoE Qwen pair gives strictly better tool-use /
# JSON-strict output for the agentic loop. moe_agentic fits in ~24 GB
# (qwen2.5-coder:7b ~5GB + qwen3-coder:30b ~18GB) so it works on both
# the 32 GB 5090 (live trading) and any 24 GB+ training box.
# Operator can still pick dense_r1 explicitly via
# `setx ACT_BRAIN_PROFILE dense_r1` if they re-add the deepseek models.
DEFAULT_PROFILE = "moe_agentic"
PROFILE_ENV = "ACT_BRAIN_PROFILE"

# Approximate Q4_K_M VRAM footprint by model name. Used for the
# startup VRAM-compatibility warning. Numbers are rough upper bounds
# including KV-cache + embeddings.
_MODEL_VRAM_GB = {
    "deepseek-r1:7b": 6.0,
    "deepseek-r1:32b": 20.0,
    "deepseek-r1:latest": 6.0,
    "qwen2.5-coder:7b": 6.0,
    "qwen3-coder:30b": 20.0,
    "qwen3:32b": 22.0,
    "devstral:24b": 16.0,
    "llama3.2:latest": 3.0,
    "mistral:latest": 6.0,
}


def _vram_estimate_gb(model: str) -> float:
    return _MODEL_VRAM_GB.get(str(model or "").lower(), 10.0)


def _profile_uses_forbidden_models(profile: Dict[str, Any]) -> bool:
    """True if ACT_FORBID_MODELS would block either of this profile's
    pair. Used by auto-downgrade to skip a fallback the operator has
    explicitly retired."""
    try:
        from src.ai.model_guard import is_forbidden  # noqa: PLC0415
        for k in ("scanner_model", "analyst_model"):
            m = profile.get(k, "")
            if m and is_forbidden(m):
                return True
    except Exception:
        pass
    return False


def _resolve_profile(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pick the active profile: env > config > default. Unknown names
    fall back to the default so typos don't break the runtime.

    AUTO-DOWNGRADE: if the resolved profile needs more VRAM than the
    detected GPU has + 2 GB headroom, automatically substitute a
    smaller profile. The fallback chain is:
        1. dense_r1   (deepseek-r1:7b + 32b ~26 GB)
        2. moe_agentic (qwen2.5-coder:7b + qwen3-coder:30b ~24 GB)

    Each candidate is skipped if ACT_FORBID_MODELS blocks either of
    its models -- so an operator who has retired deepseek (`setx
    ACT_FORBID_MODELS deepseek-r1:7b,deepseek-r1:32b`) auto-downgrades
    qwen3_r1 -> moe_agentic instead of qwen3_r1 -> dense_r1. Without
    that check the auto-downgrade was the very thing re-summoning the
    forbidden family on every tick.
    """
    name = os.environ.get(PROFILE_ENV, "").strip()
    source = "env(ACT_BRAIN_PROFILE)"
    if not name and isinstance(config, dict):
        cfg = (config.get("ai") or {}).get("dual_brain") or {}
        name = str(cfg.get("profile") or "").strip()
        if name:
            source = "config.yaml(ai.dual_brain.profile)"
    if not name or name not in BRAIN_PROFILES:
        if name and name not in BRAIN_PROFILES:
            logger.warning("unknown brain profile %r; falling back to %s",
                           name, DEFAULT_PROFILE)
        name = DEFAULT_PROFILE
        source = "DEFAULT_PROFILE"
    profile = BRAIN_PROFILES[name]
    _emit_profile_source_once(name, source, profile)
    _emit_vram_warning_once(name, profile)

    # If the SELECTED profile (not just the auto-downgrade fallback)
    # uses forbidden models, jump to a non-forbidden alternative
    # immediately. Otherwise the resolver returns a profile whose
    # every call will fail with `no_model_resolved` -- which is
    # exactly what the operator hit when ACT_BRAIN_PROFILE was unset
    # (DEFAULT_PROFILE=dense_r1 = deepseek pair, all blocked).
    if _profile_uses_forbidden_models(profile):
        for alt_name in ("moe_agentic", "dense_r1", "qwen3_r1", "hybrid"):
            if alt_name == name:
                continue
            alt = BRAIN_PROFILES.get(alt_name)
            if alt is None or _profile_uses_forbidden_models(alt):
                continue
            logger.warning(
                "[BRAIN] selected profile %r uses forbidden models "
                "(ACT_FORBID_MODELS=%r); switching to %r at runtime",
                name, os.environ.get("ACT_FORBID_MODELS", ""), alt_name,
            )
            return alt
        logger.error(
            "[BRAIN] EVERY available profile is blocked by "
            "ACT_FORBID_MODELS=%r. Either widen the forbid list, set "
            "ACT_BRAIN_PROFILE explicitly to an unblocked profile, or "
            "clear the forbid list. Returning %r unchanged -- LLM "
            "calls will fail until this is resolved.",
            os.environ.get("ACT_FORBID_MODELS", ""), name,
        )
        return profile

    # Auto-downgrade chain: try dense_r1, then moe_agentic. Skip any
    # candidate whose models are blocked by ACT_FORBID_MODELS so the
    # operator's retired family is never re-summoned by the safety
    # rail.
    if os.environ.get("ACT_DISABLE_VRAM_AUTODOWNGRADE", "").strip() == "1":
        return profile

    cap = _detect_vram_gb()
    if cap <= 0:
        return profile

    scanner = profile.get("scanner_model", "")
    analyst = profile.get("analyst_model", "")

    # Remote-analyst awareness: when OLLAMA_REMOTE_URL is set, the
    # analyst doesn't load on the local GPU — only the scanner counts
    # against local VRAM. Without this branch the auto-downgrade wrongly
    # fires on small boxes (e.g. 8 GB 4060) and routes to dense_r1
    # locally, which OOMs since deepseek-r1:32b ~20 GB doesn't fit either.
    _remote_analyst = (os.environ.get("OLLAMA_REMOTE_URL") or "").strip() != ""
    if _remote_analyst:
        needed = _vram_estimate_gb(scanner)
        if needed <= cap + 2:
            logger.info(
                "[BRAIN] remote analyst configured (OLLAMA_REMOTE_URL); "
                "scanner-only VRAM check: scanner=%s ~%.1f GB / cap %.1f GB — fits",
                scanner, needed, cap,
            )
            return profile
    else:
        needed = _vram_estimate_gb(scanner) + _vram_estimate_gb(analyst)
        if needed <= cap + 2:
            return profile

    # Profile doesn't fit. Walk fallback chain — moe_agentic first
    # (operator-preferred default, no deepseek dependency), dense_r1
    # next, local_8gb last (single 7B shared by scanner+analyst — only
    # path that fits on a 4060-class box with no remote analyst).
    for fallback_name in ("moe_agentic", "hybrid", "dense_r1", "local_8gb"):
        if fallback_name == name:
            continue   # already on this one; don't loop
        candidate = BRAIN_PROFILES.get(fallback_name)
        if candidate is None:
            continue
        if _profile_uses_forbidden_models(candidate):
            logger.warning(
                "[BRAIN] AUTO-DOWNGRADE: skipping fallback %r -- one of its "
                "models is in ACT_FORBID_MODELS=%r",
                fallback_name, os.environ.get("ACT_FORBID_MODELS", ""),
            )
            continue
        logger.warning(
            "[BRAIN] AUTO-DOWNGRADE: profile %r needs ~%.1f GB but only "
            "%.1f GB available; switching to %r at runtime. "
            "Set ACT_DISABLE_VRAM_AUTODOWNGRADE=1 to override (not "
            "recommended unless you've manually verified VRAM).",
            name, needed, cap, fallback_name,
        )
        return candidate

    # Every fallback was forbidden. Stick with the original choice
    # (which will likely error at the LLM call) so the operator
    # surfaces the contradiction rather than silently using a model
    # they banned.
    logger.error(
        "[BRAIN] AUTO-DOWNGRADE: every fallback (moe_agentic, hybrid, dense_r1) "
        "is blocked by ACT_FORBID_MODELS=%r AND profile %r doesn't fit "
        "in %.1f GB. Either widen the forbid list or shrink the profile.",
        os.environ.get("ACT_FORBID_MODELS", ""), name, cap,
    )
    return profile


_PROFILE_SOURCE_LOGGED: Dict[str, bool] = {}


def _emit_profile_source_once(name: str, source: str, profile: Dict[str, Any]) -> None:
    """Log which env / config / default chose the active profile.

    One-shot per (name,source) pair so the truth is visible exactly
    once per process. Without this, operators wonder why dual_brain
    is using a profile they thought they replaced -- it could be a
    persistent setx, a config.yaml fallback, or the hardcoded
    DEFAULT_PROFILE, and previously there was no audit trail.
    """
    key = f"{name}|{source}"
    if _PROFILE_SOURCE_LOGGED.get(key):
        return
    _PROFILE_SOURCE_LOGGED[key] = True
    logger.info(
        "[BRAIN] resolved profile=%r source=%s scanner=%r analyst=%r",
        name, source,
        profile.get("scanner_model", ""),
        profile.get("analyst_model", ""),
    )


_VRAM_CACHE_GB: Dict[str, float] = {}


def _detect_vram_gb() -> float:
    """Best-effort nvidia-smi VRAM detection. Cached for the process
    lifetime so we don't shell out per-call."""
    if "cap" in _VRAM_CACHE_GB:
        return _VRAM_CACHE_GB["cap"]
    cap = 0.0
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode().strip().splitlines()
        if out:
            cap = float(out[0]) / 1024.0  # MiB → GiB
    except Exception:
        pass
    _VRAM_CACHE_GB["cap"] = cap
    return cap


_VRAM_WARNED: Dict[str, bool] = {}


def _emit_vram_warning_once(name: str, profile: Dict[str, Any]) -> None:
    if _VRAM_WARNED.get(name):
        return
    _VRAM_WARNED[name] = True
    try:
        import os as _os
        scanner = profile.get("scanner_model", "")
        analyst = profile.get("analyst_model", "")
        needed = _vram_estimate_gb(scanner) + _vram_estimate_gb(analyst)
        cap = _detect_vram_gb()
        if cap == 0:
            return
        if needed > cap + 2:
            logger.warning(
                "[BRAIN] Profile %r needs ~%.1f GB VRAM (scanner=%s + "
                "analyst=%s) but only %.1f GB detected. Expect empty-LLM "
                "errors. Switch to dense_r1 (7B+32B, ~24 GB) or "
                "moe_agentic (7B+30B, ~24 GB) via "
                "`setx ACT_BRAIN_PROFILE dense_r1` + fresh cmd.",
                name, needed, scanner, analyst, cap,
            )
        else:
            logger.info(
                "[BRAIN] Profile %r selected (scanner=%s, analyst=%s, "
                "~%.1f GB needed, %.1f GB available)",
                name, scanner, analyst, needed, cap or 0,
            )
    except Exception:
        pass


# ── Defaults (fall-through when neither profile nor explicit cfg) ───────

_DEFAULT_PROFILE_FIELDS = BRAIN_PROFILES[DEFAULT_PROFILE]
DEFAULT_SCANNER_MODEL = _DEFAULT_PROFILE_FIELDS["scanner_model"]
DEFAULT_ANALYST_MODEL = _DEFAULT_PROFILE_FIELDS["analyst_model"]
DEFAULT_SCANNER_TEMP = _DEFAULT_PROFILE_FIELDS["scanner_temperature"]
DEFAULT_ANALYST_TEMP = _DEFAULT_PROFILE_FIELDS["analyst_temperature"]
DEFAULT_STRIP_THINK_TAGS_FROM_SCANNER = True    # compact scanner output


# ── Shared performance target (injected into both brains) ─────────────
# Operator directive (non-negotiable): 1%/day on ANY venue, ANY spread.
# Cost awareness is a tool the brain uses to filter unprofitable
# setups — it is NOT a ceiling that lowers ambition. Per memory rule
# `feedback_target_is_non_negotiable`, never frame the spread as a
# structural cap and never inject "honest caveats" that would lower
# the brain's hunt threshold. The brain's job is to FIND setups that
# clear whatever friction exists and hit the target.
def _format_performance_target() -> str:
    try:
        from src.trading.cost_gate import get_spread_pct  # noqa: PLC0415
        spread_pct = get_spread_pct("robinhood")
    except Exception:
        spread_pct = 1.69
    min_move_pct = max(2.0, spread_pct * 1.5)
    return (
        "## PERFORMANCE TARGET\n"
        "Operator goal (non-negotiable): simple +1%/day -> +7%/week -> "
        "+30%/month -> +365%/year on seed capital. Hit it on ANY venue, "
        "ANY spread.\n"
        f"Cost context: current venue round-trip is {spread_pct:.2f}% "
        f"(operator-tunable via ACT_ROBINHOOD_SPREAD_PCT). Use this as a "
        f"hard floor for setup quality -- expected move must exceed "
        f"~{min_move_pct:.2f}% so the trade can clear friction with edge "
        "left over. This is a filter, not a ceiling.\n"
        "Your job:\n"
        "  1. HUNT every tick for setups whose expected move clears "
        "friction with conviction.\n"
        "  2. SKIP confidently when the evidence is mixed -- a missed "
        "trade costs nothing; a forced trade costs spread + loss.\n"
        "  3. SIZE for the conviction, not for the daily number. "
        "Outlier 1%+ days come from clean sniper-tier setups, not from "
        "stop-tightening.\n"
        "  4. The dashboard tracks today_pct, rolling_7d_avg, best_day, "
        "gap_to_1pct_day -- outlier wins are celebrated alongside average."
    )


PERFORMANCE_TARGET = _format_performance_target()


SCANNER_SYSTEM = (
    "You are ACT's SCANNER brain (right hemisphere). Your job is pattern "
    "recognition across the current market state — surveying prices, "
    "news headlines, sentiment, regime, cross-exchange signals — and "
    "flagging opportunities for the Analyst to investigate. Do NOT "
    "compile a full trade plan; that's the Analyst's job. Keep output "
    "under 400 characters.\n\n"
    "## OUTPUT FORMAT — STRICT JSON ONLY\n"
    "Return exactly one JSON object using DOUBLE QUOTES for all keys "
    "and string values. Do not wrap in markdown fences. Do not add "
    "prose before or after. Example:\n"
    '  {"opportunity_score": 65, "top_signals": ["ema_cross", "vol_spike"], '
    '"proposed_direction": "LONG", "rationale": "BTC breaking 4h range '
    'on volume, macro supportive."}\n\n'
    + PERFORMANCE_TARGET
)

ANALYST_SYSTEM = (
    "You are ACT's ANALYST brain (left hemisphere). Given the Scanner's "
    "assessment + market context + tool access, compile a structured "
    "TradePlan via multi-turn reasoning. When emitting a plan, ensure "
    "every numeric field is grounded in VERIFIED QUANT DATA or a tool "
    "result — never hallucinate numbers. Authority rules are absolute; "
    "do not propose trades that violate them.\n\n"
    "## OUTPUT FORMAT — STRICT JSON ONLY\n"
    "Emit exactly ONE JSON object per turn. Use DOUBLE QUOTES for "
    "every key and string value. Do not wrap in markdown fences. Do "
    "not add prose, reasoning, or commentary outside the JSON. Three "
    "valid shapes:\n"
    '  {"tool_call": {"name": "<tool>", "args": {...}}}\n'
    '  {"plan": {"direction": "LONG"|"SHORT"|"FLAT", "size_pct": 1.5, '
    '"tier": "sniper"|"normal"|"skip", "thesis": "..."}}\n'
    '  {"skip": "<one-line reason>"}\n\n'
    "## RECOMMENDED TOOL USE (liberal — don't compile a plan blind)\n"
    "  * Call `ask_risk_guardian` + `ask_loss_prevention` before ANY "
    "non-skip plan — stop-distance sanity + size sanity.\n"
    "  * Call `get_readiness_state` to confirm the gate is where you "
    "expect before committing real capital.\n"
    "  * Call `backtest_hypothesis` on high-conviction setups to "
    "sanity-check Sharpe/WR on recent bars.\n"
    "  * Call `ask_debate` when your own conviction is mid (0.5–0.7) "
    "— it stress-tests your thesis against opposing agents.\n"
    "  * Call `query_knowledge_graph` for the current real-time "
    "entity/edge state if the seed digest is thin.\n"
    "  * Call `get_body_controls` to see which specialist agents the "
    "current scanner signals want you to query first.\n"
    "  * Call quant tools (fit_ou_process, hurst_exponent, hmm_regime, "
    "kalman_trend) when the setup depends on a quant claim.\n"
    "  * When unsure about system state itself, call "
    "`dispatch_skill` with {name:'status'} or {name:'diagnose-noop'}.\n\n"
    "## REASONING WEIGHTING — REGIME-AWARE\n"
    "Empirical finding (FS-ReasoningAgent, NUS+HKUST, arXiv:2410.12464):\n"
    "subjective signals dominate bull markets; factual signals dominate\n"
    "bear markets. Stronger LLMs over-index on facts and lose alpha by\n"
    "ignoring sentiment in euphoric regimes. Counter this explicitly:\n"
    "  1. FIRST CALL `get_macro_bias` — it returns signed_bias [-1, +1]\n"
    "     and a `composite_signal` string.\n"
    "  2. WEIGHT EVIDENCE BY REGIME:\n"
    "     * BULL (signed_bias > +0.20 OR composite_signal=BULLISH):\n"
    "       weight SUBJECTIVE evidence 60% (sentiment, news headlines,\n"
    "       polymarket odds, fear/greed extremes, social_sentiment\n"
    "       layer); FACTUAL evidence 40% (macro layers, regulatory).\n"
    "     * BEAR (signed_bias < -0.20 OR composite_signal=BEARISH or\n"
    "       CRISIS): weight FACTUAL evidence 60% (macro_indicators,\n"
    "       central_bank, on-chain whale flows, regulatory enforcement,\n"
    "       institutional flows); SUBJECTIVE evidence 40% — sentiment\n"
    "       lies in fear-driven markets.\n"
    "     * NEUTRAL (|signed_bias| <= 0.20): 50/50 with a slight skew\n"
    "       toward FACTUAL — chop is often confused with fading bull\n"
    "       sentiment.\n"
    "  3. TAG each item in supporting_evidence with one of\n"
    "     {factual, subjective, mixed, technical} so the post-mortem\n"
    "     can analyze which evidence type drove this trade's outcome.\n"
    "     Example: \"supporting_evidence\": [\n"
    "       {\"kind\":\"factual\", \"src\":\"get_macro_bias\", \"note\":\"USD weakening, signed_bias=+0.35\"},\n"
    "       {\"kind\":\"subjective\", \"src\":\"sentiment_decoder\", \"note\":\"Reddit bullish 78%\"},\n"
    "       {\"kind\":\"technical\", \"src\":\"hurst_exponent\", \"note\":\"H=0.62, trending\"}\n"
    "     ]\n\n"
    + PERFORMANCE_TARGET
)


@dataclass
class BrainConfig:
    """Resolved per-brain configuration (env > config > default)."""
    role: str                # 'scanner' | 'analyst'
    model: str
    temperature: float
    system_prompt: str

    def with_system_prompt(self, extra: str) -> "BrainConfig":
        sp = self.system_prompt + ("\n\n" + extra if extra else "")
        return BrainConfig(
            role=self.role, model=self.model,
            temperature=self.temperature, system_prompt=sp,
        )


@dataclass
class BrainResponse:
    """One completed brain call, with provenance for audit/credit."""
    brain: str               # 'scanner' | 'analyst'
    model: str
    text: str
    ok: bool
    fallback_used: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "brain": self.brain,
            "model": self.model,
            "text": self.text[:2000],
            "ok": self.ok,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


# ── Config resolution ───────────────────────────────────────────────────


def _resolve(config: Optional[Dict[str, Any]], brain: str) -> BrainConfig:
    """Resolve the active BrainConfig for a role.

    Precedence:
      1. Per-role env override (ACT_SCANNER_MODEL / ACT_ANALYST_MODEL).
      2. Per-role explicit config key (scanner_model / analyst_model).
      3. Active profile's default (ACT_BRAIN_PROFILE or
         config.ai.dual_brain.profile or DEFAULT_PROFILE).
    """
    assert brain in VALID_BRAINS
    cfg: Dict[str, Any] = {}
    if isinstance(config, dict):
        cfg = (config.get("ai") or {}).get("dual_brain") or {}

    profile_defaults = _resolve_profile(config)

    if brain == SCANNER:
        model = (
            os.environ.get("ACT_SCANNER_MODEL")
            or cfg.get("scanner_model")
            or profile_defaults["scanner_model"]
        )
        temp = float(cfg.get("scanner_temperature")
                     or profile_defaults["scanner_temperature"])
        sys_prompt = SCANNER_SYSTEM
    else:
        model = (
            os.environ.get("ACT_ANALYST_MODEL")
            or cfg.get("analyst_model")
            or profile_defaults["analyst_model"]
        )
        temp = float(cfg.get("analyst_temperature")
                     or profile_defaults["analyst_temperature"])
        sys_prompt = ANALYST_SYSTEM
    return BrainConfig(role=brain, model=str(model), temperature=float(temp), system_prompt=sys_prompt)


def is_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Env kill switch > config flag > default off for safety."""
    if os.environ.get(DISABLE_ENV, "0") == "1":
        return False
    if isinstance(config, dict):
        cfg = (config.get("ai") or {}).get("dual_brain") or {}
        return bool(cfg.get("enabled", False))
    return False


# ── Core call ───────────────────────────────────────────────────────────


def _llm_call(
    model: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
) -> str:
    """Dispatch through the existing LLMRouter. Never raises — returns
    empty string on any provider error so callers can fall through."""
    try:
        from src.ai.llm_provider import LLMRouter
        router = LLMRouter()
        router.add_from_env()
    except Exception as e:
        logger.debug("dual_brain: LLMRouter unavailable: %s", e)
        return ""

    # Prefer a provider that matches the requested model; otherwise
    # try each in turn. LLMRouter's provider config carries the model
    # name; we mutate it for this call only.
    providers = list(router.providers.items())
    if not providers:
        return ""

    # Try to find a provider whose .config.model matches the brain model
    # (exact match or prefix). If not, use the first one and override
    # its model field in-place for this call.
    target = None
    for _name, prov in providers:
        prov_model = getattr(getattr(prov, "config", None), "model", "")
        if prov_model and str(prov_model).lower() == str(model).lower():
            target = prov
            break
    if target is None:
        target = providers[0][1]

    # Snapshot + override + restore — ugly but avoids requiring the LLMRouter
    # to expose per-call model override. Concurrent calls with different
    # models on the same provider are rare in the agentic loop (scanner and
    # analyst serialize naturally), but we still guard with a try/finally.
    prev_model = getattr(getattr(target, "config", None), "model", None)
    prev_temp = getattr(getattr(target, "config", None), "temperature", None)
    try:
        if hasattr(target, "config") and target.config is not None:
            try:
                target.config.model = model
            except Exception:
                pass
            try:
                target.config.temperature = temperature
            except Exception:
                pass
        resp = target.generate(prompt=prompt, system_prompt=system_prompt)
        text = resp.get("response") if isinstance(resp, dict) else str(resp)
        text = str(text or "")
        if not text.strip():
            # Empty response means the provider errored silently (Ollama
            # OOM on model load, all endpoints refused, rate-limited, etc.).
            # Surface this at WARNING so zero-trade diagnostics can see it.
            resp_error = (resp.get("error") if isinstance(resp, dict) else "") or "empty response"
            logger.warning(
                "dual_brain: provider %s returned empty text for model=%s err=%r",
                type(target).__name__, model, str(resp_error)[:200],
            )
        return text
    except Exception as e:
        logger.warning("dual_brain: provider.generate failed for %s: %s", model, e)
        return ""
    finally:
        if hasattr(target, "config") and target.config is not None:
            try:
                if prev_model is not None:
                    target.config.model = prev_model
                if prev_temp is not None:
                    target.config.temperature = prev_temp
            except Exception:
                pass


# ── Public surface ──────────────────────────────────────────────────────


def call_brain(
    brain: str,
    user_prompt: str,
    *,
    extra_system: str = "",
    config: Optional[Dict[str, Any]] = None,
    allow_fallback: bool = True,
    llm_call: Optional[Callable[..., str]] = None,
) -> BrainResponse:
    """Invoke the scanner or analyst. Always returns a BrainResponse;
    never raises.

    Cross-fallback: if the primary brain returns empty/error AND
    allow_fallback is True AND `fallback_cross` is set in config (default
    True), try the other brain with a small adapter prefix.

    `llm_call` is injectable for tests: callable
    `(model, prompt, system_prompt, temperature) -> str`.
    """
    if brain not in VALID_BRAINS:
        return BrainResponse(brain=brain, model="", text="",
                             ok=False, error=f"invalid brain {brain!r}")
    cfg = _resolve(config, brain)
    if extra_system:
        cfg = cfg.with_system_prompt(extra_system)

    caller = llm_call or _llm_call
    text = ""
    try:
        text = caller(cfg.model, user_prompt, cfg.system_prompt, cfg.temperature) or ""
    except Exception as e:
        logger.debug("dual_brain.call_brain primary %s raised: %s", brain, e)
        text = ""

    if text.strip():
        # Scanner runs every tick and feeds brain_memory — strip <think>
        # traces so the next analyst cycle sees the compact answer only.
        # Analyst keeps its full reasoning trace (useful for audit +
        # learned preference from past TradePlans).
        if brain == SCANNER and DEFAULT_STRIP_THINK_TAGS_FROM_SCANNER:
            text = strip_reasoning_tags(text)
        return BrainResponse(brain=brain, model=cfg.model, text=text, ok=True)

    # Fall back to the other brain if requested.
    cross_ok = True
    if isinstance(config, dict):
        cross_ok = bool(((config.get("ai") or {}).get("dual_brain") or {}).get("fallback_cross", True))
    if not allow_fallback or not cross_ok:
        return BrainResponse(brain=brain, model=cfg.model, text="",
                             ok=False, error="primary_empty_no_fallback")

    other = ANALYST if brain == SCANNER else SCANNER
    other_cfg = _resolve(config, other)
    other_prompt = (
        f"[FALLBACK from {brain}] Please answer in the {brain}'s expected "
        f"format:\n\n{user_prompt}"
    )
    try:
        text2 = caller(other_cfg.model, other_prompt, cfg.system_prompt, cfg.temperature) or ""
    except Exception as e:
        text2 = ""
        logger.debug("dual_brain fallback %s raised: %s", other, e)
    if text2.strip():
        return BrainResponse(
            brain=brain, model=other_cfg.model, text=text2,
            ok=True, fallback_used=True,
        )
    return BrainResponse(
        brain=brain, model=cfg.model, text="",
        ok=False, error="both_brains_empty",
    )


def scan(user_prompt: str, *, config: Optional[Dict[str, Any]] = None,
         **kwargs) -> BrainResponse:
    """Shorthand — invoke the scanner (right brain)."""
    return call_brain(SCANNER, user_prompt, config=config, **kwargs)


def analyze(user_prompt: str, *, config: Optional[Dict[str, Any]] = None,
            **kwargs) -> BrainResponse:
    """Shorthand — invoke the analyst (left brain)."""
    return call_brain(ANALYST, user_prompt, config=config, **kwargs)


# ── Integration hook for the agentic loop ───────────────────────────────


def build_analyst_llm_call(
    config: Optional[Dict[str, Any]] = None,
) -> Callable[[List[Dict[str, Any]]], str]:
    """Returns a `(messages) -> str` closure suitable for
    src.ai.agentic_trade_loop.AgenticTradeLoop.llm_call that routes
    every turn to the Analyst brain.

    The agentic loop's iterated tool-use is the Analyst's natural
    habitat; scanner runs once per tick independently, not inside the
    loop.
    """
    cfg = _resolve(config, ANALYST)

    def _closure(messages: List[Dict[str, Any]]) -> str:
        # Flatten the message list into a single prompt (the existing
        # LLMRouter.generate shape).
        prompt_parts: List[str] = []
        for m in messages or []:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                import json as _json
                content = _json.dumps(content, default=str)
            if role == "system":
                continue  # system prompt is the Analyst's fixed one
            prompt_parts.append(f"[{role}]\n{content}")
        prompt = "\n\n".join(prompt_parts)
        return _llm_call(cfg.model, prompt, cfg.system_prompt, cfg.temperature)

    return _closure
