"""
Dual-brain LLM router — scanner (right-brain) + analyst (left-brain).

Operator-directed split (2026-04-23):

  * **Scanner = Qwen 3 32B** (right brain). Pattern-recognition and broad
    market-state survey. Runs every tick. Slightly creative temperature
    (0.4) so it picks up weak signals the analyst would miss.

  * **Analyst = Devstral 24B** (left brain). Mistral's agentic-coding
    model. Purpose-built for structured tool-use + Pydantic JSON
    output. Runs on-demand when scanner flags a setup. Low temperature
    (0.1) so TradePlan compilation is deterministic under Pydantic
    validation.

Both run locally via Ollama on the RTX 5090 at Q4_K_M quantization
(~20GB + ~14GB = 34GB peak). Ollama's model-swap handles the RAM
pressure when both can't be resident simultaneously.

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
    "dense_r1": {
        "scanner_model": "deepseek-r1:7b",
        "analyst_model": "deepseek-r1:32b",
        "scanner_temperature": 0.3,
        "analyst_temperature": 0.4,
        "description": "Both DeepSeek-R1 distills (dense). Most consistent; safest for paper soak.",
    },
    "moe_agentic": {
        "scanner_model": "qwen2.5-coder:7b",
        "analyst_model": "qwen3-coder:30b",     # MoE 30B-A3B (~3B active)
        "scanner_temperature": 0.2,
        "analyst_temperature": 0.3,
        "description": "MoE analyst + dense coder worker. Fastest tool use; 2026 agentic gold standard.",
    },
    "hybrid": {
        "scanner_model": "deepseek-r1:7b",      # dense reasoning worker
        "analyst_model": "qwen3-coder:30b",     # MoE agentic orchestrator
        "scanner_temperature": 0.3,
        "analyst_temperature": 0.3,
        "description": "MoE analyst + dense reasoning scanner. Post-soak recommendation.",
    },
}

DEFAULT_PROFILE = "dense_r1"
PROFILE_ENV = "ACT_BRAIN_PROFILE"


def _resolve_profile(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Pick the active profile: env > config > default. Unknown names
    fall back to the default so typos don't break the runtime."""
    name = os.environ.get(PROFILE_ENV, "").strip()
    if not name and isinstance(config, dict):
        cfg = (config.get("ai") or {}).get("dual_brain") or {}
        name = str(cfg.get("profile") or "").strip()
    if not name or name not in BRAIN_PROFILES:
        if name and name not in BRAIN_PROFILES:
            logger.warning("unknown brain profile %r; falling back to %s",
                           name, DEFAULT_PROFILE)
        name = DEFAULT_PROFILE
    return BRAIN_PROFILES[name]


# ── Defaults (fall-through when neither profile nor explicit cfg) ───────

_DEFAULT_PROFILE_FIELDS = BRAIN_PROFILES[DEFAULT_PROFILE]
DEFAULT_SCANNER_MODEL = _DEFAULT_PROFILE_FIELDS["scanner_model"]
DEFAULT_ANALYST_MODEL = _DEFAULT_PROFILE_FIELDS["analyst_model"]
DEFAULT_SCANNER_TEMP = _DEFAULT_PROFILE_FIELDS["scanner_temperature"]
DEFAULT_ANALYST_TEMP = _DEFAULT_PROFILE_FIELDS["analyst_temperature"]
DEFAULT_STRIP_THINK_TAGS_FROM_SCANNER = True    # compact scanner output


SCANNER_SYSTEM = (
    "You are ACT's SCANNER brain (right hemisphere). Your job is pattern "
    "recognition across the current market state — surveying prices, "
    "news headlines, sentiment, regime, cross-exchange signals — and "
    "flagging opportunities for the Analyst to investigate. Output a "
    "compact JSON assessment: {'opportunity_score': 0-100, "
    "'top_signals': [...], 'proposed_direction': 'LONG'|'SHORT'|'FLAT', "
    "'rationale': '<=2 sentences'}. Do NOT compile a full trade plan; "
    "that's the Analyst's job. Keep output under 400 characters."
)

ANALYST_SYSTEM = (
    "You are ACT's ANALYST brain (left hemisphere). Given the Scanner's "
    "assessment + market context + tool access, compile a structured "
    "TradePlan via multi-turn reasoning. Emit ONE JSON envelope per "
    "turn: {tool_call} / {plan} / {skip}. When emitting a plan, ensure "
    "every numeric field is grounded in VERIFIED QUANT DATA or a tool "
    "result — never hallucinate numbers. Authority rules are absolute; "
    "do not propose trades that violate them."
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
        return str(text or "")
    except Exception as e:
        logger.debug("dual_brain: provider.generate failed for %s: %s", model, e)
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
