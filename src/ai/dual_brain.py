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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Brain names — keep these stable; other modules check against them.
SCANNER = "scanner"
ANALYST = "analyst"
VALID_BRAINS = (SCANNER, ANALYST)

DISABLE_ENV = "ACT_DISABLE_DUAL_BRAIN"


# ── Defaults (env > config > these) ─────────────────────────────────────

DEFAULT_SCANNER_MODEL = "qwen3:32b"
DEFAULT_ANALYST_MODEL = "devstral:24b"
DEFAULT_SCANNER_TEMP = 0.4
DEFAULT_ANALYST_TEMP = 0.1


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
    assert brain in VALID_BRAINS
    cfg = {}
    if isinstance(config, dict):
        cfg = (config.get("ai") or {}).get("dual_brain") or {}

    if brain == SCANNER:
        model = os.environ.get("ACT_SCANNER_MODEL") or cfg.get("scanner_model") or DEFAULT_SCANNER_MODEL
        temp = float(cfg.get("scanner_temperature") or DEFAULT_SCANNER_TEMP)
        sys_prompt = SCANNER_SYSTEM
    else:
        model = os.environ.get("ACT_ANALYST_MODEL") or cfg.get("analyst_model") or DEFAULT_ANALYST_MODEL
        temp = float(cfg.get("analyst_temperature") or DEFAULT_ANALYST_TEMP)
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
