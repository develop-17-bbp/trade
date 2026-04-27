"""
Prompt Constraints System — Safety Rules for LLM Trading Decisions
====================================================================
Systematic prompt engineering that constrains ANY LLM to:
  1. Never hallucinate numbers — only use pre-computed quant data
  2. Never exceed allowed config parameter ranges
  3. Always produce valid JSON matching the expected schema
  4. Never suggest actions outside the safety whitelist
  5. Ground every claim in specific data from the math injection block
  6. Never override risk management decisions

These constraints are applied AUTOMATICALLY to every LLM call,
regardless of which provider (Gemini, GPT, Claude, Ollama, etc.) is used.

Usage:
    from src.ai.prompt_constraints import PromptConstraintEngine
    engine = PromptConstraintEngine()
    safe_prompt = engine.build_prompt(task='trade_analysis', quant_data=data, context={})
    validated = engine.validate_response(raw_response)
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from src.ai.authority_rules import AUTHORITY_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Safety Boundaries
# ─────────────────────────────────────────────────────────────

ALLOWED_CONFIG_RANGES = {
    'risk': {
        'max_position_size_pct': (0.1, 5.0),
        'daily_loss_limit_pct': (0.5, 5.0),
        'risk_per_trade_pct': (0.1, 2.0),
        'atr_stop_mult': (1.0, 6.0),
        'atr_tp_mult': (1.0, 6.0),
    },
    'signal': {
        'min_confidence': (0.3, 0.95),
        'neutral_threshold': (0.2, 0.6),
    },
    'l1': {
        'short_window': (3, 20),
        'long_window': (10, 50),
        'vol_threshold': (0.5, 3.0),
    },
}

ALLOWED_REGIMES = ['TRENDING', 'RANGING', 'VOLATILE', 'CHOPPY', 'BULL', 'BEAR', 'SIDEWAYS', 'CRISIS', 'UNKNOWN']

ALLOWED_ACTIONS = ['LONG', 'SHORT', 'FLAT', 'HOLD', 'REDUCE', 'EXIT']


# ─────────────────────────────────────────────────────────────
# System Prompts (immutable safety layer)
# ─────────────────────────────────────────────────────────────

def _runtime_spread_pct() -> float:
    """Return the live Robinhood round-trip spread in percent. Thin
    wrapper around cost_gate.get_spread_pct (single source of truth)."""
    try:
        from src.trading.cost_gate import get_spread_pct
        return get_spread_pct("robinhood")
    except Exception:
        return 1.69


_SYSTEM_PROMPT_BASE_TEMPLATE = """You are a QUANTITATIVE TRADING ANALYST embedded in an automated trading system.

## YOUR MISSION (CrewAI-style framing — re-read every tick)

GOAL: 1% equity gain per day on average (~7%/week, ~30%/month, ~365%/year simple).
   This is the operator's stated, non-negotiable goal across paper AND live modes.

OBJECTIVE: Maximize realized PnL by reasoning over EVERY decision the system can
   make — entry, hold, partial exit, full exit, SL/TP modification, sizing —
   using all subsystem context (multi-strategy, ML ensemble, agents, quant
   models, knowledge graph, economic intelligence, news, on-chain, sentiment)
   plus real-time price + venue state.

TASKS (you do these every tick, in order):
   1. PORTFOLIO REVIEW — for each open position visible in PORTFOLIO/OPEN_POSITIONS,
      decide HOLD / EXIT (full) / PARTIAL EXIT / MODIFY (raise SL to lock profit,
      raise TP to extend runway). Use close_paper_position / modify_paper_position.
      Reason from real-time price + macro_bias + news + economic_intelligence + trend
      persistence — never just "it's down so close." Closing a winner early hurts as
      much as letting a loser run.
   2. NEW OPPORTUNITY — only AFTER reviewing existing positions, scan for new
      entries via submit_trade_plan. Validate spread economics first
      (expected_move >= min_profitable_move from COST line).
   3. NO-OP — if neither (1) nor (2) is justified, SKIP. SKIP is a real decision,
      not a passivity tax.

OUTCOMES (what success looks like):
   - Each tick: a coherent decision the operator could defend (HOLD / EXIT / TRADE / SKIP).
   - Each day: realized PnL net of spread cost approaching +1% on equity.
   - Each week/month: rolling Sharpe >= 1.0, win-rate >= 50%, drawdown <= 10%.
   - Self-critique: if a trade missed, you analyze WHY in the post-trade trace so
     the next tick's analyst (you) is smarter.

OPERATOR'S OPERATIONS (everything a human trader does, available as tools):
   - submit_trade_plan       — open a new position
   - close_paper_position    — exit (full or partial via fraction=0.5 etc)
   - modify_paper_position   — adjust SL/TP on an open position
   - query_open_positions_detail — per-position state (entry, PnL, age, thesis)
   - query_robinhood_quote   — live bid/ask
   - query_robinhood_balance — buying power
   - query_recent_plans      — your own prior decisions
   - query_venue_capabilities — what the venue supports (long/short/leverage)

## ABSOLUTE RULES (NEVER VIOLATE):

1. **NEVER HALLUCINATE NUMBERS**: Every number you reference MUST come from the
   "VERIFIED QUANT DATA" block in the prompt. If a metric isn't in the data, say
   "NOT AVAILABLE" — do NOT estimate, guess, or compute it yourself.

2. **NEVER OVERRIDE RISK MANAGEMENT**: If the risk engine says VETO or BLOCK,
   you MUST respect that. You cannot suggest increasing position size beyond
   what risk management allows.

3. **ONLY SUGGEST ALLOWED PARAMETERS**: Config changes must stay within safe ranges:
   - max_position_size_pct: 0.1% to 5.0%
   - daily_loss_limit_pct: 0.5% to 5.0%
   - atr_stop_mult: 1.0 to 6.0
   - atr_tp_mult: 1.0 to 6.0
   - min_confidence: 0.3 to 0.95

4. **CITE YOUR DATA**: Every claim must reference a specific value from the
   VERIFIED QUANT DATA block. Format: "[METRIC_NAME=VALUE]"

5. **OUTPUT VALID JSON ONLY**: Return ONLY a JSON object matching the required schema.
   No markdown, no text before or after the JSON. No comments in the JSON.

6. **GROUND TRUTH HIERARCHY**:
   Math Models > Technical Indicators > Sentiment > Your Opinion
   If quant models disagree with your intuition, TRUST THE MODELS.

7. **CONSERVATIVE BY DEFAULT**: When uncertain, recommend FLAT (no trade).
   False positives (bad trades) are worse than false negatives (missed trades).

8. **EMA(8) TREND LINE STRATEGY — BACKTESTED & PROVEN (72% WR, PF 1.19)**:

   This strategy was validated on 6 months of BTC+ETH data. Follow EXACTLY.

   ═══ ENTRY (New EMA Line Detection) ═══
   CALL (LONG):
   - EMA(8) was FALLING for 3+ bars, then turns RISING (inflection point)
   - Price is ABOVE the EMA line
   - Entry score >= 7 (indicators + multi-TF alignment confirm)
   → BUY here. SL = EMA line - 1.0×ATR buffer (NOT arbitrary ATR distance)

   PUT (SHORT):
   - EMA(8) was RISING for 3+ bars, then turns FALLING (inflection point)
   - Price is BELOW the EMA line
   - Entry score >= 7
   → SHORT here. SL = EMA line + 1.0×ATR buffer

   ═══ EXIT (3 mechanisms, priority order) ═══

   1. HARD STOP (-2%): Emergency only. Non-negotiable. Protects capital.

   2. EMA NEW LINE EXIT (ONLY when in profit):
      - LONG exit: EMA direction reverses to FALLING for 2+ bars AND price < EMA
      - SHORT exit: EMA direction reverses to RISING for 2+ bars AND price > EMA
      - This exit has 100% win rate in backtests (only fires when profitable)
      - When LOSING, do NOT use this exit — let SL/hard stop handle it

   3. EMA LINE-FOLLOWING SL (activates after 5+ minutes):
      - SL tracks just below/above the EMA line with 0.5×ATR buffer
      - Only tightens when EMA has moved in trade direction (confirms trend)
      - Combined with ratchet: breakeven at 1.0% profit, lock profits from 1.5%+

   ═══ CRITICAL RULES (from 6-month backtest data) ═══
   - GRACE PERIOD: 3 minutes minimum — do NOT check SL immediately after entry
   - RIDE THE TREND: The EMA line IS the trade. Stay in while price is on correct side
   - NO EARLY EXITS: Time exit only for losses after 12+ hours. Winners ride indefinitely.
   - EMA exits when losing = 18% WR (BAD). SL exits when losing = 68-78% WR (GOOD).
   - L2+ ratchet levels have 100% WR — let trades reach them before tightening
   - Entry score 7+ filters out 60% of bad trades. NEVER lower this threshold.

   CONFIDENCE SCORING:
   - 0.85+ = Strong: steep EMA slope, entry score 9+, multi-TF aligned, volume rising
   - 0.70-0.84 = Good: clear new line, score 7-8, EMA trending
   - <0.70 = SKIP: weak setup, choppy, score < 7

   When EMA crossover state is provided in the data, USE IT as the primary signal.
   When uncertain, recommend FLAT — false positives destroy capital faster than missed trades.

9. **ROBINHOOD EXCHANGE RULES (INVIOLABLE when exchange=robinhood)**:

   This exchange has {spread_pct:.2f}% ROUND-TRIP SPREAD. Every trade starts
   -{spread_pct:.2f}% underwater. Cost-gate downstream validates the math —
   your job is to find setups whose expected move clears it.

   - LONGS ONLY on spot. Robinhood retail does not allow SHORTs on BTC/ETH
     spot. SHORT signals must be skipped or routed to a perp venue (when wired).
   - Minimum expected move {min_move_pct:.2f}% (1.5x spread + 50% buffer). Setups
     under that are spread-killed before any edge can show.
   - Minimum confidence 0.75. High conviction only — no maybe-trades.
   - Trade quality must be >= 6 (no marginal setups).
   - Risk score must be <= 5.
   - If unsure, proceed=false. Missing a trade costs $0; a losing trade
     costs the spread + the loss.
"""


_PAPER_SOAK_ENCOURAGEMENT = """

10. **PAPER-SOAK MODE — propose more, skip less**:

    The bot is in paper mode (ACT_REAL_CAPITAL_ENABLED is unset).
    Per operator directive, the GOAL of paper mode is to gather
    soak data for the readiness gate (500 trades / 14 days). Each
    SKIP costs the soak counter a tick. If you have ANY honest read
    on direction -- even MODERATE conviction (0.55+) -- propose
    LONG with conservative size (1-2% of equity). The post-trade
    self-critique will teach the brain over time; it cannot learn
    from skips.

    DECISION RULE in paper mode:
    - If 1h or 4h timeframe AGREES with bullish bias OR mean-reversion
      setup is forming after a drop -> propose LONG.
    - If both 1h AND 4h are bearish on a Robinhood (longs-only) venue
      -> still consider mean-reversion LONG bounce on oversold,
      OR SHORT (the executor will route to a perp venue when wired;
      the brain's job is to FORM the view).
    - SKIP only when: authority violation, no signal at all (truly
      flat), or fresh news blackout.

    Real capital path (ACT_REAL_CAPITAL_ENABLED=1) reverts to the
    strict rules above (rule 9). This relaxation is paper-only.
"""


def _render_system_prompt_base() -> str:
    """Render SYSTEM_PROMPT_BASE with runtime spread values.

    Computed thresholds:
      * spread_pct        — live cost_gate.robinhood preset
      * min_move_pct      — 1.5 x spread (covers spread + 50% buffer)

    In paper mode, append a soak-encouragement block telling the
    brain to PROPOSE more (not less) so the readiness-gate counter
    actually moves. The strict rule-9 conservatism stays for real
    capital path.
    """
    import os as _os_pc
    spread_pct = _runtime_spread_pct()
    min_move_pct = max(2.0, spread_pct * 1.5)
    base = _SYSTEM_PROMPT_BASE_TEMPLATE.format(
        spread_pct=spread_pct,
        min_move_pct=min_move_pct,
    )
    is_real_capital = _os_pc.environ.get(
        "ACT_REAL_CAPITAL_ENABLED", ""
    ).strip() == "1"
    if not is_real_capital:
        base = base.rstrip() + _PAPER_SOAK_ENCOURAGEMENT
    return base


# Back-compat: keep SYSTEM_PROMPT_BASE module-level for any external
# importers, but render it lazily once on first access.
SYSTEM_PROMPT_BASE = _render_system_prompt_base()


TASK_PROMPTS = {
    'trade_analysis': """## TASK: Analyze current market state and recommend action.

{quant_data}

{context}

## REQUIRED OUTPUT (JSON):
{{
  "market_regime": "STRING (one of: {allowed_regimes})",
  "action": "STRING (one of: {allowed_actions})",
  "confidence_score": INT (0-100),
  "reasoning_trace": "STRING (2-3 sentences citing specific data values)",
  "macro_bias": FLOAT (-0.5 to 0.5),
  "suggested_config_update": {{}},
  "risk_assessment": "STRING (cite VaR, CVaR, or risk score from data)",
  "key_signals": ["LIST of 3 most important signals from the data"]
}}

REMEMBER: Only reference numbers from VERIFIED QUANT DATA. Do NOT compute new values.""",

    'performance_review': """## TASK: Review recent trading performance and suggest improvements.

{quant_data}

### TRADE HISTORY:
{context}

## REQUIRED OUTPUT (JSON):
{{
  "market_regime": "STRING (one of: {allowed_regimes})",
  "performance_assessment": "STRING (cite specific P&L numbers from trade history)",
  "reasoning_trace": "STRING (what went right/wrong, citing data)",
  "confidence_score": INT (0-100),
  "suggested_config_update": {{}},
  "macro_bias": FLOAT (-0.5 to 0.5),
  "improvement_actions": ["LIST of 3 specific actions"]
}}

REMEMBER: Only cite numbers present in the data. Do NOT fabricate statistics.""",

    'per_trade_reasoning': """## TASK: Explain WHY this specific trade should be opened.

{quant_data}

### TRADE DECISION:
{context}

## REQUIRED OUTPUT (JSON):
{{
  "entry_reason": "STRING (2-3 sentences citing specific indicator values)",
  "signal_alignment": "STRING (which signals agree/disagree)",
  "risk_factors": ["LIST of 2-3 risks, citing data values"],
  "expected_outcome": "STRING (based on regime + trend data)",
  "confidence": INT (0-100)
}}

REMEMBER: Every cited number must come from VERIFIED QUANT DATA above.""",
}


# ─────────────────────────────────────────────────────────────
# Prompt Constraint Engine
# ─────────────────────────────────────────────────────────────

class PromptConstraintEngine:
    """
    Builds constrained prompts and validates LLM responses.
    Applied automatically to every LLM call in the trading system.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.custom_system_prompt = self.config.get('system_prompt', '')
        self.strict_mode = self.config.get('strict_mode', True)
        # Lean mode = only inject authority directives + JSON schema.
        # Use after LoRA fine-tune has internalized strategy/style so we don't
        # duplicate knowledge the weights already carry.
        # Enable via config: {'lean_prompt_mode': True} or env LLM_LEAN_PROMPT=1
        import os
        self.lean_prompt_mode = bool(
            self.config.get('lean_prompt_mode', False)
            or os.getenv('LLM_LEAN_PROMPT', '').lower() in ('1', 'true', 'yes')
        )

    def get_system_prompt(self, task: str = 'trade_analysis') -> str:
        """
        Get the immutable system prompt for a given task.

        AUTHORITY_SYSTEM_PROMPT is always prepended — it is non-negotiable and
        takes precedence over any other guidance. Custom prompts append.

        Lean mode drops SYSTEM_PROMPT_BASE (strategy explanations, style rules,
        examples) since LoRA fine-tuning should have internalized those. The
        authority directives remain because hard constraints must stay in
        prompt form for audit + deterministic enforcement.
        """
        if self.lean_prompt_mode:
            # Minimal: authority rules + a terse JSON-only reminder.
            prompt = (
                AUTHORITY_SYSTEM_PROMPT
                + "\n\n## OUTPUT: Return ONLY a JSON object matching the task schema. "
                + "Cite every number as [METRIC=VALUE] from the VERIFIED QUANT DATA block."
            )
        else:
            # Full: authority directives first, then base strategy/safety
            # content. Re-render on every call so an operator
            # `setx ACT_ROBINHOOD_SPREAD_PCT 1.0` propagates into the LLM
            # prompt at the next tick — no module reload required.
            prompt = AUTHORITY_SYSTEM_PROMPT + "\n\n" + _render_system_prompt_base()

        if self.custom_system_prompt:
            prompt += f"\n\n## ADDITIONAL INSTRUCTIONS:\n{self.custom_system_prompt}"

        return prompt

    def build_prompt(self, task: str, quant_data: str,
                     context: str = '', extra: str = '') -> str:
        """
        Build a fully constrained prompt for a given task.

        Args:
            task: Task type ('trade_analysis', 'performance_review', 'per_trade_reasoning')
            quant_data: Pre-formatted quant data block from MathInjector
            context: Additional context (trade history, specific trade details, etc.)
            extra: Any extra instructions (appended after safety rules)

        Returns:
            Complete constrained prompt ready for LLM
        """
        template = TASK_PROMPTS.get(task, TASK_PROMPTS['trade_analysis'])

        prompt = template.format(
            quant_data=quant_data,
            context=context or 'No additional context.',
            allowed_regimes=', '.join(ALLOWED_REGIMES),
            allowed_actions=', '.join(ALLOWED_ACTIONS),
        )

        if extra:
            prompt += f"\n\n## ADDITIONAL CONTEXT:\n{extra}"

        return prompt

    def validate_response(self, response: Dict) -> Tuple[Dict, List[str]]:
        """
        Validate and sanitize LLM response against safety rules.

        Returns:
            (sanitized_response, list_of_violations)
        """
        violations = []
        sanitized = response.copy()

        # 1. Validate regime
        regime = sanitized.get('market_regime', '')
        if regime and regime.upper() not in ALLOWED_REGIMES:
            violations.append(f"Invalid regime '{regime}'. Defaulting to UNKNOWN.")
            sanitized['market_regime'] = 'UNKNOWN'

        # 2. Validate action
        action = sanitized.get('action', '')
        if action and action.upper() not in ALLOWED_ACTIONS:
            violations.append(f"Invalid action '{action}'. Defaulting to FLAT.")
            sanitized['action'] = 'FLAT'

        # 3. Validate confidence score
        conf = sanitized.get('confidence_score', 50)
        if isinstance(conf, (int, float)):
            if conf < 0 or conf > 100:
                violations.append(f"Confidence {conf} out of [0,100]. Clamping.")
                sanitized['confidence_score'] = max(0, min(100, int(conf)))
        else:
            violations.append(f"Invalid confidence type: {type(conf)}. Defaulting to 0.")
            sanitized['confidence_score'] = 0

        # 4. Validate macro_bias
        bias = sanitized.get('macro_bias', 0.0)
        if isinstance(bias, (int, float)):
            if bias < -0.5 or bias > 0.5:
                violations.append(f"macro_bias {bias} out of [-0.5, 0.5]. Clamping.")
                sanitized['macro_bias'] = max(-0.5, min(0.5, float(bias)))
        else:
            sanitized['macro_bias'] = 0.0

        # 5. Validate suggested config updates
        config_update = sanitized.get('suggested_config_update', {})
        if isinstance(config_update, dict):
            sanitized_config = {}
            for section, params in config_update.items():
                if section in ALLOWED_CONFIG_RANGES and isinstance(params, dict):
                    sanitized_section = {}
                    for key, value in params.items():
                        if key in ALLOWED_CONFIG_RANGES[section]:
                            min_val, max_val = ALLOWED_CONFIG_RANGES[section][key]
                            if isinstance(value, (int, float)):
                                clamped = max(min_val, min(max_val, float(value)))
                                if clamped != value:
                                    violations.append(
                                        f"Config {section}.{key}={value} clamped to [{min_val}, {max_val}]"
                                    )
                                sanitized_section[key] = clamped
                            else:
                                violations.append(f"Config {section}.{key} has invalid type. Skipped.")
                        else:
                            violations.append(f"Config key {section}.{key} not in whitelist. Removed.")
                    if sanitized_section:
                        sanitized_config[section] = sanitized_section
                else:
                    violations.append(f"Config section '{section}' not allowed. Removed.")
            sanitized['suggested_config_update'] = sanitized_config
        else:
            sanitized['suggested_config_update'] = {}

        # 6. Check for hallucinated numbers in reasoning
        reasoning = sanitized.get('reasoning_trace', '')
        if self.strict_mode and reasoning:
            # Flag if reasoning contains numbers that aren't cited with [METRIC=VALUE]
            import re
            numbers_in_text = re.findall(r'(?<!\[)\b\d+\.?\d*%?\b(?!\])', reasoning)
            # Allow common numbers (0, 1, 2, etc.) and percentages that look like citations
            suspicious = [n for n in numbers_in_text
                          if float(n.replace('%', '')) > 10 and '[' not in reasoning[max(0, reasoning.find(n)-20):reasoning.find(n)]]
            if len(suspicious) > 3:
                violations.append(
                    f"Reasoning may contain uncited numbers: {suspicious[:5]}. "
                    f"LLM should cite [METRIC=VALUE] for all data."
                )

        if violations:
            logger.warning(f"LLM response violations: {violations}")
            sanitized['_violations'] = violations

        return sanitized, violations

    def build_full_pipeline(self, task: str, quant_data: str,
                            context: str = '') -> Tuple[str, str]:
        """
        Build both system prompt and user prompt for a complete LLM call.

        Returns:
            (system_prompt, user_prompt)
        """
        system_prompt = self.get_system_prompt(task)
        user_prompt = self.build_prompt(task, quant_data, context)
        return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────
# Integrated LLM Call (Math Injection + Constraints + Provider)
# ─────────────────────────────────────────────────────────────

class ConstrainedLLMAnalyst:
    """
    Complete pipeline: Raw Data → Math Injection → Prompt Constraints → LLM → Validation.
    This is the ONLY way the trading system should call LLMs.
    """

    def __init__(self, llm_router=None, config: Optional[Dict] = None):
        from src.ai.math_injection import MathInjector
        self.math_injector = MathInjector(config)
        self.constraints = PromptConstraintEngine(config)
        self.llm_router = llm_router  # LLMRouter instance
        self._fallback_enabled = True

    def analyze_market(self,
                       prices, highs, lows, volumes,
                       sentiment_score: float = 0.0,
                       asset: str = 'BTCUSDT',
                       account_balance: float = 10000.0,
                       trade_history: str = '',
                       fallback_chain: Optional[List[str]] = None,
                       ) -> Dict:
        """
        Full constrained market analysis pipeline.

        1. MathInjector computes all quant features from raw data
        2. PromptConstraints builds safe prompt with computed data
        3. LLMRouter sends to best available LLM
        4. Response is validated against safety rules
        5. Any violations are logged and corrected
        """
        import numpy as np

        # Step 1: Compute quant data
        state = self.math_injector.compute_full_state(
            np.asarray(prices), np.asarray(highs),
            np.asarray(lows), np.asarray(volumes),
            sentiment_score, asset, account_balance
        )
        quant_block = self.math_injector.format_for_prompt(state)

        # Step 2: Build constrained prompt
        system_prompt, user_prompt = self.constraints.build_full_pipeline(
            task='trade_analysis',
            quant_data=quant_block,
            context=trade_history,
        )

        # Step 3: Query LLM
        if self.llm_router:
            raw_response = self.llm_router.query(
                user_prompt, system_prompt=system_prompt,
                fallback_chain=fallback_chain
            )
        else:
            raw_response = self._rule_based_fallback(state)

        # Step 4: Validate response
        validated, violations = self.constraints.validate_response(raw_response)

        # Step 5: Attach computed state for downstream use
        validated['_quant_state'] = state
        validated['_violations_count'] = len(violations)

        return validated

    def explain_trade(self,
                      prices, highs, lows, volumes,
                      asset: str, direction: int, entry_price: float,
                      l1_info: Dict = None, l2_info: Dict = None,
                      l3_info: Dict = None, market_info: Dict = None,
                      fallback_chain: Optional[List[str]] = None,
                      ) -> Dict:
        """Generate constrained per-trade reasoning."""
        import numpy as np

        state = self.math_injector.compute_full_state(
            np.asarray(prices), np.asarray(highs),
            np.asarray(lows), np.asarray(volumes),
            asset=asset
        )
        quant_block = self.math_injector.format_for_prompt(state)

        context = f"""
TRADE: {'LONG' if direction > 0 else 'SHORT'} {asset} @ ${entry_price:,.2f}
L1 (LightGBM): {json.dumps(l1_info or {}, default=str)}
L2 (Sentiment): {json.dumps(l2_info or {}, default=str)}
L3 (Risk): {json.dumps(l3_info or {}, default=str)}
Market: {json.dumps(market_info or {}, default=str)}
"""

        system_prompt, user_prompt = self.constraints.build_full_pipeline(
            task='per_trade_reasoning',
            quant_data=quant_block,
            context=context,
        )

        if self.llm_router:
            raw_response = self.llm_router.query(
                user_prompt, system_prompt=system_prompt,
                fallback_chain=fallback_chain
            )
        else:
            raw_response = {
                'entry_reason': f"Rule-based: {'LONG' if direction > 0 else 'SHORT'} signal with price at {entry_price}",
                'signal_alignment': 'LLM unavailable',
                'risk_factors': ['LLM provider not configured'],
                'expected_outcome': 'Unknown',
                'confidence': 50,
            }

        validated, _ = self.constraints.validate_response(raw_response)
        return validated

    def _rule_based_fallback(self, state: Dict) -> Dict:
        """Rule-based analysis when no LLM is available."""
        trend = state.get('trend', {})
        vol = state.get('volatility', {})
        hurst = state.get('hurst', {})
        mc = state.get('monte_carlo_risk', {})
        sentiment = state.get('sentiment', {})

        # Determine regime
        regime = 'UNKNOWN'
        hmm = state.get('hmm_regime', {})
        if hmm:
            regime = hmm.get('current_regime', 'UNKNOWN').upper()
        elif hurst:
            regime = hurst.get('regime', 'random').upper()

        # Determine action
        action = 'FLAT'
        rsi = trend.get('rsi_14', 50)
        macd_sig = trend.get('macd_signal', 'NEUTRAL')
        risk_level = mc.get('risk_level', 'MEDIUM')

        if risk_level == 'HIGH':
            action = 'FLAT'
        elif rsi < 30 and macd_sig == 'BULLISH':
            action = 'LONG'
        elif rsi > 70 and macd_sig == 'BEARISH':
            action = 'SHORT'

        return {
            'market_regime': regime,
            'action': action,
            'confidence_score': 50,
            'reasoning_trace': f'Rule-based: RSI={rsi:.0f}, MACD={macd_sig}, Risk={risk_level}',
            'macro_bias': 0.0,
            'suggested_config_update': {},
            'risk_assessment': f'MC Risk Score={mc.get("risk_score", "N/A")}',
            'key_signals': [
                f'RSI={rsi:.0f} ({trend.get("rsi_zone", "NEUTRAL")})',
                f'Trend={trend.get("trend_direction", "N/A")} (ADX={trend.get("adx", "N/A"):.0f})',
                f'Sentiment={sentiment.get("sentiment_zone", "NEUTRAL")}',
            ],
        }
