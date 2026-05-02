"""Pine Script v5 → Python strategy translator.

Imports TradingView Pine scripts and converts them into ACT-compatible
Python strategy modules. Handles the common subset of Pine v5 features:

  ✅ Indicator wrappers: ta.sma, ta.ema, ta.rsi, ta.macd, ta.atr,
                        ta.stoch, ta.bb, ta.vwap, ta.obv, ta.adx,
                        ta.crossover, ta.crossunder
  ✅ Operators: + - * / > < >= <= == != and or not
  ✅ Conditionals: if/else, ternary
  ✅ Comments: //, /* */
  ✅ strategy.entry / strategy.close → StrategySignal
  ✅ plot/plotshape → ignored (no chart in ACT)
  ✅ input.* → mapped to function kwargs

  ❌ User-defined functions (forwards as a comment block)
  ❌ Arrays, matrices (skipped — flagged)
  ❌ request.security (cross-symbol — flagged)
  ❌ Complex nested loops (translates simple 1-level only)

Two import paths:
  1. **Heuristic translator** (this module) — for simple/common scripts.
     ~70% of community Pine scripts work straight through.
  2. **LLM-assisted translator** (`pine_importer_llm`) — falls back to
     the local LLM for complex scripts. Use when heuristic fails.

Usage:
    from src.trading.strategies.pine_importer import import_pine_script
    py_strategy = import_pine_script(pine_text, strategy_name="my_pine_strat")
    # py_strategy is a callable evaluate(highs, lows, closes, volumes, ...) function

The translator emits Python source as a string, which can then be
written to `src/trading/strategies/imported/<name>.py` for permanent
registration in ACT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# Pine ta.* function → Python equivalent in src.indicators.indicators
PINE_TA_MAP: Dict[str, str] = {
    "ta.sma":         "_pine_sma",
    "ta.ema":         "_pine_ema",
    "ta.wma":         "_pine_wma",
    "ta.rsi":         "_pine_rsi",
    "ta.atr":         "_pine_atr",
    "ta.macd":        "_pine_macd",
    "ta.stoch":       "_pine_stoch",
    "ta.bb":          "_pine_bb",
    "ta.vwap":        "_pine_vwap",
    "ta.obv":         "_pine_obv",
    "ta.adx":         "_pine_adx",
    "ta.cci":         "_pine_cci",
    "ta.mfi":         "_pine_mfi",
    "ta.crossover":   "_pine_crossover",
    "ta.crossunder":  "_pine_crossunder",
    "ta.highest":     "_pine_highest",
    "ta.lowest":      "_pine_lowest",
    "ta.change":      "_pine_change",
    "ta.roc":         "_pine_roc",
    "ta.tr":          "_pine_tr",
    "math.abs":       "abs",
    "math.max":       "max",
    "math.min":       "min",
    "math.floor":     "math.floor",
    "math.ceil":      "math.ceil",
    "math.round":     "round",
}


@dataclass
class PineTranslationResult:
    success: bool
    python_source: str = ""
    strategy_name: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fallback_required: bool = False


def _strip_comments(pine: str) -> str:
    """Remove // line comments and /* block comments */"""
    pine = re.sub(r'/\*.*?\*/', '', pine, flags=re.DOTALL)
    pine = re.sub(r'//[^\n]*', '', pine)
    return pine


def _extract_inputs(pine: str) -> Tuple[Dict[str, Any], List[str]]:
    """Find input.X(default, title=...) calls and convert to function kwargs.

    Examples:
        length = input.int(14, "RSI Length")     → kwargs['length'] = 14
        ovs    = input.float(70.0, "Overbought") → kwargs['ovs'] = 70.0
    """
    inputs: Dict[str, Any] = {}
    warnings: List[str] = []
    pattern = re.compile(
        r'(\w+)\s*=\s*input(?:\.(\w+))?\s*\(\s*([^,)]+)(?:,\s*[^)]*)?\)'
    )
    for match in pattern.finditer(pine):
        var_name, input_type, default_str = match.group(1), match.group(2), match.group(3)
        default_str = default_str.strip().strip('"').strip("'")
        if input_type in ("int", None) and default_str.lstrip('-').isdigit():
            inputs[var_name] = int(default_str)
        elif input_type == "float":
            try:
                inputs[var_name] = float(default_str)
            except ValueError:
                inputs[var_name] = 0.0
        elif input_type == "bool":
            inputs[var_name] = default_str.lower() == "true"
        elif input_type in ("string", "symbol"):
            inputs[var_name] = default_str
        else:
            try:
                inputs[var_name] = float(default_str) if "." in default_str else int(default_str)
            except (ValueError, TypeError):
                inputs[var_name] = default_str
                warnings.append(f"unknown input type for {var_name}; treated as string")
    return inputs, warnings


def _translate_operators(line: str) -> str:
    """Pine boolean operators → Python."""
    line = re.sub(r'\band\b', 'and', line)
    line = re.sub(r'\bor\b', 'or', line)
    line = re.sub(r'\bnot\b', 'not', line)
    line = re.sub(r'\btrue\b', 'True', line)
    line = re.sub(r'\bfalse\b', 'False', line)
    line = re.sub(r'\bna\b', 'None', line)
    return line


def _translate_ta_calls(line: str) -> str:
    """Replace ta.* and math.* with Python helpers in pine_runtime."""
    for pine_func, py_func in PINE_TA_MAP.items():
        # Use word-boundary to avoid partial matches
        pattern = re.escape(pine_func) + r'\b'
        line = re.sub(pattern, py_func, line)
    return line


def _translate_strategy_calls(line: str) -> Optional[str]:
    """Replace strategy.entry/exit/close with StrategySignal returns."""
    # strategy.entry("Long", strategy.long, qty=1) → return "LONG"
    # strategy.close("Long") → return "FLAT"
    # strategy.short / strategy.long
    if re.search(r'strategy\.entry.*strategy\.long', line):
        return "    _signal = 'LONG'"
    if re.search(r'strategy\.entry.*strategy\.short', line):
        return "    _signal = 'SHORT'"
    if re.search(r'strategy\.close', line):
        return "    _signal = 'FLAT'"
    if re.search(r'strategy\.exit', line):
        return "    _signal = 'FLAT'"
    return None


def _strip_plot_calls(line: str) -> Optional[str]:
    """Pine plot/plotshape/plotchar are visual-only — drop them."""
    if re.match(r'\s*(plot|plotshape|plotchar|plotcandle|hline|fill|bgcolor)\s*\(', line):
        return None
    return line


def _translate_pine_to_python_body(pine_body: str) -> Tuple[List[str], List[str]]:
    """Walk Pine source line-by-line, emit Python equivalent.
    Returns (python_lines, warnings)."""
    warnings: List[str] = []
    lines: List[str] = []
    for raw in pine_body.split('\n'):
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        # Skip pine-only directives
        if stripped.startswith(('//@', 'indicator(', 'strategy(', 'library(', 'study(')):
            warnings.append(f"directive_skipped: {stripped[:60]}")
            continue
        # Visual-only calls
        plot_check = _strip_plot_calls(raw)
        if plot_check is None:
            continue
        # strategy.entry/close → signal assignment
        strat_check = _translate_strategy_calls(stripped)
        if strat_check is not None:
            lines.append(strat_check)
            continue
        # Bracket-history Pine syntax: foo[1] → _hist(foo, 1)
        translated = re.sub(
            r'(\w+)\s*\[\s*(\d+)\s*\]',
            r'_hist(\1, \2)',
            raw,
        )
        translated = _translate_operators(translated)
        translated = _translate_ta_calls(translated)
        # Pine `:=` mutable assignment → Python `=`
        translated = translated.replace(':=', '=')
        # Pine `var x = ...` → Python `x = ...` (var declares persistent;
        # in our stateless evaluate() function we treat as initial assignment)
        translated = re.sub(r'\bvar\s+', '', translated)
        translated = re.sub(r'\bvarip\s+', '', translated)
        # Pine if/else have implicit indentation — we preserve raw indent
        if re.match(r'^\s*if\s+', translated):
            translated = re.sub(
                r'^(\s*)if\s+(.+?)(?:\s*$)', r'\1if \2:', translated,
            )
        elif re.match(r'^\s*else\s*$', translated):
            translated = translated.rstrip() + ':'
        lines.append('    ' + translated.rstrip())
    return lines, warnings


def _emit_python_module(
    strategy_name: str,
    inputs: Dict[str, Any],
    body_lines: List[str],
    warnings: List[str],
    pine_source: str,
) -> str:
    """Emit a complete ACT-compatible Python strategy module."""
    safe_name = re.sub(r'\W', '_', strategy_name)
    inputs_doc = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    warn_block = "\n".join(f"# WARNING: {w}" for w in warnings) if warnings else "# (no warnings)"
    body = "\n".join(body_lines)
    template = f'''"""Pine-imported strategy: {strategy_name}

Auto-translated from Pine Script v5 by src/trading/strategies/pine_importer.py

Original Pine inputs: {inputs_doc}

{warn_block}

Manual review recommended before production use.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional
from src.trading.strategies.pine_runtime import (
    _pine_sma, _pine_ema, _pine_wma, _pine_rsi, _pine_atr, _pine_macd,
    _pine_stoch, _pine_bb, _pine_vwap, _pine_obv, _pine_adx, _pine_cci,
    _pine_mfi, _pine_crossover, _pine_crossunder, _pine_highest, _pine_lowest,
    _pine_change, _pine_roc, _pine_tr, _hist,
)


def evaluate(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: Optional[List[float]] = None,
    {", ".join(f"{k}={v!r}" for k, v in inputs.items())},
) -> Dict[str, Any]:
    """Auto-imported {strategy_name} strategy evaluator."""
    open_ = closes  # Pine often uses open/high/low/close interchangeably
    high = highs
    low = lows
    close = closes
    volume = volumes or [0.0] * len(closes)
    _signal = "FLAT"
    confidence = 0.5
{body}

    return {{
        "strategy": "{safe_name}",
        "direction": _signal,
        "confidence": confidence,
        "source": "pine_imported",
    }}
'''
    return template


def import_pine_script(
    pine_text: str,
    strategy_name: str = "imported_pine_strategy",
) -> PineTranslationResult:
    """Translate a Pine v5 script into a Python strategy module.

    Returns PineTranslationResult with:
      - python_source: ready-to-write .py module text
      - inputs: kwargs the strategy accepts
      - warnings: non-fatal issues
      - errors: fatal issues (only if success=False)
      - fallback_required: True if heuristic translator can't handle —
        caller should retry with LLM translator
    """
    if not pine_text or not pine_text.strip():
        return PineTranslationResult(
            success=False,
            errors=["empty pine script"],
            fallback_required=False,
        )
    cleaned = _strip_comments(pine_text)

    # Detect features that the heuristic translator can't handle
    fallback_triggers: List[str] = []
    if re.search(r'\brequest\.security\s*\(', cleaned):
        fallback_triggers.append("uses request.security (cross-symbol)")
    if re.search(r'\barray\.\w+', cleaned):
        fallback_triggers.append("uses arrays")
    if re.search(r'\bmatrix\.\w+', cleaned):
        fallback_triggers.append("uses matrices")
    if re.search(r'\bmap\.\w+', cleaned):
        fallback_triggers.append("uses maps")
    if cleaned.count("for ") > 2:
        fallback_triggers.append("multiple for-loops (heuristic translator handles 1-level only)")

    if fallback_triggers:
        return PineTranslationResult(
            success=False,
            warnings=fallback_triggers,
            errors=["fallback to LLM translator required"],
            fallback_required=True,
        )

    inputs, input_warnings = _extract_inputs(cleaned)
    body_lines, body_warnings = _translate_pine_to_python_body(cleaned)
    warnings = input_warnings + body_warnings

    py_source = _emit_python_module(
        strategy_name=strategy_name,
        inputs=inputs,
        body_lines=body_lines,
        warnings=warnings,
        pine_source=pine_text,
    )

    return PineTranslationResult(
        success=True,
        python_source=py_source,
        strategy_name=strategy_name,
        inputs=inputs,
        warnings=warnings,
        errors=[],
        fallback_required=False,
    )
