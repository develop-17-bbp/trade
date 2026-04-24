"""Tests for the agentic-loop JSON parser (C23).

The parser is the critical chokepoint: when LLMs emit JSON in
non-canonical forms, plans fail to compile and zero trades fire.
These tests lock in tolerance for the specific failure modes
observed in operator logs."""

from __future__ import annotations

from src.ai.agentic_trade_loop import _extract_json


def test_parses_raw_json():
    obj = _extract_json('{"plan": {"direction": "LONG", "size_pct": 1.5}}')
    assert obj == {"plan": {"direction": "LONG", "size_pct": 1.5}}


def test_parses_fenced_json_block():
    text = "Here's my plan:\n```json\n{\"plan\": {\"direction\": \"LONG\"}}\n```\nDone."
    obj = _extract_json(text)
    assert obj is not None
    assert obj["plan"]["direction"] == "LONG"


def test_parses_fenced_block_no_json_tag():
    text = "```\n{\"skip\": \"weak setup\"}\n```"
    obj = _extract_json(text)
    assert obj == {"skip": "weak setup"}


def test_strips_deepseek_think_prefix():
    text = ("<think>Let me consider the market...I should skip.</think>\n"
            '{"skip": "thought about it"}')
    obj = _extract_json(text)
    assert obj == {"skip": "thought about it"}


def test_strips_unclosed_think_prefix():
    text = "<think>I'm still thinking when response got cut off"
    obj = _extract_json(text)
    assert obj is None  # nothing after the think — no JSON


def test_tolerates_trailing_comma():
    text = '{"skip": "done", "confidence": 0.7,}'
    obj = _extract_json(text)
    assert obj is not None
    assert obj.get("skip") == "done"


def test_tolerates_js_line_comments():
    text = '{"skip": "done" // because reasons\n}'
    obj = _extract_json(text)
    assert obj is not None
    assert obj.get("skip") == "done"


def test_tolerates_js_block_comments():
    text = '{/* header comment */ "skip": "done"}'
    obj = _extract_json(text)
    assert obj is not None
    assert obj.get("skip") == "done"


def test_prefers_larger_balanced_candidate():
    """When LLM shows an example `{}` and then a real plan, real plan wins."""
    text = ('Example: {"name": "x"}. My actual plan: '
            '{"plan": {"direction": "LONG", "size_pct": 2.0, "tier": "normal"}}')
    obj = _extract_json(text)
    assert obj is not None
    # Real plan is larger — should win
    assert "plan" in obj


def test_handles_prose_prefix_and_suffix():
    text = ("I have analyzed the market and reached this conclusion: "
            '{"plan": {"direction": "LONG"}} '
            "This is my final answer.")
    obj = _extract_json(text)
    assert obj is not None
    assert obj["plan"]["direction"] == "LONG"


def test_handles_nested_braces_in_strings():
    text = '{"skip": "the scanner said \\"regime={A}\\"", "confidence": 0.5}'
    obj = _extract_json(text)
    assert obj is not None
    assert obj.get("confidence") == 0.5


def test_empty_or_none_returns_none():
    assert _extract_json("") is None
    assert _extract_json(None) is None   # type: ignore[arg-type]


def test_garbage_returns_none():
    assert _extract_json("this is not json and has no braces") is None


def test_handles_fence_with_leading_language_tag_only():
    text = "```json\n{\"skip\": \"weak setup\"}\n```"
    obj = _extract_json(text)
    assert obj == {"skip": "weak setup"}


def test_python_dict_style_single_quotes():
    """LLMs trained on Python (devstral, qwen-coder) sometimes emit
    Python-dict syntax. ast.literal_eval must rescue these."""
    text = "{'opportunity_score': 65, 'proposed_direction': 'LONG', 'rationale': 'momentum breakout'}"
    obj = _extract_json(text)
    assert obj is not None
    assert obj["opportunity_score"] == 65
    assert obj["proposed_direction"] == "LONG"


def test_mixed_quotes_with_think_prefix():
    text = ("<think>scanning...</think>"
            "{'opportunity_score': 40, 'top_signals': ['ema_cross'], "
            "'proposed_direction': 'FLAT', 'rationale': 'weak'}")
    obj = _extract_json(text)
    assert obj is not None
    assert obj["opportunity_score"] == 40
    assert obj["proposed_direction"] == "FLAT"


def test_ast_repair_rejects_code_injection():
    """ast.literal_eval must NOT execute arbitrary code."""
    text = "{'key': __import__('os').system('echo pwn')}"
    obj = _extract_json(text)
    # Should safely fail (literal_eval refuses function calls), not crash.
    assert obj is None


def test_handles_multiline_json_with_comments_and_trailing_comma():
    """The all-at-once stress test — everything a real LLM throws at us."""
    text = """<think>
    The market is ranging. I'll skip.
    </think>
    ```json
    {
      // reasoning
      "skip": "low conviction", /* block */
      "confidence": 0.2,
    }
    ```
    """
    obj = _extract_json(text)
    assert obj is not None
    assert obj["skip"] == "low conviction"
    assert obj["confidence"] == 0.2
