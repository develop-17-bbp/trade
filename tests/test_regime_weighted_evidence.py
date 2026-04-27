"""Tests for FS-ReasoningAgent-inspired regime-weighted evidence
(arXiv:2410.12464). The Analyst prompt instructs regime-conditional
weighting; EvidenceSection now carries a `kind` tag for audit."""

from __future__ import annotations

import pytest


# ── Analyst prompt regime-aware block ──────────────────────────────────


def test_analyst_prompt_contains_regime_weighting_block():
    from src.ai.dual_brain import ANALYST_SYSTEM
    assert "REGIME-AWARE" in ANALYST_SYSTEM
    assert "FS-ReasoningAgent" in ANALYST_SYSTEM
    assert "arXiv:2410.12464" in ANALYST_SYSTEM


def test_analyst_prompt_specifies_bull_60_subjective():
    from src.ai.dual_brain import ANALYST_SYSTEM
    # Bull regime should weight subjective evidence 60%
    assert "BULL" in ANALYST_SYSTEM
    assert "SUBJECTIVE evidence 60%" in ANALYST_SYSTEM


def test_analyst_prompt_specifies_bear_60_factual():
    from src.ai.dual_brain import ANALYST_SYSTEM
    assert "BEAR" in ANALYST_SYSTEM
    assert "FACTUAL evidence 60%" in ANALYST_SYSTEM


def test_analyst_prompt_calls_macro_bias_first():
    from src.ai.dual_brain import ANALYST_SYSTEM
    # Operator must call get_macro_bias before weighting evidence
    assert "get_macro_bias" in ANALYST_SYSTEM
    assert "FIRST" in ANALYST_SYSTEM


def test_analyst_prompt_demands_kind_tag_on_evidence():
    from src.ai.dual_brain import ANALYST_SYSTEM
    # supporting_evidence items should carry kind: factual|subjective|mixed|technical
    assert "kind" in ANALYST_SYSTEM
    for tag in ("factual", "subjective", "technical", "mixed"):
        assert tag in ANALYST_SYSTEM, f"missing evidence kind tag: {tag}"


# ── EvidenceSection kind tag ────────────────────────────────────────────


def test_evidence_section_default_kind_mixed():
    from src.ai.context_builders import EvidenceSection
    s = EvidenceSection(name="X", content="body")
    assert s.kind == "mixed"


def test_evidence_section_kind_round_trips_in_dict():
    from src.ai.context_builders import EvidenceSection
    s = EvidenceSection(name="NEWS", content="b", kind="subjective")
    assert s.to_dict()["kind"] == "subjective"


def test_evidence_section_prompt_block_shows_kind_when_not_mixed():
    from src.ai.context_builders import EvidenceSection
    s = EvidenceSection(name="N", content="body", kind="factual")
    blk = s.to_prompt_block()
    assert "kind=factual" in blk


def test_evidence_section_prompt_block_hides_mixed_kind():
    from src.ai.context_builders import EvidenceSection
    # mixed is the default — no need to clutter the prompt
    s = EvidenceSection(name="N", content="body", kind="mixed",
                        confidence=1.0, age_s=0.0)
    blk = s.to_prompt_block()
    assert "kind=" not in blk


def test_build_evidence_document_assigns_correct_kinds():
    """build_evidence_document tags each section with the correct
    factual/subjective/technical kind so the analyst can apply its
    regime-weighted rules."""
    from src.ai.context_builders import build_evidence_document, clear_cache

    clear_cache()
    # Stub each fetcher to return predictable content
    import src.ai.context_builders as cb
    cb._scanner_block = lambda asset: "scanner content"
    cb._traces_block = lambda asset, limit=3: "traces content"
    cb._news_block = lambda asset, hours=12: "news content"
    cb._fear_greed_block = lambda: "fg content"
    cb._graph_block = lambda asset: "graph content"
    cb._body_controls_block = lambda: "body content"

    doc = build_evidence_document("BTC")
    by_name = {s.name: s for s in doc.sections}

    assert by_name["SCANNER_REPORT"].kind == "mixed"
    assert by_name["RECENT_ANALYST_TRACES"].kind == "technical"
    assert by_name["NEWS"].kind == "mixed"
    assert by_name["FEAR_GREED"].kind == "subjective"
    assert by_name["KNOWLEDGE_GRAPH"].kind == "mixed"
    assert by_name["BODY_CONTROLS"].kind == "technical"
