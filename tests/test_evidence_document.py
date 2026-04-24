"""Tests for EvidenceDocument / EvidenceSection (C19)."""

from __future__ import annotations

import pytest

from src.ai.context_builders import (
    EvidenceDocument,
    EvidenceSection,
    build_evidence_document,
    build_analyst_context,
    clear_cache,
)


def test_section_to_prompt_block_includes_tags():
    s = EvidenceSection(
        name="NEWS", content="BTC ETF inflow surge",
        confidence=0.7, age_s=45.0, source="rss",
    )
    blk = s.to_prompt_block()
    assert "## NEWS" in blk
    assert "conf=0.70" in blk
    assert "age=45s" in blk
    assert "BTC ETF inflow surge" in blk


def test_section_no_tags_when_perfect_confidence_and_fresh():
    s = EvidenceSection(name="SCANNER", content="hot setup", confidence=1.0, age_s=0.0)
    blk = s.to_prompt_block()
    assert "##" in blk
    # No confidence tag when conf=1.0 and age=0
    assert "conf=" not in blk
    assert "age=" not in blk


def test_document_add_skips_empty_sections():
    doc = EvidenceDocument(asset="BTC")
    doc.add(EvidenceSection(name="NEWS", content="", confidence=0.5))
    doc.add(EvidenceSection(name="SCANNER", content="  ", confidence=0.5))
    doc.add(None)
    doc.add(EvidenceSection(name="GRAPH", content="real content"))
    assert len(doc.sections) == 1
    assert doc.sections[0].name == "GRAPH"


def test_document_to_prompt_joins_sections():
    doc = EvidenceDocument(asset="BTC")
    doc.add(EvidenceSection(name="A", content="alpha"))
    doc.add(EvidenceSection(name="B", content="beta"))
    p = doc.to_prompt()
    assert "## A" in p
    assert "## B" in p
    assert "alpha" in p
    assert "beta" in p


def test_document_to_dict_audit_shape():
    doc = EvidenceDocument(asset="BTC")
    doc.add(EvidenceSection(
        name="NEWS", content="body", confidence=0.6,
        age_s=30.0, source="rss",
    ))
    d = doc.to_dict()
    assert d["asset"] == "BTC"
    assert d["section_count"] == 1
    assert d["sections"][0]["name"] == "NEWS"
    assert d["sections"][0]["confidence"] == 0.6


def test_document_section_lookup():
    doc = EvidenceDocument(asset="BTC")
    doc.add(EvidenceSection(name="NEWS", content="foo"))
    doc.add(EvidenceSection(name="GRAPH", content="bar"))
    assert doc.section("NEWS").content == "foo"
    assert doc.section("MISSING") is None


def test_build_evidence_document_returns_document_type():
    clear_cache()
    doc = build_evidence_document(
        "BTC",
        include_scanner=False, include_traces=False,
        include_news=False, include_fear_greed=False,
        include_graph=False, include_body_controls=False,
    )
    assert isinstance(doc, EvidenceDocument)
    assert doc.asset == "BTC"
    assert doc.sections == []


def test_build_analyst_context_delegates_and_caches():
    """Legacy string API still works; caches within TTL."""
    clear_cache()
    s1 = build_analyst_context(
        "BTC", include_scanner=False, include_traces=False,
        include_news=False, include_fear_greed=False,
        include_graph=False, include_body_controls=False,
    )
    s2 = build_analyst_context(
        "BTC", include_scanner=False, include_traces=False,
        include_news=False, include_fear_greed=False,
        include_graph=False, include_body_controls=False,
    )
    assert s1 == s2
