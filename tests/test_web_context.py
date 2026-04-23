"""Tests for src/ai/web_context.py — summary-first tool wrappers."""
from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from src.ai import web_context as wc


@pytest.fixture(autouse=True)
def _clear_cache_each_test():
    wc.clear_cache()
    yield
    wc.clear_cache()


# ── WebDigest shape ─────────────────────────────────────────────────────


def test_web_digest_to_dict_trims_summary():
    d = wc.WebDigest(source="t", summary="x" * 2000, confidence=0.5)
    out = d.to_dict()
    assert len(out["summary"]) <= 800
    assert out["source"] == "t"
    assert out["confidence"] == 0.5


# ── Graceful fallback when fetchers absent / erroring ───────────────────


def test_news_digest_returns_unavailable_on_fetcher_error():
    with mock.patch("src.data.news_fetcher.NewsFetcher") as M:
        M.side_effect = RuntimeError("boom")
        d = wc.get_news_digest("BTC")
    assert d.confidence == 0.0
    assert "unavailable" in d.summary or "fetcher unavailable" in d.summary


def test_sentiment_digest_returns_unavailable_on_error():
    with mock.patch("src.agents.sentiment_decoder_agent.SentimentDecoderAgent") as M:
        M.side_effect = RuntimeError("no model")
        d = wc.get_sentiment_digest("BTC")
    assert "unavailable" in d.summary
    assert d.confidence == 0.0


def test_macro_digest_returns_unavailable_on_error():
    with mock.patch("src.data.economic_intelligence.EconomicIntelligence") as M:
        M.side_effect = RuntimeError("nope")
        d = wc.get_macro_digest()
    assert "unavailable" in d.summary


# ── Happy path on news ──────────────────────────────────────────────────


def test_news_digest_compacts_headlines():
    fake_items = []
    import time as _t
    now = _t.time()
    for i in range(5):
        item = types.SimpleNamespace(
            title=f"BTC headline {i}",
            timestamp=now - 60 * (i + 1),
            event_type="regulatory" if i < 2 else "macro",
        )
        fake_items.append(item)

    fake_fetcher = mock.MagicMock()
    fake_fetcher.fetch_all.return_value = fake_items
    with mock.patch("src.data.news_fetcher.NewsFetcher", return_value=fake_fetcher):
        d = wc.get_news_digest("BTC", hours=1)

    assert d.source == "news"
    assert "5 BTC headlines" in d.summary
    assert "regulatory=2" in d.summary
    assert "macro=3" in d.summary
    # Digest should include at most ~3 titles; summary stays bounded
    assert len(d.summary) <= 800


def test_news_digest_no_recent_headlines():
    fake_fetcher = mock.MagicMock()
    fake_fetcher.fetch_all.return_value = []
    with mock.patch("src.data.news_fetcher.NewsFetcher", return_value=fake_fetcher):
        d = wc.get_news_digest("BTC", hours=6)
    assert "no headlines" in d.summary
    assert d.confidence < 0.5


# ── Happy path on sentiment ─────────────────────────────────────────────


def test_sentiment_digest_uses_decoder_vote():
    fake_vote = types.SimpleNamespace(direction=1, confidence=0.72, rationale="whales buying")
    fake_agent = mock.MagicMock()
    fake_agent.analyze.return_value = fake_vote
    with mock.patch("src.agents.sentiment_decoder_agent.SentimentDecoderAgent", return_value=fake_agent):
        d = wc.get_sentiment_digest("BTC")
    assert "bullish" in d.summary
    assert "conf 0.72" in d.summary
    assert d.confidence == pytest.approx(0.72)


# ── Fear & greed labels ─────────────────────────────────────────────────


def test_fear_greed_labels_correctly():
    def make_fn(val):
        def _fn():
            fake_ei = mock.MagicMock()
            fake_ei.fetch_all_now.return_value = {"any": {"fear_greed_index": val}}
            return fake_ei
        return _fn

    for val, label in [(10, "extreme fear"), (30, "fear"), (50, "neutral"), (65, "greed"), (85, "extreme greed")]:
        wc.clear_cache()
        with mock.patch("src.data.economic_intelligence.EconomicIntelligence", side_effect=make_fn(val)):
            d = wc.get_fear_greed_digest()
        assert label.replace(" ", "_") in d.tags, f"val={val} expected {label}"


# ── Bundle ──────────────────────────────────────────────────────────────


def test_fetch_bundle_include_subset():
    # Force every tool to error so we can assert the bundle structure
    # without touching real network.
    with mock.patch("src.data.news_fetcher.NewsFetcher", side_effect=RuntimeError), \
         mock.patch("src.agents.sentiment_decoder_agent.SentimentDecoderAgent", side_effect=RuntimeError):
        bundle = wc.fetch_bundle("BTC", include=["news", "sentiment"])
    assert set(bundle.keys()) == {"news", "sentiment"}
    # Both should be graceful "unavailable" digests.
    for d in bundle.values():
        assert d.confidence == 0.0


def test_bundle_to_prompt_block_formats_each_digest():
    bundle = {
        "news": wc.WebDigest(source="news", summary="3 headlines", confidence=0.4),
        "macro": wc.WebDigest(source="macro", summary="CPI rising", confidence=0.6),
    }
    block = wc.bundle_to_prompt_block(bundle)
    assert "WEB CONTEXT DIGESTS" in block
    assert "[news]" in block and "[macro]" in block
    assert "3 headlines" in block and "CPI rising" in block


# ── Cache behavior ──────────────────────────────────────────────────────


def test_cache_returns_same_digest_within_ttl():
    call_count = {"n": 0}

    def make_fetcher():
        call_count["n"] += 1
        fake = mock.MagicMock()
        fake.fetch_all.return_value = []
        return fake

    with mock.patch("src.data.news_fetcher.NewsFetcher", side_effect=make_fetcher):
        wc.get_news_digest("BTC")
        wc.get_news_digest("BTC")
        wc.get_news_digest("BTC")
    assert call_count["n"] == 1  # cached after first call


def test_clear_cache_releases_entries():
    fake_fetcher = mock.MagicMock()
    fake_fetcher.fetch_all.return_value = []
    with mock.patch("src.data.news_fetcher.NewsFetcher", return_value=fake_fetcher):
        wc.get_news_digest("BTC")
        wc.clear_cache()
        wc.get_news_digest("BTC")
    assert fake_fetcher.fetch_all.call_count == 2
