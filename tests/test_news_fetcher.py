import pytest
from src.data.news_fetcher import NewsFetcher, NewsItem


def test_news_fetcher_limit_default(monkeypatch):
    # monkeypatch individual fetchers to return known number of items
    nf = NewsFetcher()
    # simulate fetch_all behavior by patching source-specific methods
    def fake_fetch_reddit(q, limit=10):
        return [NewsItem(f"r{i}", "reddit", 0) for i in range(limit)]
    def fake_fetch_cryptopanic(limit=10):
        return [NewsItem(f"c{i}", "cryptopanic", 0) for i in range(limit)]
    nf._fetch_reddit = fake_fetch_reddit
    nf._fetch_cryptopanic = fake_fetch_cryptopanic
    nf.newsapi_key = None
    # when calling fetch_all with default, should respect default limit=100
    items = nf.fetch_all("crypto")
    assert len(items) <= 100
    # requesting a custom limit should override
    items2 = nf.fetch_all("crypto", limit=5)
    assert len(items2) <= 5


def test_fetch_headlines_respects_limit():
    nf = NewsFetcher()
    # patch fetch_all to control output
    nf.fetch_all = lambda q, limit=100: [NewsItem(str(i), "src", 0) for i in range(limit)]
    headlines = nf.fetch_headlines("x", limit=20)
    assert len(headlines) == 20
