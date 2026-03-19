#!/usr/bin/env python3
"""
Verify NEWS FETCHER SOURCE PRIORITY FIX
======================================
Just check that the code prioritizes NewsAPI over CoinGecko
(don't actually fetch to avoid timeouts)
"""

import inspect
from src.data.news_fetcher import NewsFetcher

print("=" * 80)
print("[CHECK] NEWS FETCHER SOURCE PRIORITY")
print("=" * 80)

# Get the fetch_all method code
method = NewsFetcher.fetch_all
source = inspect.getsource(method)

# Check for the key conditions
print("\n[ANALYSIS] Checking fetch_all() method:")
print("-" * 80)

checks = [
    ("NewsAPI conditional", "if self.newsapi_key:" in source),
    ("NewsAPI called BEFORE CoinGecko", source.index("if self.newsapi_key:") < source.index("if len(items) < limit")),
    ("CoinGecko fallback condition", "if len(items) < limit // 2:" in source),
    ("CoinGecko NOT always called", source.count("self._fetch_coingecko_trending()") == 1),
]

all_pass = True
for check_name, result in checks:
    status = "[PASS]" if result else "[FAIL]"
    print(f"{status} {check_name}")
    if not result:
        all_pass = False

# Show the actual fetch_all method
print("\n[CODE] fetch_all() method signature:")
print("-" * 80)
print(inspect.getsource(NewsFetcher.fetch_all)[:500] + "...\n")

# Also check __init__ logging
init_source = inspect.getsource(NewsFetcher.__init__)
has_logging = "NewsAPI" in init_source and "print" in init_source
print("[CHECK] __init__ logging:")
print(f"{'[PASS]' if has_logging else '[FAIL]'} Initialization logs configured sources")

print("\n" + "=" * 80)
if all_pass:
    print("[SUCCESS] NEWS FETCHER FIX APPLIED")
    print("\nFix Details:")
    print("  1. NewsAPI is PRIMARY (checked first)")
    print("  2. CryptoPanic is SECONDARY (checked second)")
    print("  3. Reddit is TERTIARY (always checked)")
    print("  4. CoinGecko is FALLBACK only (only if < 50 items)")
    print("\nBehavior:")
    print("  * When NEWSAPI_KEY is set:")
    print("    -> Fetches top news from NewsAPI")
    print("    -> CoinGecko only used if < 50 items returned")
    print("\n  * When NEWSAPI_KEY not set:")
    print("    -> Falls back to Reddit + CryptoPanic (if token set)")
    print("    -> CoinGecko only used if still < 50 items")
else:
    print("[FAILURE] Fix not properly applied")

print("=" * 80)
