#!/usr/bin/env python3
"""
🧪 NEWS FETCHER API KEY HANDLING TEST
=====================================
Verify that NewsAPI is being used when keys are provided,
and CoinGecko fallback only kicks in when needed.

Run: python verify_newsapi_priority.py
"""

import os
import time
from src.data.news_fetcher import NewsFetcher

print("=" * 80)
print("[TEST] NEWS FETCHER API KEY VERIFICATION")
print("=" * 80)

# Load keys from environment
newsapi_key = os.environ.get('NEWSAPI_KEY')
cryptopanic_token = os.environ.get('CRYPTOPANIC_TOKEN')

print("\n[KEYS] ENVIRONMENT:")
print(f"  NEWSAPI_KEY:       {'ENABLED' if newsapi_key else 'DISABLED'}")
print(f"  CRYPTOPANIC_TOKEN: {'ENABLED' if cryptopanic_token else 'DISABLED'}")

# Initialize fetcher with keys from environment
print("\n[INFO] Initializing NewsFetcher with environment keys...\n")
fetcher = NewsFetcher(
    newsapi_key=newsapi_key,
    cryptopanic_token=cryptopanic_token
)

# Test fetch
print("[INFO] Fetching headlines...\n")
headlines = fetcher.fetch_all(query='bitcoin', limit=30)

# Analyze results
print("\n" + "=" * 80)
print("[RESULTS] ANALYSIS")
print("=" * 80)

source_breakdown = {}
for item in headlines:
    source = item.source.split('/')[0] if '/' in item.source else item.source
    source_breakdown[source] = source_breakdown.get(source, 0) + 1

print(f"\nTotal headlines fetched: {len(headlines)}")
print(f"\nSource breakdown:")
for source, count in sorted(source_breakdown.items(), key=lambda x: -x[1]):
    pct = 100 * count / len(headlines)
    print(f"  * {source:20} {count:3} items ({pct:5.1f}%)")

# Show sample headlines by source
print(f"\n" + "=" * 80)
print("[SAMPLES] HEADLINES (by source)")
print("=" * 80)

for source in sorted(source_breakdown.keys()):
    items = [h for h in headlines if h.source.split('/')[0] == source][:3]
    print(f"\n{source.upper()}:")
    for i, item in enumerate(items, 1):
        age = round(fetcher._check_cache('dummy') and 0 or (time.time() - item.timestamp) / 60, 1) if time.time() > item.timestamp else 0
        print(f"  {i}. {item.title[:60]:60} (age: {age:.0f}m)")

# Verify priority
print(f"\n" + "=" * 80)
print("[VERIFY] PRIORITY")
print("=" * 80)

has_newsapi = any(item.source.startswith('newsapi') for item in headlines)
has_cryptopanic = any(item.source.startswith('cryptopanic') for item in headlines)
has_reddit = any(item.source.startswith('reddit') for item in headlines)
has_coingecko = any(item.source.startswith('coingecko') for item in headlines)

print(f"\nExpected behavior:")
print(f"  1. NewsAPI should be PRIMARY (if key configured)")
print(f"  2. CryptoPanic should be SECONDARY (if token configured)")
print(f"  3. Reddit should be TERTIARY (always available)")
print(f"  4. CoinGecko should be FALLBACK ONLY (only if < 50 items)")

print(f"\nActual results:")
if newsapi_key:
    status = "CORRECT" if has_newsapi else "MISSING (check API key validity)"
    print(f"  > NewsAPI: {status}")
else:
    print(f"  > NewsAPI: Skipped (no key configured)")

if cryptopanic_token:
    status = "CORRECT" if has_cryptopanic else "MISSING (check token validity)"
    print(f"  > CryptoPanic: {status}")
else:
    print(f"  > CryptoPanic: Skipped (no token configured)")

print(f"  > Reddit: {'CORRECT' if has_reddit else 'MISSING'}")
print(f"  > CoinGecko: {'FALLBACK' if has_coingecko and len(headlines) < 50 else 'ALWAYS USED (should be fallback)' if has_coingecko else 'Not used (sufficient items from primary sources)'}")

# Recommendations
print(f"\n" + "=" * 80)
print("[RECOMMENDATIONS]")
print("=" * 80)

if not newsapi_key:
    print("\nERROR: NewsAPI key not found!")
    print("   To enable high-quality news fetching:")
    print("   1. Get a free key from https://newsapi.org/")
    print("   2. Add to .env: export NEWSAPI_KEY='your_key_here'")
    print("   3. Restart system")

if newsapi_key and not has_newsapi:
    print("\nWARNING: NewsAPI key configured but not fetching!")
    print("   Possible issues:")
    print("   1. API key invalid or expired")
    print("   2. API quota exceeded (100/day on free tier)")
    print("   3. Network connection issue")
    print("   4. Check: curl 'https://newsapi.org/v2/everything?q=bitcoin&apiKey=YOUR_KEY'")

if has_coingecko and len(headlines) < 50:
    print("\nINFO: CoinGecko fallback active (fewer than 50 items from primary sources)")
    print("   This is normal - system will use CoinGecko when:")
    print("   * NewsAPI quota is exceeded")
    print("   * Network issues prevent fetching")
    print("   * No API keys configured")

if len(source_breakdown) == 1 and 'coingecko' in source_breakdown:
    print("\nWARNING: Only CoinGecko is being used!")
    print("   This suggests:")
    print("   * NewsAPI key not set or invalid")
    print("   * CryptoPanic token not set or invalid")
    print("   * Reddit fetch failing (should always work)")
    print("   Action: Set NEWSAPI_KEY in .env for better news coverage")

# System check
print(f"\n" + "=" * 80)
print("[SYSTEM STATUS]")
print("=" * 80)

if newsapi_key and has_newsapi:
    print("\nOPTIMAL: Using NewsAPI + Reddit + CryptoPanic (if token set)")
    print("   System has access to high-quality, real-time crypto news")
elif newsapi_key and not has_newsapi:
    print("\nDEGRADED: NewsAPI key configured but not fetching")
    print("   Falling back to secondary sources (CryptoPanic, Reddit, CoinGecko)")
else:
    print("\nLIMITED: No NewsAPI configured")
    print("   Using free-tier sources only (Reddit, CryptoPanic free, CoinGecko)")
    print("   Action: Set NEWSAPI_KEY for significantly better news coverage")

print("\n" + "=" * 80)

# Don't need to import time again
