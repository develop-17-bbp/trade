#!/usr/bin/env python3
"""
News API Priority Test - Load keys from .env directly
"""

import os
from dotenv import load_dotenv
from src.data.news_fetcher import NewsFetcher

# Load .env file explicitly
load_dotenv(override=True)

print("=" * 80)
print("[TEST] NEWS FETCHER API KEY VERIFICATION")
print("=" * 80)

# Load keys from environment (now from .env file)
newsapi_key = os.environ.get('NEWSAPI_KEY')
cryptopanic_token = os.environ.get('CRYPTOPANIC_TOKEN')

print("\n[KEYS] LOADED FROM .env:")
print(f"  NEWSAPI_KEY:       {newsapi_key[:20] + '...' if newsapi_key else 'NOT FOUND'}")
print(f"  CRYPTOPANIC_TOKEN: {cryptopanic_token[:20] + '...' if cryptopanic_token else 'NOT FOUND'}")

if not newsapi_key:
    print("\n[ERROR] NEWSAPI_KEY not found in .env!")
    print("Action: Add NEWSAPI_KEY=your_key to .env file")

# Initialize fetcher
print("\n[INIT] Creating NewsFetcher with keys...\n")
fetcher = NewsFetcher(
    newsapi_key=newsapi_key,
    cryptopanic_token=cryptopanic_token
)

# Fetch headlines
print("[FETCH] Requesting headlines...\n")
try:
    headlines = fetcher.fetch_all(query='bitcoin', limit=20)
    
    # Analyze sources
    sources = {}
    for h in headlines:
        src = h.source.split('/')[0]
        sources[src] = sources.get(src, 0) + 1
    
    print("\n" + "=" * 80)
    print("[SUMMARY] NEWS SOURCES USED")
    print("=" * 80)
    print(f"\nTotal headlines: {len(headlines)}")
    print(f"Source breakdown:")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  * {src:20} {cnt:3} items")
    
    # Verdict
    print("\n" + "=" * 80)
    print("[VERDICT]")
    print("=" * 80)
    
    if newsapi_key and 'newsapi' in sources:
        print("\n[OK] NewsAPI IS BEING USED (fix worked!)")
        print(f"     {sources.get('newsapi', 0)} news articles fetched from NewsAPI")
        if sources.get('coingecko', 0) > 0:
            print(f"     CoinGecko used as fallback: {sources.get('coingecko', 0)} items")
            print("     -> This is normal when NewsAPI quota/results are limited")
    elif newsapi_key:
        print("\n[WARNING] NewsAPI key configured but NOT BEING USED")
        print("Possible causes:")
        print("  * API quota exceeded")
        print("  * Network issue")
        print("  * Invalid API key")
    else:
        print("\n[INFO] NewsAPI key not configured")
        print("       System using fallback sources: Reddit, CryptoPanic, CoinGecko")
        
except Exception as e:
    print(f"\n[ERROR] Fetch failed: {e}")

print("\n" + "=" * 80)
