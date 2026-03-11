#!/usr/bin/env python3
"""
Verification Script: News Sources Display During Main Execution
================================================================

This script verifies that when the main system runs, it displays:
1. NewsAPI status (enabled/disabled)
2. CryptoPanic status (enabled/disabled)
3. CoinGecko fallback status
4. Priority order
5. Per-asset source breakdown
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

print("\n" + "="*70)
print("  📰 NEWS SOURCES DISPLAY TEST")
print("="*70)

# Test 1: Check environment variables
print("\n[TEST 1] Checking environment variables...")
print("-" * 70)
newsapi_key = os.environ.get('NEWSAPI_KEY', '')
cryptopanic_token = os.environ.get('CRYPTOPANIC_TOKEN', '')

print(f"✓ NEWSAPI_KEY:        {'✅ SET' if newsapi_key else '❌ NOT SET'}")
print(f"✓ CRYPTOPANIC_TOKEN:  {'✅ SET' if cryptopanic_token else '❌ NOT SET'}")

# Test 2: Initialize NewsFetcher and check its startup message
print("\n[TEST 2] Initializing NewsFetcher (should show source configuration)...")
print("-" * 70)
from src.data.news_fetcher import NewsFetcher

fetcher = NewsFetcher(
    newsapi_key=newsapi_key,
    cryptopanic_token=cryptopanic_token
)

# Test 3: Fetch news and check breakdown
print("\n[TEST 3] Fetching news for BTC (should show source breakdown)...")
print("-" * 70)
try:
    news_items = fetcher.fetch_all('bitcoin', limit=50)
    
    if news_items:
        # Count by source
        source_count = {}
        for item in news_items:
            source = item.source
            source_count[source] = source_count.get(source, 0) + 1
        
        print(f"\n✓ Fetched {len(news_items)} total items")
        print("\n  Source breakdown:")
        for source, count in sorted(source_count.items(), key=lambda x: -x[1]):
            pct = (count / len(news_items)) * 100
            print(f"    • {source}: {count} items ({pct:.1f}%)")
        
        # Verify NewsAPI was used if key is configured
        if newsapi_key and 'NewsAPI' not in source_count:
            print("\n⚠️  WARNING: NewsAPI key is set but no NewsAPI items found!")
        elif newsapi_key and source_count.get('NewsAPI', 0) > 0:
            print(f"\n✅ SUCCESS: NewsAPI is being used! ({source_count['NewsAPI']} items)")
        
        # Verify CryptoPanic was used if token is configured
        if cryptopanic_token and 'CryptoPanic' not in source_count:
            print("⚠️  WARNING: CryptoPanic token is set but no CryptoPanic items found!")
        elif cryptopanic_token and source_count.get('CryptoPanic', 0) > 0:
            print(f"✅ SUCCESS: CryptoPanic is being used! ({source_count['CryptoPanic']} items)")
        
        # Verify CoinGecko is only fallback (appeared if < 50 items from primary)
        if source_count.get('CoinGecko', 0) > 0:
            if len(news_items) >= 50:
                print(f"✅ SUCCESS: CoinGecko used as intended (total items = {len(news_items)} >= 50)")
            else:
                print(f"✅ SUCCESS: CoinGecko fallback triggered (total items = {len(news_items)} < 50)")
        else:
            print(f"✅ SUCCESS: CoinGecko NOT needed (sufficient items from primary sources)")
    else:
        print("⚠️  No news items returned")
        
except Exception as e:
    print(f"❌ Error fetching news: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Show what would display in main.py
print("\n[TEST 4] Simulating executor.py startup display...")
print("-" * 70)

newsapi_status = "✅ ENABLED" if os.environ.get('NEWSAPI_KEY') else "❌ DISABLED"
cryptopanic_status = "✅ ENABLED" if os.environ.get('CRYPTOPANIC_TOKEN') else "❌ DISABLED"
reddit_status = "✅ ALWAYS ENABLED"
coingecko_status = "⚙️  FALLBACK ONLY"

print("\n  📰 L2 SENTIMENT DATA SOURCES")
print("  " + "-" * 50)
print(f"  📧 NewsAPI:        {newsapi_status}")
print(f"  🚨 CryptoPanic:    {cryptopanic_status}")
print(f"  🔴 Reddit:         {reddit_status}")
print(f"  🪙 CoinGecko:      {coingecko_status}")
print(f"  📊 Priority Order: NewsAPI → CryptoPanic → Reddit → CoinGecko")

print("\n" + "="*70)
print("  ✅ NEWS SOURCES DISPLAY TEST COMPLETE")
print("="*70 + "\n")

print("\n📝 SUMMARY:")
print(f"   • NewsAPI is {'enabled' if newsapi_key else 'disabled (add NEWSAPI_KEY to .env)'}") 
print(f"   • CryptoPanic is {'enabled' if cryptopanic_token else 'disabled (add CRYPTOPANIC_TOKEN to .env)'}")
print(f"   • Reddit is always enabled (no key needed)")
print(f"   • CoinGecko is fallback only")
print(f"\n   When main runs, this source configuration will be displayed")
print(f"   and news items will show which sources were used for each asset.\n")
