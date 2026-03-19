📰 NEWS SOURCES DISPLAY GUIDE
============================
March 11, 2026

┌─────────────────────────────────────────────────────────────────┐
│ WHAT DISPLAYS WHEN THE MAIN SYSTEM RUNS                         │
└─────────────────────────────────────────────────────────────────┘

When you run: `python -m src.main` or `python src/main.py`

The system will display the following during startup:

===== STARTUP SEQUENCE =====

1. EXCHANGE WALLET BALANCES
   ├─ USDT balance
   ├─ BTC, ETH holdings (or configured assets)
   └─ Total portfolio value

2. LIVE SPOT PRICES
   ├─ BTC/USDT price
   ├─ ETH/USDT price
   └─ Holding values

3. 9-LAYER INTELLIGENCE STATUS
   ├─ L1: Quantitative Engine        [ONLINE]
   ├─ L2: Sentiment Intelligence     [ONLINE or DEGRADED]
   ├─ L3: Risk Fortress              [ONLINE]
   ├─ L4: Signal Fusion              [ONLINE]
   ├─ L5: Execution Engine           [PAPER or ONLINE]
   ├─ L6: Strategist Hub             [ONLINE or OFFLINE]
   ├─ L7: Advanced Learning          [ONLINE]
   ├─ L8: Tactical Memory            [ONLINE]
   └─ L9: Evolution Portal           [STANDBY]

   ✅ NEW: 4. L2 SENTIMENT DATA SOURCES  ← YOU ARE HERE
   ├─ 📧 NewsAPI:        ✅ ENABLED (or ❌ DISABLED)
   ├─ 🚨 CryptoPanic:    ✅ ENABLED (or ❌ DISABLED)
   ├─ 🔴 Reddit:         ✅ ALWAYS ENABLED
   ├─ 🪙 CoinGecko:      ⚙️  FALLBACK ONLY
   └─ 📊 Priority: NewsAPI → CryptoPanic → Reddit → CoinGecko

>=== SYSTEM STARTING... ===

5. FOR EACH ASSET (BTC, ETH, etc.), PER CYCLE:

   [NewsAPI] Source Configuration:
     - NewsAPI:      ENABLED
     - CryptoPanic:  ENABLED
     - Reddit:       ALWAYS ENABLED (no key needed)
     - CoinGecko:    FALLBACK ONLY (used if other sources return <50 items)
     - Priority: NewsAPI -> CryptoPanic -> Reddit -> CoinGecko fallback

   [NewsAPI] Fetch Summary:
     • NewsAPI (75 items)
     • CryptoPanic (8 items)
     • Reddit (100 items)
     • CoinGecko (skipped - sufficient items from primary sources)
     └─ Total: 168 unique items (after dedup)

   [NewsAPI] Returned Items Breakdown (50 items):
     • NewsAPI: 25 items (50.0%)
     • reddit/r/Bitcoin: 15 items (30.0%)
     • CryptoPanic: 8 items (16.0%)
     • reddit/r/ethereum: 2 items (4.0%)

   [L2 SENTIMENT] Fetched 50 news items for BTC
   [L2 SENTIMENT] Source Breakdown:
     • NewsAPI: 25 items
     • reddit/r/Bitcoin: 15 items
     • CryptoPanic: 8 items
     • reddit/r/ethereum: 2 items
     📰 NewsAPI: Bitcoin surges amid institutional adoption wave...
     📰 reddit/r/Bitcoin: Major exchange adds BTC support...
     📰 CryptoPanic: Market update: BTC breaks resistance...

┌─────────────────────────────────────────────────────────────────┐
│ EXAMPLE: WHAT YOU'LL SEE WITH API KEYS CONFIGURED              │
└─────────────────────────────────────────────────────────────────┘

File: .env
--------
NEWSAPI_KEY=your_newsapi_key_here
CRYPTOPANIC_TOKEN=your_token_here

Startup Display:
  📰 L2 SENTIMENT DATA SOURCES
  --------------------------------------------------
  📧 NewsAPI:        ✅ ENABLED
  🚨 CryptoPanic:    ✅ ENABLED
  🔴 Reddit:         ✅ ALWAYS ENABLED
  🪙 CoinGecko:      ⚙️  FALLBACK ONLY
  📊 Priority Order: NewsAPI → CryptoPanic → Reddit → CoinGecko

During Trading:
  [NewsAPI] Fetch Summary:
    • NewsAPI (75 items)          ← HIGH QUALITY NEWS
    • CryptoPanic (8 items)       ← REAL-TIME CRYPTO
    • Reddit (100 items)          ← COMMUNITY SENTIMENT
    • CoinGecko (skipped)         ← NOT USED (sufficient data)

  [L2 SENTIMENT] Source Breakdown:
    • NewsAPI: 35 items (70%)     ← PRIMARY
    • CryptoPanic: 10 items (20%) ← SECONDARY
    • reddit/...: 5 items (10%)   ← TERTIARY

┌─────────────────────────────────────────────────────────────────┐
│ EXAMPLE: WITHOUT API KEYS (DEGRADED MODE)                      │
└─────────────────────────────────────────────────────────────────┘

File: .env
--------
# No NEWSAPI_KEY or CRYPTOPANIC_TOKEN

Startup Display:
  📰 L2 SENTIMENT DATA SOURCES
  --------------------------------------------------
  📧 NewsAPI:        ❌ DISABLED
  🚨 CryptoPanic:    ❌ DISABLED
  🔴 Reddit:         ✅ ALWAYS ENABLED
  🪙 CoinGecko:      ⚙️  FALLBACK ONLY
  📊 Priority Order: Reddit → CoinGecko fallback

During Trading:
  [NewsAPI] Fetch Summary:
    • NewsAPI (⚠️  no key configured)        ← NOT AVAILABLE
    • CryptoPanic (⚠️  no token configured)  ← NOT AVAILABLE
    • Reddit (100 items)                     ← USED
    • CoinGecko (22 items fallback)          ← USED (< 50 items)

  [L2 SENTIMENT] Source Breakdown:
    • reddit/r/Bitcoin: 35 items
    • reddit/r/cryptocurrency: 30 items
    • reddit/r/CryptoMarkets: 25 items
    • CoinGecko: 20 items

┌─────────────────────────────────────────────────────────────────┐
│ KEY POINTS: WHAT'S BEEN FIXED                                  │
└─────────────────────────────────────────────────────────────────┘

BEFORE THE FIX:
  ❌ CoinGecko was ALWAYS called
  ❌ NewsAPI/CryptoPanic often ignored
  ❌ No visibility into which sources were being used
  ❌ API quota wasted on unnecessary CoinGecko calls

AFTER THE FIX:
  ✅ NewsAPI is PRIMARY source (when key available)
  ✅ CryptoPanic is SECONDARY source (when token available)
  ✅ Reddit is TERTIARY source (always available)
  ✅ CoinGecko is FALLBACK ONLY (when < 50 items from above)
  ✅ Clear visibility: Fetch Summary shows what was fetched
  ✅ Item Breakdown shows which sources made it to final results
  ✅ L2 Sentiment logs show exact source mix per asset

┌─────────────────────────────────────────────────────────────────┐
│ HOW TO VERIFY IT'S WORKING                                     │
└─────────────────────────────────────────────────────────────────┘

Run 1: Start the system normally
  $ python -m src.main
  
  Look for:
  1. At startup: "[NewsAPI] Source Configuration:" section
  2. Shows your keys as ENABLED (✅) or DISABLED (❌)
  3. During trading: "[NewsAPI] Fetch Summary:" section
  4. Shows breakdown of sources actually used

Run 2: Check logs
  $ tail -f logs/backtest_full.txt | grep -i "newsapi\|source\|fetch"
  
  Look for:
  1. "[NewsAPI] Source Configuration:" (at startup)
  2. "[NewsAPI] Fetch Summary:" (per asset, per cycle)
  3. "[L2 SENTIMENT] Source Breakdown:" (detailed per-asset breakdown)

Run 3: Verify CoinGecko is now fallback only
  $ grep -i "coingecko" logs/backtest_full.txt
  
  Should show:
  ✅ "CoinGecko (skipped - sufficient items)" OR
  ✅ "CoinGecko FALLBACK" (only if < 50 items)
  
  NOT show:
  ❌ "CoinGecko" on every single line

┌─────────────────────────────────────────────────────────────────┐
│ NEXT STEPS                                                      │
└─────────────────────────────────────────────────────────────────┘

1. Ensure API Keys are in .env:
   NEWSAPI_KEY=your_key_here
   CRYPTOPANIC_TOKEN=your_token_here

2. Run: python -m src.main

3. Observe:
   • Startup shows which sources are enabled
   • Each fetch cycle shows source breakdown
   • CoinGecko appears only when needed

4. Monitor quality:
   • Better news sources = better sentiment scores
   • Better sentiment = more profitable signals
   • Cleaner signals = higher win rate

═══════════════════════════════════════════════════════════════════

SUCCESS CRITERIA:
✅ L2 Sentiment Data Sources shows NewsAPI ✅ ENABLED
✅ Fetch Summary shows NewsAPI items fetched
✅ Returned Items Breakdown includes NewsAPI items
✅ CoinGecko shows as "skipped" or "FALLBACK" (not always used)
✅ NewsAPI appears before CoinGecko in priority order

═══════════════════════════════════════════════════════════════════
