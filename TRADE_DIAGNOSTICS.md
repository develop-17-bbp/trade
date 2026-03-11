#!/usr/bin/env python
"""
🚀 TRADING RESUME GUIDE
Why the system stopped trading and how to fix it
"""

print("""
═══════════════════════════════════════════════════════════════════════
  🔴 PROBLEM: System made initial profit but then STOPPED trading
═══════════════════════════════════════════════════════════════════════

ROOT CAUSES IDENTIFIED:
─────────────────────────────────────────────────────────────────────

1️⃣  STRICT RL VETO WAS BLOCKING TRADES (FIXED ✅)
   Location: src/trading/meta_controller.py:65
   
   Before: if rl_prob < 0.45: return 0 (ZERO OUT SIGNAL)
   Impact: ANY trade with RL confidence below 45% was blocked
   
   After: if rl_prob < 0.30 AND lgb_conf < 0.55: return 0
   Impact: Only blocks in EXTREME disagreement (both models doubt)
   Result: Much more trading allowed! ✅

2️⃣  RISK LIMITS WERE TOO RESTRICTIVE (FIXED ✅)
   Location: config.yaml
   
   Before:
     daily_loss_limit_pct: 3.0%      ← Hit after small drawdown
     max_drawdown_pct: 10.0%          ← Circuit breaker too tight
   
   After:
     daily_loss_limit_pct: 5.0%      ← More tolerance for volatility
     max_drawdown_pct: 15.0%          ← Reasonable for testnet
   
   Why: Testnet is for testing! Should not halt on small temporary losses

3️⃣  NO DEBUG VISIBILITY (DIAGNOSTICS ADDED ✅)
   New tools:
     • quick_diagnose.py   - Check if system components working
     • TRADE_DIAGNOSTICS.md - Document what each layer is doing

═══════════════════════════════════════════════════════════════════════
  ✅ FIXES APPLIED
═══════════════════════════════════════════════════════════════════════

Changes made:
✅ Relaxed RL veto threshold (0.45 → 0.30)
✅ Increased daily loss limit (3% → 5%)
✅ Increased max drawdown limit (10% → 15%)
✅ Added min_confidence config (0.55) for signal generation
✅ Added testnet_aggressive flag for more aggressive trading
✅ Max trades/hour limit (4) to prevent overtrading

═══════════════════════════════════════════════════════════════════════
  🚀 TO RESUME TRADING
═══════════════════════════════════════════════════════════════════════

Step 1: Restart the system
  $ python -m src.main --mode testnet --symbol BTC

Step 2: Wait for first signal (usually within 1-2 minutes)
  Look for log line: "[PHASE 5] Evaluating trade for BTC (Signal: 1)" or "-1"

Step 3: Monitor trades executing
  Watch for: "[PHASE 5] ORDER INITIATED: ..."

Step 4: Check profit
  Look for: "[LIVE] Wallet: $XX,XXX.XX | Return: +X.XX%"

═══════════════════════════════════════════════════════════════════════
  🔍 DEBUGGING: If trades STILL not executing
═══════════════════════════════════════════════════════════════════════

Run diagnostic:
  $ python quick_diagnose.py

Check these conditions:

1. Signal Generation Check:
   "Last Signal: 1 (LONG)" means good signal
   "Last Signal: 0 (NEUTRAL)" means no signal - need to boost confidence
   
2. Risk Block Check:
   "LONG trade is safe" = good to go
   "Risk check blocked trade: Reason: Daily loss..." = profit taking too aggressive
   
3. API Connection:
   "Balance: $10,000.00 USDT" = connected OK
   "Error: Invalid Api-Key" = need to update config.yaml keys

═══════════════════════════════════════════════════════════════════════
  ⚙️  TUNING PARAMETERS FOR BETTER TRADING
═══════════════════════════════════════════════════════════════════════

To make MORE trades happen:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Edit config.yaml:

# Trade frequency
poll_interval: 30            # Check every 30 seconds (instead of 60)
max_trades_per_hour: 6       # Allow up to 6 trades/hour

# Signal generation
min_confidence: 0.50         # Lower = more trades (default 0.55)

# Risk tolerance (CAUTION: testnet only!)
daily_loss_limit_pct: 10.0   # Don't halt on small losses
max_drawdown_pct: 20.0       # More room for swings

To make LESS trades (safer):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

poll_interval: 300           # Check every 5 minutes
min_confidence: 0.65         # Higher = fewer, higher-quality signals
max_trades_per_hour: 2       # Conservative

To target SPECIFIC profit:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

min_return_pct: 1.0          # Exit when up 1% per trade
risk_per_trade_pct: 0.5      # Risk only 0.5% to protect profits

═══════════════════════════════════════════════════════════════════════
  💡 WHY THIS HAPPENED
═══════════════════════════════════════════════════════════════════════

The system has 3 independent signal sources (L1, L2, L3):
  • L1: LightGBM technical classifier
  • L2: RL agent (reinforcement learning)
  • L3: Risk limits

For a trade to execute:
  ✅ L1 (LightGBM) must generate buy/sell signal
  ✅ L2 (RL agent) must agree somewhat (was too strict at 0.45!)
  ✅ L3 (Risk) must allow it

The ORIGINAL config had L2 veto too harsh:
  RL confidence 0.42 + LightGBM confident?
  -> BLOCKED! Even though LightGBM wanted to trade

NEW LOGIC is smarter:
  RL confidence 0.42 + LightGBM confident (0.70)?
  -> If LightGBM_conf > 0.55, ALLOW it!
  -> Only block if BOTH models unsure

═══════════════════════════════════════════════════════════════════════
  📊 EXPECTED BEHAVIOR NOW
═══════════════════════════════════════════════════════════════════════

Watch for this pattern:

[LIVE] BAR 1 | Wallet: $10,000.00 | Return: +0.00%
  Signal generated: 1 (LONG)
  [PHASE 5] Evaluating trade for BTC (Signal: 1)
  [PHASE 5] ORDER INITIATED: exec_001

[LIVE] BAR 2 | Wallet: $10,002.50 | Return: +0.02%
  Signal generated: -1 (SHORT)
  [PHASE 5] ORDER INITIATED: exec_002

[LIVE] BAR 3 | Wallet: $10,025.75 | Return: +0.26%
  Profit accumulating! ✅

═══════════════════════════════════════════════════════════════════════
  🛠️  IF ISSUES PERSIST
═══════════════════════════════════════════════════════════════════════

1. Check signal generation:
   $ python -c "from src.trading.strategy import HybridStrategy;
                s = HybridStrategy({});
                r = s.generate_signals(prices=[100,101,102,103,104]);
                print('Signals:', r['signals'])"

2. Check if risk is halted:
   Look for: "[SYSTEM] HALTED:" in logs

3. Check meta-controller arbitration:
   Add debug print to meta_controller.py:arbitrate() method
   See what lgb/rl scores are

4. Check if using PAPER mode:
   Paper mode doesn't execute real trades, only simulates
   Run with: --mode testnet (not 'paper')

═══════════════════════════════════════════════════════════════════════
""")

print("✅ Summary: System should now resume trading!")
print("   Start: python -m src.main --mode testnet --symbol BTC")
print("   Monitor: Look for 'ORDER INITIATED' messages")
print("═════════════════════════════════════════════════════════════════")
