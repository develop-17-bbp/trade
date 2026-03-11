#!/usr/bin/env python
"""
Trade Execution Diagnostic Tool
Identifies why the system stops making trades
"""
import sys
import os
sys.path.insert(0, '.')

import yaml
import time
from src.data.fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.trading.strategy import HybridStrategy
from src.models.lightgbm_classifier import LightGBMClassifier
from src.risk.manager import RiskManager

print("\n" + "="*70)
print("  🔍 TRADING SYSTEM DIAGNOSTIC")
print("="*70)

# Load config
with open('config.yaml') as f:
    config = yaml.safe_load(f)

print(f"\n1. SYSTEM CONFIGURATION")
print("-" * 70)
print(f"   Mode: {config.get('mode', 'unknown')}")
print(f"   Assets: {config.get('assets', [])}")
print(f"   Initial Capital: ${config.get('initial_capital', 100000):,.2f}")
print(f"   Poll Interval: {config.get('poll_interval', 3600)}s")

# Test API connection
print(f"\n2. API CONNECTION")
print("-" * 70)
exchange_cfg = config.get('exchange', {})
price_source = PriceFetcher(
    exchange_name=exchange_cfg.get('name', 'binance'),
    testnet=config.get('mode') == 'testnet',
    api_key=exchange_cfg.get('api_key'),
    api_secret=exchange_cfg.get('api_secret'),
)

if not price_source.is_available:
    print("   ❌ CRITICAL: Exchange not available!")
    sys.exit(1)

print(f"   ✅ Authenticated: {price_source.is_authenticated}")
print(f"   ✅ Testnet Mode: {price_source.testnet}")

# Fetch balances
balance = price_source.get_balance()
if 'error' not in balance:
    usdt_balance = balance.get('free', {}).get('USDT', 0.0)
    print(f"   ✅ Account Balance: ${usdt_balance:,.2f} USDT")
else:
    print(f"   ❌ Balance Error: {balance.get('error')}")
    sys.exit(1)

# Test data fetching
print(f"\n3. MARKET DATA")
print("-" * 70)
asset = config.get('assets', ['BTC'])[0]
symbol = f"{asset}/USDT"
try:
    price = price_source.fetch_latest_price(symbol)
    print(f"   ✅ {symbol}: ${price:,.2f}")
except Exception as e:
    print(f"   ❌ Error fetching price: {e}")

# Fetch OHLCV for strategy testing
try:
    ohlcv = price_source.fetch_ohlcv(symbol, timeframe='1h', limit=200)
    if ohlcv:
        print(f"   ✅ Fetched {len(ohlcv)} OHLCV bars")
        closes = [bar[4] for bar in ohlcv]
        print(f"   ✅ Price range: ${min(closes):,.2f} - ${max(closes):,.2f}")
    else:
        print(f"   ❌ No OHLCV data returned")
except Exception as e:
    print(f"   ❌ Error fetching OHLCV: {e}")
    sys.exit(1)

# Test signal generation
print(f"\n4. SIGNAL GENERATION")
print("-" * 70)
try:
    strategy = HybridStrategy(config)
    
    # Generate signals
    opens = closes[:-1] + [closes[-1]]
    highs = [p * 1.01 for p in closes]
    lows = [p * 0.99 for p in closes]
    volumes = [1000000.0] * len(closes)
    
    result = strategy.generate_signals(
        prices=closes,
        highs=highs,
        lows=lows,
        volumes=volumes,
        headlines=[],
        account_balance=usdt_balance
    )
    
    signals = result.get('signals', [])
    if signals:
        last_signal = signals[-1]
        print(f"   ✅ Generated {len(signals)} signals")
        print(f"   Signal Distribution:")
        print(f"      - LONG (+1):  {sum(1 for s in signals if s > 0)} signals")
        print(f"      - FLAT (0):   {sum(1 for s in signals if s == 0)} signals")
        print(f"      - SHORT (-1): {sum(1 for s in signals if s < 0)} signals")
        print(f"   Last Signal: {last_signal} ({'LONG' if last_signal > 0 else 'SHORT' if last_signal < 0 else 'NEUTRAL'})")
        
        if last_signal == 0:
            print(f"   ⚠️  ISSUE 1: Last signal is NEUTRAL (0) - no trade will execute")
            final_decisions = result.get('final_decisions', [])
            if final_decisions:
                last_decision = final_decisions[-1]
                print(f"        Reason: {last_decision.get('reason', 'Unknown')}")
    else:
        print(f"   ❌ ISSUE 1: No signals generated!")
        
except Exception as e:
    print(f"   ❌ Error generating signals: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test risk checks
print(f"\n5. RISK CHECKS")
print("-" * 70)
try:
    risk_mgr = RiskManager(
        initial_capital=usdt_balance,
        daily_loss_limit_pct=3.0,
        max_drawdown_pct=10.0
    )
    
    # Check if trading is safe
    current_price = closes[-1]
    atr_val = (max(closes[-20:]) - min(closes[-20:])) / 20
    
    is_safe, reason = risk_mgr.is_trade_safe(
        current_price=current_price,
        direction=1,  # Test LONG
        atr_value=atr_val,
        account_balance=usdt_balance
    )
    
    if is_safe:
        print(f"   ✅ LONG trade is safe")
    else:
        print(f"   ⚠️  ISSUE 2: Risk check blocked trade")
        print(f"        Reason: {reason}")
        
except Exception as e:
    print(f"   ❌ Error in risk checks: {e}")

# Summary
print(f"\n" + "="*70)
print("  🎯 DIAGNOSTIC SUMMARY")
print("="*70)
print("""
If LONG signal was generated but no trades are executing, check:

1. ⚠️  SIGNAL ISSUE
   → Last signal is 0 (NEUTRAL): Strategy not generating BUY/SELL signals
   → Likely causes:
      • Low sentiment score (headlines required for strong signals)
      • LightGBM classifier not confident
      • Model drift detected (volatility too high)
   → Fix: Add sentiment context (headlines) or reduce model drift

2. ⚠️  RISK ISSUE
   → Risk manager blocking trades
   → Likely causes:
      • Daily loss limit reached
      • Maximum drawdown exceeded
      • Position too large for account size
   → Fix: Check config.yaml risk parameters

3. ⚠️  VPIN TOXICITY
   → VPIN flow toxicity check blocking entry
   → Fix: Wait for market to stabilize, or reduce VPIN threshold

4. ⚠️  CYCLE TIME
   → Poll interval too long (default 3600s = 1 hour)
   → System only checks for trades once per hour
   → Fix: Reduce poll_interval in config.yaml for faster trading

NEXT STEP: Run system with debug mode:
  python -m src.main --mode testnet --symbol BTC --debug
""")

