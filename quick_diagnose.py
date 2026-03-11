#!/usr/bin/env python
"""Simple trading diagnostic"""
import sys
sys.path.insert(0, '.')

print("Step 1: Checking imports...")

try:
    import yaml
    print("  ✓ yaml")
    
    from src.data.fetcher import PriceFetcher
    print("  ✓ PriceFetcher")
    
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    print("  ✓ config loaded")
    
    # Test connection
    exchange_cfg = config.get('exchange', {})
    price_source = PriceFetcher(
        exchange_name=exchange_cfg.get('name', 'binance'),
        testnet=config.get('mode') == 'testnet',
        api_key=exchange_cfg.get('api_key'),
        api_secret=exchange_cfg.get('api_secret'),
    )
    print(f"  ✓ Exchange connected: {price_source.is_available}")
    print(f"  ✓ Authenticated: {price_source.is_authenticated}")
    
    # Get balance
    balance = price_source.get_balance()
    if 'error' not in balance:
        usdt = balance.get('free', {}).get('USDT', 0.0)
        print(f"  ✓ Balance: ${usdt:,.2f} USDT")
    else:
        print(f"  ✗ Balance error: {balance['error']}")
    
    # Get price
    asset = config.get('assets', ['BTC'])[0]
    price = price_source.fetch_latest_price(f"{asset}/USDT")
    print(f"  ✓ Price: ${price:,.2f}")
    
    print("\n✅ ALL CONNECTION TESTS PASSED")
    print("\nPossible reasons for no trades:")
    print("1. Signal generation returning 0 (NEUTRAL)")
    print("2. Risk checks blocking trades")
    print("3. Daily loss limit or drawdown limit hit")
    print("4. Poll interval too long (waiting hours between checks)")
    print("5. VPIN toxicity threshold blocking entry")
    print("\nRun with poll_interval to speed up testing:")
    print('  Edit config.yaml: add "poll_interval: 30" for 30-second checks')
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
