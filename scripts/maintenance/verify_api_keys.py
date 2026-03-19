#!/usr/bin/env python
"""Quick API Key Verification Tool"""
import sys
import yaml
from pathlib import Path

# Load config
config_path = Path('config.yaml')
if not config_path.exists():
    print("❌ config.yaml not found!")
    sys.exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

exchange_config = config.get('exchange', {})
api_key = exchange_config.get('api_key', '').strip()
api_secret = exchange_config.get('api_secret', '').strip()
mode = config.get('mode', 'paper')

print("\n📋 API Configuration Verification")
print("=" * 60)
print(f"Mode: {mode}")
print(f"API Key Set: {'✅ Yes' if api_key else '❌ No'}")
print(f"API Secret Set: {'✅ Yes' if api_secret else '❌ No'}")

if not api_key or not api_secret:
    print("\n❌ API keys are missing!")
    print("\n📖 To fix this:")
    if mode == 'testnet':
        print("   1. Go to: https://testnet.binance.vision/key/publicKey")
        print("   2. Create new API key (Label: Trading Bot)")
        print("   3. Copy API Key and Secret")
        print("   4. Paste into config.yaml exchange.api_key and exchange.api_secret")
    else:
        print("   1. Go to: https://www.binance.com/en/user/settings/api-management")
        print("   2. Create new API key (Label: Trading Bot)")
        print("   3. Copy API Key and Secret")
        print("   4. Paste into config.yaml exchange.api_key and exchange.api_secret")
    print("\n📚 Full guide: See API_KEYS_SETUP.md")
    sys.exit(1)

print("\n✅ API keys are configured!")
print("\n🔌 Testing connection...")

try:
    import ccxt
    
    exchange_config_ccxt = {
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'adjustForTimeDifference': True,
            'recvWindow': 60000,
        },
        'apiKey': api_key,
        'secret': api_secret,
    }
    
    exchange = ccxt.binance(exchange_config_ccxt)
    
    if mode == 'testnet':
        exchange.set_sandbox_mode(True)
        print("   [TESTNET] Sandbox mode enabled")
    
    # Try to fetch balance
    balance = exchange.fetch_balance()
    print("   ✅ Connection successful!")
    print("\n💰 Account Balances:")
    print("   ────────────────────")
    
    free = balance.get('free', {})
    total = balance.get('total', {})
    
    # Show USDT
    usdt_free = free.get('USDT', 0)
    usdt_total = total.get('USDT', 0)
    print(f"   USDT: {usdt_free:>12,.4f} / {usdt_total:,.4f}")
    
    # Show other assets
    for asset in sorted(free.keys()):
        if asset in ('USDT', 'free', 'used', 'total') or not free.get(asset):
            continue
        amt_free = free.get(asset, 0)
        amt_total = total.get(asset, 0)
        print(f"   {asset:<6}: {amt_free:>12,.8f} / {amt_total:,.8f}")
    
    print("\n✅ Ready to trade!")
    
except Exception as e:
    if '2008' in str(e) or 'Invalid Api-Key' in str(e):
        print(f"   ❌ Invalid API key! Error: {e}")
        print("\n   The API key/secret in config.yaml is invalid.")
        print("   Make sure you're using the correct key for the mode (testnet vs live).")
    elif 'Unauthorized' in str(e):
        print(f"   ❌ Unauthorized! Error: {e}")
        print("\n   Check that:")
        print("   - API key has 'Enable Spot Trading' enabled")
        print("   - Your IP is whitelisted (or no IP restriction)")
    else:
        print(f"   ❌ Connection failed! Error: {e}")
    sys.exit(1)
