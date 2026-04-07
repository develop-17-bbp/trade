"""
MT5 Setup & Diagnostic Tool
=============================
Run this to verify your MetaTrader 5 connection and see available symbols.

Usage:
    python -m src.scripts.mt5_setup
    python -m src.scripts.mt5_setup --status        # Show open positions & P&L
    python -m src.scripts.mt5_setup --symbols       # List all available crypto symbols
    python -m src.scripts.mt5_setup --history 30    # Show last 30 days trade history
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description='MetaTrader 5 Setup & Diagnostics')
    parser.add_argument('--status', action='store_true', help='Show open positions and P&L')
    parser.add_argument('--symbols', action='store_true', help='List all available crypto symbols')
    parser.add_argument('--history', type=int, default=0, help='Show trade history for N days')
    parser.add_argument('--test-order', action='store_true', help='Place a tiny test order (use with caution)')
    args = parser.parse_args()

    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("MetaTrader5 package not installed.")
        print("Run: pip install MetaTrader5")
        sys.exit(1)

    # Run basic setup check
    from src.trading.mt5_bridge import check_mt5_setup, MT5Bridge
    import yaml

    if not args.status and not args.symbols and not args.history and not args.test_order:
        # Default: run full diagnostic
        check_mt5_setup()
        return

    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f) or {}

    # Force enable for diagnostics
    if 'mt5' not in config:
        config['mt5'] = {'enabled': True, 'mode': 'execute'}
    config['mt5']['enabled'] = True

    bridge = MT5Bridge(config)
    if not bridge.connected:
        print("Failed to connect to MT5. Run without flags for full diagnostic.")
        sys.exit(1)

    if args.status:
        bridge.print_status()

    if args.symbols:
        print("\n" + "=" * 65)
        print("  ALL CRYPTO SYMBOLS ON THIS BROKER")
        print("=" * 65)
        all_symbols = mt5.symbols_get()
        if all_symbols:
            crypto_keywords = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'BNB',
                             'CRYPTO', 'BITCOIN', 'ETHER', 'COIN']
            found = []
            for s in all_symbols:
                name_upper = s.name.upper()
                if any(kw in name_upper for kw in crypto_keywords):
                    found.append(s)

            if found:
                print(f"  Found {len(found)} crypto-related symbols:\n")
                print(f"  {'Symbol':<20} {'Spread':>8} {'MinLot':>8} {'MaxLot':>10} {'Contract':>10}")
                print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
                for s in sorted(found, key=lambda x: x.name):
                    print(f"  {s.name:<20} {s.spread:>8} {s.volume_min:>8.3f} {s.volume_max:>10.1f} {s.trade_contract_size:>10.3f}")
            else:
                print("  No crypto symbols found on this broker.")
                print("  Recommended brokers with crypto CFDs:")
                print("    - Exness (BTCUSD, ETHUSD)")
                print("    - RoboForex (BTCUSD, ETHUSD)")
                print("    - IC Markets (BTC/USD, ETH/USD)")
                print("    - FP Markets (Crypto CFDs)")
                print("    - Pepperstone (Crypto CFDs)")
        print("=" * 65)

    if args.history > 0:
        print(f"\n  Trade History (last {args.history} days):")
        print("=" * 75)
        trades = bridge.get_trade_history(days=args.history)
        if trades:
            total_pnl = 0
            wins = 0
            losses = 0
            for t in trades:
                pnl = t['profit']
                total_pnl += pnl
                if pnl > 0:
                    wins += 1
                elif pnl < 0:
                    losses += 1

                from datetime import datetime
                dt = datetime.fromtimestamp(t['time']).strftime('%Y-%m-%d %H:%M')
                print(f"  {dt} | {t['symbol']:<12} {t['type']:<5} {t['volume']:.3f} lots "
                      f"@ ${t['price']:,.2f} | P&L: ${pnl:+,.2f}")

            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            print(f"\n  Summary: {total} trades | {wins}W / {losses}L | WR: {wr:.1f}% | Total P&L: ${total_pnl:+,.2f}")
        else:
            print("  No trades found in this period.")
        print("=" * 75)

    if args.test_order:
        print("\n  TEST ORDER (smallest possible size)")
        print("  This will open and immediately close a tiny BTC position")
        confirm = input("  Type 'yes' to proceed: ").strip().lower()
        if confirm == 'yes':
            # Open tiny position
            result = bridge.open_position('BTC', 'LONG', 0, 0.01, 0, "TEST_ORDER")
            print(f"  Open result: {result}")
            if result.get('status') == 'success':
                import time
                time.sleep(2)
                close_result = bridge.close_position('BTC', 0, "TEST_CLOSE")
                print(f"  Close result: {close_result}")
        else:
            print("  Cancelled.")

    bridge.shutdown()


if __name__ == '__main__':
    main()
