"""Quick MT5 symbol check."""
import MetaTrader5 as mt5

mt5.initialize(
    path="C:/Program Files/MetaTrader 5/terminal64.exe",
    login=5048969803,
    password="@c0sOyFd",
    server="MetaQuotes-Demo"
)

account = mt5.account_info()
print(f"Account: {account.login} | Balance: ${account.balance:,.2f} | Equity: ${account.equity:,.2f}")

# Check BTC and related symbols
print("\n=== Symbol Status ===")
for sym in ['BTC', 'BTCUSD', 'GBTC', 'ETH', 'ETHUSD', 'ETHE', 'FBTC', 'COIN']:
    info = mt5.symbol_info(sym)
    if info:
        mt5.symbol_select(sym, True)
        tick = mt5.symbol_info_tick(sym)
        bid = tick.bid if tick else 0
        ask = tick.ask if tick else 0
        # trade_mode: 0=disabled, 1=long only, 2=short only, 3=close only, 4=full
        modes = {0: 'DISABLED', 1: 'LONG_ONLY', 2: 'SHORT_ONLY', 3: 'CLOSE_ONLY', 4: 'FULL'}
        mode_str = modes.get(info.trade_mode, str(info.trade_mode))
        print(f"  {sym:12} bid={bid:>12.2f}  ask={ask:>12.2f}  mode={mode_str:12}  path={info.path}")
    else:
        print(f"  {sym:12} NOT FOUND")

# Show all categories
print("\n=== Symbol Categories ===")
symbols = mt5.symbols_get()
paths = {}
for s in symbols:
    parts = s.path.split("\\") if s.path else ['unknown']
    cat = parts[0]
    if cat not in paths:
        paths[cat] = 0
    paths[cat] += 1

for cat, count in sorted(paths.items()):
    print(f"  {cat}: {count} symbols")

# Try to get recent price data for BTC
print("\n=== BTC Recent Prices ===")
import datetime
rates = mt5.copy_rates_from_pos("BTC", mt5.TIMEFRAME_D1, 0, 5)
if rates is not None and len(rates) > 0:
    for r in rates:
        dt = datetime.datetime.fromtimestamp(r['time'])
        print(f"  {dt.strftime('%Y-%m-%d')} O={r['open']:.2f} H={r['high']:.2f} L={r['low']:.2f} C={r['close']:.2f} V={r['tick_volume']}")
else:
    print("  No data available for BTC")

# Check open positions
print("\n=== Open Positions ===")
positions = mt5.positions_get()
if positions:
    for p in positions:
        print(f"  {p.symbol} {'LONG' if p.type==0 else 'SHORT'} {p.volume} lots @ {p.price_open} -> {p.price_current} P&L=${p.profit:+.2f}")
else:
    print("  No open positions")

# Try placing a test order on BTC
print("\n=== Test Order Check ===")
info = mt5.symbol_info("BTC")
if info and info.trade_mode == 4:  # FULL trading mode
    tick = mt5.symbol_info_tick("BTC")
    if tick and tick.ask > 0:
        print(f"  BTC is tradeable! Ask={tick.ask:.2f}")
        # Place minimum size order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "BTC",
            "volume": 1.0,
            "type": mt5.ORDER_TYPE_BUY,
            "price": tick.ask,
            "deviation": 20,
            "magic": 888888,
            "comment": "TEST_EMA8",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result:
            print(f"  Order result: retcode={result.retcode} comment={result.comment}")
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"  ORDER FILLED! Ticket={result.order} Price={result.price}")
        else:
            print(f"  Order send returned None. Last error: {mt5.last_error()}")
    else:
        print(f"  BTC tick not available (market closed?)")
else:
    mode_val = info.trade_mode if info else 'N/A'
    print(f"  BTC trade_mode={mode_val} — not in FULL mode, cannot trade")

mt5.shutdown()
