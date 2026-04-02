import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
trades = []
with open('logs/trading_journal.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            trades.append(json.loads(line))

today = [t for t in trades if t.get('timestamp','').startswith('2026-04-02')]
today.sort(key=lambda t: t.get('timestamp',''))

print("TOTAL TRADES TODAY:", len(today))
print()

total_bybit = 0
total_delta = 0

for t in today:
    pnl = float(t.get('pnl_usd',0))
    is_delta = '-' in str(t.get('order_id',''))
    ex = 'DELTA' if is_delta else 'BYBIT'
    if is_delta:
        total_delta += pnl
    else:
        total_bybit += pnl

    w = 'WIN ' if pnl > 0 else 'LOSS' if pnl < 0 else 'BE  '
    asset = t.get('asset','?')
    action = t.get('action','?')
    sl = str(t.get('sl_progression','?'))
    exit_r = str(t.get('exit_reason','?'))[:35]
    conf = float(t.get('confidence',0))
    ts = t.get('timestamp','')[11:16]
    print(w, ts, ex.ljust(5), asset, action.ljust(5), "pnl=$" + format(pnl, '+8.2f'), "conf=" + format(conf, '.2f'), "SL=" + sl.ljust(15), "|", exit_r)

print()
print("BYBIT total P&L today: $" + format(total_bybit, '+.2f'))
print("DELTA total P&L today: $" + format(total_delta, '+.2f'))
print("COMBINED: $" + format(total_bybit + total_delta, '+.2f'))
