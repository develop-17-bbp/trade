
import json
import requests
j = json.load(open('logs/trading_journal.json'))
r = requests.get('https://testnet.binance.vision/api/v3/ticker/price').json()
prices = {i['symbol']: float(i['price']) for i in r if 'symbol' in i}
u_pnl = 0
for t in j:
    sym = t['asset'].replace('/', '')
    if sym not in prices and sym.endswith('T'):
        sym = sym[:-1]
    p = prices.get(sym, prices.get(sym + 'T', prices.get(t['asset'].replace('/',''), 0)))
    if p and t['status'] == 'OPEN':
        pnl = (p - t['price']) * t['quantity'] if t['side'].lower() == 'buy' else (t['price'] - p) * t['quantity']
        u_pnl += pnl
print(f'Unrealized PNL for {len(j)} open trades: ')

