"""Direct Alpaca paper-API test — bypasses the bot entirely.

Run this when the bot is silent and you want to know whether the issue
is bot-side (no decisions, gate refusing plans, etc.) or API-side
(auth bad, permissions, account issue).

Usage (on the 4060 or any box with APCA_API_KEY_ID + APCA_API_SECRET_KEY set):
    python scripts/test_alpaca_paper_order.py SPY 1
    python scripts/test_alpaca_paper_order.py "BTC/USD" 0.001
    python scripts/test_alpaca_paper_order.py NVDA 1 --sell

If this places an order successfully -> Alpaca paper is healthy and the
bug is in the bot's decision pipeline (not generating BUY/SELL signals
or refusing all plans at the gates). Check trade_decisions.jsonl for
why decisions are flat.

If this fails -> there's an Alpaca-side issue (auth, account status,
unsupported symbol, fractional restrictions, etc). The error message
points at which.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("symbol", help="e.g. SPY, NVDA, 'BTC/USD'")
    p.add_argument("qty", type=float, help="number of shares / crypto quantity")
    p.add_argument("--sell", action="store_true", help="sell instead of buy")
    p.add_argument("--limit", type=float, default=None,
                   help="limit price (default: market order)")
    args = p.parse_args()

    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        print("[FAIL] APCA_API_KEY_ID / APCA_API_SECRET_KEY not set in env.")
        print("       setx APCA_API_KEY_ID 'PK...' / setx APCA_API_SECRET_KEY '...'")
        print("       open fresh PowerShell, then re-run.")
        return 2

    print(f"[INFO] Key={key[:6]}... ; secret_len={len(secret)}")
    print(f"[INFO] Symbol={args.symbol}  qty={args.qty}  side={'sell' if args.sell else 'buy'}  type={'limit' if args.limit else 'market'}")

    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    print(f"[INFO] Base URL: {base}")

    import requests
    sess = requests.Session()
    sess.headers.update({
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type":        "application/json",
    })

    # 1. Health: account + clock
    try:
        a = sess.get(f"{base}/v2/account", timeout=10)
        if a.status_code != 200:
            print(f"[FAIL] /v2/account HTTP {a.status_code}: {a.text[:300]}")
            return 3
        acc = a.json()
        print(f"[OK]   Account status={acc.get('status')}  equity=${acc.get('equity')}  cash=${acc.get('cash')}  buying_power=${acc.get('buying_power')}")
        if acc.get("trading_blocked"):
            print(f"[FAIL] account.trading_blocked = True ({acc.get('account_blocked', '?')}). Cannot place orders.")
            return 4
        if acc.get("transfers_blocked") or acc.get("trade_suspended_by_user"):
            print(f"[WARN] transfers_blocked or trade_suspended_by_user — may affect orders")
    except Exception as e:
        print(f"[FAIL] /v2/account request failed: {e}")
        return 3

    try:
        c = sess.get(f"{base}/v2/clock", timeout=10)
        if c.status_code == 200:
            ck = c.json()
            print(f"[OK]   Market clock: is_open={ck.get('is_open')}  next_open={ck.get('next_open')}  next_close={ck.get('next_close')}")
            if not ck.get("is_open") and "/" not in args.symbol:
                print("[WARN] Market is CLOSED and you're trying a stock order — Alpaca may reject. Crypto symbols (BTC/USD) work 24/7.")
    except Exception as e:
        print(f"[WARN] /v2/clock failed: {e}")

    # 2. Place the test order
    payload = {
        "symbol":        args.symbol,
        "qty":           str(args.qty),
        "side":          "sell" if args.sell else "buy",
        "type":          "limit" if args.limit else "market",
        "time_in_force": "gtc" if "/" in args.symbol else "day",
    }
    if args.limit:
        payload["limit_price"] = str(args.limit)

    print(f"[INFO] POST /v2/orders payload: {payload}")
    try:
        r = sess.post(f"{base}/v2/orders", json=payload, timeout=15)
    except Exception as e:
        print(f"[FAIL] order request raised: {e}")
        return 5

    print(f"[INFO] HTTP {r.status_code}")
    try:
        body = r.json()
    except Exception:
        body = {"raw": r.text[:500]}
    if r.status_code in (200, 201):
        print(f"[OK]   ORDER ACCEPTED")
        print(f"       order_id={body.get('id')}")
        print(f"       status={body.get('status')}")
        print(f"       symbol={body.get('symbol')}  qty={body.get('qty')}  side={body.get('side')}")
        print(f"       submitted_at={body.get('submitted_at')}")
        print()
        print("If you see this order on the Alpaca dashboard within 10 seconds,")
        print("the API path is healthy and the bug is in the bot's decision")
        print("pipeline (gates refusing plans, scanner not seeing setups,")
        print("LLM returning parse_failures, etc). Check trade_decisions.jsonl.")
        return 0
    else:
        print(f"[FAIL] order rejected by Alpaca: {body}")
        return 6


if __name__ == "__main__":
    raise SystemExit(main())
