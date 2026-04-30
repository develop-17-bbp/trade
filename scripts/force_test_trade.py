"""Force a single test paper trade through the bot's full executor path.

Use this when the bot has been running for hours without trades and
you want to know whether the pipeline (conviction gate -> authority
-> executor -> Alpaca API -> warm_store) is healthy or broken.

Stocks (NVDA / SPY / etc) + crypto (BTC / ETH on Alpaca):
    python scripts/force_test_trade.py NVDA buy 1
    python scripts/force_test_trade.py SPY sell 1
    python scripts/force_test_trade.py BTC buy 0.001
    python scripts/force_test_trade.py ETH buy 0.01

What it does:
  1. Validates env (APCA_* set, market hours for stocks, etc.)
  2. Constructs the ACTUAL AlpacaExecutor / AlpacaClient the bot uses
  3. Submits a paper order
  4. Reports order_id + writes a warm_store decision row tagged
     'force_test' so the audit trail shows it wasn't a bot-generated
     trade

If this places an order successfully -> pipeline is healthy. The
"no trades" issue is the bot's decision logic (conviction gate too
strict for current setups, scanner not firing, LLM returning
parse_failures, etc).

If this fails -> the failure mode points at the actual bug.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("force_test_trade")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("asset", help="e.g. NVDA, SPY, BTC, ETH")
    p.add_argument("side", choices=["buy", "sell"])
    p.add_argument("qty", type=float, help="Number of shares (stocks) or quantity (crypto)")
    p.add_argument("--limit", type=float, default=None, help="Optional limit price")
    args = p.parse_args()

    asset = args.asset.upper().strip()
    is_crypto = asset in ("BTC", "ETH", "BTC/USD", "ETH/USD")

    if not os.environ.get("APCA_API_KEY_ID") or not os.environ.get("APCA_API_SECRET_KEY"):
        logger.error("APCA_API_KEY_ID / APCA_API_SECRET_KEY missing from env")
        logger.error("Set them via setx and open a fresh PowerShell")
        return 2

    if is_crypto:
        return _force_crypto(asset, args.side, args.qty, args.limit)
    return _force_stock(asset, args.side, int(args.qty), args.limit)


def _force_stock(asset: str, side: str, qty: int, limit_price: float | None) -> int:
    """Stocks path goes through AlpacaExecutor (RTH-gated)."""
    from src.utils.market_hours import is_us_market_open
    from src.trading.alpaca_executor import AlpacaExecutor

    if not is_us_market_open():
        logger.error("[FAIL] NYSE is currently CLOSED. Stocks orders cannot")
        logger.error("       fire outside RTH (13:30-20:00 UTC, weekdays).")
        logger.error("       Try a crypto symbol (BTC / ETH) which trades 24/7,")
        logger.error("       OR wait for next RTH open.")
        return 3

    ex = AlpacaExecutor(paper=True)
    if not ex.available:
        logger.error("[FAIL] AlpacaExecutor.available=False (creds rejected by /v2/account)")
        return 4

    health = ex.health()
    logger.info(f"[OK]   AlpacaExecutor health: {health}")

    logger.info(f"[INFO] Submitting test order: {side.upper()} {qty} x {asset}")
    # `intent` kwarg is only present in the operator's local edit on
    # some boxes — pass it via **kwargs only when the signature accepts
    # it so this script works against any committed version of
    # AlpacaExecutor.submit_order.
    import inspect
    extra: dict = {}
    try:
        if "intent" in inspect.signature(ex.submit_order).parameters:
            extra["intent"] = "open"
    except (TypeError, ValueError):
        pass
    res = ex.submit_order(
        symbol=asset, side=side, qty=qty,
        limit_price=limit_price,
        conviction_tier="force_test",
        plan={
            "asset": asset, "direction": "LONG" if side == "buy" else "SHORT",
            "size_pct": 1.0, "reasoning": "force_test_trade.py diagnostic",
            "force_test": True,
        },
        **extra,
    )
    if res.submitted:
        logger.info(f"[OK]   ORDER ACCEPTED order_id={res.order_id} decision_id={res.decision_id}")
        logger.info("       Check Alpaca paper dashboard - should appear within 5s.")
        return 0
    logger.error(f"[FAIL] ORDER REJECTED reason={res.reason}")
    logger.error("       This rejection reason points at which gate is blocking trades.")
    return 5


def _force_crypto(asset: str, side: str, qty: float, limit_price: float | None) -> int:
    """Crypto path goes through PriceFetcher.place_order -> AlpacaClient."""
    from src.data.fetcher import PriceFetcher

    fetcher = PriceFetcher(exchange_name="alpaca_crypto", testnet=True)
    if not fetcher._available:
        logger.error("[FAIL] PriceFetcher not available for alpaca_crypto - check creds")
        return 4

    # Alpaca crypto symbol = "BTC/USD" / "ETH/USD"
    if "/" not in asset:
        asset_with_slash = f"{asset}/USD"
    else:
        asset_with_slash = asset

    logger.info(f"[INFO] Submitting crypto test order: {side.upper()} {qty} x {asset_with_slash}")
    result = fetcher.place_order(
        symbol=asset_with_slash, side=side, amount=qty,
        order_type="limit" if limit_price else "market",
        price=limit_price,
    )
    if result.get("status") == "success":
        logger.info(f"[OK]   ORDER ACCEPTED order_id={result.get('order_id')}")
        logger.info("       Check Alpaca paper dashboard - should appear within 5s.")
        logger.info(f"       Full response: {result}")
        return 0
    logger.error(f"[FAIL] ORDER REJECTED: {result}")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
