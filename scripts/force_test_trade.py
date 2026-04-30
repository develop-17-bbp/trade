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

# Ensure project root is on sys.path so `from src.X` works when this
# script is run as `python scripts/force_test_trade.py ...` from the
# repo root. Without this, the 5090's Python (different env from the
# 4060's) raises ModuleNotFoundError: No module named 'src' on the
# robinhood path because that's the first import that happens before
# any sibling module triggers path setup.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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
    p.add_argument("--venue", choices=["alpaca", "robinhood"], default="alpaca",
                   help="Target venue. alpaca (default) hits Alpaca paper API; "
                        "robinhood writes to the bot's internal paper-sim state.")
    p.add_argument("--target-pct", type=float, default=1.0,
                   help="Take-profit target as %% above (LONG) / below (SHORT) entry. "
                        "Default 1%%. The bot's position-monitor auto-closes when TP hits.")
    p.add_argument("--stop-pct", type=float, default=5.0,
                   help="Stop-loss as %% below (LONG) / above (SHORT) entry. Default 5%%.")
    args = p.parse_args()

    asset = args.asset.upper().strip()
    is_crypto = asset in ("BTC", "ETH", "BTC/USD", "ETH/USD")

    if args.venue == "robinhood":
        # Robinhood paper-sim path - doesn't need APCA keys, doesn't
        # talk to Robinhood servers.
        if not is_crypto:
            logger.error("Robinhood paper-sim only supports crypto (BTC, ETH)")
            return 2
        return _force_robinhood_paper(asset, args.side, args.qty,
                                       target_pct=args.target_pct,
                                       stop_pct=args.stop_pct)

    # Alpaca path requires APCA keys
    if not os.environ.get("APCA_API_KEY_ID") or not os.environ.get("APCA_API_SECRET_KEY"):
        logger.error("APCA_API_KEY_ID / APCA_API_SECRET_KEY missing from env")
        logger.error("Set them via setx and open a fresh PowerShell")
        return 2

    if is_crypto:
        return _force_crypto(asset, args.side, args.qty, args.limit)
    return _force_stock(asset, args.side, int(args.qty), args.limit)


def _force_robinhood_paper(asset: str, side: str, qty: float,
                            target_pct: float = 1.0,
                            stop_pct: float = 5.0) -> int:
    """Robinhood paper-sim: writes a synthetic position to the bot's
    RobinhoodPaperFetcher state. Visible in the bot's dashboard +
    logs/robinhood_paper_state.json. Does NOT touch Robinhood servers.

    Operator-configured TP/SL: defaults to TP=+1% / SL=-5% (LONG).
    The bot's existing position-monitor (when running) auto-closes
    the position when live price hits TP or SL — i.e. fire-and-hold
    until +1% gain (or -5% drawdown if it goes the wrong way first).

    Use this on the 5090 to verify the paper-sim pipeline writes state
    correctly AND to simulate "buy spot, hold for 1% profit" behavior.
    """
    import yaml
    from pathlib import Path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    cfg_path = PROJECT_ROOT / "config.yaml"
    try:
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"[FAIL] could not load config.yaml: {e}")
        return 4

    from src.data.robinhood_fetcher import RobinhoodPaperFetcher
    pf = RobinhoodPaperFetcher(config=config)

    # Pull live price. Try sources in priority order:
    #   1. RH client mid (best — same data the bot's monitor uses if RH connected)
    #   2. Alpaca crypto endpoint (almost always reachable; APCA keys in env)
    #   3. LiveCoinWatch (broad crypto coverage, separate API)
    #   4. Hardcoded approx (LAST RESORT — produces synthetic-fill warnings).
    # The hardcoded fallback was causing test entries to open at $75K
    # while live BTC was $76K+, making positions show $0 unrealized PnL
    # forever instead of an immediate +1.5% gain. Real-price fills
    # match what the bot's position-monitor sees on the next tick, so
    # PnL accounting is honest.
    fill_price = None
    if pf.connected:
        try:
            quote = pf.get_live_price(asset)
            if quote.get("mid"):
                fill_price = float(quote["mid"])
                logger.info(f"[PRICE] tier-1 RH mid: ${fill_price:,.2f}")
        except Exception as e:
            logger.debug(f"RH live price failed: {e}")
    if not fill_price:
        try:
            from src.data.alpaca_fetcher import AlpacaFetcher
            af = AlpacaFetcher(paper=True)
            if af.available:
                # Crypto symbols on Alpaca need slash form: BTC -> BTC/USD
                alpaca_sym = asset if "/" in asset else f"{asset}/USD"
                tk = af.fetch_ticker(alpaca_sym)
                bid = float(tk.get("bid") or 0)
                ask = float(tk.get("ask") or 0)
                last = float(tk.get("last") or 0)
                if bid > 0 and ask > 0:
                    fill_price = (bid + ask) / 2.0
                elif last > 0:
                    fill_price = last
                if fill_price:
                    logger.info(f"[PRICE] tier-2 Alpaca crypto: ${fill_price:,.2f}")
        except Exception as e:
            logger.debug(f"Alpaca crypto price probe failed: {e}")
    if not fill_price:
        try:
            from src.data.livecoinwatch_fetcher import LiveCoinWatchFetcher
            import os as _os
            lcw_key = _os.environ.get("LIVECOINWATCH_API_KEY", "")
            if lcw_key:
                lcw = LiveCoinWatchFetcher(api_key=lcw_key)
                px = lcw.get_price(asset)
                if px:
                    fill_price = float(px)
                    logger.info(f"[PRICE] tier-3 LiveCoinWatch: ${fill_price:,.2f}")
        except Exception as e:
            logger.debug(f"LiveCoinWatch price probe failed: {e}")
    if not fill_price:
        fill_price = {"BTC": 75000.0, "ETH": 2200.0}.get(asset, 100.0)
        logger.warning(
            f"[WARN] all live-price sources failed - using SYNTHETIC fallback ${fill_price} for {asset}. "
            "Position will show wrong unrealized PnL until next monitor tick. "
            "Set ROBINHOOD_API_KEY or LIVECOINWATCH_API_KEY for real fills."
        )

    direction = "LONG" if side == "buy" else "SHORT"
    if direction == "LONG":
        tp_price = fill_price * (1.0 + target_pct / 100.0)
        sl_price = fill_price * (1.0 - stop_pct / 100.0)
    else:
        tp_price = fill_price * (1.0 - target_pct / 100.0)
        sl_price = fill_price * (1.0 + stop_pct / 100.0)

    logger.info(f"[INFO] RH paper-sim: {direction} {qty} {asset} @ ~${fill_price:,.2f}")
    logger.info(f"       TP={tp_price:,.2f} ({target_pct:+.1f}%)  SL={sl_price:,.2f} ({-stop_pct:+.1f}%)")
    pos = pf.record_entry(
        asset=asset, direction=direction, price=fill_price,
        score=10, quantity=qty,
        sl_price=sl_price, tp_price=tp_price,
        ml_confidence=0.5, llm_confidence=0.5,
        size_pct=1.0,
        reasoning=f"force_test_trade.py: spot-long hold to +{target_pct:.1f}% gain",
    )
    if pos is None:
        logger.error("[FAIL] record_entry returned None - check RobinhoodPaperFetcher logs")
        return 5
    # Persist to logs/robinhood_paper_state.json so the bot's running
    # fetcher (which now calls load_state on init) sees this position
    # on next tick / next restart and can monitor it for TP/SL auto-close.
    # Without this, the position lives only in this script's in-memory
    # fetcher and dies when the script exits.
    try:
        pf.save_state()
        logger.info("       state persisted to logs/robinhood_paper_state.json")
    except Exception as e:
        logger.warning(f"       state persistence soft-fail: {e}")
    logger.info(f"[OK]   RH paper position recorded:")
    logger.info(f"       trade_id={getattr(pos, 'trade_id', '?')}")
    logger.info(f"       asset={getattr(pos, 'asset', '?')}")
    logger.info(f"       direction={getattr(pos, 'direction', '?')}")
    logger.info(f"       entry_price=${getattr(pos, 'entry_price', 0):,.2f}")
    logger.info(f"       qty={getattr(pos, 'quantity', '?')}")
    logger.info(f"       tp_price=${getattr(pos, 'tp_price', tp_price):,.2f}")
    logger.info(f"       sl_price=${getattr(pos, 'sl_price', sl_price):,.2f}")
    logger.info(f"       Position is now in logs/robinhood_paper_state.json (will appear in bot's dashboard).")
    logger.info(f"       It does NOT appear on Robinhood's app or website - this is internal sim only.")
    logger.info(f"       The bot's position-monitor will auto-close when price reaches TP (+{target_pct:.1f}%) or SL (-{stop_pct:.1f}%).")
    logger.info(f"       Until then it holds as spot {direction}.")
    return 0


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
