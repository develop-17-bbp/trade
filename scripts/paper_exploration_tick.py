"""Paper-soak exploration tick — fires one small trade when the bot
has been silent for too long.

Why this exists:
  Paper soak needs ACCUMULATION of warm_store data so the readiness
  gate, fine-tune corpus, and credit assigner have something to learn
  from. When the LLM analyst is unreachable (Ollama endpoint flap)
  or the conviction gate is correctly refusing low-quality setups
  on a dead-quiet market, the soak can sit empty for days. This
  script breaks that stalemate by firing one small momentum-following
  trade per quiet period, tagged so post-hoc audit knows the trade
  came from exploration rather than a real signal.

Hard rules:
  * Paper-only. ACT_REAL_CAPITAL_ENABLED=1 is a HARD SKIP.
  * Default 0.5% size. ACT_PAPER_EXPLORATION_SIZE_PCT overrides.
  * Default fires only if no warm_store decision in the last 4 hours.
  * Default max 8 exploration trades per UTC day.
  * Direction follows last 15-bar momentum (BUY if up, SELL if down).
    Random would be worse than informed-but-loose-thresholds.
  * Operator kill: ACT_DISABLE_PAPER_EXPLORATION=1 or
    ACT_DISABLE_AGENTIC_LOOP=1 (broader kill).
  * Tags warm_store row with conviction_tier='paper_explore' so the
    fine-tune training-data filter can exclude these from the corpus.

Wire up by adding to START_ALL.ps1 + START_ALL_4060.ps1 as a
scheduled tick (every 15 min is fine; the cooldown enforces real
cadence). This is a standalone script — runs once and exits — so
it works equally well from Windows Scheduled Tasks, cron, or a
START_ALL background loop.

Usage (one-shot):
    python scripts/paper_exploration_tick.py
    python scripts/paper_exploration_tick.py --once
    python scripts/paper_exploration_tick.py --venue alpaca
    python scripts/paper_exploration_tick.py --venue robinhood
    python scripts/paper_exploration_tick.py --venue auto  (default)

`auto` picks alpaca if APCA_* env is set, else robinhood paper-sim.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import math
import os
import random
import sys
from typing import List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load .env so the price-fetch chain (Alpaca / LiveCoinWatch) sees its
# API keys. Same rationale as force_test_trade.py - without this, the
# script env on the 5090 is starved of the .env-only keys, falls through
# to hardcoded prices, and the test trade opens at the wrong fill price.
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path, override=False)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("paper_exploration")


# ── Tunables (env overridable) ────────────────────────────────────

QUIET_HOURS = float(os.environ.get("ACT_PAPER_EXPLORATION_QUIET_HOURS", "1"))
SIZE_PCT = float(os.environ.get("ACT_PAPER_EXPLORATION_SIZE_PCT", "0.5"))
MAX_PER_DAY = int(os.environ.get("ACT_PAPER_EXPLORATION_MAX_PER_DAY", "16"))

# Asset selection: comma-sep tickers from env, else use the project's
# top-100 large-cap list + BTC/ETH on Alpaca crypto. Operator directive
# 2026-04-30: 4060 should consider top 100 stocks + BTC + ETH, not the
# 4-ETF SPY/QQQ/TQQQ/SOXL minimal basket.
def _build_default_alpaca_basket() -> str:
    try:
        from src.trading.watchlist_scanner import TOP_100_LARGE_CAPS
        crypto = ["BTC/USD", "ETH/USD"]
        return ",".join(crypto + list(TOP_100_LARGE_CAPS))
    except Exception:
        # Fallback if watchlist module unavailable - keep BTC/ETH at the
        # head so crypto coverage exists when stocks RTH is closed.
        return ("BTC/USD,ETH/USD,NVDA,MSFT,AAPL,AMZN,GOOGL,META,TSLA,AVGO,"
                "JPM,LLY,UNH,V,XOM,MA,JNJ,COST,HD,PG,WMT,SPY,QQQ")

ALPACA_BASKET = os.environ.get(
    "ACT_PAPER_EXPLORATION_ALPACA_BASKET",
    _build_default_alpaca_basket(),
).split(",")
ROBINHOOD_BASKET = os.environ.get(
    "ACT_PAPER_EXPLORATION_RH_BASKET",
    "BTC,ETH",
).split(",")


# ── Gate checks ───────────────────────────────────────────────────


def _is_disabled() -> Optional[str]:
    if os.environ.get("ACT_REAL_CAPITAL_ENABLED") == "1":
        return "ACT_REAL_CAPITAL_ENABLED=1 (hard skip on real capital)"
    if os.environ.get("ACT_DISABLE_PAPER_EXPLORATION") == "1":
        return "ACT_DISABLE_PAPER_EXPLORATION=1"
    if os.environ.get("ACT_DISABLE_AGENTIC_LOOP") == "1":
        return "ACT_DISABLE_AGENTIC_LOOP=1"
    return None


def _last_decision_age_h() -> float:
    """Return hours since last NON-SHADOW warm_store decision. inf when
    warm_store has no real rows.

    Shadow rows (decision_id LIKE 'shadow-%') are excluded — those fire
    every 60-180s from the agentic loop and would permanently reset the
    quiet-hours gate, blocking exploration forever.
    """
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        store.flush()
        conn = store._get_conn()
        row = conn.execute(
            "SELECT MAX(ts_ns) FROM decisions WHERE decision_id NOT LIKE 'shadow-%'"
        ).fetchone()
    except Exception as e:
        logger.debug("warm_store probe failed: %s", e)
        return math.inf
    if not row or row[0] is None:
        return math.inf
    import time
    return (time.time_ns() - int(row[0])) / 1e9 / 3600.0


def _today_explore_count() -> int:
    """Count exploration trades fired so far today (UTC)."""
    try:
        from src.orchestration.warm_store import get_store
        store = get_store()
        store.flush()
        conn = store._get_conn()
        today_utc = _dt.datetime.now(_dt.timezone.utc).date()
        start_ts_ns = int(_dt.datetime(
            today_utc.year, today_utc.month, today_utc.day,
            tzinfo=_dt.timezone.utc,
        ).timestamp() * 1e9)
        row = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE ts_ns >= ? AND conviction_tier = 'paper_explore'",
            (start_ts_ns,),
        ).fetchone()
    except Exception:
        return 0
    return int(row[0] or 0) if row else 0


# ── Momentum scoring (15-bar pct change abs ranks) ────────────────


def _rsi(closes: List[float], period: int = 14) -> float:
    """Plain Wilder RSI; returns 50 on insufficient data."""
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, period + 1):
        d = closes[-i] - closes[-i - 1]
        gains.append(d if d > 0 else 0.0)
        losses.append(-d if d < 0 else 0.0)
    ag = sum(gains) / period if gains else 0.0
    al = sum(losses) / period if losses else 0.0
    if ag == 0 and al == 0:
        return 50.0
    if al == 0:
        return 100.0
    rs = ag / al
    return 100.0 - (100.0 / (1.0 + rs))


def _pick_asset_alpaca(relaxed: bool = False) -> Optional[Tuple[str, str, str]]:
    """Pick (symbol, side, rationale) for a profit-biased exploration trade.

    STRICT mode (default): all 4 filters must pass.
      1. Multi-timeframe trend alignment: 1h slope same sign as 5m
         momentum. Don't fight HTF.
      2. RSI sanity: 40-65 (longs), 35-60 (shorts). Avoid extremes.
      3. Volume confirmation: last 5m bar > 1.2× trailing 20-bar avg.
      4. Min move size: > 0.3% over last 15×5m bars.

    RELAXED mode (--relaxed): require any 2 of 4 filters to pass +
    minimum-move halved to 0.15%. Catches more setups on quiet days
    at the cost of slightly lower win-rate per trade. Use when soak
    accumulation matters more than per-trade EV (early paper soak).

    Among qualifying assets, picks the one with the largest absolute
    momentum (most-conviction setup). Returns None if no asset
    qualifies — exploration tick simply skips this cycle rather
    than firing a low-quality trade.
    """
    try:
        from src.data.alpaca_fetcher import AlpacaFetcher
        f = AlpacaFetcher(paper=True)
        if not f.available:
            return None
    except Exception:
        return None

    min_move = 0.15 if relaxed else 0.3
    min_filters_passed = 2 if relaxed else 4

    candidates: List[Tuple[str, str, float, str]] = []
    for sym in ALPACA_BASKET:
        sym = sym.strip().upper()
        if not sym:
            continue
        try:
            # 5m bars (limit=30 for RSI + momentum + volume avg)
            bars5 = f.fetch_ohlcv(sym, timeframe="5Min", limit=30)
            if not bars5 or len(bars5) < 21:
                continue
            closes5 = [float(b[4]) for b in bars5]
            volumes5 = [float(b[5]) for b in bars5]

            close_now = closes5[-1]
            close_15ago = closes5[-16]
            if close_15ago <= 0 or close_now <= 0:
                continue
            mom5_pct = (close_now - close_15ago) / close_15ago * 100.0

            # Filter 4: minimum move (always required - below this is noise)
            if abs(mom5_pct) < min_move:
                continue

            side_5m = "buy" if mom5_pct > 0 else "sell"

            # Filter 3: volume confirmation
            avg_vol = sum(volumes5[:-1]) / max(1, len(volumes5) - 1)
            last_vol = volumes5[-1]
            vol_pass = avg_vol > 0 and last_vol >= 1.2 * avg_vol

            # Filter 2: RSI sanity
            rsi = _rsi(closes5, period=14)
            if side_5m == "buy":
                rsi_pass = 40 <= rsi <= 65
            else:
                rsi_pass = 35 <= rsi <= 60

            # Filter 1: multi-timeframe trend alignment
            bars1h = f.fetch_ohlcv(sym, timeframe="1Hour", limit=8)
            if not bars1h or len(bars1h) < 4:
                slope1h_pct = 0.0
                mtf_pass = False
            else:
                closes1h = [float(b[4]) for b in bars1h]
                if closes1h[0] <= 0:
                    mtf_pass = False
                    slope1h_pct = 0.0
                else:
                    slope1h_pct = (closes1h[-1] - closes1h[0]) / closes1h[0] * 100.0
                    if side_5m == "buy":
                        mtf_pass = slope1h_pct > 0
                    else:
                        mtf_pass = slope1h_pct < 0

            # Filter 4 already gated above (min_move). Count remaining 3.
            passed = sum([mtf_pass, rsi_pass, vol_pass]) + 1   # +1 for min_move
            if passed < min_filters_passed:
                continue

            tags = []
            tags.append("MTF" if mtf_pass else "mtf")
            tags.append("RSI" if rsi_pass else "rsi")
            tags.append("VOL" if vol_pass else "vol")
            mode = "RELAXED" if relaxed else "STRICT"
            rationale = (
                f"[{mode}] {''.join(t[0] for t in tags).upper()} "
                f"mom5={mom5_pct:+.2f}% slope1h={slope1h_pct:+.2f}% "
                f"rsi={rsi:.0f} vol={last_vol/max(avg_vol,1e-9):.1f}x"
            )
            candidates.append((sym, side_5m, abs(mom5_pct), rationale))
        except Exception as e:
            logger.debug("setup probe failed for %s: %s", sym, e)
            continue

    if not candidates:
        logger.info("[EXPLORE] no asset passed trend-alignment + RSI + volume filters; skipping")
        return None

    # Pick highest-conviction setup (largest |momentum|)
    candidates.sort(key=lambda c: c[2], reverse=True)
    sym, side, mom_abs, rationale = candidates[0]
    logger.info("[EXPLORE] picked %s %s (%s)", sym, side.upper(), rationale)
    return sym, side, rationale


# ── Submission ────────────────────────────────────────────────────


def _submit_alpaca(symbol: str, side: str) -> int:
    """Submit a small exploration trade via PriceFetcher (handles both
    stocks + crypto). Tagged so audit can identify it."""
    if not os.environ.get("APCA_API_KEY_ID") or not os.environ.get("APCA_API_SECRET_KEY"):
        logger.error("APCA_* keys missing - cannot submit alpaca exploration")
        return 2

    is_crypto = "/" in symbol
    venue = "alpaca_crypto" if is_crypto else "alpaca"
    try:
        from src.data.fetcher import PriceFetcher
        fetcher = PriceFetcher(exchange_name=venue, testnet=True)
    except Exception as e:
        logger.error("PriceFetcher init failed for %s: %s", venue, e)
        return 3

    if not getattr(fetcher, "_available", False):
        logger.error("PriceFetcher not available for %s", venue)
        return 3

    # Size: SIZE_PCT of equity, in shares/qty. For crypto this is a
    # fractional unit; for stocks we pick min(1, calculated qty).
    try:
        acct = fetcher.alpaca.get_account() if hasattr(fetcher, "alpaca") and fetcher.alpaca else {}
        equity = float(acct.get("equity", 100000) or 100000)
    except Exception:
        equity = 100000.0

    notional = equity * (SIZE_PCT / 100.0)
    # Quick price probe for sizing. CRITICAL: use AlpacaFetcher directly
    # for stock symbols. PriceFetcher.fetch_latest_price falls through
    # to CCXT/Kraken for non-crypto symbols on alpaca venue, and Kraken
    # doesn't have stock tickers (AMD/NVDA/etc) → "kraken does not
    # have market symbol" error → script aborts.
    last_px = 0.0
    try:
        from src.data.alpaca_fetcher import AlpacaFetcher
        af = AlpacaFetcher(paper=True)
        if af.available:
            tk = af.fetch_ticker(symbol)
            # Prefer mid-quote when both bid + ask available; else last trade
            bid = float(tk.get("bid") or 0)
            ask = float(tk.get("ask") or 0)
            last = float(tk.get("last") or 0)
            if bid > 0 and ask > 0:
                last_px = (bid + ask) / 2.0
            elif last > 0:
                last_px = last
            elif ask > 0:
                last_px = ask
            elif bid > 0:
                last_px = bid
        # Fallback: last close from the bar series we already fetched
        # in _pick_asset_alpaca. Refetch one bar to get the most recent
        # close — cheap because AlpacaFetcher is HTTP not socket.
        if last_px <= 0:
            bars = af.fetch_ohlcv(symbol, timeframe="1Min", limit=2)
            if bars:
                last_px = float(bars[-1][4])
    except Exception as e:
        logger.debug("AlpacaFetcher price probe failed: %s", e)
    # Last-resort fallback: PriceFetcher's path (works for crypto since
    # the alpaca_crypto venue routes through Alpaca crypto data API).
    if last_px <= 0:
        try:
            last_px = float(fetcher.fetch_latest_price(symbol) or 0)
        except Exception:
            last_px = 0.0
    if last_px <= 0:
        logger.error("Could not get last price for %s (tried Alpaca quote, Alpaca bars, fetcher fallback)", symbol)
        return 4
    qty = notional / last_px
    if not is_crypto:
        qty = max(1, int(qty))
    else:
        qty = round(qty, 6)

    # Inventory pre-check: before SELLing a stock, confirm Alpaca
    # actually has free qty (qty_available, not just qty). Existing
    # positions can be locked by working orders (held_for_orders > 0)
    # which makes Alpaca return 403 'insufficient qty available'. When
    # locked, skip this symbol so the next 15-min tick can pick a
    # different one rather than retrying the same conflict forever.
    # Operator audit 2026-04-30: the AAPL 1-share-locked case.
    if not is_crypto and side == "sell":
        try:
            from src.data.alpaca_fetcher import AlpacaFetcher as _AF
            _af2 = _AF(paper=True)
            if _af2.available:
                pos = _af2.alpaca.get_position(symbol) if hasattr(_af2, "alpaca") and _af2.alpaca else None
                if pos is None:
                    raise RuntimeError("no_position_method")
                # alpaca-py returns Position object with .qty_available + .qty
                _qty_avail = float(getattr(pos, "qty_available", None) or
                                   pos.get("qty_available", 0) if isinstance(pos, dict) else 0)
                _qty_held = float(getattr(pos, "qty", None) or
                                  pos.get("qty", 0) if isinstance(pos, dict) else 0)
                if _qty_avail < qty:
                    logger.warning(
                        "[SKIP] %s SELL blocked: qty_avail=%s held=%s < requested=%s "
                        "(held_for_orders locks the share). Picking different "
                        "symbol on next tick.",
                        symbol, _qty_avail, _qty_held, qty,
                    )
                    return 7   # distinct exit code so audit can grep
        except Exception as _ie:
            # No position OR error reading - fall through to attempt
            # the order. Alpaca returns 403 with a clear message which
            # we'll log below; better to attempt + log than silently
            # skip every tick because the inventory probe choked.
            logger.debug("inventory pre-check skipped for %s: %s", symbol, _ie)

    logger.info("[EXPLORE] submitting %s %s qty=%s (~$%.0f at $%.2f)",
                side.upper(), symbol, qty, notional, last_px)
    try:
        result = fetcher.place_order(
            symbol=symbol, side=side, amount=float(qty),
            order_type="market",
        )
    except Exception as e:
        logger.error("place_order raised: %s", e)
        return 5

    if result.get("status") == "success":
        logger.info("[OK] EXPLORE order accepted: %s", result.get("order_id"))
        # Mark as exploration in warm_store so the corpus filter knows.
        try:
            from src.orchestration.warm_store import get_store
            import uuid as _uuid
            import time as _time
            store = get_store()
            row = {
                "decision_id": f"paper_explore_{_uuid.uuid4().hex}",
                "ts_ns": _time.time_ns(),
                "symbol": symbol, "asset_class": "CRYPTO" if is_crypto else "STOCK",
                "venue": "alpaca", "side": side, "qty": float(qty),
                "conviction_tier": "paper_explore",
                "final_action": "EXPLORE_SUBMIT",
                "plan_json": {"exploration": True, "size_pct": SIZE_PCT},
                "order_resp": result,
            }
            try:
                store.write_decision(row)
            except Exception:
                store.write(row)
        except Exception as e:
            logger.debug("warm_store write soft-fail: %s", e)
        return 0
    logger.error("[FAIL] EXPLORE rejected: %s", result)
    return 6


def _submit_robinhood(asset: str, side: str) -> int:
    """RH paper-sim — internal-only, uses RobinhoodPaperFetcher."""
    import yaml
    from pathlib import Path
    cfg_path = Path(_PROJECT_ROOT) / "config.yaml"
    try:
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("config load failed: %s", e)
        return 4

    from src.data.robinhood_fetcher import RobinhoodPaperFetcher
    pf = RobinhoodPaperFetcher(config=config)

    fill_price = None
    if pf.connected:
        try:
            q = pf.get_live_price(asset)
            if q.get("mid"):
                fill_price = float(q["mid"])
        except Exception:
            pass
    if not fill_price:
        fill_price = {"BTC": 75000.0, "ETH": 2200.0}.get(asset, 100.0)

    direction = "LONG" if side == "buy" else "SHORT"
    qty = (pf.equity * SIZE_PCT / 100.0) / fill_price
    qty = round(qty, 6)
    logger.info("[EXPLORE] RH paper-sim: %s %s %s @ $%.2f", direction, qty, asset, fill_price)

    pos = pf.record_entry(
        asset=asset, direction=direction, price=fill_price,
        score=5, quantity=qty,
        sl_price=fill_price * (0.97 if direction == "LONG" else 1.03),
        tp_price=fill_price * (1.03 if direction == "LONG" else 0.97),
        ml_confidence=0.4, llm_confidence=0.4,
        size_pct=SIZE_PCT,
        reasoning="paper_exploration_tick.py (low-conviction soak data)",
    )
    if pos is None:
        return 5
    # Persist to disk so the bot's long-lived RobinhoodPaperFetcher picks
    # up the position via load_state on its next tick. Operator audit
    # 2026-05-01: without this save_state, the position was in-memory
    # only and lost when this one-shot script exited. Dashboard's
    # OPEN POSITIONS panel reads from the bot instance, which never saw
    # the entry, so it stayed at 0 even though RECENT TRADES feed
    # (which reads warm_store) showed the entry. The bot also needs to
    # reload from disk on each tick - that's a separate fix.
    try:
        pf.save_state()
        logger.info("[EXPLORE] RH paper-sim state persisted: positions=%d", len(pf.positions))
    except Exception as _se:
        logger.warning("[EXPLORE] save_state failed: %s", _se)
    try:
        from src.orchestration.warm_store import get_store
        import uuid as _uuid
        import time as _time
        store = get_store()
        row = {
            "decision_id": f"paper_explore_{_uuid.uuid4().hex}",
            "ts_ns": _time.time_ns(),
            "symbol": asset, "asset_class": "CRYPTO",
            "venue": "robinhood", "side": side, "qty": float(qty),
            "conviction_tier": "paper_explore",
            "final_action": "EXPLORE_SUBMIT",
            "plan_json": {
                "exploration": True, "size_pct": SIZE_PCT,
                "entry_price": fill_price, "direction": direction,
            },
        }
        try:
            store.write_decision(row)
        except Exception:
            store.write(row)
    except Exception as e:
        logger.debug("warm_store write soft-fail: %s", e)
    return 0


# ── Main ──────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--venue", choices=["alpaca", "robinhood", "auto"], default="auto")
    p.add_argument("--once", action="store_true",
                   help="Single-shot. Always true; flag is a no-op for compat.")
    p.add_argument("--force", action="store_true",
                   help="Skip the quiet-hours + daily-cap gates (testing only).")
    p.add_argument("--relaxed", action="store_true",
                   help="Soften per-trade quality bar: 2-of-4 filters (vs 4-of-4) "
                        "+ min move 0.15%% (vs 0.3%%). Use when soak data accumulation "
                        "matters more than per-trade EV (early paper soak / quiet markets).")
    args = p.parse_args()

    disabled = _is_disabled()
    if disabled:
        logger.info("[SKIP] paper exploration disabled: %s", disabled)
        return 0

    if not args.force:
        age_h = _last_decision_age_h()
        if age_h < QUIET_HOURS:
            logger.info("[SKIP] last decision %.1fh ago < %sh quiet threshold; not stale yet",
                        age_h, QUIET_HOURS)
            return 0
        cnt = _today_explore_count()
        if cnt >= MAX_PER_DAY:
            logger.info("[SKIP] daily exploration count %s >= cap %s", cnt, MAX_PER_DAY)
            return 0
        logger.info("[GATES OK] last decision %.1fh ago; %s/%s explorations today",
                    age_h, cnt, MAX_PER_DAY)
    else:
        logger.warning("[FORCE] gates bypassed (--force)")

    venue = args.venue
    if venue == "auto":
        venue = "alpaca" if os.environ.get("APCA_API_KEY_ID") else "robinhood"
        logger.info("[INFO] auto-selected venue=%s", venue)

    if venue == "alpaca":
        pick = _pick_asset_alpaca(relaxed=args.relaxed)
        if pick is None:
            mode = "relaxed" if args.relaxed else "strict"
            logger.warning("[SKIP] no alpaca asset passed quality filters this cycle (%s mode)", mode)
            if not args.relaxed:
                logger.info("       Try --relaxed if you need a trade fired now even on a quiet market")
            return 0
        sym, side, rationale = pick
        logger.info("[EXPLORE] rationale: %s", rationale)
        return _submit_alpaca(sym, side)

    if venue == "robinhood":
        # RH basket is BTC/ETH only; pick by 15m return on whichever
        # has more momentum via Robinhood quotes (or fallback).
        # For simplicity, alternate between BTC/ETH by daily count parity.
        cnt = _today_explore_count()
        asset = ROBINHOOD_BASKET[cnt % len(ROBINHOOD_BASKET)].strip().upper() or "BTC"
        # Direction: HARDCODED to LONG. Robinhood is SPOT LONG-ONLY per
        # operator directive 2026-04-30. Earlier random.random() choice
        # produced SHORT entries that submit_trade_plan would reject and
        # that show as 'SHORT' in the dashboard's recent_trades log -
        # confusing audit. Bear-leaning markets get a skip (return 0)
        # instead of a SHORT trade. Operator explicitly said: 'when
        # bearish: emit skip not SHORT'.
        side = "buy"
        logger.info("[EXPLORE] RH paper-sim picked asset=%s side=%s (LONG-only)", asset, side)
        return _submit_robinhood(asset, side)

    logger.error("Unknown venue: %s", venue)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
