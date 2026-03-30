"""Main Trading Loop - EMA Crossover + LLM Decisions on Bybit."""

import os
import sys
import time
import yaml
import logging
from dotenv import load_dotenv

# Force trade/ as the ONLY project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Nuke sys.path and rebuild with only trade/ and stdlib
_stdlib = [p for p in sys.path if 'site-packages' in p or 'DLLs' in p or 'Lib' in p or p.endswith('.zip')]
sys.path = [PROJECT_ROOT] + _stdlib
os.chdir(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'), override=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Use importlib with ABSOLUTE paths to guarantee correct files
import importlib.util
def _load(name, rel_path):
    full = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"{full} not found")
    spec = importlib.util.spec_from_file_location(name, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # Register to prevent re-import from wrong path
    spec.loader.exec_module(mod)
    return mod

# Print debug to see what path resolves to
print(f"  [PATH] PROJECT_ROOT={PROJECT_ROOT}")
print(f"  [PATH] fetcher.py exists: {os.path.exists(os.path.join(PROJECT_ROOT, 'src/data/fetcher.py'))}")

_f = _load('_trade_fetcher', 'src/data/fetcher.py')
_l = _load('_trade_llm', 'src/ai/llm_provider.py')
_s = _load('_trade_strategy', 'src/trading/strategy.py')
_e = _load('_trade_executor', 'src/trading/executor1.py')
_j = _load('_trade_journal', 'src/monitoring/journal1.py')

# Get correct classes
DataFetcher = getattr(_f, 'DataFetcher', getattr(_f, 'PriceFetcher', None))
LLMProvider = _l.LLMProvider
EMAStrategy = _s.EMAStrategy
TradeExecutor = _e.TradeExecutor
TradeJournal = _j.TradeJournal

if DataFetcher is None:
    print(f"  [ERROR] No DataFetcher or PriceFetcher found in {_f.__file__}")
    print(f"  [ERROR] Available: {[x for x in dir(_f) if x[0].isupper()]}")
    sys.exit(1)


LLM_PROMPT = """You are a crypto trading AI. Analyze this {symbol} 1-minute data and decide: LONG, SHORT, or FLAT.

=== MARKET DATA (Last 20 candles) ===
{candle_table}

=== CURRENT STATE ===
Price: ${price:,.2f} | EMA(8): ${ema:,.2f} | ATR(14): ${atr:,.2f}
EMA direction: {ema_dir} | Price vs EMA: {price_vs_ema}

=== TREND DETECTION ===
>>> {trend_bias} <<<

=== EMA CROSSOVER RULES ===
ENTRY RULES:
- LONG (CALL): Previous candle has *CROSS* AND current candle ENTIRELY ABOVE EMA AND EMA RISING
- SHORT (PUT): Previous candle has *CROSS* AND current candle ENTIRELY BELOW EMA AND EMA FALLING
- If NO *CROSS* marker in recent candles -> FLAT
STOP-LOSS: Use dynamic trailing SL. Max giveback 30% of peak profit. NEVER widen SL.

=== RESPOND WITH VALID JSON ONLY ===
{{"action": "LONG" or "SHORT" or "FLAT", "order_type": "market" or "limit", "confidence": 0.0 to 1.0, "position_size_pct": 2.0 to 10.0, "limit_price": 0, "stop_loss_price": 0.0, "reasoning": "cite candle numbers and *CROSS* markers"}}"""


def build_prompt(symbol, df, strategy):
    """Build LLM prompt with candle table and indicators."""
    price = df["close"].iloc[-1]
    ema = df["ema8"].iloc[-1]
    atr = df["atr14"].iloc[-1] if not df["atr14"].isna().iloc[-1] else 0
    ema_dir = "RISING" if df["ema_rising"].iloc[-1] else "FALLING"
    price_vs = "ABOVE" if price > ema else "BELOW"

    trend_5 = (price - df["close"].iloc[-6]) / df["close"].iloc[-6] * 100 if len(df) >= 6 else 0

    if trend_5 < -0.3 and price < ema:
        trend_bias = "DOWNTREND DETECTED -> FAVOR SHORT (PUT)"
    elif trend_5 > 0.3 and price > ema:
        trend_bias = "UPTREND DETECTED -> FAVOR LONG (CALL)"
    else:
        trend_bias = "SIDEWAYS/UNCLEAR -> Only trade on clear *CROSS* pattern"

    candle_table = strategy.build_candle_table(df, n=20)

    return LLM_PROMPT.format(
        symbol=symbol, candle_table=candle_table,
        price=price, ema=ema, atr=atr,
        ema_dir=ema_dir, price_vs_ema=price_vs,
        trend_bias=trend_bias,
    )


def main():
    # Load config (same path as main.py)
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize components
    fetcher = DataFetcher(config)
    llm = LLMProvider(config)
    strategy = EMAStrategy(config)
    journal = TradeJournal()

    equity = fetcher.get_equity()
    executor = TradeExecutor(fetcher, equity=equity if equity > 0 else 100000)

    assets = config.get("assets", ["BTC/USDT:USDT", "ETH/USDT:USDT"])
    timeframe = config.get("timeframe", "1m")
    poll = config.get("poll_interval_sec", 5)
    bar_count = 0

    print("=" * 60)
    print("  EMA CROSSOVER + LLM TRADING SYSTEM")
    print(f"  Assets: {', '.join(assets)}")
    print(f"  Equity: ${executor.equity:,.2f}")
    print(f"  LLM: {'ONLINE' if llm.is_alive() else 'OFFLINE'}")
    print("=" * 60)

    try:
        while True:
            bar_count += 1
            # Update equity from exchange
            equity = fetcher.get_equity()
            if equity > 0:
                executor.equity = equity
            ret = (executor.equity - executor.initial_equity) / executor.initial_equity * 100
            n_pos = len(executor.positions)
            print(f"\n  [BAR {bar_count}] Equity: ${executor.equity:,.2f} | "
                  f"Return: {ret:+.2f}% | Positions: {n_pos}")

            # Risk check
            ok, reason = executor.check_risk_limits()
            if not ok:
                print(f"  [RISK] {reason}")
                time.sleep(60)
                continue

            for symbol in assets:
                asset = symbol.split("/")[0]  # BTC or ETH

                # Fetch candle data with indicators
                df = fetcher.fetch_ohlcv(symbol, timeframe)
                if df is None or len(df) < 30:
                    continue

                price = df["close"].iloc[-1]
                ema = df["ema8"].iloc[-1]
                atr = df["atr14"].iloc[-1] if not df["atr14"].isna().iloc[-1] else 0
                ema_dir = "RISING" if df["ema_rising"].iloc[-1] else "FALLING"

                # Display current state
                has_cross = "*CROSS*" if df["cross"].iloc[-2] else ""
                print(f"  [{asset}] ${price:,.2f} | EMA: ${ema:,.2f} | {ema_dir} | "
                      f"ATR: ${atr:,.2f} {has_cross}")

                # ====== POSITION MANAGEMENT ======
                if executor.positions.get(asset):
                    pos = executor.positions[asset]

                    # Check EMA reversal exit
                    exit_sig = strategy.check_exit(df, pos.side)
                    if exit_sig.action.startswith("EXIT"):
                        record = executor.close_position(asset, symbol, price, "EMA_REVERSAL")
                        if record:
                            journal.log_trade(record)
                            print(f"  [{asset}] EXIT {pos.side.upper()} @ ${price:,.2f} "
                                  f"PnL=${record['pnl_usd']:+,.2f} | {pos.sl_str()}")
                        continue

                    # Check stop-loss hit
                    if executor.check_sl_hit(asset, price):
                        record = executor.close_position(asset, symbol, price, "STOP_LOSS")
                        if record:
                            journal.log_trade(record)
                            print(f"  [{asset}] SL HIT @ ${price:,.2f} "
                                  f"PnL=${record['pnl_usd']:+,.2f} | {pos.sl_str()}")
                        continue

                    # Update trailing stop
                    lows = df["low"].iloc[-15:].tolist()
                    highs = df["high"].iloc[-15:].tolist()
                    swings = lows if pos.side == "long" else highs
                    executor.update_trailing_sl(asset, price, swings)

                    # Display position status
                    if pos.side == "long":
                        pnl_pct = (price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - price) / pos.entry_price * 100
                    print(f"  [{asset}] HOLD {pos.side.upper()} @ ${pos.entry_price:,.2f} | "
                          f"Now: ${price:,.2f} | SL: {pos.sl_str()} | P&L: {pnl_pct:+.2f}%")

                # ====== ENTRY LOGIC ======
                else:
                    sig = strategy.analyze(df)
                    if sig.action in ("LONG", "SHORT"):
                        print(f"  [{asset}] Signal: {sig.action} | {sig.reason[:80]}")

                        # Query LLM for confirmation
                        prompt = build_prompt(symbol, df, strategy)
                        decision = llm.query(prompt)

                        if decision and isinstance(decision, dict) and "error" not in decision:
                            action = decision.get("action", "FLAT").upper()
                            conf = float(decision.get("confidence", 0))
                            size = float(decision.get("position_size_pct", 5.0))
                            sl = float(decision.get("stop_loss_price", sig.structure_sl))
                            reasoning = decision.get("reasoning", "")
                            order_type = decision.get("order_type", "market")
                            limit_price = float(decision.get("limit_price", 0))

                            short_reason = reasoning[:80] if reasoning else "no reasoning"
                            print(f"  [{asset}] LLM: {action} {order_type.upper()} "
                                  f"conf={conf:.2f} size={size:.1f}% "
                                  f"SL=${sl:,.2f} | {short_reason}")

                            if action in ("LONG", "SHORT"):
                                # Quality check
                                expected_move = abs(price - ema) / price * 100
                                print(f"  [{asset}] QUALITY OK: conf={conf:.2f} "
                                      f"expected_move={expected_move:.2f}%")

                                side = "long" if action == "LONG" else "short"
                                notional = executor.equity * (size / 100)
                                qty = notional / price

                                print(f"  [{asset}] EXECUTING: "
                                      f"{'BUY' if side == 'long' else 'SELL'} "
                                      f"{qty:.6f} {order_type.upper()} "
                                      f"(${notional:,.0f} = {size:.1f}% of "
                                      f"${executor.equity:,.0f})")

                                pos = executor.enter_trade(
                                    asset, symbol, side, price,
                                    sl if sl > 0 else sig.structure_sl,
                                    order_type=order_type,
                                    limit_price=limit_price if limit_price > 0 else None,
                                    size_pct=size, confidence=conf, atr=atr,
                                )
                                if pos:
                                    print(f"  [{asset}] ORDER OK: {pos.order_id}")
                                    print(f"  [{asset}] SL L1 set at "
                                          f"${pos.current_sl:,.2f} (LLM-decided)")
                                    journal.log_trade({
                                        "asset": asset, "side": side,
                                        "entry_price": pos.entry_price,
                                        "qty": pos.qty, "order_id": pos.order_id,
                                    }, llm_reasoning=reasoning, confidence=conf,
                                       order_type=order_type)
                            else:
                                print(f"  [{asset}] LLM says FLAT - no trade")
                        else:
                            raw = decision.get("raw_text", "")[:60] if isinstance(decision, dict) else ""
                            print(f"  [{asset}] LLM response unparseable: {raw}")
                    else:
                        # No signal - just show status
                        pass

            # Sleep until next poll
            remaining = max(1, poll - 1)
            print(f"  [SLEEP] {remaining}s")
            time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        # Close all open positions
        for asset in list(executor.positions.keys()):
            symbol_map = {a.split("/")[0]: a for a in assets}
            sym = symbol_map.get(asset, f"{asset}/USDT:USDT")
            df = fetcher.fetch_ohlcv(sym)
            price = df["close"].iloc[-1] if df is not None and len(df) > 0 else 0
            record = executor.close_position(asset, sym, price, "SHUTDOWN")
            if record:
                journal.log_trade(record)
                print(f"  Closed {asset}: PnL=${record['pnl_usd']:+,.2f}")

        # Print performance summary
        print("\n" + "=" * 50)
        print("  PERFORMANCE SUMMARY")
        print("=" * 50)
        summary = executor.get_summary()
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print()
        journal.print_summary()


if __name__ == "__main__":
    main()
