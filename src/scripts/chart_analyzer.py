"""
Chart Data Analyzer for Claude Code MCP
=========================================
Provides real-time and historical chart analysis that Claude Code
scheduled tasks can call to make data-driven adaptation decisions.

Reads OHLCV from:
  1. Live Robinhood prices (real-time via API)
  2. Kraken CCXT (1h, 4h candles)
  3. Local parquet files (historical)
  4. Backtest cache (years of data)

Outputs analysis in JSON format for Claude Code agents to consume.

Usage:
    python -m src.scripts.chart_analyzer --asset BTC --timeframe 4h --bars 100
    python -m src.scripts.chart_analyzer --all --output logs/chart_analysis.json
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


def fetch_live_ohlcv(asset='BTC', timeframe='4h', bars=100):
    """Fetch OHLCV from CCXT (Kraken) for real-time data."""
    try:
        import ccxt
        exchange = ccxt.kraken({'enableRateLimit': True})
        symbol = f"{asset}/USD"
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=bars)
        return [{
            'time': int(bar[0] / 1000),
            'open': bar[1], 'high': bar[2], 'low': bar[3], 'close': bar[4],
            'volume': bar[5],
        } for bar in ohlcv]
    except Exception as e:
        logger.warning(f"CCXT fetch failed: {e}")
        return []


def load_parquet_ohlcv(asset='BTC', timeframe='4h', bars=100):
    """Load from local parquet files."""
    try:
        import pandas as pd
        path = f"data/{asset}USDT-{timeframe}.parquet"
        if not os.path.exists(path):
            return []
        df = pd.read_parquet(path)
        df.columns = [c.lower() for c in df.columns]
        df = df.tail(bars)
        return [{
            'time': int(row.get('timestamp', 0)),
            'open': float(row['open']), 'high': float(row['high']),
            'low': float(row['low']), 'close': float(row['close']),
            'volume': float(row.get('volume', 0)),
        } for _, row in df.iterrows()]
    except Exception as e:
        logger.warning(f"Parquet load failed: {e}")
        return []


def analyze_chart(bars, asset='BTC', timeframe='4h', spread_pct=3.34):
    """Comprehensive chart analysis for Claude Code agents."""
    if len(bars) < 20:
        return {"error": "Insufficient data", "bars": len(bars)}

    closes = [b['close'] for b in bars]
    highs = [b['high'] for b in bars]
    lows = [b['low'] for b in bars]
    volumes = [b['volume'] for b in bars if b.get('volume', 0) > 0]

    c = np.array(closes)
    h = np.array(highs)
    l = np.array(lows)

    # Current price
    current = closes[-1]
    prev = closes[-2] if len(closes) >= 2 else current

    # Trend analysis
    sma20 = np.mean(c[-20:]) if len(c) >= 20 else current
    sma50 = np.mean(c[-50:]) if len(c) >= 50 else current
    ema8 = c[-1]  # simplified
    mult = 2 / 9
    for price in c[-20:]:
        ema8 = price * mult + ema8 * (1 - mult)

    # RSI
    deltas = np.diff(c[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Volatility
    returns = np.diff(c) / c[:-1]
    volatility = np.std(returns[-20:]) * 100 if len(returns) >= 20 else 0
    atr_pct = (np.mean(h[-14:] - l[-14:]) / current * 100) if len(h) >= 14 else 0

    # Support / Resistance
    recent_lows = sorted(l[-20:])
    recent_highs = sorted(h[-20:], reverse=True)
    support = recent_lows[2] if len(recent_lows) > 2 else recent_lows[0]
    resistance = recent_highs[2] if len(recent_highs) > 2 else recent_highs[0]

    # Trend direction
    if current > sma20 > sma50:
        trend = "STRONG_UP"
    elif current > sma20:
        trend = "UP"
    elif current < sma20 < sma50:
        trend = "STRONG_DOWN"
    elif current < sma20:
        trend = "DOWN"
    else:
        trend = "SIDEWAYS"

    # Volume trend
    if len(volumes) >= 10:
        vol_recent = np.mean(volumes[-5:])
        vol_avg = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        volume_trend = "RISING" if vol_recent > vol_avg * 1.2 else ("FALLING" if vol_recent < vol_avg * 0.8 else "NORMAL")
        volume_ratio = round(vol_recent / vol_avg, 2) if vol_avg > 0 else 1.0
    else:
        volume_trend = "UNKNOWN"
        volume_ratio = 1.0

    # Distance to key levels
    dist_to_support = round((current - support) / current * 100, 2)
    dist_to_resistance = round((resistance - current) / current * 100, 2)

    # Move potential (can it clear spread?)
    potential_up = dist_to_resistance
    potential_down = dist_to_support
    clears_spread = potential_up > spread_pct or potential_down > spread_pct

    # Candle patterns (last 3 bars)
    patterns = []
    if len(bars) >= 3:
        last = bars[-1]
        prev_bar = bars[-2]
        body = abs(last['close'] - last['open'])
        wick_up = last['high'] - max(last['open'], last['close'])
        wick_down = min(last['open'], last['close']) - last['low']
        total_range = last['high'] - last['low']

        if total_range > 0:
            if body / total_range < 0.1:
                patterns.append("DOJI")
            if wick_down > body * 2 and last['close'] > last['open']:
                patterns.append("HAMMER")
            if wick_up > body * 2 and last['close'] < last['open']:
                patterns.append("SHOOTING_STAR")
        if last['close'] > last['open'] and prev_bar['close'] < prev_bar['open']:
            if last['close'] > prev_bar['open'] and last['open'] < prev_bar['close']:
                patterns.append("BULLISH_ENGULFING")
        if last['close'] < last['open'] and prev_bar['close'] > prev_bar['open']:
            if last['open'] > prev_bar['close'] and last['close'] < prev_bar['open']:
                patterns.append("BEARISH_ENGULFING")

    # 242 strategy universe consensus (if available)
    universe_result = {}
    try:
        from src.trading.strategy_universe import StrategyUniverse
        u = StrategyUniverse()
        signals = u.evaluate_all(closes, highs, lows, volumes if volumes else [0] * len(closes))
        consensus, confidence = u.get_consensus(signals)
        buy_count = sum(1 for s in signals.values() if s > 0)
        sell_count = sum(1 for s in signals.values() if s < 0)
        flat_count = sum(1 for s in signals.values() if s == 0)
        universe_result = {
            "consensus": consensus,
            "confidence": round(confidence, 3),
            "buy": buy_count,
            "sell": sell_count,
            "flat": flat_count,
            "total": len(signals),
        }
    except Exception:
        pass

    # 16 named strategy signals (if available)
    named_strategies = {}
    try:
        from src.trading.sub_strategies import SubStrategy
        for cls in SubStrategy.__subclasses__():
            if cls.__name__ == 'PairsStrategy':
                continue
            try:
                strat = cls()
                sig = strat.generate_signal(closes, highs, lows, volumes if volumes else [0] * len(closes))
                if sig != 0:
                    named_strategies[cls.__name__] = "LONG" if sig > 0 else "SHORT"
            except Exception:
                pass
    except Exception:
        pass

    return {
        "asset": asset,
        "timeframe": timeframe,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "bars_analyzed": len(bars),
        "current_price": round(current, 2),
        "price_change_pct": round((current - prev) / prev * 100, 3),

        "trend": {
            "direction": trend,
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2),
            "ema8": round(ema8, 2),
            "price_vs_sma20": "ABOVE" if current > sma20 else "BELOW",
        },

        "momentum": {
            "rsi": round(rsi, 1),
            "rsi_zone": "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "NEUTRAL"),
        },

        "volatility": {
            "daily_vol_pct": round(volatility, 3),
            "atr_pct": round(atr_pct, 3),
            "regime": "HIGH" if volatility > 3 else ("LOW" if volatility < 1 else "NORMAL"),
        },

        "volume": {
            "trend": volume_trend,
            "ratio_vs_avg": volume_ratio,
        },

        "levels": {
            "support": round(support, 2),
            "resistance": round(resistance, 2),
            "dist_to_support_pct": dist_to_support,
            "dist_to_resistance_pct": dist_to_resistance,
            "room_to_profit": round(max(potential_up, potential_down), 2),
            "clears_spread": clears_spread,
        },

        "patterns": patterns,

        "spread_analysis": {
            "spread_cost_pct": spread_pct,
            "max_upside_pct": round(potential_up, 2),
            "net_after_spread": round(potential_up - spread_pct, 2),
            "viable_trade": potential_up > spread_pct * 1.5,
        },

        "universe_242": universe_result,
        "active_named_strategies": named_strategies,

        "recommendation": (
            "STRONG_BUY" if trend in ["STRONG_UP", "UP"] and rsi < 60 and clears_spread and volume_trend != "FALLING" else
            "BUY" if trend == "UP" and rsi < 70 and clears_spread else
            "WAIT_OVERSOLD" if rsi < 30 and dist_to_support < 1 else
            "WAIT_BREAKOUT" if trend == "SIDEWAYS" and volatility < 1.5 else
            "AVOID" if trend in ["STRONG_DOWN", "DOWN"] and rsi > 40 else
            "NEUTRAL"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description='Chart Data Analyzer for Claude Code MCP')
    parser.add_argument('--asset', default='BTC', help='BTC or ETH')
    parser.add_argument('--timeframe', default='4h', help='1h, 4h, 1d')
    parser.add_argument('--bars', type=int, default=100, help='Number of bars')
    parser.add_argument('--all', action='store_true', help='Analyze BTC + ETH')
    parser.add_argument('--live', action='store_true', help='Fetch live from CCXT')
    parser.add_argument('--output', default=None, help='Output JSON file')
    args = parser.parse_args()

    assets = ['BTC', 'ETH'] if args.all else [args.asset]
    results = {}

    for asset in assets:
        print(f"\n=== Analyzing {asset} {args.timeframe} ({args.bars} bars) ===")

        if args.live:
            bars = fetch_live_ohlcv(asset, args.timeframe, args.bars)
            print(f"  Live CCXT: {len(bars)} bars fetched")
        else:
            bars = load_parquet_ohlcv(asset, args.timeframe, args.bars)
            print(f"  Parquet: {len(bars)} bars loaded")

        if not bars:
            print(f"  WARNING: No data for {asset}")
            continue

        analysis = analyze_chart(bars, asset, args.timeframe)
        results[asset] = analysis

        # Print summary
        t = analysis['trend']
        m = analysis['momentum']
        v = analysis['volatility']
        l = analysis['levels']
        s = analysis['spread_analysis']
        print(f"  Price: ${analysis['current_price']:,.2f} ({analysis['price_change_pct']:+.2f}%)")
        print(f"  Trend: {t['direction']} | SMA20=${t['sma20']:,.2f} | EMA8=${t['ema8']:,.2f}")
        print(f"  RSI: {m['rsi']} ({m['rsi_zone']}) | Vol: {v['daily_vol_pct']:.2f}% ({v['regime']})")
        print(f"  Support: ${l['support']:,.2f} ({l['dist_to_support_pct']:.1f}%) | Resistance: ${l['resistance']:,.2f} ({l['dist_to_resistance_pct']:.1f}%)")
        print(f"  Spread viable: {s['viable_trade']} (upside {s['max_upside_pct']:.1f}% - spread {s['spread_cost_pct']:.1f}% = net {s['net_after_spread']:.1f}%)")
        print(f"  Patterns: {analysis['patterns'] or 'none'}")
        print(f"  Active strategies: {analysis['active_named_strategies'] or 'none firing'}")
        if analysis['universe_242']:
            u = analysis['universe_242']
            print(f"  Universe 242: {u['buy']}↑ {u['sell']}↓ {u['flat']}— | {u['consensus']} (conf={u['confidence']})")
        print(f"  >>> RECOMMENDATION: {analysis['recommendation']}")

    # Save output
    output_path = args.output or 'logs/chart_analysis.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
