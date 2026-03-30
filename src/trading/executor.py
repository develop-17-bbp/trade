"""
Trading Executor — EMA(8) Crossover with LLM Confirmation
==========================================================
Bybit USDT perpetual futures. LONG (CALL) and SHORT (PUT).
Dynamic trailing stop-loss L1 -> L2 -> L3 -> L4 ...
"""

import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.data.fetcher import PriceFetcher
from src.data.microstructure import MicrostructureAnalyzer
from src.ai.agentic_strategist import AgenticStrategist
from src.monitoring.journal import TradeJournal
from src.indicators.indicators import ema, atr

logger = logging.getLogger(__name__)


class TradingExecutor:
    """EMA(8) crossover strategy with LLM confirmation on Bybit testnet."""

    def __init__(self, config: dict):
        self.config = config
        self.assets: List[str] = config.get('assets', ['BTC', 'ETH'])
        self.poll_interval: int = config.get('poll_interval', 10)
        self.initial_capital: float = config.get('initial_capital', 100000.0)

        # EMA / adaptive settings
        adaptive = config.get('adaptive', {})
        self.ema_period: int = adaptive.get('ema_period', 8)
        self.struct_window: int = adaptive.get('struct_window', 5)

        # Risk settings
        risk = config.get('risk', {})
        self.daily_loss_limit_pct: float = risk.get('daily_loss_limit_pct', 3.0)

        # AI / LLM settings
        ai_cfg = config.get('ai', {})
        self.ollama_base_url: str = (
            os.environ.get('OLLAMA_REMOTE_URL', '')
            or ai_cfg.get('ollama_base_url', 'http://localhost:11434')
        ).rstrip('/')
        self.ollama_model: str = (
            os.environ.get('OLLAMA_REMOTE_MODEL', '')
            or ai_cfg.get('reasoning_model', 'llama3.2:latest')
        )
        self.llm_conf_threshold: float = ai_cfg.get('llm_trade_conf_threshold', 0.40)

        # Quality gates
        self.min_confidence: float = 0.70
        self.min_atr_ratio: float = 0.0003
        self.trade_cooldown: float = 120.0       # 2 min between any trades
        self.post_close_cooldown: float = 180.0   # 3 min after closing before new entry

        # Exchange
        exchange_name = config.get('exchange', {}).get('name', 'bybit')
        testnet = config.get('mode', 'testnet') in ('testnet', 'paper')
        self.price_source = PriceFetcher(exchange_name=exchange_name, testnet=testnet)

        # LLM strategist (used as fallback / for deeper analysis)
        provider = ai_cfg.get('reasoning_provider', 'auto')
        model = ai_cfg.get('reasoning_model', 'llama3.2:latest')
        use_local = ai_cfg.get('use_local_on_failure', False)
        self.strategist = AgenticStrategist(
            provider=provider,
            model=model,
            use_local_on_failure=use_local,
        )

        # Order book microstructure analyzer
        self.microstructure = MicrostructureAnalyzer(depth=20)

        # Journal
        self.journal = TradeJournal()

        # Set leverage to 1x on Bybit (prevent amplified losses)
        try:
            if self.price_source.bybit and self.price_source.bybit.available:
                for asset in self.assets:
                    sym = f"{asset}/USDT:USDT"
                    try:
                        self.price_source.bybit.exchange.set_leverage(1, sym)
                        print(f"  [{asset}] Leverage set to 1x")
                    except Exception:
                        pass
        except Exception:
            pass

        # State
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity: float = self.initial_capital
        self.cash: float = self.initial_capital
        self.bar_count: int = 0
        self.last_trade_time: Dict[str, float] = {}
        self.last_close_time: Dict[str, float] = {}
        self.last_signal_candle: Dict[str, float] = {}  # Track candle timestamp to avoid re-entry on same candle
        self.failed_close_assets: Dict[str, float] = {}  # Assets that failed to close — skip until manual resolution

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_symbol(self, asset: str) -> str:
        """BTC -> BTC/USDT:USDT  (Bybit linear perpetual)"""
        return f"{asset}/USDT:USDT"

    def _get_spot_symbol(self, asset: str) -> str:
        """BTC -> BTC/USDT"""
        return f"{asset}/USDT"

    def _extract_ob_levels(self, order_book: dict, price: float) -> dict:
        """
        Extract key support/resistance levels from L2 order book.
        Finds bid walls (support) and ask walls (resistance) where
        large volume clusters sit — these are real levels the market respects.

        Returns:
            {
                'bid_wall': strongest bid support price (or 0),
                'ask_wall': strongest ask resistance price (or 0),
                'bid_walls': [(price, vol), ...] sorted by volume desc,
                'ask_walls': [(price, vol), ...] sorted by volume desc,
                'imbalance': float (-1 to 1, positive = bid heavy = bullish),
                'bid_depth_usd': total bid volume in USD,
                'ask_depth_usd': total ask volume in USD,
            }
        """
        result = {
            'bid_wall': 0.0, 'ask_wall': 0.0,
            'bid_walls': [], 'ask_walls': [],
            'imbalance': 0.0, 'bid_depth_usd': 0.0, 'ask_depth_usd': 0.0,
        }

        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])

        if len(bids) < 5 or len(asks) < 5:
            return result

        # Calculate imbalance using only levels NEAR current price (within 3%)
        # Testnet has fake walls far from price that skew imbalance to -1.00
        nearby_bid_vol = 0
        nearby_ask_vol = 0
        price_range = price * 0.03  # 3% from current price
        for b in bids[:20]:
            bp = float(b[0])
            if bp >= price - price_range:
                nearby_bid_vol += float(b[1])
        for a in asks[:20]:
            ap = float(a[0])
            if ap <= price + price_range:
                nearby_ask_vol += float(a[1])
        total_nearby = nearby_bid_vol + nearby_ask_vol
        if total_nearby > 0:
            result['imbalance'] = (nearby_bid_vol - nearby_ask_vol) / total_nearby
        result['bid_depth_usd'] = nearby_bid_vol * price
        result['ask_depth_usd'] = nearby_ask_vol * price

        # Find bid walls (large volume clusters = support)
        bid_vols = [(float(b[0]), float(b[1])) for b in bids[:20] if float(b[0]) >= price - price_range]
        if bid_vols:
            avg_bid_vol = sum(v for _, v in bid_vols) / len(bid_vols)
            # Walls = levels with 2x+ average volume
            bid_walls = [(p, v) for p, v in bid_vols if v >= avg_bid_vol * 2.0]
            bid_walls.sort(key=lambda x: x[1], reverse=True)
            result['bid_walls'] = bid_walls[:5]
            if bid_walls:
                result['bid_wall'] = bid_walls[0][0]  # Strongest support

        # Find ask walls (large volume clusters = resistance) — only near price
        ask_vols = [(float(a[0]), float(a[1])) for a in asks[:20] if float(a[0]) <= price + price_range]
        if ask_vols:
            avg_ask_vol = sum(v for _, v in ask_vols) / len(ask_vols)
            ask_walls = [(p, v) for p, v in ask_vols if v >= avg_ask_vol * 2.0]
            ask_walls.sort(key=lambda x: x[1], reverse=True)
            result['ask_walls'] = ask_walls[:5]
            if ask_walls:
                result['ask_wall'] = ask_walls[0][0]  # Strongest resistance

        return result

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        print("  EMA(8) Crossover + LLM | Bybit USDT Perpetual")
        print(f"  Assets: {self.assets} | Poll: {self.poll_interval}s")
        print(f"  LLM: {self.ollama_model} @ {self.ollama_base_url}")
        print("=" * 60)
        self._run_live()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run_live(self):
        while True:
            try:
                loop_start = time.time()
                self.bar_count += 1

                # Fetch account equity from Bybit (USDT + BTC collateral value)
                try:
                    if self.price_source.bybit and self.price_source.bybit.available:
                        acct = self.price_source.bybit.get_account()
                        total_equity = float(acct.get('equity', 0) or 0)
                        usdt_free = float(acct.get('cash', 0) or 0)
                        # Use USDT free for cash display, but total equity for sizing
                        # (Bybit unified uses BTC as collateral)
                        if total_equity > 0:
                            self.equity = total_equity
                        if usdt_free > 0:
                            self.cash = usdt_free
                except Exception:
                    pass

                ret_pct = ((self.equity - self.initial_capital) / self.initial_capital) * 100.0
                n_pos = len(self.positions)
                print(f"\n[BAR {self.bar_count}] Equity: ${self.equity:,.2f} | Cash: ${self.cash:,.2f} | Return: {ret_pct:+.2f}% | Positions: {n_pos}")

                for asset in self.assets:
                    try:
                        self._process_asset(asset)
                    except Exception as e:
                        print(f"  [{asset}] ERROR: {e}")
                        logger.exception(f"Error processing {asset}")

                # Sleep until next bar
                elapsed = time.time() - loop_start
                sleep_time = max(1, self.poll_interval - int(elapsed))
                print(f"  [SLEEP] {sleep_time}s")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Graceful exit.")
                break
            except Exception as e:
                print(f"  [LOOP ERROR] {e}")
                logger.exception("Main loop error")
                time.sleep(5)

    # ------------------------------------------------------------------
    # Per-asset processing
    # ------------------------------------------------------------------
    def _process_asset(self, asset: str):
        # If asset has a stuck position that can't close:
        # - Still show signal analysis (so user sees BTC trends)
        # - Skip entry (already have a position)
        # - Don't spam close attempts — retry every 10 minutes
        if asset in self.failed_close_assets:
            elapsed = time.time() - self.failed_close_assets[asset]
            if elapsed >= 600:  # Retry close every 10 minutes
                del self.failed_close_assets[asset]
                print(f"  [{asset}] Retrying stuck position close...")
            # Don't return — let it analyze and show signals

        symbol = self._get_symbol(asset)

        # ══════════════════════════════════════════════════════════════
        # 5m CANDLES ONLY — analyze 5m, trade 5m, check every 10s
        # ══════════════════════════════════════════════════════════════

        try:
            raw_5m = self.price_source.fetch_ohlcv(symbol, timeframe='5m', limit=100)
        except Exception as e:
            print(f"  [{asset}] OHLCV fetch failed: {e}")
            return
        ohlcv = PriceFetcher.extract_ohlcv(raw_5m)

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        opens = ohlcv['opens']
        volumes = ohlcv['volumes']

        if len(closes) < 20:
            print(f"  [{asset}] Not enough 5m data ({len(closes)} candles)")
            return

        # ══════════════════════════════════════════════════════════════
        # USE CONFIRMED CANDLES for signal — last candle may be incomplete
        # Signal analysis: use closes[:-1] (confirmed candles only)
        # Position management: use live price from ticker for SL checks
        # ══════════════════════════════════════════════════════════════
        # Get live tick price for position management (SL needs real-time)
        try:
            ticker = self.price_source.exchange.fetch_ticker(symbol) if self.price_source.exchange else {}
            live_price = float(ticker.get('last', 0)) if ticker.get('last') else closes[-1]
        except Exception:
            live_price = closes[-1]

        # For signal generation, use the LAST CONFIRMED candle close
        # The final element in OHLCV may be an incomplete candle
        price = closes[-2]  # Last confirmed close (signal reference)
        tick_price = live_price  # Real-time price (execution/SL)

        # Compute EMA(8) and ATR(14) on 5m candles (including current for freshness)
        ema_vals = ema(closes, self.ema_period)
        atr_vals = atr(highs, lows, closes, 14)

        # Use confirmed candle EMA for signal direction
        current_ema = ema_vals[-2]  # EMA at last confirmed candle
        prev_ema = ema_vals[-3] if len(ema_vals) >= 3 else current_ema
        current_atr = atr_vals[-1] if atr_vals else 0
        ema_direction = "RISING" if current_ema > prev_ema else "FALLING"

        # Fetch L2 order book for support/resistance walls
        try:
            order_book = self.price_source.fetch_order_book(symbol, limit=25)
            ob_levels = self._extract_ob_levels(order_book, tick_price)
        except Exception:
            ob_levels = {'bid_wall': 0, 'ask_wall': 0, 'bid_walls': [], 'ask_walls': [], 'imbalance': 0, 'bid_depth_usd': 0, 'ask_depth_usd': 0}

        # ── Signal from 5m EMA crossover (CONFIRMED candles only) ──
        signal = "NEUTRAL"

        # Check if EMA crossed through recent CONFIRMED candles (skip last incomplete)
        ema_crossed = False
        for i in range(2, min(5, len(highs))):  # Start at -2 (last confirmed)
            h = highs[-i]
            l = lows[-i]
            e = ema_vals[-i] if i <= len(ema_vals) else 0
            if l <= e <= h:
                ema_crossed = True
                break

        if ema_direction == "RISING" and price > current_ema and ema_crossed:
            signal = "BUY"
        elif ema_direction == "FALLING" and price < current_ema and ema_crossed:
            signal = "SELL"
        # Strong trend: price >1% from EMA
        elif ema_direction == "FALLING" and price < current_ema * 0.99:
            signal = "SELL"
        elif ema_direction == "RISING" and price > current_ema * 1.01:
            signal = "BUY"

        ob_imb = ob_levels.get('imbalance', 0)
        ob_bid = ob_levels.get('bid_wall', 0)
        ob_ask = ob_levels.get('ask_wall', 0)
        ob_info = f"OB[imb={ob_imb:+.2f}"
        if ob_bid > 0:
            ob_info += f" sup=${ob_bid:,.2f}"
        if ob_ask > 0:
            ob_info += f" res=${ob_ask:,.2f}"
        ob_info += "]"
        print(f"  [{asset}] ${tick_price:,.2f} (sig=${price:,.2f}) | EMA(5m): ${current_ema:.2f} {ema_direction} | Signal: {signal} | ATR: ${current_atr:.2f} | {ob_info}")

        # ── Stale position check: if internal state says position but exchange is clean ──
        if asset in self.positions and asset not in self.failed_close_assets:
            try:
                if self.price_source.bybit and self.price_source.bybit.available:
                    ex_pos = self.price_source.bybit.get_positions()
                    has_exchange_pos = any(asset in pp.get('symbol','') and float(pp.get('qty',0)) > 0 for pp in ex_pos)
                    if not has_exchange_pos:
                        print(f"  [{asset}] STALE position cleared (not on exchange)")
                        del self.positions[asset]
            except Exception:
                pass

        # ── Position management uses LIVE price, entry uses CONFIRMED price ──
        if asset in self.positions:
            self._manage_position(asset, tick_price, ohlcv, ema_vals, atr_vals, ema_direction, signal, ob_levels)
        else:
            self._evaluate_entry(asset, tick_price, ohlcv, ema_vals, atr_vals,
                                 ema_direction, signal, closes, highs, lows,
                                 opens, volumes, current_ema, current_atr, ob_levels)

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------
    def _evaluate_entry(self, asset: str, price: float, ohlcv: dict,
                        ema_vals: list, atr_vals: list, ema_direction: str,
                        signal: str, closes: list, highs: list, lows: list,
                        opens: list, volumes: list, current_ema: float,
                        current_atr: float, ob_levels: dict = None):

        ob_levels = ob_levels or {}

        # Only trade when there's a signal (BUY/SELL from crossover or strong trend)
        if signal == "NEUTRAL":
            return

        # ══════════════════════════════════════════════════════════
        # PATTERN-BASED ENTRY FILTER
        # Instead of rigid range detection, score the crossover quality
        # High score = likely to trend (L10+), Low score = likely to chop (L1-L2)
        # ══════════════════════════════════════════════════════════
        entry_score = 0
        score_reasons = []

        if len(ema_vals) >= 5:
            # 1. EMA SLOPE STRENGTH (0-3 points)
            # Steep slope = strong momentum = likely to continue
            ema_slope = abs(ema_vals[-1] - ema_vals[-3]) / ema_vals[-3] * 100 if ema_vals[-3] > 0 else 0
            if ema_slope > 0.3:
                entry_score += 3
                score_reasons.append(f"steep_slope={ema_slope:.2f}%")
            elif ema_slope > 0.1:
                entry_score += 2
                score_reasons.append(f"good_slope={ema_slope:.2f}%")
            elif ema_slope > 0.03:
                entry_score += 1
                score_reasons.append(f"mild_slope={ema_slope:.2f}%")

            # 2. CONSECUTIVE EMA DIRECTION (0-3 points)
            # EMA moving same direction for multiple bars = trend established
            consec = 0
            for i in range(len(ema_vals)-1, max(0, len(ema_vals)-10), -1):
                if i > 0:
                    if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]:
                        consec += 1
                    elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]:
                        consec += 1
                    else:
                        break
            if consec >= 5:
                entry_score += 3
                score_reasons.append(f"trend_{consec}bars")
            elif consec >= 3:
                entry_score += 2
                score_reasons.append(f"trend_{consec}bars")
            elif consec >= 2:
                entry_score += 1
                score_reasons.append(f"trend_{consec}bars")

            # 3. PRICE vs EMA SEPARATION (0-2 points)
            # Price clearly above/below EMA = momentum confirmed
            separation = abs(price - ema_vals[-1]) / ema_vals[-1] * 100 if ema_vals[-1] > 0 else 0
            if separation > 0.5:
                entry_score += 2
                score_reasons.append(f"sep={separation:.2f}%")
            elif separation > 0.2:
                entry_score += 1
                score_reasons.append(f"sep={separation:.2f}%")

            # 4. CANDLE MOMENTUM (0-2 points)
            # Last 3 candles closing in trend direction
            if len(closes) >= 4:
                if signal == "BUY" and closes[-1] > closes[-2] > closes[-3]:
                    entry_score += 2
                    score_reasons.append("3_green")
                elif signal == "BUY" and closes[-1] > closes[-2]:
                    entry_score += 1
                    score_reasons.append("2_green")
                elif signal == "SELL" and closes[-1] < closes[-2] < closes[-3]:
                    entry_score += 2
                    score_reasons.append("3_red")
                elif signal == "SELL" and closes[-1] < closes[-2]:
                    entry_score += 1
                    score_reasons.append("2_red")

        # ── RANGE DETECTION ──
        # Only block if TRULY flat — allow trades when EMA has clear direction
        if len(closes) >= 20:
            range_high = max(closes[-20:])
            range_low = min(closes[-20:])
            range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0
            atr_pct = (current_atr / price * 100) if price > 0 else 0
            atr_range_ratio = atr_pct / range_pct if range_pct > 0 else 0

            # Check if EMA has clear directional momentum (overrides range filter)
            ema_has_momentum = False
            if len(ema_vals) >= 5:
                ema_slope_check = abs(ema_vals[-1] - ema_vals[-3]) / ema_vals[-3] * 100 if ema_vals[-3] > 0 else 0
                if ema_slope_check > 0.03:  # EMA moving > 0.03% over 3 bars
                    ema_has_momentum = True

            # Only skip if: tight range AND no EMA momentum AND very choppy
            if range_pct > 0 and atr_range_ratio > 0.5 and range_pct < 1.5 and not ema_has_momentum:
                print(f"  [{asset}] RANGING: range={range_pct:.1f}% ATR={atr_pct:.2f}% ratio={atr_range_ratio:.0%} — SKIP")
                return
            elif range_pct > 0 and atr_range_ratio > 0.4:
                # Log but DON'T skip — EMA momentum overrides
                print(f"  [{asset}] RANGE NOTE: range={range_pct:.1f}% ratio={atr_range_ratio:.0%} but EMA momentum={ema_has_momentum} — ALLOWING")

        # ORDER BOOK IMBALANCE FILTER
        # Only block on EXTREME imbalance — testnet OB is thin and flips rapidly
        # -0.9 = 95% sellers, +0.9 = 95% buyers
        ob_imbalance = ob_levels.get('imbalance', 0)
        if signal == "BUY" and ob_imbalance < -0.9:
            print(f"  [{asset}] OB CONFLICT: signal=BUY but OB imbalance={ob_imbalance:+.2f} (extreme sell) — SKIP")
            return
        elif signal == "SELL" and ob_imbalance > 0.9:
            print(f"  [{asset}] OB CONFLICT: signal=SELL but OB imbalance={ob_imbalance:+.2f} (extreme buy) — SKIP")
            return

        # MINIMUM SCORE: 7 out of 10 to enter
        # Only trade STRONG patterns that have real trend potential (L38+ trails)
        # Score 7-8 = strong trend setup, Score 9-10 = excellent (catch the big moves)
        min_score = 7
        if entry_score < min_score:
            print(f"  [{asset}] WEAK: score={entry_score}/10 ({', '.join(score_reasons) or 'no momentum'}) — need {min_score}+")
            return
        else:
            quality = "EXCELLENT" if entry_score >= 9 else "STRONG"
            print(f"  [{asset}] {quality} PATTERN: score={entry_score}/10 ({', '.join(score_reasons)})")

        # CRITICAL: Check EXCHANGE positions (not just internal dict)
        # Prevents stacking when internal state is out of sync
        try:
            if self.price_source.bybit and self.price_source.bybit.available:
                exchange_positions = self.price_source.bybit.get_positions()
                for p in exchange_positions:
                    sym = p.get('symbol', '')
                    contracts = abs(float(p.get('qty', 0) or p.get('contracts', 0)))
                    if asset in sym and contracts > 0:
                        pos_side = p.get('side', 'long')
                        entry_p = float(p.get('avg_entry_price', price) or price)
                        synced_direction = 'LONG' if pos_side == 'long' else 'SHORT'

                        # Check if signal is OPPOSITE to current position
                        # e.g., holding SHORT but signal is BUY — should close and flip
                        signal_conflicts = (
                            (synced_direction == 'SHORT' and signal == 'BUY') or
                            (synced_direction == 'LONG' and signal == 'SELL')
                        )

                        if signal_conflicts and asset not in self.failed_close_assets:
                            flip_dir = 'CALL' if signal == 'BUY' else 'PUT'
                            print(f"  [{asset}] WRONG SIDE: holding {synced_direction} but signal={signal} — closing to flip to {flip_dir}")
                            close_side = 'sell' if synced_direction == 'LONG' else 'buy'
                            result = self.price_source.place_order(
                                symbol=self._get_symbol(asset),
                                side=close_side,
                                amount=contracts,
                                order_type='market',
                                price=None,
                            )
                            time.sleep(1)
                            # Verify closed
                            ex_pos2 = self.price_source.bybit.get_positions()
                            still_open = any(asset in pp.get('symbol','') and float(pp.get('qty',0)) > 0 for pp in ex_pos2)
                            if still_open:
                                print(f"  [{asset}] CLOSE FAILED — stuck, blacklisting")
                                self.failed_close_assets[asset] = time.time()
                                if asset in self.positions:
                                    del self.positions[asset]
                                return
                            else:
                                print(f"  [{asset}] CLOSED {synced_direction} — now free to enter {flip_dir}")
                                if asset in self.positions:
                                    del self.positions[asset]
                                # Don't return — fall through to entry evaluation
                                break
                        else:
                            # Same direction or no signal — sync it and manage
                            if asset not in self.positions:
                                sl_dist = current_atr * 1.5 if current_atr > 0 else price * 0.01
                                if pos_side == 'long':
                                    sync_sl = entry_p - sl_dist
                                else:
                                    sync_sl = entry_p + sl_dist
                                self.positions[asset] = {
                                    'direction': synced_direction,
                                    'side': 'buy' if pos_side == 'long' else 'sell',
                                    'entry_price': entry_p,
                                    'qty': contracts,
                                    'sl': sync_sl, 'sl_levels': ['L1'],
                                    'sl_order_id': None,
                                    'peak_price': price, 'entry_time': time.time(),
                                    'confidence': 0, 'reasoning': 'synced from exchange',
                                    'breakeven_moved': False,
                                }
                                print(f"  [{asset}] SYNCED {pos_side} position ({contracts}) from exchange | SL=${sync_sl:,.2f}")
                            # Don't return — let _manage_position handle it
                            return
        except Exception as e:
            logger.debug(f"Exchange position check failed: {e}")

        # CANDLE DEDUP: Only enter once per confirmed 5m candle
        # Prevents re-entering the same signal 6 times (10s poll x 5m candle)
        timestamps = ohlcv.get('timestamps', [])
        if len(timestamps) >= 2:
            confirmed_candle_ts = timestamps[-2]  # Last confirmed candle
            if asset in self.last_signal_candle and self.last_signal_candle[asset] == confirmed_candle_ts:
                return  # Already evaluated this candle
            self.last_signal_candle[asset] = confirmed_candle_ts

        # Minimum 30s between trades to prevent same-second churning
        now = time.time()
        if asset in self.last_trade_time:
            elapsed = now - self.last_trade_time[asset]
            if elapsed < 30:
                return

        # Quality gate: ATR ratio
        atr_ratio = current_atr / price if price > 0 else 0
        if atr_ratio < self.min_atr_ratio:
            return

        # Build candle data for LLM prompt (last 20 candles)
        n_candles = min(20, len(closes))
        candle_lines = []
        for i in range(-n_candles, 0):
            idx = len(closes) + i
            ema_v = ema_vals[idx] if idx < len(ema_vals) else 0
            cross_marker = ""
            if ema_v <= highs[idx] and ema_v >= lows[idx]:
                cross_marker = " *CROSS*"
            candle_lines.append(
                f"  O={opens[idx]:.2f} H={highs[idx]:.2f} L={lows[idx]:.2f} "
                f"C={closes[idx]:.2f} V={volumes[idx]:.0f} EMA={ema_v:.2f}{cross_marker}"
            )

        # Support / resistance (swing points from last 30 candles)
        lookback = min(30, len(highs))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        resistance = max(recent_highs) if recent_highs else price
        support = min(recent_lows) if recent_lows else price

        # Compute trend strength metrics for LLM
        ema_slope_pct = ((ema_vals[-1] - ema_vals[-4]) / ema_vals[-4] * 100) if len(ema_vals) >= 4 else 0
        consecutive_trend = 0
        for i in range(len(ema_vals)-1, max(0, len(ema_vals)-20), -1):
            if i > 0:
                if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]:
                    consecutive_trend += 1
                elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]:
                    consecutive_trend += 1
                else:
                    break

        forced_action = "SHORT" if signal == "SELL" else "LONG" if signal == "BUY" else "FLAT"

        prompt = f"""You are an EMA crossover pattern analyst for {asset}/USDT perpetual futures.
Your goal: FIND CROSSOVERS THAT LEAD TO STRONG TRENDS (L1→L2→L3→...→L38+ trailing stop progression).

═══ CURRENT MARKET ═══
Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction})
ATR(14): ${current_atr:.2f} | Support: ${support:.2f} | Resistance: ${resistance:.2f}
Signal: {signal} | EMA slope: {ema_slope_pct:+.3f}%/bar | Trend bars: {consecutive_trend}

═══ LAST {n_candles} CANDLES (5m) ═══
{chr(10).join(candle_lines)}

═══ EMA REVERSAL STRATEGY (CALL + PUT) ═══

CALL (LONG) ENTRY — When downtrend REVERSES to uptrend:
  1. EMA was FALLING, now crosses UP through a candle (marked *CROSS*)
  2. Next candle forms ENTIRELY ABOVE EMA
  3. EMA direction turns RISING
  → Entry point P1: BUY here
  → Trailing SL starts at L1 (recent swing low, 0.5% below entry)
  → As price rises: L1→L2→L3→L4→...→L38+ (SL pushes up every tick)
  → Exit: when EMA reverses back down OR SL hit (whichever is higher)

PUT (SHORT) ENTRY — When uptrend REVERSES to downtrend:
  1. EMA was RISING, now crosses DOWN through a candle (marked *CROSS*)
  2. Next candle forms ENTIRELY BELOW EMA
  3. EMA direction turns FALLING
  → Entry point P1: SELL/SHORT here
  → Trailing SL starts at L1 (recent swing high, 0.5% above entry)
  → As price falls: L1→L2→L3→L4→...→L38+ (SL pushes down every tick)
  → Exit: when EMA reverses back up OR SL hit (whichever is lower)

═══ PATTERN QUALITY ASSESSMENT ═══
Analyze these factors to determine if this crossover leads to L38+ or just L2:

HIGH CONFIDENCE (0.90-1.00) — likely L10+ trail:
  - EMA slope is steep ({ema_slope_pct:+.3f}%/bar) — momentum is strong
  - ATR is high (${current_atr:.2f}) — big moves expected
  - Consecutive trend bars: {consecutive_trend} (5+ is excellent)
  - No recent *CROSS* markers in last 3 candles (not choppy)
  - Clear break from support/resistance level
  - Volume increasing in trend direction

LOW CONFIDENCE (0.40-0.60) — likely L1-L2 then stop:
  - EMA is flat/sideways
  - Multiple *CROSS* markers in recent candles (choppy/ranging)
  - ATR is low (price not moving much)
  - Price near support AND resistance (squeeze)
  - Consecutive trend bars < 3

═══ DECISION ═══
Signal is {signal}. You MUST respond: {forced_action}.
Your confidence score determines if we trade (>=0.70) or skip (<0.70).
Rate confidence based on how many L-levels this trend could reach.

Respond ONLY with JSON:
{{"action": "{forced_action}", "confidence": <0.0-1.0>, "position_size_pct": 5, "reasoning": "brief pattern analysis"}}"""

        # Query LLM
        try:
            llm_response = self._query_llm(prompt)
            decision = json.loads(llm_response)
        except Exception as e:
            logger.warning(f"[{asset}] LLM query failed: {e}")
            # Fallback: use signal directly with moderate confidence
            if signal == "BUY":
                decision = {"action": "LONG", "confidence": 0.75, "position_size_pct": 5, "reasoning": "EMA crossover BUY (no LLM)"}
            elif signal == "SELL":
                decision = {"action": "SHORT", "confidence": 0.75, "position_size_pct": 5, "reasoning": "EMA crossover SELL (no LLM)"}
            else:
                return

        action = str(decision.get('action', 'FLAT')).upper()
        confidence = float(decision.get('confidence', 0.0))
        size_pct = float(decision.get('position_size_pct', 5))
        reasoning = str(decision.get('reasoning', ''))[:120]

        # CRITICAL: EMA signal ALWAYS overrides LLM direction
        # The LLM has a known long bias — force it to follow the strategy
        if signal == "BUY":
            if action != "LONG":
                print(f"  [{asset}] LLM override: {action} → LONG (EMA says BUY)")
            action = "LONG"
        elif signal == "SELL":
            if action != "SHORT":
                print(f"  [{asset}] LLM override: {action} → SHORT (EMA says SELL)")
            action = "SHORT"

        # Map action to direction label
        if action == "LONG":
            direction_label = "CALL"
        elif action == "SHORT":
            direction_label = "PUT"
        else:
            return

        print(f"  [{asset}] LLM: {direction_label} MARKET conf={confidence:.2f} size={size_pct:.0f}% | {reasoning}")

        # Quality gate: confidence
        if confidence < self.min_confidence:
            print(f"  [{asset}] SKIP: confidence {confidence:.2f} < {self.min_confidence}")
            return

        # Calculate position size from total equity (USDT + BTC collateral)
        size_pct = max(1, min(5, size_pct))  # Cap at 5%
        if self.equity <= 0:
            print(f"  [{asset}] SKIP: no equity available")
            return

        notional = self.equity * (size_pct / 100.0)

        # Hard cap: max $2,000 per trade (safe for ~$70K account with 1x leverage)
        notional = min(notional, 2000.0)

        qty = notional / price if price > 0 else 0
        qty = round(qty, 6)

        if qty <= 0:
            return

        # Minimum qty check (Bybit minimums: BTC=0.001, ETH=0.01)
        min_qty = {'BTC': 0.001, 'ETH': 0.01}
        asset_min = min_qty.get(asset, 0.001)
        if qty < asset_min:
            # Round up to minimum if we can afford it (< 10% of equity)
            min_notional = asset_min * price
            if min_notional <= self.equity * 0.10:
                qty = asset_min
                notional = min_notional
                print(f"  [{asset}] Adjusted qty to minimum {asset_min} (${notional:,.0f})")
            else:
                print(f"  [{asset}] SKIP: min order ${min_notional:,.0f} > 10% of equity ${self.equity:,.0f}")
                return

        # Determine order side and price
        side = 'buy' if action == 'LONG' else 'sell'

        # Execution type from config
        entry_type = self.config.get('execution', {}).get('entry_type', 'market')

        if entry_type == 'limit':
            order_price = current_ema
            print(f"  [{asset}] {direction_label}: {side.upper()} {qty:.6f} LIMIT@${order_price:,.2f} (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")
        else:
            order_price = None  # CRITICAL: market orders must NOT have a price
            print(f"  [{asset}] {direction_label}: {side.upper()} {qty:.6f} MARKET (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")

        # Place order — fallback to limit if market order is rejected (testnet price cap)
        symbol = self._get_symbol(asset)
        result = self.price_source.place_order(
            symbol=symbol,
            side=side,
            amount=qty,
            order_type=entry_type,
            price=order_price,
        )

        # If market order rejected (Bybit 30208 = price cap), retry as limit at best ask/bid
        if result.get('status') != 'success' and '30208' in str(result.get('message', '')):
            try:
                ob = self.price_source.fetch_order_book(symbol, limit=5)
                if side == 'buy' and ob.get('asks'):
                    limit_price = float(ob['asks'][0][0])
                elif side == 'sell' and ob.get('bids'):
                    limit_price = float(ob['bids'][0][0])
                else:
                    limit_price = None
                if limit_price:
                    print(f"  [{asset}] Market rejected (price cap) — retrying LIMIT @ ${limit_price:,.2f}")
                    result = self.price_source.place_order(
                        symbol=symbol, side=side, amount=qty,
                        order_type='limit', price=limit_price,
                    )
            except Exception as e:
                print(f"  [{asset}] Limit fallback failed: {e}")

        if result.get('status') == 'success':
            order_id = result.get('order_id', 'unknown')

            # Compute initial stop-loss (L1) using ORDER BOOK + ATR
            # Priority: order book wall > ATR-based > percentage fallback
            sl_distance = current_atr * 1.5  # ATR fallback
            sl_distance = max(sl_distance, price * 0.003)  # minimum 0.3%
            sl_distance = min(sl_distance, price * 0.02)   # maximum 2%

            sl_source = "ATR"
            if action == 'LONG':
                sl_price = price - sl_distance
                # Use order book bid wall as support-based SL if available
                # Place SL just below the strongest bid wall (support)
                bid_wall = ob_levels.get('bid_wall', 0)
                if bid_wall > 0 and bid_wall < price:
                    ob_sl = bid_wall * 0.999  # 0.1% below the wall
                    # Only use if within reasonable range (0.3% to 2% from entry)
                    ob_dist_pct = (price - ob_sl) / price * 100
                    if 0.3 <= ob_dist_pct <= 2.0:
                        sl_price = ob_sl
                        sl_source = f"OB_BID_WALL@${bid_wall:,.2f}"
            else:  # SHORT
                sl_price = price + sl_distance
                # Use order book ask wall as resistance-based SL
                # Place SL just above the strongest ask wall (resistance)
                ask_wall = ob_levels.get('ask_wall', 0)
                if ask_wall > 0 and ask_wall > price:
                    ob_sl = ask_wall * 1.001  # 0.1% above the wall
                    ob_dist_pct = (ob_sl - price) / price * 100
                    if 0.3 <= ob_dist_pct <= 2.0:
                        sl_price = ob_sl
                        sl_source = f"OB_ASK_WALL@${ask_wall:,.2f}"

            imb = ob_levels.get('imbalance', 0)
            print(f"  [{asset}] ORDER OK: {order_id} | SL L1=${sl_price:,.2f} ({sl_source}) | OB imbalance={imb:+.2f}")

            # SL managed by polling loop (10s) — no exchange stop orders
            # Exchange SL was creating orphan positions that cost $2-3 each to close
            sl_order_id = None

            # Record position
            self.positions[asset] = {
                'direction': action,          # LONG or SHORT
                'side': side,
                'entry_price': price,
                'qty': qty,
                'order_id': order_id,
                'sl': sl_price,
                'sl_order_id': sl_order_id,
                'sl_levels': ['L1'],
                'peak_price': price,
                'entry_time': time.time(),
                'confidence': confidence,
                'reasoning': reasoning,
                'breakeven_moved': False,
            }
            self.last_trade_time[asset] = time.time()
        else:
            err = result.get('message', str(result))
            print(f"  [{asset}] ORDER FAILED: {err}")

    # ------------------------------------------------------------------
    # Position management — Aggressive Trailing SL (L1→L2→...→L38+)
    # Core idea: profit becomes investment, investment becomes safe
    # On every favorable tick, push SL forward so losses come from profits only
    # ------------------------------------------------------------------
    def _manage_position(self, asset: str, price: float, ohlcv: dict,
                         ema_vals: list, atr_vals: list, ema_direction: str,
                         signal: str, ob_levels: dict = None):
        ob_levels = ob_levels or {}
        pos = self.positions[asset]
        direction = pos['direction']   # LONG or SHORT
        entry = pos['entry_price']
        sl = pos['sl']
        sl_levels = pos['sl_levels']
        peak = pos['peak_price']

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']

        # ── 1. Update peak price (maximum favorable excursion) ──
        if direction == 'LONG':
            if price > peak:
                pos['peak_price'] = price
                peak = price
            pnl_pct = ((price - entry) / entry) * 100.0
            pnl_from_peak = ((price - peak) / peak) * 100.0 if peak > 0 else 0
        else:  # SHORT
            if price < peak:
                pos['peak_price'] = price
                peak = price
            pnl_pct = ((entry - price) / entry) * 100.0
            pnl_from_peak = ((peak - price) / peak) * 100.0 if peak > 0 else 0

        # ── 1b. OB REVERSAL EXIT — order book turns heavily against us ──
        # Only exit on EXTREME + SUSTAINED imbalance AND losing > 1%
        # Testnet OB is thin — don't overreact to momentary flips
        ob_imbalance = ob_levels.get('imbalance', 0)
        if not (asset in self.failed_close_assets):
            if direction == 'LONG' and ob_imbalance < -0.9 and pnl_pct < -1.0:
                print(f"  [{asset}] OB EXIT: LONG but OB={ob_imbalance:+.2f} (extreme sell) & P&L={pnl_pct:+.2f}% — closing")
                self._close_position(asset, price, f"OB reversal (imb={ob_imbalance:+.2f})")
                self.last_trade_time.pop(asset, None)
                self.last_close_time.pop(asset, None)
                return
            elif direction == 'SHORT' and ob_imbalance > 0.9 and pnl_pct < -1.0:
                print(f"  [{asset}] OB EXIT: SHORT but OB={ob_imbalance:+.2f} (extreme buy) & P&L={pnl_pct:+.2f}% — closing")
                self._close_position(asset, price, f"OB reversal (imb={ob_imbalance:+.2f})")
                self.last_trade_time.pop(asset, None)
                self.last_close_time.pop(asset, None)
                return

        # ── 2. HARD STOP: max -2% loss — non-negotiable ──
        # But if asset is blacklisted (stuck, can't close), just log and skip
        is_stuck = asset in self.failed_close_assets
        if pnl_pct <= -2.0:
            if is_stuck:
                print(f"  [{asset}] STUCK {pnl_pct:+.2f}% (can't close — no liquidity)")
                return
            print(f"  [{asset}] HARD STOP at ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
            self._close_position(asset, price, f"Hard stop -2%")
            return

        # ── 3. Check if current SL is hit ──
        sl_hit = False
        if direction == 'LONG' and price <= sl:
            sl_hit = True
        elif direction == 'SHORT' and price >= sl:
            sl_hit = True

        if sl_hit:
            if is_stuck:
                print(f"  [{asset}] STUCK SL {pnl_pct:+.2f}% (can't close — no liquidity)")
                return
            print(f"  [{asset}] SL {sl_levels[-1]} HIT at ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
            self._close_position(asset, price, f"SL {sl_levels[-1]} hit at ${sl:,.2f}")
            return

        # ── 4. TRAILING SL — ATR-based + minimum profit protection ──
        #
        # Balance between riding trends and locking profits:
        # - ATR trailing gives room for normal pullbacks
        # - Minimum profit floor guarantees we keep a % of gains
        # - Whichever is TIGHTER wins (more protective)
        #
        # Phase 1 (pnl < 0.5%):  Initial SL at L1 — hold through noise
        # Phase 2 (pnl >= 0.5%): BREAKEVEN — can't lose capital
        # Phase 3 (pnl >= 1.5%): Protect 40% of profit + ATR trail (1.5x)
        # Phase 4 (pnl >= 3%):   Protect 50% of profit + ATR trail (1.2x)
        # Phase 5 (pnl >= 5%):   Protect 60% of profit + ATR trail (1.0x)
        # Phase 6 (pnl >= 10%):  Protect 70% of profit + ATR trail (0.8x)
        #
        # SL = MAX(profit_floor, atr_trail) — always pick the tighter one

        new_sl = sl
        qty = pos.get('qty', 0)
        current_atr = atr_vals[-1] if atr_vals else price * 0.01

        if direction == 'LONG':
            if pnl_pct >= 0.5:
                # Phase 2: Breakeven
                if sl < entry:
                    new_sl = entry

            if pnl_pct >= 1.5:
                # Determine protection level
                if pnl_pct >= 10.0:
                    protect_pct = 0.70
                    atr_mult = 0.8
                elif pnl_pct >= 5.0:
                    protect_pct = 0.60
                    atr_mult = 1.0
                elif pnl_pct >= 3.0:
                    protect_pct = 0.50
                    atr_mult = 1.2
                else:
                    protect_pct = 0.40
                    atr_mult = 1.5

                # Profit floor: lock in X% of peak profit
                profit_range = peak - entry
                floor_sl = entry + (profit_range * protect_pct)

                # ATR trail: peak - multiplier * ATR
                atr_trail_sl = peak - (current_atr * atr_mult)

                # Use whichever is TIGHTER (higher for LONG = more safe)
                best_sl = max(floor_sl, atr_trail_sl)
                if best_sl > new_sl and best_sl < price:
                    new_sl = best_sl

            # Swing lows + order book as additional tightening
            if pnl_pct >= 2.0:
                lookback = min(15, len(lows))
                if lookback >= 3:
                    recent_lows = lows[-lookback:]
                    for i in range(1, len(recent_lows) - 1):
                        if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                            if recent_lows[i] > entry and recent_lows[i] > new_sl and recent_lows[i] < price:
                                new_sl = recent_lows[i]

                for wall_price, wall_vol in ob_levels.get('bid_walls', []):
                    if wall_price > new_sl and wall_price < price and wall_price > entry:
                        ob_sl = wall_price * 0.999
                        if ob_sl > new_sl:
                            new_sl = ob_sl

        else:  # SHORT
            if pnl_pct >= 0.5:
                if sl > entry:
                    new_sl = entry

            if pnl_pct >= 1.5:
                if pnl_pct >= 10.0:
                    protect_pct = 0.70
                    atr_mult = 0.8
                elif pnl_pct >= 5.0:
                    protect_pct = 0.60
                    atr_mult = 1.0
                elif pnl_pct >= 3.0:
                    protect_pct = 0.50
                    atr_mult = 1.2
                else:
                    protect_pct = 0.40
                    atr_mult = 1.5

                # Profit floor (SHORT: SL moves DOWN)
                profit_range = entry - peak
                floor_sl = entry - (profit_range * protect_pct)

                # ATR trail (SHORT: SL = peak + multiplier * ATR)
                atr_trail_sl = peak + (current_atr * atr_mult)

                # Use whichever is TIGHTER (lower for SHORT = more safe)
                best_sl = min(floor_sl, atr_trail_sl)
                if best_sl < new_sl and best_sl > price:
                    new_sl = best_sl

            if pnl_pct >= 2.0:
                lookback = min(15, len(highs))
                if lookback >= 3:
                    recent_highs = highs[-lookback:]
                    for i in range(1, len(recent_highs) - 1):
                        if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                            if recent_highs[i] < entry and recent_highs[i] < new_sl and recent_highs[i] > price:
                                new_sl = recent_highs[i]

                for wall_price, wall_vol in ob_levels.get('ask_walls', []):
                    if wall_price < new_sl and wall_price > price and wall_price < entry:
                        ob_sl = wall_price * 1.001
                        if ob_sl < new_sl:
                            new_sl = ob_sl

        # SL can only move FORWARD (tighter), never backward
        # Minimum move: $0.50 or 0.01% of entry (whichever is larger) to avoid noise
        new_sl = round(new_sl, 2)
        min_move = max(0.50, entry * 0.0001)
        sl_moved = False
        if direction == 'LONG' and new_sl > sl + min_move:
            old_level = sl_levels[-1]
            pos['sl'] = new_sl
            sl_levels.append(f"L{len(sl_levels) + 1}")
            print(f"  [{asset}] SL {old_level}->{sl_levels[-1]}: ${sl:,.2f} -> ${new_sl:,.2f} | P&L: {pnl_pct:+.2f}%")
            sl = new_sl
            sl_moved = True
        elif direction == 'SHORT' and new_sl < sl - min_move:
            old_level = sl_levels[-1]
            pos['sl'] = new_sl
            sl_levels.append(f"L{len(sl_levels) + 1}")
            print(f"  [{asset}] SL {old_level}->{sl_levels[-1]}: ${sl:,.2f} -> ${new_sl:,.2f} | P&L: {pnl_pct:+.2f}%")
            sl = new_sl
            sl_moved = True

        # SL managed by polling (10s check) — no exchange stop orders
        # This avoids orphan positions from exchange SL fills

        # ── 5. EMA reversal exit (E1) — only on SIGNIFICANT profit ──
        # Don't exit on minor pullbacks — those caused the churning.
        # Only flip when we have meaningful profit (>= 2%) AND EMA confirms reversal
        # Skip if asset is stuck (no liquidity) — don't spam close attempts
        if not is_stuck:
            current_ema = ema_vals[-1]
            if direction == 'LONG' and ema_direction == 'FALLING' and price < current_ema and pnl_pct >= 2.0:
                print(f"  [{asset}] EMA REVERSAL (E1): CALL->PUT | exit ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
                self._close_position(asset, price, "EMA reversal (E1) - flipping to PUT")
                self.last_trade_time.pop(asset, None)
                self.last_close_time.pop(asset, None)
                return
            elif direction == 'SHORT' and ema_direction == 'RISING' and price > current_ema and pnl_pct >= 2.0:
                print(f"  [{asset}] EMA REVERSAL (E1): PUT->CALL | exit ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
                self._close_position(asset, price, "EMA reversal (E1) - flipping to CALL")
                self.last_trade_time.pop(asset, None)
                self.last_close_time.pop(asset, None)
                return

        # ── 6. Print HOLD status with profit-buffer % ──
        dir_label = "CALL" if direction == "LONG" else "PUT"
        sl_chain = '->'.join(sl_levels)
        n_levels = len(sl_levels)

        # Calculate protected vs at-risk as %
        if direction == 'LONG':
            sl_pnl_pct = ((sl - entry) / entry) * 100.0
        else:
            sl_pnl_pct = ((entry - sl) / entry) * 100.0

        status = ""
        if sl_pnl_pct > 0:
            risk_pct = pnl_pct - sl_pnl_pct
            if pnl_pct >= 10:
                phase = "LOCK-70%"
            elif pnl_pct >= 5:
                phase = "LOCK-60%"
            elif pnl_pct >= 3:
                phase = "LOCK-50%"
            elif pnl_pct >= 1.5:
                phase = "LOCK-40%"
            elif pnl_pct >= 0.5:
                phase = "BREAKEVEN"
            else:
                phase = "INITIAL"
            status = f"SAFE={sl_pnl_pct:+.2f}% RISK={risk_pct:.2f}% [{phase}]"
        elif sl_pnl_pct >= 0:
            status = "BREAKEVEN"

        imb = ob_levels.get('imbalance', 0)
        ob_tag = f"OB={imb:+.2f}" if imb != 0 else "OB=N/A"
        print(f"  [{asset}] HOLD {dir_label} @ ${entry:,.2f} | Now: ${price:,.2f} | {sl_chain} SL=${sl:,.2f} | P&L: {pnl_pct:+.2f}% | {status} | {ob_tag}")

    # ------------------------------------------------------------------
    # Trailing stop
    # ------------------------------------------------------------------
    def _trail_stop(self, asset: str, price: float, direction: str,
                    entry: float, peak: float, sl: float,
                    sl_levels: list, highs: list, lows: list,
                    pnl_pct: float) -> Optional[float]:
        """
        Structure-based trailing stop.
        Activates at 0.05% profit.
        LONG: higher swing lows tighten SL upward.
        SHORT: lower swing highs tighten SL downward.
        Fallback: 20% max giveback of peak profit.
        """
        if pnl_pct < 0.05:
            return None

        lookback = min(15, len(highs))
        if lookback < 3:
            return None

        new_sl = sl

        if direction == 'LONG':
            # Find swing lows in last 15 candles
            recent_lows = lows[-lookback:]
            swing_lows = []
            for i in range(1, len(recent_lows) - 1):
                if recent_lows[i] < recent_lows[i - 1] and recent_lows[i] < recent_lows[i + 1]:
                    swing_lows.append(recent_lows[i])

            if swing_lows:
                # Use the highest swing low that is below current price
                valid = [s for s in swing_lows if s < price and s > sl]
                if valid:
                    new_sl = max(valid)

            # Fallback: 15% max giveback (tighter trailing)
            if peak > entry:
                profit_from_peak = peak - entry
                giveback_sl = peak - (profit_from_peak * 0.15)
                if giveback_sl > new_sl and giveback_sl < price:
                    new_sl = giveback_sl

        else:  # SHORT
            # Find swing highs in last 15 candles
            recent_highs = highs[-lookback:]
            swing_highs = []
            for i in range(1, len(recent_highs) - 1):
                if recent_highs[i] > recent_highs[i - 1] and recent_highs[i] > recent_highs[i + 1]:
                    swing_highs.append(recent_highs[i])

            if swing_highs:
                # Use the lowest swing high that is above current price
                valid = [s for s in swing_highs if s > price and s < sl]
                if valid:
                    new_sl = min(valid)

            # Fallback: 15% max giveback (tighter trailing)
            if peak < entry:
                profit_from_peak = entry - peak
                giveback_sl = peak + (profit_from_peak * 0.15)
                if giveback_sl < new_sl and giveback_sl > price:
                    new_sl = giveback_sl

        if new_sl != sl:
            return round(new_sl, 2)
        return None

    # ------------------------------------------------------------------
    # Close position
    # ------------------------------------------------------------------
    def _close_position(self, asset: str, price: float, reason: str):
        if asset not in self.positions:
            return

        pos = self.positions[asset]
        direction = pos['direction']
        entry = pos['entry_price']
        qty = pos['qty']
        symbol = self._get_symbol(asset)

        # Get actual position qty from Bybit
        actual_qty = qty
        try:
            if self.price_source.bybit and self.price_source.bybit.available:
                positions = self.price_source.bybit.get_positions()
                for p in positions:
                    if asset in p.get('symbol', ''):
                        actual_qty = float(p.get('qty', qty))
                        break
        except Exception:
            pass

        # Close side is opposite of entry side
        close_side = 'sell' if direction == 'LONG' else 'buy'

        # Place market close order — retry up to 3 times for partial fills
        remaining_qty = actual_qty
        for close_attempt in range(3):
            result = self.price_source.place_order(
                symbol=symbol,
                side=close_side,
                amount=remaining_qty,
                order_type='market',
                price=None,
            )

            if result.get('status') != 'success':
                err = result.get('message', str(result))
                if 'NoImmediate' in str(err) or 'cancel' in str(err).lower():
                    print(f"  [{asset}] CLOSE FAILED (no liquidity): {err}")
                    self.failed_close_assets[asset] = time.time()
                    return
                else:
                    print(f"  [{asset}] CLOSE WARNING: {err}")

            # Check remaining position on exchange
            time.sleep(1)
            try:
                if self.price_source.bybit and self.price_source.bybit.available:
                    ex_positions = self.price_source.bybit.get_positions()
                    still_open = False
                    for p in ex_positions:
                        if asset in p.get('symbol', '') and float(p.get('qty', 0)) > 0:
                            remaining_qty = float(p.get('qty', 0))
                            still_open = True
                    if not still_open:
                        break  # Fully closed
                    if close_attempt < 2:
                        print(f"  [{asset}] Partial fill — {remaining_qty} remaining, retrying...")
                    else:
                        print(f"  [{asset}] CLOSE INCOMPLETE — {remaining_qty} still open after 3 attempts")
                        self.failed_close_assets[asset] = time.time()
                        return
                else:
                    break
            except Exception:
                break

        # Calculate P&L
        if direction == 'LONG':
            pnl_pct = ((price - entry) / entry) * 100.0
            pnl_usd = (price - entry) * qty
        else:
            pnl_pct = ((entry - price) / entry) * 100.0
            pnl_usd = (entry - price) * qty

        sl_chain = '->'.join(pos.get('sl_levels', ['L1']))
        duration_min = (time.time() - pos.get('entry_time', time.time())) / 60.0

        print(f"  [{asset}] CLOSED: P&L {pnl_pct:+.2f}% | {reason}")

        # Log to journal
        try:
            self.journal.log_trade(
                asset=asset,
                action=direction,
                entry_price=entry,
                exit_price=price,
                qty=qty,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                sl_progression=sl_chain,
                exit_reason=reason,
                llm_reasoning=pos.get('reasoning', ''),
                confidence=pos.get('confidence', 0.0),
                order_type='market',
                duration_minutes=duration_min,
                order_id=pos.get('order_id', ''),
            )
        except Exception as e:
            logger.warning(f"Journal log failed: {e}")

        # Remove from positions
        del self.positions[asset]
        now = time.time()
        self.last_close_time[asset] = now
        self.last_trade_time[asset] = now  # Also set trade cooldown to prevent immediate re-entry

    # ------------------------------------------------------------------
    # LLM query (direct Ollama HTTP)
    # ------------------------------------------------------------------
    def _query_llm(self, prompt: str) -> str:
        """Query remote Ollama and return the raw text response."""
        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 256,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data.get('response', '').strip()

            # Extract JSON from response (may be wrapped in markdown)
            if '```' in text:
                # Find JSON between code fences
                parts = text.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{'):
                        return part
            # Try direct parse
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return text[start:end + 1]
            return text
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            raise
