#!/usr/bin/env python3
"""
================================================================================
EMA CROSSOVER + DYNAMIC TRAILING STOP-LOSS AUTONOMOUS TRADING SYSTEM
================================================================================
Single-file, end-to-end: data fetch → indicators → LLM analysis → paper trade → log

Strategy (from chart reference):
  DOWNTREND: EMA crosses candle bearish → next candle below EMA → SELL (P1)
  UPTREND:   EMA crosses candle bullish → next candle above EMA → BUY (P1)
  EXIT:      Reverse crossover confirmed (E1)
  STOP-LOSS: Dynamic trailing L1→L2→L3→L4, always locking profits

Author: Autonomous Trading Research System
================================================================================
"""

import os
import sys
import json
import csv
import time
import logging
import hashlib
import signal
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ─── Core Dependencies ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ─── Exchange Data ────────────────────────────────────────────────────────────
try:
    import ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False
    print("[WARN] ccxt not installed. Install with: pip install ccxt")

# ─── HTTP for live data ──────────────────────────────────────────────────────
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ─── LLM (transformers) ──────────────────────────────────────────────────────
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARN] transformers/torch not installed. LLM features disabled.")

# ─── Visualization ────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Central configuration for the entire system."""
    # ── Symbol & Timeframe ──
    symbol: str = "ETH/USDT"
    timeframe: str = "5m"
    timeframe_minutes: int = 5

    # ── Data Sources ──
    exchange_1: str = "binance"
    exchange_2: str = "coinbasepro"
    live_api_url: str = "https://api.coingecko.com/api/v3"
    live_poll_seconds: int = 300  # 5 minutes

    # ── EMA Strategy Parameters ──
    ema_period: int = 8
    sma_short: int = 20
    sma_long: int = 50
    trend_lookback: int = 10  # candles to determine trend

    # ── Dynamic Stop-Loss ──
    stop_loss_buffer_pct: float = 0.001  # 0.1% buffer above/below structure
    min_profit_lock_pct: float = 0.002   # min 0.2% profit to start trailing
    trailing_step_pct: float = 0.001     # move SL in 0.1% steps
    max_loss_from_profit_pct: float = 0.30  # max 30% giveback of peak profit

    # ── Paper Trading ──
    initial_capital: float = 10000.0
    position_size_pct: float = 0.10  # 10% of capital per trade
    fee_pct: float = 0.001  # 0.1% per side

    # ── LLM Configuration ──
    # Remote Ollama endpoint (Cloudflare tunnel → GPU server running ollama serve)
    llm_endpoint: str = "https://cent-gmbh-yields-amended.trycloudflare.com"
    llm_model_name: str = "llama3.2"  # model name as registered in ollama
    llm_use_local: bool = False  # False = use remote Ollama endpoint first
    llm_use_4bit: bool = True  # quantize for memory efficiency (local fallback)
    llm_max_tokens: int = 512
    llm_temperature: float = 0.3
    llm_request_timeout: int = 60  # seconds; Ollama can be slow on first load
    llm_fallback_model: str = "Sengil/turkish-gemma-9b-finance-sft"  # HF local fallback

    # ── Historical Data ──
    historical_candles: int = 1000  # candles to fetch for backtest/training

    # ── Logging ──
    log_dir: str = "logs"
    trade_log_file: str = "ema_trades.csv"
    decision_log_file: str = "ema_decisions.json"
    summary_file: str = "ema_summary.json"
    chart_file: str = "ema_chart.png"

    # ── Backtest Mode ──
    backtest_mode: bool = False  # set True to backtest before live sim


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

class TrendDirection(Enum):
    UP = "UPTREND"
    DOWN = "DOWNTREND"
    NEUTRAL = "NEUTRAL"

class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class StopLevel:
    """A single stop-loss structure point."""
    level: float
    label: str  # L1, L2, L3, ...
    timestamp: datetime
    reason: str

@dataclass
class Trade:
    trade_id: str
    entry_time: datetime
    entry_price: float
    trade_type: str  # BUY or SELL
    quantity: float
    status: str = "OPEN"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    stop_loss_progression: List[Dict] = field(default_factory=list)
    current_stop_loss: Optional[float] = None
    peak_favorable: float = 0.0  # max favorable excursion
    entry_reason: str = ""
    exit_reason: str = ""
    llm_analysis: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("EMA_TRADER")
    logger.setLevel(logging.DEBUG)
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"
    ))
    logger.addHandler(ch)
    # File
    fh = logging.FileHandler(os.path.join(log_dir, "ema_system.log"))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    ))
    logger.addHandler(fh)
    return logger


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class IndicatorEngine:
    """Computes SMA, EMA, and structural points on OHLCV data."""

    @staticmethod
    def sma(values: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        cumsum = np.cumsum(values)
        cumsum[period:] = cumsum[period:] - cumsum[:-period]
        result[period - 1:] = cumsum[period - 1:] / period
        return result

    @staticmethod
    def ema(values: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        multiplier = 2.0 / (period + 1)
        # Seed with SMA
        result[period - 1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (values[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    @staticmethod
    def find_local_highs(highs: np.ndarray, window: int = 5) -> List[Tuple[int, float]]:
        """Find local high points (structure resistance for downtrend SL)."""
        points = []
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                points.append((i, highs[i]))
        return points

    @staticmethod
    def find_local_lows(lows: np.ndarray, window: int = 5) -> List[Tuple[int, float]]:
        """Find local low points (structure support for uptrend SL)."""
        points = []
        for i in range(window, len(lows) - window):
            if lows[i] == min(lows[i - window:i + window + 1]):
                points.append((i, lows[i]))
        return points

    @staticmethod
    def compute_all(df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """Compute all indicators and add as columns."""
        closes = df["close"].values.astype(float)
        highs = df["high"].values.astype(float)
        lows = df["low"].values.astype(float)

        df["ema"] = IndicatorEngine.ema(closes, config.ema_period)
        df["sma_short"] = IndicatorEngine.sma(closes, config.sma_short)
        df["sma_long"] = IndicatorEngine.sma(closes, config.sma_long)

        # Candle-EMA relationship
        df["close_above_ema"] = df["close"] > df["ema"]
        df["open_above_ema"] = df["open"] > df["ema"]
        df["high_above_ema"] = df["high"] > df["ema"]
        df["low_below_ema"] = df["low"] < df["ema"]

        # EMA crosses through candle: EMA is between high and low
        df["ema_crosses_candle"] = (df["ema"] <= df["high"]) & (df["ema"] >= df["low"])

        # Trend direction via SMA slope
        df["trend"] = TrendDirection.NEUTRAL.value
        for i in range(config.trend_lookback, len(df)):
            recent_ema = df["ema"].iloc[i - config.trend_lookback:i + 1]
            if recent_ema.is_monotonic_decreasing or (recent_ema.iloc[-1] < recent_ema.iloc[0]):
                df.iloc[i, df.columns.get_loc("trend")] = TrendDirection.DOWN.value
            elif recent_ema.is_monotonic_increasing or (recent_ema.iloc[-1] > recent_ema.iloc[0]):
                df.iloc[i, df.columns.get_loc("trend")] = TrendDirection.UP.value

        return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHER (Multi-Source)
# ══════════════════════════════════════════════════════════════════════════════

class DataFetcher:
    """Fetches historical and live OHLCV data from multiple exchanges."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.exchanges = {}

        if HAS_CCXT:
            # Source 1: Binance
            try:
                self.exchanges["binance"] = ccxt.binance({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                })
                self.logger.info("✓ Binance exchange initialized")
            except Exception as e:
                self.logger.warning(f"Binance init failed: {e}")

            # Source 2: Coinbase (via coinbasepro / coinbaseexchange)
            try:
                exc_class = getattr(ccxt, "coinbase", None) or getattr(ccxt, "coinbasepro", None)
                if exc_class:
                    self.exchanges["coinbase"] = exc_class({"enableRateLimit": True})
                    self.logger.info("✓ Coinbase exchange initialized")
            except Exception as e:
                self.logger.warning(f"Coinbase init failed: {e}")

            # Source 3: Kraken as fallback
            try:
                self.exchanges["kraken"] = ccxt.kraken({"enableRateLimit": True})
                self.logger.info("✓ Kraken exchange initialized")
            except Exception as e:
                self.logger.warning(f"Kraken init failed: {e}")

    def fetch_historical(self, source: str = "binance") -> pd.DataFrame:
        """Fetch historical OHLCV from specified exchange."""
        exchange = self.exchanges.get(source)
        if not exchange:
            self.logger.warning(f"Exchange {source} not available")
            return pd.DataFrame()

        symbol = self.config.symbol
        # Some exchanges use different symbol format
        if source == "coinbase" and "/" in symbol:
            # coinbase might need different pair format
            pass

        try:
            self.logger.info(f"Fetching {self.config.historical_candles} candles "
                             f"from {source} ({symbol} {self.config.timeframe})")
            ohlcv = exchange.fetch_ohlcv(
                symbol,
                timeframe=self.config.timeframe,
                limit=self.config.historical_candles
            )
            df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["source"] = source
            self.logger.info(f"✓ Got {len(df)} candles from {source}")
            return df
        except Exception as e:
            self.logger.error(f"Fetch from {source} failed: {e}")
            return pd.DataFrame()

    def fetch_multi_source(self) -> pd.DataFrame:
        """Fetch from multiple sources, merge, and validate."""
        frames = []
        for src in ["binance", "kraken", "coinbase"]:
            df = self.fetch_historical(src)
            if not df.empty:
                frames.append(df)
            if len(frames) >= 2:
                break  # got at least 2 sources

        if not frames:
            self.logger.error("No data from any exchange!")
            return self._generate_synthetic_data()

        # Use primary source, cross-validate with secondary
        primary = frames[0].copy()
        if len(frames) > 1:
            secondary = frames[1]
            self.logger.info(f"Cross-validating {len(primary)} candles with "
                             f"{frames[1]['source'].iloc[0]}")
            # Merge on nearest timestamp for validation
            overlap = min(len(primary), len(secondary))
            if overlap > 0:
                price_diff = abs(
                    primary["close"].iloc[-overlap:].values -
                    secondary["close"].iloc[-overlap:].values
                ).mean()
                self.logger.info(f"Average price difference between sources: "
                                 f"${price_diff:.2f}")

        primary = primary.drop(columns=["source"], errors="ignore")
        primary = primary.sort_values("timestamp").reset_index(drop=True)
        return primary

    def fetch_latest_candle(self) -> Optional[pd.DataFrame]:
        """Fetch latest candle for live trading."""
        for src in ["binance", "kraken", "coinbase"]:
            exchange = self.exchanges.get(src)
            if not exchange:
                continue
            try:
                ohlcv = exchange.fetch_ohlcv(
                    self.config.symbol,
                    timeframe=self.config.timeframe,
                    limit=3
                )
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                return df
            except Exception as e:
                self.logger.debug(f"Live fetch from {src} failed: {e}")
                continue
        return None

    def fetch_live_price(self) -> Optional[float]:
        """Fetch current price via REST API as backup."""
        if not HAS_REQUESTS:
            return None
        try:
            coin_id = self.config.symbol.split("/")[0].lower()
            if coin_id == "eth":
                coin_id = "ethereum"
            elif coin_id == "btc":
                coin_id = "bitcoin"
            resp = requests.get(
                f"{self.config.live_api_url}/simple/price",
                params={"ids": coin_id, "vs_currencies": "usd"},
                timeout=10
            )
            data = resp.json()
            return data.get(coin_id, {}).get("usd")
        except Exception as e:
            self.logger.debug(f"Live price fetch failed: {e}")
            return None

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic 5m candles if no exchange data available."""
        self.logger.warning("Generating synthetic data for testing")
        n = self.config.historical_candles
        np.random.seed(42)
        base_price = 2100.0
        returns = np.random.normal(0, 0.002, n)
        # Add trend: down then up
        trend = np.concatenate([
            np.linspace(0, -0.05, n // 3),
            np.linspace(-0.05, 0.03, n // 3),
            np.linspace(0.03, -0.02, n - 2 * (n // 3))
        ])
        prices = base_price * np.exp(np.cumsum(returns + trend / n))

        now = datetime.now(timezone.utc)
        timestamps = [now - timedelta(minutes=5 * (n - i)) for i in range(n)]

        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.003, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.003, n))),
            "close": prices,
            "volume": np.random.uniform(100, 5000, n),
        })
        return df


# ══════════════════════════════════════════════════════════════════════════════
# EMA CROSSOVER PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

class PatternDetector:
    """
    Detects EMA crossover entry/exit patterns exactly as described:

    DOWNTREND ENTRY (P1):
      1. Market is in downtrend
      2. EMA crosses through previous candle (bearish: EMA cuts candle body)
      3. NEXT candle forms BELOW EMA entirely
      4. That next candle = entry point P1 → SELL

    DOWNTREND EXIT (E1):
      1. EMA crosses candle upward (bullish crossover)
      2. Next candle forms ABOVE EMA
      3. Exit trade

    UPTREND: Exact mirror of above.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def detect_entry(self, df: pd.DataFrame, idx: int) -> Optional[TradeAction]:
        """
        Check if candle at idx is a valid entry point.
        The PREVIOUS candle must have been crossed by EMA.
        THIS candle confirms direction.
        """
        if idx < 2 or idx >= len(df):
            return None

        ema_prev = df["ema"].iloc[idx - 1]
        ema_curr = df["ema"].iloc[idx]
        prev_candle = df.iloc[idx - 1]
        curr_candle = df.iloc[idx]
        trend = df["trend"].iloc[idx]

        if pd.isna(ema_prev) or pd.isna(ema_curr):
            return None

        # ── DOWNTREND ENTRY ──
        # Previous candle: EMA crosses through it (EMA between high and low)
        # Current candle: forms entirely below EMA (high < ema)
        if trend == TrendDirection.DOWN.value:
            prev_ema_crosses = (ema_prev <= prev_candle["high"]) and (ema_prev >= prev_candle["low"])
            curr_below_ema = curr_candle["high"] < ema_curr
            # Additional: EMA should be descending
            ema_descending = ema_curr < ema_prev

            if prev_ema_crosses and curr_below_ema and ema_descending:
                self.logger.debug(f"[PATTERN] Downtrend entry at idx={idx}, "
                                  f"price={curr_candle['close']:.2f}, ema={ema_curr:.2f}")
                return TradeAction.SELL

        # ── UPTREND ENTRY ──
        # Previous candle: EMA crosses through it
        # Current candle: forms entirely above EMA (low > ema)
        if trend == TrendDirection.UP.value:
            prev_ema_crosses = (ema_prev <= prev_candle["high"]) and (ema_prev >= prev_candle["low"])
            curr_above_ema = curr_candle["low"] > ema_curr
            ema_ascending = ema_curr > ema_prev

            if prev_ema_crosses and curr_above_ema and ema_ascending:
                self.logger.debug(f"[PATTERN] Uptrend entry at idx={idx}, "
                                  f"price={curr_candle['close']:.2f}, ema={ema_curr:.2f}")
                return TradeAction.BUY

        return None

    def detect_exit(self, df: pd.DataFrame, idx: int, trade: Trade) -> bool:
        """
        Check if current candle signals exit for an open trade.
        EXIT (reverse of entry):
          - EMA crosses through current candle (reversal cross)
          - NEXT candle forms on opposite side of EMA

        Since we check in real-time, we look at idx-1 as the cross candle
        and idx as the confirmation candle.
        """
        if idx < 2 or idx >= len(df):
            return False

        ema_prev = df["ema"].iloc[idx - 1]
        ema_curr = df["ema"].iloc[idx]
        prev_candle = df.iloc[idx - 1]
        curr_candle = df.iloc[idx]

        if pd.isna(ema_prev) or pd.isna(ema_curr):
            return False

        prev_ema_crosses = (ema_prev <= prev_candle["high"]) and (ema_prev >= prev_candle["low"])

        if trade.trade_type == "SELL":
            # Exit SELL: EMA crosses candle upward, next candle above EMA
            curr_above_ema = curr_candle["low"] > ema_curr
            ema_rising = ema_curr > ema_prev
            if prev_ema_crosses and curr_above_ema and ema_rising:
                return True

        elif trade.trade_type == "BUY":
            # Exit BUY: EMA crosses candle downward, next candle below EMA
            curr_below_ema = curr_candle["high"] < ema_curr
            ema_falling = ema_curr < ema_prev
            if prev_ema_crosses and curr_below_ema and ema_falling:
                return True

        return False


# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC STOP-LOSS ENGINE (L1 → L2 → L3 → L4 ...)
# ══════════════════════════════════════════════════════════════════════════════

class DynamicStopLoss:
    """
    Implements the trailing stop-loss system from the chart:

    DOWNTREND (SELL trade):
      - L1 = initial SL = recent local HIGH (above entry)
      - As price drops (favorable), find new local highs as trade progresses
      - L2 = new structure high that's LOWER than L1 (tighter stop)
      - L3 = even lower structure high → lock more profit
      - L4 = lowest structure high before reversal
      - Exit at last SL level when price reverses above it

    UPTREND (BUY trade): Mirror
      - L1 = recent local LOW (below entry)
      - L2, L3, L4 = progressively HIGHER lows

    KEY RULE: Stop must only move in profitable direction. Never widen SL.
    Losses must always be covered by previously secured profits.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def initialize_stop(self, df: pd.DataFrame, entry_idx: int,
                        trade: Trade) -> StopLevel:
        """Set initial stop loss L1 at the nearest structure point."""
        if trade.trade_type == "SELL":
            # L1 = recent local high BEFORE entry (resistance above)
            highs = df["high"].values[:entry_idx + 1].astype(float)
            local_highs = IndicatorEngine.find_local_highs(highs, window=3)
            if local_highs:
                # Nearest local high
                best = max(local_highs, key=lambda x: x[1])
                sl_price = best[1] * (1 + self.config.stop_loss_buffer_pct)
            else:
                # Fallback: highest of last N candles
                lookback = min(10, entry_idx)
                sl_price = float(np.max(highs[-lookback:])) * (1 + self.config.stop_loss_buffer_pct)

            level = StopLevel(
                level=sl_price,
                label="L1",
                timestamp=df["timestamp"].iloc[entry_idx],
                reason=f"Initial SL at structure high {sl_price:.2f}"
            )

        else:  # BUY trade
            # L1 = recent local low BEFORE entry (support below)
            lows = df["low"].values[:entry_idx + 1].astype(float)
            local_lows = IndicatorEngine.find_local_lows(lows, window=3)
            if local_lows:
                best = min(local_lows, key=lambda x: x[1])
                sl_price = best[1] * (1 - self.config.stop_loss_buffer_pct)
            else:
                lookback = min(10, entry_idx)
                sl_price = float(np.min(lows[-lookback:])) * (1 - self.config.stop_loss_buffer_pct)

            level = StopLevel(
                level=sl_price,
                label="L1",
                timestamp=df["timestamp"].iloc[entry_idx],
                reason=f"Initial SL at structure low {sl_price:.2f}"
            )

        self.logger.info(f"  [SL] {level.label} set at {level.level:.2f} — {level.reason}")
        return level

    def update_stop(self, df: pd.DataFrame, current_idx: int,
                    trade: Trade, current_levels: List[StopLevel]) -> Optional[StopLevel]:
        """
        Check if stop-loss should be moved forward (trailing).

        Rules:
        1. Trade must be in profit (favorable excursion > min_profit_lock_pct)
        2. Find new structure point that's CLOSER to price (tighter SL)
        3. New SL must be more favorable than current SL (never widen)
        4. SL ensures accumulated profit is protected (max giveback = 30% of peak)
        """
        if current_idx < 3:
            return None

        current_price = float(df["close"].iloc[current_idx])
        current_sl = current_levels[-1].level
        next_label = f"L{len(current_levels) + 1}"

        if trade.trade_type == "SELL":
            # Profit = entry_price - current_price (favorable when price drops)
            profit_pct = (trade.entry_price - current_price) / trade.entry_price

            if profit_pct < self.config.min_profit_lock_pct:
                return None  # Not enough profit to trail yet

            # Find structure highs in the profitable zone (since entry)
            entry_idx_approx = max(0, current_idx - 50)
            highs = df["high"].values[entry_idx_approx:current_idx + 1].astype(float)
            local_highs = IndicatorEngine.find_local_highs(highs, window=2)

            if not local_highs:
                # No structure point, but if very profitable, trail anyway
                # New SL = peak favorable price + buffer (trailing)
                peak_low = float(df["low"].iloc[entry_idx_approx:current_idx + 1].min())
                max_giveback = abs(trade.entry_price - peak_low) * self.config.max_loss_from_profit_pct
                new_sl = peak_low + max_giveback

                if new_sl < current_sl:
                    level = StopLevel(
                        level=new_sl,
                        label=next_label,
                        timestamp=df["timestamp"].iloc[current_idx],
                        reason=f"Trail SL: peak_low={peak_low:.2f}, max_giveback={max_giveback:.2f}"
                    )
                    return level
                return None

            # Best new SL: lowest local high that's still BELOW current SL
            candidates = [(i, h) for i, h in local_highs
                          if h < current_sl and h > current_price]

            if candidates:
                # Pick the one closest to current price (tightest protection)
                best = min(candidates, key=lambda x: x[1])
                new_sl = best[1] * (1 + self.config.stop_loss_buffer_pct)

                if new_sl < current_sl:  # Must be tighter (lower for SELL)
                    level = StopLevel(
                        level=new_sl,
                        label=next_label,
                        timestamp=df["timestamp"].iloc[current_idx],
                        reason=f"Structure high at {best[1]:.2f}, tighter trail"
                    )
                    return level

        elif trade.trade_type == "BUY":
            # Profit = current_price - entry_price
            profit_pct = (current_price - trade.entry_price) / trade.entry_price

            if profit_pct < self.config.min_profit_lock_pct:
                return None

            entry_idx_approx = max(0, current_idx - 50)
            lows = df["low"].values[entry_idx_approx:current_idx + 1].astype(float)
            local_lows = IndicatorEngine.find_local_lows(lows, window=2)

            if not local_lows:
                peak_high = float(df["high"].iloc[entry_idx_approx:current_idx + 1].max())
                max_giveback = abs(peak_high - trade.entry_price) * self.config.max_loss_from_profit_pct
                new_sl = peak_high - max_giveback

                if new_sl > current_sl:
                    level = StopLevel(
                        level=new_sl,
                        label=next_label,
                        timestamp=df["timestamp"].iloc[current_idx],
                        reason=f"Trail SL: peak_high={peak_high:.2f}, max_giveback={max_giveback:.2f}"
                    )
                    return level
                return None

            candidates = [(i, l) for i, l in local_lows
                          if l > current_sl and l < current_price]

            if candidates:
                best = max(candidates, key=lambda x: x[1])
                new_sl = best[1] * (1 - self.config.stop_loss_buffer_pct)

                if new_sl > current_sl:
                    level = StopLevel(
                        level=new_sl,
                        label=next_label,
                        timestamp=df["timestamp"].iloc[current_idx],
                        reason=f"Structure low at {best[1]:.2f}, tighter trail"
                    )
                    return level

        return None

    def check_stop_hit(self, candle: pd.Series, trade: Trade) -> bool:
        """Check if current candle hit the stop-loss level."""
        if trade.current_stop_loss is None:
            return False

        if trade.trade_type == "SELL":
            # SL hit if price goes ABOVE stop level
            return float(candle["high"]) >= trade.current_stop_loss

        elif trade.trade_type == "BUY":
            # SL hit if price goes BELOW stop level
            return float(candle["low"]) <= trade.current_stop_loss

        return False


# ══════════════════════════════════════════════════════════════════════════════
# LLM ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class LLMEngine:
    """
    LLM for trade analysis and decision support.
    Priority: Remote Ollama GPU endpoint → Local transformers fallback → Rule-based.
    """

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        self.is_ready = False
        self.remote_ready = False
        self._inference_count = 0
        self._remote_failures = 0

        # 1. Try remote Ollama endpoint first
        if config.llm_endpoint and HAS_REQUESTS:
            self._test_remote_connection()

        # 2. Fall back to local transformers if remote unavailable
        if not self.remote_ready and HAS_TRANSFORMERS and config.llm_use_local:
            self._load_local_model()

        if not self.remote_ready and not self.is_ready:
            self.logger.info("LLM: Using rule-based fallback (no LLM available)")

    def _test_remote_connection(self):
        """Test connectivity to the remote Ollama endpoint."""
        endpoint = self.config.llm_endpoint.rstrip("/")
        self.logger.info(f"Testing remote Ollama endpoint: {endpoint}")
        try:
            # Try the /v1/models or /api/tags endpoint to verify
            for test_path in ["/v1/models", "/api/tags", "/"]:
                try:
                    resp = requests.get(
                        f"{endpoint}{test_path}",
                        timeout=15,
                        headers={"Accept": "application/json"},
                    )
                    if resp.status_code == 200:
                        self.logger.info(f"✓ Remote Ollama reachable via {test_path}")
                        self.remote_ready = True
                        break
                except requests.exceptions.ConnectionError:
                    continue

            if not self.remote_ready:
                # Even if health check fails, try a small completion
                resp = requests.post(
                    f"{endpoint}/v1/chat/completions",
                    json={
                        "model": self.config.llm_model_name,
                        "messages": [{"role": "user", "content": "Say OK"}],
                        "max_tokens": 5,
                        "temperature": 0.0,
                    },
                    timeout=self.config.llm_request_timeout,
                )
                if resp.status_code == 200:
                    self.remote_ready = True
                    self.logger.info(f"✓ Remote Ollama responding (model: {self.config.llm_model_name})")
                else:
                    self.logger.warning(f"Remote Ollama returned status {resp.status_code}")

        except Exception as e:
            self.logger.warning(f"Remote Ollama connection failed: {e}")
            self.remote_ready = False

        if self.remote_ready:
            self.logger.info(f"✓ LLM: Using remote Ollama GPU ({self.config.llm_endpoint})")
            self.logger.info(f"  Model: {self.config.llm_model_name}")

    def _load_local_model(self):
        """Download and load the local transformers LLM as fallback."""
        model_name = self.config.llm_fallback_model
        try:
            self.logger.info(f"Loading local fallback LLM: {model_name}")
            self.logger.info("This may take a few minutes on first run...")

            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }

            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                if self.config.llm_use_4bit:
                    try:
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                        model_kwargs["quantization_config"] = bnb_config
                        model_kwargs["device_map"] = "auto"
                    except Exception:
                        model_kwargs["torch_dtype"] = torch.float16
                        model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
            else:
                self.logger.info("No local GPU; loading on CPU (slower)")
                model_kwargs["torch_dtype"] = torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **model_kwargs
            )
            if self.device == "cpu" and "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)

            self.model.eval()
            self.is_ready = True
            self.logger.info(f"✓ Local LLM loaded on {self.device}")

        except Exception as e:
            self.logger.warning(f"Local LLM loading failed: {e}")
            self.is_ready = False

    def _query_remote(self, prompt: str) -> Optional[str]:
        """Query the remote Ollama GPU endpoint via OpenAI-compatible API."""
        if not self.remote_ready or not HAS_REQUESTS:
            return None

        endpoint = self.config.llm_endpoint.rstrip("/")
        try:
            self._inference_count += 1
            self.logger.debug(f"[LLM] Remote query #{self._inference_count} → {endpoint}")

            resp = requests.post(
                f"{endpoint}/v1/chat/completions",
                json={
                    "model": self.config.llm_model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert crypto trading analyst. "
                                "Always respond in valid JSON. "
                                "Strategy: EMA crossover entries with dynamic trailing stop-loss. "
                                "Maximize profit by aggressively trailing stop-loss while protecting gains. "
                                "Never exit too early. Only exit on confirmed reversal."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": self.config.llm_max_tokens,
                    "temperature": self.config.llm_temperature,
                    "stream": False,
                },
                timeout=self.config.llm_request_timeout,
                headers={"Content-Type": "application/json"},
            )

            if resp.status_code == 200:
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                self._remote_failures = 0  # reset on success
                self.logger.debug(f"[LLM] Remote response received ({len(content)} chars)")
                return content
            else:
                self._remote_failures += 1
                self.logger.warning(f"[LLM] Remote returned status {resp.status_code}: "
                                    f"{resp.text[:200]}")
                # If too many failures, disable remote
                if self._remote_failures >= 5:
                    self.logger.warning("[LLM] Too many remote failures, disabling remote endpoint")
                    self.remote_ready = False
                return None

        except requests.exceptions.Timeout:
            self._remote_failures += 1
            self.logger.warning(f"[LLM] Remote timeout ({self.config.llm_request_timeout}s)")
            return None
        except Exception as e:
            self._remote_failures += 1
            self.logger.debug(f"[LLM] Remote query error: {e}")
            return None

    def analyze_market(self, df: pd.DataFrame, idx: int,
                       active_trade: Optional[Trade] = None) -> Dict[str, Any]:
        """
        Ask LLM to analyze current market state.
        Returns structured analysis with action recommendation.
        """
        if idx < 5:
            return self._rule_based_analysis(df, idx, active_trade)

        # Build context for LLM
        recent = df.iloc[max(0, idx - 10):idx + 1]
        context = self._build_market_context(recent, active_trade)

        prompt = f"""You are an expert crypto trader. Analyze this market data and recommend an action.

STRATEGY RULES:
- DOWNTREND SELL: EMA crosses candle bearish, next candle below EMA → SELL
- UPTREND BUY: EMA crosses candle bullish, next candle above EMA → BUY
- Always trail stop-loss to lock profits
- Never exit too early; only exit on confirmed reversal
- Maximize profit by aggressively trailing stop-loss while protecting accumulated gains

CURRENT MARKET STATE:
{context}

Respond ONLY in this JSON format:
{{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, "trend": "UP|DOWN|NEUTRAL", "reasoning": "brief explanation", "stop_loss_action": "HOLD|TIGHTEN|EXIT"}}"""

        # Try remote endpoint first, then local, then rule-based
        response = self._query_remote(prompt)

        if not response and self.is_ready:
            response = self._local_inference(prompt)

        if response:
            return self._parse_llm_response(response, df, idx, active_trade)
        else:
            return self._rule_based_analysis(df, idx, active_trade)

    def _build_market_context(self, recent: pd.DataFrame,
                              active_trade: Optional[Trade]) -> str:
        """Build human-readable market context string."""
        last = recent.iloc[-1]
        lines = [
            f"Symbol: {Config.symbol}",
            f"Current Price: {last['close']:.2f}",
            f"EMA({Config.ema_period}): {last['ema']:.2f}",
            f"SMA({Config.sma_short}): {last.get('sma_short', 'N/A')}",
            f"Trend: {last.get('trend', 'NEUTRAL')}",
            f"Close vs EMA: {'ABOVE' if last['close'] > last['ema'] else 'BELOW'}",
            f"EMA crosses candle: {last.get('ema_crosses_candle', False)}",
            f"Recent candles (last 5):",
        ]
        for i in range(-5, 0):
            if abs(i) <= len(recent):
                c = recent.iloc[i]
                body = "GREEN" if c["close"] > c["open"] else "RED"
                lines.append(f"  {body}: O={c['open']:.2f} H={c['high']:.2f} "
                             f"L={c['low']:.2f} C={c['close']:.2f}")

        if active_trade:
            lines.extend([
                f"\nACTIVE TRADE:",
                f"  Type: {active_trade.trade_type}",
                f"  Entry: {active_trade.entry_price:.2f}",
                f"  Current SL: {active_trade.current_stop_loss:.2f}" if active_trade.current_stop_loss else "",
                f"  SL Levels: {len(active_trade.stop_loss_progression)}",
                f"  Unrealized P&L: {active_trade.pnl:.2f}",
            ])

        return "\n".join(lines)

    def _local_inference(self, prompt: str) -> Optional[str]:
        """Run inference on local model."""
        if not self.is_ready:
            return None
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            if self.device == "cuda" and not hasattr(self.model, "hf_device_map"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.llm_max_tokens,
                    temperature=self.config.llm_temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()
        except Exception as e:
            self.logger.debug(f"Local inference error: {e}")
            return None

    def _parse_llm_response(self, response: str, df: pd.DataFrame,
                            idx: int, active_trade: Optional[Trade]) -> Dict[str, Any]:
        """Parse LLM response, fall back to rule-based if parsing fails."""
        try:
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return {
                    "action": TradeAction[data.get("action", "HOLD")],
                    "confidence": float(data.get("confidence", 0.5)),
                    "trend": data.get("trend", "NEUTRAL"),
                    "reasoning": data.get("reasoning", "LLM analysis"),
                    "stop_loss_action": data.get("stop_loss_action", "HOLD"),
                    "source": "llm",
                }
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

        return self._rule_based_analysis(df, idx, active_trade)

    def _rule_based_analysis(self, df: pd.DataFrame, idx: int,
                             active_trade: Optional[Trade]) -> Dict[str, Any]:
        """Deterministic rule-based fallback matching the strategy exactly."""
        if idx < 2 or idx >= len(df):
            return {
                "action": TradeAction.HOLD,
                "confidence": 0.0,
                "trend": "NEUTRAL",
                "reasoning": "Insufficient data",
                "stop_loss_action": "HOLD",
                "source": "rule_based",
            }

        last = df.iloc[idx]
        prev = df.iloc[idx - 1]
        ema_curr = last["ema"]
        ema_prev = prev["ema"]
        trend = last.get("trend", TrendDirection.NEUTRAL.value)

        if pd.isna(ema_curr) or pd.isna(ema_prev):
            return {
                "action": TradeAction.HOLD,
                "confidence": 0.0,
                "trend": "NEUTRAL",
                "reasoning": "EMA not ready",
                "stop_loss_action": "HOLD",
                "source": "rule_based",
            }

        # Default
        action = TradeAction.HOLD
        confidence = 0.0
        sl_action = "HOLD"
        reasoning = "No clear signal"

        prev_cross = (ema_prev <= prev["high"]) and (ema_prev >= prev["low"])

        if not active_trade:
            # Look for entry
            if trend == TrendDirection.DOWN.value:
                if prev_cross and last["high"] < ema_curr and ema_curr < ema_prev:
                    action = TradeAction.SELL
                    confidence = 0.85
                    reasoning = "Downtrend: EMA crossed prev candle, curr below EMA → SELL entry"
            elif trend == TrendDirection.UP.value:
                if prev_cross and last["low"] > ema_curr and ema_curr > ema_prev:
                    action = TradeAction.BUY
                    confidence = 0.85
                    reasoning = "Uptrend: EMA crossed prev candle, curr above EMA → BUY entry"
        else:
            # Manage existing trade
            if active_trade.trade_type == "SELL":
                # Check for exit
                if prev_cross and last["low"] > ema_curr and ema_curr > ema_prev:
                    action = TradeAction.BUY  # close SELL = BUY
                    confidence = 0.80
                    reasoning = "Reversal: bullish crossover confirmed → EXIT SELL"
                    sl_action = "EXIT"
                else:
                    # Check if we should tighten SL
                    unrealized = (active_trade.entry_price - last["close"]) / active_trade.entry_price
                    if unrealized > self.config.min_profit_lock_pct:
                        sl_action = "TIGHTEN"
                        reasoning = f"In profit {unrealized*100:.2f}%, tighten SL"
                    action = TradeAction.HOLD
                    confidence = 0.70

            elif active_trade.trade_type == "BUY":
                if prev_cross and last["high"] < ema_curr and ema_curr < ema_prev:
                    action = TradeAction.SELL  # close BUY = SELL
                    confidence = 0.80
                    reasoning = "Reversal: bearish crossover confirmed → EXIT BUY"
                    sl_action = "EXIT"
                else:
                    unrealized = (last["close"] - active_trade.entry_price) / active_trade.entry_price
                    if unrealized > self.config.min_profit_lock_pct:
                        sl_action = "TIGHTEN"
                        reasoning = f"In profit {unrealized*100:.2f}%, tighten SL"
                    action = TradeAction.HOLD
                    confidence = 0.70

        return {
            "action": action,
            "confidence": confidence,
            "trend": trend,
            "reasoning": reasoning,
            "stop_loss_action": sl_action,
            "source": "rule_based",
        }


# ══════════════════════════════════════════════════════════════════════════════
# PAPER TRADING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class PaperTrader:
    """Manages paper trading: positions, P&L, equity tracking."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.capital = config.initial_capital
        self.equity_curve: List[float] = [config.initial_capital]
        self.active_trade: Optional[Trade] = None
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        self.peak_equity = config.initial_capital

    def open_trade(self, action: TradeAction, price: float,
                   timestamp: datetime, reason: str,
                   llm_analysis: str = "") -> Trade:
        """Open a new paper trade."""
        self.trade_counter += 1
        position_value = self.capital * self.config.position_size_pct
        quantity = position_value / price
        fee = position_value * self.config.fee_pct

        trade = Trade(
            trade_id=f"T{self.trade_counter:04d}",
            entry_time=timestamp,
            entry_price=price,
            trade_type=action.value,
            quantity=quantity,
            entry_reason=reason,
            llm_analysis=llm_analysis,
            peak_favorable=price,
        )

        self.capital -= fee  # Entry fee
        self.active_trade = trade

        self.logger.info(f"{'='*60}")
        self.logger.info(f"  📈 TRADE OPENED: {trade.trade_id}")
        self.logger.info(f"  Type: {trade.trade_type} | Price: ${price:.2f}")
        self.logger.info(f"  Qty: {quantity:.6f} | Value: ${position_value:.2f}")
        self.logger.info(f"  Reason: {reason}")
        self.logger.info(f"{'='*60}")

        return trade

    def close_trade(self, price: float, timestamp: datetime,
                    reason: str) -> Trade:
        """Close the active trade and calculate P&L."""
        trade = self.active_trade
        if not trade:
            return None

        trade.exit_time = timestamp
        trade.exit_price = price
        trade.exit_reason = reason
        trade.status = "CLOSED"

        # Calculate P&L
        position_value = trade.quantity * trade.entry_price
        fee = position_value * self.config.fee_pct  # exit fee

        if trade.trade_type == "SELL":
            # Profit when price drops: (entry - exit) * qty
            trade.pnl = (trade.entry_price - price) * trade.quantity - 2 * fee
        else:  # BUY
            # Profit when price rises: (exit - entry) * qty
            trade.pnl = (price - trade.entry_price) * trade.quantity - 2 * fee

        trade.pnl_pct = trade.pnl / position_value * 100

        self.capital += position_value + trade.pnl  # return position + profit
        self.equity_curve.append(self.capital)
        self.peak_equity = max(self.peak_equity, self.capital)

        self.closed_trades.append(trade)
        self.active_trade = None

        emoji = "✅" if trade.pnl > 0 else "❌"
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  {emoji} TRADE CLOSED: {trade.trade_id}")
        self.logger.info(f"  Type: {trade.trade_type} | Entry: ${trade.entry_price:.2f} → Exit: ${price:.2f}")
        self.logger.info(f"  P&L: ${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%)")
        self.logger.info(f"  SL Progression: {' → '.join(sl['label'] for sl in trade.stop_loss_progression)}")
        self.logger.info(f"  Reason: {reason}")
        self.logger.info(f"  Capital: ${self.capital:.2f}")
        self.logger.info(f"{'='*60}")

        return trade

    def update_unrealized(self, current_price: float):
        """Update unrealized P&L for active trade."""
        if not self.active_trade:
            return
        t = self.active_trade
        if t.trade_type == "SELL":
            t.pnl = (t.entry_price - current_price) * t.quantity
            # Track peak favorable (lowest price for SELL)
            t.peak_favorable = min(t.peak_favorable, current_price)
        else:
            t.pnl = (current_price - t.entry_price) * t.quantity
            t.peak_favorable = max(t.peak_favorable, current_price)

    def get_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance summary."""
        trades = self.closed_trades
        if not trades:
            return {"total_trades": 0, "message": "No trades executed"}

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]

        total_pnl = sum(pnls)
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001

        # Max drawdown
        equity = self.equity_curve
        peak = equity[0]
        max_dd = 0
        for eq in equity:
            peak = max(peak, eq)
            dd = (peak - eq) / peak * 100
            max_dd = max(max_dd, dd)

        # Average holding time
        holding_times = []
        for t in trades:
            if t.entry_time and t.exit_time:
                delta = (t.exit_time - t.entry_time).total_seconds() / 60
                holding_times.append(delta)

        summary = {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": len(wins) / len(trades) * 100,
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / self.config.initial_capital * 100, 2),
            "avg_profit_per_trade": round(np.mean(pnls), 2),
            "avg_pnl_pct": round(np.mean(pnl_pcts), 2),
            "max_win": round(max(pnls), 2) if pnls else 0,
            "max_loss": round(min(pnls), 2) if pnls else 0,
            "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf"),
            "max_drawdown_pct": round(max_dd, 2),
            "final_capital": round(self.capital, 2),
            "return_pct": round((self.capital - self.config.initial_capital) / self.config.initial_capital * 100, 2),
            "avg_holding_minutes": round(np.mean(holding_times), 1) if holding_times else 0,
            "sharpe_ratio": round(np.mean(pnl_pcts) / (np.std(pnl_pcts) + 1e-8) * np.sqrt(252), 2),
        }
        return summary


# ══════════════════════════════════════════════════════════════════════════════
# TRADE LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class TradeLogger:
    """Logs all trades and decisions to CSV and JSON files."""

    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize CSV
        self.csv_path = os.path.join(config.log_dir, config.trade_log_file)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "entry_time", "entry_price", "exit_time", "exit_price",
                    "trade_type", "quantity", "pnl", "pnl_pct",
                    "stop_loss_progression", "entry_reason", "exit_reason", "llm_analysis"
                ])

        # Initialize decisions JSON
        self.decisions_path = os.path.join(config.log_dir, config.decision_log_file)
        self.decisions: List[Dict] = []

    def log_trade(self, trade: Trade):
        """Log a closed trade to CSV."""
        sl_prog = " → ".join(
            f"{sl['label']}({sl['level']:.2f})" for sl in trade.stop_loss_progression
        )
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.trade_id,
                trade.entry_time.isoformat() if trade.entry_time else "",
                f"{trade.entry_price:.2f}",
                trade.exit_time.isoformat() if trade.exit_time else "",
                f"{trade.exit_price:.2f}" if trade.exit_price else "",
                trade.trade_type,
                f"{trade.quantity:.6f}",
                f"{trade.pnl:.2f}",
                f"{trade.pnl_pct:.2f}",
                sl_prog,
                trade.entry_reason,
                trade.exit_reason,
                trade.llm_analysis[:200] if trade.llm_analysis else "",
            ])
        self.logger.debug(f"Trade {trade.trade_id} logged to CSV")

    def log_decision(self, timestamp: datetime, analysis: Dict[str, Any],
                     candle: Dict = None):
        """Log every decision (even HOLD) to JSON."""
        entry = {
            "timestamp": timestamp.isoformat() if timestamp else "",
            "action": analysis["action"].value if isinstance(analysis["action"], TradeAction) else str(analysis["action"]),
            "confidence": analysis.get("confidence", 0),
            "trend": analysis.get("trend", ""),
            "reasoning": analysis.get("reasoning", ""),
            "stop_loss_action": analysis.get("stop_loss_action", ""),
            "source": analysis.get("source", ""),
        }
        if candle:
            entry["price"] = candle.get("close", 0)
            entry["ema"] = candle.get("ema", 0)
        self.decisions.append(entry)

        # Write periodically
        if len(self.decisions) % 50 == 0:
            self._flush_decisions()

    def _flush_decisions(self):
        try:
            with open(self.decisions_path, "w") as f:
                json.dump(self.decisions, f, indent=2, default=str)
        except Exception as e:
            self.logger.debug(f"Decision flush error: {e}")

    def save_summary(self, summary: Dict[str, Any]):
        """Save performance summary."""
        path = os.path.join(self.config.log_dir, self.config.summary_file)
        summary["generated_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Summary saved to {path}")

    def finalize(self):
        """Flush all pending logs."""
        self._flush_decisions()


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """Generate charts showing trades, EMA, and stop-loss levels."""

    @staticmethod
    def plot_backtest(df: pd.DataFrame, trades: List[Trade],
                      equity_curve: List[float], config: Config,
                      logger: logging.Logger):
        if not HAS_MATPLOTLIB:
            logger.info("matplotlib not available, skipping chart")
            return

        fig, axes = plt.subplots(3, 1, figsize=(20, 14), height_ratios=[3, 1, 1])
        fig.suptitle("EMA Crossover Strategy — Backtest Results", fontsize=14, fontweight="bold")

        # ── Price + EMA + Trades ──
        ax1 = axes[0]
        timestamps = df["timestamp"].values
        ax1.plot(timestamps, df["close"], color="#888888", linewidth=0.8, label="Price", alpha=0.7)
        ax1.plot(timestamps, df["ema"], color="#2196F3", linewidth=1.5, label=f"EMA({config.ema_period})")
        if "sma_short" in df.columns:
            ax1.plot(timestamps, df["sma_short"], color="#FF9800", linewidth=0.8,
                     label=f"SMA({config.sma_short})", alpha=0.5)

        # Plot trades
        for t in trades:
            color = "#4CAF50" if t.pnl > 0 else "#F44336"
            marker = "^" if t.trade_type == "BUY" else "v"

            # Entry
            ax1.scatter(t.entry_time, t.entry_price, marker=marker,
                        color=color, s=100, zorder=5, edgecolors="black", linewidth=0.5)
            ax1.annotate(f"P({t.trade_id})", (t.entry_time, t.entry_price),
                         fontsize=7, ha="center", va="bottom" if t.trade_type == "BUY" else "top")

            # Exit
            if t.exit_time and t.exit_price:
                ax1.scatter(t.exit_time, t.exit_price, marker="x",
                            color=color, s=80, zorder=5, linewidth=2)
                ax1.annotate(f"E({t.trade_id})", (t.exit_time, t.exit_price),
                             fontsize=7, ha="center", va="top" if t.trade_type == "BUY" else "bottom")

            # Stop-loss levels
            for sl in t.stop_loss_progression:
                ax1.axhline(y=sl["level"], color="#FFC107", linewidth=0.5,
                            linestyle="--", alpha=0.4)
                # Small label
                try:
                    ax1.annotate(sl["label"], (sl.get("timestamp", t.entry_time), sl["level"]),
                                 fontsize=6, color="#FFC107", alpha=0.7)
                except Exception:
                    pass

        ax1.set_ylabel("Price ($)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Volume ──
        ax2 = axes[1]
        colors = ["#4CAF50" if c > o else "#F44336"
                  for c, o in zip(df["close"], df["open"])]
        ax2.bar(timestamps, df["volume"], color=colors, alpha=0.5, width=0.003)
        ax2.set_ylabel("Volume")
        ax2.grid(True, alpha=0.3)

        # ── Equity Curve ──
        ax3 = axes[2]
        ax3.plot(range(len(equity_curve)), equity_curve, color="#2196F3", linewidth=1.5)
        ax3.axhline(y=config.initial_capital, color="#888888", linestyle="--", alpha=0.5)
        ax3.fill_between(range(len(equity_curve)), config.initial_capital,
                         equity_curve, alpha=0.1, color="#2196F3")
        ax3.set_ylabel("Equity ($)")
        ax3.set_xlabel("Trade #")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(config.log_dir, config.chart_file)
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"✓ Chart saved to {chart_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRADING SYSTEM (Orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class TradingSystem:
    """
    End-to-end orchestrator:
    1. Fetch data (multi-source)
    2. Compute indicators
    3. Run pattern detection
    4. LLM analysis
    5. Execute paper trades with dynamic SL
    6. Log everything
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = setup_logging(self.config.log_dir)
        self.running = True

        # Signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)

        self.logger.info("=" * 70)
        self.logger.info("  EMA CROSSOVER + DYNAMIC TRAILING SL — AUTONOMOUS TRADING SYSTEM")
        self.logger.info("=" * 70)

        # Initialize components
        self.logger.info("[1/5] Initializing data fetcher...")
        self.fetcher = DataFetcher(self.config, self.logger)

        self.logger.info("[2/5] Initializing pattern detector...")
        self.detector = PatternDetector(self.config, self.logger)

        self.logger.info("[3/5] Initializing dynamic stop-loss engine...")
        self.stop_loss_engine = DynamicStopLoss(self.config, self.logger)

        self.logger.info("[4/5] Initializing LLM engine...")
        self.llm = LLMEngine(self.config, self.logger)

        self.logger.info("[5/5] Initializing paper trader & logger...")
        self.trader = PaperTrader(self.config, self.logger)
        self.trade_logger = TradeLogger(self.config, self.logger)

        self.df: Optional[pd.DataFrame] = None
        self.last_processed_idx = 0

    def _shutdown_handler(self, signum, frame):
        self.logger.info("\n⚠ Shutdown signal received. Closing gracefully...")
        self.running = False

    # ── Phase 1: Data Collection ─────────────────────────────────────────────

    def collect_data(self) -> pd.DataFrame:
        """Fetch and prepare historical data."""
        self.logger.info("\n" + "─" * 60)
        self.logger.info("PHASE 1: DATA COLLECTION")
        self.logger.info("─" * 60)

        df = self.fetcher.fetch_multi_source()
        if df.empty:
            self.logger.error("No data available!")
            return df

        self.logger.info(f"Data range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        self.logger.info(f"Candles: {len(df)} | Timeframe: {self.config.timeframe}")
        return df

    # ── Phase 2: Indicator Computation ───────────────────────────────────────

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators."""
        self.logger.info("\n" + "─" * 60)
        self.logger.info("PHASE 2: INDICATOR COMPUTATION")
        self.logger.info("─" * 60)

        df = IndicatorEngine.compute_all(df, self.config)

        # Log summary
        last = df.iloc[-1]
        self.logger.info(f"EMA({self.config.ema_period}): {last['ema']:.2f}")
        self.logger.info(f"SMA({self.config.sma_short}): {last['sma_short']:.2f}" if not pd.isna(last.get("sma_short")) else "")
        self.logger.info(f"Current Trend: {last.get('trend', 'N/A')}")
        self.logger.info(f"Price vs EMA: {'ABOVE' if last['close'] > last['ema'] else 'BELOW'}")

        return df

    # ── Phase 3: Backtest Mode ───────────────────────────────────────────────

    def run_backtest(self, df: pd.DataFrame):
        """Run strategy over historical data."""
        self.logger.info("\n" + "─" * 60)
        self.logger.info("PHASE 3: BACKTESTING")
        self.logger.info("─" * 60)

        start_idx = max(self.config.ema_period + 2, self.config.trend_lookback + 2)

        for idx in range(start_idx, len(df)):
            if not self.running:
                break

            candle = df.iloc[idx]
            current_price = float(candle["close"])
            timestamp = candle["timestamp"]

            # Update unrealized P&L
            self.trader.update_unrealized(current_price)

            # ── If in a trade: manage it ──
            if self.trader.active_trade:
                trade = self.trader.active_trade

                # 1. Check stop-loss hit
                if self.stop_loss_engine.check_stop_hit(candle, trade):
                    reason = (f"Stop-loss hit at {trade.current_stop_loss:.2f} "
                              f"(last level: {trade.stop_loss_progression[-1]['label']})")
                    closed = self.trader.close_trade(
                        price=trade.current_stop_loss,
                        timestamp=timestamp,
                        reason=reason
                    )
                    if closed:
                        self.trade_logger.log_trade(closed)
                    continue

                # 2. Check pattern-based exit
                if self.detector.detect_exit(df, idx, trade):
                    reason = "EMA crossover reversal confirmed (pattern exit)"
                    closed = self.trader.close_trade(
                        price=current_price,
                        timestamp=timestamp,
                        reason=reason
                    )
                    if closed:
                        self.trade_logger.log_trade(closed)
                    continue

                # 3. Update trailing stop-loss
                new_sl = self.stop_loss_engine.update_stop(
                    df, idx, trade, self._get_stop_levels(trade)
                )
                if new_sl:
                    trade.current_stop_loss = new_sl.level
                    trade.stop_loss_progression.append({
                        "label": new_sl.label,
                        "level": new_sl.level,
                        "timestamp": str(new_sl.timestamp),
                        "reason": new_sl.reason,
                    })
                    self.logger.info(f"  [SL] Updated: {new_sl.label} = {new_sl.level:.2f}")

            # ── If no trade: look for entry ──
            else:
                entry_action = self.detector.detect_entry(df, idx)
                if entry_action:
                    # Get LLM confirmation (or rule-based)
                    analysis = self.llm.analyze_market(df, idx)
                    self.trade_logger.log_decision(timestamp, analysis, candle.to_dict())

                    # Only enter if LLM agrees or confidence is high
                    if (analysis["action"] == entry_action or
                            analysis["confidence"] >= 0.7):
                        trade = self.trader.open_trade(
                            action=entry_action,
                            price=current_price,
                            timestamp=timestamp,
                            reason=analysis.get("reasoning", "Pattern detected"),
                            llm_analysis=analysis.get("reasoning", ""),
                        )

                        # Set initial stop-loss (L1)
                        sl = self.stop_loss_engine.initialize_stop(df, idx, trade)
                        trade.current_stop_loss = sl.level
                        trade.stop_loss_progression.append({
                            "label": sl.label,
                            "level": sl.level,
                            "timestamp": str(sl.timestamp),
                            "reason": sl.reason,
                        })
                    else:
                        self.trade_logger.log_decision(timestamp, {
                            "action": TradeAction.HOLD,
                            "confidence": analysis["confidence"],
                            "trend": analysis["trend"],
                            "reasoning": f"LLM disagrees: {analysis['reasoning']}",
                            "stop_loss_action": "NONE",
                            "source": analysis["source"],
                        }, candle.to_dict())

            # Periodic progress
            if idx % 100 == 0:
                pct = (idx - start_idx) / (len(df) - start_idx) * 100
                self.logger.info(f"  Backtest progress: {pct:.1f}% | "
                                 f"Trades: {len(self.trader.closed_trades)} | "
                                 f"Capital: ${self.trader.capital:.2f}")

        # Close any open trade at end
        if self.trader.active_trade:
            self.trader.close_trade(
                price=float(df["close"].iloc[-1]),
                timestamp=df["timestamp"].iloc[-1],
                reason="End of backtest period"
            )
            self.trade_logger.log_trade(self.trader.closed_trades[-1])

        self.logger.info(f"\nBacktest complete. Total trades: {len(self.trader.closed_trades)}")

    # ── Phase 4: Live Simulation ─────────────────────────────────────────────

    def run_live_simulation(self, df: pd.DataFrame):
        """Continuously fetch new candles and trade in real-time (paper)."""
        self.logger.info("\n" + "─" * 60)
        self.logger.info("PHASE 4: LIVE SIMULATION (Paper Trading)")
        self.logger.info("─" * 60)
        self.logger.info(f"Polling every {self.config.live_poll_seconds}s for new candles...")
        self.logger.info("Press Ctrl+C to stop.\n")

        self.df = df.copy()
        self.last_processed_idx = len(self.df) - 1

        while self.running:
            try:
                # Fetch latest candles
                new_data = self.fetcher.fetch_latest_candle()
                if new_data is not None and not new_data.empty:
                    # Append new candles not yet in df
                    last_ts = self.df["timestamp"].iloc[-1]
                    new_candles = new_data[new_data["timestamp"] > last_ts]

                    if not new_candles.empty:
                        self.df = pd.concat([self.df, new_candles], ignore_index=True)
                        self.df = IndicatorEngine.compute_all(self.df, self.config)

                        # Process each new candle
                        for idx in range(self.last_processed_idx + 1, len(self.df)):
                            self._process_candle(idx)

                        self.last_processed_idx = len(self.df) - 1

                # Print dashboard
                self._print_dashboard()

                # Wait for next candle
                time.sleep(self.config.live_poll_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Live loop error: {e}")
                time.sleep(10)

    def _process_candle(self, idx: int):
        """Process a single candle in live mode (same logic as backtest)."""
        candle = self.df.iloc[idx]
        current_price = float(candle["close"])
        timestamp = candle["timestamp"]

        self.trader.update_unrealized(current_price)

        if self.trader.active_trade:
            trade = self.trader.active_trade

            # Check stop-loss
            if self.stop_loss_engine.check_stop_hit(candle, trade):
                reason = f"Stop-loss hit at {trade.current_stop_loss:.2f}"
                closed = self.trader.close_trade(trade.current_stop_loss, timestamp, reason)
                if closed:
                    self.trade_logger.log_trade(closed)
                return

            # Check exit pattern
            if self.detector.detect_exit(self.df, idx, trade):
                analysis = self.llm.analyze_market(self.df, idx, trade)
                reason = f"Pattern exit: {analysis.get('reasoning', 'reversal')}"
                closed = self.trader.close_trade(current_price, timestamp, reason)
                if closed:
                    self.trade_logger.log_trade(closed)
                return

            # Update SL
            new_sl = self.stop_loss_engine.update_stop(
                self.df, idx, trade, self._get_stop_levels(trade)
            )
            if new_sl:
                trade.current_stop_loss = new_sl.level
                trade.stop_loss_progression.append({
                    "label": new_sl.label, "level": new_sl.level,
                    "timestamp": str(new_sl.timestamp), "reason": new_sl.reason,
                })
                self.logger.info(f"  [SL] {new_sl.label} = {new_sl.level:.2f}")

        else:
            entry_action = self.detector.detect_entry(self.df, idx)
            if entry_action:
                analysis = self.llm.analyze_market(self.df, idx)
                self.trade_logger.log_decision(timestamp, analysis, candle.to_dict())

                if analysis["action"] == entry_action or analysis["confidence"] >= 0.7:
                    trade = self.trader.open_trade(
                        entry_action, current_price, timestamp,
                        analysis.get("reasoning", "Pattern detected"),
                        analysis.get("reasoning", ""),
                    )
                    sl = self.stop_loss_engine.initialize_stop(self.df, idx, trade)
                    trade.current_stop_loss = sl.level
                    trade.stop_loss_progression.append({
                        "label": sl.label, "level": sl.level,
                        "timestamp": str(sl.timestamp), "reason": sl.reason,
                    })

    def _get_stop_levels(self, trade: Trade) -> List[StopLevel]:
        """Convert trade's stop progression to StopLevel objects."""
        levels = []
        for sl in trade.stop_loss_progression:
            levels.append(StopLevel(
                level=sl["level"],
                label=sl["label"],
                timestamp=datetime.fromisoformat(sl["timestamp"]) if isinstance(sl["timestamp"], str) else sl["timestamp"],
                reason=sl.get("reason", ""),
            ))
        return levels

    def _print_dashboard(self):
        """Print real-time console dashboard."""
        if self.df is None or self.df.empty:
            return

        last = self.df.iloc[-1]
        now = datetime.now(timezone.utc).strftime("%H:%M:%S")

        lines = [
            f"\n┌{'─'*58}┐",
            f"│ {'EMA STRATEGY DASHBOARD':^56} │",
            f"│ {now:^56} │",
            f"├{'─'*58}┤",
            f"│ Price: ${last['close']:<10.2f} EMA: ${last['ema']:<10.2f}         │" if not pd.isna(last['ema']) else f"│ Price: ${last['close']:<10.2f} EMA: computing...          │",
            f"│ Trend: {str(last.get('trend', 'N/A')):<15}                              │",
            f"├{'─'*58}┤",
            f"│ Capital: ${self.trader.capital:<12.2f}                          │",
            f"│ Trades: {len(self.trader.closed_trades):<5}  Win Rate: {self._calc_winrate():<6}            │",
        ]

        if self.trader.active_trade:
            t = self.trader.active_trade
            lines.extend([
                f"├{'─'*58}┤",
                f"│ ACTIVE: {t.trade_type:<6} @ ${t.entry_price:<10.2f}                  │",
                f"│ P&L: ${t.pnl:<10.2f}  SL: ${t.current_stop_loss if t.current_stop_loss else 0:<10.2f}        │",
                f"│ SL Levels: {len(t.stop_loss_progression):<3}                                    │",
            ])

        lines.append(f"└{'─'*58}┘")
        print("\n".join(lines))

    def _calc_winrate(self) -> str:
        trades = self.trader.closed_trades
        if not trades:
            return "N/A"
        wins = sum(1 for t in trades if t.pnl > 0)
        return f"{wins/len(trades)*100:.1f}%"

    # ── Phase 5: Results & Logging ───────────────────────────────────────────

    def finalize(self):
        """Generate summary, save logs, create charts."""
        self.logger.info("\n" + "─" * 60)
        self.logger.info("PHASE 5: RESULTS & EVALUATION")
        self.logger.info("─" * 60)

        # Summary
        summary = self.trader.get_summary()
        self.trade_logger.save_summary(summary)

        # Print summary
        self.logger.info("\n📊 PERFORMANCE SUMMARY:")
        self.logger.info(f"  Total Trades:     {summary.get('total_trades', 0)}")
        self.logger.info(f"  Win Rate:         {summary.get('win_rate_pct', 0):.1f}%")
        self.logger.info(f"  Total P&L:        ${summary.get('total_pnl', 0):.2f}")
        self.logger.info(f"  Return:           {summary.get('return_pct', 0):.2f}%")
        self.logger.info(f"  Avg Trade:        ${summary.get('avg_profit_per_trade', 0):.2f}")
        self.logger.info(f"  Best Trade:       ${summary.get('max_win', 0):.2f}")
        self.logger.info(f"  Worst Trade:      ${summary.get('max_loss', 0):.2f}")
        self.logger.info(f"  Profit Factor:    {summary.get('profit_factor', 0):.2f}")
        self.logger.info(f"  Max Drawdown:     {summary.get('max_drawdown_pct', 0):.2f}%")
        self.logger.info(f"  Sharpe Ratio:     {summary.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"  Final Capital:    ${summary.get('final_capital', 0):.2f}")

        # Chart
        if self.df is not None:
            Visualizer.plot_backtest(
                self.df, self.trader.closed_trades,
                self.trader.equity_curve, self.config, self.logger
            )

        # Flush logs
        self.trade_logger.finalize()

        self.logger.info("\n✓ All logs saved to: " + self.config.log_dir)
        self.logger.info("  - Trades:    " + self.config.trade_log_file)
        self.logger.info("  - Decisions: " + self.config.decision_log_file)
        self.logger.info("  - Summary:   " + self.config.summary_file)
        self.logger.info("  - Chart:     " + self.config.chart_file)

    # ── Main Entry Point ─────────────────────────────────────────────────────

    def run(self):
        """Execute the full pipeline."""
        try:
            # 1. Collect data
            df = self.collect_data()
            if df.empty:
                self.logger.error("Cannot proceed without data. Exiting.")
                return

            # 2. Compute indicators
            df = self.compute_indicators(df)
            self.df = df

            # 3. Backtest on historical data
            self.logger.info("\n── Running backtest on historical data ──")
            self.run_backtest(df)

            # 4. Show backtest results
            self.finalize()

            # 5. Transition to live simulation
            if not self.config.backtest_mode and self.running:
                self.logger.info("\n── Transitioning to live simulation ──")
                # Reset trader for live mode (keep learned patterns)
                self.trader = PaperTrader(self.config, self.logger)
                self.trade_logger = TradeLogger(self.config, self.logger)
                self.run_live_simulation(df)

                # Final results from live sim
                self.finalize()

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user")
        except Exception as e:
            self.logger.error(f"Fatal error: {e}", exc_info=True)
        finally:
            self.trade_logger.finalize()
            self.logger.info("\n🏁 System shutdown complete.")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point — configure and run."""
    config = Config()

    # Parse command-line overrides
    import argparse
    parser = argparse.ArgumentParser(description="EMA Crossover Autonomous Trading System")
    parser.add_argument("--symbol", default="ETH/USDT", help="Trading pair (default: ETH/USDT)")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe (default: 5m)")
    parser.add_argument("--ema-period", type=int, default=8, help="EMA period (default: 8)")
    parser.add_argument("--capital", type=float, default=10000, help="Starting capital (default: 10000)")
    parser.add_argument("--backtest-only", action="store_true", help="Run backtest only, no live sim")
    parser.add_argument("--candles", type=int, default=1000, help="Historical candles to fetch")
    parser.add_argument("--llm-endpoint", default="", help="Remote LLM API endpoint URL")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM, use rule-based only")
    parser.add_argument("--position-size", type=float, default=0.10, help="Position size as fraction of capital")

    args = parser.parse_args()

    config.symbol = args.symbol
    config.timeframe = args.timeframe
    config.ema_period = args.ema_period
    config.initial_capital = args.capital
    config.backtest_mode = args.backtest_only
    config.historical_candles = args.candles
    config.position_size_pct = args.position_size

    if args.llm_endpoint:
        config.llm_endpoint = args.llm_endpoint
    if args.no_llm:
        config.llm_use_local = False
        config.llm_endpoint = ""  # disable remote too

    # Timeframe to minutes mapping
    tf_map = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240}
    config.timeframe_minutes = tf_map.get(config.timeframe, 5)
    config.live_poll_seconds = config.timeframe_minutes * 60

    # Run the system
    system = TradingSystem(config)
    system.run()


if __name__ == "__main__":
    main()
