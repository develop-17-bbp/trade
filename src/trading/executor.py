"""
Trading Executor — EMA(8) Crossover with LLM Confirmation
==========================================================
Bybit USDT perpetual futures. LONG (CALL) and SHORT (PUT).
Dynamic trailing stop-loss L1 -> L2 -> L3 -> L4 ...
"""

import os
import re
import json
import math
import time
import logging
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.data.fetcher import PriceFetcher
from src.data.microstructure import MicrostructureAnalyzer
from src.ai.agentic_strategist import AgenticStrategist
from src.monitoring.journal import TradeJournal
from src.indicators.indicators import ema, atr

# Agent Orchestrator + Math Injection (optional, graceful degradation)
try:
    from src.agents.orchestrator import AgentOrchestrator
    from src.agents.base_agent import EnhancedDecision, AgentVote
    from src.ai.math_injection import MathInjector
    ORCHESTRATOR_AVAILABLE = True
except Exception:
    ORCHESTRATOR_AVAILABLE = False

# On-Chain data fetcher (free APIs: whale flows, mempool, DeFi TVL)
try:
    from src.data.on_chain_fetcher import OnChainFetcher
    ONCHAIN_AVAILABLE = True
except Exception:
    ONCHAIN_AVAILABLE = False

# Memory Vault (trade experience database with semantic search)
try:
    from src.ai.memory_vault import MemoryVault
    MEMORY_AVAILABLE = True
except Exception:
    MEMORY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _signal_bar_open_utc(ts_ms: Any) -> str:
    """Format exchange OHLCV open time (ms) for logs."""
    try:
        sec = float(ts_ms) / 1000.0
        return datetime.utcfromtimestamp(sec).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError, OSError):
        return "?"


def _early_ema_p1_switch_signal(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    ema_vals: List[float],
    tick_price: float,
    ema_period: int,
    *,
    overextension_pct: float = 10.0,
    signal_trend_phase: str = "",
    price_falling: bool = False,
    price_rising: bool = False,
    entry_block_stabilizing: bool = False,
) -> Optional[str]:
    """
    P1-style entry on the forming candle: previous CLOSED bar had EMA touch the range;
    current bar (live) is entirely on one side of EMA and live EMA vs prior bar EMA
    shows the turn. Mirrors TRADING_SYSTEM_PROMPT but uses tick-updated EMA on [-1].
    """
    if len(closes) < 3 or len(highs) < 3 or len(lows) < 3 or len(ema_vals) < 3:
        return None
    ema_prev_bar = float(ema_vals[-2])
    low2, high2 = float(lows[-2]), float(highs[-2])
    if not (low2 <= ema_prev_bar <= high2):
        return None

    closes_live = list(closes[:-1]) + [float(tick_price)]
    ema_live = ema(closes_live, ema_period)
    ema_now = float(ema_live[-1])
    if ema_now <= 0:
        return None

    sep = abs(float(tick_price) - ema_now) / ema_now * 100.0
    if sep > overextension_pct:
        return None

    # LONG: bar fully above EMA, EMA turning up vs value anchored at prior close
    if float(lows[-1]) > ema_now and ema_now > ema_prev_bar:
        if price_falling:
            return None
        if entry_block_stabilizing and signal_trend_phase == "RISING_STABILIZING":
            return None
        return "BUY"

    # SHORT: bar fully below EMA, EMA turning down vs prior bar
    if float(highs[-1]) < ema_now and ema_now < ema_prev_bar:
        if price_rising:
            return None
        if entry_block_stabilizing and signal_trend_phase == "FALLING_STABILIZING":
            return None
        return "SELL"

    return None


def _pred_l_level_chop_zone(pred_l: str) -> bool:
    """
    True when predicted_l_level implies exit at L1/L2 only (no L4+ runway).
    Small LLMs often return L1-L2 + proceed=true + high confidence; we veto that mismatch.
    """
    t = str(pred_l).strip().lower().replace("–", "-").replace("—", "-")
    if not t or t in ("?", "none", "nan"):
        return False
    nums = [int(m.group(1)) for m in re.finditer(r"l\s*(\d+)", t)]
    if nums:
        return max(nums) <= 2
    if "chop" in t or "l1-l2" in t or "l1 l2" in t:
        return True
    if "l1" in t or "l2" in t:
        if any(x in t for x in ("l4", "l5", "l6", "l7", "l8", "l9", "l10")):
            return False
        return True
    return False


def _linear_qty_multiplier(qty: float, contract_size: float) -> float:
    """USDT linear PnL scale: Bybit qty in coin → mult=qty; Delta qty in contracts → qty×contract_size."""
    if contract_size and contract_size > 0:
        return float(qty) * float(contract_size)
    return float(qty)


def _loss_usd_at_price(
    direction: str,
    entry: float,
    exit_price: float,
    qty: float,
    contract_size: float = 0.0,
) -> float:
    """Positive USD loss vs entry if flat at exit_price (ignores fees)."""
    if entry <= 0 or qty <= 0:
        return 0.0
    m = _linear_qty_multiplier(qty, contract_size)
    if direction == 'LONG':
        return max(0.0, m * (entry - exit_price))
    return max(0.0, m * (exit_price - entry))


def _unrealized_pnl_usd(
    direction: str,
    entry: float,
    mark: float,
    qty: float,
    contract_size: float = 0.0,
) -> float:
    m = _linear_qty_multiplier(qty, contract_size)
    if direction == 'LONG':
        return m * (mark - entry)
    return m * (entry - mark)


def _bounce_relief_from_ob(action: str, ob_levels: Optional[dict], sl_source: str) -> bool:
    """
    Wider loss cap when book suggests absorption (support for longs, resistance for shorts).
    """
    if sl_source and ('OB_BID_WALL' in sl_source or 'OB_ASK_WALL' in sl_source):
        return True
    if not ob_levels:
        return False
    imb = float(ob_levels.get('imbalance', 0) or 0)
    if action == 'LONG' and imb >= 0.07:
        return True
    if action == 'SHORT' and imb <= -0.07:
        return True
    return False


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
        self.entry_pattern_min_score: int = int(adaptive.get('entry_pattern_min_score', 3))
        self.entry_min_trend_bars: int = int(adaptive.get('entry_min_trend_bars', 1))
        # EMA slope deceleration → FALLING_STABILIZING / RISING_STABILIZING (potential reversal)
        self.trend_stabilize_decel_ratio: float = float(
            adaptive.get('trend_stabilize_decel_ratio', 0.55)
        )
        self.trend_stabilize_atr_k: float = float(adaptive.get('trend_stabilize_atr_k', 0.06))
        self.entry_block_stabilizing_crossover: bool = bool(
            adaptive.get('entry_block_stabilizing_crossover', False)
        )

        _exec = config.get('execution', {})
        self.signal_timeframe: str = (_exec.get('signal_timeframe') or '1h').strip()
        self.context_timeframe: str = (_exec.get('context_timeframe') or '1m').strip()
        self.require_context_alignment: bool = bool(_exec.get('require_context_alignment', False))
        _mtf_raw = _exec.get('analysis_timeframes')
        if isinstance(_mtf_raw, list) and _mtf_raw:
            self.analysis_timeframes = [str(x).strip() for x in _mtf_raw if str(x).strip()]
        else:
            self.analysis_timeframes = ['1m', '5m', '15m', '30m', '1h']
        if self.signal_timeframe not in self.analysis_timeframes:
            self.analysis_timeframes.append(self.signal_timeframe)
        _seen: set = set()
        self.analysis_timeframes = [
            x for x in self.analysis_timeframes if not (x in _seen or _seen.add(x))
        ]
        self.mtf_confluence_min: int = max(0, int(_exec.get('mtf_confluence_min', 0)))
        # signal_timing: confirmed = EMA/crossover from last CLOSED signal bar (wait for bar close).
        # live = treat current tick as the forming bar's close for EMA/signals; no candle-close wait for entries.
        _stg = str(_exec.get('signal_timing', 'confirmed') or 'confirmed').strip().lower()
        self.signal_timing: str = _stg if _stg in ('confirmed', 'live') else 'confirmed'
        # Trailing SL: confirmed_close = only when last CLOSED bar's close crosses SL (slow, fewer wicks).
        # intrabar_wick = exit when live tick OR current (forming) bar low/high on signal TF touches SL (faster).
        _sem = str(_exec.get('sl_exit_mode', 'confirmed_close') or 'confirmed_close').strip().lower()
        self.sl_exit_mode: str = _sem if _sem in ('confirmed_close', 'intrabar_wick') else 'confirmed_close'
        if self.signal_timing == 'live':
            self.sl_exit_mode = 'intrabar_wick'
        # Entry dedup: after_llm = one LLM+orchestrator run per closed signal bar (reject locks the bar).
        # after_order = only lock after a successful order; LLM re-runs each poll if rejected (higher Ollama load).
        _eds = str(_exec.get('entry_dedup_scope', 'after_llm') or 'after_llm').strip().lower()
        self.entry_dedup_scope: str = _eds if _eds in ('after_llm', 'after_order') else 'after_llm'
        # Intrabar P1: if classic signal is still NEUTRAL, allow entry when forming bar completes
        # the crossover pattern vs live EMA (previous closed bar must have touched EMA).
        self.early_ema_switch_entry: bool = bool(_exec.get('early_ema_switch_entry', False))
        self.early_switch_size_mult: float = max(1.0, float(_exec.get('early_switch_size_mult', 1.3)))
        # USDT linear: exchange leverage sets IM ≈ notional/leverage; notional = margin% × equity × leverage.
        _lm = str(_exec.get('leverage_mode', 'fixed') or 'fixed').strip().lower()
        self._leverage_mode: str = _lm if _lm in ('fixed', 'dynamic') else 'fixed'
        self._leverage_min: float = max(1.0, float(_exec.get('leverage_min', 1) or 1))
        _lmax_raw = _exec.get('leverage_max', _exec.get('leverage', 10))
        self._leverage_max: float = max(
            self._leverage_min,
            float(_lmax_raw if _lmax_raw is not None else 10) or 10,
        )
        if self._leverage_mode == 'fixed':
            self._order_leverage: float = max(
                1.0, float(_exec.get('leverage', 1) or 1)
            )
            self._leverage_max = self._order_leverage
            self._leverage_min = self._order_leverage
        else:
            self._order_leverage: float = self._leverage_max
        # Taker fee % per side (e.g. Bybit 0.055 = 0.055%). Round-trip ≈ 2×; use for min move vs fees.
        self._taker_fee_pct_per_side: float = max(
            0.0, float(_exec.get('taker_fee_pct_per_side', 0.055) or 0.0)
        )
        _fee_buf = max(1.0, float(_exec.get('fee_coverage_safety_buffer', 1.35) or 1.0))
        self._fee_coverage_safety_buffer: float = _fee_buf
        _side_frac = self._taker_fee_pct_per_side / 100.0
        self._roundtrip_taker_fee_frac: float = 2.0 * _side_frac
        self._min_price_edge_pct_after_fees: float = (
            self._roundtrip_taker_fee_frac * 100.0 * _fee_buf
        )
        self._fee_gate_skip_if_atr_low: bool = bool(
            _exec.get('fee_gate_skip_if_atr_low', True)
        )
        # Subtract model taker fees from PnL for peak trail / early giveback / journal extras (est.; not exchange audit).
        self._account_fees_in_pnl: bool = bool(
            _exec.get('account_taker_fees_in_pnl_logic', True)
        )

        # Trailing SL: early-window giveback protection + dynamic ATR trail (see _manage_position)
        _tr = _exec.get('trailing') if isinstance(_exec.get('trailing'), dict) else {}
        self._trail_early_window_sec: float = max(0.0, float(_tr.get('early_window_seconds', 150)))
        self._trail_early_breakeven_pct: float = max(0.0, float(_tr.get('early_breakeven_min_pct', 0.18)))
        self._trail_profit_floor_start_pct: float = max(0.1, float(_tr.get('profit_floor_start_pct', 1.0)))
        self._trail_early_atr_scale: float = max(0.45, min(1.0, float(_tr.get('early_window_atr_mult_scale', 0.68))))
        self._trail_giveback_mfe_frac: float = max(0.05, min(0.95, float(_tr.get('giveback_mfe_fraction', 0.38))))
        self._trail_giveback_lock_ratio: float = max(0.1, min(0.95, float(_tr.get('giveback_lock_of_mfe_ratio', 0.62))))
        self._trail_giveback_min_mfe_pct: float = max(0.05, float(_tr.get('giveback_min_mfe_price_pct', 0.22)))
        self._trail_close_on_early_giveback: bool = bool(_tr.get('close_on_early_giveback', True))
        self._trail_giveback_close_mfe_frac: float = max(0.15, min(0.98, float(_tr.get('giveback_close_mfe_fraction', 0.58))))
        self._trail_peak_usd_pullback: float = max(0.1, min(0.95, float(_tr.get('peak_usd_giveback_close_frac', 0.42))))
        self._trail_min_sl_move_usd: float = max(0.01, float(_tr.get('min_sl_move_usd', 0.50)))
        self._trail_min_sl_move_pct: float = max(1e-6, float(_tr.get('min_sl_move_price_pct', 0.0001)))
        self._trail_in_profit_min_move_scale: float = max(0.15, min(1.0, float(_tr.get('in_profit_min_move_scale', 0.40))))
        # Peak PnL% trail: once favorable price PnL% >= activate, track max PnL%; market-close if PnL% drops
        # by giveback (absolute percentage points) below that peak — ride trends, exit on first meaningful dip.
        self._peak_pnl_trail_enabled: bool = bool(_tr.get('peak_pnl_trail_enabled', False))
        self._peak_pnl_trail_activate_pct: float = max(
            0.0, float(_tr.get('peak_pnl_trail_activate_pct', 0.10))
        )
        self._peak_pnl_trail_giveback_pct: float = max(
            1e-6, float(_tr.get('peak_pnl_trail_giveback_pct', 0.05))
        )
        # Gross-only floor for peak trail activate (ATR gate uses fees separately). When net PnL drives trail, skip this bump.
        if (
            self._peak_pnl_trail_enabled
            and self._min_price_edge_pct_after_fees > 0
            and not self._account_fees_in_pnl
        ):
            self._peak_pnl_trail_activate_pct = max(
                self._peak_pnl_trail_activate_pct,
                self._min_price_edge_pct_after_fees,
            )

        # Position sizing — % of equity + USD cap; dynamic scales with conviction + optional daily target
        _sz = _exec.get('sizing') if isinstance(_exec.get('sizing'), dict) else {}
        _sm = str(_sz.get('mode', 'dynamic') or 'dynamic').strip().lower()
        self._sizing_mode: str = (
            _sm if _sm in ('dynamic', 'conservative', 'llm_direct') else 'dynamic'
        )
        self._sizing_small_threshold: float = float(_sz.get('small_account_equity_usd', 500))
        self._sizing_max_pct_small: float = float(_sz.get('max_pct_small_account', 25))
        self._sizing_max_pct: float = float(_sz.get('max_pct', 28))
        self._sizing_min_pct: float = float(_sz.get('min_pct', 2))
        self._sizing_conviction_power: float = float(_sz.get('conviction_power', 1.15))
        self._sizing_llm_weight: float = max(0.0, min(1.0, float(_sz.get('llm_weight', 0.32))))
        _mtu = _sz.get('max_trade_usd', 15000)
        if _mtu is None or (isinstance(_mtu, (int, float)) and float(_mtu) <= 0):
            self._sizing_max_trade_usd = float('inf')
        else:
            self._sizing_max_trade_usd = max(0.0, float(_mtu))
        self._sizing_max_equity_fraction: float = max(
            0.05, min(1.0, float(_sz.get('max_equity_fraction', 0.42)))
        )
        self._sizing_target_trades: int = max(1, int(_sz.get('target_successful_trades_per_day', 10)))
        self._sizing_target_profit_per_trade_usd: float = max(
            0.0, float(_sz.get('target_profit_usd_per_trade', 20))
        )
        _plan_daily = self._sizing_target_trades * self._sizing_target_profit_per_trade_usd
        _dpt_raw = _sz.get('daily_profit_target_usd')
        if _dpt_raw is not None and float(_dpt_raw) > 0:
            self._sizing_daily_target_usd = max(0.0, float(_dpt_raw))
            self._sizing_daily_target_is_override = True
        else:
            self._sizing_daily_target_usd = _plan_daily
            self._sizing_daily_target_is_override = False
        self._sizing_daily_ramp_max_mult: float = max(
            1.0, min(2.5, float(_sz.get('daily_ramp_max_mult', 1.45)))
        )

        # Risk settings
        risk = config.get('risk', {})
        self.daily_loss_limit_pct: float = risk.get('daily_loss_limit_pct', 3.0)
        self.atr_stop_mult: float = float(risk.get('atr_stop_mult', 1.5))
        # If SL fills, target max loss ≈ this % of account equity at entry (before bounce relief).
        self.max_loss_per_trade_equity_pct: float = float(
            risk.get('max_loss_per_trade_equity_pct', 1.0)
        )
        # When OB shows bid/ask wall SL or strong imbalance, allow this × equity loss budget for SL placement.
        self.bounce_loss_relief_mult: float = max(
            1.0, float(risk.get('bounce_loss_relief_mult', 1.25))
        )
        # Flat position if unrealized loss exceeds (capped SL loss USD) × this (gap/slippage buffer).
        self.hard_stop_loss_slippage_mult: float = max(
            1.0, float(risk.get('hard_stop_loss_slippage_mult', 1.15))
        )
        # Optional hard USD ceiling on loss per trade (see _max_loss_price_move_usd).
        _mlusd_raw = risk.get('max_loss_usd_total')
        if _mlusd_raw is None:
            self.max_loss_usd_total: Optional[float] = None
        else:
            _mlv = float(_mlusd_raw)
            self.max_loss_usd_total = _mlv if _mlv > 0 else None

        # AI / LLM settings — local Ollama first (OLLAMA_HOST / BASE), then tunnel (OLLAMA_REMOTE_URL)
        ai_cfg = config.get('ai', {})
        _ollama_local = (
            os.environ.get('OLLAMA_HOST', '').strip()
            or os.environ.get('OLLAMA_BASE_URL', '').strip()
        )
        _ollama_remote = os.environ.get('OLLAMA_REMOTE_URL', '').strip()
        _ollama_cfg = (ai_cfg.get('ollama_base_url') or 'http://127.0.0.1:11434').strip()
        if _ollama_local:
            self.ollama_base_url = _ollama_local.rstrip('/')
        elif _ollama_remote:
            self.ollama_base_url = _ollama_remote.rstrip('/')
        else:
            self.ollama_base_url = _ollama_cfg.rstrip('/') or 'http://127.0.0.1:11434'

        _h = (urlparse(self.ollama_base_url).hostname or '').lower()
        _using_local_ollama = _h in ('127.0.0.1', 'localhost', '0.0.0.0', '')
        if _using_local_ollama:
            self.ollama_model = (
                os.environ.get('OLLAMA_MODEL', '').strip()
                or ai_cfg.get('reasoning_model', 'mistral:latest')
            )
        else:
            self.ollama_model = (
                os.environ.get('OLLAMA_REMOTE_MODEL', '').strip()
                or os.environ.get('OLLAMA_MODEL', '').strip()
                or ai_cfg.get('reasoning_model', 'mistral:latest')
            )
        self.llm_conf_threshold: float = ai_cfg.get('llm_trade_conf_threshold', 0.40)

        # Claude API settings (used for Delta Exchange)
        self._claude_api_key: str = os.environ.get('ANTHROPIC_API_KEY', '')
        self._claude_model: str = ai_cfg.get('claude_model', 'claude-haiku-4-5')
        self._claude_client = None
        # Use Claude for this exchange? Check per-exchange override or exchange name
        _ex_name = config.get('exchange', {}).get('name', 'bybit').lower()
        exchange_llm = ai_cfg.get('llm_provider', 'auto')  # 'ollama', 'claude', 'auto'
        if exchange_llm == 'claude':
            self._use_claude = True
        elif exchange_llm == 'ollama':
            self._use_claude = False
        else:
            # Auto: use Claude for Delta, Ollama for everything else
            self._use_claude = (_ex_name == 'delta')

        if self._use_claude and ANTHROPIC_AVAILABLE and self._claude_api_key:
            self._claude_client = anthropic.Anthropic(api_key=self._claude_api_key)
            print(f"  [AI] Claude API enabled ({self._claude_model}) for {_ex_name.upper()}")
        elif self._use_claude and not self._claude_api_key:
            print(f"  [AI] WARNING: Claude requested but ANTHROPIC_API_KEY not set -- falling back to Ollama")
            self._use_claude = False
        elif self._use_claude and not ANTHROPIC_AVAILABLE:
            print(f"  [AI] WARNING: anthropic SDK not installed -- falling back to Ollama")
            self._use_claude = False

        # Quality gates (spec §4–§5)
        self.min_confidence: float = float(ai_cfg.get('llm_min_confidence', 0.50))
        self.llm_fallback_confidence: float = float(ai_cfg.get('llm_fallback_confidence', 0.75))
        self.llm_fallback_confidence_gate: float = float(
            ai_cfg.get('llm_fallback_confidence_gate', self.llm_fallback_confidence)
        )
        # When true: do not apply server-side LLM overrides (tq chop, size halving, conf boost/cap) — gate on confidence only.
        self._confidence_only_gates: bool = bool(ai_cfg.get('confidence_only_gates', False))
        self.min_atr_ratio: float = 0.0003
        self.trade_cooldown: float = float(config.get('trade_cooldown_seconds', 30))
        self.post_close_cooldown: float = 180.0   # spec §4.4: 180s after close

        # Exchange
        exchange_name = config.get('exchange', {}).get('name', 'bybit')
        testnet = config.get('mode', 'testnet') in ('testnet', 'paper')
        self.price_source = PriceFetcher(exchange_name=exchange_name, testnet=testnet)

        # LLM strategist — same model/endpoint as trade-time Ollama (env + config)
        provider = ai_cfg.get('reasoning_provider', 'auto')
        use_local = ai_cfg.get('use_local_on_failure', False)
        self.strategist = AgenticStrategist(
            provider=provider,
            model=self.ollama_model,
            use_local_on_failure=use_local,
        )

        # Order book microstructure analyzer
        self.microstructure = MicrostructureAnalyzer(depth=20)

        # Journal
        self.journal = TradeJournal()

        # Exchange tag for output (prevents interleaved confusion in multi-exchange mode)
        self._ex_tag: str = config.get('exchange', {}).get('name', 'bybit').upper()

        try:
            if self._exchange_client:
                _lev = int(round(min(self._order_leverage, 125)))
                for asset in self.assets:
                    sym = self._get_symbol(asset)  # Uses correct format per exchange
                    try:
                        self._exchange_client.exchange.set_leverage(_lev, sym)
                        print(
                            f"  [{self._ex_tag}:{asset}] Leverage set to {_lev}x on {self._exchange_name} "
                            f"(config execution.leverage)"
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        # State — equity/cash filled only from broker (see _sync_balance_from_broker)
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity: float = 0.0
        self.cash: float = 0.0
        self.bar_count: int = 0
        self.last_trade_time: Dict[str, float] = {}
        self.last_close_time: Dict[str, float] = {}
        self.last_signal_candle: Dict[str, float] = {}  # Track candle timestamp to avoid re-entry on same candle
        self.failed_close_assets: Dict[str, float] = {}  # Assets that failed to close — skip until manual resolution

        # Bear/Risk veto agent
        self.bear_enabled: bool = ai_cfg.get('bear_agent_enabled', True)
        self.bear_veto_threshold: int = int(ai_cfg.get('bear_veto_threshold', 10))
        self.bear_reduce_threshold: int = int(ai_cfg.get('bear_reduce_threshold', 8))
        self.bear_veto_stats: Dict[str, Dict[str, int]] = {}
        for a in self.assets:
            self.bear_veto_stats[a] = {'vetoed': 0, 'reduced': 0, 'passed': 0}

        # ── Portfolio-level drawdown limit (Freqtrade pattern) ──
        # Halt ALL trading if cumulative realized losses exceed threshold
        self.max_drawdown_pct: float = risk.get('max_drawdown_pct', 10.0)
        self.daily_loss_limit_pct: float = risk.get('daily_loss_limit_pct', 3.0)
        self.session_start_equity: float = 0.0
        self._session_baseline_written: bool = False
        self.session_realized_pnl: float = 0.0
        self.daily_realized_pnl: float = 0.0
        self._session_unrealized_pnl: float = 0.0  # refreshed each bar for sizing / logs
        self.daily_reset_date: str = datetime.utcnow().strftime('%Y-%m-%d')
        self.trading_halted: bool = False
        self.halt_reason: str = ""

        # ── Time-based exits (Freqtrade pattern) ──
        # Close positions held longer than max_hold_minutes
        self.max_hold_minutes: float = float(config.get('max_hold_minutes', 360))  # 6 hours default

        # ── Facilitator agent (TradingAgents pattern) ──
        self.facilitator_enabled: bool = ai_cfg.get('facilitator_enabled', True)

        # ── 3 Risk Personas (TradingAgents pattern) ──
        self.risk_personas_enabled: bool = ai_cfg.get('risk_personas_enabled', True)

        # ── Edge Positioning (Freqtrade pattern) ──
        # Dynamic sizing based on historical win rate per asset
        self.edge_enabled: bool = ai_cfg.get('edge_positioning_enabled', True)
        self.edge_stats: Dict[str, Dict[str, float]] = {}
        self._load_edge_stats()

        # ── Agent Orchestrator + Debate Engine ──
        # 10 specialized math agents + adversarial debate → feeds into LLM prompt
        self.orchestrator_enabled: bool = ai_cfg.get('orchestrator_enabled', True)
        # Spec mode: agents inform LLM only — no entry veto / no confidence dilution
        self.orchestrator_entry_veto: bool = bool(ai_cfg.get('orchestrator_entry_veto', False))
        self.orchestrator_blend_confidence: bool = bool(ai_cfg.get('orchestrator_blend_confidence', False))
        self._orchestrator = None
        self._math_injector = None
        if self.orchestrator_enabled and ORCHESTRATOR_AVAILABLE:
            try:
                self._orchestrator = AgentOrchestrator(config)
                self._math_injector = MathInjector(config)
                n_agents = len(self._orchestrator.agents)
                print(f"  [AGENTS] Orchestrator + Debate Engine loaded ({n_agents} agents)")
            except Exception as e:
                logger.warning(f"Orchestrator init failed (degraded): {e}")
                self._orchestrator = None

        # ── On-Chain Data Fetcher (free APIs) ──
        self._onchain = None
        if ONCHAIN_AVAILABLE:
            try:
                self._onchain = OnChainFetcher()
                print(f"  [DATA] On-chain fetcher loaded (BTC hashrate, mempool, DeFi TVL)")
            except Exception as e:
                logger.warning(f"On-chain fetcher init failed: {e}")

        # ── Memory Vault (trade experience database) ──
        self._memory = None
        if MEMORY_AVAILABLE:
            try:
                self._memory = MemoryVault()
                print(f"  [MEMORY] Trade memory vault loaded (semantic search)")
            except Exception as e:
                logger.warning(f"Memory vault init failed (needs sentence-transformers): {e}")

        self._sync_balance_from_broker()
        if self.equity <= 0 and not self._exchange_client:
            print(f"  [BROKER] No exchange client — equity unset until Bybit/Delta connects (not using config initial_capital)")

    def _broker_session_file(self) -> str:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(root, "logs", "broker_session.json")

    def _persist_broker_session_baseline(self, equity: float) -> None:
        path = self._broker_session_file()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_start_equity": equity,
                    "exchange": self._exchange_name,
                    "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Session baseline write failed: {e}")

    def _sync_balance_from_broker(self) -> None:
        """Set equity and cash only from exchange get_account() — no config NAV fallback."""
        try:
            client = self._exchange_client
            if not client:
                return
            acct = client.get_account()
            if not acct or acct.get("error"):
                return
            self.equity = float(acct.get("equity", 0) or 0)
            self.cash = float(acct.get("cash", 0) or 0)
            if not self._session_baseline_written:
                self.session_start_equity = self.equity
                self._persist_broker_session_baseline(self.equity)
                self._session_baseline_written = True
        except Exception as e:
            logger.debug(f"Broker balance sync failed: {e}")

    # ------------------------------------------------------------------
    # Agent Orchestrator — run 10 math agents + debate, format for LLM
    # ------------------------------------------------------------------
    def _run_orchestrator(self, asset: str, price: float, signal: str,
                          closes: list, highs: list, lows: list,
                          opens: list, volumes: list,
                          ema_vals: list = None, atr_vals: list = None,
                          ema_direction: str = '') -> Optional[Dict]:
        """
        Run 10-agent orchestrator + debate engine, filtered through
        our CALL/PUT EMA crossover + trailing SL strategy.
        Every agent sees strategy context so they vote on:
        'Will this crossover reach L10+ or just L1-L2 then stop?'
        """
        if not self._orchestrator or not self._math_injector:
            return None

        try:
            import numpy as np

            # Build quant_state from MathInjector
            quant_state = self._math_injector.compute_full_state(
                prices=np.array(closes[-200:], dtype=float),
                highs=np.array(highs[-200:], dtype=float),
                lows=np.array(lows[-200:], dtype=float),
                volumes=np.array(volumes[-200:], dtype=float),
                asset=f"{asset}USDT",
                account_balance=self.equity,
            )

            # ── Inject CALL/PUT strategy context so agents know what we're doing ──
            ema_slope = 0.0
            if ema_vals and len(ema_vals) >= 3 and ema_vals[-3] > 0:
                ema_slope = (ema_vals[-1] - ema_vals[-3]) / ema_vals[-3] * 100
            current_atr_v = atr_vals[-1] if atr_vals and len(atr_vals) > 0 else 0
            current_ema_v = ema_vals[-1] if ema_vals and len(ema_vals) > 0 else price
            ema_distance_pct = abs(price - current_ema_v) / current_ema_v * 100 if current_ema_v > 0 else 0

            quant_state['strategy'] = {
                'name': 'EMA_CROSSOVER_TRAILING_SL',
                'entry_type': 'CALL' if signal == 'BUY' else 'PUT',
                'signal': signal,
                'ema_period': self.ema_period,
                'ema_direction': ema_direction,
                'ema_slope_pct': round(ema_slope, 4),
                'ema_distance_pct': round(ema_distance_pct, 3),
                'atr': round(current_atr_v, 2) if current_atr_v else 0,
                'trailing_sl': 'L1->Ln progressive (profit becomes investment)',
                'goal': 'Find crossovers that reach L10+ trailing SL levels',
                'capital_rule': 'Never risk more than SL distance. Safety first.',
                'profit_rule': 'Once profitable, SL locks gains at 40/50/60/70%',
            }

            raw_signal = 1 if signal == "BUY" else (-1 if signal == "SELL" else 0)

            # Build OHLCV context
            ohlcv_ctx = {
                'closes': closes[-50:],
                'highs': highs[-50:],
                'lows': lows[-50:],
                'opens': opens[-50:],
                'volumes': volumes[-50:],
            }

            # Fetch on-chain data (whale flows, mempool, DeFi TVL)
            on_chain_data = {}
            if self._onchain and asset == 'BTC':
                try:
                    on_chain_data = self._onchain.get_market_context(asset=asset, current_price=price)
                except Exception:
                    pass  # On-chain is optional, don't block trading

            # Run full orchestrator pipeline (agents + debate + combine + audit)
            decision = self._orchestrator.run_cycle(
                quant_state=quant_state,
                raw_signal=raw_signal,
                raw_confidence=0.5,
                on_chain=on_chain_data,
                ohlcv_data=ohlcv_ctx,
                asset=asset,
                daily_pnl=self.daily_realized_pnl,
                account_balance=self.equity,
                open_positions=list(self.positions.values()),
                trade_history=[],
            )

            # ── Strategy Filter: Interpret agent votes for CALL/PUT trailing SL ──
            result = self._strategy_filter(decision, signal, ema_slope, ema_distance_pct, asset)
            return result

        except Exception as e:
            logger.warning(f"[{asset}] Orchestrator failed (degraded): {e}")
            return None

    # ------------------------------------------------------------------
    # Strategy Filter — reinterpret agent votes for CALL/PUT trailing SL
    # ------------------------------------------------------------------
    def _strategy_filter(self, decision, signal: str, ema_slope: float,
                         ema_distance_pct: float, asset: str) -> Optional[Dict]:
        """
        Reinterpret EnhancedDecision through CALL/PUT EMA crossover + trailing SL lens.

        Each agent's vote is scored by:
        - Will this crossover reach L10+ trailing SL levels?
        - Is capital protected (SL distance reasonable)?
        - Can trailing SL lock gains (strong trend vs choppy)?

        Returns dict with consensus, confidence, position_scale, agent_summary, veto,
        plus strategy-specific scores: trend_reach_score, safety_score, profit_lock_score.
        """
        if decision is None:
            return None

        votes = decision.agent_votes  # Dict[str, AgentVote]
        if not votes:
            return None

        # ── Strategy relevance weights per agent ──
        STRATEGY_WEIGHTS = {
            'trend_momentum':       1.5,   # Core: will the trend continue to L10+?
            'regime_intelligence':  1.4,   # Trending regime = good for us
            'trade_timing':         1.3,   # Are we entering at the crossover, not late?
            'risk_guardian':        1.3,   # Capital safety, SL adequacy
            'loss_prevention':      1.2,   # Streak/drawdown protection
            'market_structure':     1.1,   # Support/resistance awareness
            'pattern_matcher':      1.0,   # Historical pattern similarity
            'mean_reversion':       0.8,   # Counter-trend warning (lower weight)
            'portfolio_optimizer':  0.7,   # Portfolio-level (less relevant)
            'sentiment_decoder':    0.6,   # Sentiment (indirect)
            'polymarket_arb':       0.5,   # Arbitrage (least relevant)
            'data_integrity':       1.0,   # Data quality gate
            'decision_auditor':     1.0,   # Audit gate
        }

        trend_scores = []
        safety_scores = []
        profit_lock_scores = []
        agent_summary = []
        weighted_directions = []
        total_weight = 0.0
        veto = False

        for agent_name, vote in votes.items():
            w = STRATEGY_WEIGHTS.get(agent_name, 1.0)

            if vote.veto:
                veto = True

            # ── Strategy interpretation per agent type ──
            if 'trend' in agent_name or 'momentum' in agent_name:
                reach = vote.confidence * (1.0 if vote.direction != 0 else 0.3)
                if ema_slope > 0.2:
                    reach = min(1.0, reach * 1.2)
                trend_scores.append(reach * w)
                safety_scores.append(vote.position_scale * w)
                label = f"TREND L{'10+' if reach > 0.7 else '5-9' if reach > 0.4 else '1-4'}"

            elif 'regime' in agent_name:
                is_trending = vote.confidence > 0.5 and vote.direction != 0
                reach = vote.confidence if is_trending else vote.confidence * 0.3
                trend_scores.append(reach * w)
                profit_lock_scores.append((1.0 if is_trending else 0.3) * w)
                label = f"REGIME {'TREND' if is_trending else 'RANGE'}"

            elif 'timing' in agent_name:
                timing_quality = max(0, 1.0 - ema_distance_pct / 2.0)
                combined = (vote.confidence * 0.6 + timing_quality * 0.4)
                safety_scores.append(combined * w)
                trend_scores.append(combined * w * 0.5)
                label = f"TIMING {'EARLY' if timing_quality > 0.7 else 'OK' if timing_quality > 0.4 else 'LATE'}"

            elif 'risk' in agent_name or 'loss' in agent_name:
                safety_scores.append(vote.confidence * w)
                profit_lock_scores.append(vote.position_scale * w)
                label = f"SAFETY {'OK' if vote.confidence > 0.5 else 'WARN'}"

            elif 'mean_reversion' in agent_name:
                dir_match = (vote.direction > 0 and signal == 'BUY') or \
                            (vote.direction < 0 and signal == 'SELL') or \
                            vote.direction == 0
                if not dir_match and vote.confidence > 0.6:
                    trend_scores.append(-0.3 * w)
                    label = "MR WARNS REVERSAL"
                else:
                    trend_scores.append(0.2 * w)
                    label = f"MR {'OK' if dir_match else 'MILD'}"

            elif 'structure' in agent_name:
                profit_lock_scores.append(vote.confidence * w)
                trend_scores.append(vote.confidence * 0.5 * w)
                label = f"STRUCT {'CLEAR' if vote.confidence > 0.6 else 'BLOCKED'}"

            elif 'pattern' in agent_name:
                trend_scores.append(vote.confidence * 0.5 * w)
                label = f"PATTERN conf={vote.confidence:.0%}"

            else:
                trend_scores.append(vote.confidence * 0.3 * w)
                label = f"{agent_name.upper()[:8]}"

            weighted_directions.append(vote.direction * w * vote.confidence)
            total_weight += w

            dir_str = "CALL" if vote.direction > 0 else "PUT" if vote.direction < 0 else "FLAT"
            agent_summary.append(
                f"  {agent_name}: {dir_str} conf={vote.confidence:.0%} scale={vote.position_scale:.0%} | {label} | {vote.reasoning[:60]}"
            )

        # ── Compute strategy scores ──
        trend_reach_score = max(0.0, min(1.0, sum(trend_scores) / max(1, len(trend_scores))))
        safety_score = max(0.0, min(1.0, sum(safety_scores) / max(1, len(safety_scores)))) if safety_scores else 0.5
        profit_lock_score = max(0.0, min(1.0, sum(profit_lock_scores) / max(1, len(profit_lock_scores)))) if profit_lock_scores else 0.5

        # Composite: 50% trend reach + 30% safety + 20% profit lock
        strategy_confidence = trend_reach_score * 0.50 + safety_score * 0.30 + profit_lock_score * 0.20

        # ── Consensus direction ──
        net_direction = sum(weighted_directions) / total_weight if total_weight > 0 else 0
        if net_direction > 0.15:
            consensus_dir, direction_int = "CALL", 1
        elif net_direction < -0.15:
            consensus_dir, direction_int = "PUT", -1
        else:
            consensus_dir, direction_int = "FLAT", 0

        abs_net = abs(net_direction)
        consensus = "STRONG" if abs_net > 0.6 else "MODERATE" if abs_net > 0.3 else "WEAK" if abs_net > 0.15 else "CONFLICT"

        # ── Position scale ──
        position_scale = decision.position_scale
        if trend_reach_score < 0.3:
            position_scale = min(position_scale, 0.5)
        if safety_score < 0.3:
            position_scale = min(position_scale, 0.3)

        # ── L-level prediction ──
        if trend_reach_score > 0.7:
            l_prediction = "L10+"
        elif trend_reach_score > 0.5:
            l_prediction = "L5-L9"
        elif trend_reach_score > 0.3:
            l_prediction = "L3-L5"
        else:
            l_prediction = "L1-L2"

        agent_summary.insert(0,
            f"  STRATEGY: {consensus_dir} | trend_reach={trend_reach_score:.0%}({l_prediction}) "
            f"safety={safety_score:.0%} profit_lock={profit_lock_score:.0%} | "
            f"EMA_slope={ema_slope:+.3f}% dist={ema_distance_pct:.2f}%"
        )

        print(f"  [{self._ex_tag}:{asset}] STRATEGY FILTER: {consensus_dir} {consensus} | "
              f"reach={trend_reach_score:.0%}({l_prediction}) safe={safety_score:.0%} lock={profit_lock_score:.0%} | "
              f"conf={strategy_confidence:.2f} scale={position_scale:.2f}")

        return {
            'consensus': consensus,
            'consensus_dir': consensus_dir,
            'direction': direction_int,
            'confidence': strategy_confidence,
            'position_scale': position_scale,
            'agent_count': len(votes),
            'agent_summary': agent_summary,
            'debate_summary': decision.strategy_recommendation[:100] if decision.strategy_recommendation else 'N/A',
            'data_quality': decision.data_quality,
            'veto': veto or decision.veto,
            'agent_votes': {k: {'dir': v.direction, 'conf': v.confidence} for k, v in votes.items()},
            'trend_reach_score': trend_reach_score,
            'safety_score': safety_score,
            'profit_lock_score': profit_lock_score,
            'l_prediction': l_prediction,
        }

    # ------------------------------------------------------------------
    # Edge Positioning — load historical win/loss stats per asset
    # ------------------------------------------------------------------
    def _load_edge_stats(self):
        """Load win rate and expectancy from trade journal for position sizing."""
        for asset in self.assets:
            self.edge_stats[asset] = {
                'wins': 0, 'losses': 0, 'total': 0,
                'win_rate': 0.5, 'avg_win': 0.0, 'avg_loss': 0.0,
                'expectancy': 0.0, 'edge_multiplier': 1.0,
            }
        try:
            trades = self.journal.load_trades()
            for t in trades[-200:]:  # last 200 trades
                a = t.get('asset', '')
                if a not in self.edge_stats:
                    continue
                pnl = float(t.get('pnl_pct', 0))
                self.edge_stats[a]['total'] += 1
                if pnl > 0:
                    self.edge_stats[a]['wins'] += 1
                    self.edge_stats[a]['avg_win'] += pnl
                else:
                    self.edge_stats[a]['losses'] += 1
                    self.edge_stats[a]['avg_loss'] += abs(pnl)

            for a in self.assets:
                s = self.edge_stats[a]
                if s['total'] < 5:
                    continue  # Not enough data
                s['win_rate'] = s['wins'] / s['total'] if s['total'] > 0 else 0.5
                s['avg_win'] = s['avg_win'] / s['wins'] if s['wins'] > 0 else 0
                s['avg_loss'] = s['avg_loss'] / s['losses'] if s['losses'] > 0 else 0
                # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
                s['expectancy'] = (s['win_rate'] * s['avg_win']) - ((1 - s['win_rate']) * s['avg_loss'])
                # Edge multiplier: scale position size by expectancy
                # Positive expectancy > 1x, negative < 1x, minimum 0.25x
                if s['expectancy'] > 0:
                    s['edge_multiplier'] = min(2.0, 1.0 + s['expectancy'] / 5.0)
                else:
                    s['edge_multiplier'] = max(0.25, 1.0 + s['expectancy'] / 5.0)
                print(f"  [{self._ex_tag}:{a}] EDGE: {s['wins']}W/{s['losses']}L rate={s['win_rate']:.0%} exp={s['expectancy']:+.2f} mult={s['edge_multiplier']:.2f}x")
        except Exception as e:
            logger.debug(f"Edge stats load failed: {e}")

    # ------------------------------------------------------------------
    # Historical L-Level Pattern Analysis (feeds into LLM prompt)
    # ------------------------------------------------------------------
    def _build_historical_pattern_context(self, asset: str) -> str:
        """
        Analyze closed trades from journal to build a statistical summary
        of L-level exits, win rates, and winning vs losing entry conditions.
        This gets injected into the unified LLM prompt so Mistral knows
        what historically works and what doesn't.

        Returns a multi-line string for direct prompt embedding.
        """
        try:
            all_trades = self.journal.load_trades(asset=asset)
            # Only use closed trades with PnL (last 300 for recency)
            closed = [t for t in all_trades if t.get('pnl_usd') is not None][-300:]
            if len(closed) < 10:
                return f"[{asset}] Insufficient history ({len(closed)} trades) — no pattern data."

            # ── Categorize exits by L-level ──
            l_stats = {}   # {'L1': {'count': N, 'wins': N, 'total_pnl': X}, ...}
            hard_stop = {'count': 0, 'total_pnl': 0.0}
            generic_sl = {'count': 0, 'wins': 0, 'total_pnl': 0.0}
            winners = []
            losers = []

            for t in closed:
                pnl = float(t.get('pnl_usd', 0))
                exit_reason = str(t.get('exit_reason', '')).lower()
                sl_prog = str(t.get('sl_progression', ''))
                conf = float(t.get('confidence', 0))
                is_win = pnl > 0

                # Determine highest L-level reached
                highest_l = 0
                if sl_prog:
                    import re
                    l_matches = re.findall(r'L(\d+)', sl_prog)
                    if l_matches:
                        highest_l = max(int(x) for x in l_matches)

                # Categorize
                if 'hard stop' in exit_reason or 'hard_stop' in exit_reason:
                    hard_stop['count'] += 1
                    hard_stop['total_pnl'] += pnl
                elif highest_l > 0:
                    bucket = f"L{highest_l}" if highest_l <= 5 else f"L6+"
                    if bucket not in l_stats:
                        l_stats[bucket] = {'count': 0, 'wins': 0, 'total_pnl': 0.0}
                    l_stats[bucket]['count'] += 1
                    if is_win:
                        l_stats[bucket]['wins'] += 1
                    l_stats[bucket]['total_pnl'] += pnl
                else:
                    generic_sl['count'] += 1
                    if is_win:
                        generic_sl['wins'] += 1
                    generic_sl['total_pnl'] += pnl

                # Collect entry stats for winners vs losers
                entry_data = {
                    'confidence': conf,
                    'highest_l': highest_l,
                    'pnl': pnl,
                }
                if is_win:
                    winners.append(entry_data)
                else:
                    losers.append(entry_data)

            # ── Build summary ──
            total = len(closed)
            n_wins = len(winners)
            win_rate = n_wins / total * 100 if total > 0 else 0
            avg_win_conf = sum(w['confidence'] for w in winners) / n_wins if n_wins > 0 else 0
            avg_lose_conf = sum(l['confidence'] for l in losers) / len(losers) if losers else 0

            lines = []
            lines.append(f"=== HISTORICAL PATTERNS ({asset}, last {total} trades) ===")
            lines.append(f"Overall: {n_wins}W/{total - n_wins}L ({win_rate:.0f}% win rate)")
            lines.append(f"Winners avg entry confidence: {avg_win_conf:.2f} | Losers avg: {avg_lose_conf:.2f}")
            lines.append("")

            # L-level breakdown
            lines.append("EXIT LEVEL PERFORMANCE (our trailing SL L1→Ln):")
            for lev in ['L1', 'L2', 'L3', 'L4', 'L5', 'L6+']:
                if lev in l_stats:
                    s = l_stats[lev]
                    wr = s['wins'] / s['count'] * 100 if s['count'] > 0 else 0
                    avg_pnl = s['total_pnl'] / s['count'] if s['count'] > 0 else 0
                    lines.append(f"  {lev}: {s['count']} trades, {wr:.0f}% win, avg ${avg_pnl:+.2f}")

            if hard_stop['count'] > 0:
                avg_hs = hard_stop['total_pnl'] / hard_stop['count']
                lines.append(f"  Hard Stop -2%: {hard_stop['count']} trades, 0% win, avg ${avg_hs:+.2f} ← CATASTROPHIC")

            if generic_sl['count'] > 0:
                wr = generic_sl['wins'] / generic_sl['count'] * 100
                avg_gs = generic_sl['total_pnl'] / generic_sl['count']
                lines.append(f"  Generic SL: {generic_sl['count']} trades, {wr:.0f}% win, avg ${avg_gs:+.2f}")

            lines.append("")

            # Key insights for the LLM
            lines.append("KEY INSIGHTS FOR DECISION:")
            # High L-level trades
            l5_plus = l_stats.get('L6+', {'count': 0, 'wins': 0})
            l4 = l_stats.get('L4', {'count': 0, 'wins': 0})
            l5 = l_stats.get('L5', {'count': 0, 'wins': 0})
            big_winners = l5_plus['count'] + l4.get('count', 0) + l5.get('count', 0)
            big_win_wins = l5_plus['wins'] + l4.get('wins', 0) + l5.get('wins', 0)
            if big_winners > 0:
                big_wr = big_win_wins / big_winners * 100
                lines.append(f"  - Trades reaching L4+ have {big_wr:.0f}% win rate ({big_winners} trades) ← THESE ARE PROFITABLE")
            lines.append(f"  - Most trades die at L1-L2 or hard stop ← THESE LOSE MONEY")
            lines.append(f"  - Winners enter with avg confidence {avg_win_conf:.2f}, losers {avg_lose_conf:.2f}")
            if avg_win_conf > avg_lose_conf + 0.1:
                lines.append(f"  - Confidence gap: {avg_win_conf - avg_lose_conf:.2f} — HIGH confidence signals are significantly better")
            lines.append(f"  - ONLY approve trades you believe will reach L4+ (strong trend, not choppy)")
            lines.append(f"  - REJECT trades that look like L1-L2 exits (low slope, weak volume, no trend)")

            return chr(10).join(lines)

        except Exception as e:
            logger.debug(f"Historical pattern build failed for {asset}: {e}")
            return f"[{asset}] Pattern history unavailable — be extra cautious."

    # ------------------------------------------------------------------
    # Portfolio Drawdown Check
    # ------------------------------------------------------------------
    def _check_drawdown_limits(self) -> bool:
        """Check if trading should be halted due to drawdown limits.
        Returns True if trading is OK, False if halted."""
        # Daily reset
        today = datetime.utcnow().strftime('%Y-%m-%d')
        if today != self.daily_reset_date:
            self.daily_realized_pnl = 0.0
            self.daily_reset_date = today
            if self.trading_halted and 'daily' in self.halt_reason:
                self.trading_halted = False
                self.halt_reason = ""
                print(f"  [{self._ex_tag}] DRAWDOWN: Daily limit reset — trading resumed")

        # Check daily loss limit
        if self.session_start_equity > 0:
            daily_loss_pct = abs(self.daily_realized_pnl) / self.session_start_equity * 100
            if self.daily_realized_pnl < 0 and daily_loss_pct >= self.daily_loss_limit_pct:
                self.trading_halted = True
                self.halt_reason = f"daily loss {daily_loss_pct:.1f}% >= {self.daily_loss_limit_pct}%"
                return False

        # Check max drawdown from session start
        if self.session_start_equity > 0:
            session_dd_pct = abs(self.session_realized_pnl) / self.session_start_equity * 100
            if self.session_realized_pnl < 0 and session_dd_pct >= self.max_drawdown_pct:
                self.trading_halted = True
                self.halt_reason = f"max drawdown {session_dd_pct:.1f}% >= {self.max_drawdown_pct}%"
                return False

        return True

    def _sizing_effective_caps(self) -> Tuple[float, float, float]:
        """One new entry: (max_pct_of_equity, max_notional_usd, equity_fraction for cap)."""
        eq = max(self.equity, 0.0)
        lev = max(1.0, float(self._order_leverage))
        if self._sizing_mode == 'conservative':
            max_pct = 20.0 if eq < self._sizing_small_threshold else 5.0
            max_usd = min(2000.0, eq * 0.25 * lev)
            return max_pct, max_usd, 0.25
        max_pct = (
            self._sizing_max_pct_small
            if eq < self._sizing_small_threshold
            else self._sizing_max_pct
        )
        cap_from_eq = eq * self._sizing_max_equity_fraction * lev
        if math.isinf(self._sizing_max_trade_usd):
            max_usd = cap_from_eq
        else:
            max_usd = min(self._sizing_max_trade_usd, cap_from_eq)
        return max_pct, max_usd, self._sizing_max_equity_fraction

    def _leverage_for_trade(self, confidence: float, unified: dict) -> float:
        """Fixed: config leverage. Dynamic: interpolate between leverage_min and leverage_max from LLM signals."""
        if self._leverage_mode == 'fixed':
            return max(1.0, float(self._order_leverage))
        conf = max(0.0, min(1.0, float(confidence)))
        tq = max(0.0, min(10.0, float(unified.get('trade_quality', 5))))
        blend = 0.55 * conf + 0.45 * (tq / 10.0)
        span = self._leverage_max - self._leverage_min
        lev = self._leverage_min + span * blend
        lev = int(round(max(self._leverage_min, min(self._leverage_max, lev))))
        return max(1, lev)

    def _set_exchange_leverage_symbol(self, asset: str, leverage: float) -> None:
        try:
            if not self._exchange_client:
                return
            sym = self._get_symbol(asset)
            lv = int(round(min(max(1.0, float(leverage)), 125)))
            self._exchange_client.exchange.set_leverage(lv, sym)
        except Exception:
            pass

    def _coin_mult_for_position(self, asset: str, contract_size: float) -> float:
        cs = float(contract_size or 0.0)
        if cs > 0:
            return cs
        if self._exchange_name == 'delta':
            return {'BTC': 0.001, 'ETH': 0.01}.get(asset, 0.001)
        return 1.0

    def _notional_usd(self, qty: float, px: float, asset: str, contract_size: float) -> float:
        m = self._coin_mult_for_position(asset, contract_size)
        return abs(float(qty)) * float(px) * m

    def _est_taker_fee_usd(self, notional_usd: float) -> float:
        if self._taker_fee_pct_per_side <= 0:
            return 0.0
        return max(0.0, float(notional_usd)) * (self._taker_fee_pct_per_side / 100.0)

    def _max_loss_price_move_usd(
        self,
        eq_ref: float,
        relief: float,
        entry_notional_usd: float,
    ) -> float:
        """
        Max adverse *price* PnL at SL in USD (gross move vs entry), before exit taker fee.
        Uses min(equity % budget, optional max_loss_usd_total); when fee PnL accounting is on,
        reserves one exit-side taker fee on entry notional so net loss stays within the cap.
        """
        cap = float(eq_ref) * (self.max_loss_per_trade_equity_pct / 100.0) * max(1.0, float(relief))
        if self.max_loss_usd_total is not None:
            cap = min(cap, float(self.max_loss_usd_total))
        if (
            self._account_fees_in_pnl
            and self._taker_fee_pct_per_side > 0
            and entry_notional_usd > 0
        ):
            cap = max(0.0, cap - self._est_taker_fee_usd(entry_notional_usd))
        return cap

    def _fee_adjusted_upnl_usd(
        self,
        asset: str,
        pos: dict,
        direction: str,
        entry: float,
        mark: float,
        qty: float,
        contract_size: float,
    ) -> Tuple[float, float, float, float]:
        """
        Returns (gross_upnl_usd, entry_fee_est, exit_fee_est, net_upnl_usd).
        Net ≈ gross − entry taker est − exit taker est at current mark.
        """
        gross = _unrealized_pnl_usd(direction, entry, mark, qty, contract_size)
        if not self._account_fees_in_pnl or self._taker_fee_pct_per_side <= 0:
            return gross, 0.0, 0.0, gross
        n_ent = float(
            pos.get('entry_notional_usd')
            or self._notional_usd(qty, entry, asset, contract_size)
        )
        n_mk = self._notional_usd(qty, mark, asset, contract_size)
        ef = float(pos.get('entry_fee_usd_est') or self._est_taker_fee_usd(n_ent))
        xf = self._est_taker_fee_usd(n_mk)
        return gross, ef, xf, gross - ef - xf

    def _pnl_pct_peak_trail(
        self,
        asset: str,
        pos: dict,
        direction: str,
        entry: float,
        mark: float,
        qty: float,
        contract_size: float,
        gross_pnl_pct: float,
    ) -> float:
        """Peak trail uses net % vs entry notional when fee accounting is on."""
        if not self._account_fees_in_pnl or self._taker_fee_pct_per_side <= 0:
            return gross_pnl_pct
        _, _, _, net_u = self._fee_adjusted_upnl_usd(
            asset, pos, direction, entry, mark, qty, contract_size
        )
        den = max(
            float(pos.get('entry_notional_usd') or self._notional_usd(qty, entry, asset, contract_size)),
            1e-9,
        )
        return (net_u / den) * 100.0

    def _conviction_score(self, unified: dict) -> float:
        """0–1 aggregate from LLM + MTF + risk (higher = scale toward max position %)."""
        conf = max(0.0, min(1.0, float(unified.get('confidence', 0.5))))
        tq = int(unified.get('trade_quality', 5))
        rs = int(unified.get('risk_score', 5))
        t_norm = max(0.0, min(1.0, tq / 10.0))
        r_norm = max(0.0, min(1.0, 1.0 - rs / 10.0))
        mtf_n = int(getattr(self, '_mtf_n', 0))
        mtf_a = int(getattr(self, '_mtf_aligned', 0))
        if mtf_n > 0:
            m_norm = max(0.0, min(1.0, mtf_a / mtf_n))
        else:
            m_norm = 0.5
        raw = 0.38 * conf + 0.22 * t_norm + 0.22 * m_norm + 0.18 * r_norm
        return max(0.0, min(1.0, raw))

    def _blend_dynamic_size_pct(self, llm_pct: float, unified: dict) -> Tuple[float, str]:
        """
        Blend LLM position_size_pct with conviction-based %; optional daily PnL target ramp.
        conservative mode: only clamps llm_pct to legacy-style caps (no conviction curve).
        """
        max_pct, _, _ = self._sizing_effective_caps()
        mn = (
            max(1.0, self._sizing_min_pct)
            if self._sizing_mode in ('dynamic', 'llm_direct')
            else 1.0
        )
        llm_pct = float(llm_pct)
        if self._sizing_mode == 'conservative':
            out = max(1.0, min(max_pct, llm_pct))
            return out, "conservative (LLM % only)"
        if self._sizing_mode == 'llm_direct':
            out = max(mn, min(max_pct, llm_pct))
            return out, "llm_direct (LLM position_size_pct; caps + leverage apply)"

        conv = self._conviction_score(unified)
        power = max(0.5, min(3.0, self._sizing_conviction_power))
        dyn = mn + (max_pct - mn) * (conv ** power)
        w = self._sizing_llm_weight
        blended = w * llm_pct + (1.0 - w) * dyn
        out = max(mn, min(max_pct, blended))
        parts = [f"conv={conv:.2f}", f"curve={dyn:.1f}%", f"blend→{out:.1f}%"]

        if self._sizing_daily_target_usd > 0:
            total = self.session_realized_pnl + float(
                getattr(self, '_session_unrealized_pnl', 0.0)
            )
            tgt = self._sizing_daily_target_usd
            if total < tgt * 0.55:
                deficit_ratio = max(0.0, (tgt - total) / tgt)
                mult = min(
                    self._sizing_daily_ramp_max_mult,
                    1.0 + deficit_ratio * (self._sizing_daily_ramp_max_mult - 1.0),
                )
                out = min(max_pct, out * mult)
                parts.append(f"ramp×{mult:.2f} (sess ${total:+,.0f} vs tgt ${tgt:,.0f})")
            elif total > tgt * 1.25:
                out = max(mn, out * 0.88)
                parts.append("ahead-of-tgt −12%")

        out = max(mn, min(max_pct, out))
        return out, " ".join(parts)

    # ------------------------------------------------------------------
    # Orphan Position Closer
    # ------------------------------------------------------------------
    def _close_orphan_positions(self):
        """Find and close exchange positions that the bot doesn't track internally.
        Runs every loop. These orphans cause hidden unrealized losses."""
        try:
            if not self._exchange_client:
                return
            ex_positions = self._exchange_client.get_positions()
            for p in ex_positions:
                sym = p.get('symbol', '')
                qty = abs(float(p.get('qty', 0) or p.get('contracts', 0)))
                if qty <= 0:
                    continue

                # Find which asset this is
                asset = None
                for a in self.assets:
                    if a in sym:
                        asset = a
                        break
                if not asset:
                    continue

                # Skip if bot is already tracking this position
                if asset in self.positions:
                    continue

                # Skip if in failed_close blacklist (retry handled elsewhere)
                if asset in self.failed_close_assets:
                    elapsed = time.time() - self.failed_close_assets[asset]
                    if elapsed < 600:  # Only retry every 10 min
                        continue
                    del self.failed_close_assets[asset]

                # This is an ORPHAN — exchange has it, bot doesn't
                side = p.get('side', 'long')
                entry_p = float(p.get('avg_entry_price', 0) or 0)
                close_side = 'sell' if side == 'long' else 'buy'

                print(f"  [{self._ex_tag}:{asset}] ORPHAN DETECTED: {side} {qty} @ ${entry_p:,.2f} — CLOSING")

                # Close it (reduceOnly so Delta doesn't open a new position)
                symbol = self._get_symbol(asset)
                result = self.price_source.place_order(
                    symbol=symbol,
                    side=close_side,
                    amount=qty,
                    order_type='market',
                    price=None,
                    reduce_only=True,
                )

                if result.get('status') == 'success':
                    print(f"  [{self._ex_tag}:{asset}] ORPHAN CLOSED: {result.get('order_id', '?')}")
                else:
                    err = result.get('message', str(result))
                    # "no_position_for_reduce_only" = position is phantom/liquidated
                    # Stop retrying — it doesn't actually exist on exchange
                    if 'no_position_for_reduce_only' in str(err):
                        print(f"  [{self._ex_tag}:{asset}] ORPHAN PHANTOM: position already gone (liquidated?) — suppressing")
                        self.failed_close_assets[asset] = time.time() + 86400  # Suppress for 24h
                    else:
                        print(f"  [{self._ex_tag}:{asset}] ORPHAN CLOSE FAILED: {err}")
                        self.failed_close_assets[asset] = time.time()

                time.sleep(1)  # Brief pause between closes

        except Exception as e:
            logger.debug(f"Orphan check failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def _exchange_client(self):
        """Return the active exchange client (Bybit, Delta, or None)."""
        ps = self.price_source
        if hasattr(ps, 'bybit') and ps.bybit and ps.bybit.available:
            return ps.bybit
        if hasattr(ps, 'delta') and ps.delta and ps.delta.available:
            return ps.delta
        return None

    @property
    def _exchange_name(self):
        """Return active exchange name."""
        ps = self.price_source
        if hasattr(ps, 'bybit') and ps.bybit and ps.bybit.available:
            return 'bybit'
        if hasattr(ps, 'delta') and ps.delta and ps.delta.available:
            return 'delta'
        return 'unknown'

    def _get_symbol(self, asset: str) -> str:
        """BTC -> exchange-specific symbol format."""
        if self._exchange_name == 'delta':
            return f"{asset}USD"  # Delta uses BTCUSD
        return f"{asset}/USDT:USDT"  # Bybit uses BTC/USDT:USDT

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
        ex_name = self._exchange_name.upper()
        print(f"  EMA(8) Crossover + LLM | {ex_name} Futures")
        print(
            f"  Assets: {self.assets} | Poll: {self.poll_interval}s | "
            f"Signal: {self.signal_timeframe} | Context: {self.context_timeframe} | "
            f"MTF stack: {', '.join(self.analysis_timeframes)}"
        )
        _sig_note = (
            f"live tick as forming {self.signal_timeframe} close"
            if self.signal_timing == 'live'
            else f"last closed {self.signal_timeframe} bar"
        )
        print(f"  Signal timing: {self.signal_timing} ({_sig_note}) | SL exit: {self.sl_exit_mode}")
        print(f"  Entry dedup: {self.entry_dedup_scope}")
        _mxp, _mxu, _mxf = self._sizing_effective_caps()
        if self.equity > 0:
            _cap_note = "∞" if math.isinf(_mxu) else f"${_mxu:,.0f}"
        elif math.isinf(self._sizing_max_trade_usd):
            _cap_note = "∞ max_trade (equity×lev cap after connect)"
        else:
            _cap_note = f"≤${self._sizing_max_trade_usd:,.0f} (after connect)"
        _lev_banner = (
            f"{self._order_leverage:.0f}x fixed"
            if self._leverage_mode == 'fixed'
            else f"dynamic {self._leverage_min:.0f}x–{self._leverage_max:.0f}x (caps use max)"
        )
        print(
            f"  Leverage: {_lev_banner} | "
            f"Sizing: {self._sizing_mode} | max {_mxp:.0f}% margin / "
            f"{_cap_note} notional cap ({_mxf:.0%} equity × {self._order_leverage:.0f}x max)"
            + (
                (
                    f" | daily plan ramp ${self._sizing_daily_target_usd:,.0f} "
                    f"({self._sizing_target_trades} wins × ${self._sizing_target_profit_per_trade_usd:.0f})"
                    if self._sizing_daily_target_usd > 0
                    and not self._sizing_daily_target_is_override
                    else (
                        f" | daily tgt ramp ${self._sizing_daily_target_usd:,.0f} (override)"
                        if self._sizing_daily_target_usd > 0
                        else ""
                    )
                )
            )
        )
        if self._taker_fee_pct_per_side > 0:
            print(
                f"  Fees: {self._taker_fee_pct_per_side:.3f}%/side (model taker) → "
                f"~{self._roundtrip_taker_fee_frac * 100:.3f}% round-trip; "
                f"min ATR% for entry ≥ {self._min_price_edge_pct_after_fees:.3f}% "
                f"(×{self._fee_coverage_safety_buffer:.2f} buffer, "
                f"gate {'on' if self._fee_gate_skip_if_atr_low else 'off'})"
            )
        if self._account_fees_in_pnl:
            print(
                "  Fee accounting: ON — peak trail / early uPnL giveback use est. net vs taker fees; "
                "journal logs gross pnl_usd + extra pnl_usd_net_est"
            )
        if self._use_claude and self._claude_client:
            print(f"  LLM: Claude API ({self._claude_model}) -- Anthropic")
        else:
            print(f"  LLM: {self.ollama_model} @ {self.ollama_base_url}")
        bear_status = "ACTIVE" if self.bear_enabled else "OFF"
        print(f"  Bear Veto: {bear_status} (veto>={self.bear_veto_threshold} reduce>={self.bear_reduce_threshold})")
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

                self._sync_balance_from_broker()
                unrealized_pnl = 0.0
                wallet_balance = 0.0
                try:
                    if self._exchange_client:
                        acct = self._exchange_client.get_account() or {}
                        unrealized_pnl = float(acct.get('unrealized_pnl', 0) or 0)
                        wallet_balance = float(acct.get('wallet_balance', 0) or 0)
                except Exception:
                    pass

                self._session_unrealized_pnl = unrealized_pnl

                base = self.session_start_equity
                if base and base > 0:
                    ret_pct = ((self.equity - base) / base) * 100.0
                else:
                    ret_pct = 0.0
                n_pos = len(self.positions)
                total_vetoed = sum(s['vetoed'] for s in self.bear_veto_stats.values())
                total_reduced = sum(s['reduced'] for s in self.bear_veto_stats.values())
                bear_tag = f" | Bear: {total_vetoed}v/{total_reduced}r" if (total_vetoed + total_reduced) > 0 else ""
                pnl_tag = f" | UPnL: ${unrealized_pnl:+,.2f}" if unrealized_pnl != 0 else ""
                wallet_tag = f" | Wallet: ${wallet_balance:,.2f}" if wallet_balance > 0 and abs(wallet_balance - self.equity) > 1 else ""
                print(f"\n[{self._ex_tag} BAR {self.bar_count}] Equity: ${self.equity:,.2f}{wallet_tag}{pnl_tag} | Cash: ${self.cash:,.2f} | Return: {ret_pct:+.2f}% | Positions: {n_pos}{bear_tag}")

                # ── Portfolio drawdown check — halt new entries if limit breached ──
                if not self._check_drawdown_limits():
                    print(f"  [{self._ex_tag}] HALTED: {self.halt_reason} | Still managing open positions")
                    # Still manage existing positions (SL, exits) but no new entries
                    for asset in self.assets:
                        if asset in self.positions:
                            try:
                                self._process_asset(asset)
                            except Exception as e:
                                print(f"  [{self._ex_tag}:{asset}] ERROR: {e}")
                    elapsed = time.time() - loop_start
                    sleep_time = max(1, self.poll_interval - int(elapsed))
                    print(f"  [{self._ex_tag} SLEEP] {sleep_time}s")
                    time.sleep(sleep_time)
                    continue

                # ── Orphan position check — close exchange positions bot doesn't track ──
                self._close_orphan_positions()

                for asset in self.assets:
                    try:
                        self._process_asset(asset)
                    except Exception as e:
                        print(f"  [{self._ex_tag}:{asset}] ERROR: {e}")
                        logger.exception(f"Error processing {asset}")

                # Sleep until next bar
                elapsed = time.time() - loop_start
                sleep_time = max(1, self.poll_interval - int(elapsed))
                print(f"  [{self._ex_tag} SLEEP] {sleep_time}s")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Graceful exit.")
                break
            except Exception as e:
                print(f"  [{self._ex_tag} ERROR] {e}")
                logger.exception("Main loop error")
                time.sleep(5)

    # ------------------------------------------------------------------
    # Multi-timeframe snapshots (EMA8 + momentum on confirmed bars)
    # ------------------------------------------------------------------
    def _classify_trend_phase(self, ema_vals: list, atr_val: float) -> str:
        """
        Beyond binary R/F: detect slope deceleration (trend stabilising, possible turn).
        Returns RISING | RISING_STABILIZING | FALLING | FALLING_STABILIZING.
        """
        if len(ema_vals) < 3:
            return "RISING" if ema_vals[-2] > ema_vals[-3] else "FALLING"
        cur_e = ema_vals[-2]
        prev_e = ema_vals[-3]
        if abs(prev_e) < 1e-12:
            return "RISING" if cur_e > prev_e else "FALLING"
        pct_1 = (cur_e - prev_e) / prev_e * 100.0
        base_up = cur_e > prev_e
        base_down = cur_e < prev_e

        if len(ema_vals) < 5:
            return "RISING" if base_up else "FALLING"

        e5 = ema_vals[-5]
        if abs(e5) < 1e-12:
            return "RISING" if base_up else "FALLING"
        pct_3 = (cur_e - e5) / e5 * 100.0
        avg3 = pct_3 / 3.0

        atr = max(0.0, float(atr_val or 0.0))
        flat_pct = max(
            0.018,
            min(0.09, (atr / prev_e * 100.0) * self.trend_stabilize_atr_k if atr else 0.03),
        )
        decel = self.trend_stabilize_decel_ratio

        def stabilizing_down() -> bool:
            if not base_down:
                return False
            if abs(pct_1) < flat_pct:
                return True
            if pct_3 >= 0:
                return False
            return abs(pct_1) < decel * abs(avg3)

        def stabilizing_up() -> bool:
            if not base_up:
                return False
            if abs(pct_1) < flat_pct:
                return True
            if pct_3 <= 0:
                return False
            return abs(pct_1) < decel * abs(avg3)

        if base_down:
            return "FALLING_STABILIZING" if stabilizing_down() else "FALLING"
        if base_up:
            return "RISING_STABILIZING" if stabilizing_up() else "RISING"
        return "FALLING"

    @staticmethod
    def _mtf_compact_char(snap: Dict[str, Any]) -> str:
        """R/F = strong phase; r/f = stabilising (slope decelerating)."""
        phase = snap.get("trend_phase") or snap.get("direction", "FALLING")
        if phase == "RISING_STABILIZING":
            return "r"
        if phase == "FALLING_STABILIZING":
            return "f"
        return "R" if snap.get("direction") == "RISING" else "F"

    def _tf_snapshot(
        self,
        closes: list,
        highs: list,
        lows: list,
        atr_hint: Optional[float] = None,
        *,
        style: str = "confirmed",
    ) -> Optional[Dict[str, Any]]:
        """Per-timeframe trend stats. style=confirmed uses last closed bar; style=live uses forming bar (last close)."""
        need = max(self.ema_period + 5, 20)
        if len(closes) < need:
            return None
        ema_vals = ema(closes, self.ema_period)
        if len(ema_vals) < 5:
            return None
        live = style == "live"
        if live:
            cur_e = ema_vals[-1]
            prev_e = ema_vals[-2]
            sig_close = closes[-1]
            ema_term = ema_vals[-1]
            slope_ref = ema_vals[-1]
            c1, c2, c3 = closes[-1], closes[-2], closes[-3]
            atr_idx = -1
        else:
            cur_e = ema_vals[-2]
            prev_e = ema_vals[-3]
            sig_close = closes[-2]
            ema_term = ema_vals[-1]
            slope_ref = ema_vals[-2]
            c1, c2, c3 = closes[-2], closes[-3], closes[-4]
            atr_idx = -2
        direction = "RISING" if cur_e > prev_e else "FALLING"
        if atr_hint is None and len(highs) >= 15 and len(lows) >= 15:
            av = atr(highs, lows, closes, 14)
            if av and len(av) >= abs(atr_idx):
                atr_hint = float(av[atr_idx])
            else:
                atr_hint = 0.0
        trend_phase = self._classify_trend_phase(ema_vals, atr_hint or 0.0)
        base = ema_vals[-5] if abs(ema_vals[-5]) > 1e-12 else cur_e
        slope_pct = ((slope_ref - base) / base * 100) if base else 0.0
        sep_pct = (
            abs(sig_close - ema_term) / ema_term * 100 if ema_term > 0 else 0.0
        )
        if c1 > c2 and c1 > c3:
            mom = "up"
        elif c1 < c2 and c1 < c3:
            mom = "down"
        else:
            mom = "flat"
        return {
            "direction": direction,
            "trend_phase": trend_phase,
            "slope_pct": float(slope_pct),
            "sep_pct": float(sep_pct),
            "mom": mom,
            "ema": float(cur_e),
        }

    def _collect_mtf_analysis(
        self,
        symbol: str,
        signal_closes: list,
        signal_highs: list,
        signal_lows: list,
        signal_ema_direction: str,
        signal_atr: float = 0.0,
        *,
        signal_snapshot_style: str = "confirmed",
    ) -> Tuple[str, int, int, Dict[str, Dict[str, Any]]]:
        """
        On every poll: refresh each analysis timeframe. Reuses signal TF OHLCV when tf matches.
        Returns (llm_block, aligned_count, total_valid, snapshots_by_tf).
        """
        lines: List[str] = []
        snaps: Dict[str, Dict[str, Any]] = {}
        for tf in self.analysis_timeframes:
            try:
                if tf == self.signal_timeframe:
                    cl, hi, lo = signal_closes, signal_highs, signal_lows
                    snap = self._tf_snapshot(
                        cl, hi, lo, atr_hint=signal_atr, style=signal_snapshot_style
                    )
                else:
                    raw = self.price_source.fetch_ohlcv(
                        symbol, timeframe=tf, limit=120
                    )
                    ox = PriceFetcher.extract_ohlcv(raw)
                    cl, hi, lo = ox['closes'], ox['highs'], ox['lows']
                    snap = self._tf_snapshot(cl, hi, lo)
                if snap is None:
                    lines.append(f"  {tf}: (insufficient data)")
                    continue
                snaps[tf] = snap
                s = snap
                phase = s.get("trend_phase", s["direction"])
                stab = " · stabilising" if phase.endswith("_STABILIZING") else ""
                lines.append(
                    f"  {tf}: EMA{self.ema_period} {s['direction']}{stab} "
                    f"slope={s['slope_pct']:+.3f}% sep={s['sep_pct']:.2f}% "
                    f"mom={s['mom']} EMA=${s['ema']:,.2f}"
                )
            except Exception as e:
                lines.append(f"  {tf}: error ({type(e).__name__})")
        block = "\n".join(lines)
        aligned = sum(
            1 for s in snaps.values() if s.get("direction") == signal_ema_direction
        )
        total = len(snaps)
        return block, aligned, total, snaps

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
                print(f"  [{self._ex_tag}:{asset}] Retrying stuck position close...")
            # Don't return — let it analyze and show signals

        symbol = self._get_symbol(asset)

        # ══════════════════════════════════════════════════════════════
        # Signal TF (e.g. 1h): EMA crossover + entries + candle dedup
        # analysis_timeframes: full stack refreshed each poll → logs + unified LLM
        # poll_interval: how often we wake and refetch (not the chart bar size)
        # ══════════════════════════════════════════════════════════════
        self._mtf_block_for_llm = ''
        self._mtf_aligned = 0
        self._mtf_n = 0
        self._mtf_snaps: Dict[str, Dict[str, Any]] = {}

        try:
            raw_sig = self.price_source.fetch_ohlcv(
                symbol, timeframe=self.signal_timeframe, limit=100
            )
        except Exception as e:
            print(f"  [{self._ex_tag}:{asset}] OHLCV fetch failed: {e}")
            return
        ohlcv = PriceFetcher.extract_ohlcv(raw_sig)

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        opens = ohlcv['opens']
        volumes = ohlcv['volumes']

        if len(closes) < 20:
            print(
                f"  [{self._ex_tag}:{asset}] Not enough {self.signal_timeframe} data "
                f"({len(closes)} candles)"
            )
            return

        # ══════════════════════════════════════════════════════════════
        # Tick for execution/SL; signal EMA uses either last CLOSED bar (confirmed)
        # or the same logic with current tick as the forming bar's close (live).
        # ══════════════════════════════════════════════════════════════
        try:
            ticker = self.price_source.exchange.fetch_ticker(symbol) if self.price_source.exchange else {}
            live_price = float(ticker.get('last', 0)) if ticker.get('last') else closes[-1]
        except Exception:
            live_price = closes[-1]

        tick_price = live_price
        closes_sig = list(closes)
        highs_sig = list(highs)
        lows_sig = list(lows)
        if self.signal_timing == 'live':
            closes_sig[-1] = float(tick_price)
            highs_sig[-1] = max(float(highs_sig[-1]), float(tick_price))
            lows_sig[-1] = min(float(lows_sig[-1]), float(tick_price))

        if self.signal_timing == 'confirmed' and len(closes) >= 2:
            price = float(closes[-2])
        else:
            price = float(tick_price)

        ema_vals = ema(closes_sig, self.ema_period)
        atr_vals = atr(highs_sig, lows_sig, closes_sig, 14)
        current_atr = atr_vals[-1] if atr_vals else 0

        if self.signal_timing == 'live':
            current_ema = ema_vals[-1]
            prev_ema = ema_vals[-2] if len(ema_vals) >= 2 else current_ema
            signal_atr_for_phase = (
                float(atr_vals[-1]) if atr_vals else float(current_atr)
            )
        else:
            current_ema = ema_vals[-2]
            prev_ema = ema_vals[-3] if len(ema_vals) >= 3 else current_ema
            signal_atr_for_phase = (
                float(atr_vals[-2]) if len(atr_vals) >= 2 else float(current_atr)
            )

        ema_direction = "RISING" if current_ema > prev_ema else "FALLING"
        signal_trend_phase = self._classify_trend_phase(ema_vals, signal_atr_for_phase)

        _snap_style = "live" if self.signal_timing == "live" else "confirmed"
        mtf_block, mtf_aligned, mtf_n, mtf_snaps = self._collect_mtf_analysis(
            symbol,
            closes_sig,
            highs_sig,
            lows_sig,
            ema_direction,
            signal_atr=signal_atr_for_phase,
            signal_snapshot_style=_snap_style,
        )
        self._mtf_block_for_llm = mtf_block
        self._mtf_aligned = mtf_aligned
        self._mtf_n = mtf_n
        self._mtf_snaps = mtf_snaps

        # Context TF (e.g. 1m): prefer MTF snapshot if that TF is in the stack
        ctx_suffix = ""
        self._ctx_gate = {'blocks_long': False, 'blocks_short': False}
        _ctx = self.context_timeframe
        ctx_snap = mtf_snaps.get(_ctx) if _ctx and mtf_snaps else None
        if ctx_snap:
            mom = "↓" if ctx_snap['mom'] == 'down' else (
                "↑" if ctx_snap['mom'] == 'up' else "~"
            )
            micro_ema = self._mtf_compact_char(ctx_snap)
            ctx_suffix = f" | {_ctx}: mom={mom} EMA={micro_ema}"
            self._ctx_gate['blocks_long'] = ctx_snap['mom'] == 'down'
            self._ctx_gate['blocks_short'] = ctx_snap['mom'] == 'up'
        elif _ctx and _ctx != self.signal_timeframe:
            try:
                raw_ctx = self.price_source.fetch_ohlcv(symbol, timeframe=_ctx, limit=150)
                cx = PriceFetcher.extract_ohlcv(raw_ctx)
                cc = cx['closes']
                if len(cc) >= 5:
                    c1, c2, c3 = cc[-2], cc[-3], cc[-4]
                    micro_down = c1 < c2 and c1 < c3
                    micro_up = c1 > c2 and c1 > c3
                    mom = "↓" if micro_down else ("↑" if micro_up else "~")
                    ema_c = ema(cc, self.ema_period)
                    e2 = ema_c[-2] if len(ema_c) >= 2 else cc[-2]
                    e3 = ema_c[-3] if len(ema_c) >= 3 else e2
                    micro_ema = "R" if e2 > e3 else "F"
                    ctx_suffix = f" | {_ctx}: mom={mom} EMA={micro_ema}"
                    self._ctx_gate['blocks_long'] = micro_down
                    self._ctx_gate['blocks_short'] = micro_up
            except Exception:
                ctx_suffix = f" | {_ctx}: n/a"

        # Fetch L2 order book for support/resistance walls
        try:
            order_book = self.price_source.fetch_order_book(symbol, limit=25)
            ob_levels = self._extract_ob_levels(order_book, tick_price)
        except Exception:
            ob_levels = {'bid_wall': 0, 'ask_wall': 0, 'bid_walls': [], 'ask_walls': [], 'imbalance': 0, 'bid_depth_usd': 0, 'ask_depth_usd': 0}

        # ── Signal from signal_timeframe EMA crossover (confirmed bar vs live tick as forming close) ──
        signal = "NEUTRAL"

        ema_crossed = False
        _cross_start = 1 if self.signal_timing == 'live' else 2
        for i in range(_cross_start, min(5, len(highs_sig))):
            h = highs_sig[-i]
            l = lows_sig[-i]
            e = ema_vals[-i] if i <= len(ema_vals) else 0
            if l <= e <= h:
                ema_crossed = True
                break

        price_falling = False
        price_rising = False
        if len(closes_sig) >= 5:
            if self.signal_timing == 'live':
                c1, c2, c3 = float(tick_price), float(closes_sig[-2]), float(closes_sig[-3])
            else:
                c1, c2, c3 = closes_sig[-2], closes_sig[-3], closes_sig[-4]
            if c1 < c2 and c1 < c3:
                price_falling = True
            elif c1 > c2 and c1 > c3:
                price_rising = True

        # OVEREXTENSION CHECK — if price is >10% from EMA, it's parabolic
        # Don't enter LONG at the top of a parabolic move or SHORT at the bottom
        ema_separation = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
        if ema_separation > 10.0:
            signal = "NEUTRAL"
            print(f"  [{self._ex_tag}:{asset}] OVEREXTENDED: {ema_separation:.1f}% from EMA — too far, waiting for pullback")
        elif ema_direction == "RISING" and price > current_ema and ema_crossed:
            if price_falling:
                signal = "NEUTRAL"  # EMA says buy but price is reversing down
            elif (
                self.entry_block_stabilizing_crossover
                and signal_trend_phase == "RISING_STABILIZING"
            ):
                signal = "NEUTRAL"
                print(
                    f"  [{self._ex_tag}:{asset}] STABILISING: skip crossover BUY "
                    f"(EMA rise decelerating — possible reversal)"
                )
            else:
                signal = "BUY"
        elif ema_direction == "FALLING" and price < current_ema and ema_crossed:
            if price_rising:
                signal = "NEUTRAL"  # EMA says sell but price is reversing up
            elif (
                self.entry_block_stabilizing_crossover
                and signal_trend_phase == "FALLING_STABILIZING"
            ):
                signal = "NEUTRAL"
                print(
                    f"  [{self._ex_tag}:{asset}] STABILISING: skip crossover SELL "
                    f"(EMA fall decelerating — possible bounce)"
                )
            else:
                signal = "SELL"
        # Strong trend: price >1% from EMA — skip when slope stabilising (reversal risk)
        elif ema_direction == "FALLING" and price < current_ema * 0.99:
            if not price_rising and signal_trend_phase != "FALLING_STABILIZING":
                signal = "SELL"
        elif ema_direction == "RISING" and price > current_ema * 1.01:
            if not price_falling and signal_trend_phase != "RISING_STABILIZING":
                signal = "BUY"

        # Intrabar P1: previous CLOSED bar touched EMA; forming bar commits + live EMA turn (spec), before next close.
        entry_size_mult = 1.0
        dedup_candle_ts_override: Optional[float] = None
        early_ema_entry = False
        if (
            self.signal_timing != 'live'
            and self.early_ema_switch_entry
            and signal == "NEUTRAL"
            and ema_separation <= 10.0
        ):
            early_sig = _early_ema_p1_switch_signal(
                closes,
                highs,
                lows,
                ema(closes, self.ema_period),
                tick_price,
                self.ema_period,
                overextension_pct=10.0,
                signal_trend_phase=signal_trend_phase,
                price_falling=price_falling,
                price_rising=price_rising,
                entry_block_stabilizing=self.entry_block_stabilizing_crossover,
            )
            if early_sig:
                _ts_early = ohlcv.get('timestamps') or []
                if len(_ts_early) >= 1:
                    signal = early_sig
                    entry_size_mult = self.early_switch_size_mult
                    dedup_candle_ts_override = float(_ts_early[-1])
                    early_ema_entry = True
                    print(
                        f"  [{self._ex_tag}:{asset}] EARLY_EMA_SWITCH: {early_sig} "
                        f"(forming {self.signal_timeframe} vs prior bar EMA touch) | size x{entry_size_mult:.2f}"
                    )

        ob_imb = ob_levels.get('imbalance', 0)
        ob_bid = ob_levels.get('bid_wall', 0)
        ob_ask = ob_levels.get('ask_wall', 0)
        ob_info = f"OB[imb={ob_imb:+.2f}"
        if ob_bid > 0:
            ob_info += f" sup=${ob_bid:,.2f}"
        if ob_ask > 0:
            ob_info += f" res=${ob_ask:,.2f}"
        ob_info += "]"
        _phase_disp = (
            "RISING (stabilising)"
            if signal_trend_phase == "RISING_STABILIZING"
            else (
                "FALLING (stabilising)"
                if signal_trend_phase == "FALLING_STABILIZING"
                else ema_direction
            )
        )
        print(
            f"  [{self._ex_tag}:{asset}] ${tick_price:,.2f} (sig=${price:,.2f}) | "
            f"EMA({self.signal_timeframe}): ${current_ema:.2f} {_phase_disp} | "
            f"Signal: {signal} | ATR: ${current_atr:.2f} | {ob_info}{ctx_suffix}"
        )
        if mtf_n > 0:
            compact = " ".join(
                f"{tf}:{self._mtf_compact_char(mtf_snaps[tf])}"
                for tf in self.analysis_timeframes
                if tf in mtf_snaps
            )
            print(
                f"  [{self._ex_tag}:{asset}] MTF EMA{self.ema_period} "
                f"align {mtf_aligned}/{mtf_n} vs {ema_direction} | {compact}"
            )

        _ts_ohlcv = ohlcv.get('timestamps') or []
        _rise_fall = "RISE" if ema_direction == "RISING" else "FALL"
        if self.signal_timing == 'live':
            _fo = _ts_ohlcv[-1] if len(_ts_ohlcv) >= 1 else None
            _bar_open = _signal_bar_open_utc(_fo) if _fo is not None else "?"
            print(
                f"  [{self._ex_tag}:{asset}] SIGNAL (live): forming {self.signal_timeframe} "
                f"bar OPEN={_bar_open} | tick as close | EMA({self.ema_period}) vs prior → {_rise_fall} ({ema_direction})"
            )
        else:
            _sig_open = _ts_ohlcv[-2] if len(_ts_ohlcv) >= 2 else None
            _bar_open = _signal_bar_open_utc(_sig_open) if _sig_open is not None else "?"
            print(
                f"  [{self._ex_tag}:{asset}] SIGNAL CANDLE: last CLOSED {self.signal_timeframe} "
                f"bar OPEN={_bar_open} | EMA({self.ema_period}) vs prior bar → {_rise_fall} ({ema_direction})"
            )
        if asset in self.positions:
            _ps = self.positions[asset].get("direction", "?")
            print(f"  [{self._ex_tag}:{asset}] FINAL DECISION (trade): HOLD {_ps}")
        elif signal == "BUY":
            print(f"  [{self._ex_tag}:{asset}] FINAL DECISION (signal): LONG / CALL (if pattern+LLM pass)")
        elif signal == "SELL":
            print(f"  [{self._ex_tag}:{asset}] FINAL DECISION (signal): SHORT / PUT (if pattern+LLM pass)")
        else:
            print(f"  [{self._ex_tag}:{asset}] FINAL DECISION (signal): FLAT")

        # ── Stale position check: if internal state says position but exchange is clean ──
        if asset in self.positions and asset not in self.failed_close_assets:
            try:
                if self._exchange_client:
                    ex_pos = self._exchange_client.get_positions()
                    has_exchange_pos = any(asset in pp.get('symbol','') and float(pp.get('qty',0)) > 0 for pp in ex_pos)
                    if not has_exchange_pos:
                        print(f"  [{self._ex_tag}:{asset}] STALE position cleared (not on exchange)")
                        del self.positions[asset]
            except Exception:
                pass

        ohlcv_manage = ohlcv
        if self.signal_timing == 'live':
            _cm = list(closes)
            _hm = list(highs)
            _lm = list(lows)
            _cm[-1] = float(tick_price)
            _hm[-1] = max(float(_hm[-1]), float(tick_price))
            _lm[-1] = min(float(_lm[-1]), float(tick_price))
            ohlcv_manage = {**ohlcv, 'closes': _cm, 'highs': _hm, 'lows': _lm}

        if asset in self.positions:
            self._manage_position(
                asset, tick_price, ohlcv_manage, ema_vals, atr_vals, ema_direction, signal, ob_levels
            )
        else:
            self._evaluate_entry(
                asset,
                tick_price,
                ohlcv,
                ema_vals,
                atr_vals,
                ema_direction,
                signal,
                closes_sig,
                highs_sig,
                lows_sig,
                opens,
                volumes,
                current_ema,
                current_atr,
                ob_levels,
                signal_trend_phase=signal_trend_phase,
                dedup_candle_ts_override=dedup_candle_ts_override,
                entry_size_mult=entry_size_mult,
                early_ema_entry=early_ema_entry,
            )

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------
    def _evaluate_entry(
        self,
        asset: str,
        price: float,
        ohlcv: dict,
        ema_vals: list,
        atr_vals: list,
        ema_direction: str,
        signal: str,
        closes: list,
        highs: list,
        lows: list,
        opens: list,
        volumes: list,
        current_ema: float,
        current_atr: float,
        ob_levels: dict = None,
        signal_trend_phase: str = "",
        dedup_candle_ts_override: Optional[float] = None,
        entry_size_mult: float = 1.0,
        early_ema_entry: bool = False,
    ):

        ob_levels = ob_levels or {}
        entry_size_mult = max(1.0, float(entry_size_mult or 1.0))

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
            # Use confirmed candles only (skip [-1] incomplete candle)
            consec = 0
            for i in range(len(ema_vals)-2, max(0, len(ema_vals)-12), -1):
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

            # 3. PRICE vs EMA SEPARATION (0-2 points) — confirmed: last closed close; live: forming close (= tick)
            if self.signal_timing == 'live' and len(closes) >= 1:
                sig_close = float(closes[-1])
            else:
                sig_close = closes[-2] if len(closes) >= 2 else price
            separation = abs(sig_close - ema_vals[-1]) / ema_vals[-1] * 100 if ema_vals[-1] > 0 else 0
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

        if early_ema_entry:
            entry_score += 4
            score_reasons.append("early_ema_switch+4")

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
                print(f"  [{self._ex_tag}:{asset}] RANGING: range={range_pct:.1f}% ATR={atr_pct:.2f}% ratio={atr_range_ratio:.0%} — SKIP")
                return
            elif range_pct > 0 and atr_range_ratio > 0.4:
                # Log but DON'T skip — EMA momentum overrides
                print(f"  [{self._ex_tag}:{asset}] RANGE NOTE: range={range_pct:.1f}% ratio={atr_range_ratio:.0%} but EMA momentum={ema_has_momentum} — ALLOWING")

        # ORDER BOOK IMBALANCE — log only, don't block
        # Testnet OB is too thin/unreliable to filter entries
        # On live exchange with real liquidity, re-enable filtering
        ob_imbalance = ob_levels.get('imbalance', 0)

        # MINIMUM SCORE — spec §4.1: need >= 3 / 10; >= 1 confirmed EMA trend bar
        min_score = self.entry_pattern_min_score
        min_trend_need = self.entry_min_trend_bars

        # Count consecutive trend bars — CONFIRMED candles only (skip [-1] incomplete)
        min_trend_bars = 0
        for i in range(len(ema_vals)-2, max(0, len(ema_vals)-12), -1):
            if i > 0:
                if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]:
                    min_trend_bars += 1
                elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]:
                    min_trend_bars += 1
                else:
                    break

        if entry_score < min_score:
            print(f"  [{self._ex_tag}:{asset}] WEAK: score={entry_score}/10 ({', '.join(score_reasons) or 'no momentum'}) -- need {min_score}+")
            return
        elif min_trend_bars < min_trend_need and not early_ema_entry:
            print(f"  [{self._ex_tag}:{asset}] TOO EARLY: only {min_trend_bars} trend bars -- need {min_trend_need}+")
            return

        # ── HARD GATE: EMA slope must agree with signal direction ──
        # Prevents LLM from confirming LONG when EMA is falling (or SHORT when rising)
        # Skipped for intrabar EMA-switch entries (slope uses forming bar; live turn already checked)
        if not early_ema_entry and len(ema_vals) >= 4:
            slope_pct = (ema_vals[-1] - ema_vals[-4]) / ema_vals[-4] * 100 if ema_vals[-4] > 0 else 0
            if signal == "BUY" and slope_pct < -0.01:
                print(f"  [{self._ex_tag}:{asset}] SLOPE REJECT: CALL but EMA slope={slope_pct:+.3f}% (negative) — skip")
                return
            elif signal == "SELL" and slope_pct > 0.01:
                print(f"  [{self._ex_tag}:{asset}] SLOPE REJECT: PUT but EMA slope={slope_pct:+.3f}% (positive) — skip")
                return

        quality = "EXCELLENT" if entry_score >= 9 else "STRONG" if entry_score >= 7 else "OK"
        print(f"  [{self._ex_tag}:{asset}] {quality} PATTERN: score={entry_score}/10 ({', '.join(score_reasons)}) trend={min_trend_bars}bars")

        # CRITICAL: Check EXCHANGE positions (not just internal dict)
        # CHECK OPEN ORDERS — if we already have a pending limit order, don't place another
        try:
            if self._exchange_client:
                symbol = self._get_symbol(asset)
                open_orders = self._exchange_client.exchange.fetch_open_orders(symbol)
                if open_orders:
                    print(f"  [{self._ex_tag}:{asset}] SKIP: {len(open_orders)} pending order(s) already on exchange")
                    return
        except Exception:
            pass

        # Prevents stacking when internal state is out of sync
        try:
            if self._exchange_client:
                exchange_positions = self._exchange_client.get_positions()
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
                            print(f"  [{self._ex_tag}:{asset}] WRONG SIDE: holding {synced_direction} but signal={signal} — closing to flip to {flip_dir}")
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
                            ex_pos2 = self._exchange_client.get_positions()
                            still_open = any(asset in pp.get('symbol','') and float(pp.get('qty',0)) > 0 for pp in ex_pos2)
                            if still_open:
                                print(f"  [{self._ex_tag}:{asset}] CLOSE FAILED — stuck, blacklisting")
                                self.failed_close_assets[asset] = time.time()
                                if asset in self.positions:
                                    del self.positions[asset]
                                return
                            else:
                                print(f"  [{self._ex_tag}:{asset}] CLOSED {synced_direction} — now free to enter {flip_dir}")
                                if asset in self.positions:
                                    del self.positions[asset]
                                # Don't return — fall through to entry evaluation
                                break
                        else:
                            # Same direction or no signal — sync it and manage
                            if asset not in self.positions:
                                # SAFETY: Check if synced position is too large for our account
                                max_pct, _, _ = self._sizing_effective_caps()
                                if self._exchange_name == 'delta':
                                    cs = {'BTC': 0.001, 'ETH': 0.01}.get(asset, 0.001)
                                    pos_notional = contracts * cs * price
                                else:
                                    pos_notional = contracts * price
                                max_notional = self.equity * (max_pct / 100)
                                if pos_notional > max_notional * 3:
                                    # Position is >3x our max — likely a leftover from manual trading
                                    print(f"  [{self._ex_tag}:{asset}] WARNING: exchange pos ${pos_notional:,.0f} >> max ${max_notional:,.0f} — NOT syncing (close manually)")
                                    return

                                sl_dist = current_atr * self.atr_stop_mult if current_atr > 0 else price * 0.01
                                if pos_side == 'long':
                                    sync_sl = entry_p - sl_dist
                                else:
                                    sync_sl = entry_p + sl_dist
                                _sync_cs = 0.0
                                if self._exchange_name == 'delta':
                                    _sync_cs = {'BTC': 0.001, 'ETH': 0.01}.get(asset, 0.001)
                                _sync_n = self._notional_usd(contracts, entry_p, asset, _sync_cs)
                                _eqs = max(float(self.equity), 1.0)
                                _max_l = self._max_loss_price_move_usd(_eqs, 1.0, _sync_n)
                                _ls = _loss_usd_at_price(
                                    synced_direction, entry_p, sync_sl, contracts, _sync_cs
                                )
                                if _ls > _max_l > 0 and contracts > 0:
                                    _qm = _linear_qty_multiplier(contracts, _sync_cs)
                                    if _qm > 0:
                                        _md = _max_l / _qm
                                        if synced_direction == 'LONG':
                                            sync_sl = entry_p - _md
                                        else:
                                            sync_sl = entry_p + _md
                                _sync_fe = self._est_taker_fee_usd(_sync_n)
                                self.positions[asset] = {
                                    'direction': synced_direction,
                                    'side': 'buy' if pos_side == 'long' else 'sell',
                                    'entry_price': entry_p,
                                    'qty': contracts,
                                    'contract_size': _sync_cs,
                                    'sl': sync_sl, 'sl_levels': ['L1'],
                                    'sl_order_id': None,
                                    'peak_price': price, 'entry_time': time.time(),
                                    'confidence': 0, 'reasoning': 'synced from exchange',
                                    'breakeven_moved': False,
                                    'risk_equity_ref': _eqs,
                                    'max_loss_usd_cap': _max_l,
                                    'bounce_loss_relief': False,
                                    'peak_upnl_usd': 0.0,
                                    'peak_trail_armed': False,
                                    'peak_trail_pnl_pct': 0.0,
                                    'entry_notional_usd': _sync_n,
                                    'entry_fee_usd_est': _sync_fe,
                                }
                                print(f"  [{self._ex_tag}:{asset}] SYNCED {pos_side} position ({contracts}) ${pos_notional:,.0f} | SL=${sync_sl:,.2f}")
                            # Don't return — let _manage_position handle it
                            return
        except Exception as e:
            logger.debug(f"Exchange position check failed: {e}")

        # CANDLE DEDUP: default = last CLOSED signal bar (-2). Intrabar early entries use forming bar open (-1).
        # Timestamp is recorded only after pre-LLM gates pass (see below) — not here — so context/ATR
        # skips do not burn the candle and incorrectly show DEDUP on every later poll.
        timestamps = ohlcv.get('timestamps', [])
        dedup_candle_ts: Optional[float] = None
        if dedup_candle_ts_override is not None:
            dedup_candle_ts = float(dedup_candle_ts_override)
        elif len(timestamps) >= 2 and self.signal_timing != 'live':
            dedup_candle_ts = float(timestamps[-2])
        if dedup_candle_ts is not None:
            if asset in self.last_signal_candle and self.last_signal_candle[asset] == dedup_candle_ts:
                print(
                    f"  [{self._ex_tag}:{asset}] DEDUP: waiting for next "
                    f"{self.signal_timeframe} candle"
                )
                return

        if self.require_context_alignment:
            g = getattr(self, '_ctx_gate', None) or {}
            if signal == "BUY" and g.get('blocks_long'):
                print(f"  [{self._ex_tag}:{asset}] SKIP: {self.context_timeframe} momentum vs LONG")
                return
            if signal == "SELL" and g.get('blocks_short'):
                print(f"  [{self._ex_tag}:{asset}] SKIP: {self.context_timeframe} momentum vs SHORT")
                return

        if self.mtf_confluence_min > 0:
            ma = getattr(self, '_mtf_aligned', 0)
            mn = getattr(self, '_mtf_n', 0)
            if mn > 0 and ma < self.mtf_confluence_min:
                print(
                    f"  [{self._ex_tag}:{asset}] MTF CONFLUENCE: {ma}/{mn} "
                    f"< min {self.mtf_confluence_min} for {ema_direction} — skip"
                )
                return

        # spec §4.4: trade cooldown (default 30s)
        now = time.time()
        if asset in self.last_close_time:
            elapsed_pc = now - self.last_close_time[asset]
            if elapsed_pc < self.post_close_cooldown:
                return
        if asset in self.last_trade_time:
            elapsed = now - self.last_trade_time[asset]
            if elapsed < self.trade_cooldown:
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

        # ── Run Agent Orchestrator + Debate Engine (math agents) ──
        orch_result = self._run_orchestrator(
            asset, price, signal, closes, highs, lows, opens, volumes,
            ema_vals=ema_vals, atr_vals=atr_vals, ema_direction=ema_direction
        )

        # ══════════════════════════════════════════════════════════
        # UNIFIED LLM: ALL 7 roles in ONE call
        # Agent Synthesis + Bull + Bear + 3 Personas + Facilitator
        # ══════════════════════════════════════════════════════════
        if self.orchestrator_entry_veto and orch_result and orch_result.get('veto'):
            print(f"  [{self._ex_tag}:{asset}] AGENTS VETO: consensus={orch_result['consensus']} — skipping trade")
            if dedup_candle_ts is not None:
                self.last_signal_candle[asset] = dedup_candle_ts
            return

        # Build P&L history string for context
        edge = self.edge_stats.get(asset, {})
        pnl_history = f"{edge.get('wins',0)}W/{edge.get('losses',0)}L rate={edge.get('win_rate',0.5):.0%}"

        _stp = signal_trend_phase or ema_direction
        _msnaps = getattr(self, '_mtf_snaps', None) or {}
        _mtf_stab = sum(
            1
            for s in _msnaps.values()
            if str(s.get('trend_phase', '') or '').endswith('_STABILIZING')
        )
        unified = self._query_unified_llm(
            asset=asset, signal=signal, price=price,
            current_ema=current_ema, current_atr=current_atr,
            ema_direction=ema_direction, ema_slope_pct=ema_slope_pct,
            consecutive_trend=consecutive_trend, candle_lines=candle_lines,
            support=support, resistance=resistance,
            closes=closes, highs=highs, lows=lows,
            volumes=volumes, orch_result=orch_result,
            pnl_history=pnl_history,
            mtf_block=getattr(self, '_mtf_block_for_llm', '') or '',
            mtf_aligned=int(getattr(self, '_mtf_aligned', 0)),
            mtf_n=int(getattr(self, '_mtf_n', 0)),
            entry_pattern_score=entry_score,
            signal_trend_bars=min_trend_bars,
            signal_trend_phase=_stp,
            mtf_stabilizing_count=_mtf_stab,
        )

        if self.entry_dedup_scope == 'after_llm' and dedup_candle_ts is not None:
            self.last_signal_candle[asset] = dedup_candle_ts

        # Extract unified decision
        if not unified.get('proceed', False):
            risk_score = unified.get('risk_score', 5)
            tq = unified.get('trade_quality', 3)
            pred_l = unified.get('predicted_l_level', '?')
            fac = str(unified.get('facilitator_verdict', ''))[:80]
            print(f"  [{self._ex_tag}:{asset}] REJECTED: quality={tq}/10 risk={risk_score}/10 pred={pred_l} | {fac}")

            # Track bear veto stats
            if asset not in self.bear_veto_stats:
                self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
            if risk_score >= self.bear_veto_threshold:
                self.bear_veto_stats[asset]['vetoed'] += 1
            else:
                self.bear_veto_stats[asset]['reduced'] += 1
            return

        # Trade approved — extract parameters
        confidence = unified.get('confidence', 0.5)
        size_pct = unified.get('position_size_pct', 3) * entry_size_mult
        reasoning = str(unified.get('facilitator_verdict', ''))[:120]
        risk_score = unified.get('risk_score', 5)

        # Optional: blend agent math into confidence (off by default — spec §5.3 uses LLM gate only)
        if self.orchestrator_blend_confidence and orch_result and orch_result.get('confidence', 0) > 0:
            agent_conf = orch_result['confidence']
            agent_scale = orch_result.get('position_scale', 1.0)
            blended_conf = confidence * 0.4 + agent_conf * 0.6

            agent_dir = orch_result.get('direction', 0)
            signal_dir = 1 if signal == "BUY" else -1
            if agent_dir != 0 and agent_dir != signal_dir:
                blended_conf *= 0.5
                print(f"  [{self._ex_tag}:{asset}] AGENTS DISAGREE: agents={orch_result['consensus_dir']} vs EMA={signal} — conf halved")

            size_pct = size_pct * agent_scale
            print(f"  [{self._ex_tag}:{asset}] BLENDED: LLM={confidence:.2f} + Agents={agent_conf:.2f} -> {blended_conf:.2f} | scale={agent_scale:.2f}")
            confidence = blended_conf

        # Direction from EMA signal (always)
        action = "LONG" if signal == "BUY" else "SHORT"
        direction_label = "CALL" if signal == "BUY" else "PUT"

        # spec §6.2: elevated risk → halve size (skipped when ai.confidence_only_gates)
        if (
            not self._confidence_only_gates
            and self.bear_reduce_threshold <= risk_score < self.bear_veto_threshold
        ):
            size_pct *= 0.5
            print(f"  [{self._ex_tag}:{asset}] BEAR REDUCE: risk={risk_score} >= {self.bear_reduce_threshold} — size halved")
            if asset not in self.bear_veto_stats:
                self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
            self.bear_veto_stats[asset]['reduced'] += 1

        # Quality gate: confidence (llm_min_confidence in config; LLM failure uses llm_fallback_confidence_gate)
        conf_floor = (
            self.llm_fallback_confidence_gate
            if unified.get('llm_unavailable')
            else self.min_confidence
        )
        if confidence < conf_floor:
            print(
                f"  [{self._ex_tag}:{asset}] SKIP: confidence {confidence:.2f} < {conf_floor}"
                f"{' (LLM fallback gate)' if unified.get('llm_unavailable') else ''}"
            )
            return

        # Track bear stats
        if asset not in self.bear_veto_stats:
            self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
        self.bear_veto_stats[asset]['passed'] += 1

        # ── Fee-aware gate: skip if typical volatility (ATR %) can't cover round-trip taker fees × buffer ──
        if (
            self._fee_gate_skip_if_atr_low
            and self._min_price_edge_pct_after_fees > 0
            and price > 0
            and current_atr > 0
        ):
            atr_pct = (current_atr / price) * 100.0
            if atr_pct + 1e-12 < self._min_price_edge_pct_after_fees:
                print(
                    f"  [{self._ex_tag}:{asset}] SKIP (fees): ATR {atr_pct:.3f}% < "
                    f"{self._min_price_edge_pct_after_fees:.3f}% min edge "
                    f"(2×{self._taker_fee_pct_per_side:.3f}% taker × {self._fee_coverage_safety_buffer:.2f} buffer) "
                    f"— unlikely net profit after ~{self._roundtrip_taker_fee_frac*100:.3f}% round-trip fees"
                )
                return

        # ── Edge Positioning: adjust size by historical win rate ──
        if (
            self.edge_enabled
            and not self._confidence_only_gates
            and asset in self.edge_stats
        ):
            edge = self.edge_stats[asset]
            if edge['total'] >= 5:
                mult = edge['edge_multiplier']
                old_size = size_pct
                size_pct = size_pct * mult
                if abs(mult - 1.0) > 0.05:
                    print(f"  [{self._ex_tag}:{asset}] EDGE: {mult:.2f}x ({edge['wins']}W/{edge['losses']}L) size {old_size:.0f}% -> {size_pct:.0f}%")

        # Calculate position size — equity from broker; caps from sizing config / mode
        max_size_pct, max_trade, _eq_frac = self._sizing_effective_caps()
        pre_dyn = size_pct
        size_pct, sizing_note = self._blend_dynamic_size_pct(size_pct, unified)
        size_pct = max(1.0, min(max_size_pct, size_pct))
        if self.equity <= 0:
            print(f"  [{self._ex_tag}:{asset}] SKIP: no equity available")
            return

        trade_lev = self._leverage_for_trade(confidence, unified)
        self._set_exchange_leverage_symbol(asset, trade_lev)

        margin_alloc = self.equity * (size_pct / 100.0)
        notional = margin_alloc * float(trade_lev)
        notional = min(notional, max_trade)

        if (
            abs(size_pct - pre_dyn) > 0.05
            or self._sizing_mode in ('dynamic', 'llm_direct')
        ):
            _cap_s = "∞" if math.isinf(max_trade) else f"${max_trade:,.0f}"
            _lm = (
                f"{trade_lev:.0f}x"
                if self._leverage_mode == 'fixed'
                else f"{trade_lev:.0f}x (dynamic {self._leverage_min:.0f}-{self._leverage_max:.0f})"
            )
            print(
                f"  [{self._ex_tag}:{asset}] SIZING: {size_pct:.1f}% margin of ${self.equity:,.0f} "
                f"× {_lm} → ${notional:,.0f} notional (cap {_cap_s}) | {sizing_note}"
            )
        else:
            _cap_s = "∞" if math.isinf(max_trade) else f"${max_trade:,.0f}"
            _lm = (
                f"{trade_lev:.0f}x"
                if self._leverage_mode == 'fixed'
                else f"{trade_lev:.0f}x (dyn)"
            )
            print(
                f"  [{self._ex_tag}:{asset}] SIZING: {size_pct:.0f}% margin × {_lm} "
                f"→ ${notional:,.0f} notional (cap {_cap_s})"
            )

        qty = notional / price if price > 0 else 0
        qty = round(qty, 6)

        if qty <= 0:
            return

        # Minimum qty check — different per exchange
        if self._exchange_name == 'delta':
            # Delta: contract-based sizing
            # BTC: 1 contract = 0.001 BTC. So contract_value = 0.001 * price
            # ETH: 1 contract = 0.01 ETH. So contract_value = 0.01 * price
            contract_size = {'BTC': 0.001, 'ETH': 0.01}
            cs = contract_size.get(asset, 0.001)
            contract_value = cs * price  # value of 1 contract in USD
            qty = max(1, int(notional / contract_value))  # number of contracts
            actual_notional = qty * contract_value
            print(f"  [{self._ex_tag}:{asset}] DELTA: {qty} contracts x ${contract_value:,.2f} = ${actual_notional:,.2f} notional")
            _max_n = self.equity * self._sizing_max_equity_fraction * float(trade_lev)
            if actual_notional > _max_n + 1e-9 and qty > 1:
                qty = max(1, int(_max_n / contract_value))
                actual_notional = qty * contract_value
                print(
                    f"  [{self._ex_tag}:{asset}] CAPPED to {qty} contracts (${actual_notional:,.2f}) "
                    f"(max notional vs equity×lev cap)"
                )
            min_qty = {'BTC': 1, 'ETH': 1}
        else:
            # Bybit: coin-based sizing
            min_qty = {'BTC': 0.001, 'ETH': 0.01}

        asset_min = min_qty.get(asset, 1 if self._exchange_name == 'delta' else 0.001)
        if qty < asset_min:
            if self._sizing_mode in ('dynamic', 'llm_direct'):
                max_pct = max(
                    0.12, min(0.30, self._sizing_max_equity_fraction * 0.65)
                )
            else:
                max_pct = 0.50 if self.equity < 500 else 0.10
            if self._exchange_name == 'delta':
                cs = contract_size.get(asset, 0.001)
                min_notional = asset_min * cs * price
            else:
                min_notional = asset_min * price
            if min_notional <= self.equity * max_pct:
                qty = asset_min
                notional = min_notional
                print(f"  [{self._ex_tag}:{asset}] Adjusted qty to minimum {asset_min} (${notional:,.0f})")
            else:
                print(f"  [{self._ex_tag}:{asset}] SKIP: min order ${min_notional:,.0f} > {max_pct:.0%} of equity ${self.equity:,.0f}")
                return

        # Determine order side and price
        side = 'buy' if action == 'LONG' else 'sell'

        # Execution type from config
        entry_type = self.config.get('execution', {}).get('entry_type', 'market')

        if entry_type == 'limit':
            order_price = current_ema
            print(f"  [{self._ex_tag}:{asset}] {direction_label}: {side.upper()} {qty:.6f} LIMIT@${order_price:,.2f} (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")
        else:
            order_price = None  # CRITICAL: market orders must NOT have a price
            print(f"  [{self._ex_tag}:{asset}] {direction_label}: {side.upper()} {qty:.6f} MARKET (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")

        # PRE-CHECK: Liquidity sanity check (widened for testnet)
        # Only block if order book is completely empty — testnet has thin but usable books
        ob_check = self.price_source.fetch_order_book(self._get_symbol(asset), limit=10)
        max_dev = price * 0.10  # 10% — wide enough for testnet spreads

        reasonable_asks = [a for a in ob_check.get('asks', []) if float(a[0]) <= price + max_dev]
        reasonable_bids = [b for b in ob_check.get('bids', []) if float(b[0]) >= price - max_dev]

        # Only block if NO levels at all within 10%
        if len(reasonable_asks) == 0:
            print(f"  [{self._ex_tag}:{asset}] NO ASKS within 10% -- cannot buy/close short")
            return
        if len(reasonable_bids) == 0:
            print(f"  [{self._ex_tag}:{asset}] NO BIDS within 10% -- cannot sell/close long")
            return

        # Log spread for monitoring but don't block
        best_ask = float(reasonable_asks[0][0])
        best_bid = float(reasonable_bids[0][0])
        spread_pct = (best_ask - best_bid) / best_bid * 100
        print(f"  [{self._ex_tag}:{asset}] LIQUIDITY: spread={spread_pct:.1f}% bids={len(reasonable_bids)} asks={len(reasonable_asks)}")

        # Volume warning (log only, don't block)
        if side == 'buy':
            exit_vol = sum(float(b[1]) for b in reasonable_bids)
            if exit_vol < qty:
                print(f"  [{self._ex_tag}:{asset}] LOW EXIT VOL: bids={exit_vol:.3f} vs qty={qty:.3f} -- may have trouble closing")
        elif side == 'sell':
            # Will need to BUY to close — check ask volume
            exit_vol = sum(float(a[1]) for a in reasonable_asks)
            if exit_vol < qty * 2:
                print(f"  [{self._ex_tag}:{asset}] EXIT VOL LOW: asks={exit_vol:.3f} need {qty*2:.3f}+ to close safely")
                return

        print(f"  [{self._ex_tag}:{asset}] LIQUIDITY OK: spread={spread_pct:.2f}% bids={len(reasonable_bids)} asks={len(reasonable_asks)}")

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
        # Only use prices within 5% of current price — testnet has fake walls at 2x price
        if result.get('status') != 'success' and '30208' in str(result.get('message', '')):
            try:
                ob = self.price_source.fetch_order_book(symbol, limit=10)
                limit_price = None
                max_deviation = price * 0.05  # 5% max from current price

                if side == 'buy' and ob.get('asks'):
                    for ask_price, ask_vol in ob['asks']:
                        if float(ask_price) <= price + max_deviation:
                            limit_price = float(ask_price)
                            break
                elif side == 'sell' and ob.get('bids'):
                    for bid_price, bid_vol in ob['bids']:
                        if float(bid_price) >= price - max_deviation:
                            limit_price = float(bid_price)
                            break

                if limit_price:
                    print(f"  [{self._ex_tag}:{asset}] Market rejected (price cap) -- retrying LIMIT @ ${limit_price:,.2f}")
                    result = self.price_source.place_order(
                        symbol=symbol, side=side, amount=qty,
                        order_type='limit', price=limit_price,
                    )
                else:
                    print(f"  [{self._ex_tag}:{asset}] Market rejected & no reasonable limit price within 5% -- SKIP")
            except Exception as e:
                print(f"  [{self._ex_tag}:{asset}] Limit fallback failed: {e}")

        if result.get('status') == 'success':
            order_id = result.get('order_id', 'unknown')
            if self.entry_dedup_scope == 'after_order' and dedup_candle_ts is not None:
                self.last_signal_candle[asset] = dedup_candle_ts

            # Get ACTUAL fill price from exchange (not signal price)
            # Testnet has thin liquidity — market fills can be far from expected price
            fill_price = price
            try:
                time.sleep(0.5)  # Brief wait for fill
                if self._exchange_client:
                    ex_positions = self._exchange_client.get_positions()
                    for p in ex_positions:
                        if asset in p.get('symbol', '') and float(p.get('qty', 0)) > 0:
                            ep = float(p.get('avg_entry_price', 0) or 0)
                            if ep > 0:
                                fill_price = ep
                                if abs(fill_price - price) / price > 0.005:  # >0.5% slippage
                                    print(f"  [{self._ex_tag}:{asset}] SLIPPAGE: expected ${price:,.2f} filled ${fill_price:,.2f} ({(fill_price-price)/price*100:+.2f}%)")
                            break
            except Exception:
                pass
            price = fill_price  # Use actual fill for all subsequent calculations

            _pos_cs = 0.0
            if self._exchange_name == 'delta':
                _pos_cs = {'BTC': 0.001, 'ETH': 0.01}.get(asset, 0.001)
            _ent_n = self._notional_usd(qty, price, asset, _pos_cs)

            # Compute initial stop-loss (L1) using ORDER BOOK + ATR
            # Priority: order book wall > ATR-based > percentage fallback
            sl_distance = current_atr * self.atr_stop_mult
            sl_distance = max(sl_distance, price * 0.003)  # minimum 0.3%
            sl_distance = min(sl_distance, price * 0.02)   # maximum 2%

            sl_source = "ATR"
            _obl = ob_levels or {}
            if action == 'LONG':
                sl_price = price - sl_distance
                # Use order book bid wall as support-based SL if available
                # Place SL just below the strongest bid wall (support)
                bid_wall = _obl.get('bid_wall', 0)
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
                ask_wall = _obl.get('ask_wall', 0)
                if ask_wall > 0 and ask_wall > price:
                    ob_sl = ask_wall * 1.001  # 0.1% above the wall
                    ob_dist_pct = (ob_sl - price) / price * 100
                    if 0.3 <= ob_dist_pct <= 2.0:
                        sl_price = ob_sl
                        sl_source = f"OB_ASK_WALL@${ask_wall:,.2f}"

            # Cap loss at SL: equity %, optional max_loss_usd_total, exit-fee reserve when fee PnL on.
            eq_ref = max(float(self.equity), 1.0)
            bounce_ok = _bounce_relief_from_ob(action, _obl, sl_source)
            relief = self.bounce_loss_relief_mult if bounce_ok else 1.0
            max_loss_usd = self._max_loss_price_move_usd(eq_ref, relief, _ent_n)
            loss_sl = _loss_usd_at_price(action, price, sl_price, qty, _pos_cs)
            if loss_sl > max_loss_usd > 0 and qty > 0:
                qm = _linear_qty_multiplier(qty, _pos_cs)
                if qm > 0:
                    max_dist = max_loss_usd / qm
                    if action == 'LONG':
                        sl_price = price - max_dist
                    else:
                        sl_price = price + max_dist
                    _cap_note = (
                        f"${self.max_loss_usd_total:,.0f} max/trade"
                        if self.max_loss_usd_total is not None
                        else f"{self.max_loss_per_trade_equity_pct:.2f}% equity"
                    )
                    print(
                        f"  [{self._ex_tag}:{asset}] RISK CAP: SL tightened — price PnL @ SL ≤ ~${max_loss_usd:,.2f} "
                        f"({_cap_note}"
                        f"{' × ' + str(round(relief, 2)) + ' bounce relief' if bounce_ok else ''})"
                    )

            imb = _obl.get('imbalance', 0)
            print(f"  [{self._ex_tag}:{asset}] ORDER OK: {order_id} | SL L1=${sl_price:,.2f} ({sl_source}) | OB imbalance={imb:+.2f}")
            print(
                f"  [{self._ex_tag}:{asset}] FINAL DECISION (trade): OPEN {action} / {direction_label}"
            )

            # SL managed by polling loop (10s) — no exchange stop orders
            # Exchange SL was creating orphan positions that cost $2-3 each to close
            sl_order_id = None

            _ent_fe = self._est_taker_fee_usd(_ent_n)
            if self._account_fees_in_pnl and _ent_fe > 0:
                print(
                    f"  [{self._ex_tag}:{asset}] FEE ACCT: est entry taker ~${_ent_fe:,.2f} "
                    f"on ${_ent_n:,.0f} notional ({self._taker_fee_pct_per_side:.3f}%/side model)"
                )

            # Record position
            self.positions[asset] = {
                'direction': action,          # LONG or SHORT
                'side': side,
                'entry_price': price,
                'qty': qty,
                'contract_size': _pos_cs,
                'order_id': order_id,
                'sl': sl_price,
                'sl_order_id': sl_order_id,
                'sl_levels': ['L1'],
                'peak_price': price,
                'entry_time': time.time(),
                'confidence': confidence,
                'reasoning': reasoning,
                'breakeven_moved': False,
                'agent_votes': orch_result.get('agent_votes', {}) if orch_result else {},
                'risk_equity_ref': eq_ref,
                'max_loss_usd_cap': max_loss_usd,
                'bounce_loss_relief': bounce_ok,
                'peak_upnl_usd': max(
                    0.0, _unrealized_pnl_usd(action, price, price, qty, _pos_cs)
                ),
                'peak_trail_armed': False,
                'peak_trail_pnl_pct': 0.0,
                'trade_leverage': float(trade_lev),
                'entry_notional_usd': _ent_n,
                'entry_fee_usd_est': _ent_fe,
            }
            self.last_trade_time[asset] = time.time()
        else:
            err = result.get('message', str(result))
            print(f"  [{self._ex_tag}:{asset}] ORDER FAILED: {err}")

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
        qty = float(pos.get('qty', 0) or 0)

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

        # OB imbalance — tracked for display only (testnet OB too unreliable for exits)
        ob_imbalance = ob_levels.get('imbalance', 0)

        age_sec = time.time() - float(pos.get('entry_time', time.time()))
        in_early = self._trail_early_window_sec > 0 and age_sec < self._trail_early_window_sec
        _pcs = float(pos.get('contract_size', 0) or 0)
        upnl_usd = _unrealized_pnl_usd(direction, entry, price, qty, _pcs)
        _, _, _, _net_u = self._fee_adjusted_upnl_usd(
            asset, pos, direction, entry, price, qty, _pcs
        )
        upnl_for_peak_rules = _net_u if self._account_fees_in_pnl else upnl_usd
        _pku = float(pos.get('peak_upnl_usd', 0) or 0)
        if upnl_for_peak_rules > _pku:
            pos['peak_upnl_usd'] = upnl_for_peak_rules
            _pku = upnl_for_peak_rules

        extra_sl_long_floor: Optional[float] = None
        extra_sl_short_ceiling: Optional[float] = None
        is_stuck = asset in self.failed_close_assets

        # ── 1b. Early window: giveback from peak → tighten SL or cash out before uPnL fades ──
        if in_early and not is_stuck and qty > 0:
            if (
                self._trail_close_on_early_giveback
                and _pku > 1e-9
                and upnl_for_peak_rules > 0
                and upnl_for_peak_rules <= _pku * (1.0 - self._trail_peak_usd_pullback)
            ):
                _ulab = "net" if self._account_fees_in_pnl else "gross"
                print(
                    f"  [{self._ex_tag}:{asset}] EARLY uPnL GIVEBACK ({_ulab}): ${upnl_for_peak_rules:,.0f} vs peak ${_pku:,.0f} "
                    f"(>{self._trail_peak_usd_pullback:.0%} off peak @ {age_sec:.0f}s) — banking profit"
                )
                self._close_position(
                    asset,
                    price,
                    f"Early uPnL giveback ({_ulab} ${upnl_for_peak_rules:,.0f} <= {(1-self._trail_peak_usd_pullback):.0%} of peak ${_pku:,.0f})",
                )
                return
            if direction == 'LONG' and peak > entry:
                mfe = peak - entry
                mfe_pct = (mfe / entry) * 100.0
                if mfe > 0 and mfe_pct >= self._trail_giveback_min_mfe_pct:
                    retr = peak - price
                    frac = retr / mfe
                    if (
                        frac >= self._trail_giveback_close_mfe_frac
                        and self._trail_close_on_early_giveback
                        and (_net_u > 0 if self._account_fees_in_pnl else pnl_pct > 0)
                    ):
                        print(
                            f"  [{self._ex_tag}:{asset}] EARLY MFE GIVEBACK: retraced {frac:.0%} of spike "
                            f"({age_sec:.0f}s) — cash out @ P&L {pnl_pct:+.2f}%"
                        )
                        self._close_position(
                            asset,
                            price,
                            f"Early MFE giveback ({frac:.0%} of favorable excursion)",
                        )
                        return
                    if frac >= self._trail_giveback_mfe_frac:
                        extra_sl_long_floor = entry + mfe * self._trail_giveback_lock_ratio
            elif direction == 'SHORT' and peak < entry:
                mfe = entry - peak
                mfe_pct = (mfe / entry) * 100.0
                if mfe > 0 and mfe_pct >= self._trail_giveback_min_mfe_pct:
                    retr = price - peak
                    frac = retr / mfe
                    if (
                        frac >= self._trail_giveback_close_mfe_frac
                        and self._trail_close_on_early_giveback
                        and (_net_u > 0 if self._account_fees_in_pnl else pnl_pct > 0)
                    ):
                        print(
                            f"  [{self._ex_tag}:{asset}] EARLY MFE GIVEBACK: retraced {frac:.0%} of dip "
                            f"({age_sec:.0f}s) — cash out @ P&L {pnl_pct:+.2f}%"
                        )
                        self._close_position(
                            asset,
                            price,
                            f"Early MFE giveback ({frac:.0%} of favorable excursion)",
                        )
                        return
                    if frac >= self._trail_giveback_mfe_frac:
                        extra_sl_short_ceiling = entry - mfe * self._trail_giveback_lock_ratio

        # ── 2. HARD STOP: max loss vs equity / max_loss_usd_total (+ slip) — USD-based ──
        eq_ref = float(pos.get('risk_equity_ref', self.equity) or self.equity)
        max_loss_cap = pos.get('max_loss_usd_cap')
        if max_loss_cap is None or float(max_loss_cap) <= 0:
            _n_fb = float(
                pos.get('entry_notional_usd')
                or self._notional_usd(qty, entry, asset, _pcs)
            )
            max_loss_cap = self._max_loss_price_move_usd(eq_ref, 1.0, _n_fb)
        else:
            max_loss_cap = float(max_loss_cap)
        emergency_usd = max_loss_cap * self.hard_stop_loss_slippage_mult
        upnl_for_hard = _net_u if self._account_fees_in_pnl else upnl_usd
        if upnl_for_hard <= -emergency_usd:
            if is_stuck:
                print(
                    f"  [{self._ex_tag}:{asset}] STUCK hard-stop zone "
                    f"(${upnl_for_hard:,.0f} vs -${emergency_usd:,.0f} cap) — can't close"
                )
                return
            _hn = "net" if self._account_fees_in_pnl else "gross"
            print(
                f"  [{self._ex_tag}:{asset}] HARD STOP at ${price:,.2f} | "
                f"{_hn} uPnL ${upnl_for_hard:,.0f} ≤ -${emergency_usd:,.0f} "
                f"(price-move cap ~${max_loss_cap:,.2f} × {self.hard_stop_loss_slippage_mult:.2f} slip) "
                f"| price P&L {pnl_pct:+.2f}%"
            )
            self._close_position(
                asset,
                price,
                f"Hard stop ({_hn} ${upnl_for_hard:,.0f} <= -${emergency_usd:,.0f})",
            )
            return

        # ── 2a. Peak PnL% trail — uses net % vs notional (after est. taker fees) when fee accounting on ──
        pnl_trail_pct = self._pnl_pct_peak_trail(
            asset, pos, direction, entry, price, qty, _pcs, pnl_pct
        )
        _trail_ok = pnl_trail_pct > 0 if self._account_fees_in_pnl else pnl_pct > 0
        if (
            self._peak_pnl_trail_enabled
            and not is_stuck
            and qty > 0
            and _trail_ok
        ):
            peak_tp = float(pos.get('peak_trail_pnl_pct', 0.0) or 0.0)
            if pnl_trail_pct >= self._peak_pnl_trail_activate_pct:
                pos['peak_trail_armed'] = True
                if pnl_trail_pct > peak_tp:
                    pos['peak_trail_pnl_pct'] = pnl_trail_pct
                    peak_tp = pnl_trail_pct
            if (
                pos.get('peak_trail_armed')
                and peak_tp >= self._peak_pnl_trail_activate_pct
            ):
                giveback_pp = peak_tp - pnl_trail_pct
                if giveback_pp >= self._peak_pnl_trail_giveback_pct:
                    _tlab = "net" if self._account_fees_in_pnl else "gross"
                    print(
                        f"  [{self._ex_tag}:{asset}] PEAK PnL% TRAIL ({_tlab}): peak +{peak_tp:.2f}% "
                        f"→ now +{pnl_trail_pct:.2f}% (Δ {giveback_pp:.2f}% ≥ "
                        f"{self._peak_pnl_trail_giveback_pct:.2f}% giveback) — exit to lock profit "
                        f"| price P&L {pnl_pct:+.2f}%"
                    )
                    self._close_position(
                        asset,
                        price,
                        f"Peak PnL% trail ({_tlab} +{peak_tp:.2f}% peak, +{pnl_trail_pct:.2f}% now)",
                    )
                    return

        # ── 2b. TIME-BASED EXIT: close zombie positions (Freqtrade pattern) ──
        duration_min = (time.time() - pos.get('entry_time', time.time())) / 60.0
        if duration_min >= self.max_hold_minutes and not is_stuck:
            if pnl_pct <= 0:
                print(f"  [{self._ex_tag}:{asset}] TIME EXIT: held {duration_min:.0f}min (max {self.max_hold_minutes:.0f}) P&L={pnl_pct:+.2f}% — closing stale loser")
                self._close_position(asset, price, f"Time exit ({duration_min:.0f}min, P&L={pnl_pct:+.2f}%)")
                return
            elif pnl_pct < 1.0:
                print(f"  [{self._ex_tag}:{asset}] TIME EXIT: held {duration_min:.0f}min with tiny P&L={pnl_pct:+.2f}% — closing")
                self._close_position(asset, price, f"Time exit ({duration_min:.0f}min, P&L={pnl_pct:+.2f}%)")
                return
            # If profitable >1%, let trailing SL handle it (don't cut winners)

        # ── 3. Check if current SL is hit ──
        # confirmed_close: wait for last CLOSED bar — direction aligns with chart close but exits lag (up to 1 signal bar).
        # intrabar_wick: use ticker + forming bar extreme on signal TF — faster; more stop-outs on spikes.
        # Hard stop (step 2) always uses live price.
        confirmed_close = closes[-2] if len(closes) >= 2 else price
        sl_hit = False
        sl_detail = ""
        if self.sl_exit_mode == 'intrabar_wick':
            form_lo = float(lows[-1]) if lows else price
            form_hi = float(highs[-1]) if highs else price
            if direction == 'LONG':
                probe = min(price, form_lo)
                sl_hit = probe <= sl
                sl_detail = f"probe=min(tick,barLow)={probe:,.2f} vs SL"
            else:
                probe = max(price, form_hi)
                sl_hit = probe >= sl
                sl_detail = f"probe=max(tick,barHigh)={probe:,.2f} vs SL"
        else:
            if direction == 'LONG' and confirmed_close <= sl:
                sl_hit = True
            elif direction == 'SHORT' and confirmed_close >= sl:
                sl_hit = True
            sl_detail = f"last closed C={confirmed_close:,.2f} vs SL"

        if sl_hit:
            if is_stuck:
                print(f"  [{self._ex_tag}:{asset}] STUCK SL {pnl_pct:+.2f}% (can't close — no liquidity)")
                return
            print(
                f"  [{self._ex_tag}:{asset}] SL {sl_levels[-1]} HIT at ${price:,.2f} "
                f"({sl_detail} ${sl:,.2f}) | P&L: {pnl_pct:+.2f}%"
            )
            self._close_position(asset, price, f"SL {sl_levels[-1]} hit ({sl_detail} ${sl:,.2f})")
            return

        # ── 4. TRAILING SL — ATR-based + minimum profit protection ──
        #
        # Balance between riding trends and locking profits:
        # - ATR trailing gives room for normal pullbacks
        # - Minimum profit floor guarantees we keep a % of gains
        # - Whichever is TIGHTER wins (more protective)
        #
        # Phase 1: Initial SL at L1 — hold through noise (early window: faster BE at early_breakeven_min_pct)
        # Phase 2: BREAKEVEN — can't lose capital (0.5% normally; early_breakeven_min_pct in first trailing.early_window_seconds)
        # Phase 3 (pnl >= profit_floor_start_pct): Protect 40% of peak + ATR trail; early_window tightens ATR via early_window_atr_mult_scale
        # Phase 4 (pnl >= 3%):   Protect 50% of profit + ATR trail (1.2x)
        # Phase 5 (pnl >= 5%):   Protect 60% of profit + ATR trail (1.0x)
        # Phase 6 (pnl >= 10%):  Protect 70% of profit + ATR trail (0.8x)
        #
        # SL = MAX(profit_floor, atr_trail) — always pick the tighter one

        new_sl = sl
        current_atr = atr_vals[-1] if atr_vals else price * 0.01
        be_thresh = self._trail_early_breakeven_pct if in_early else 0.5
        pf0 = self._trail_profit_floor_start_pct

        if direction == 'LONG':
            if pnl_pct >= be_thresh:
                if sl < entry:
                    new_sl = entry

            if pnl_pct >= pf0:
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

                if in_early:
                    atr_mult *= self._trail_early_atr_scale

                # Profit floor: lock in X% of peak profit
                profit_range = peak - entry
                floor_sl = entry + (profit_range * protect_pct)

                # ATR trail: peak - multiplier * ATR
                atr_trail_sl = peak - (current_atr * atr_mult)

                # Use whichever is TIGHTER (higher for LONG = more safe)
                best_sl = max(floor_sl, atr_trail_sl)
                if best_sl > new_sl and best_sl < price:
                    new_sl = best_sl

            if extra_sl_long_floor is not None:
                cap_sl = min(extra_sl_long_floor, price * 0.9995)
                if cap_sl > new_sl:
                    new_sl = cap_sl

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
            if pnl_pct >= be_thresh:
                if sl > entry:
                    new_sl = entry

            if pnl_pct >= pf0:
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

                if in_early:
                    atr_mult *= self._trail_early_atr_scale

                # Profit floor (SHORT: SL moves DOWN)
                profit_range = entry - peak
                floor_sl = entry - (profit_range * protect_pct)

                # ATR trail (SHORT: SL = peak + multiplier * ATR)
                atr_trail_sl = peak + (current_atr * atr_mult)

                # Use whichever is TIGHTER (lower for SHORT = more safe)
                best_sl = min(floor_sl, atr_trail_sl)
                if best_sl < new_sl and best_sl > price:
                    new_sl = best_sl

            if extra_sl_short_ceiling is not None and extra_sl_short_ceiling > price * 1.0003:
                if extra_sl_short_ceiling < new_sl:
                    new_sl = extra_sl_short_ceiling

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
        new_sl = round(new_sl, 2)
        min_move = max(self._trail_min_sl_move_usd, entry * self._trail_min_sl_move_pct)
        if pnl_pct > 0.35:
            min_move *= self._trail_in_profit_min_move_scale
        sl_moved = False
        if direction == 'LONG' and new_sl > sl + min_move:
            old_level = sl_levels[-1]
            pos['sl'] = new_sl
            sl_levels.append(f"L{len(sl_levels) + 1}")
            print(f"  [{self._ex_tag}:{asset}] SL {old_level}->{sl_levels[-1]}: ${sl:,.2f} -> ${new_sl:,.2f} | P&L: {pnl_pct:+.2f}%")
            sl = new_sl
            sl_moved = True
        elif direction == 'SHORT' and new_sl < sl - min_move:
            old_level = sl_levels[-1]
            pos['sl'] = new_sl
            sl_levels.append(f"L{len(sl_levels) + 1}")
            print(f"  [{self._ex_tag}:{asset}] SL {old_level}->{sl_levels[-1]}: ${sl:,.2f} -> ${new_sl:,.2f} | P&L: {pnl_pct:+.2f}%")
            sl = new_sl
            sl_moved = True

        # SL managed by polling (10s check) — no exchange stop orders
        # This avoids orphan positions from exchange SL fills

        # ── 5. EMA reversal exit (E1) — only on CONFIRMED candle + SIGNIFICANT profit ──
        # Use confirmed EMA (not live tick EMA) to avoid false reversals
        # Only flip when >= 5% profit AND 2 consecutive confirmed candles show reversal
        if not is_stuck and len(ema_vals) >= 3:
            confirmed_ema = ema_vals[-2]
            prev_confirmed_ema = ema_vals[-3]
            # Need 2 consecutive confirmed EMA bars falling/rising (not just one tick)
            ema_confirmed_falling = confirmed_ema < prev_confirmed_ema
            ema_confirmed_rising = confirmed_ema > prev_confirmed_ema

            if direction == 'LONG' and ema_confirmed_falling and confirmed_close < confirmed_ema and pnl_pct >= 5.0:
                print(f"  [{self._ex_tag}:{asset}] EMA REVERSAL (E1): CALL->PUT | exit ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
                self._close_position(asset, price, "EMA reversal (E1) - flipping to PUT")
                self.last_trade_time.pop(asset, None)
                self.last_close_time.pop(asset, None)
                return
            elif direction == 'SHORT' and ema_confirmed_rising and confirmed_close > confirmed_ema and pnl_pct >= 5.0:
                print(f"  [{self._ex_tag}:{asset}] EMA REVERSAL (E1): PUT->CALL | exit ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
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
            elif pnl_pct >= self._trail_profit_floor_start_pct:
                phase = "LOCK-40%"
            elif pnl_pct >= self._trail_early_breakeven_pct:
                phase = "BREAKEVEN"
            else:
                phase = "INITIAL"
            status = f"SAFE={sl_pnl_pct:+.2f}% RISK={risk_pct:.2f}% [{phase}]"
        elif sl_pnl_pct >= 0:
            status = "BREAKEVEN"

        imb = ob_levels.get('imbalance', 0)
        ob_tag = f"OB={imb:+.2f}" if imb != 0 else "OB=N/A"
        print(f"  [{self._ex_tag}:{asset}] HOLD {dir_label} @ ${entry:,.2f} | Now: ${price:,.2f} | {sl_chain} SL=${sl:,.2f} | P&L: {pnl_pct:+.2f}% | {status} | {ob_tag}")

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
            if self._exchange_client:
                positions = self._exchange_client.get_positions()
                for p in positions:
                    if asset in p.get('symbol', ''):
                        actual_qty = float(p.get('qty', qty))
                        break
        except Exception:
            pass

        # Close side is opposite of entry side
        close_side = 'sell' if direction == 'LONG' else 'buy'

        # Place close order — try limit at best price first, then market fallback
        # Testnet market orders fill at terrible prices due to thin books
        remaining_qty = actual_qty
        for close_attempt in range(3):
            # Try limit order at best bid/ask for better fill
            close_price = None
            try:
                ob = self.price_source.fetch_order_book(symbol, limit=10)
                if close_side == 'sell' and ob.get('bids'):
                    close_price = float(ob['bids'][0][0])
                elif close_side == 'buy' and ob.get('asks'):
                    close_price = float(ob['asks'][0][0])
            except Exception:
                pass

            if close_price and close_attempt == 0:
                # First attempt: aggressive limit order (reduceOnly to close, not open)
                result = self.price_source.place_order(
                    symbol=symbol,
                    side=close_side,
                    amount=remaining_qty,
                    order_type='limit',
                    price=close_price,
                    reduce_only=True,
                )
                print(f"  [{self._ex_tag}:{asset}] CLOSE LIMIT @ ${close_price:,.2f}")
                time.sleep(2)  # Wait for fill
                # Check if filled
                try:
                    ex_positions = self._exchange_client.get_positions()
                    still_has = any(asset in p.get('symbol','') and float(p.get('qty',0)) > 0 for p in ex_positions)
                    if not still_has:
                        price = close_price  # Use limit price for P&L
                        break
                    # Limit didn't fill — cancel and use market
                    open_orders = self._exchange_client.exchange.fetch_open_orders(symbol)
                    for o in open_orders:
                        try:
                            self._exchange_client.exchange.cancel_order(o['id'], symbol)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Fallback: market order (reduceOnly to close, not open new position)
            result = self.price_source.place_order(
                symbol=symbol,
                side=close_side,
                amount=remaining_qty,
                order_type='market',
                price=None,
                reduce_only=True,
            )

            if result.get('status') != 'success':
                err = result.get('message', str(result))
                if 'NoImmediate' in str(err) or 'cancel' in str(err).lower():
                    print(f"  [{self._ex_tag}:{asset}] CLOSE FAILED (no liquidity): {err}")
                    self.failed_close_assets[asset] = time.time()
                    return
                else:
                    print(f"  [{self._ex_tag}:{asset}] CLOSE WARNING: {err}")

            # Check remaining position on exchange
            time.sleep(1)
            try:
                if self._exchange_client:
                    ex_positions = self._exchange_client.get_positions()
                    still_open = False
                    for p in ex_positions:
                        if asset in p.get('symbol', '') and float(p.get('qty', 0)) > 0:
                            remaining_qty = float(p.get('qty', 0))
                            still_open = True
                    if not still_open:
                        break  # Fully closed
                    if close_attempt < 2:
                        print(f"  [{self._ex_tag}:{asset}] Partial fill — {remaining_qty} remaining, retrying...")
                    else:
                        print(f"  [{self._ex_tag}:{asset}] CLOSE INCOMPLETE — {remaining_qty} still open after 3 attempts")
                        self.failed_close_assets[asset] = time.time()
                        return
                else:
                    break
            except Exception:
                break

        # ── Fetch actual fill price from exchange (don't trust trigger price) ──
        actual_exit = price
        try:
            if self._exchange_client:
                # Check if position is fully closed and get last trade price
                time.sleep(0.5)
                ex_positions = self._exchange_client.get_positions()
                # If position closed, fetch recent fills
                still_open = any(
                    asset in p.get('symbol', '') and float(p.get('qty', 0)) > 0
                    for p in ex_positions
                )
                if not still_open:
                    # Try to get actual fill price from recent trades
                    try:
                        recent = self._exchange_client.exchange.fetch_my_trades(symbol, limit=3)
                        if recent:
                            last_fill = recent[-1]
                            actual_exit = float(last_fill.get('price', price))
                            if abs(actual_exit - price) / price > 0.002:  # >0.2% slippage
                                print(f"  [{self._ex_tag}:{asset}] EXIT SLIPPAGE: expected ${price:,.2f} filled ${actual_exit:,.2f} ({(actual_exit-price)/price*100:+.2f}%)")
                    except Exception:
                        pass  # Use trigger price if can't fetch fills
        except Exception:
            pass
        price = actual_exit

        # ── Calculate P&L (correct for contract-based exchanges) ──
        # Delta: qty = number of contracts. 1 BTC contract = 0.001 BTC, 1 ETH contract = 0.01 ETH
        # Bybit: qty = coin amount (e.g., 0.001 BTC)
        # P&L formula: (exit - entry) * qty * contract_size
        # Use exchange-reported size when it differed from our book (fills partials / manual edits).
        q_for_pnl = float(actual_qty) if actual_qty and float(actual_qty) > 0 else float(qty)
        if self._exchange_name == 'delta':
            contract_sizes = {'BTC': 0.001, 'ETH': 0.01}
            cs = contract_sizes.get(asset, 0.001)
        else:
            cs = 1.0  # Bybit qty is already in coin units

        if direction == 'LONG':
            pnl_pct = ((price - entry) / entry) * 100.0
            pnl_usd = (price - entry) * q_for_pnl * cs
        else:
            pnl_pct = ((entry - price) / entry) * 100.0
            pnl_usd = (entry - price) * q_for_pnl * cs

        n_ent = self._notional_usd(q_for_pnl, entry, asset, cs)
        n_x = self._notional_usd(q_for_pnl, price, asset, cs)
        est_fees_rt = 0.0
        pnl_net_est = float(pnl_usd)
        if self._account_fees_in_pnl and self._taker_fee_pct_per_side > 0:
            est_fees_rt = self._est_taker_fee_usd(n_ent) + self._est_taker_fee_usd(n_x)
            pnl_net_est = float(pnl_usd) - est_fees_rt

        sl_chain = '->'.join(pos.get('sl_levels', ['L1']))
        duration_min = (time.time() - pos.get('entry_time', time.time())) / 60.0

        if self._account_fees_in_pnl and est_fees_rt > 0:
            print(
                f"  [{self._ex_tag}:{asset}] CLOSED: gross {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | "
                f"net≈${pnl_net_est:+,.2f} after est. fees ${est_fees_rt:,.2f} "
                f"(2×{self._taker_fee_pct_per_side:.3f}% taker model) | {reason}"
            )
        else:
            print(
                f"  [{self._ex_tag}:{asset}] CLOSED: P&L {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | {reason} "
                f"(gross; exchange fees separate)"
            )

        # Track realized PnL for drawdown limits (net est. when fee accounting on)
        _pnl_sess = pnl_net_est if self._account_fees_in_pnl else pnl_usd
        self.session_realized_pnl += _pnl_sess
        self.daily_realized_pnl += _pnl_sess

        # Update agent orchestrator weights (learn from outcome)
        if self._orchestrator and pos.get('agent_votes'):
            try:
                self._orchestrator.record_outcome(
                    asset=asset,
                    direction=1 if direction == 'LONG' else -1,
                    pnl=_pnl_sess,
                )
            except Exception:
                pass  # Don't block close on feedback error

        # Update edge stats (wins use net est. when fee accounting on)
        if asset in self.edge_stats:
            self.edge_stats[asset]['total'] += 1
            _win = pnl_net_est > 0 if self._account_fees_in_pnl else pnl_pct > 0
            if _win:
                self.edge_stats[asset]['wins'] += 1
            else:
                self.edge_stats[asset]['losses'] += 1
            s = self.edge_stats[asset]
            s['win_rate'] = s['wins'] / s['total'] if s['total'] > 0 else 0.5

        # Log to journal (pnl_usd = gross; extra holds model fees + net est.)
        _extra = None
        if self._account_fees_in_pnl:
            _extra = {
                'est_roundtrip_fees_usd': round(est_fees_rt, 4),
                'pnl_usd_net_est': round(pnl_net_est, 2),
                'pnl_usd_gross': round(float(pnl_usd), 2),
            }
        try:
            self.journal.log_trade(
                asset=asset,
                action=direction,
                entry_price=entry,
                exit_price=price,
                qty=q_for_pnl,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                sl_progression=sl_chain,
                exit_reason=reason,
                llm_reasoning=pos.get('reasoning', ''),
                confidence=pos.get('confidence', 0.0),
                order_type='market',
                duration_minutes=duration_min,
                order_id=pos.get('order_id', ''),
                extra=_extra,
            )
        except Exception as e:
            logger.warning(f"Journal log failed: {e}")

        # Remove from positions
        del self.positions[asset]
        now = time.time()
        self.last_close_time[asset] = now
        self.last_trade_time[asset] = now  # Also set trade cooldown to prevent immediate re-entry

    # ------------------------------------------------------------------
    # Robust JSON extraction from LLM text (handles markdown, comments, etc.)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract valid JSON from LLM response text. Handles:
        - Markdown code fences (```json ... ```)
        - Preamble/postamble text around JSON
        - Nested braces
        - Trailing commas (common LLM mistake)
        - Single quotes instead of double quotes
        Returns the extracted JSON string, or raises ValueError.
        """
        text = text.strip()

        # 1. Try direct parse first (fast path)
        try:
            json.loads(text)
            return text
        except (json.JSONDecodeError, ValueError):
            pass

        # 2. Extract from markdown code fences
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if fence_match:
            candidate = fence_match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass

        # 3. Find outermost { ... } with brace matching
        start = text.find('{')
        if start >= 0:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            # Try fixing trailing commas
                            fixed = re.sub(r',\s*}', '}', candidate)
                            fixed = re.sub(r',\s*]', ']', fixed)
                            try:
                                json.loads(fixed)
                                return fixed
                            except (json.JSONDecodeError, ValueError):
                                pass
                        break

        # 4. Regex fallback for common patterns
        # Bear agent: {"risk_score": N, "reasoning": "..."}
        m = re.search(r'\{\s*"risk_score"\s*:\s*(\d+)\s*,\s*"reasoning"\s*:\s*"([^"]*)"', text)
        if m:
            return json.dumps({"risk_score": int(m.group(1)), "reasoning": m.group(2)})

        # Bull/main: {"action": "...", "confidence": N, ...}
        m = re.search(r'"action"\s*:\s*"(\w+)"\s*,\s*"confidence"\s*:\s*([\d.]+)', text)
        if m:
            action = m.group(1)
            conf = float(m.group(2))
            # Try to get position_size_pct and reasoning too
            size_m = re.search(r'"position_size_pct"\s*:\s*([\d.]+)', text)
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            return json.dumps({
                "action": action,
                "confidence": conf,
                "position_size_pct": float(size_m.group(1)) if size_m else 5,
                "reasoning": reason_m.group(1) if reason_m else "",
            })

        # Facilitator: {"proceed": true/false, ...}
        m = re.search(r'"proceed"\s*:\s*(true|false)', text, re.IGNORECASE)
        if m:
            proceed = m.group(1).lower() == 'true'
            size_m = re.search(r'"position_size_pct"\s*:\s*([\d.]+)', text)
            reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            return json.dumps({
                "proceed": proceed,
                "position_size_pct": float(size_m.group(1)) if size_m else 5,
                "reasoning": reason_m.group(1) if reason_m else "",
                "override_notes": "",
            })

        raise ValueError(f"No valid JSON found in LLM response: {text[:200]}")

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
            return self._extract_json(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Ollama returned invalid JSON: {e}")
            raise
        except Exception as e:
            logger.warning(f"Ollama query failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Claude API query (Anthropic SDK)
    # ------------------------------------------------------------------
    def _query_claude(self, prompt: str) -> str:
        """Query Claude API and return the raw text response (JSON extracted)."""
        if not self._claude_client:
            raise RuntimeError("Claude client not initialized")

        try:
            message = self._claude_client.messages.create(
                model=self._claude_model,
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )

            text = message.content[0].text.strip()
            return self._extract_json(text)
        except Exception as e:
            logger.warning(f"Claude query failed: {e} -- falling back to Ollama ({self.ollama_model})")
            print(f"  [AI] Claude failed ({type(e).__name__}): {str(e)[:80]} -- using Ollama fallback")
            return self._query_llm(prompt)

    # ------------------------------------------------------------------
    # Unified LLM router — picks Claude or Ollama based on exchange
    # ------------------------------------------------------------------
    def _query_llm_auto(self, prompt: str) -> str:
        """Route LLM query to Claude (Delta) or Ollama (Bybit) automatically.
        Any Claude failure falls back to Ollama llama3.2 — trading never stops."""
        if self._use_claude and self._claude_client:
            try:
                return self._query_claude(prompt)
            except Exception as e:
                logger.warning(f"Claude+fallback both failed, trying Ollama direct: {e}")
                return self._query_llm(prompt)
        return self._query_llm(prompt)

    # ------------------------------------------------------------------
    # UNIFIED LLM CALL — ALL 7 roles in ONE prompt, ONE response
    # Replaces: agent-LLM + bull + bear + 3 personas + facilitator
    # ------------------------------------------------------------------
    def _query_unified_llm(self, asset: str, signal: str, price: float,
                            current_ema: float, current_atr: float,
                            ema_direction: str, ema_slope_pct: float,
                            consecutive_trend: int, candle_lines: list,
                            support: float, resistance: float,
                            closes: list, highs: list, lows: list,
                            volumes: list, orch_result: dict,
                            pnl_history: str = '',
                            mtf_block: str = '',
                            mtf_aligned: int = 0,
                            mtf_n: int = 0,
                            entry_pattern_score: int = 0,
                            signal_trend_bars: int = 0,
                            signal_trend_phase: str = '',
                            mtf_stabilizing_count: int = 0) -> dict:
        """
        SINGLE LLM call that performs ALL 7 analysis roles:
          1. Agent Synthesis — interpret 13 math agents together
          2. Bull Case — why enter this trade
          3. Bear Case — what could go wrong (risk 0-10)
          4. Aggressive Persona — max profit recommendation
          5. Neutral Persona — balanced recommendation
          6. Conservative Persona — capital safety recommendation
          7. Facilitator — final verdict weighing all perspectives

        Returns:
            {
                'proceed': bool,
                'confidence': float (0-1),
                'position_size_pct': float,
                'risk_score': int (0-10),
                'trade_quality': int (0-10),
                'reasoning': str,
                'bull_case': str,
                'bear_case': str,
                'facilitator_verdict': str,
            }
        """
        forced_action = "LONG" if signal == "BUY" else "SHORT"
        direction_word = "CALL/LONG" if signal == "BUY" else "PUT/SHORT"

        # Agent data
        agent_lines = orch_result.get('agent_summary', []) if orch_result else []
        agent_data = "\n".join(agent_lines[:13])
        consensus_dir = orch_result.get('consensus_dir', '?') if orch_result else '?'
        consensus = orch_result.get('consensus', '?') if orch_result else '?'
        trend_reach = orch_result.get('trend_reach_score', 0) if orch_result else 0
        safety = orch_result.get('safety_score', 0) if orch_result else 0
        profit_lock = orch_result.get('profit_lock_score', 0) if orch_result else 0
        l_pred = orch_result.get('l_prediction', '?') if orch_result else '?'
        debate = orch_result.get('debate_summary', 'N/A') if orch_result else 'N/A'
        agent_conf = orch_result.get('confidence', 0.5) if orch_result else 0.5

        # Volume trend
        vol_trend = "UNKNOWN"
        if len(volumes) >= 10:
            recent_vol = sum(volumes[-5:]) / 5
            prev_vol = sum(volumes[-10:-5]) / 5
            if recent_vol < prev_vol * 0.7:
                vol_trend = "DECLINING"
            elif recent_vol > prev_vol * 1.3:
                vol_trend = "INCREASING"
            else:
                vol_trend = "FLAT"

        # EMA separation
        separation_pct = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0

        # Last N candles
        n_candles = min(15, len(candle_lines))
        candle_block = chr(10).join(candle_lines[-n_candles:]) if candle_lines else "N/A"

        # Account context
        equity = self.equity
        max_position_pct, _, _ = self._sizing_effective_caps()

        # Historical L-level patterns from journal (teaches LLM what works)
        historical_patterns = self._build_historical_pattern_context(asset)

        # ── Computed JSON anchor (no static demo numbers in the prompt) ──
        hint_tq = max(0, min(10, int(entry_pattern_score)))
        if mtf_n >= 5 and mtf_aligned >= 4:
            hint_tq = min(10, hint_tq + 1)
        _phase = (signal_trend_phase or ema_direction).strip()
        _stabilizing = _phase.endswith("_STABILIZING")
        if _stabilizing:
            hint_tq = max(0, hint_tq - 1)
        # Trend-following signal while slope decelerates: legacy direction still "right"
        # on paper but reversal / crossover risk dominates (e.g. SHORT + FALLING_STABILIZING).
        _trend_follow = (
            (signal == "SELL" and ema_direction == "FALLING")
            or (signal == "BUY" and ema_direction == "RISING")
        )
        _stab_trend_conflict = _stabilizing and _trend_follow
        if _stab_trend_conflict:
            hint_tq = max(0, hint_tq - 1)
        mtf_ratio = (mtf_aligned / mtf_n) if mtf_n > 0 else 0.0
        ac = max(0.0, min(1.0, float(agent_conf)))
        tr = max(0.0, min(1.0, float(trend_reach)))
        sf = max(0.0, min(1.0, float(safety)))
        hint_conf = 0.26 + 0.36 * ac + 0.24 * mtf_ratio + 0.14 * (hint_tq / 10.0)
        hint_conf = max(0.38, min(0.93, hint_conf))
        hint_risk = int(round(10.0 * (1.0 - sf)))
        hint_risk = max(0, min(10, hint_risk))
        if _stab_trend_conflict:
            hint_risk = min(10, hint_risk + 2)
            hint_conf = max(0.38, min(0.88, hint_conf - 0.12))
        if hint_tq >= 6 and mtf_n >= 5 and mtf_aligned >= 4 and not _stab_trend_conflict:
            hint_conf = max(hint_conf, 0.65)
        _tq_bar = 5 if _stab_trend_conflict else 4
        hint_proceed = bool(hint_tq >= _tq_bar and hint_risk < self.bear_veto_threshold)
        span = max(0.5, float(max_position_pct) - 2.0)
        hint_size = 2.0 + (hint_tq / 10.0) * span
        hint_size = max(1.0, min(float(max_position_pct), hint_size))
        hint_size = round(hint_size, 1)
        pred_anchor = str(l_pred).strip() if l_pred else ''
        if not pred_anchor or pred_anchor in ('?', 'None', 'nan'):
            pred_anchor = 'L4'
        debate_snip = str(debate).replace('"', "'")[:120]
        _regime = (
            "trend_follow_into_stabilization"
            if _stab_trend_conflict
            else "standard"
        )
        _stab_note = (
            f"stab_vs_trend=yes mtf_stab={mtf_stabilizing_count}/{mtf_n or 0} "
            f"(do not extrapolate past trend alone; crossover/bounce risk)"
            if _stab_trend_conflict
            else f"mtf_stab={mtf_stabilizing_count}/{mtf_n or 0}"
        )
        computed_anchor = {
            'proceed': hint_proceed,
            'confidence': round(hint_conf, 2),
            'position_size_pct': hint_size,
            'risk_score': hint_risk,
            'trade_quality': hint_tq,
            'predicted_l_level': pred_anchor,
            'trend_phase': _phase,
            'ema_direction': ema_direction,
            'regime': _regime,
            'bull_case': (
                f"[DATA] pattern={entry_pattern_score}/10 trendBars={signal_trend_bars} "
                f"phase={_phase} dir={ema_direction} {_stab_note} "
                f"MTF_align={mtf_aligned}/{mtf_n} slope={ema_slope_pct:+.3f}%/bar "
                f"reach={tr:.0%} vol={vol_trend}"
            ),
            'bear_case': (
                f"[DATA] safety={sf:.0%} sep_vs_EMA={separation_pct:.2f}% risk_hint={hint_risk}/10 "
                f"{_stab_note} | {debate_snip}"
            ),
            'facilitator_verdict': (
                f"[DATA] anchor proceed={hint_proceed} conf={hint_conf:.2f} tq={hint_tq} "
                f"size%={hint_size} agents_conf={ac:.2f}"
            ),
            'agent_conflicts': f"consensus={consensus} dir={consensus_dir}"[:120],
        }
        computed_json_line = json.dumps(computed_anchor, ensure_ascii=False)

        _ema_dir_prompt_note = (
            "Signal timing is LIVE: ema_direction uses current tick as the forming bar's close on the signal "
            "timeframe (no wait for that bar to close)."
            if self.signal_timing == 'live'
            else "ema_direction is the last closed bar vs the prior bar (can still read FALLING after a long decline)."
        )
        _mtf_snapshot_note = (
            "signal TF uses live tick as forming-bar close; other TFs use last closed bar in snapshots"
            if self.signal_timing == 'live'
            else "last closed bar per TF"
        )

        prompt = f"""You are a COMPLETE trading decision system for {asset} on {self._exchange_name.upper()}.
You must play ALL 7 roles below and produce ONE final decision in a SINGLE JSON response.

=== INSTRUMENT: USDT PERPETUAL FUTURES (LONG + SHORT) ===
Both directions are valid: CALL/LONG (buy) and PUT/SHORT (sell) must be evaluated with equal rigor — do not default to long-only bias when the setup is bearish.

=== OUR STRATEGY: EMA(8) CROSSOVER + TRAILING STOP-LOSS (L1→Ln) ===
This is a CALL/PUT strategy based on EMA(8) crossovers with a multi-level trailing SL system:

CALL (LONG): Price crosses ABOVE EMA(8) → enter LONG → trail SL through L-levels:
  L1: Move SL to breakeven (0.5% above entry) when price moves +1 ATR
  L2: Lock 40% of unrealized profit
  L3: Lock 50% of unrealized profit
  L4: Lock 55% ... and so on, each level locks more profit
  L10+: Deep trend — lock 70%+ of profit. These are the BIG winners.
  L20+: Exceptional trend — lock 80%+ of profit.

PUT (SHORT): Price crosses BELOW EMA(8) → enter SHORT → same L-level trailing but inverted.

CRITICAL INSIGHT: The trailing SL system means we ONLY profit when price trends strongly
AFTER our entry. Choppy/ranging markets kill us at L1-L2. We need MOMENTUM.

Current signal: {"CALL" if signal == "BUY" else "PUT"} ({direction_word})
EMA just crossed {"UP (bullish)" if signal == "BUY" else "DOWN (bearish)"}

=== WHAT MAKES A WINNING TRADE (L4+ EXIT) ===
- Strong EMA slope ({ema_slope_pct:+.3f}%/bar) — steeper = more momentum = reaches higher L-levels
- Multiple consecutive trend bars ({consecutive_trend} now) — 5+ bars = strong trend building
- Increasing volume — confirms institutional participation, trend has legs
- Price just crossed EMA (not already 5%+ away) — early entry catches full move
- Support/resistance gives room to run — CALL with resistance far away, PUT with support far away

=== WHAT MAKES A LOSING TRADE (L1-L2 EXIT / HARD STOP) ===
- Flat or weak EMA slope — no momentum, SL gets hit at L1-L2
- Low trend bars (0-3) — choppy market, no direction
- Declining volume — move is exhausted, about to reverse
- Late entry (price already 3%+ from EMA) — missed the move, entering at top/bottom
- Near key level ({"resistance" if signal == "BUY" else "support"}) — price will bounce back

=== ACCOUNT ===
Equity: ${equity:,.2f} | Max position: {max_position_pct}% | History: {pnl_history}

=== MARKET DATA ===
Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} (direction={ema_direction}, trend_phase={_phase}) | EMA-price gap: {separation_pct:.2f}%
ATR: ${current_atr:.2f} | Slope: {ema_slope_pct:+.3f}%/bar | Trend bars: {consecutive_trend}
Support: ${support:.2f} | Resistance: ${resistance:.2f} | Volume: {vol_trend}
Note: trend_phase uses EMA slope deceleration vs recent bars; *_STABILIZING means momentum is fading — possible direction change soon (not pure R/F chase).

=== STABILISATION VS PRIOR TREND (do not ignore) ===
{_ema_dir_prompt_note}
trend_phase={_phase} adds whether that move is still accelerating or stabilising (slowing).
If phase is FALLING_STABILIZING or RISING_STABILIZING, the *history* may still look like one trend, but structurally the edge is weaker: mean-reversion, whipsaw, and EMA crossovers *against* the legacy trend are more likely next.
Current signal is {"trend-following (same side as ema_direction)" if _trend_follow else "not pure trend-follow vs EMA step"}.
If trend-following AND *_STABILIZING: do NOT justify aggressive {"SHORT" if signal == "SELL" else "LONG" if signal == "BUY" else "entry"} only because "price was falling/rising for many bars". Explicitly address bounce/crossover risk, support/resistance, and L1-L2 failure; lean proceed=false or low trade_quality unless fresh breakdown/continuation is visible in candles and agents.
MTF: {mtf_stabilizing_count} of {mtf_n} analyzed timeframes are in *_STABILIZING on the last bar (broad slowing increases reversal pressure).

=== MULTI-TIMEFRAME STACK (EMA{self.ema_period} on each TF; {_mtf_snapshot_note}; refreshed every engine poll) ===
Signal timeframe is {self.signal_timeframe} — that is the bar size for ENTRY rules above.
MTF compact uses R/F for strong trend and r/f when that TF is stabilising (slope decelerating).
Alignment: {mtf_aligned} of {mtf_n} analyzed timeframes agree on EMA direction ({ema_direction}) with the signal TF.
{(mtf_block if mtf_block.strip() else '  (no MTF snapshot)')}

Use this stack for REALISTIC trend judgment: broad agreement across TFs supports a stable move; lower TFs strongly opposing higher TFs suggests chop or pullback — favor SKIP or lower confidence when the stack is mixed.

Last {n_candles} candles ({self.signal_timeframe}):
{candle_block}

=== 13 MATH AGENT OUTPUTS ===
{agent_data}
Consensus: {consensus_dir} {consensus} | Agents conf={agent_conf:.2f}
Trend Reach: {trend_reach:.0%} ({l_pred}) | Safety: {safety:.0%} | Lock: {profit_lock:.0%}
Debate: {debate}

{historical_patterns}

=== YOUR JOB: PREDICT IF THIS TRADE REACHES L4+ OR DIES AT L1-L2 ===

ROLE 1 - AGENT SYNTHESIS: Do the 13 math agents agree this trend has legs? What L-level do they predict?
ROLE 2 - BULL CASE: Based on slope, trend bars, volume — can this reach L4+? What's the realistic target?
ROLE 3 - BEAR CASE: What kills this trade early? Score risk 0-10.
  Specifically check: flat slope → L1 exit | declining volume → reversal | near {"resistance" if signal == "BUY" else "support"} → bounce | late entry → L2 max
ROLE 4 - AGGRESSIVE: If this reaches L10+, what's the potential? Size up to {max_position_pct}%?
ROLE 5 - NEUTRAL: Does the data justify entry? Or is this a L1-L2 chop trap?
ROLE 6 - CONSERVATIVE: Given our historical L-level distribution, is this worth the risk?
ROLE 7 - FACILITATOR: FINAL VERDICT — Will this trade reach L4+ (ENTER) or die at L1-L2 (SKIP)?

=== DECISION RULES ===
- Weigh the MULTI-TIMEFRAME STACK: if most TFs align with {ema_direction}, trend is more credible; if many TFs disagree, treat as unstable and lower confidence or set proceed=false.
- If trend_phase ends with _STABILIZING on the signal TF, treat reversal/bounce risk as elevated vs a fresh R/F trend — reflect in risk_score, trade_quality, and confidence.
- If regime is trend_follow_into_stabilization (see anchor): past trend direction must NOT dominate — weight possible reversal and next-candle crossover; default skeptical on full-size trend entries.
- SERVER-ENFORCED FLOOR (post-LLM): if parsed trade_quality >= 6 and MTF >= 4/5 ({mtf_aligned}/{mtf_n}), confidence is raised to at least 0.65 EXCEPT when trend-following signal meets *_STABILIZING (no floor in that case — reversal risk).
- SERVER MAY cap confidence when trend_follow + stabilizing — do not assume high conviction.
- predicted_l_level: Your best estimate of where trailing SL exits (L1, L2, L4, L10+, etc.)
- risk_score: 0=safe, 5=moderate — veto entry if >={self.bear_veto_threshold}; size reduction zone if >={self.bear_reduce_threshold}
- trade_quality: 0=terrible, 5=marginal, 8+=excellent (measures L-level potential)
  quality 8+: Strong trend indicators → likely L4+ → ENTER with full size
  quality 5-7: Moderate setup → might reach L3-L4 → enter with reduced size
  quality ≤4: Weak/choppy → likely L1-L2 → SKIP (save capital for better setup)
- confidence: 0.0-1.0 (your HONEST estimate based on historical patterns above)
- If agents conflict OR risk>={self.bear_veto_threshold} OR quality<=3: set proceed=false
- Self-consistency: if predicted_l_level is only L1/L2 (chop, scratch), proceed must be false unless trade_quality>=7 with explicit continuation evidence; do not output L1-L2 prediction with proceed=true and high confidence (server will reject tq<5).
- position_size_pct: 1-{max_position_pct}% (scale with trade_quality and predicted L-level)
- REMEMBER: It's BETTER to skip a marginal trade than to lose at L1. We profit from PATIENCE.

=== COMPUTED JSON ANCHOR (code-derived from pattern score, MTF, orchestrator — not placeholders) ===
Use the SAME keys. Keep booleans/ints/floats within ±1 step of anchor values unless a specific candle/agent fact contradicts (say which in prose).
Expand bull_case, bear_case, facilitator_verdict into full sentences; you may keep the [DATA] prefix facts and add reasoning after.
{computed_json_line}

Respond with ONLY that JSON object shape (no markdown, no extra text)."""

        try:
            raw = self._query_llm_auto(prompt)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                raw = self._extract_json(raw)
                result = json.loads(raw)

            # Validate and clamp all fields
            result['proceed'] = bool(result.get('proceed', False))
            result['confidence'] = max(0.0, min(1.0, float(result.get('confidence', 0.5))))
            result['position_size_pct'] = max(1, min(max_position_pct, float(result.get('position_size_pct', 3))))
            result['risk_score'] = max(0, min(10, int(result.get('risk_score', 5))))
            result['trade_quality'] = max(0, min(10, int(result.get('trade_quality', 5))))

            # Server-side enforcement — override LLM if it ignores rules (skipped when ai.confidence_only_gates)
            tq = result['trade_quality']
            rs = result['risk_score']
            if not self._confidence_only_gates:
                if tq <= 3:
                    result['proceed'] = False
                    result['confidence'] = min(result['confidence'], 0.3)
            if rs >= self.bear_veto_threshold:
                result['proceed'] = False
            if not self._confidence_only_gates:
                if tq <= 5:
                    result['position_size_pct'] = min(
                        result['position_size_pct'], max_position_pct * 0.5
                    )

            chop_override_msg = ''
            pred_l_raw = str(result.get('predicted_l_level', ''))
            if (
                not self._confidence_only_gates
                and result['proceed']
                and _pred_l_level_chop_zone(pred_l_raw)
                and tq < 5
            ):
                result['proceed'] = False
                result['confidence'] = min(result['confidence'], 0.45)
                chop_override_msg = (
                    f"  [{self._ex_tag}:{asset}] LLM OVERRIDE: predicted_l_level={pred_l_raw!r} "
                    f"is L1-L2/chop zone — cannot ENTER with trade_quality={tq}<5"
                )

            # Deterministic confidence floor: small local LLMs often echo ~0.5; MTF + quality override.
            mtf_ok = mtf_n >= 5 and mtf_aligned >= 4
            _pg = (signal_trend_phase or ema_direction).strip()
            _stab_gate = _pg.endswith("_STABILIZING")
            _tf_gate = (
                (signal == "SELL" and ema_direction == "FALLING")
                or (signal == "BUY" and ema_direction == "RISING")
            )
            _no_conf_floor = _stab_gate and _tf_gate
            if not self._confidence_only_gates:
                if tq >= 6 and mtf_ok and not _no_conf_floor:
                    floor_c = 0.65
                    if result['confidence'] < floor_c:
                        result['confidence'] = floor_c
                        print(
                            f"  [{self._ex_tag}:{asset}] CONF BOOST: LLM conf raised to {floor_c:.2f} "
                            f"(trade_quality={tq}>=6, MTF {mtf_aligned}/{mtf_n}>=4/5)"
                        )
                if _no_conf_floor:
                    cap_c = 0.60
                    prev_c = result['confidence']
                    result['confidence'] = min(prev_c, cap_c)
                    if prev_c > cap_c:
                        print(
                            f"  [{self._ex_tag}:{asset}] CONF CAP: LLM conf capped to {cap_c:.2f} "
                            f"(trend-follow + stabilising phase={_pg} — reversal/crossover risk)"
                        )

            # Print compact summary of all 7 roles
            bull = str(result.get('bull_case', ''))[:60]
            bear = str(result.get('bear_case', ''))[:60]
            fac = str(result.get('facilitator_verdict', ''))[:80]
            conflicts = str(result.get('agent_conflicts', ''))[:40]
            pred_l = str(result.get('predicted_l_level', '?'))

            print(f"  [{self._ex_tag}:{asset}] UNIFIED LLM (1 call, 7 roles):")
            print(f"    AGENTS: quality={tq}/10 conflicts=[{conflicts}]")
            print(f"    BULL: {bull}")
            print(f"    BEAR: risk={rs}/10 | {bear}")
            print(f"    L-LEVEL PREDICTION: {pred_l} | {'ENTER' if result['proceed'] else 'REJECT'} conf={result['confidence']:.2f} size={result['position_size_pct']:.0f}%")
            print(f"    FACILITATOR: {fac}")
            if chop_override_msg:
                print(chop_override_msg)

            _trade_dir = (
                "LONG / CALL"
                if signal == "BUY"
                else ("SHORT / PUT" if signal == "SELL" else "FLAT")
            )
            if result["proceed"]:
                print(
                    f"  [{self._ex_tag}:{asset}] FINAL DECISION (LLM): APPROVED {_trade_dir}"
                )
            else:
                print(
                    f"  [{self._ex_tag}:{asset}] FINAL DECISION (LLM): NO ENTER — {_trade_dir} rejected"
                )

            return result

        except Exception as e:
            logger.warning(f"[{asset}] Unified LLM failed: {e}")
            # spec §5.3: fallback confidence with EMA direction — proceed; gate uses llm_fallback_confidence_gate
            return {
                'proceed': True,
                'confidence': self.llm_fallback_confidence,
                'llm_unavailable': True,
                'position_size_pct': 3,
                'risk_score': 5,
                'trade_quality': 6,
                'predicted_l_level': '?',
                'bull_case': f'LLM unavailable ({type(e).__name__}) — following EMA signal',
                'bear_case': 'Risk not LLM-scored — using default risk_score=5',
                'facilitator_verdict': 'EMA indicator gate passed; LLM offline',
                'agent_conflicts': 'unknown',
            }

    def _query_bear_agent(self, asset: str, signal: str, price: float,
                          current_ema: float, current_atr: float,
                          ema_direction: str, ema_slope_pct: float,
                          consecutive_trend: int, candle_lines: list,
                          support: float, resistance: float,
                          closes: list, highs: list, lows: list,
                          volumes: list, bull_confidence: float,
                          bull_reasoning: str) -> dict:
        """
        Bear/Risk veto agent — same LLM, contrarian prompt.
        Scores risk 0-10. Higher = more dangerous trade.
        """
        separation_pct = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0

        # Key level distance
        if signal == "BUY":
            level_dist_pct = (resistance - price) / price * 100 if price > 0 and resistance > price else 99
        else:
            level_dist_pct = (price - support) / price * 100 if price > 0 and support < price else 99

        # Volume trend (last 5 candles)
        vol_trend = "UNKNOWN"
        if len(volumes) >= 7:
            recent_vol = sum(volumes[-5:]) / 5
            prev_vol = sum(volumes[-10:-5]) / 5 if len(volumes) >= 10 else recent_vol
            if recent_vol < prev_vol * 0.7:
                vol_trend = "DECLINING (weak conviction)"
            elif recent_vol > prev_vol * 1.3:
                vol_trend = "INCREASING (strong)"
            else:
                vol_trend = "FLAT"

        # Candle wicks (last 3 candles)
        wick_warning = ""
        if len(highs) >= 3 and len(lows) >= 3 and len(closes) >= 3:
            for i in range(-3, 0):
                body = abs(closes[i] - (closes[i-1] if i > -len(closes) else closes[i]))
                upper_wick = highs[i] - max(closes[i], closes[i-1] if i > -len(closes) else closes[i])
                lower_wick = min(closes[i], closes[i-1] if i > -len(closes) else closes[i]) - lows[i]
                total_range = highs[i] - lows[i]
                if total_range > 0 and (upper_wick > total_range * 0.5 or lower_wick > total_range * 0.5):
                    wick_warning = "REJECTION WICKS detected"
                    break

        # Recent loss count
        recent_losses = 0
        try:
            recent_trades = self.journal.get_recent_trades(asset=asset, limit=10)
            if recent_trades:
                for t in recent_trades:
                    if float(t.get('pnl_usd', 0)) < 0:
                        recent_losses += 1
        except Exception:
            try:
                recent_trades = self.journal.load_trades()
                asset_trades = [t for t in recent_trades[-10:] if t.get('asset') == asset]
                for t in asset_trades:
                    if float(t.get('pnl_usd', 0)) < 0:
                        recent_losses += 1
            except Exception:
                pass

        n_candles = min(10, len(candle_lines))
        candle_data = chr(10).join(candle_lines[-n_candles:]) if candle_lines else "N/A"

        bear_prompt = f"""You are a RISK ANALYST for {asset}/USDT perpetual futures.
Your job is to find reasons this trade SHOULD NOT be taken.
You are the last defense before capital is risked.

A bull analyst recommends: {signal} with confidence {bull_confidence:.2f}
Bull says: "{bull_reasoning}"

MARKET DATA:
Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction})
ATR: ${current_atr:.2f} | Support: ${support:.2f} | Resistance: ${resistance:.2f}
EMA slope: {ema_slope_pct:+.3f}%/bar | Trend bars: {consecutive_trend}
Distance from EMA: {separation_pct:.2f}%
Volume trend: {vol_trend}
{wick_warning}

LAST {n_candles} CANDLES ({self.signal_timeframe}):
{candle_data}

SCORE EACH RISK 0-2 POINTS:

1. LATE ENTRY: Price {separation_pct:.2f}% from EMA.
   - Crypto is VERY volatile especially on testnet. 15%+ from EMA = clearly late (2pts)
   - 8-15% = somewhat extended (1pt)
   - Under 8% = normal for crypto momentum, score 0

2. REVERSAL SIGNS: Any rejection wicks, dojis, shrinking bodies? {wick_warning or "Check candles above."}
   - Clear reversal pattern (multiple wicks against trend) = 2pts
   - Single wick or doji = 1pt
   - No reversal signs = 0pts

3. VOLUME: Volume is {vol_trend}.
   - Declining volume on strong price move = weak, likely to reverse (2pts)
   - Flat volume = uncertain (1pt)
   - Increasing volume = confirming trend (0pts)

4. KEY LEVEL: {"LONG near resistance $"+f"{resistance:,.2f} ({level_dist_pct:.1f}% away)" if signal=="BUY" else "SHORT near support $"+f"{support:,.2f} ({level_dist_pct:.1f}% away)"}.
   - Within 0.5% of key level = very dangerous (2pts)
   - 0.5-2% = caution (1pt)
   - More than 2% away = fine (0pts)

5. LOSS STREAK: {recent_losses} losses in last 10 {asset} trades.
   - 5+ recent losses = regime clearly unfavorable (2pts)
   - 3-4 losses = caution (1pt)
   - 0-2 losses = normal (0pts)

TOTAL risk_score = sum (0-10)
0-6: Acceptable (proceed or reduce)
7-8: High risk (reduce position)
9-10: Extreme risk ONLY (veto)

IMPORTANT: Crypto markets are volatile. A 3% move from EMA is NORMAL, not "late."
Only score 9-10 if MULTIPLE serious risks combine (e.g. 5%+ from EMA AND reversal wicks AND declining volume).
A single risk factor alone should NOT push score above 7.

Respond ONLY with JSON:
{{"risk_score": <0-10>, "reasoning": "specific risks found"}}"""

        try:
            raw = self._query_llm_auto(bear_prompt)
            # Double-safety: _query_llm_auto already extracts JSON, but parse might still fail
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                raw = self._extract_json(raw)
                result = json.loads(raw)
            risk_score = int(result.get('risk_score') or 5)
            risk_score = max(0, min(10, risk_score))  # clamp 0-10
            reasoning = str(result.get('reasoning', ''))[:120]
            return {'risk_score': risk_score, 'reasoning': reasoning}
        except Exception as e:
            logger.warning(f"[{asset}] Bear agent failed: {e}")
            # SAFETY DEFAULT: assume moderate risk when bear is down, not 0 (which disables veto)
            return {'risk_score': 5, 'reasoning': f'bear unavailable ({type(e).__name__}) — cautious default'}

    # ------------------------------------------------------------------
    # 3 Risk Personas (TradingAgents pattern)
    # ------------------------------------------------------------------
    def _query_risk_personas(self, asset: str, signal: str, price: float,
                             current_ema: float, current_atr: float,
                             ema_direction: str, pnl_history: str,
                             bull_confidence: float, bull_reasoning: str,
                             bear_risk_score: int, bear_reasoning: str) -> dict:
        """
        Three risk perspectives on the same trade: Aggressive, Neutral, Conservative.
        Each returns a position_size_pct recommendation. Facilitator synthesizes.
        """
        separation_pct = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
        context = f"""Asset: {asset} | Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction})
ATR: ${current_atr:.2f} | Signal: {signal} | Distance from EMA: {separation_pct:.2f}%
Bull says: conf={bull_confidence:.2f} "{bull_reasoning}"
Bear says: risk={bear_risk_score}/10 "{bear_reasoning}"
Recent P&L history: {pnl_history}"""

        personas = {}
        for persona in ['AGGRESSIVE', 'NEUTRAL', 'CONSERVATIVE']:
            if persona == 'AGGRESSIVE':
                style = """You are an AGGRESSIVE risk manager. You believe in maximizing returns when signals align.
Your bias: bigger positions on strong setups, smaller only when multiple risks converge.
Position range: 3-20% of equity. Only go below 3% if risk_score >= 8."""
            elif persona == 'NEUTRAL':
                style = """You are a NEUTRAL risk manager. You seek balanced risk-reward.
Your bias: standard positions, hedge when uncertain, scale with confidence.
Position range: 2-10% of equity. Scale linearly with bull confidence."""
            else:
                style = """You are a CONSERVATIVE risk manager. Capital preservation is your #1 goal.
Your bias: small positions always, skip marginal setups, only full size on perfect setups.
Position range: 1-5% of equity. Only go above 3% if bull_conf >= 0.95 AND bear_risk <= 3."""

            prompt = f"""{style}

TRADE CONTEXT:
{context}

Based on the bull/bear analysis above, recommend:
1. position_size_pct (within your range)
2. Should this trade proceed? (yes/no)
3. One-line reasoning

Respond ONLY with JSON:
{{"position_size_pct": <number>, "proceed": true/false, "reasoning": "brief"}}"""

            try:
                raw = self._query_llm_auto(prompt)
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    raw = self._extract_json(raw)
                    result = json.loads(raw)
                personas[persona] = {
                    'size_pct': float(result.get('position_size_pct') or 5),
                    'proceed': bool(result.get('proceed', True)),
                    'reasoning': str(result.get('reasoning', ''))[:80],
                }
            except Exception:
                # SAFE defaults: conservative when LLM fails
                defaults = {'AGGRESSIVE': 5, 'NEUTRAL': 3, 'CONSERVATIVE': 1}
                personas[persona] = {
                    'size_pct': defaults[persona],
                    'proceed': persona == 'AGGRESSIVE',  # Only aggressive proceeds on failure
                    'reasoning': f'{persona.lower()} default (LLM failed)',
                }

        return personas

    # ------------------------------------------------------------------
    # Facilitator Agent (TradingAgents pattern)
    # ------------------------------------------------------------------
    def _query_facilitator(self, asset: str, signal: str, price: float,
                           bull_confidence: float, bull_reasoning: str,
                           bear_risk_score: int, bear_reasoning: str,
                           risk_personas: dict) -> dict:
        """
        Facilitator evaluates bull vs bear arguments + risk personas,
        makes the FINAL trade decision. Returns action + size.
        """
        agg = risk_personas.get('AGGRESSIVE', {})
        neu = risk_personas.get('NEUTRAL', {})
        con = risk_personas.get('CONSERVATIVE', {})

        prompt = f"""You are the TRADE FACILITATOR for {asset}. You review all agent opinions and make the final call.
Your goal: maximize risk-adjusted returns. You are the last gate before capital is deployed.

SIGNAL: {signal}

BULL ANALYST:
  Confidence: {bull_confidence:.2f}
  Reasoning: "{bull_reasoning}"

BEAR RISK ANALYST:
  Risk Score: {bear_risk_score}/10
  Reasoning: "{bear_reasoning}"

RISK COMMITTEE:
  Aggressive: size={agg.get('size_pct',5)}% proceed={agg.get('proceed',True)} — "{agg.get('reasoning','')}"
  Neutral:    size={neu.get('size_pct',5)}% proceed={neu.get('proceed',True)} — "{neu.get('reasoning','')}"
  Conservative: size={con.get('size_pct',2)}% proceed={con.get('proceed',True)} — "{con.get('reasoning','')}"

DECISION RULES:
- If 2+ risk personas say "proceed: false" → you should also reject
- If bull_conf < {self.min_confidence:.2f} → reject
- If bear_risk >= 9 → reject unless aggressive makes compelling case
- Position size = weighted average (aggressive 20%, neutral 50%, conservative 30%)
- You can override any individual agent if their reasoning is weak

Evaluate which arguments are STRONGEST and decide:

Respond ONLY with JSON:
{{"proceed": true/false, "position_size_pct": <number>, "reasoning": "which arguments won and why", "override_notes": "any agent overrides"}}"""

        try:
            raw = self._query_llm_auto(prompt)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                raw = self._extract_json(raw)
                result = json.loads(raw)
            return {
                'proceed': bool(result.get('proceed', True)),
                'size_pct': float(result.get('position_size_pct') or 5),
                'reasoning': str(result.get('reasoning', ''))[:150],
                'override_notes': str(result.get('override_notes', ''))[:80],
            }
        except Exception as e:
            logger.warning(f"[{asset}] Facilitator failed: {e}")
            # SAFE FALLBACK: weighted average of personas, but cap size conservatively
            w_size = (
                agg.get('size_pct', 5) * 0.2 +
                neu.get('size_pct', 5) * 0.5 +
                con.get('size_pct', 2) * 0.3
            )
            votes = sum(1 for p in [agg, neu, con] if p.get('proceed', True))
            return {
                'proceed': votes >= 2,
                'size_pct': min(w_size, 5),  # Cap at 5% when facilitator is broken
                'reasoning': f'facilitator unavailable ({type(e).__name__}) — conservative weighted avg',
                'override_notes': '',
            }
