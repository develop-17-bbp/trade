"""
Trading Executor — EMA(8) Crossover with LLM Confirmation
==========================================================
Robinhood Crypto (real account). LONG (CALL) and SHORT (PUT).
Dynamic trailing stop-loss L1 -> L2 -> L3 -> L4 ...
"""

import os
import re
import sys
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

# ── scipy.stats guard (takes 20+ min to import on Python 3.14 + scipy 1.17) ──
# Modules that depend on scipy.stats (hmmlearn, statsmodels) are skipped unless
# scipy.stats is already cached from a prior import.
_SCIPY_STATS_OK = 'scipy.stats' in sys.modules

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.api.state import DashboardState
from src.data.fetcher import PriceFetcher
from src.data.microstructure import MicrostructureAnalyzer
from src.ai.agentic_strategist import AgenticStrategist
from src.monitoring.journal import TradeJournal
from src.indicators.indicators import (
    ema, atr, rsi, macd, bollinger_bands, stochastic, vwap, obv, adx,
    bb_width, roc, williams_r, chaikin_money_flow, mfi, supertrend,
    volume_delta, choppiness_index
)

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

# Trading Brain v2 — Multi-model consensus + CoT + Memory + Regime + Kelly + Session
try:
    from src.ai.trading_brain import TradingBrainV2
    BRAIN_V2_AVAILABLE = True
except Exception:
    BRAIN_V2_AVAILABLE = False

# Trade Protections — freqtrade-inspired guards, ROI table, partial exits, tagging
try:
    from src.trading.protections import TradeProtections
    PROTECTIONS_AVAILABLE = True
except Exception:
    PROTECTIONS_AVAILABLE = False

# Price Action Analyzer — FVG + Order Blocks (institutional liquidity zones)
try:
    from src.indicators.price_action import PriceActionAnalyzer
    PRICE_ACTION_AVAILABLE = True
except Exception:
    PRICE_ACTION_AVAILABLE = False

# Market Structure — HH/HL/LH/LL + BOS/CHoCH detection
try:
    from src.indicators.market_structure import MarketStructureAnalyzer
    MARKET_STRUCTURE_AVAILABLE = True
except Exception:
    MARKET_STRUCTURE_AVAILABLE = False

# Profit Protector — trade quality rating + loss aversion
try:
    from src.risk.profit_protector import ProfitProtector
    PROFIT_PROTECTOR_AVAILABLE = True
except Exception:
    PROFIT_PROTECTOR_AVAILABLE = False

# VPIN Guard — Volume-synchronous Probability of Informed Trading (adverse selection)
try:
    from src.risk.vpin_guard import VPINGuard
    VPIN_AVAILABLE = True
except Exception:
    VPIN_AVAILABLE = False

# Hurst Exponent — regime detection (trending vs mean-reverting vs random walk)
try:
    from src.models.hurst import HurstExponent
    HURST_AVAILABLE = True
except Exception:
    HURST_AVAILABLE = False

# ML Models — feed predictions to LLM for richer pattern analysis
# HMM: hmmlearn imports scipy.stats internally → 20+ min hang on Py 3.14
if _SCIPY_STATS_OK:
    try:
        from src.models.hmm_regime import HMMRegimeDetector
        HMM_AVAILABLE = True
    except Exception:
        HMM_AVAILABLE = False
else:
    HMM_AVAILABLE = False
    HMMRegimeDetector = None
    print("  [SKIP] HMMRegimeDetector — scipy.stats not available (Py 3.14 hang)")

try:
    from src.models.kalman_filter import KalmanTrendFilter
    KALMAN_AVAILABLE = True
except Exception:
    KALMAN_AVAILABLE = False

try:
    from src.models.volatility import classify_volatility_regime, log_returns
    VOLATILITY_MODEL_AVAILABLE = True
except Exception:
    VOLATILITY_MODEL_AVAILABLE = False

try:
    from src.models.cycle_detector import detect_dominant_cycles
    CYCLE_AVAILABLE = True
except Exception:
    CYCLE_AVAILABLE = False

# Cointegration Engine — BTC-ETH pairs trading spread signal
try:
    from src.models.cointegration import CointegrationEngine
    COINTEGRATION_AVAILABLE = True
except Exception:
    COINTEGRATION_AVAILABLE = False

# LightGBM 3-Class Classifier (pre-filter gate: blocks L1-death trades before LLM call)
try:
    from src.models.lightgbm_classifier import LightGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

# LSTM/GRU/BiLSTM Ensemble (neural net direction prediction)
try:
    from src.models.lstm_ensemble import LSTMEnsemble
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# PatchTST Transformer (patch-based time-series forecaster)
try:
    from src.ai.patchtst_model import PatchTSTClassifier
    PATCHTST_AVAILABLE = True
except Exception:
    PATCHTST_AVAILABLE = False

# Alpha Decay (optimal holding period estimator)
try:
    from src.models.alpha_decay import AlphaDecayModel
    ALPHA_DECAY_AVAILABLE = True
except Exception:
    ALPHA_DECAY_AVAILABLE = False

# RL EMA Strategy Agent (Q-learning optimizer for entry/sizing/SL)
try:
    from src.ai.reinforcement_learning import EMAStrategyRL
    RL_AVAILABLE = True
except Exception:
    RL_AVAILABLE = False

# MetaTrader 5 Bridge — mirror or execute trades on MT5 terminal
try:
    from src.trading.mt5_bridge import MT5Bridge
    MT5_AVAILABLE = True
except Exception:
    MT5_AVAILABLE = False

# Multi-Strategy Engine — regime-aware 4-strategy consensus (replaces EMA-only gatekeeper)
try:
    from src.trading.multi_strategy_engine import MultiStrategyEngine
    MULTI_STRATEGY_AVAILABLE = True
except Exception:
    MULTI_STRATEGY_AVAILABLE = False

# LLM Router — universal multi-provider LLM abstraction (Ollama, Claude, Gemini, etc.)
try:
    from src.ai.llm_provider import LLMRouter, LLMConfig
    LLM_ROUTER_AVAILABLE = True
except Exception:
    LLM_ROUTER_AVAILABLE = False

# Prompt Constraints — safety layer + response validation for LLM outputs
try:
    from src.ai.prompt_constraints import PromptConstraintEngine
    PROMPT_CONSTRAINTS_AVAILABLE = True
except Exception:
    PROMPT_CONSTRAINTS_AVAILABLE = False

# AlertManager — Slack/Telegram/webhook notifications for critical events
try:
    from src.monitoring.alerting import AlertManager
    ALERTING_AVAILABLE = True
except Exception:
    ALERTING_AVAILABLE = False

# Position Sizing — ATR-based dynamic sizing (replaces fixed %)
try:
    from src.risk.position_sizing import atr_position_size, optimal_position_size
    POSITION_SIZING_AVAILABLE = True
except Exception:
    POSITION_SIZING_AVAILABLE = False

# Dynamic Risk Manager — circuit breakers, VaR, kill switch (monitoring only, NOT stop management)
try:
    from src.risk.dynamic_manager import DynamicRiskManager, RiskLimits
    DYNAMIC_RISK_AVAILABLE = True
except Exception:
    DYNAMIC_RISK_AVAILABLE = False

# MetaSizer — Half-Kelly position scaling (risk-calibrated multiplier 0.1-1.0)
try:
    from src.models.meta_sizer import MetaSizer
    META_SIZER_AVAILABLE = True
except Exception:
    META_SIZER_AVAILABLE = False

# VolatilityRegimeDetector — ATR+realized vol regime classification
try:
    from src.models.volatility_regime import VolatilityRegimeDetector, VolatilityRegime
    VOL_REGIME_DETECTOR_AVAILABLE = True
except Exception:
    VOL_REGIME_DETECTOR_AVAILABLE = False

# TradeTrace — structured trade records for memory/audit
try:
    from src.models.trade_trace import TradeTrace
    TRADE_TRACE_AVAILABLE = True
except Exception:
    TRADE_TRACE_AVAILABLE = False

# FFT Cycle Detection — dominant cycle period estimation
try:
    from src.models.cycle import rolling_fft_period
    FFT_CYCLE_AVAILABLE = True
except Exception:
    FFT_CYCLE_AVAILABLE = False

# SystemHealthChecker — background 24/7 health monitoring
try:
    from src.monitoring.health_checker import SystemHealthChecker
    HEALTH_CHECKER_AVAILABLE = True
except Exception:
    HEALTH_CHECKER_AVAILABLE = False

# ── Advanced Learning Engine (runtime meta-optimizer — online training loop) ──
try:
    from src.ai.advanced_learning import (
        AdvancedLearningEngine, MarketAnomalyDetector, MarketRegimeClassifier,
        AlphaDecayTracker, PipelineOverlay,
    )
    ADVANCED_LEARNING_AVAILABLE = True
except Exception:
    ADVANCED_LEARNING_AVAILABLE = False

# ── EVT Tail Risk (Extreme Value Theory — fat-tail VaR for crypto) ──
try:
    from src.risk.evt_risk import EVTRisk
    EVT_RISK_AVAILABLE = True
except Exception:
    EVT_RISK_AVAILABLE = False

# ── Monte Carlo Risk (forward-looking VaR/CVaR simulation) ──
try:
    from src.risk.monte_carlo_risk import MonteCarloRisk
    MC_RISK_AVAILABLE = True
except Exception:
    MC_RISK_AVAILABLE = False

# ── Sentiment Pipeline (rule-based fast + optional FinBERT transformer) ──
try:
    from src.ai.sentiment import SentimentPipeline
    SENTIMENT_AVAILABLE = True
except Exception:
    SENTIMENT_AVAILABLE = False

# ── Temporal Transformer (multi-horizon attention-based forecaster) ──
try:
    from src.ai.temporal_transformer import TemporalTransformer
    TEMPORAL_TRANSFORMER_AVAILABLE = True
except Exception:
    TEMPORAL_TRANSFORMER_AVAILABLE = False

# ── Hawkes Process (self-exciting event clustering — predicts vol spikes) ──
try:
    from src.models.hawkes_process import HawkesProcess
    HAWKES_AVAILABLE = True
except Exception:
    HAWKES_AVAILABLE = False

# ── Market Event Guard (calendar-based trading pause for high-risk events) ──
try:
    from src.monitoring.event_guard import MarketEventGuard
    EVENT_GUARD_AVAILABLE = True
except Exception:
    EVENT_GUARD_AVAILABLE = False

# SL Crash Persistence — survive bot crashes with open positions
try:
    from src.persistence.sl_persistence import SLPersistenceManager
    SL_PERSIST_AVAILABLE = True
except Exception:
    SL_PERSIST_AVAILABLE = False

# API Circuit Breaker — prevent cascade failures on Robinhood
try:
    from src.monitoring.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
    CIRCUIT_BREAKER_AVAILABLE = True
except Exception:
    CIRCUIT_BREAKER_AVAILABLE = False

# Adaptive Feedback Loop — closed-loop learning from every trade
try:
    from src.trading.adaptive_feedback import AdaptiveFeedbackLoop, TradeOutcome
    ADAPTIVE_FEEDBACK_AVAILABLE = True
except Exception:
    ADAPTIVE_FEEDBACK_AVAILABLE = False

# Strategy Universe — 242 auto-generated strategies for consensus
try:
    from src.trading.strategy_universe import StrategyUniverse
    STRATEGY_UNIVERSE_AVAILABLE = True
except Exception:
    STRATEGY_UNIVERSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradingExecutor:
    """EMA(8) crossover strategy with LLM confirmation on Robinhood Crypto."""

    def __init__(self, config: dict):
        self.config = config
        self._paper = None  # Early init — referenced in _run_live before full setup
        self._config_exchange = config.get('exchange', {}).get('name', 'bybit').lower()
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
            or ai_cfg.get('reasoning_model', 'qwen3:32b')
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

        # Quality gates — bear veto handles dangerous trades, so confidence bar can be lower
        _conf_threshold = ai_cfg.get('llm_trade_conf_threshold', 0.70)
        self.min_confidence: float = max(float(_conf_threshold), 0.70)  # Never below 0.70
        self.min_atr_ratio: float = 0.0003
        self.trade_cooldown: float = 600.0       # 10 min between trades (swing: less churn)
        self.post_close_cooldown: float = 1200.0  # 20 min after closing before new entry (swing: wait for setup)
        self.asset_loss_streak: Dict[str, int] = {}    # consecutive losses per asset
        self.asset_cooldown_until: Dict[str, float] = {}  # timestamp when asset can trade again

        # Exchange tag for output (prevents interleaved confusion in multi-exchange mode)
        # MUST be set early — brain, edge stats, journal all use it
        self._ex_tag: str = config.get('exchange', {}).get('name', 'bybit').upper()

        # Exchange
        exchange_name = config.get('exchange', {}).get('name', 'bybit')
        testnet = config.get('mode', 'live') in ('testnet', 'paper')
        self.price_source = PriceFetcher(exchange_name=exchange_name, testnet=testnet)

        # LLM strategist (used as fallback / for deeper analysis)
        provider = ai_cfg.get('reasoning_provider', 'auto')
        model = ai_cfg.get('reasoning_model', 'qwen3:32b')
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

        # ── v8.0: Initialize dynamic intelligence systems ──
        self._economic_intelligence = None
        self._accuracy_engine = None
        self._sharpe_optimizer = None
        self._llm_memory = None
        self._finetune_enricher = None
        try:
            from src.data.economic_intelligence import EconomicIntelligence
            self._economic_intelligence = EconomicIntelligence(config.get('economic_intelligence', {}))
            self._economic_intelligence.start()
            print(f"  [v8.0] EconomicIntelligence: 12 macro layers (background fetch)")
        except Exception as e:
            print(f"  [v8.0] EconomicIntelligence init failed: {e}")
        try:
            from src.optimization.sharpe_optimizer import SharpeOptimizer
            self._sharpe_optimizer = SharpeOptimizer(config.get('sharpe', {}))
            print(f"  [v8.0] SharpeOptimizer: target={self._sharpe_optimizer.target_sharpe}")
        except Exception as e:
            print(f"  [v8.0] SharpeOptimizer init failed: {e}")
        try:
            from src.learning.accuracy_engine import AccuracyEngine
            self._accuracy_engine = AccuracyEngine(config.get('accuracy', {}))
            print(f"  [v8.0] AccuracyEngine: dynamic weights + consistency guardian")
        except Exception as e:
            print(f"  [v8.0] AccuracyEngine init failed: {e}")
        try:
            from src.memory.llm_memory import LLMMemory
            from src.memory.quant_memory import QuantMemory
            self._llm_memory = LLMMemory('mistral_scanner')
            self._quant_memories = {
                'lgbm': QuantMemory('lgbm'),
                'patchtst': QuantMemory('patchtst'),
                'rl': QuantMemory('rl'),
            }
            print(f"  [v8.0] Memory: LLM + 3 quant models (persistent SQLite)")
        except Exception as e:
            self._llm_memory = None
            self._quant_memories = {}
            print(f"  [v8.0] Memory init failed: {e}")
        try:
            from src.learning.finetune_enricher import FinetuneEnricher
            self._finetune_enricher = FinetuneEnricher(
                econ_intel=self._economic_intelligence,
                accuracy_engine=self._accuracy_engine,
                sharpe_optimizer=self._sharpe_optimizer,
                llm_memory=self._llm_memory,
                quant_memories=self._quant_memories if hasattr(self, '_quant_memories') else {},
            )
            print(f"  [v8.0] FinetuneEnricher: macro+memory+accuracy in every LLM prompt")
        except Exception as e:
            print(f"  [v8.0] FinetuneEnricher init failed: {e}")
        self._dynamic_limits = None
        try:
            from src.risk.dynamic_position_limits import DynamicPositionLimits
            self._dynamic_limits = DynamicPositionLimits(
                config=config.get('position_limits', {}),
                accuracy_engine=self._accuracy_engine,
                sharpe_optimizer=self._sharpe_optimizer,
            )
            self._dynamic_limits.update_equity(self.initial_capital)
            print(f"  [v8.0] DynamicPositionLimits: memory-driven sizing + leverage")
        except Exception as e:
            print(f"  [v8.0] DynamicPositionLimits init failed: {e}")

        # Trading Brain v2 — multi-model consensus (mistral + llama3.2), CoT, memory, regime, Kelly, session
        self._brain: Optional[TradingBrainV2] = None
        if BRAIN_V2_AVAILABLE:
            try:
                journal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'trading_journal.jsonl')
                self._brain = TradingBrainV2(
                    ollama_base_url=self.ollama_base_url,
                    journal_path=journal_path,
                    exchange=self._ex_tag.lower(),
                    config=config,
                    economic_intelligence=self._economic_intelligence,
                    llm_memory=self._llm_memory,
                    finetune_enricher=self._finetune_enricher,
                )
                # Wire fine-tuning data collector into brain
                try:
                    from src.ai.training_data_collector import TrainingDataCollector
                    self._training_collector = TrainingDataCollector(
                        spread_cost_pct=self._round_trip_spread,
                    )
                    self._brain._training_collector = self._training_collector
                    print(f"  [AI] Fine-tuning data collector ACTIVE")
                except Exception:
                    self._training_collector = None
                print(f"  [AI] Trading Brain v2 ACTIVE — multi-model + v8.0 intelligence")
            except Exception as e:
                print(f"  [AI] Trading Brain v2 init failed ({e}) — using legacy LLM")
                self._brain = None

        # ── EARLY SPREAD CONFIG (needed by protections below) ──
        # The full spread block runs later at ~line 800+, but ROITable needs
        # this value at construction time so its targets are net-of-spread.
        # Safe to run twice: both reads are idempotent.
        self._spread_per_side = 0.05
        self._round_trip_spread = 0.10
        try:
            for _ex_cfg_early in config.get('exchanges', []):
                if _ex_cfg_early.get('name', '').lower() == self._exchange_name.lower():
                    self._spread_per_side = _ex_cfg_early.get('spread_pct_per_side', 0.05)
                    self._round_trip_spread = _ex_cfg_early.get(
                        'round_trip_spread_pct', self._spread_per_side * 2
                    )
                    break
        except Exception:
            pass

        # ── Trade Protections (freqtrade-inspired) ──
        self._protections = None
        if PROTECTIONS_AVAILABLE:
            try:
                # Round-trip spread (already a percentage, e.g. 1.69 for Robinhood) —
                # critical for ROI table so targets are NET profit. Without this,
                # ROI table closes profitable trades at spread-sized losses on Robinhood.
                _rt_spread_pct = float(getattr(self, '_round_trip_spread', 0.0) or 0.0)
                prot_cfg = {
                    "sl_guard": {"trade_limit": 3, "lookback_minutes": 60, "cooldown_minutes": 30},
                    "max_drawdown": {"max_drawdown_pct": risk.get('max_drawdown_pct', 10.0),
                                     "lookback_minutes": 120, "cooldown_minutes": 60},
                    "pair_lock": {"min_profit_pct": -8.0, "lookback_trades": 8, "lock_hours": 0.5},
                    # ROI targets are NET of spread. On Robinhood (~1.69% round-trip) a 1%
                    # NET target fires when gross ~4.34% — so the exit is truly profitable.
                    "roi_table": {0: 0.10, 30: 0.05, 60: 0.025, 120: 0.01, 240: 0.0},
                    "spread_cost_pct": _rt_spread_pct,
                    "confirm": {"max_spread_pct": 1.0, "max_price_drift_pct": 0.5,
                                "max_concurrent_trades": len(self.assets) * 2},
                    "position_adjust": {
                        "max_dca_entries": 1, "dca_threshold_pct": -3.0,
                        "dca_multiplier": 0.3,
                        "partial_exit_levels": [(4.0, 0.25), (7.0, 0.25), (12.0, 0.25)],
                    },
                }
                self._protections = TradeProtections(prot_cfg)
                print(f"  [PROTECT] Trade Protections ACTIVE — SL guard, drawdown, pair lock, ROI table (spread-aware {_rt_spread_pct:.2f}%), partial exits")
            except Exception as e:
                print(f"  [PROTECT] Init failed ({e}) — proceeding without protections")
                self._protections = None

        # ── Price Action Analyzer (FVG + Order Blocks) ──
        self._price_action = None
        if PRICE_ACTION_AVAILABLE:
            try:
                self._price_action = PriceActionAnalyzer(window=50)
                print(f"  [PA] Price Action Analyzer ACTIVE — FVG + Order Blocks for LLM")
            except Exception as e:
                print(f"  [PA] Price Action init failed ({e})")

        # ── Market Structure Analyzer (BOS/CHoCH) ──
        self._market_structure = None
        if MARKET_STRUCTURE_AVAILABLE:
            try:
                self._market_structure = MarketStructureAnalyzer(window=5)
                print(f"  [MS] Market Structure ACTIVE — HH/HL/LH/LL + BOS/CHoCH for LLM")
            except Exception as e:
                print(f"  [MS] Market Structure init failed ({e})")

        # ── Profit Protector (trade quality + loss aversion) ──
        self._profit_protector = None
        if PROFIT_PROTECTOR_AVAILABLE:
            try:
                self._profit_protector = ProfitProtector(initial_capital=self.initial_capital)
                print(f"  [PP] Profit Protector ACTIVE — trade quality rating for LLM")
            except Exception as e:
                print(f"  [PP] Profit Protector init failed ({e})")

        # ── VPIN Guard (adverse selection / toxic flow detection) ──
        self._vpin_guards: Dict[str, Any] = {}
        if VPIN_AVAILABLE:
            try:
                for asset in self.assets:
                    # BTC: larger bucket (1.0), ETH: smaller (10.0 contracts)
                    bucket = 1.0 if asset == 'BTC' else 10.0
                    self._vpin_guards[asset] = VPINGuard(bucket_size=bucket, window_buckets=50, threshold=0.7)
                print(f"  [VPIN] VPIN Guard ACTIVE — toxic flow detection for {list(self._vpin_guards.keys())}")
            except Exception as e:
                print(f"  [VPIN] VPIN Guard init failed ({e})")

        # ── Hurst Exponent (regime detection: trending vs random vs mean-reverting) ──
        self._hurst = None
        if HURST_AVAILABLE:
            try:
                self._hurst = HurstExponent(min_window=20)
                print(f"  [HURST] Hurst Exponent ACTIVE — regime detection (H>0.55=trend, H<0.45=revert)")
            except Exception as e:
                print(f"  [HURST] Hurst init failed ({e})")

        # ── ML Models (feed predictions to LLM) ──
        self._hmm = None
        if HMM_AVAILABLE:
            try:
                self._hmm = HMMRegimeDetector(n_states=4)
                print(f"  [ML] HMM Regime Detector ACTIVE — 4-state (BULL/BEAR/SIDEWAYS/CRISIS)")
            except Exception as e:
                print(f"  [ML] HMM init failed ({e})")

        self._kalman = None
        if KALMAN_AVAILABLE:
            try:
                self._kalman = KalmanTrendFilter()
                print(f"  [ML] Kalman Trend Filter ACTIVE — adaptive trend + SNR")
            except Exception as e:
                print(f"  [ML] Kalman init failed ({e})")

        # ── LightGBM Binary Classifier (SKIP/TRADE pre-filter gate) ──
        self._lgbm = None
        self._lgbm_raw = {}  # Per-asset raw trained binary models
        self._lgbm_calibration = {}  # Per-asset CalibrationBundle (isotonic + score-delta lookup)
        # Meta-label model (López de Prado style): trained ONLY on rule-signaled bars
        # labeled by actual forward-simulated SL/TP outcome. Consulted AFTER the rule
        # strategy says "enter", as a veto-only gate. Separate from _lgbm_raw so
        # both can coexist and fall back to each other.
        self._lgbm_meta = {}            # Per-asset Booster
        self._lgbm_meta_calibration = {}  # Per-asset CalibrationBundle
        self._lgbm_meta_threshold = {}  # Per-asset TAKE cutoff from thresholds JSON
        if os.environ.get('ACT_DISABLE_ML', '').strip().lower() in ('1', 'true', 'yes', 'on'):
            print(f"  [ML] DISABLED via ACT_DISABLE_ML=1 — LightGBM gate skipped, rule-only entries")
        if LIGHTGBM_AVAILABLE:
            try:
                self._lgbm = LightGBMClassifier(config={
                    'confidence_threshold': 0.55,
                })
                import os as _os
                # Load per-asset BINARY trained models (trained on 30 strategy features)
                _models_root = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'models')
                try:
                    from src.ml import calibration as _calib_mod
                except Exception:
                    _calib_mod = None
                for _asset in self.config.get('assets', ['BTC', 'ETH']):
                    asset_model_path = _os.path.join(_models_root, f'lgbm_{_asset.lower()}_trained.txt')
                    if _os.path.exists(asset_model_path):
                        try:
                            import lightgbm as _lgb
                            self._lgbm_raw[_asset] = _lgb.Booster(model_file=asset_model_path)
                            print(f"  [ML] LightGBM ({_asset}) ACTIVE — binary SKIP/TRADE model loaded")
                        except Exception as e2:
                            print(f"  [ML] LightGBM ({_asset}) raw load failed: {e2}")
                    # Load calibration bundle (optional — falls back to hand-tuned deltas if absent)
                    if _calib_mod is not None:
                        _cal_path = _calib_mod.calibration_path_for(_models_root, _asset)
                        _bundle = _calib_mod.load_calibration(_cal_path)
                        if _bundle is not None:
                            self._lgbm_calibration[_asset] = _bundle
                            print(f"  [ML] LightGBM ({_asset}) calibrated — base_wr={_bundle.baseline_win_rate:.3f} deltas={_bundle.deltas}")

                    # Meta-label model (rule-conditional). Loaded separately — present
                    # only if `src/scripts/train_meta_label.py` has been run. When
                    # present, executor decision path uses it AFTER rule signals fire
                    # as a veto-only gate (can subtract score, never add).
                    _meta_path = _os.path.join(_models_root, f'lgbm_{_asset.lower()}_meta.txt')
                    if _os.path.exists(_meta_path):
                        try:
                            import lightgbm as _lgb
                            self._lgbm_meta[_asset] = _lgb.Booster(model_file=_meta_path)
                            print(f"  [ML] LightGBM ({_asset}) META model loaded (rule-conditional, veto-only)")
                        except Exception as _me:
                            print(f"  [ML] LightGBM ({_asset}) meta load failed: {_me}")
                        # Meta calibration (separate file from the base calibration)
                        if _calib_mod is not None:
                            _meta_cal_path = _os.path.join(_models_root, f'lgbm_{_asset.lower()}_meta_calibration.json')
                            _meta_bundle = _calib_mod.load_calibration(_meta_cal_path)
                            if _meta_bundle is not None:
                                self._lgbm_meta_calibration[_asset] = _meta_bundle
                                print(f"  [ML] META ({_asset}) calibrated — base_wr={_meta_bundle.baseline_win_rate:.3f} deltas={_meta_bundle.deltas}")
                        # Meta threshold (TAKE cutoff) from json
                        _thresh_path = _os.path.join(_models_root, f'lgbm_{_asset.lower()}_meta_thresholds.json')
                        if _os.path.exists(_thresh_path):
                            try:
                                import json as _json
                                with open(_thresh_path, 'r', encoding='utf-8') as _fh:
                                    _tj = _json.load(_fh)
                                self._lgbm_meta_threshold[_asset] = float(_tj.get('take_threshold', 0.5))
                            except Exception:
                                self._lgbm_meta_threshold[_asset] = 0.5
                # Also load generic model for backward compat
                model_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'models', 'lgbm_latest.txt')
                if _os.path.exists(model_path):
                    self._lgbm.load_model(model_path)
                    print(f"  [ML] LightGBM fallback loaded from {model_path}")
                else:
                    print(f"  [ML] LightGBM ACTIVE — rule-based mode (no trained model)")
            except Exception as e:
                print(f"  [ML] LightGBM init failed ({e})")
                self._lgbm = None

        # ── MetaTrader 5 Bridge (skip entirely in paper mode) ──
        self._mt5: Optional[MT5Bridge] = None
        if MT5_AVAILABLE and not self._paper_mode:
            mt5_cfg = config.get('mt5', {})
            if mt5_cfg.get('enabled', False):
                try:
                    self._mt5 = MT5Bridge(config)
                    if self._mt5.connected:
                        print(f"  [MT5] Bridge ACTIVE — mode: {self._mt5.mode.upper()}")
                        self._mt5.print_status()
                    else:
                        print(f"  [MT5] Bridge enabled but not connected — check MT5 terminal")
                        self._mt5 = None
                except Exception as e:
                    print(f"  [MT5] Bridge init failed: {e}")
                    self._mt5 = None
        elif self._paper_mode:
            print(f"  [MT5] Skipped — paper mode (no real orders)")

        # v8.0: Dynamic leverage — learned from memory, not hardcoded 1x
        try:
            if self._exchange_client:
                for asset in self.assets:
                    sym = self._get_symbol(asset)
                    try:
                        if self._dynamic_limits:
                            _lev = self._dynamic_limits.get_optimal_leverage(
                                asset=asset, exchange=self._exchange_name or 'robinhood')
                        else:
                            _lev = 1  # safe default
                        self._exchange_client.exchange.set_leverage(int(_lev), sym)
                        print(f"  [{self._ex_tag}:{asset}] Leverage set to {_lev}x on {self._exchange_name}")
                    except Exception:
                        pass
        except Exception:
            pass

        # State
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity: float = self.initial_capital
        self.cash: float = self.initial_capital
        self.bar_count: int = 0

        # Safe-entries gate (ACT_SAFE_ENTRIES=1). Structural interventions for +EV /
        # Sharpe ≥ 1.0 before readiness-gate soak can complete. See src/trading/safe_entries.py
        # for the full rationale. When disabled, executor behaves identically to pre-change.
        try:
            from src.trading import safe_entries as _safe
            self._safe_enabled = _safe.is_enabled(self.config)
            self._safe_config = _safe.merged_config(self.config)
            self._safe_state = _safe.SafeEntryState.load(_safe.default_state_path())
            if self._safe_enabled:
                print(f"  [SAFE] ACTIVE — risk_pct={self._safe_config['risk_pct']}% "
                      f"min_rr={self._safe_config['min_rr']} "
                      f"stop_atr_mult={self._safe_config['stop_atr_mult']}x "
                      f"consec_halve/pause={self._safe_config['consec_losses_halve']}/"
                      f"{self._safe_config['consec_losses_pause']}")
        except Exception as _se:
            self._safe_enabled = False
            self._safe_config = {}
            self._safe_state = None
            if os.environ.get("ACT_SAFE_ENTRIES"):
                print(f"  [SAFE] init failed: {_se}")

        # Phase 1: observability — Prometheus exporter + OTel tracer. Both are
        # idempotent and guarded by env vars (ACT_METRICS_ENABLED, ACT_TRACING_ENABLED).
        try:
            from src.orchestration import init_tracer, set_equity, start_exporter
            start_exporter()
            init_tracer()
            set_equity("TOTAL", self.equity)
        except Exception as _e:
            # Observability is best-effort — never block bot boot on it.
            print(f"  [OBS] Phase 1 init skipped: {_e}")

        # Phase 3: crash-resume banner + seed checkpoint table. Soft-fail.
        try:
            from src.orchestration.checkpoint import log_startup_diagnostic
            log_startup_diagnostic()
        except Exception as _e:
            print(f"  [CHECKPOINT] startup diag skipped: {_e}")

        # Phase 4: seed GPU-lease metric at 0 so dashboards show a series.
        try:
            from src.orchestration.gpu_scheduler import init as _gpu_init
            _gpu_init()
        except Exception:
            pass

        # Phase 4.5a: register the meta-coordinator as a PeriodicJob and
        # start the supervisor. Env-gated so we can A/B without a deploy.
        if os.getenv("ACT_LEARNING_MESH_ENABLED", "1") == "1":
            try:
                from src.learning.meta_coordinator import register_scheduler_job
                from src.orchestration.scheduler import get_scheduler
                register_scheduler_job(interval_s=float(os.getenv("ACT_META_COORD_INTERVAL_S", "10")))
                get_scheduler().start_all()
                print("  [MESH] MetaCoordinator registered + scheduler started")
            except Exception as _e:
                print(f"  [MESH] Phase 4.5a startup skipped: {_e}")

        # Readiness-gate wrapper: on a live exchange, intercept every
        # new-entry order and block it if the soak gate is closed. Closes
        # (reduce_only=True) always pass — stuck positions must remain
        # exitable. Paper mode skips the wrapper entirely so soak trades
        # can accumulate in the warm store.
        try:
            if (not self._paper_mode
                    and getattr(self, "price_source", None) is not None
                    and hasattr(self.price_source, "place_order")):
                _orig_place_order = self.price_source.place_order

                def _gated_place_order(*args, **kwargs):
                    if kwargs.get("reduce_only"):
                        return _orig_place_order(*args, **kwargs)
                    try:
                        from src.orchestration.readiness_gate import (
                            evaluate as _gate_eval,
                        )
                        state = _gate_eval()
                        if not state.open_:
                            print(
                                "  [GATE] BLOCKED live entry — readiness gate closed "
                                f"({len(state.reasons)} condition(s) failing). "
                                f"Reasons: {state.reasons[:2]}"
                            )
                            return {
                                "status": "blocked",
                                "reason": "readiness_gate_closed",
                                "failing": state.reasons,
                            }
                    except Exception:
                        # Gate evaluation failure must not block trading —
                        # default-allow so a broken metric path never jams
                        # the execution engine.
                        pass
                    return _orig_place_order(*args, **kwargs)

                self.price_source.place_order = _gated_place_order
                print("  [GATE] Live-entry orders are gated by readiness_gate.evaluate()")
        except Exception as _e:
            print(f"  [GATE] live-entry wrapper skipped: {_e}")

        # Pre-flight: data-layer freshness probe. Non-fatal; logs stale layers
        # so the operator sees them before the first decision cycle.
        try:
            from src.data.health import probe_all, summary as _health_summary
            if self._economic_intelligence is not None:
                healths = probe_all(self._economic_intelligence)
                s = _health_summary(healths)
                if s["critical_stale"] > 0:
                    print(f"  [HEALTH] ⚠ {s['critical_stale']} CRITICAL layer(s) stale: "
                          f"{', '.join(s['critical_stale_names'])}")
                else:
                    print(f"  [HEALTH] {s['fresh']}/{s['total']} data layers fresh")
        except Exception as _e:
            print(f"  [HEALTH] pre-flight skipped: {_e}")

        # Readiness gate: log whether the bot would currently be cleared
        # to place real-capital orders. Never blocks boot — the per-order
        # check in the order-placement path is what actually enforces this.
        try:
            from src.orchestration.readiness_gate import evaluate as _gate_eval
            _gate = _gate_eval()
            if _gate.open_:
                print("  [GATE] Readiness gate OPEN — real-capital orders permitted")
            else:
                n = len(_gate.reasons)
                print(f"  [GATE] Readiness gate CLOSED — {n} condition(s) failing "
                      f"(paper-only until soak completes)")
                for _r in _gate.reasons[:4]:
                    print(f"          • {_r}")
        except Exception as _e:
            print(f"  [GATE] readiness evaluation skipped: {_e}")
        self.last_trade_time: Dict[str, float] = {}
        self.last_close_time: Dict[str, float] = {}
        self.last_signal_candle: Dict[str, float] = {}  # Track candle timestamp to avoid re-entry on same candle
        self.failed_close_assets: Dict[str, float] = {}  # Assets that failed to close — skip until manual resolution

        # ── LSTM Ensemble (binary SKIP/TRADE from trailing SL simulation) ──
        # Loads per-asset models in _process_asset; here we just init the dict
        self._lstm = None
        self._lstm_per_asset = {}
        if LSTM_AVAILABLE:
            for _asset in self.config.get('assets', ['BTC', 'ETH']):
                try:
                    model_dir = f'models/lstm_ensemble_{_asset.lower()}'
                    n_features = 50  # 30 strategy + 5 Kalman + 5 EMA inflection + 10 Category B risk/ML
                    _lstm = LSTMEnsemble(input_dim=n_features, seq_len=30,
                                         num_classes=2, model_dir=model_dir)
                    self._lstm_per_asset[_asset] = _lstm
                    print(f"  [ML] LSTM Ensemble ({_asset}) ACTIVE — {n_features} features (30+Kalman+inflection), binary SKIP/TRADE")
                except Exception as e:
                    print(f"  [ML] LSTM Ensemble ({_asset}) init failed ({e})")
            # Default to first asset for backward compat
            if self._lstm_per_asset:
                self._lstm = list(self._lstm_per_asset.values())[0]

        # ── PatchTST Transformer ──
        self._patchtst = None
        if PATCHTST_AVAILABLE:
            try:
                self._patchtst = PatchTSTClassifier()
                if self._patchtst.is_ready:
                    print(f"  [ML] PatchTST Transformer ACTIVE — pre-trained model loaded")
                else:
                    print(f"  [ML] PatchTST initialized (no pre-trained model)")
            except Exception as e:
                print(f"  [ML] PatchTST init failed ({e})")

        # ── ML context cache for cross-agent sharing ──
        self._last_ml_context = {}

        # ── Alpha Decay (optimal hold period) ──
        self._alpha_decay = None
        if ALPHA_DECAY_AVAILABLE:
            try:
                import pickle as _pkl
                alpha_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
                for asset_name in self.assets:
                    ap = os.path.join(alpha_path, f'alpha_decay_{asset_name.lower()}.pkl')
                    if os.path.exists(ap):
                        with open(ap, 'rb') as _f:
                            self._alpha_decay = _pkl.load(_f)
                        print(f"  [ML] Alpha Decay ACTIVE — optimal hold={self._alpha_decay.optimal_hold:.0f} bars ({self._alpha_decay.optimal_hold*5:.0f}min)")
                        break
                if self._alpha_decay is None:
                    self._alpha_decay = AlphaDecayModel()
                    print(f"  [ML] Alpha Decay initialized (default params)")
            except Exception as e:
                print(f"  [ML] Alpha Decay init failed ({e})")

        # ── RL EMA Strategy Agent (Q-learning optimizer) ──
        self._rl_per_asset: Dict[str, Any] = {}
        if RL_AVAILABLE:
            for _asset in self.assets:
                try:
                    rl_path = os.path.join('models', f'rl_ema_{_asset.lower()}.json')
                    if os.path.exists(rl_path):
                        _rl = EMAStrategyRL({'rl_model_path': rl_path})
                        self._rl_per_asset[_asset] = _rl
                        print(f"  [ML] RL Agent ({_asset}) ACTIVE — {len(_rl.q_table)} states, epsilon={_rl.epsilon:.3f}")
                    else:
                        print(f"  [ML] RL Agent ({_asset}): no model at {rl_path}")
                except Exception as e:
                    print(f"  [ML] RL Agent ({_asset}) init failed ({e})")

        # ── Pre-initialize GARCH models (avoid ad-hoc loading per tick) ──
        self._garch_per_asset: Dict[str, Any] = {}
        try:
            import pickle as _pkl
            for _asset in self.assets:
                garch_path = os.path.join('models', f'garch_{_asset.lower()}.pkl')
                if os.path.exists(garch_path):
                    with open(garch_path, 'rb') as _f:
                        self._garch_per_asset[_asset] = _pkl.load(_f)
                    print(f"  [ML] GARCH ({_asset}) ACTIVE — pre-loaded from {garch_path}")
        except Exception as e:
            print(f"  [ML] GARCH pre-load failed ({e})")

        # ── Load fitted HMM from disk if available ──
        if self._hmm:
            try:
                import pickle as _pkl
                hmm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models')
                for asset_name in self.assets:
                    hp = os.path.join(hmm_path, f'hmm_{asset_name.lower()}.pkl')
                    if os.path.exists(hp):
                        with open(hp, 'rb') as _f:
                            loaded_hmm = _pkl.load(_f)
                        if hasattr(loaded_hmm, 'is_fitted') and loaded_hmm.is_fitted:
                            self._hmm = loaded_hmm
                            print(f"  [ML] HMM Regime: loaded TRAINED model from {hp}")
                            break
            except Exception as e:
                logger.debug(f"HMM load error: {e}")

        # LightGBM auto-retrain counter
        self._lgbm_trades_since_retrain: int = 0
        self._lgbm_retrain_interval: int = 50  # Retrain every 50 closed trades

        # Bear/Risk veto agent
        self.bear_enabled: bool = ai_cfg.get('bear_agent_enabled', True)
        self.bear_veto_threshold: int = ai_cfg.get('bear_veto_threshold', 9)
        self.bear_reduce_threshold: int = ai_cfg.get('bear_reduce_threshold', 7)
        self.bear_veto_stats: Dict[str, Dict[str, int]] = {}
        for a in self.assets:
            self.bear_veto_stats[a] = {'vetoed': 0, 'reduced': 0, 'passed': 0}

        # ── EXCHANGE SPREAD CONFIGURATION ──
        # Load spread constants from exchange config (critical for Robinhood profitability)
        self._spread_per_side = 0.05   # Default: Bybit/Delta ~0.05%
        self._round_trip_spread = 0.10
        self._longs_only = False
        # ── Robinhood-specific trading rules ──
        self._rh_min_profit_exit = 0.0       # Minimum profit % before any exit
        self._rh_min_hold_minutes = 0        # Minimum hold time
        self._rh_trailing_lock_pct = 50      # Trail SL at X% of max profit after min_profit
        self._rh_max_hold_days = 7           # Hard exit after N days
        self._rh_trend_only = False          # Disable mean-reversion strategies
        self._rh_tf_alignment_override = False  # Bypass agents when timeframes agree
        self._rh_compound_pct = 0            # Reinvest % of profits

        for _ex_cfg in config.get('exchanges', []):
            if _ex_cfg.get('name', '').lower() == self._exchange_name.lower():
                self._spread_per_side = _ex_cfg.get('spread_pct_per_side', 0.05)
                self._round_trip_spread = _ex_cfg.get('round_trip_spread_pct', self._spread_per_side * 2)
                self._longs_only = _ex_cfg.get('longs_only', False)
                # Robinhood-specific rules
                self._rh_min_profit_exit = _ex_cfg.get('min_profit_before_exit_pct', 0.0)
                self._rh_min_hold_minutes = _ex_cfg.get('min_hold_minutes', 0)
                self._rh_trailing_lock_pct = _ex_cfg.get('trailing_lock_pct', 50)
                self._rh_max_hold_days = _ex_cfg.get('max_hold_days', 7)
                self._rh_trend_only = _ex_cfg.get('trend_only', False)
                self._rh_tf_alignment_override = _ex_cfg.get('tf_alignment_override', False)
                self._rh_compound_pct = _ex_cfg.get('compound_pct', 0)
                break
        if self._round_trip_spread > 1.0:
            print(f"  [SPREAD] HIGH-SPREAD EXCHANGE: {self._exchange_name} | per-side={self._spread_per_side:.2f}% | round-trip={self._round_trip_spread:.2f}%")
            if self._longs_only:
                print(f"  [SPREAD] LONGS-ONLY mode enabled — all SHORT signals will be blocked")
            if self._rh_trend_only:
                print(f"  [SPREAD] TREND-ONLY mode — mean-reversion/grid strategies suppressed")
            if self._rh_min_profit_exit > 0:
                print(f"  [SPREAD] Minimum {self._rh_min_profit_exit}% profit before ANY exit")
            if self._rh_min_hold_minutes > 0:
                print(f"  [SPREAD] Minimum hold: {self._rh_min_hold_minutes} min ({self._rh_min_hold_minutes/60:.0f}h)")
            if self._rh_tf_alignment_override:
                print(f"  [SPREAD] Timeframe alignment override: ENABLED (bypass agents when 1h+4h+1d agree)")

        # ── SNIPER MODE: Patient, high-conviction, capital-protecting trading ──
        # Only enters on multi-confluence setups, compounds profits into bigger positions
        sniper_cfg = config.get('sniper', {})
        self.sniper_enabled: bool = sniper_cfg.get('enabled', self._paper_mode)  # Default ON for Robinhood
        self.sniper_min_confluence: int = sniper_cfg.get('min_confluence', 4)  # Need 4+ signals agreeing
        self.sniper_min_score: int = sniper_cfg.get('min_score', 8)  # Higher bar than normal min_entry_score
        self.sniper_min_expected_move_pct: float = sniper_cfg.get('min_expected_move_pct', 5.0)  # Must expect 5%+ move
        self.sniper_compound_pct: float = sniper_cfg.get('compound_pct', 50.0)  # Reinvest 50% of profits
        self.sniper_protect_principal: bool = sniper_cfg.get('protect_principal', True)  # Never risk original capital after first win
        # Robinhood-hardening intervention A: tiered min-move thresholds. The sniper
        # threshold was 5%+ expected move (correctly rare) — on a 1.69%-spread venue
        # that's the only setup class worth 3x sizing. But waiting ONLY for 5%+ moves
        # means weeks of zero trades in normal volatility. The "normal" tier at
        # 2.5%+ expected move lets the bot take 1x-sized swings in between sniper
        # setups so training/soak data accumulates.
        self.normal_min_expected_move_pct: float = sniper_cfg.get(
            'normal_min_expected_move_pct', 2.5
        )
        self.sniper_stats: Dict[str, int] = {'signals_seen': 0, 'filtered': 0, 'entered': 0, 'wins': 0, 'losses': 0}
        self.sniper_profit_pool: float = 0.0  # Accumulated realized profits for compounding
        self.sniper_principal: float = self.initial_capital  # Protected base capital
        if self.sniper_enabled:
            print(f"  [SNIPER] MODE ACTIVE — min confluence: {self.sniper_min_confluence} | min score: {self.sniper_min_score} | min move: {self.sniper_min_expected_move_pct}%")

        # ── MULTI-STRATEGY ENGINE (replaces EMA-only gatekeeper) ──
        self._multi_engine = None
        if MULTI_STRATEGY_AVAILABLE:
            try:
                self._multi_engine = MultiStrategyEngine(config)
            except Exception as e:
                logger.warning(f"[MULTI-STRATEGY] Init failed: {e}")
        if not self._multi_engine:
            print(f"  [MULTI-STRATEGY] Unavailable — EMA-only mode")
            print(f"  [SNIPER] Compound {self.sniper_compound_pct}% of profits | Protect principal: {self.sniper_protect_principal}")

        # ── SL Crash Persistence — survive restarts with open positions ──
        self._sl_persist = None
        if SL_PERSIST_AVAILABLE:
            self._sl_persist = SLPersistenceManager()
            orphans = self._sl_persist.recover_all()
            if orphans:
                for asset, pos_data in orphans.items():
                    self.positions[asset] = pos_data
                print(f"  [SL-PERSIST] Recovered {len(orphans)} position(s) from crash")
            else:
                print(f"  [SL-PERSIST] No orphaned positions — clean start")

        # ── API Circuit Breaker — prevent Robinhood cascade failures ──
        self._circuit_breaker = None
        if CIRCUIT_BREAKER_AVAILABLE:
            alert_fn = self._send_alert if hasattr(self, '_send_alert') else None
            self._circuit_breaker = CircuitBreaker(
                failure_threshold=3,
                window_seconds=300,
                recovery_timeout=1800,
                alert_fn=alert_fn,
                name='Robinhood',
            )

        # ── Adaptive Feedback Loop — every trade outcome teaches the system ──
        self._adaptive = None
        if ADAPTIVE_FEEDBACK_AVAILABLE:
            self._adaptive = AdaptiveFeedbackLoop(config)

        # ── Strategy Universe — 242 auto-generated strategies for consensus ──
        self._universe = None
        if STRATEGY_UNIVERSE_AVAILABLE:
            self._universe = StrategyUniverse()
            print(f"  [UNIVERSE] {self._universe.total_strategies} strategies loaded for consensus voting")

        # ── Genetic Strategy Engine — self-evolving strategy discovery ──
        self._genetic_engine = None
        self._genetic_hall_of_fame = []
        self._genetic_last_reload = 0
        self._genetic_hof_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs/genetic_evolution_results.json')
        try:
            from src.trading.genetic_strategy_engine import GeneticStrategyEngine, backtest_dna
            self._genetic_engine = GeneticStrategyEngine(spread_pct=self._round_trip_spread * 100 if self._round_trip_spread else 1.69)
            self._reload_genetic_hall_of_fame()
        except Exception as e:
            logger.debug(f"GeneticStrategyEngine init failed: {e}")

    def _reload_genetic_hall_of_fame(self):
        """Load or refresh genetic hall-of-fame from results file.

        NOTE: This method also initializes many subsystems (cointegration,
        paper tracker, meta controller, etc.) because they were originally
        part of __init__ and share the same scope.  We reconstruct the
        local variables from self.config so they are available here.
        """
        # Reconstruct __init__ locals that the subsystem init code references
        config = self.config
        adaptive = config.get('adaptive', {})
        risk = config.get('risk', {})
        ai_cfg = config.get('ai', {})
        try:
            import json as _json
            if os.path.exists(self._genetic_hof_path):
                with open(self._genetic_hof_path) as _f:
                    _hof = _json.load(_f)
                self._genetic_hall_of_fame = _hof.get('hall_of_fame', [])
                print(f"  [GENETIC] Engine ACTIVE — {len(self._genetic_hall_of_fame)} evolved strategies from hall of fame")
            else:
                print(f"  [GENETIC] Engine ACTIVE — no prior evolution results (run genetic evolution to populate)")
            self._genetic_last_reload = time.time()
        except Exception as e:
            logger.debug(f"Genetic hall-of-fame reload failed: {e}")

        # ── Cointegration Engine — BTC-ETH pairs spread trading signal ──
        self._coint_engine = None
        if COINTEGRATION_AVAILABLE:
            try:
                self._coint_engine = CointegrationEngine(entry_z=2.0, exit_z=0.5, lookback=200)
                print(f"  [PAIRS] Cointegration engine ACTIVE — BTC-ETH spread signal enabled")
            except Exception as e:
                logger.warning(f"[PAIRS] CointegrationEngine init failed: {e}")
        self._last_pairs_signal: Dict[str, Any] = {}

        # ── Per-timeframe performance tracking (LLM learns which TFs profit) ──
        # Configurable via adaptive.signal_timeframes (default: all TFs)
        _cfg_tfs = adaptive.get('signal_timeframes', ['1m', '5m', '15m', '1h', '4h'])
        self.SIGNAL_TIMEFRAMES = [str(tf) for tf in _cfg_tfs]
        self.TF_FETCH_LIMITS = {'1m': 100, '5m': 100, '15m': 60, '1h': 50, '4h': 30, '1d': 20}
        # Scale ratchet thresholds by timeframe (higher TF = bigger moves = wider %)
        # Ratchet scale: how many % PnL to trigger breakeven/profit lock per TF
        # Old: 4h=5.0, 1d=10.0 → breakeven needed +5%/+10% = too late, spread already ate profit
        # New: 4h=3.0, 1d=5.0 → breakeven at +3%/+5% = just above spread cost (1.69%)
        self.TF_RATCHET_SCALE = {'1m': 0.5, '5m': 1.0, '15m': 1.5, '1h': 2.5, '4h': 3.0, '1d': 5.0}
        # Min/Max SL distance as % of price, per timeframe
        self.TF_SL_MIN_PCT = {'1m': 0.003, '5m': 0.005, '15m': 0.008, '1h': 0.012, '4h': 0.02, '1d': 0.03}
        self.TF_SL_MAX_PCT = {'1m': 0.01, '5m': 0.02, '15m': 0.03, '1h': 0.05, '4h': 0.08, '1d': 0.12}
        # Widen SL/TP/ratchet bounds for high-spread exchanges (Robinhood ~1.69% round-trip)
        if self._round_trip_spread > 1.0:
            self.TF_SL_MIN_PCT = {k: max(v, 0.04) for k, v in self.TF_SL_MIN_PCT.items()}
            self.TF_SL_MAX_PCT = {k: max(v, 0.15) for k, v in self.TF_SL_MAX_PCT.items()}
            self.TF_RATCHET_SCALE = {k: max(v * 2.5, 5.0) for k, v in self.TF_RATCHET_SCALE.items()}
            print(f"  [SPREAD] SL/ratchet widened: min_SL=4%+, max_SL=15%+, ratchet=2.5x scaled")
        self.tf_performance: Dict[str, Dict[str, Any]] = {}
        for tf in self.SIGNAL_TIMEFRAMES:
            self.tf_performance[tf] = {'wins': 0, 'losses': 0, 'total_pnl': 0.0}
        # Cache HTF candles (1h/4h only refresh every few minutes, not every 10s poll)
        self._tf_cache: Dict[str, Dict] = {}   # key = f"{asset}_{tf}" -> {'ohlcv': ..., 'ts': time.time()}
        self._tf_cache_ttl = {'1m': 30, '5m': 60, '15m': 120, '1h': 300, '4h': 600, '1d': 3600}

        # ── Portfolio-level drawdown limit (Freqtrade pattern) ──
        # Halt ALL trading if cumulative realized losses exceed threshold
        self.max_drawdown_pct: float = risk.get('max_drawdown_pct', 10.0)
        self.daily_loss_limit_pct: float = risk.get('daily_loss_limit_pct', 3.0)
        self.session_start_equity: float = self.initial_capital
        self.session_realized_pnl: float = 0.0
        self._trade_log: List[Dict] = []  # Complete trade history for session summary
        self._last_tick_prices: Dict[str, float] = {}  # Latest price per asset for paper updates
        self.daily_realized_pnl: float = 0.0
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

        # ── LLM Router (universal multi-provider abstraction) ──
        self._llm_router = None
        if LLM_ROUTER_AVAILABLE:
            try:
                self._llm_router = LLMRouter()
                # Register Ollama as primary provider
                self._llm_router.add_provider('ollama', LLMConfig(
                    provider='ollama',
                    base_url=self.ollama_base_url,
                    model=self.ollama_model,
                    temperature=0.3,
                    max_tokens=1024,
                    timeout=60,
                ))
                # Register Claude as fallback if available
                if self._claude_api_key:
                    self._llm_router.add_provider('claude', LLMConfig(
                        provider='anthropic',
                        api_key=self._claude_api_key,
                        model=self._claude_model,
                        temperature=0.3,
                        max_tokens=1024,
                    ))
                # Auto-detect any other providers from env vars
                self._llm_router.add_from_env()
                providers = self._llm_router.list_providers()
                print(f"  [LLM] Router ACTIVE — providers: {providers}")
            except Exception as e:
                print(f"  [LLM] Router init failed ({e}) — using legacy direct calls")
                self._llm_router = None

        # ── Prompt Constraint Engine (safety + response validation) ──
        self._prompt_constraints = None
        if PROMPT_CONSTRAINTS_AVAILABLE:
            try:
                self._prompt_constraints = PromptConstraintEngine()
                print(f"  [SAFETY] Prompt Constraints ACTIVE — LLM response validation enabled")
            except Exception as e:
                print(f"  [SAFETY] Prompt Constraints init failed ({e})")

        # ── Alert Manager (Slack/Telegram/webhook notifications) ──
        self._alert_manager = None
        if ALERTING_AVAILABLE:
            try:
                alert_cfg = config.get('alerts', {})
                self._alert_manager = AlertManager(alert_cfg)
                print(f"  [ALERT] AlertManager ACTIVE — notifications enabled")
            except Exception as e:
                print(f"  [ALERT] AlertManager init failed ({e})")

        # ── MetaSizer (Half-Kelly position scaling) ──
        self._meta_sizer = None
        if META_SIZER_AVAILABLE:
            try:
                self._meta_sizer = MetaSizer()
                print(f"  [SIZING] MetaSizer ACTIVE — Half-Kelly position scaling (0.1x-1.0x)")
            except Exception as e:
                print(f"  [SIZING] MetaSizer init failed ({e})")

        # ── VolatilityRegimeDetector (ATR + realized vol classification) ──
        self._vol_regime_detector = None
        if VOL_REGIME_DETECTOR_AVAILABLE:
            try:
                self._vol_regime_detector = VolatilityRegimeDetector(lookback=100)
                print(f"  [VOL] VolatilityRegimeDetector ACTIVE — LOW_VOL_RANGE/TREND_EXPANSION/HIGH_VOL_PANIC/NORMAL")
            except Exception as e:
                print(f"  [VOL] VolatilityRegimeDetector init failed ({e})")

        # ── SystemHealthChecker (background monitoring) ──
        self._health_checker = None
        if HEALTH_CHECKER_AVAILABLE:
            try:
                self._health_checker = SystemHealthChecker(check_interval_sec=120)
                # Register critical components
                self._health_checker.register_component('price_source',
                    lambda: self.price_source is not None and self.price_source.exchange is not None)
                self._health_checker.register_component('equity_positive',
                    lambda: self.equity > 0)
                self._health_checker.register_component('not_halted',
                    lambda: not self.trading_halted)
                self._health_checker.start()
                print(f"  [HEALTH] SystemHealthChecker ACTIVE — monitoring {len(self._health_checker.components)} components every 120s")
            except Exception as e:
                print(f"  [HEALTH] SystemHealthChecker init failed ({e})")

        # ── Dynamic Risk Manager (circuit breakers + VaR — monitoring only) ──
        self._dynamic_risk = None
        if DYNAMIC_RISK_AVAILABLE:
            try:
                self._dynamic_risk = DynamicRiskManager(initial_capital=self.initial_capital)
                # Align risk limits with executor's config
                self._dynamic_risk.risk_limits.max_daily_loss_pct = self.daily_loss_limit_pct / 100.0
                self._dynamic_risk.risk_limits.max_drawdown_limit_pct = self.max_drawdown_pct / 100.0
                self._dynamic_risk.risk_limits.kill_switch_daily_loss_pct = (self.daily_loss_limit_pct * 1.5) / 100.0
                # Match executor's actual sizing: 5% for normal, 20% for small accounts
                max_trade_pct = 0.20 if self.initial_capital < 500 else 0.05
                self._dynamic_risk.risk_limits.max_single_trade_pct = max_trade_pct
                self._dynamic_risk.risk_limits.max_portfolio_heat_pct = max_trade_pct * len(self.assets)
                print(f"  [RISK] DynamicRiskManager ACTIVE — 7 circuit breakers, VaR/ES, kill switch")
            except Exception as e:
                print(f"  [RISK] DynamicRiskManager init failed ({e})")

        # ── Advanced Learning Engine (runtime meta-optimizer + online training) ──
        self._advanced_learning = None
        if ADVANCED_LEARNING_AVAILABLE:
            try:
                meta_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'meta_learning_model.json')
                self._advanced_learning = AdvancedLearningEngine(meta_model_path=meta_path)
                print(f"  [META] AdvancedLearningEngine ACTIVE — online training (anomaly + regime + overlay + alpha decay)")
            except Exception as e:
                print(f"  [META] AdvancedLearningEngine init failed ({e})")

        # ── EVT Tail Risk (fat-tail VaR) ──
        self._evt_risk = None
        if EVT_RISK_AVAILABLE:
            try:
                self._evt_risk = EVTRisk(threshold_quantile=0.90, var_level=0.99)
                print(f"  [RISK] EVT Tail Risk ACTIVE — GPD-based fat-tail VaR (99%)")
            except Exception as e:
                print(f"  [RISK] EVT init failed ({e})")

        # ── Monte Carlo Risk (forward VaR/CVaR simulation) ──
        self._mc_risk = None
        if MC_RISK_AVAILABLE:
            try:
                self._mc_risk = MonteCarloRisk(n_simulations=5000, horizon=24, var_confidence=0.95)
                print(f"  [RISK] Monte Carlo Risk ACTIVE — 5K sims, 24-bar horizon, VaR/CVaR")
            except Exception as e:
                print(f"  [RISK] Monte Carlo init failed ({e})")

        # ── Sentiment Pipeline (rule-based fast tier + optional FinBERT) ──
        self._sentiment = None
        if SENTIMENT_AVAILABLE:
            try:
                _sent_model = 'rule-based'
                _use_transformer = False
                if config:
                    _sent_cfg = config.get('sentiment', {})
                    _sent_model = _sent_cfg.get('model', 'rule-based')
                    _use_transformer = _sent_cfg.get('use_transformer', False)
                    if 'finbert' in str(_sent_model).lower():
                        _use_transformer = True
                if _use_transformer and _sent_model != 'rule-based':
                    self._sentiment = SentimentPipeline(
                        sentiment_model=_sent_model,
                        use_transformer=True,
                        device=config.get('sentiment', {}).get('device', 'cpu') if config else 'cpu',
                    )
                    print(f"  [SENT] SentimentPipeline ACTIVE — FinBERT transformer ({_sent_model})")
                else:
                    self._sentiment = SentimentPipeline()
                    print(f"  [SENT] SentimentPipeline ACTIVE — rule-based (set sentiment.model=finbert to enable transformer)")
            except Exception as e:
                try:
                    self._sentiment = SentimentPipeline()
                    print(f"  [SENT] SentimentPipeline ACTIVE — rule-based fallback (transformer init failed: {e})")
                except Exception as e2:
                    print(f"  [SENT] SentimentPipeline init failed ({e2})")

        # ── RSS News Aggregator (replaces dead CryptoPanic API) ──
        self._news_rss = None
        try:
            from src.ai.rss_news_aggregator import RSSNewsAggregator
            self._news_rss = RSSNewsAggregator(
                fetch_interval=int((config or {}).get('news', {}).get('fetch_interval_sec', 300)),
                max_age_hours=float((config or {}).get('news', {}).get('max_age_hours', 48.0)),
            )
            # Warm the cache in background so first trade decision has data
            import threading as _th
            _th.Thread(
                target=lambda: self._news_rss.get_headlines(asset='BTC', force_refresh=True),
                daemon=True,
            ).start()
            print(f"  [NEWS] RSSNewsAggregator ACTIVE — 9 sources, warming cache")
        except Exception as e:
            print(f"  [NEWS] RSSNewsAggregator init failed: {e}")

        # ── Temporal Transformer (attention-based multi-horizon forecaster) ──
        self._temporal_transformer = None
        if TEMPORAL_TRANSFORMER_AVAILABLE:
            try:
                self._temporal_transformer = TemporalTransformer(d_model=64, n_heads=4, context_len=120)
                print(f"  [ML] TemporalTransformer ACTIVE — attention-based forecaster")
            except Exception as e:
                print(f"  [ML] TemporalTransformer init failed ({e})")

        # ── Hawkes Process (event clustering intensity) ──
        self._hawkes = None
        if HAWKES_AVAILABLE:
            try:
                self._hawkes = HawkesProcess(mu=0.1, alpha=0.5, beta=1.0)
                print(f"  [ML] HawkesProcess ACTIVE — self-exciting event clustering")
            except Exception as e:
                print(f"  [ML] HawkesProcess init failed ({e})")

        # ── Market Event Guard (calendar-based risk pause) ──
        self._event_guard = None
        if EVENT_GUARD_AVAILABLE:
            try:
                self._event_guard = MarketEventGuard()
                print(f"  [GUARD] MarketEventGuard ACTIVE — calendar-based high-risk pause")
            except Exception as e:
                print(f"  [GUARD] MarketEventGuard init failed ({e})")

        # ── Meta Controller — ML model arbitration (LGB + RL + PatchTST) ──
        self._meta_controller = None
        try:
            from src.trading.meta_controller import MetaController
            self._meta_controller = MetaController(config)
            print(f"  [META-CTRL] Meta Controller ACTIVE — LGB+RL+PatchTST arbitration")
        except Exception as e:
            logger.debug(f"MetaController init failed: {e}")

        # ── Signal Combiner — formal L1+L2+L3 fusion with VETO ──
        self._signal_combiner = None
        try:
            from src.trading.signal_combiner import SignalCombiner
            self._signal_combiner = SignalCombiner(config)
            print(f"  [COMBINER] Signal Combiner ACTIVE — L1(50%)+L2(30%)+L3(20%) fusion")
        except Exception as e:
            logger.debug(f"SignalCombiner init failed: {e}")

        # ── Portfolio Allocator — cross-asset Kelly sizing ──
        self._allocator = None
        try:
            from src.portfolio.allocator import PortfolioAllocator
            _alloc_capital = config.get('portfolio', {}).get('total_capital', 100_000.0) if config else 100_000.0
            _alloc_max = config.get('portfolio', {}).get('max_allocation_pct', 0.05) if config else 0.05
            self._allocator = PortfolioAllocator(total_capital=_alloc_capital, max_allocation_pct=_alloc_max)
            print(f"  [ALLOC] Portfolio Allocator ACTIVE — Kelly + risk-parity sizing (${_alloc_capital:,.0f})")
        except Exception as e:
            logger.debug(f"PortfolioAllocator init failed: {e}")

        # ── Adaptive Engine — regime-based strategy selection with learning ──
        # ── Self-Evolving Overlay — makes ALL static parts adaptive ──
        self._evolution_overlay = None
        try:
            from src.trading.self_evolving_overlay import SelfEvolvingOverlay
            self._evolution_overlay = SelfEvolvingOverlay()
            _ov = self._evolution_overlay.get_overrides()
            print(f"  [EVOLVE] Self-Evolving Overlay ACTIVE — risk/agents/LLM/indicators all adapting")
            print(f"  [EVOLVE] Evolved params: EMA={_ov['indicator_params'].get('ema_fast',8)}/{_ov['indicator_params'].get('ema_slow',21)} RSI={_ov['indicator_params'].get('rsi_period',14)}")
        except Exception as e:
            logger.debug(f"SelfEvolvingOverlay init failed: {e}")

        # ── Drift Detector — monitor feature distributions for model staleness ──
        self._drift_detector = None
        try:
            from src.monitoring.drift_detector import DriftDetector
            self._drift_detector = DriftDetector(window_size=500)
            print(f"  [DRIFT] Feature drift detector ACTIVE — PSI monitoring for ML model staleness")
        except Exception as e:
            logger.debug(f"DriftDetector init failed: {e}")

        self._adaptive_engine = None
        try:
            from src.trading.adaptive_engine import AdaptiveEngine
            self._adaptive_engine = AdaptiveEngine(config.get('adaptive', {}))
            print(f"  [ADAPT-ENGINE] Adaptive Engine ACTIVE — 5 strategies with performance learning")
        except Exception as e:
            logger.debug(f"AdaptiveEngine init failed: {e}")

        # ── Polymarket failure tracking ──
        self._polymarket_consecutive_failures = 0
        self._polymarket_disabled = False

        # ── Robinhood Paper Trading Tracker ──
        self._paper = None
        try:
            from src.data.robinhood_fetcher import RobinhoodPaperFetcher
            self._paper = RobinhoodPaperFetcher(config)
            self._paper.load_state()
            if self._paper.connected:
                print(f"  [PAPER] Robinhood Paper Tracker ACTIVE — logging signals vs real prices")
            else:
                print(f"  [PAPER] Paper Tracker loaded (no Robinhood — using CCXT prices)")
        except Exception as e:
            print(f"  [PAPER] Paper Tracker init failed ({e})")
            self._paper = None

        # ── Online training state: last full analysis timestamp ──
        self._last_full_meta_analysis = 0  # timestamp of last full cross-asset analysis
        self._meta_analysis_interval = 1800  # every 30 minutes

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

        # Phase 0: ULID per cycle. Phase 1: enter OTel span + Prometheus timer.
        # We use manual __enter__/__exit__ (not a `with`) to avoid a 200-line
        # re-indent of the existing method body. Every observability call is a
        # no-op when deps aren't installed or ACT_{METRICS,TRACING}_ENABLED=0.
        import time as _time
        from src.orchestration import (
            decision_span,
            new_decision_id,
            record_agent_vote,
            record_authority_violation,
            record_decision,
        )
        decision_id = new_decision_id()
        _span_cm = decision_span(decision_id=decision_id, symbol=asset)
        _span_ctx = _span_cm.__enter__()
        trace_id = _span_ctx.get("trace_id", decision_id)
        _cycle_start = _time.perf_counter()
        _cycle_action = "FLAT"
        _cycle_consensus = "UNKNOWN"

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

            # ── Build sentiment_data from real RSS headlines (not price proxy) ──
            sentiment_data = {}
            try:
                if self._news_rss is not None and self._sentiment is not None:
                    headline_objs = self._news_rss.get_headlines(asset=asset, limit=20)
                    headlines = [h.text for h in headline_objs]
                    timestamps = [h.timestamp for h in headline_objs]
                    if headlines:
                        scored = self._sentiment.analyze(headlines, timestamps=timestamps)
                        agg = self._sentiment.aggregate_sentiment(scored, timestamps=timestamps)
                        if isinstance(agg, dict):
                            sentiment_data = dict(agg)
                            # Map fields the sentiment_decoder_agent reads
                            sentiment_data['score'] = agg.get('aggregate_score', 0.0)
                            sentiment_data['label'] = agg.get('aggregate_label', 'NEUTRAL')
                        sentiment_data['headline_count'] = len(headlines)
                        sentiment_data['recent_headlines'] = headlines[:5]
                        sentiment_data['sources'] = list({h.source for h in headline_objs})
            except Exception as e:
                logger.debug(f"[SENT] news->sentiment aggregation failed: {e}")

            # ── Build ext_feats from economic intelligence (real data) ──
            ext_feats = {}
            try:
                if self._economic_intelligence is not None:
                    # Fear & Greed from social_sentiment layer
                    social = self._economic_intelligence._layers.get('social_sentiment')
                    if social is not None and hasattr(social, 'get_cached'):
                        fng = social.get_cached() or {}
                        if isinstance(fng, dict) and 'value' in fng:
                            ext_feats['fear_greed_index'] = float(fng.get('value', 50))
                            ext_feats['fear_greed_signal'] = fng.get('signal', 'NEUTRAL')
                    # Derivatives layer — funding rate + OI live from Bybit/Binance
                    deriv = self._economic_intelligence._layers.get('derivatives')
                    if deriv is not None and hasattr(deriv, 'get_cached'):
                        d = deriv.get_cached() or {}
                        if isinstance(d, dict) and not d.get('stale', True):
                            # Agent reads 'funding_rate' (fractional form)
                            ext_feats['funding_rate'] = float(d.get('funding_rate', 0.0) or 0.0)
                            # OI in USD notional for regime reasoning
                            ext_feats['open_interest'] = float(d.get('open_interest_usd', 0.0) or 0.0)
                            ext_feats['put_call_ratio'] = float(d.get('put_call_ratio', 1.0) or 1.0)
            except Exception as e:
                logger.debug(f"[ECON] ext_feats build failed: {e}")

            # ── Economic macro snapshot for agent context ──
            economic_data = {}
            try:
                if self._economic_intelligence is not None:
                    economic_data = self._economic_intelligence.get_macro_summary() or {}
            except Exception as e:
                logger.debug(f"[ECON] macro summary failed: {e}")

            # Run full orchestrator pipeline (agents + debate + combine + audit)
            decision = self._orchestrator.run_cycle(
                quant_state=quant_state,
                raw_signal=raw_signal,
                raw_confidence=0.5,
                ext_feats=ext_feats,
                on_chain=on_chain_data,
                sentiment_data=sentiment_data,
                ohlcv_data=ohlcv_ctx,
                asset=asset,
                daily_pnl=self.daily_realized_pnl,
                account_balance=self.equity,
                open_positions=list(self.positions.values()),
                trade_history=[],
                economic_data=economic_data,
            )

            # ── Push live intelligence snapshot to DashboardState + audit log ──
            # Each side is wrapped INDEPENDENTLY so a failure in live-intelligence
            # push (dashboard state, non-critical) doesn't mask a failure in the
            # audit-log write (Phase 0 soak gate depends on this).
            try:
                self._publish_live_intelligence(
                    asset=asset,
                    sentiment_data=sentiment_data,
                    ext_feats=ext_feats,
                    economic_data=economic_data,
                )
            except Exception as e:
                logger.warning(
                    f"[LIVE_INTEL] publish failed for {asset}: {type(e).__name__}: {e}",
                    exc_info=True,
                )

            try:
                self._log_decision_audit(
                    asset=asset,
                    raw_signal=raw_signal,
                    decision=decision,
                    sentiment_data=sentiment_data,
                    ext_feats=ext_feats,
                    economic_data=economic_data,
                    decision_id=decision_id,
                    trace_id=trace_id,
                )
            except Exception as e:
                # Phase 0 gate depends on this log — surface loudly so we find
                # the failing code path instead of silently dropping audit rows.
                logger.warning(
                    f"[AUDIT] decision-log write failed for {asset} "
                    f"decision_id={decision_id}: {type(e).__name__}: {e}",
                    exc_info=True,
                )

            # Phase 2: publish decision envelope to the decision.cycle stream
            # for downstream learners. Soft-fail: a dead Redis returns None
            # and logs at debug level — decision path continues.
            _decision_row = {
                "decision_id": decision_id,
                "trace_id": trace_id,
                "symbol": asset,
                "raw_signal": int(raw_signal or 0),
                "direction": int(getattr(decision, "direction", 0) or 0),
                "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
                "consensus": getattr(decision, "consensus_level", "UNKNOWN") or "UNKNOWN",
                "veto": bool(getattr(decision, "veto", False)),
                "ts_ns": _time.time_ns(),
            }
            try:
                from src.orchestration import STREAM_DECISION_CYCLE, stream_publish
                stream_publish(STREAM_DECISION_CYCLE, _decision_row)
            except Exception:
                pass

            # Phase 3: durable warm-tier write + crash-resume checkpoint.
            try:
                from src.orchestration.warm_store import get_store
                get_store().write_decision(_decision_row)
            except Exception as _e:
                logger.debug(f"[WARM] decision write failed: {_e}")
            try:
                from src.orchestration.checkpoint import checkpoint_cycle
                checkpoint_cycle(
                    decision_id=decision_id,
                    trace_id=trace_id,
                    symbol=asset,
                    open_positions={k: {
                        'direction': v.get('direction'),
                        'entry_price': v.get('entry_price'),
                        'size': v.get('size'),
                    } for k, v in self.positions.items()},
                    equity=self.equity,
                )
            except Exception as _e:
                logger.debug(f"[CHECKPOINT] write failed: {_e}")

            # ── Pass RAW agent outputs to LLM — LLM is the brain, not Python ──
            # No strategy filter, no Python voting — LLM sees everything and decides
            votes = decision.agent_votes if decision else {}
            agent_summary = []
            for agent_name, vote in votes.items():
                dir_str = "LONG" if vote.direction > 0 else "SHORT" if vote.direction < 0 else "FLAT"
                veto_str = " [VETO]" if vote.veto else ""
                agent_summary.append(
                    f"  {agent_name}: {dir_str} conf={vote.confidence:.0%} scale={vote.position_scale:.0%}{veto_str} | {vote.reasoning[:80]}"
                )
                record_agent_vote(agent=agent_name, direction=int(vote.direction or 0))

            # Phase 1: emit authority violations (risk_params.violations) as counters.
            try:
                _viol = getattr(decision, "risk_params", None) or {}
                for _rule in (_viol.get("violations", []) if isinstance(_viol, dict) else []):
                    record_authority_violation(rule=str(_rule))
            except Exception:
                pass

            # Basic consensus for logging
            directions = [v.direction * v.confidence for v in votes.values()]
            net = sum(directions) / len(directions) if directions else 0
            avg_conf = sum(v.confidence for v in votes.values()) / len(votes) if votes else 0.5

            consensus_dir = "CALL" if net > 0.15 else "PUT" if net < -0.15 else "FLAT"
            consensus = "STRONG" if abs(net) > 0.6 else "MODERATE" if abs(net) > 0.3 else "WEAK"

            # Phase 1: expose final action + consensus on the span and queue
            # them for the latency histogram emitted in the finally block.
            _cycle_action = "LONG" if net > 0 else "SHORT" if net < 0 else "FLAT"
            _cycle_consensus = consensus
            _span_ctx["final_action"] = _cycle_action
            _span_ctx["consensus"] = consensus

            print(f"  [{self._ex_tag}:{asset}] AGENTS RAW: {consensus_dir} {consensus} (net={net:+.2f}) | {len(votes)} agents | avg_conf={avg_conf:.2f}")

            return {
                'consensus': consensus,
                'consensus_dir': consensus_dir,
                'direction': 1 if net > 0 else (-1 if net < 0 else 0),
                'confidence': avg_conf,
                'position_scale': decision.position_scale if decision else 0.5,
                'agent_count': len(votes),
                'agent_summary': agent_summary,
                'debate_summary': decision.strategy_recommendation[:100] if decision and decision.strategy_recommendation else 'N/A',
                'data_quality': decision.data_quality if decision else 1.0,
                'veto': False,  # No Python veto — LLM decides
                'agent_votes': {k: {'dir': v.direction, 'conf': v.confidence, 'reasoning': v.reasoning[:100]} for k, v in votes.items()},
                'trend_reach_score': 0,
                'safety_score': 0,
                'profit_lock_score': 0,
                'l_prediction': '?',
            }

        except Exception as e:
            _span_ctx["error"] = f"{type(e).__name__}: {e}"
            logger.warning(f"[{asset}] Orchestrator failed (degraded): {e}")
            return None
        finally:
            # Phase 1: record latency + equity + close the span exactly once,
            # on every exit path. Exceptions here must never mask the original.
            try:
                record_decision(
                    symbol=asset,
                    action=_cycle_action,
                    consensus=_cycle_consensus,
                    latency_s=_time.perf_counter() - _cycle_start,
                )
            except Exception:
                pass
            try:
                from src.orchestration import set_equity as _set_equity
                _set_equity("TOTAL", float(self.equity))
            except Exception:
                pass
            try:
                _span_cm.__exit__(None, None, None)
            except Exception:
                pass

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
    def _publish_live_intelligence(self, asset, sentiment_data, ext_feats, economic_data):
        """Push the live sentiment/macro snapshot to DashboardState.

        Makes the data the agents just saw visible on the dashboard so
        operators can verify real feeds are flowing (not stuck at defaults).
        """
        try:
            from src.api.state import DashboardState
        except Exception:
            return
        sd = sentiment_data or {}
        ef = ext_feats or {}
        ec = economic_data or {}
        from datetime import datetime as _dt
        snapshot = {
            'sentiment': {
                'score': float(sd.get('score', sd.get('aggregate_score', 0.0)) or 0.0),
                'label': sd.get('label') or sd.get('aggregate_label') or 'NEUTRAL',
                'confidence': float(sd.get('confidence', 0.0) or 0.0),
                'headline_count': int(sd.get('headline_count', 0) or 0),
                'sources': sd.get('sources', []),
                'recent_headlines': sd.get('recent_headlines', [])[:5],
            },
            'fear_greed': {
                'value': ef.get('fear_greed_index'),
                'signal': ef.get('fear_greed_signal'),
            },
            'funding_rate': ef.get('funding_rate'),
            'open_interest_usd': ef.get('open_interest'),
            'put_call_ratio': ef.get('put_call_ratio'),
            'macro_composite': ec.get('composite'),
            'macro_risk': ec.get('macro_risk'),
            'timestamp': _dt.utcnow().isoformat() + 'Z',
        }
        try:
            DashboardState().update_live_intelligence(asset, snapshot)
        except Exception:
            pass

    def _log_decision_audit(self, asset, raw_signal, decision, sentiment_data,
                            ext_feats, economic_data, decision_id=None, trace_id=None):
        """Append a full decision record to logs/trade_decisions.jsonl.

        Each line is one cycle: the orchestrator's direction/confidence/veto,
        the agent votes, and the news + macro context that produced them.

        Phase 0: added decision_id (ULID) + stubbed trace_id = decision_id.
        Phase 1: trace_id now comes from the OTel span (hex-encoded 128-bit)
        and lets Tempo cross-link the audit row. If the caller didn't pass
        one (legacy callers, shutdown paths), we fall back to decision_id so
        the JSONL schema stays stable.
        """
        import os, json
        from datetime import datetime as _dt
        from src.orchestration.envelope import synthetic_decision_id
        try:
            if not decision_id:
                decision_id = synthetic_decision_id("audit_no_ctx")
            if not trace_id:
                trace_id = decision_id

            # Defensive attr access — EnhancedDecision fields can be None on
            # degraded-data cycles, and getattr shouldn't raise before we've
            # composed a minimal audit row.
            votes = {}
            agent_votes = getattr(decision, 'agent_votes', None) or {}
            for name, vote in agent_votes.items():
                votes[name] = {
                    'direction': getattr(vote, 'direction', 0),
                    'confidence': round(float(getattr(vote, 'confidence', 0.0) or 0.0), 3),
                    'veto': bool(getattr(vote, 'veto', False)),
                    'reasoning': (getattr(vote, 'reasoning', '') or '')[:160],
                }
            sd = sentiment_data or {}
            ef = ext_feats or {}
            ec = economic_data or {}
            risk_params = getattr(decision, 'risk_params', None) or {}
            record = {
                'ts': _dt.utcnow().isoformat() + 'Z',
                'decision_id': decision_id,
                'trace_id': trace_id,  # Phase 1: real OTel hex trace_id from the cycle span
                'asset': asset,
                'raw_signal': int(raw_signal or 0),
                'decision': {
                    'direction': int(getattr(decision, 'direction', 0) or 0),
                    'confidence': round(float(getattr(decision, 'confidence', 0.0) or 0.0), 3),
                    'position_scale': round(float(getattr(decision, 'position_scale', 0.0) or 0.0), 3),
                    'consensus': getattr(decision, 'consensus_level', 'UNKNOWN') or 'UNKNOWN',
                    'veto': bool(getattr(decision, 'veto', False)),
                    'violations': risk_params.get('violations', []) if isinstance(risk_params, dict) else [],
                },
                'sentiment': {
                    'score': round(float(sd.get('score', sd.get('aggregate_score', 0.0)) or 0.0), 3),
                    'label': sd.get('label') or sd.get('aggregate_label'),
                    'headline_count': sd.get('headline_count', 0),
                    'recent_headlines': sd.get('recent_headlines', [])[:5],
                    'sources': sd.get('sources', []),
                },
                'macro': {
                    'fear_greed': ef.get('fear_greed_index'),
                    'funding_rate': ef.get('funding_rate'),
                    'open_interest_usd': ef.get('open_interest'),
                    'put_call_ratio': ef.get('put_call_ratio'),
                    'composite': ec.get('composite'),
                    'macro_risk': ec.get('macro_risk'),
                },
                'agents': votes,
                # Phase 0: provenance block present with empty strings.
                # Phase 1 will populate data_snapshot_hash, prompt_hash,
                # model_versions, authority_rules_version, config_hash.
                'provenance': {
                    'data_snapshot_hash': '',
                    'prompt_hash': '',
                    'model_versions': {},
                    'authority_rules_version': '',
                    'config_hash': '',
                },
            }
            log_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'logs', 'trade_decisions.jsonl',
            )
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            # Phase 0 gate depends on this log. Do NOT swallow silently.
            # Re-raise so the caller's warning-level handler surfaces it with
            # full traceback. Before Phase 0 this was `except: pass` and cost
            # us 2 days of silent audit-log drops.
            logger.warning(
                f"[AUDIT] inner write failed decision_id={decision_id}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            raise

    def _load_edge_stats(self):
        """Load win rate and expectancy from trade journal for position sizing."""
        for asset in self.assets:
            self.edge_stats[asset] = {
                'wins': 0, 'losses': 0, 'total': 0,
                'win_rate': 0.5, 'avg_win': 0.0, 'avg_loss': 0.0,
                'expectancy': 0.0, 'edge_multiplier': 1.0,
            }
        try:
            trades = self.journal.load_trades(exchange=self._ex_tag.lower())
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
    def _send_alert(self, level: str, title: str, message: str, data: dict = None):
        """Send alert via AlertManager if available. Never blocks trading on failure."""
        if self._alert_manager:
            try:
                self._alert_manager.send_alert(level, title, message, data)
            except Exception:
                pass  # Alerting failure must never block trading

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
                self._send_alert('CRITICAL', f'{self._ex_tag} HALTED',
                    f'Daily loss limit breached: {daily_loss_pct:.1f}% >= {self.daily_loss_limit_pct}%',
                    {'equity': self.equity, 'daily_pnl': self.daily_realized_pnl})
                return False

        # Check max drawdown from session start
        if self.session_start_equity > 0:
            session_dd_pct = abs(self.session_realized_pnl) / self.session_start_equity * 100
            if self.session_realized_pnl < 0 and session_dd_pct >= self.max_drawdown_pct:
                self.trading_halted = True
                self.halt_reason = f"max drawdown {session_dd_pct:.1f}% >= {self.max_drawdown_pct}%"
                self._send_alert('CRITICAL', f'{self._ex_tag} MAX DRAWDOWN',
                    f'Session drawdown limit breached: {session_dd_pct:.1f}% >= {self.max_drawdown_pct}%',
                    {'equity': self.equity, 'session_pnl': self.session_realized_pnl})
                return False

        # ── Dynamic Risk Manager circuit breakers (advanced monitoring) ──
        if self._dynamic_risk:
            try:
                self._dynamic_risk.current_capital = self.equity
                halt, reason, severity = self._dynamic_risk.check_halt_conditions(
                    self.equity, self.daily_realized_pnl)
                if halt and severity == 'HALT':
                    self.trading_halted = True
                    self.halt_reason = f"DRM: {reason}"
                    self._send_alert('CRITICAL', f'{self._ex_tag} CIRCUIT BREAKER', reason,
                        {'severity': severity, 'equity': self.equity})
                    return False
                elif halt and severity == 'PAUSE':
                    # Pause = warn but don't kill
                    print(f"  [{self._ex_tag}] DRM CAUTION: {reason} (severity={severity})")
            except Exception as drm_err:
                logger.debug(f"DRM check error: {drm_err}")

        return True

    # ------------------------------------------------------------------
    # Orphan Position Closer
    # ------------------------------------------------------------------
    def _close_orphan_positions(self):
        """Find and close exchange positions that the bot doesn't track internally.
        Runs every loop. These orphans cause hidden unrealized losses."""
        try:
            if self._paper_mode:
                return  # Paper positions don't exist on exchange — skip
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

                # Close with limit order at best bid/ask for better fill
                symbol = self._get_symbol(asset)
                close_price = None
                try:
                    ob = self.price_source.fetch_order_book(symbol, limit=5)
                    if close_side == 'sell' and ob.get('bids'):
                        close_price = float(ob['bids'][0][0])
                    elif close_side == 'buy' and ob.get('asks'):
                        close_price = float(ob['asks'][0][0])
                except Exception:
                    pass

                if close_price:
                    result = self._api_call(
                        self.price_source.place_order,
                        symbol=symbol,
                        side=close_side,
                        amount=qty,
                        order_type='limit',
                        price=close_price,
                        reduce_only=True,
                    )
                else:
                    result = self._api_call(
                        self.price_source.place_order,
                        symbol=symbol,
                        side=close_side,
                        amount=qty,
                        order_type='market',
                        price=None,
                        reduce_only=True,
                    )

                if result.get('status') == 'success':
                    print(f"  [{self._ex_tag}:{asset}] ORPHAN CLOSED: {result.get('order_id', '?')}")
                    # Verify it actually closed — if not, blacklist to stop the loop
                    time.sleep(2)
                    try:
                        verify_pos = self._exchange_client.get_positions()
                        still_has = any(asset in pp.get('symbol','') and abs(float(pp.get('qty',0) or pp.get('contracts',0))) > 0 for pp in verify_pos)
                        if still_has:
                            print(f"  [{self._ex_tag}:{asset}] ORPHAN STILL OPEN after close — suppressing for 30min")
                            self.failed_close_assets[asset] = time.time() - 600 + 1800  # retry in 30min
                    except Exception:
                        pass
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
        """Return the active exchange client (Bybit, Delta, or None). Robinhood is read-only."""
        ps = self.price_source
        if hasattr(ps, 'bybit') and ps.bybit and ps.bybit.available:
            return ps.bybit
        if hasattr(ps, 'delta') and ps.delta and ps.delta.available:
            return ps.delta
        return None

    @property
    def _exchange_name(self):
        """Return active exchange name.

        Checks live-authenticated exchanges first, then falls back to the
        configured exchange name so the rest of the system (paper mode,
        symbol formatting, spread config) still works even when auth
        hasn't completed or credentials are missing.
        """
        ps = self.price_source
        if hasattr(ps, 'robinhood') and ps.robinhood and ps.robinhood.authenticated:
            return 'robinhood'
        if hasattr(ps, 'bybit') and ps.bybit and ps.bybit.available:
            return 'bybit'
        if hasattr(ps, 'delta') and ps.delta and ps.delta.available:
            return 'delta'
        # Fall back to config — so paper mode, symbol format, spread config
        # all work even when the exchange hasn't authenticated yet
        return getattr(self, '_config_exchange', 'unknown')

    def _api_call(self, fn, *args, **kwargs):
        """Wrap API calls with circuit breaker protection."""
        if self._circuit_breaker and self._circuit_breaker.is_available:
            return self._circuit_breaker.call(fn, *args, **kwargs)
        return fn(*args, **kwargs)

    def _api_call_safe(self, fn, *args, default=None, **kwargs):
        """Wrap non-critical API calls — returns default on failure/breaker open."""
        if self._circuit_breaker:
            return self._circuit_breaker.call_safe(fn, *args, default=default, **kwargs)
        try:
            return fn(*args, **kwargs)
        except Exception:
            return default

    @property
    def _paper_mode(self) -> bool:
        """True when running paper-only (Robinhood read-only, no order execution)."""
        return self._exchange_name == 'robinhood'

    def _get_symbol(self, asset: str) -> str:
        """BTC -> exchange-specific symbol format."""
        if self._exchange_name == 'robinhood':
            return f"{asset}/USD"  # Kraken CCXT uses BTC/USD
        if self._exchange_name == 'delta':
            return f"{asset}USD"  # Delta uses BTCUSD
        return f"{asset}/USDT:USDT"  # Bybit uses BTC/USDT:USDT

    def _get_spot_symbol(self, asset: str) -> str:
        """BTC -> spot symbol."""
        if self._exchange_name == 'robinhood':
            return f"{asset}/USD"
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
    # Connection Health — detect stale CCXT and reconnect
    # ------------------------------------------------------------------
    def _try_reconnect(self, asset: str = ''):
        """
        Reconnect the exchange client if OHLCV data looks stale/frozen.
        This happens when CCXT HTTP connections die silently after hours.
        """
        try:
            import ccxt
            import os
            if self._exchange_name == 'bybit' and self.price_source.bybit:
                print(f"  [{self._ex_tag}:{asset}] RECONNECTING Bybit exchange client...")
                old_ex = self.price_source.bybit.exchange
                # Create fresh CCXT instance with same config
                new_ex = ccxt.bybit({
                    'apiKey': old_ex.apiKey,
                    'secret': old_ex.secret,
                    'sandbox': True,
                    'options': {
                        'defaultType': 'linear',
                        'recvWindow': 60000,
                        'adjustForTimeDifference': True,
                    },
                    'enableRateLimit': True,
                })
                new_ex.load_time_difference()
                new_ex.load_markets()
                # Replace in both the client and PriceFetcher
                self.price_source.bybit.exchange = new_ex
                self.price_source.exchange = new_ex
                offset = getattr(new_ex, 'timeOffset', 0)
                print(f"  [{self._ex_tag}:{asset}] RECONNECTED OK (time offset: {offset}ms)")

            elif self._exchange_name == 'delta' and self.price_source.delta:
                print(f"  [{self._ex_tag}:{asset}] RECONNECTING Delta exchange client...")
                old_ex = self.price_source.delta.exchange
                new_ex = ccxt.delta({
                    'apiKey': old_ex.apiKey,
                    'secret': old_ex.secret,
                    'options': {'defaultType': 'future'},
                    'enableRateLimit': True,
                })
                new_ex.load_markets()
                self.price_source.delta.exchange = new_ex
                self.price_source.exchange = new_ex
                print(f"  [{self._ex_tag}:{asset}] RECONNECTED OK")

        except Exception as e:
            print(f"  [{self._ex_tag}:{asset}] RECONNECT FAILED: {e}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self):
        print("=" * 60)
        ex_name = self._exchange_name.upper()
        mode_tag = "PAPER TRADING" if self._paper_mode else "LIVE"
        print(f"  EMA(8) Crossover + LLM | {ex_name} {mode_tag}")
        print(f"  Assets: {self.assets} | Poll: {self.poll_interval}s")
        if self._use_claude and self._claude_client:
            print(f"  LLM: Claude API ({self._claude_model}) -- Anthropic")
        else:
            print(f"  LLM: {self.ollama_model} @ {self.ollama_base_url}")
        bear_status = "ACTIVE" if self.bear_enabled else "OFF"
        print(f"  Bear Veto: {bear_status} (veto>={self.bear_veto_threshold} reduce>={self.bear_reduce_threshold})")
        if self.sniper_enabled:
            print(f"  SNIPER MODE: confluence>={self.sniper_min_confluence} | score>={self.sniper_min_score} | move>={self.sniper_min_expected_move_pct}% | compound={self.sniper_compound_pct}%")
        print("=" * 60)

        # ── Subsystem Health Report ──
        # Explicitly log which subsystems loaded vs failed (Risk 3: no silent degradation)
        _subsystems = {
            'Anthropic SDK': ANTHROPIC_AVAILABLE,
            'AgentOrchestrator': ORCHESTRATOR_AVAILABLE,
            'OnChain': ONCHAIN_AVAILABLE,
            'MemoryVault': MEMORY_AVAILABLE,
            'TradingBrainV2': BRAIN_V2_AVAILABLE,
            'TradeProtections': PROTECTIONS_AVAILABLE,
            'PriceAction': PRICE_ACTION_AVAILABLE,
            'MarketStructure': MARKET_STRUCTURE_AVAILABLE,
            'ProfitProtector': PROFIT_PROTECTOR_AVAILABLE,
            'VPIN': VPIN_AVAILABLE,
            'Hurst': HURST_AVAILABLE,
            'HMM Regime': HMM_AVAILABLE,
            'Kalman Filter': KALMAN_AVAILABLE,
            'Volatility Model': VOLATILITY_MODEL_AVAILABLE,
            'Cycle Detector': CYCLE_AVAILABLE,
            'LightGBM': LIGHTGBM_AVAILABLE,
            'LSTM Ensemble': LSTM_AVAILABLE,
            'PatchTST': PATCHTST_AVAILABLE,
            'Alpha Decay': ALPHA_DECAY_AVAILABLE,
            'RL Agent': RL_AVAILABLE,
            'MT5 Bridge': MT5_AVAILABLE,
            'LLM Router': LLM_ROUTER_AVAILABLE,
            'Prompt Constraints': PROMPT_CONSTRAINTS_AVAILABLE,
            'Alerting': ALERTING_AVAILABLE,
            'Position Sizing': POSITION_SIZING_AVAILABLE,
            'Dynamic Risk': DYNAMIC_RISK_AVAILABLE,
            'MetaSizer': META_SIZER_AVAILABLE,
            'Vol Regime Detector': VOL_REGIME_DETECTOR_AVAILABLE,
            'TradeTrace': TRADE_TRACE_AVAILABLE,
            'FFT Cycle': FFT_CYCLE_AVAILABLE,
            'Health Checker': HEALTH_CHECKER_AVAILABLE,
            'Advanced Learning': ADVANCED_LEARNING_AVAILABLE,
            'EVT Risk': EVT_RISK_AVAILABLE,
            'Monte Carlo Risk': MC_RISK_AVAILABLE,
            'Sentiment': SENTIMENT_AVAILABLE,
            'Temporal Transformer': TEMPORAL_TRANSFORMER_AVAILABLE,
            'Hawkes Process': HAWKES_AVAILABLE,
            'Event Guard': EVENT_GUARD_AVAILABLE,
        }
        _ok = [k for k, v in _subsystems.items() if v]
        _fail = [k for k, v in _subsystems.items() if not v]
        print(f"\n  SUBSYSTEM HEALTH: {len(_ok)}/{len(_subsystems)} loaded")
        if _fail:
            print(f"  DEGRADED ({len(_fail)}): {', '.join(_fail)}")
            logger.warning(f"[STARTUP] Degraded subsystems: {', '.join(_fail)}")
        else:
            print(f"  ALL SUBSYSTEMS OPERATIONAL")
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

                # Fetch account equity
                unrealized_pnl = 0.0
                wallet_balance = 0.0
                if self._paper_mode:
                    # Paper mode: equity tracked internally from simulated P&L
                    if self._paper:
                        wallet_balance = self._paper.equity
                else:
                    # Live mode: fetch from exchange
                    try:
                        if self._exchange_client:
                            acct = self._exchange_client.get_account()
                            total_equity = float(acct.get('equity', 0) or 0)
                            available = float(acct.get('cash', 0) or 0)
                            unrealized_pnl = float(acct.get('unrealized_pnl', 0) or 0)
                            wallet_balance = float(acct.get('wallet_balance', 0) or 0)
                            if total_equity > 0:
                                self.equity = total_equity
                            if available > 0:
                                self.cash = available
                    except Exception:
                        pass

                ret_pct = ((self.equity - self.initial_capital) / self.initial_capital) * 100.0
                n_pos = len(self.positions)
                total_vetoed = sum(s['vetoed'] for s in self.bear_veto_stats.values())
                total_reduced = sum(s['reduced'] for s in self.bear_veto_stats.values())
                bear_tag = f" | Bear: {total_vetoed}v/{total_reduced}r" if (total_vetoed + total_reduced) > 0 else ""
                pnl_tag = f" | UPnL: ${unrealized_pnl:+,.2f}" if unrealized_pnl != 0 else ""
                wallet_tag = f" | Wallet: ${wallet_balance:,.2f}" if wallet_balance > 0 and abs(wallet_balance - self.equity) > 1 else ""
                sniper_tag = ""
                if self.sniper_enabled:
                    _ss = self.sniper_stats
                    sniper_tag = f" | Sniper: {_ss['signals_seen']}seen/{_ss['filtered']}filtered/{_ss['entered']}entered pool=${self.sniper_profit_pool:,.2f}"
                print(f"\n[{self._ex_tag} BAR {self.bar_count}] Equity: ${self.equity:,.2f}{wallet_tag}{pnl_tag} | Cash: ${self.cash:,.2f} | Return: {ret_pct:+.2f}% | Positions: {n_pos}{bear_tag}{sniper_tag}")

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

                # ── Paper Trading: update positions with prices from OHLCV feed ──
                if self._paper:
                    try:
                        # Pass latest tick prices (already fetched from Kraken OHLCV)
                        _live = {a: self._last_tick_prices.get(a, 0) for a in self.assets}
                        self._paper.update_positions(live_prices=_live)
                        self._paper.save_state()  # Save every bar so dashboard sees updates
                        if self._paper.positions or self._paper.stats['exits'] > 0:
                            print(f"  {self._paper.positions_status()}")
                    except Exception as _pe:
                        logger.debug(f"Paper update error: {_pe}")

                # ── AUTONOMOUS ML RETRAIN ──
                # 1. HMM quick retrain: every 6 hours (fast, uses live OHLCV)
                # 2. LightGBM full retrain: weekly Sunday 3 AM UTC (background thread)
                if not hasattr(self, '_last_retrain_time'):
                    self._last_retrain_time = time.time()
                if not hasattr(self, '_last_lgbm_retrain_check'):
                    self._last_lgbm_retrain_check = 0
                if not hasattr(self, '_lgbm_retrain_running'):
                    self._lgbm_retrain_running = False

                retrain_interval = 6 * 3600  # 6 hours
                if time.time() - self._last_retrain_time > retrain_interval:
                    try:
                        print(f"  [{self._ex_tag}] AUTO-RETRAIN: 6h interval -- retraining HMM + fitting models")
                        self._last_retrain_time = time.time()
                        # Retrain HMM on recent data
                        if self._hmm:
                            for asset in self.assets:
                                try:
                                    symbol = self._get_symbol(asset)
                                    raw = self.price_source.fetch_ohlcv(symbol, timeframe='5m', limit=100)
                                    if raw:
                                        from src.data.fetcher import PriceFetcher
                                        _oh = PriceFetcher.extract_ohlcv(raw)
                                        _closes = _oh['closes']
                                        _volumes = _oh['volumes']
                                        if len(_closes) >= 80:
                                            import numpy as _np
                                            lr = _np.diff(_np.log(_np.array(_closes) + 1e-12))
                                            v20 = _np.array([_np.std(lr[max(0,j-20):j]) for j in range(1, len(lr)+1)])
                                            vc = _np.zeros(len(lr))
                                            mn = min(len(lr), len(v20), len(vc))
                                            self._hmm.fit(lr[-mn:], v20[-mn:], vc[-mn:])
                                            print(f"  [{self._ex_tag}:{asset}] HMM retrained on {mn} observations")
                                except Exception as he:
                                    logger.debug(f"HMM retrain error: {he}")
                    except Exception as re:
                        logger.debug(f"Auto-retrain error: {re}")

                # Weekly LightGBM retrain: Sunday 3 AM UTC (background thread)
                now_utc = datetime.utcnow()
                if (now_utc.weekday() == 6 and now_utc.hour == 3
                        and not self._lgbm_retrain_running
                        and time.time() - self._last_lgbm_retrain_check > 3600):
                    self._last_lgbm_retrain_check = time.time()
                    self._lgbm_retrain_running = True
                    print(f"  [{self._ex_tag}] WEEKLY RETRAIN: launching LightGBM retrain in background thread")

                    def _retrain_lgbm_background():
                        try:
                            from src.models.scheduled_retrain import retrain_all
                            results = retrain_all(assets=[a for a in self.assets], dry_run=False)
                            for r in results:
                                status = r.get('status', 'unknown')
                                acc = r.get('accuracy', 0)
                                print(f"  [{self._ex_tag}] RETRAIN DONE: {r['asset']} — {status} (acc={acc:.4f})")
                            # Hot-reload updated models
                            for asset in self.assets:
                                model_path = os.path.join('models', f'lgbm_{asset.lower()}_trained.txt')
                                if os.path.exists(model_path):
                                    try:
                                        import lightgbm as lgb
                                        self._lgbm_raw[asset] = lgb.Booster(model_file=model_path)
                                        print(f"  [{self._ex_tag}:{asset}] HOT-RELOAD: LightGBM model updated from retrain")
                                    except Exception as le:
                                        logger.warning(f"LightGBM hot-reload failed for {asset}: {le}")
                        except Exception as e:
                            logger.error(f"Weekly LightGBM retrain failed: {e}")
                        finally:
                            self._lgbm_retrain_running = False

                    import threading as _thr
                    _thr.Thread(target=_retrain_lgbm_background, daemon=True, name="lgbm-retrain").start()

                # ── AUTONOMOUS ONLINE TRAINING: Advanced Learning Engine ──
                # Per-bar: update anomaly detector, regime classifier, pipeline overlay for each asset
                # Every 30 min: full cross-asset analysis (patterns, correlations, meta-learning)
                if self._advanced_learning:
                    try:
                        for asset in self.assets:
                            try:
                                symbol = self._get_symbol(asset)
                                raw = self.price_source.fetch_ohlcv(symbol, timeframe='5m', limit=200)
                                if raw:
                                    _oh = PriceFetcher.extract_ohlcv(raw)
                                    if len(_oh['closes']) >= 50:
                                        import numpy as _np
                                        _c = _np.array(_oh['closes'], dtype=float)
                                        _h = _np.array(_oh['highs'], dtype=float)
                                        _l = _np.array(_oh['lows'], dtype=float)
                                        _v = _np.array(_oh['volumes'], dtype=float)
                                        # Per-bar online update (anomaly + regime + overlay)
                                        bar_result = self._advanced_learning.process_bar(asset, _c, _h, _l, _v)
                                        _ov = bar_result.get('overlay')
                                        if _ov:
                                            _regime_tag = bar_result.get('regime')
                                            _regime_str = _regime_tag.regime_type if _regime_tag else '?'
                                            _anom = bar_result.get('anomaly', {})
                                            _anom_tag = f" ANOMALY:{_anom.get('type', 'NONE')}" if _anom and _anom.get('is_anomaly') else ""
                                            print(f"  [{self._ex_tag}:{asset}] META: regime={_regime_str} bear_veto={_ov.bear_veto_threshold} conf>={_ov.min_confidence:.2f} risk_mult={_ov.risk_multiplier:.2f}{_anom_tag}")

                                        # ── Category B online learning: update models with new bar data ──
                                        _log_rets = _np.diff(_np.log(_c + 1e-12))
                                        _bar_ret = float(_log_rets[-1]) if len(_log_rets) > 0 else 0.0

                                        # EVT: update rolling tail risk estimate per bar
                                        if self._evt_risk and len(_log_rets) >= 50:
                                            try:
                                                self._evt_risk.online_update(_bar_ret)
                                            except Exception:
                                                pass

                                        # Hawkes: update with new event time if large move detected
                                        if self._hawkes and len(_log_rets) >= 10:
                                            try:
                                                _abs_rets = _np.abs(_log_rets[-50:])
                                                _thresh = float(_np.mean(_abs_rets) + 2.0 * _np.std(_abs_rets))
                                                if abs(_bar_ret) > _thresh:
                                                    _ev_times = _np.where(_np.abs(_log_rets[-50:]) > _thresh)[0].astype(float)
                                                    self._hawkes.online_update(float(len(_log_rets)), _ev_times.tolist())
                                            except Exception:
                                                pass

                                        # Temporal Transformer: online gradient update with realized return
                                        if self._temporal_transformer and len(_c) >= 120:
                                            try:
                                                _pct = _np.diff(_c[-121:]) / _c[-121:-1]
                                                _hpct = (_h[-120:] - _c[-121:-1]) / _c[-121:-1]
                                                _lpct = (_l[-120:] - _c[-121:-1]) / _c[-121:-1]
                                                _vpct = _np.diff(_v[-121:]) / (_v[-121:-1] + 1e-12)
                                                _hist = _np.column_stack([_pct, _hpct, _lpct, _vpct])
                                                if _hist.shape[1] < self._temporal_transformer.d_model:
                                                    _pad = _np.zeros((_hist.shape[0], self._temporal_transformer.d_model - _hist.shape[1]))
                                                    _hist = _np.hstack([_hist, _pad])
                                                self._temporal_transformer.online_update(_hist, float(_bar_ret))
                                            except Exception:
                                                pass

                                        # Sentiment: compute real values from price-action proxy
                                        if self._sentiment and len(_c) >= 20:
                                            try:
                                                _up_v = sum(float(_v[j]) for j in range(-20, 0) if _c[j] > _c[j-1])
                                                _dn_v = sum(float(_v[j]) for j in range(-20, 0) if _c[j] < _c[j-1])
                                                _total = _up_v + _dn_v + 1e-10
                                                _sent_mean = (_up_v - _dn_v) / _total
                                                # Z-score: compare current sentiment to 50-bar rolling
                                                _sent_history = []
                                                for _si in range(max(0, len(_c)-50), len(_c)-20):
                                                    _uv = sum(float(_v[j]) for j in range(_si, _si+20) if j+1 < len(_c) and _c[j+1] > _c[j])
                                                    _dv = sum(float(_v[j]) for j in range(_si, _si+20) if j+1 < len(_c) and _c[j+1] < _c[j])
                                                    _sent_history.append((_uv - _dv) / (_uv + _dv + 1e-10))
                                                _sent_std = float(_np.std(_sent_history)) if len(_sent_history) > 5 else 0.3
                                                _sent_z = (_sent_mean - float(_np.mean(_sent_history))) / (_sent_std + 1e-10) if _sent_history else 0.0
                                                # Store for use in _evaluate_entry
                                                if not hasattr(self, '_sentiment_cache'):
                                                    self._sentiment_cache = {}
                                                self._sentiment_cache[asset] = {
                                                    'sentiment_mean': float(_np.clip(_sent_mean, -1, 1)),
                                                    'sentiment_z_score': float(_np.clip(_sent_z, -3, 3)),
                                                }
                                            except Exception:
                                                pass

                            except Exception as ae:
                                logger.debug(f"Meta bar update error for {asset}: {ae}")

                        # Full cross-asset analysis every 30 min
                        if time.time() - self._last_full_meta_analysis > self._meta_analysis_interval:
                            self._last_full_meta_analysis = time.time()
                            try:
                                import pandas as _pd
                                multi_data = {}
                                for asset in self.assets:
                                    symbol = self._get_symbol(asset)
                                    raw = self.price_source.fetch_ohlcv(symbol, timeframe='5m', limit=200)
                                    if raw:
                                        _oh = PriceFetcher.extract_ohlcv(raw)
                                        multi_data[asset] = _pd.DataFrame({
                                            'close': _oh['closes'], 'high': _oh['highs'],
                                            'low': _oh['lows'], 'volume': _oh['volumes'],
                                        })
                                if len(multi_data) >= 2:
                                    full_result = self._advanced_learning.process_market_data(multi_data)
                                    _corrs = full_result.get('correlations', {})
                                    _pats = full_result.get('patterns', {})
                                    _active_pats = [k for k, v in _pats.items() if v]
                                    print(f"  [{self._ex_tag}] META FULL: correlations={_corrs} active_patterns={_active_pats}")
                                    self._advanced_learning.save_learned_models()
                            except Exception as fe:
                                logger.debug(f"Full meta-analysis error: {fe}")
                    except Exception as me:
                        logger.debug(f"Advanced learning loop error: {me}")

                # Paper report every 100 bars (~17min at 10s poll)
                if self._paper and self.bar_count % 100 == 0 and self._paper.stats['exits'] > 0:
                    print(self._paper.report())

                # Sleep until next bar
                elapsed = time.time() - loop_start
                sleep_time = max(1, self.poll_interval - int(elapsed))
                print(f"  [{self._ex_tag} SLEEP] {sleep_time}s")
                time.sleep(sleep_time)

            except KeyboardInterrupt:
                if self._paper:
                    self._paper.save_state()
                    print(self._paper.report())
                self._print_session_summary()
                print("\n[SHUTDOWN] Graceful exit.")
                break
            except Exception as e:
                print(f"  [{self._ex_tag} ERROR] {e}")
                logger.exception("Main loop error")
                time.sleep(5)

    # ------------------------------------------------------------------
    # Session Summary — printed on shutdown
    # ------------------------------------------------------------------
    def _print_session_summary(self):
        """Print complete trade log table and session P&L summary."""
        trades = self._trade_log
        if not trades:
            print("\n" + "=" * 80)
            print("  SESSION SUMMARY: No trades executed")
            print("=" * 80)
            return

        total_pnl = sum(t['realized_pnl'] for t in trades)
        wins = sum(1 for t in trades if t['realized_pnl'] > 0)
        losses = sum(1 for t in trades if t['realized_pnl'] <= 0)
        n = len(trades)
        wr = wins / n * 100 if n > 0 else 0
        total_spread = sum(t.get('spread_cost', 0) for t in trades)
        largest_win = max((t['realized_pnl'] for t in trades), default=0)
        largest_loss = min((t['realized_pnl'] for t in trades), default=0)
        avg_pnl = total_pnl / n if n > 0 else 0
        gross_profit = sum(t['realized_pnl'] for t in trades if t['realized_pnl'] > 0)
        gross_loss = abs(sum(t['realized_pnl'] for t in trades if t['realized_pnl'] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        print("\n" + "=" * 100)
        print("  SESSION TRADE LOG")
        print("=" * 100)
        print(f"  {'#':<4} {'Market':<10} {'Entry Price':>14} {'Exit Price':>14} {'Qty':>12} {'Type':<6} {'Realized P&L':>14} {'Spread':>10} {'Time':<20}")
        print("  " + "-" * 96)

        running_pnl = 0.0
        for i, t in enumerate(trades, 1):
            running_pnl += t['realized_pnl']
            pnl_str = f"${t['realized_pnl']:+,.2f}"
            spread_str = f"${t.get('spread_cost', 0):,.2f}" if t.get('spread_cost', 0) > 0 else "-"
            print(f"  {i:<4} {t['market']:<10} ${t['entry_price']:>12,.2f} ${t['exit_price']:>12,.2f} {t['qty']:>12.6f} {t['direction']:<6} {pnl_str:>14} {spread_str:>10} {t['time']:<20}")

        print("  " + "-" * 96)
        print(f"\n  {'SUMMARY':^100}")
        print("  " + "=" * 96)
        pnl_color = "PROFIT" if total_pnl >= 0 else "LOSS"
        print(f"  Total Trades:    {n:<6}  |  Wins: {wins}  |  Losses: {losses}  |  Win Rate: {wr:.1f}%")
        print(f"  Net P&L:         ${total_pnl:+,.2f} ({pnl_color})")
        print(f"  Gross Profit:    ${gross_profit:+,.2f}  |  Gross Loss: ${gross_loss:,.2f}  |  Profit Factor: {pf:.2f}")
        print(f"  Largest Win:     ${largest_win:+,.2f}  |  Largest Loss: ${largest_loss:+,.2f}  |  Avg P&L: ${avg_pnl:+,.2f}")
        if total_spread > 0:
            print(f"  Total Spread:    ${total_spread:,.2f} (Robinhood)")
        print("  " + "=" * 96)

    # ------------------------------------------------------------------
    # BTC-ETH Pairs Trading Signal (cointegration spread z-score)
    # ------------------------------------------------------------------
    def _check_pairs_signal(self) -> Dict[str, Any]:
        """
        Check BTC-ETH cointegration and return spread z-score signal.

        Uses cached OHLCV data from the timeframe cache to avoid extra API calls.
        Returns informational signal for LLM context (not a direct trade trigger).

        Returns:
            {signal: str, z_score: float, cointegrated: bool, hedge_ratio: float, ...}
        """
        result = {
            'signal': 'NONE',
            'z_score': 0.0,
            'cointegrated': False,
            'context_line': '',
        }

        if not self._coint_engine:
            return result

        try:
            # Get BTC and ETH close prices from cached 4h data (preferred) or 1h
            btc_closes = None
            eth_closes = None

            for tf in ['4h', '1h', '5m']:
                btc_cache = self._tf_cache.get(f"BTC_{tf}")
                eth_cache = self._tf_cache.get(f"ETH_{tf}")

                if btc_cache and eth_cache:
                    btc_ohlcv = btc_cache.get('ohlcv', {})
                    eth_ohlcv = eth_cache.get('ohlcv', {})
                    btc_c = btc_ohlcv.get('closes', [])
                    eth_c = eth_ohlcv.get('closes', [])

                    if len(btc_c) >= 50 and len(eth_c) >= 50:
                        # Align lengths (take last min(len_a, len_b, 100) bars)
                        n = min(len(btc_c), len(eth_c), 100)
                        btc_closes = btc_c[-n:]
                        eth_closes = eth_c[-n:]
                        break

            if btc_closes is None or eth_closes is None:
                logger.debug("[PAIRS] Not enough cached BTC/ETH data for cointegration check")
                return result

            import numpy as np
            btc_arr = np.array(btc_closes, dtype=float)
            eth_arr = np.array(eth_closes, dtype=float)

            # Run cointegration test: BTC = A, ETH = B
            spread_result = self._coint_engine.spread_signal(btc_arr, eth_arr)

            z_score = spread_result.get('spread_z_score', 0.0)
            cointegrated = spread_result.get('cointegrated', False)
            hedge_ratio = spread_result.get('hedge_ratio', 0.0)
            signal_int = spread_result.get('signal', 0)
            half_life = spread_result.get('spread_half_life', 999.0)

            # Determine human-readable signal
            if not cointegrated:
                signal_str = 'NOT_COINTEGRATED'
                context = f"BTC-ETH NOT cointegrated (p={spread_result.get('adf_pvalue', 1.0):.3f}) -> pairs signal inactive"
            elif z_score < -2.0:
                signal_str = 'BUY_BTC_REL_ETH'
                context = (f"BTC-ETH spread OVERSOLD z={z_score:.2f} (half-life={half_life:.0f} bars, "
                           f"hedge={hedge_ratio:.3f}) -> BTC undervalued vs ETH, expect BTC outperformance")
            elif z_score > 2.0:
                signal_str = 'SELL_BTC_REL_ETH'
                context = (f"BTC-ETH spread OVERBOUGHT z={z_score:.2f} (half-life={half_life:.0f} bars, "
                           f"hedge={hedge_ratio:.3f}) -> BTC overvalued vs ETH, expect ETH outperformance")
            else:
                signal_str = 'NEUTRAL'
                context = (f"BTC-ETH spread NEUTRAL z={z_score:.2f} (cointegrated, "
                           f"half-life={half_life:.0f} bars) -> no pairs signal")

            result = {
                'signal': signal_str,
                'z_score': z_score,
                'cointegrated': cointegrated,
                'hedge_ratio': hedge_ratio,
                'half_life': half_life,
                'adf_pvalue': spread_result.get('adf_pvalue', 1.0),
                'context_line': context,
            }

            print(f"  [PAIRS] BTC-ETH: z={z_score:.2f} signal={signal_str}")
            self._last_pairs_signal = result

        except Exception as e:
            logger.warning(f"[PAIRS] Cointegration check failed: {e}")
            # Don't crash — pairs signal is informational only

        return result

    # ------------------------------------------------------------------
    # Per-asset processing
    # ------------------------------------------------------------------
    def _process_asset(self, asset: str):
        # Position tracking diagnostic
        if asset in self.positions:
            _p = self.positions[asset]
            print(f"  [{self._ex_tag}:{asset}] TRACKING: {_p['direction']} @ ${_p['entry_price']:,.2f} age={int((time.time()-_p.get('entry_time',time.time()))/60)}min")
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
        # MULTI-TIMEFRAME SIGNAL DETECTION
        # Fetch ALL timeframes, compute EMA(8) crossover on each,
        # LLM picks the BEST timeframe to trade on
        # ══════════════════════════════════════════════════════════════

        # 5m is the anchor — must succeed for heartbeat/staleness checks
        try:
            raw_5m = self.price_source.fetch_ohlcv(symbol, timeframe='5m', limit=100)
        except Exception as e:
            print(f"  [{self._ex_tag}:{asset}] OHLCV fetch failed: {e}")
            self._try_reconnect(asset)
            return
        ohlcv = PriceFetcher.extract_ohlcv(raw_5m)

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        opens = ohlcv['opens']
        volumes = ohlcv['volumes']

        # L1: Data Ingestion log
        try:
            DashboardState().add_layer_log('L1', f"{asset}: fetched {len(closes)} candles @ ${closes[-1]:,.2f}", "info")
        except Exception:
            pass

        if len(closes) < 20:
            print(f"  [{self._ex_tag}:{asset}] Not enough 5m data ({len(closes)} candles)")
            return

        # ── STALENESS DETECTOR ──
        if len(closes) >= 10:
            last_10 = closes[-10:]
            if len(set(round(c, 2) for c in last_10)) <= 1:
                print(f"  [{self._ex_tag}:{asset}] STALE DATA: last 10 candles all ${last_10[-1]:,.2f} - reconnecting")
                self._try_reconnect(asset)
                return

        # ── Agentic shadow loop (C4d) ─────────────────────────────────
        # When ACT_AGENTIC_LOOP=1 (or config.agentic_loop.enabled=true),
        # run the LLM-driven TradePlan compiler in SHADOW MODE alongside
        # the existing decision path. The compiled plan is logged to
        # warm_store for A/B analysis but does NOT replace the executor's
        # live decision — that flip comes after paper soak shows shadow
        # plans match reality. Never raises; errors logged + ignored.
        self._run_agentic_shadow(asset, closes)

        if raw_5m and len(raw_5m) > 0:
            newest_ts = raw_5m[-1][0] / 1000.0
            age_minutes = (time.time() - newest_ts) / 60
            if age_minutes > 15:
                print(f"  [{self._ex_tag}:{asset}] STALE CANDLES: newest is {age_minutes:.0f}min old - reconnecting")
                self._try_reconnect(asset)
                return

        # ── PRICE SANITY ──
        # Reject candle-to-candle jumps > 25% (data quality check)
        if len(closes) >= 3:
            for i in range(-2, 0):
                prev_c = closes[i - 1]
                curr_c = closes[i]
                if prev_c > 0:
                    jump_pct = abs(curr_c - prev_c) / prev_c * 100
                    if jump_pct > 25.0:
                        print(f"  [{self._ex_tag}:{asset}] PRICE JUMP: {jump_pct:.1f}% between candles (${prev_c:,.0f} -> ${curr_c:,.0f}) — skipping")
                        return

        # Reject prices outside sane ranges (data quality)
        _sane_ranges = {'BTC': (5_000, 500_000), 'ETH': (200, 50_000)}
        _sane = _sane_ranges.get(asset)
        if _sane and len(closes) >= 1:
            last_price = closes[-1]
            if last_price < _sane[0] or last_price > _sane[1]:
                print(f"  [{self._ex_tag}:{asset}] BAD PRICE: ${last_price:,.0f} outside sane range ${_sane[0]:,}-${_sane[1]:,} — skipping")
                return

        # ── Live tick price for SL management ──
        try:
            ticker = self.price_source.exchange.fetch_ticker(symbol) if self.price_source.exchange else {}
            live_price = float(ticker.get('last', 0)) if ticker.get('last') else closes[-1]
        except Exception:
            live_price = closes[-1]
        tick_price = live_price

        # Store tick price for paper position updates (avoids extra API calls)
        if not hasattr(self, '_last_tick_prices'):
            self._last_tick_prices = {}
        self._last_tick_prices[asset] = tick_price

        # 5m EMA/ATR (always computed — used for position management + as fallback)
        ema_vals = ema(closes, self.ema_period)
        atr_vals = atr(highs, lows, closes, 14)
        current_ema = ema_vals[-2] if len(ema_vals) >= 2 else closes[-1]
        prev_ema = ema_vals[-3] if len(ema_vals) >= 3 else current_ema
        current_atr = atr_vals[-1] if atr_vals else 0
        ema_direction = "RISING" if current_ema > prev_ema else "FALLING"
        price = closes[-2]  # Last confirmed close

        # ══════════════════════════════════════════════════════════════
        # FETCH ALL TIMEFRAMES & COMPUTE SIGNALS ON EACH
        # ══════════════════════════════════════════════════════════════
        tf_data = {'5m': ohlcv}  # 5m always fetched for price/staleness/ATR
        tf_signals = {}

        # Compute 5m signal (always — used for price management even if not traded)
        tf_signals['5m'] = self._compute_tf_signal(ohlcv, '5m')

        # Fetch and compute signals for configured timeframes (with caching)
        for tf in self.SIGNAL_TIMEFRAMES:
            if tf == '5m':
                continue
            try:
                tf_ohlcv = self._fetch_tf_ohlcv(symbol, asset, tf, self.TF_FETCH_LIMITS[tf])
                if len(tf_ohlcv.get('closes', [])) >= 20:
                    tf_data[tf] = tf_ohlcv
                    tf_signals[tf] = self._compute_tf_signal(tf_ohlcv, tf)
            except Exception as e:
                logger.debug(f"{tf} fetch failed for {asset}: {e}")

        # Active signals = ONLY from configured SIGNAL_TIMEFRAMES (not 5m unless configured)
        # This prevents 5m scalp signals from triggering trades on Robinhood swing config
        active_tf_signals = {
            tf: sig for tf, sig in tf_signals.items()
            if sig.get('signal') != 'NEUTRAL' and tf in self.SIGNAL_TIMEFRAMES
        }

        # Store best active TF's ATR for spread filter (higher TF = more accurate for swing)
        if not hasattr(self, '_last_chosen_tf_atr'):
            self._last_chosen_tf_atr = {}
        if active_tf_signals:
            _best_tf = max(active_tf_signals.keys(),
                          key=lambda t: active_tf_signals[t].get('current_atr', 0))
            self._last_chosen_tf_atr[asset] = active_tf_signals[_best_tf].get('current_atr', current_atr)

        # HTF alignment (computed from tf_signals)
        htf_1h_direction = tf_signals.get('1h', {}).get('ema_direction', 'FLAT')
        htf_4h_direction = tf_signals.get('4h', {}).get('ema_direction', 'FLAT')
        htf_alignment = 1  # 5m always counts
        if htf_1h_direction == ema_direction:
            htf_alignment += 1
        if htf_4h_direction == ema_direction:
            htf_alignment += 1

        # ── Build multi-timeframe signal summary for LLM ──
        tf_perf = self.tf_performance
        mtf_signal_lines = []
        for tf in self.SIGNAL_TIMEFRAMES:
            if tf in tf_signals:
                s = tf_signals[tf]
                sig = s.get('signal', 'NEUTRAL')
                perf = tf_perf.get(tf, {})
                w, l = perf.get('wins', 0), perf.get('losses', 0)
                pnl = perf.get('total_pnl', 0)
                wr = (w / (w + l) * 100) if (w + l) > 0 else 0
                perf_str = f"WR={wr:.0f}% ({w}W/{l}L ${pnl:+,.0f})" if (w + l) > 0 else "no history"

                if sig != 'NEUTRAL':
                    mtf_signal_lines.append(
                        f"  [{tf}] {sig} | EMA: ${s['current_ema']:.2f} ({s['ema_direction']}) "
                        f"| ATR: ${s['current_atr']:.2f} ({s['atr_pct']:.2f}%) "
                        f"| Slope: {s['ema_slope_pct']:+.4f}%/bar | Trend: {s['trend_bars']} bars "
                        f"| Gap: {s['ema_separation_pct']:.2f}% | Vol: {s['vol_trend']} "
                        f"| History: {perf_str}"
                    )
                else:
                    mtf_signal_lines.append(f"  [{tf}] NEUTRAL (no crossover) | History: {perf_str}")
            else:
                mtf_signal_lines.append(f"  [{tf}] N/A (fetch failed)")

        mtf_signal_block = chr(10).join(mtf_signal_lines)

        # ── Build MTF candle summary for LLM context ──
        mtf_candle_summary = ""
        try:
            mtf_parts = []
            for tf in ['1m', '15m', '1h', '4h']:
                tf_ohlcv = tf_data.get(tf)
                if not tf_ohlcv or len(tf_ohlcv.get('closes', [])) < 5:
                    continue
                c_tf = tf_ohlcv['closes']
                h_tf = tf_ohlcv['highs']
                l_tf = tf_ohlcv['lows']
                o_tf = tf_ohlcv['opens']
                v_tf = tf_ohlcv['volumes']
                n_show = {'1m': 10, '15m': 8, '1h': 6, '4h': 4}.get(tf, 6)
                ema_tf = ema(c_tf, self.ema_period) if len(c_tf) >= self.ema_period else []
                tf_dir = tf_signals.get(tf, {}).get('ema_direction', 'FLAT')
                lines_tf = []
                for i in range(-min(n_show, len(c_tf)), 0):
                    idx = len(c_tf) + i
                    e_val = ema_tf[idx] if idx < len(ema_tf) else 0
                    lines_tf.append(f"  O={o_tf[idx]:.2f} H={h_tf[idx]:.2f} L={l_tf[idx]:.2f} C={c_tf[idx]:.2f} V={v_tf[idx]:.0f} EMA={e_val:.2f}")
                label = {'1m': '1-MINUTE', '15m': '15-MINUTE', '1h': '1-HOUR', '4h': '4-HOUR'}.get(tf, tf)
                mtf_parts.append(f"{label} CANDLES (EMA direction: {tf_dir}):" + chr(10) + chr(10).join(lines_tf))
            if mtf_parts:
                mtf_candle_summary = chr(10) + chr(10).join(mtf_parts)
        except Exception as e:
            logger.debug(f"MTF candle summary error for {asset}: {e}")

        # ── FULL INDICATOR SUITE (from 5m — feeds ML models + LLM) ──
        indicator_context = {}
        try:
            if len(closes) >= 20:
                rsi_vals = rsi(closes, 14)
                indicator_context['rsi'] = round(rsi_vals[-2], 1) if len(rsi_vals) >= 2 else 50

                macd_line, signal_line, hist = macd(closes)
                indicator_context['macd_hist'] = round(hist[-2], 4) if len(hist) >= 2 else 0
                indicator_context['macd_cross'] = 'BULLISH' if len(hist) >= 3 and hist[-2] > 0 and hist[-3] < 0 else ('BEARISH' if len(hist) >= 3 and hist[-2] < 0 and hist[-3] > 0 else 'NONE')

                bb_upper, bb_mid, bb_lower = bollinger_bands(closes, 20)
                if len(bb_upper) >= 2 and bb_upper[-2] > 0:
                    bb_pos = (closes[-2] - bb_lower[-2]) / (bb_upper[-2] - bb_lower[-2]) if (bb_upper[-2] - bb_lower[-2]) > 0 else 0.5
                    indicator_context['bb_position'] = round(bb_pos, 2)
                    indicator_context['bb_width'] = round((bb_upper[-2] - bb_lower[-2]) / bb_mid[-2] * 100, 2) if bb_mid[-2] > 0 else 0

                k_vals, d_vals = stochastic(highs, lows, closes)
                if len(k_vals) >= 2:
                    indicator_context['stoch_k'] = round(k_vals[-2], 1)
                    indicator_context['stoch_d'] = round(d_vals[-2], 1)

                obv_vals = obv(closes, volumes)
                if len(obv_vals) >= 10:
                    obv_slope = (obv_vals[-2] - obv_vals[-6]) / 4 if obv_vals[-6] != 0 else 0
                    indicator_context['obv_trend'] = 'RISING' if obv_slope > 0 else 'FALLING'

                adx_vals = adx(highs, lows, closes, 14)
                if len(adx_vals) >= 2:
                    indicator_context['adx'] = round(adx_vals[-2], 1)
                    indicator_context['trend_strength'] = 'STRONG' if adx_vals[-2] > 25 else 'WEAK'

                roc_vals = roc(closes, 12)
                if len(roc_vals) >= 2:
                    indicator_context['roc_12'] = round(roc_vals[-2], 2)

                wr_vals = williams_r(highs, lows, closes, 14)
                if len(wr_vals) >= 2:
                    indicator_context['williams_r'] = round(wr_vals[-2], 1)

                chop_vals = choppiness_index(highs, lows, closes, 14)
                if len(chop_vals) >= 2:
                    indicator_context['choppiness'] = round(chop_vals[-2], 1)
                    indicator_context['market_type'] = 'RANGING' if chop_vals[-2] > 61.8 else 'TRENDING'

                vd_vals = volume_delta(closes, opens, volumes)
                if len(vd_vals) >= 2:
                    indicator_context['vol_delta'] = round(vd_vals[-2], 0)

                cmf_vals = chaikin_money_flow(highs, lows, closes, volumes, 20)
                if len(cmf_vals) >= 2:
                    indicator_context['cmf'] = round(cmf_vals[-2], 3)
                    indicator_context['money_flow'] = 'INFLOW' if cmf_vals[-2] > 0 else 'OUTFLOW'

                # Feed indicators to drift detector for PSI monitoring
                if self._drift_detector:
                    self._drift_detector.update_batch({
                        k: float(v) for k, v in indicator_context.items()
                        if isinstance(v, (int, float))
                    })
        except Exception:
            pass

        # L2: Feature Engineering log
        try:
            n_ind = len(indicator_context)
            rsi_v = indicator_context.get('rsi', '?')
            DashboardState().add_layer_log('L2', f"{asset}: {n_ind} indicators computed (RSI={rsi_v}, EMA={ema_direction})", "info")
        except Exception:
            pass

        # Fetch L2 order book for support/resistance walls
        try:
            order_book = self.price_source.fetch_order_book(symbol, limit=25)
            ob_levels = self._extract_ob_levels(order_book, tick_price)
        except Exception:
            ob_levels = {'bid_wall': 0, 'ask_wall': 0, 'bid_walls': [], 'ask_walls': [], 'imbalance': 0, 'bid_depth_usd': 0, 'ask_depth_usd': 0}

        # ── Feed VPIN guard with recent candle volume ──
        if asset in self._vpin_guards and len(closes) >= 3:
            try:
                for i in range(-3, -1):
                    idx = len(closes) + i
                    c, o, v = closes[idx], opens[idx], volumes[idx]
                    side = 'buy' if c >= o else 'sell'
                    self._vpin_guards[asset].add_trade(c, v, side)
            except Exception:
                pass

        # ── Determine primary signal (5m for backward compat + position mgmt) ──
        ema_signal_raw = tf_signals.get('5m', {}).get('signal', 'NEUTRAL')
        is_reversal_signal = False

        # ── MULTI-STRATEGY CONSENSUS (replaces EMA-only gatekeeper) ──
        _ema_int = 1 if ema_signal_raw == 'BUY' else (-1 if ema_signal_raw == 'SELL' else 0)
        _multi_details = {}
        if self._multi_engine:
            try:
                _hurst_val = getattr(self, '_last_hurst', {}).get(asset, 0.5)
                _hmm_reg = getattr(self, '_last_hmm_regime', {}).get(asset, 'SIDEWAYS')
                _vol_reg = getattr(self, '_last_vol_regime', {}).get(asset, 'NORMAL')
                _multi_signals = self._multi_engine.generate_all_signals(
                    closes, highs, lows, volumes, ema_signal=_ema_int,
                )
                _multi_weights = self._multi_engine.compute_regime_weights(
                    hurst=_hurst_val, hmm_regime=_hmm_reg, volatility_regime=_vol_reg,
                )
                signal, _meta_conf, _multi_details = self._multi_engine.combine(
                    _multi_signals, _multi_weights,
                )
                # Format for log
                _strat_summary = " | ".join(
                    f"{n}={d['signal_word']}" for n, d in _multi_details.items() if not n.startswith('_')
                )
                _consensus = _multi_details.get('_agreement', '?')
                # Cache for conviction-gate consumption in _evaluate_entry
                if not hasattr(self, '_last_multi_details'):
                    self._last_multi_details = {}
                self._last_multi_details[asset] = _multi_details
                print(f"  [{self._ex_tag}:{asset}] MULTI-STRATEGY: {_strat_summary} | consensus={_consensus} score={_multi_details.get('_consensus_score', 0):.3f}")
            except Exception as e:
                signal = ema_signal_raw
                logger.debug(f"[MULTI-STRATEGY] {asset} failed, EMA fallback: {e}")
        else:
            signal = ema_signal_raw

        # ── STRATEGY UNIVERSE CONSENSUS (242 strategies vote) ──
        _universe_consensus = None
        _universe_confidence = 0
        if self._universe and len(closes) >= 50:
            try:
                _all_signals = self._universe.evaluate_all(closes, highs, lows, volumes)
                _universe_consensus, _universe_confidence = self._universe.get_consensus(_all_signals)
                _buy_count = sum(1 for s in _all_signals.values() if s > 0)
                _sell_count = sum(1 for s in _all_signals.values() if s < 0)
                _flat_count = sum(1 for s in _all_signals.values() if s == 0)
                print(f"  [{self._ex_tag}:{asset}] UNIVERSE: {_buy_count}↑ {_sell_count}↓ {_flat_count}— | consensus={_universe_consensus} conf={_universe_confidence:.2f}")

                # Universe can OVERRIDE multi-strategy signal if strong consensus
                if _universe_confidence > 0.60 and _universe_consensus == 'BUY' and signal != 'BUY':
                    print(f"  [{self._ex_tag}:{asset}] UNIVERSE OVERRIDE: {signal} → BUY (60%+ strategies agree)")
                    signal = 'BUY'
                elif _universe_confidence > 0.60 and _universe_consensus == 'SELL' and signal != 'SELL':
                    if not self._longs_only:
                        signal = 'SELL'
            except Exception as e:
                logger.debug(f"[UNIVERSE] {asset} eval failed: {e}")

        # ── GENETIC EVOLVED STRATEGIES (hall-of-fame strategies contribute signals) ──
        _genetic_vote = 0
        _genetic_count = 0
        # Periodic refresh of hall-of-fame (every 30 min)
        if self._genetic_engine and time.time() - self._genetic_last_reload > 1800:
            self._reload_genetic_hall_of_fame()
        if self._genetic_engine and self._genetic_hall_of_fame and len(closes) >= 100:
            try:
                from src.trading.genetic_strategy_engine import StrategyDNA, execute_strategy
                for _hof_entry in self._genetic_hall_of_fame[:5]:  # Top 5 evolved strategies
                    _dna = StrategyDNA()
                    _dna.genes = _hof_entry.get('genes', _dna.genes)
                    _dna.entry_rule = _hof_entry.get('entry_rule', _dna.entry_rule)
                    _dna.exit_rule = _hof_entry.get('exit_rule', _dna.exit_rule)
                    _sig = execute_strategy(_dna, closes, highs, lows, volumes)
                    if _sig != 0:  # execute_strategy returns int, not list
                        _genetic_vote += _sig
                        _genetic_count += 1
                if _genetic_count > 0:
                    _genetic_dir = "BUY" if _genetic_vote > 0 else ("SELL" if _genetic_vote < 0 else "FLAT")
                    print(f"  [{self._ex_tag}:{asset}] GENETIC: {_genetic_count} evolved strategies vote {_genetic_dir} (net={_genetic_vote:+d})")
            except Exception as e:
                logger.debug(f"[GENETIC] {asset} eval failed: {e}")

        # Print status with active signals across all timeframes
        active_tfs_str = ", ".join(f"{tf}={s['signal']}" for tf, s in active_tf_signals.items()) or "none"
        ob_imb = ob_levels.get('imbalance', 0)
        ob_bid = ob_levels.get('bid_wall', 0)
        ob_ask = ob_levels.get('ask_wall', 0)
        ob_info = f"OB[imb={ob_imb:+.2f}"
        if ob_bid > 0:
            ob_info += f" sup=${ob_bid:,.2f}"
        if ob_ask > 0:
            ob_info += f" res=${ob_ask:,.2f}"
        ob_info += "]"
        print(f"  [{self._ex_tag}:{asset}] ${tick_price:,.2f} | EMA(5m): ${current_ema:.2f} {ema_direction} | Signals: [{active_tfs_str}] | ATR: ${current_atr:.2f} | {ob_info}")

        # ── BTC-ETH Pairs Trading Signal (informational — feeds LLM context) ──
        pairs_signal = {}
        if self._coint_engine and asset in ('BTC', 'ETH'):
            try:
                pairs_signal = self._check_pairs_signal()
            except Exception as e:
                logger.debug(f"[PAIRS] {asset} pairs check failed: {e}")

        # ── Stale position check (SKIP in paper mode — exchange has no paper positions) ──
        if asset in self.positions and asset not in self.failed_close_assets and not self._paper_mode:
            try:
                if self._exchange_client:
                    ex_pos = self._exchange_client.get_positions()
                    has_exchange_pos = any(asset in pp.get('symbol','') and float(pp.get('qty',0)) > 0 for pp in ex_pos)
                    if not has_exchange_pos:
                        print(f"  [{self._ex_tag}:{asset}] STALE position cleared (not on exchange)")
                        del self.positions[asset]
            except Exception:
                pass

        # ── Position management uses LIVE price; new entries use multi-TF signals ──
        if asset in self.positions:
            _pos_before = list(self.positions.keys())
            self._manage_position(asset, tick_price, ohlcv, ema_vals, atr_vals, ema_direction, signal, ob_levels)
            if asset not in self.positions:
                print(f"  [{self._ex_tag}:{asset}] *** POSITION REMOVED by _manage_position *** (was: {_pos_before})")
        else:
            # ── LONGS-ONLY GATE (Robinhood spot) — only block new SHORT entries, not status/management ──
            if self._longs_only and signal == 'SELL' and not active_tf_signals:
                return  # 5m is SELL and no higher TF has a BUY — skip

            # ── TIMEFRAME ALIGNMENT OVERRIDE (Fix 3 — Robinhood) ──
            # When 1h+4h+1d ALL agree on direction, force entry evaluation
            # even if individual strategy signals are weak. This catches
            # the START of big multi-day moves before momentum builds.
            _tf_override = False
            if self._rh_tf_alignment_override and htf_alignment >= 2:
                # 2+ timeframes agree on direction
                _all_buy = htf_1h_direction == 'RISING' and htf_4h_direction in ('RISING', 'FLAT')
                if _all_buy:
                    _tf_override = True
                    signal = 'BUY'  # Force BUY signal from TF alignment
                    print(f"  [{self._ex_tag}:{asset}] TF ALIGNMENT OVERRIDE: 1h={htf_1h_direction} 4h={htf_4h_direction} → forcing BUY evaluation")

            # Need at least ONE active signal on any timeframe to evaluate entry
            if not active_tf_signals and not _tf_override:
                return  # No crossover on any TF — nothing to evaluate

            self._evaluate_entry(asset, tick_price, ohlcv, ema_vals, atr_vals,
                                 ema_direction, signal, closes, highs, lows,
                                 opens, volumes, current_ema, current_atr, ob_levels,
                                 htf_1h_direction=htf_1h_direction,
                                 htf_4h_direction=htf_4h_direction,
                                 htf_alignment=htf_alignment,
                                 is_reversal_signal=is_reversal_signal,
                                 mtf_candle_summary=mtf_candle_summary,
                                 indicator_context=indicator_context,
                                 tf_signals=tf_signals,
                                 active_tf_signals=active_tf_signals,
                                 mtf_signal_block=mtf_signal_block,
                                 pairs_signal=pairs_signal,
                                 genetic_vote=_genetic_vote,
                                 genetic_count=_genetic_count)

    # ------------------------------------------------------------------
    # Multi-timeframe signal computation
    # ------------------------------------------------------------------
    def _compute_tf_signal(self, ohlcv: dict, tf_label: str):
        """Compute EMA(8) NEW LINE detection for a single timeframe.

        The strategy (from reference images):
        - ENTRY: Detect when EMA line CHANGES DIRECTION (new line forms)
          Image 1: EMA was falling → price cuts through → EMA starts rising = NEW UP LINE = CALL
          Image 3: EMA was rising → price cuts through → EMA starts falling = NEW DOWN LINE = PUT
        - RIDE: Stay in as long as the new EMA line continues (same direction)
        - EXIT: When EMA direction changes AGAIN (new opposite line forms)

        Key detection: INFLECTION POINT = where EMA slope changes sign after
        trending in opposite direction for 3+ bars. This is the exact moment
        the "new line" starts forming on the chart.
        """
        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        opens = ohlcv['opens']
        volumes = ohlcv['volumes']

        if len(closes) < 20:
            return {'timeframe': tf_label, 'signal': 'NEUTRAL', 'reason': 'not enough data'}

        ema_vals = ema(closes, self.ema_period)
        atr_vals = atr(highs, lows, closes, 14)

        current_ema = ema_vals[-2] if len(ema_vals) >= 2 else closes[-1]
        prev_ema = ema_vals[-3] if len(ema_vals) >= 3 else current_ema
        current_atr = atr_vals[-1] if atr_vals and len(atr_vals) > 0 else (closes[-1] * 0.01 if len(closes) > 0 else 1.0)
        price = closes[-2]  # Last confirmed close

        ema_direction = "RISING" if current_ema > prev_ema else "FALLING"

        # EMA slope as %/bar
        slope_pct = ((current_ema - prev_ema) / prev_ema * 100) if prev_ema > 0 else 0

        # EMA separation
        ema_separation = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0

        # ATR as % of price
        atr_pct = (current_atr / price * 100) if price > 0 else 0

        # ══════════════════════════════════════════════════════════════
        # NEW LINE DETECTION (inflection point)
        # Walk backward through EMA values to find:
        #   1. How many bars the CURRENT direction has been going (new_line_bars)
        #   2. How many bars the PREVIOUS direction went (prior_trend_bars)
        #   3. Whether this is a genuine NEW LINE (prior trend was 3+ bars)
        # ══════════════════════════════════════════════════════════════
        new_line_bars = 0   # How many bars the NEW line has been forming
        prior_trend_bars = 0  # How many bars the PRIOR trend lasted

        if len(ema_vals) >= 5:
            # Count bars in CURRENT direction
            for i in range(2, min(30, len(ema_vals))):
                if ema_direction == "RISING" and ema_vals[-i] > ema_vals[-i - 1]:
                    new_line_bars += 1
                elif ema_direction == "FALLING" and ema_vals[-i] < ema_vals[-i - 1]:
                    new_line_bars += 1
                else:
                    break

            # Now count bars in the PRIOR (opposite) direction
            inflection_idx = 2 + new_line_bars  # Where the direction changed
            if inflection_idx < len(ema_vals) - 1:
                for i in range(inflection_idx, min(inflection_idx + 30, len(ema_vals) - 1)):
                    if ema_direction == "RISING":
                        # Prior trend was FALLING
                        if ema_vals[-i] < ema_vals[-i - 1]:
                            prior_trend_bars += 1
                        else:
                            break
                    else:
                        # Prior trend was RISING
                        if ema_vals[-i] > ema_vals[-i - 1]:
                            prior_trend_bars += 1
                        else:
                            break

        # A genuine NEW LINE = prior trend was 3+ bars, then direction flipped
        is_new_line = prior_trend_bars >= 3 and new_line_bars >= 1
        # FRESH new line = just formed (1-5 bars) — best entry zone
        is_fresh_entry = is_new_line and new_line_bars <= 5

        # EMA crossover: price bar contains the EMA line (price cut through)
        ema_crossed = False
        for i in range(2, min(5, len(highs))):
            h = highs[-i]
            l = lows[-i]
            e = ema_vals[-i] if i <= len(ema_vals) else 0
            if l <= e <= h:
                ema_crossed = True
                break

        # Price momentum filter
        price_falling = False
        price_rising = False
        if len(closes) >= 5:
            c1, c2, c3 = closes[-2], closes[-3], closes[-4]
            if c1 < c2 and c1 < c3:
                price_falling = True
            elif c1 > c2 and c1 > c3:
                price_rising = True

        # ══════════════════════════════════════════════════════════════
        # SIGNAL: Detect NEW EMA LINE formation (the exact pattern from images)
        #
        # CALL (BUY): EMA was falling 3+ bars → EMA starts rising → price above EMA
        #   = "New line formed from DOWN to UP" (Image 1 & 2)
        #
        # PUT (SELL): EMA was rising 3+ bars → EMA starts falling → price below EMA
        #   = "New line formed from UP to Down" (Image 3)
        #
        # Also allow continuation entry if the new line is strong (5+ bars, price on right side)
        # ══════════════════════════════════════════════════════════════
        signal = "NEUTRAL"

        # PRIMARY: New line just formed (best entry — fresh inflection point)
        if is_fresh_entry and ema_direction == "RISING" and price > current_ema:
            if not price_falling:
                signal = "BUY"
        elif is_fresh_entry and ema_direction == "FALLING" and price < current_ema:
            if not price_rising:
                signal = "SELL"

        # SECONDARY: Established new line with crossover (price just cut through EMA)
        elif is_new_line and ema_crossed:
            if ema_direction == "RISING" and price > current_ema and not price_falling:
                signal = "BUY"
            elif ema_direction == "FALLING" and price < current_ema and not price_rising:
                signal = "SELL"

        # TERTIARY: Strong continuation (EMA trending 5+ bars, not overextended)
        elif new_line_bars >= 5 and ema_separation < 3.0:
            if ema_direction == "RISING" and price > current_ema and ema_crossed and not price_falling:
                signal = "BUY"
            elif ema_direction == "FALLING" and price < current_ema and ema_crossed and not price_rising:
                signal = "SELL"

        # Volume trend
        vol_trend = "FLAT"
        if len(volumes) >= 10:
            recent_vol = sum(volumes[-5:]) / 5
            prev_vol = sum(volumes[-10:-5]) / 5
            if prev_vol > 0:
                if recent_vol > prev_vol * 1.2:
                    vol_trend = "RISING"
                elif recent_vol < prev_vol * 0.7:
                    vol_trend = "DECLINING"

        return {
            'timeframe': tf_label,
            'signal': signal,
            'ema_direction': ema_direction,
            'current_ema': current_ema,
            'current_atr': current_atr,
            'atr_pct': round(atr_pct, 4),
            'ema_slope_pct': round(slope_pct, 4),
            'ema_separation_pct': round(ema_separation, 2),
            'trend_bars': new_line_bars,       # Bars of the NEW line
            'prior_trend_bars': prior_trend_bars,  # Bars of the PRIOR trend (before inflection)
            'is_new_line': is_new_line,         # Genuine new line (prior trend 3+ bars)
            'is_fresh_entry': is_fresh_entry,   # Fresh new line (1-5 bars old)
            'vol_trend': vol_trend,
            'price': price,
            'ohlcv': ohlcv,
            'ema_vals': ema_vals,
            'atr_vals': atr_vals,
        }

    def _fetch_tf_ohlcv(self, symbol: str, asset: str, tf: str, limit: int):
        """Fetch OHLCV with caching for higher timeframes to reduce API calls."""
        cache_key = f"{asset}_{tf}"
        ttl = self._tf_cache_ttl.get(tf, 60)
        cached = self._tf_cache.get(cache_key)
        if cached and (time.time() - cached['ts']) < ttl:
            return cached['ohlcv']
        raw = self.price_source.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        ohlcv = PriceFetcher.extract_ohlcv(raw)
        self._tf_cache[cache_key] = {'ohlcv': ohlcv, 'ts': time.time()}

        # Periodic cache cleanup — evict expired entries to prevent memory leak
        if len(self._tf_cache) > 50:
            now = time.time()
            expired = [k for k, v in self._tf_cache.items() if now - v.get('ts', 0) > 600]
            for k in expired:
                del self._tf_cache[k]

        return ohlcv

    # ------------------------------------------------------------------
    # Entry evaluation
    # ------------------------------------------------------------------
    def _evaluate_entry(self, asset: str, price: float, ohlcv: dict,
                        ema_vals: list, atr_vals: list, ema_direction: str,
                        signal: str, closes: list, highs: list, lows: list,
                        opens: list, volumes: list, current_ema: float,
                        current_atr: float, ob_levels: dict = None,
                        htf_1h_direction: str = "FLAT",
                        htf_4h_direction: str = "FLAT",
                        htf_alignment: int = 2,
                        is_reversal_signal: bool = False,
                        mtf_candle_summary: str = "",
                        indicator_context: dict = None,
                        tf_signals: dict = None,
                        active_tf_signals: dict = None,
                        mtf_signal_block: str = "",
                        pairs_signal: dict = None,
                        genetic_vote: int = 0,
                        genetic_count: int = 0):

        ob_levels = ob_levels or {}
        pairs_signal = pairs_signal or {}
        indicator_context = indicator_context or {}
        tf_signals = tf_signals or {}
        active_tf_signals = active_tf_signals or {}

        # Store tf_signals for use in later gates (e.g., SHORT direction gate)
        if not hasattr(self, '_last_tf_signals'):
            self._last_tf_signals = {}
        self._last_tf_signals[asset] = tf_signals

        # ── ROBINHOOD-HARDENING INTERVENTIONS (A+B+C) ──
        # Gated by ACT_ROBINHOOD_HARDEN=1. Runs BEFORE LLM/ML so we don't waste
        # expensive calls on trades the conviction gate would reject anyway.
        #
        # (A) Tiered move threshold — signals with expected_move < normal_min
        #     are rejected, between normal_min and sniper_min are 'normal' tier,
        #     >= sniper_min are 'sniper' tier. Size multiplier applied at sizing.
        # (B) Macro bias — EconomicIntelligence's 12-layer summary becomes a
        #     signed bias + size multiplier, applied at sizing time.
        # (C) Conviction gate — TF + Hurst + multi-strategy + macro must all
        #     align for 'sniper'; a looser subset for 'normal'; else reject.
        self._last_conviction_tier = getattr(self, '_last_conviction_tier', {})
        self._last_macro_bias = getattr(self, '_last_macro_bias', {})
        _harden_enabled = (os.environ.get('ACT_ROBINHOOD_HARDEN') or '').strip().lower() in ('1', 'true', 'yes', 'on')
        if _harden_enabled and signal in ('BUY', 'SELL'):
            try:
                from src.trading.conviction_gate import evaluate as _conv_evaluate
                from src.trading.macro_bias import compute_macro_bias

                # Macro summary (already fetched elsewhere in the cycle; read fresh)
                _summary = None
                if self._economic_intelligence is not None:
                    try:
                        _summary = self._economic_intelligence.get_macro_summary()
                    except Exception:
                        _summary = None
                _macro = compute_macro_bias(_summary)

                # Multi-strategy counts from the _process_asset call that just ran
                _md = getattr(self, '_last_multi_details', {}).get(asset, {})
                _ms_long = sum(1 for k, v in _md.items()
                               if not k.startswith('_') and isinstance(v, dict)
                               and v.get('signal_word') == 'LONG')
                _ms_short = sum(1 for k, v in _md.items()
                                if not k.startswith('_') and isinstance(v, dict)
                                and v.get('signal_word') == 'SHORT')
                _ms_flat = sum(1 for k, v in _md.items()
                               if not k.startswith('_') and isinstance(v, dict)
                               and v.get('signal_word') == 'FLAT')

                # Hurst regime — float stored in self._last_hurst[asset]
                _h = getattr(self, '_last_hurst', {}).get(asset, 0.5)
                _regime = 'trending' if _h > 0.55 else ('mean_reverting' if _h < 0.45 else 'random')

                _direction = 'LONG' if signal == 'BUY' else 'SHORT'
                _conv = _conv_evaluate(
                    direction=_direction,
                    tf_1h_direction=htf_1h_direction,
                    tf_4h_direction=htf_4h_direction,
                    hurst_regime=_regime,
                    multi_strategy_counts={'long': _ms_long, 'short': _ms_short, 'flat': _ms_flat},
                    macro_bias=_macro,
                )
                self._last_conviction_tier[asset] = _conv.tier
                self._last_macro_bias[asset] = _macro

                if not _conv.passed:
                    # Log the conviction-tier rejection reason and skip. This is the
                    # intervention working as designed — low-conviction trades are
                    # rejected before they eat spread.
                    print(f"  [{self._ex_tag}:{asset}] CONVICTION REJECT: "
                          f"tier=reject  reasons={_conv.reasons}")
                    return

                print(f"  [{self._ex_tag}:{asset}] CONVICTION PASS: tier={_conv.tier} "
                      f"size_mult={_conv.size_multiplier}  macro_bias={_macro.signed_bias:+.2f}"
                      f"{'  (crisis)' if _macro.crisis else ''}")
            except Exception as _hardexc:
                logger.debug(f"[HARDEN] conviction gate failed: {_hardexc}")

        # Need at least one active signal on a configured timeframe
        if not active_tf_signals:
            return

        # ══════════════════════════════════════════════════════════
        # SNIPER MODE: Multi-confluence gate — wait for PERFECT setups
        # Counts how many independent signals agree before allowing entry
        # This is the PRIMARY quality filter for Robinhood's 3.3% spread
        # ══════════════════════════════════════════════════════════
        if self.sniper_enabled:
            self.sniper_stats['signals_seen'] += 1
            confluence_count = 0
            confluence_reasons = []

            # 1. Multiple timeframes agree (each active TF with same direction = +1)
            _sniper_direction = signal  # BUY or SELL from 5m anchor
            for tf, sig in active_tf_signals.items():
                tf_sig = sig.get('signal', 'NEUTRAL')
                if tf_sig != 'NEUTRAL':
                    if (tf_sig == 'BUY' and _sniper_direction == 'BUY') or \
                       (tf_sig == 'SELL' and _sniper_direction == 'SELL'):
                        confluence_count += 1
                        confluence_reasons.append(f"{tf}={tf_sig}")

            # 2. HTF alignment bonus (+1 if 4h agrees, +1 if 1d agrees)
            _htf_4h = tf_signals.get('4h', {}).get('ema_direction', 'FLAT')
            _htf_1d = tf_signals.get('1d', {}).get('ema_direction', 'FLAT')
            if (_sniper_direction == 'BUY' and _htf_4h == 'RISING') or \
               (_sniper_direction == 'SELL' and _htf_4h == 'FALLING'):
                confluence_count += 1
                confluence_reasons.append(f"4h_ema={_htf_4h}")
            if (_sniper_direction == 'BUY' and _htf_1d == 'RISING') or \
               (_sniper_direction == 'SELL' and _htf_1d == 'FALLING'):
                confluence_count += 1
                confluence_reasons.append(f"1d_ema={_htf_1d}")

            # 3. Volume confirmation (+1 if volume trending up SIGNIFICANTLY)
            # Old: 20% jump fired on noise. New: 50% jump = real conviction
            if len(volumes) >= 10:
                _vol_avg_recent = sum(volumes[-3:]) / 3
                _vol_avg_prior = sum(volumes[-10:-3]) / 7 if len(volumes) >= 10 else _vol_avg_recent
                if _vol_avg_prior > 0 and _vol_avg_recent / _vol_avg_prior > 1.5:
                    confluence_count += 1
                    confluence_reasons.append(f"vol_up={_vol_avg_recent/_vol_avg_prior:.1f}x")

            # 4. EMA slope acceleration (+1 if slope increasing AND strong)
            # Old: 0.05% threshold = any random candle. New: 0.15% = real momentum
            if len(ema_vals) >= 5:
                _slope_now = abs(ema_vals[-1] - ema_vals[-2]) / ema_vals[-2] * 100 if ema_vals[-2] > 0 else 0
                _slope_prev = abs(ema_vals[-3] - ema_vals[-4]) / ema_vals[-4] * 100 if ema_vals[-4] > 0 else 0
                if _slope_now > _slope_prev * 1.5 and _slope_now > 0.15:
                    confluence_count += 1
                    confluence_reasons.append(f"accel={_slope_now:.3f}%")

            # 5. Expected move vs spread cost (+1 if move > 2× spread)
            _sniper_atr = current_atr
            if hasattr(self, '_last_chosen_tf_atr') and self._last_chosen_tf_atr.get(asset, 0) > 0:
                _sniper_atr = self._last_chosen_tf_atr[asset]
            _atr_tp = self.config.get('risk', {}).get('atr_tp_mult', 10.0)
            _expected_move = (_sniper_atr * _atr_tp / price) * 100 if price > 0 and _sniper_atr > 0 else 0
            if _expected_move >= self.sniper_min_expected_move_pct:
                confluence_count += 1
                confluence_reasons.append(f"move={_expected_move:.1f}%")

            # 6. Trend structure: higher highs (LONG) or lower lows (SHORT)
            # Proves the trend is structurally intact, not just EMA noise
            if len(closes) >= 20:
                _recent_high = max(closes[-5:])
                _prior_high = max(closes[-15:-5])
                _recent_low = min(closes[-5:])
                _prior_low = min(closes[-15:-5])
                if _sniper_direction == 'BUY' and _recent_high > _prior_high and _recent_low > _prior_low:
                    confluence_count += 1
                    confluence_reasons.append("HH+HL")
                elif _sniper_direction == 'SELL' and _recent_low < _prior_low and _recent_high < _prior_high:
                    confluence_count += 1
                    confluence_reasons.append("LL+LH")

            # 7. Price above both EMA(8) for LONG / below for SHORT (+1)
            # Confirms price has momentum, not just EMA slope
            if len(ema_vals) >= 2:
                _ema_now = ema_vals[-2]
                if _sniper_direction == 'BUY' and closes[-1] > _ema_now and closes[-2] > _ema_now:
                    confluence_count += 1
                    confluence_reasons.append("price>EMA")
                elif _sniper_direction == 'SELL' and closes[-1] < _ema_now and closes[-2] < _ema_now:
                    confluence_count += 1
                    confluence_reasons.append("price<EMA")

            # Initialize math_filter_warnings early (before sniper advisory needs it)
            if 'math_filter_warnings' not in dir():
                math_filter_warnings = []

            # SNIPER: Advisory, NOT a hard gate — LLM and agents make final decision
            # This ensures the LLM always sees market data and can learn even from skipped setups
            if confluence_count < self.sniper_min_confluence:
                self.sniper_stats['filtered'] += 1
                print(f"  [{self._ex_tag}:{asset}] SNIPER ADVISORY: confluence {confluence_count}/{self.sniper_min_confluence} — {', '.join(confluence_reasons) or 'none'} — LOW (LLM will decide)")
                math_filter_warnings.append(f"SNIPER: only {confluence_count}/{self.sniper_min_confluence} confluence — weak setup")
                # DON'T return — let LLM + agents see the data and decide
            else:
                print(f"  [{self._ex_tag}:{asset}] SNIPER PASS: confluence {confluence_count}/{self.sniper_min_confluence} — {', '.join(confluence_reasons)}")

            # Store for RL state (accessible later in evaluation)
            if not hasattr(self, '_last_sniper_confluence'):
                self._last_sniper_confluence = {}
            self._last_sniper_confluence[asset] = confluence_count

        # Initialize math filter warnings list (used throughout evaluation)
        math_filter_warnings = []

        # ── TIME-OF-DAY CONTEXT (informational, not a block) ──
        # Data analysis: 06:00-17:00 UTC = profitable, overnight = losing
        # Passed to LLM as context but NOT blocking — user wants 24/7 trading
        import datetime
        utc_hour = datetime.datetime.utcnow().hour
        if utc_hour < 6 or utc_hour >= 18:
            math_filter_warnings.append(f"TIME: overnight session (UTC {utc_hour}:00) -- historically weaker")

        # ── DANGEROUS HOURS: DISABLED for crypto (24/7 market) ──
        # v13 backtest finding: blocking hours caused artificial trade clustering at UTC 04
        # (all pent-up signals fire when block lifts → 17% of trades in worst hour)
        # Removing the filter spread trades evenly and improved PF from 0.97 → 1.01
        dangerous_hours = self.config.get('filters', {}).get('dangerous_hours', [])  # Empty = no block
        if dangerous_hours and utc_hour in dangerous_hours:
            print(f"  [{self._ex_tag}:{asset}] HOUR BLOCK: UTC {utc_hour}:00 is a high-loss hour")
            return

        # Flag for position sizing: reduce size during off-hours
        self._is_profitable_hour = utc_hour in [6, 7, 8, 9, 10, 12, 13, 16]

        # ══════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME TREND ALIGNMENT FILTER
        # For trend-following: block if 1h AND 4h both disagree (alignment < 2)
        # For reversals: INVERTED — we WANT HTFs to disagree (going against trend)
        # ══════════════════════════════════════════════════════════
        # HTF alignment — passed as context to LLM brain (NOT a hard block)
        # The LLM sees all timeframes and decides if misalignment matters
        if is_reversal_signal:
            if htf_alignment >= 3:
                print(f"  [{self._ex_tag}:{asset}] HTF NOTE: all TFs agree — reversal less likely (LLM decides)")
        else:
            if htf_alignment < 2:
                print(f"  [{self._ex_tag}:{asset}] HTF WARNING: 1h={htf_1h_direction} 4h={htf_4h_direction} vs 5m={ema_direction} — alignment={htf_alignment}/3 (LLM decides)")

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

        # ══════════════════════════════════════════════════════════
        # 5-8. ML-DRIVEN ENTRY SCORE BOOST (uses indicator_context + ml_context)
        # Indicators confirm direction → higher score → LLM gets cleaner setup
        # ══════════════════════════════════════════════════════════
        try:
            # 5. RSI CONFIRMATION (0-2 points)
            # RSI below 40 confirms SHORT, above 60 confirms LONG
            _rsi = indicator_context.get('rsi', 50)
            if signal == "BUY" and _rsi > 55 and _rsi < 75:
                entry_score += 2
                score_reasons.append(f"rsi_bull={_rsi:.0f}")
            elif signal == "SELL" and _rsi < 45 and _rsi > 25:
                entry_score += 2
                score_reasons.append(f"rsi_bear={_rsi:.0f}")
            elif (signal == "BUY" and _rsi < 30) or (signal == "SELL" and _rsi > 70):
                entry_score -= 1  # Overbought/oversold against direction
                score_reasons.append(f"rsi_against={_rsi:.0f}")

            # 6. ADX TREND STRENGTH (0-2 points)
            # ADX > 25 = trending market (our strategy works), < 20 = choppy (fails)
            _adx = indicator_context.get('adx', 20)
            if _adx > 30:
                entry_score += 2
                score_reasons.append(f"adx_strong={_adx:.0f}")
            elif _adx > 25:
                entry_score += 1
                score_reasons.append(f"adx_trend={_adx:.0f}")
            elif _adx < 15:
                entry_score -= 1
                score_reasons.append(f"adx_weak={_adx:.0f}")

            # 7. MACD ALIGNMENT (0-2 points)
            # MACD histogram agrees with signal direction
            _macd_cross = indicator_context.get('macd_cross', 'NONE')
            _macd_hist = indicator_context.get('macd_hist', 0)
            if signal == "BUY" and (_macd_cross == 'BULLISH' or _macd_hist > 0):
                entry_score += 2
                score_reasons.append("macd_bull")
            elif signal == "SELL" and (_macd_cross == 'BEARISH' or _macd_hist < 0):
                entry_score += 2
                score_reasons.append("macd_bear")
            elif (signal == "BUY" and _macd_hist < 0) or (signal == "SELL" and _macd_hist > 0):
                entry_score -= 1
                score_reasons.append("macd_against")

            # 8. CHOPPINESS + MONEY FLOW (0-2 points)
            _chop = indicator_context.get('choppiness', 50)
            _mflow = indicator_context.get('money_flow', '')
            if _chop < 50:
                entry_score += 1
                score_reasons.append(f"chop_trending={_chop:.0f}")
            elif _chop > 65:
                entry_score -= 1
                score_reasons.append(f"chop_ranging={_chop:.0f}")
            if (signal == "BUY" and _mflow == 'INFLOW') or (signal == "SELL" and _mflow == 'OUTFLOW'):
                entry_score += 1
                score_reasons.append(f"flow_{_mflow}")

            # 9. OBV VOLUME CONFIRMATION (0-1 point)
            _obv = indicator_context.get('obv_trend', '')
            if (signal == "BUY" and _obv == 'RISING') or (signal == "SELL" and _obv == 'FALLING'):
                entry_score += 1
                score_reasons.append(f"obv_{_obv}")

            # 10. HORIZONTAL S/R LEVELS (v14: helps ETH, neutral for BTC)
            # Per-asset: enabled for ETH (PF +0.02), disabled for BTC (PF -0.01)
            sr_assets = self.config.get('adaptive', {}).get('sr_assets', ['ETH'])
            if asset in sr_assets:
                try:
                    from src.indicators.trendlines import get_sr_score_adjustment
                    sr_ctx = get_sr_score_adjustment(highs, lows, closes, signal, lookback=100)
                    sr_adj = sr_ctx.get('sr_score_adj', 0)
                    if sr_adj != 0:
                        entry_score += sr_adj
                        sr_detail = sr_ctx.get('sr_details', f'sr={sr_adj}')
                        score_reasons.append(sr_detail)
                except Exception:
                    pass

        except Exception:
            pass  # Don't block if indicator boosting fails

        # ── GENETIC EVOLVED STRATEGIES SCORE BOOST (±1 point) ──
        if genetic_count >= 2:
            if signal == "BUY" and genetic_vote > 0:
                entry_score += 1
                score_reasons.append(f"genetic_agree({genetic_count}v,+{genetic_vote})")
            elif signal == "BUY" and genetic_vote < 0:
                entry_score -= 1
                score_reasons.append(f"genetic_disagree({genetic_count}v,{genetic_vote})")
            elif signal == "SELL" and genetic_vote < 0:
                entry_score += 1
                score_reasons.append(f"genetic_agree({genetic_count}v,{genetic_vote})")
            elif signal == "SELL" and genetic_vote > 0:
                entry_score -= 1
                score_reasons.append(f"genetic_disagree({genetic_count}v,+{genetic_vote})")

        # ── RANGE DETECTION — computed for LLM, NOT a hard block ──
        range_pct = 0
        atr_pct = 0
        atr_range_ratio = 0
        ema_has_momentum = False
        is_ranging = False
        if len(closes) >= 20:
            range_high = max(closes[-20:])
            range_low = min(closes[-20:])
            range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0
            atr_pct = (current_atr / price * 100) if price > 0 else 0
            atr_range_ratio = atr_pct / range_pct if range_pct > 0 else 0

            if len(ema_vals) >= 5:
                ema_slope_check = abs(ema_vals[-1] - ema_vals[-3]) / ema_vals[-3] * 100 if ema_vals[-3] > 0 else 0
                if ema_slope_check > 0.03:
                    ema_has_momentum = True

            if range_pct > 0 and atr_range_ratio > 0.5 and range_pct < 1.5 and not ema_has_momentum:
                is_ranging = True

        ob_imbalance = ob_levels.get('imbalance', 0)

        # Count consecutive trend bars — CONFIRMED candles only
        min_trend_bars = 0
        for i in range(len(ema_vals)-2, max(0, len(ema_vals)-12), -1):
            if i > 0:
                if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]:
                    min_trend_bars += 1
                elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]:
                    min_trend_bars += 1
                else:
                    break

        # EMA slope direction
        slope_pct = 0
        slope_conflicts = False
        if len(ema_vals) >= 4:
            slope_pct = (ema_vals[-1] - ema_vals[-4]) / ema_vals[-4] * 100 if ema_vals[-4] > 0 else 0
            if signal == "BUY" and slope_pct < -0.01:
                slope_conflicts = True
            elif signal == "SELL" and slope_pct > 0.01:
                slope_conflicts = True

        # EMA separation (overextension)
        ema_separation = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
        is_overextended = ema_separation > 10.0

        # ── Math context for LLM (informational — LLM decides what matters) ──
        # Only warn on genuinely concerning conditions, not routine
        if entry_score <= 2:
            math_filter_warnings.append(f"LOW score {entry_score}/10 ({', '.join(score_reasons) or 'no momentum'})")
        if slope_conflicts:
            direction = "CALL" if signal == "BUY" else "PUT"
            math_filter_warnings.append(f"SLOPE NOTE: {direction} but slope={slope_pct:+.3f}%")
        if is_ranging and entry_score < 4:
            math_filter_warnings.append(f"RANGING: no EMA momentum")
        if is_overextended:
            math_filter_warnings.append(f"EXTENDED: {ema_separation:.1f}% from EMA")

        # HTF alignment — only note if fully against
        if htf_alignment < 2:
            math_filter_warnings.append(f"HTF: 1h={htf_1h_direction} 4h={htf_4h_direction} vs 5m={ema_direction}")

        # Log what the LLM will see
        if math_filter_warnings:
            print(f"  [{self._ex_tag}:{asset}] WARNINGS (LLM decides): {' | '.join(math_filter_warnings)}")
        else:
            quality = "EXCELLENT" if entry_score >= 9 else "STRONG" if entry_score >= 7 else "OK"
            print(f"  [{self._ex_tag}:{asset}] {quality} PATTERN: score={entry_score}/10 ({', '.join(score_reasons)}) trend={min_trend_bars}bars")

        # Entry quality context already captured above — no duplicate warnings needed

        # CRITICAL: Check EXCHANGE positions (not just internal dict)
        # CHECK OPEN ORDERS — if we already have a pending limit order, cancel stale ones
        try:
            if self._exchange_client:
                symbol = self._get_symbol(asset)
                open_orders = self._exchange_client.exchange.fetch_open_orders(symbol)
                if open_orders:
                    # Auto-cancel orders older than 2 minutes (stale limit orders)
                    now_ms = time.time() * 1000
                    stale = []
                    fresh = []
                    for o in open_orders:
                        created = o.get('timestamp', 0) or 0
                        age_s = (now_ms - created) / 1000 if created > 0 else 999
                        if age_s > 30:  # older than 30s
                            stale.append(o)
                        else:
                            fresh.append(o)
                    # Cancel stale orders
                    for o in stale:
                        try:
                            self._exchange_client.exchange.cancel_order(o['id'], symbol)
                            print(f"  [{self._ex_tag}:{asset}] CANCELLED stale order {o['id']} (age={int((now_ms - (o.get('timestamp',0) or 0))/1000)}s)")
                        except Exception as ce:
                            print(f"  [{self._ex_tag}:{asset}] Cancel failed: {ce}")
                    if fresh:
                        print(f"  [{self._ex_tag}:{asset}] SKIP: {len(fresh)} recent pending order(s)")
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
                        raw_entry = p.get('avg_entry_price', None)
                        entry_p = float(raw_entry) if raw_entry and float(raw_entry) > 0 else None

                        # SAFETY: Validate entry price is sane
                        # If entry price is missing or wildly different from current price, skip sync
                        if entry_p is None:
                            print(f"  [{self._ex_tag}:{asset}] SYNC SKIP: no avg_entry_price from exchange — can't calculate safe SL")
                            return
                        # Reject if entry is >30% away from current price (data artifact)
                        entry_deviation = abs(entry_p - price) / price * 100 if price > 0 else 999
                        if entry_deviation > 30:
                            print(f"  [{self._ex_tag}:{asset}] SYNC SKIP: entry ${entry_p:,.0f} is {entry_deviation:.0f}% from current ${price:,.0f} — bad data")
                            return

                        synced_direction = 'LONG' if pos_side == 'long' else 'SHORT'

                        # Check if signal is OPPOSITE to current position
                        # e.g., holding SHORT but signal is BUY — should close and flip
                        signal_conflicts = (
                            (synced_direction == 'SHORT' and signal == 'BUY') or
                            (synced_direction == 'LONG' and signal == 'SELL')
                        )

                        if signal_conflicts and asset not in self.failed_close_assets:
                            # ANTI-CHURN: Don't flip if position was entered recently
                            # Rapid flipping (LONG->SHORT->LONG) in ranges = death by fees
                            pos_entry_time = self.positions.get(asset, {}).get('entry_time', 0)
                            pos_age_min = (time.time() - pos_entry_time) / 60 if pos_entry_time > 0 else 999
                            if pos_age_min < 10:
                                # Position is too young to flip — let trailing SL handle exit
                                print(f"  [{self._ex_tag}:{asset}] FLIP BLOCKED: position only {pos_age_min:.0f}min old (need 10min). Let SL handle it.")
                                return

                            flip_dir = 'CALL' if signal == 'BUY' else 'PUT'
                            print(f"  [{self._ex_tag}:{asset}] WRONG SIDE: holding {synced_direction} but signal={signal} — closing to flip to {flip_dir}")
                            close_side = 'sell' if synced_direction == 'LONG' else 'buy'
                            # Use limit at best bid/ask for better fill
                            flip_price = None
                            try:
                                flip_ob = self.price_source.fetch_order_book(self._get_symbol(asset), limit=5)
                                if close_side == 'sell' and flip_ob.get('bids'):
                                    flip_price = float(flip_ob['bids'][0][0])
                                elif close_side == 'buy' and flip_ob.get('asks'):
                                    flip_price = float(flip_ob['asks'][0][0])
                            except Exception:
                                pass
                            result = self._api_call(
                                self.price_source.place_order,
                                symbol=self._get_symbol(asset),
                                side=close_side,
                                amount=contracts,
                                order_type='limit' if flip_price else 'market',
                                price=flip_price,
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
                                # Set close time to enforce post-close cooldown before re-entry
                                self.last_close_time[asset] = time.time()
                                if asset in self.positions:
                                    del self.positions[asset]
                                # After a flip close, DON'T immediately re-enter
                                # The post-close cooldown will prevent re-entry for 10 min
                                return
                        else:
                            # Same direction or no signal — sync it and manage
                            if asset not in self.positions:
                                # SAFETY: Check if synced position is too large for our account
                                # Max position: 5% of equity (or 20% for small accounts)
                                max_pct = 20 if self.equity < 500 else 5
                                if self._exchange_name == 'delta':
                                    cs = {'BTC': 0.001, 'ETH': 0.01}.get(asset, 0.001)
                                    pos_notional = contracts * cs * entry_p
                                else:
                                    pos_notional = contracts * entry_p
                                max_notional = self.equity * (max_pct / 100)
                                if pos_notional > max_notional * 3:
                                    # Position is >3x our max — likely a leftover from manual trading
                                    print(f"  [{self._ex_tag}:{asset}] WARNING: exchange pos ${pos_notional:,.0f} >> max ${max_notional:,.0f} — NOT syncing (close manually)")
                                    return

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
                                    'peak_price': entry_p, 'entry_time': time.time(),
                                    'confidence': 0, 'reasoning': 'synced from exchange',
                                    'predicted_l_level': '?', 'risk_score': 0,
                                    'bear_risk': 0, 'hurst': 0.5, 'hurst_regime': 'unknown',
                                    'breakeven_moved': False,
                                }
                                print(f"  [{self._ex_tag}:{asset}] SYNCED {pos_side} position ({contracts}) ${pos_notional:,.0f} | SL=${sync_sl:,.2f}")
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
                return  # Same candle — wait for next
            self.last_signal_candle[asset] = confirmed_candle_ts

        now = time.time()

        # ── POST-CLOSE COOLDOWN (HARD BLOCK) ──
        # After closing ANY trade, wait before re-entering to avoid churn
        if asset in self.last_close_time:
            since_close = now - self.last_close_time[asset]
            if since_close < self.post_close_cooldown:
                remaining = int(self.post_close_cooldown - since_close)
                if remaining % 60 < 10:  # Print once per minute to reduce spam
                    print(f"  [{self._ex_tag}:{asset}] COOLDOWN: {remaining}s left after last close")
                return

        # ── MINIMUM TIME BETWEEN TRADES (HARD BLOCK) ──
        if asset in self.last_trade_time:
            since_trade = now - self.last_trade_time[asset]
            if since_trade < self.trade_cooldown:
                return

        # ── LOSS STREAK COOLDOWN (HARD BLOCK) ──
        # Was soft (LLM context only) — now enforced. LLM kept ignoring it.
        cooldown_until = self.asset_cooldown_until.get(asset, 0)
        if now < cooldown_until:
            streak = self.asset_loss_streak.get(asset, 0)
            remaining_min = int((cooldown_until - now) / 60)
            if remaining_min % 5 == 0:  # Print every 5 min
                print(f"  [{self._ex_tag}:{asset}] STREAK BLOCK: {streak} losses, {remaining_min}min left")
            return

        # ── RANGE DETECTION (HARD BLOCK) ──
        # If price has been in a tight band (<0.5%) for last 10 candles, market is ranging
        # EMA(8) oscillates in ranges causing rapid BUY/SELL flip = churn
        if len(closes) >= 12:
            last_12 = closes[-12:-2]  # 10 confirmed candles
            if len(last_12) >= 10:
                range_high = max(last_12)
                range_low = min(last_12)
                range_pct = (range_high - range_low) / range_low * 100 if range_low > 0 else 0
                if range_pct < 0.15:
                    print(f"  [{self._ex_tag}:{asset}] RANGING: {range_pct:.2f}% range over 10 candles — sitting out")
                    return

        # ── HURST REGIME GATE ──
        # If market is random walk or mean-reverting, EMA trend signals get stopped at L1
        # Only block when confidence is HIGH that regime is anti-trend (R^2 > 0.7)
        hurst_regime = 'unknown'
        hurst_value = 0.5
        if self._hurst and len(closes) >= 50:
            try:
                import numpy as _np
                hurst_result = self._hurst.compute(_np.array(closes), window=min(200, len(closes)))
                hurst_value = hurst_result['hurst']
                hurst_regime = hurst_result['regime']
                hurst_conf = hurst_result['r_squared']

                if hurst_regime == 'mean_reverting' and hurst_conf > 0.7:
                    # Market is mean-reverting — TREND signals unreliable, but
                    # mean-reversion/grid/VWAP strategies SHOULD trade here
                    # Old behavior: hard return (blocked everything)
                    # New: add warning for LLM, let multi-strategy handle regime
                    math_filter_warnings.append(f"HURST: H={hurst_value:.2f} MEAN-REVERTING — trend strategies unreliable, favor grid/mean-reversion/VWAP")
                    print(f"  [{self._ex_tag}:{asset}] HURST NOTE: H={hurst_value:.2f} ({hurst_regime}) R2={hurst_conf:.2f} — mean-reversion strategies favored")
                elif hurst_regime == 'random' and hurst_conf > 0.7:
                    # Random walk — add warning but don't block (weaker signal)
                    math_filter_warnings.append(f"HURST: H={hurst_value:.2f} random walk — trend may not persist")
            except Exception as e:
                logger.debug(f"Hurst computation error: {e}")

        # ── VPIN TOXIC FLOW CHECK ──
        # If order flow is one-sided (informed traders), our entries get front-run
        vpin_status = None
        if asset in self._vpin_guards:
            try:
                vpin_status = self._vpin_guards[asset].is_flow_toxic()
                if vpin_status['is_toxic']:
                    math_filter_warnings.append(f"VPIN: {vpin_status['vpin']:.2f} TOXIC flow — informed traders detected")
                elif vpin_status['risk_action'] == 'REDUCE':
                    math_filter_warnings.append(f"VPIN: {vpin_status['vpin']:.2f} elevated — watch for adverse selection")
                # Push VPIN to dashboard
                try:
                    DashboardState().add_layer_log('L1', f"VPIN updated: {vpin_status['vpin']:.4f} ({vpin_status.get('risk_action', 'OK')})", "info")
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"VPIN check error: {e}")

        # ── ML MODEL PREDICTIONS ──
        ml_context = {}

        # HMM Regime Detection
        if self._hmm and len(closes) >= 50:
            try:
                import numpy as _np
                log_ret = _np.diff(_np.log(_np.array(closes[-100:]) + 1e-12))
                vol_20 = _np.array([_np.std(log_ret[max(0,i-20):i]) for i in range(1, len(log_ret)+1)])
                vol_change = _np.diff(volumes[-100:]) / (_np.array(volumes[-101:-1]) + 1e-12) if len(volumes) >= 101 else _np.zeros(len(log_ret))
                if len(vol_change) > len(log_ret):
                    vol_change = vol_change[-len(log_ret):]
                elif len(vol_change) < len(log_ret):
                    vol_change = _np.pad(vol_change, (len(log_ret)-len(vol_change), 0))

                min_len = min(len(log_ret), len(vol_20), len(vol_change))
                obs = _np.column_stack([log_ret[-min_len:], vol_20[-min_len:], vol_change[-min_len:]])

                # Try predict() first (new API), fall back to detect() (old API)
                if hasattr(self._hmm, 'predict') and self._hmm.is_fitted:
                    regime_result = self._hmm.predict(log_ret[-min_len:], vol_20[-min_len:], vol_change[-min_len:])
                elif hasattr(self._hmm, 'detect'):
                    regime_result = self._hmm.detect(obs)
                else:
                    regime_result = None

                if regime_result:
                    ml_context['hmm_regime'] = regime_result.get('regime', 'UNKNOWN').upper()
                    ml_context['hmm_confidence'] = round(regime_result.get('stability', 0), 2)
                    crisis_prob = regime_result.get('crisis_prob', 0)
                    ml_context['crisis_probability'] = round(crisis_prob, 3)

                    # Store for multi-strategy engine
                    if not hasattr(self, '_last_hmm_regime'):
                        self._last_hmm_regime = {}
                    self._last_hmm_regime[asset] = ml_context['hmm_regime']

                    # ── REGIME TRANSITION PREDICTION ──
                    if hasattr(self._hmm, 'predict_transition') and self._hmm.is_fitted:
                        try:
                            transition = self._hmm.predict_transition(
                                log_ret[-min_len:], vol_20[-min_len:], vol_change[-min_len:],
                                horizon_bars=6,
                            )
                            ml_context['regime_transition'] = transition
                            ml_context['regime_strategy_bias'] = transition.get('strategy_bias', 'HOLD_CURRENT')

                            # Log transition prediction
                            if transition.get('is_about_to_change'):
                                print(f"  [{self._ex_tag}:{asset}] REGIME SHIFT: {transition['current_regime']}→{transition['next_probable_regime']} "
                                      f"({transition['next_regime_probability']:.0%} in {transition['horizon_bars']} bars) | "
                                      f"Bias: {transition['strategy_bias']}")
                            else:
                                persistence = transition.get('persistence_score', 0)
                                if persistence < 0.5:
                                    print(f"  [{self._ex_tag}:{asset}] REGIME WEAK: {transition['current_regime']} persistence={persistence:.2f} | {transition['bias_reason']}")
                        except Exception:
                            pass

                    # CRISIS: advisory for LLM (not hard block — let LLM decide)
                    if ml_context['hmm_regime'] == 'CRISIS' and crisis_prob > 0.5:
                        math_filter_warnings.append(f"HMM CRISIS: prob={crisis_prob:.2f} — extreme caution")
                        print(f"  [{self._ex_tag}:{asset}] HMM CRISIS WARNING: {ml_context['hmm_regime']} (crisis_prob={crisis_prob:.2f}) — LLM informed")
            except Exception as e:
                logger.debug(f"HMM regime error: {e}")

        # Kalman Trend Filter
        if self._kalman and len(closes) >= 30:
            try:
                import numpy as _np
                kalman_result = self._kalman.filter(_np.array(closes[-100:]))
                if kalman_result:
                    ml_context['kalman_slope'] = round(kalman_result.get('slope', 0), 4)
                    ml_context['kalman_snr'] = round(kalman_result.get('snr', 0), 2)
                    slope_dir = 'UP' if kalman_result.get('slope', 0) > 0 else 'DOWN'
                    ml_context['kalman_trend'] = slope_dir
            except Exception as e:
                logger.debug(f"Kalman filter error: {e}")

        # Volatility Regime — dual detection: basic + advanced detector
        if VOLATILITY_MODEL_AVAILABLE and len(closes) >= 30:
            try:
                import numpy as _np
                returns = log_returns(_np.array(closes[-100:]))
                vol_regime = classify_volatility_regime(returns)
                if vol_regime:
                    ml_context['vol_regime'] = str(vol_regime.get('regime', 'NORMAL'))
                    ml_context['vol_percentile'] = round(vol_regime.get('percentile', 50), 0)
            except Exception as e:
                logger.debug(f"Volatility regime error: {e}")

        # ── Advanced Volatility Regime (ATR + clustering + z-score) ──
        if self._vol_regime_detector and len(closes) >= 50:
            try:
                vrd_result = self._vol_regime_detector.detect_regime(closes, highs, lows)
                ml_context['vol_regime_adv'] = vrd_result.get('vol_regime', 'NORMAL')
                ml_context['vol_cluster_score'] = round(vrd_result.get('vol_cluster_score', 1.0), 2)
                ml_context['atr_z_score'] = round(vrd_result.get('atr_z_score', 0.0), 2)
                ml_context['realized_vol_annual'] = round(vrd_result.get('realized_vol_annual', 0.0), 4)

                # HIGH_VOL_PANIC: reduce entry score (EMA signals are unreliable in panic)
                if vrd_result.get('vol_regime') == 'HIGH_VOL_PANIC':
                    entry_score -= 2
                    score_reasons.append("vol_PANIC")
                    math_filter_warnings.append(f"VolRegime: HIGH_VOL_PANIC (cluster={vrd_result['vol_cluster_score']:.1f}x, ATR z={vrd_result['atr_z_score']:.1f})")
                # TREND_EXPANSION: boost entry score (EMA thrives in expanding trends)
                elif vrd_result.get('vol_regime') == 'TREND_EXPANSION':
                    entry_score += 1
                    score_reasons.append("vol_EXPANSION")
                # LOW_VOL_RANGE: penalize (EMA chops in ranges)
                elif vrd_result.get('vol_regime') == 'LOW_VOL_RANGE':
                    entry_score -= 1
                    score_reasons.append("vol_LOW_RANGE")
                    math_filter_warnings.append("VolRegime: LOW_VOL_RANGE — EMA signals chop in tight ranges")
            except Exception as e:
                logger.debug(f"Advanced volatility regime error: {e}")

        # Cycle Detection (primary: detect_dominant_cycles, secondary: FFT rolling)
        if CYCLE_AVAILABLE and len(closes) >= 50:
            try:
                import numpy as _np
                cycle_result = detect_dominant_cycles(_np.array(closes[-200:]))
                if cycle_result:
                    ml_context['dominant_cycle'] = cycle_result.get('period', 0)
                    ml_context['cycle_phase'] = cycle_result.get('phase', 'UNKNOWN')
            except Exception as e:
                logger.debug(f"Cycle detection error: {e}")

        # ── FFT Cycle Period (complementary — precise period estimation) ──
        if FFT_CYCLE_AVAILABLE and len(closes) >= 128:
            try:
                fft_periods = rolling_fft_period(closes[-200:], window=128, top_k=1)
                # Use the latest valid period
                latest_fft = None
                for p in reversed(fft_periods):
                    if p is not None and 4 < p < 100:  # Sane range: 4-100 bars
                        latest_fft = p
                        break
                if latest_fft:
                    ml_context['fft_cycle_period'] = round(latest_fft, 1)
                    # Cross-validate: if FFT and primary cycle detector agree, boost confidence
                    primary_cycle = ml_context.get('dominant_cycle', 0)
                    if primary_cycle > 0 and abs(latest_fft - primary_cycle) < primary_cycle * 0.3:
                        ml_context['cycle_agreement'] = True
                    else:
                        ml_context['cycle_agreement'] = False
            except Exception as e:
                logger.debug(f"FFT cycle detection error: {e}")

        # ── LSTM ENSEMBLE PREDICTION (BINARY: SKIP vs TRADE) ──
        # ML learned from trailing SL simulation: TRADE = setup where SL locks profit (L2+)
        _lstm_model = self._lstm_per_asset.get(asset, self._lstm) if hasattr(self, '_lstm_per_asset') else self._lstm
        if _lstm_model and len(closes) >= 80:
            try:
                from src.scripts.train_all_models import compute_strategy_features
                import numpy as _np
                opens_list = ohlcv.get('opens', closes)
                X_seq, _ = compute_strategy_features(closes, highs, lows, opens_list, volumes, seq_len=30, n_features=50)
                if X_seq is not None and len(X_seq) > 0:
                    lstm_pred = _lstm_model.predict(X_seq[-1])
                    if lstm_pred and lstm_pred.get('confidence', 0) > 0.1:
                        # Binary: signal=0=SKIP, signal=1=TRADE
                        trade_quality = lstm_pred.get('trade_quality', 'UNKNOWN')
                        trade_conf = lstm_pred.get('confidence', 0)
                        ml_context['lstm_signal'] = trade_quality  # 'TRADE' or 'SKIP'
                        ml_context['lstm_confidence'] = round(trade_conf, 2)
                        ml_context['lstm_probs'] = lstm_pred.get('probs', [])

                        # TRADE signal with high confidence = boost entry
                        if trade_quality == 'TRADE' and trade_conf > 0.55:
                            entry_score += 2
                            score_reasons.append(f"lstm_TRADE({trade_conf:.0%})")
                        elif trade_quality == 'TRADE' and trade_conf > 0.40:
                            entry_score += 1
                            score_reasons.append(f"lstm_trade_weak({trade_conf:.0%})")
                        elif trade_quality == 'SKIP' and trade_conf > 0.75:
                            # HARD BLOCK: LSTM is very confident this setup dies at L1
                            print(f"  [{self._ex_tag}:{asset}] LSTM HARD SKIP: conf={trade_conf:.0%} — ML predicts L1 death")
                            return
                        elif trade_quality == 'SKIP' and trade_conf > 0.60:
                            entry_score -= 2
                            score_reasons.append(f"lstm_SKIP({trade_conf:.0%})")
                            math_filter_warnings.append(f"LSTM: SKIP signal ({trade_conf:.0%}) - setup predicts L1 death")
                        elif trade_quality == 'SKIP' and trade_conf > 0.40:
                            entry_score -= 1
                            score_reasons.append(f"lstm_skip_weak({trade_conf:.0%})")
            except Exception as e:
                logger.debug(f"LSTM prediction error: {e}")

        # ── PatchTST TRANSFORMER PREDICTION ──
        if self._patchtst and self._patchtst.is_ready and len(closes) >= 401:
            try:
                import numpy as _np
                ptst_pred = self._patchtst.predict(_np.array(closes))
                if ptst_pred and ptst_pred.get('confidence', 0) > 0.1:
                    ml_context['patchtst_direction'] = {1: 'UP', -1: 'DOWN', 0: 'NEUTRAL'}.get(ptst_pred.get('prediction', 0), 'NEUTRAL')
                    ml_context['patchtst_prob_up'] = round(ptst_pred.get('prob_up', 0.5), 2)
                    ml_context['patchtst_shock_prob'] = round(ptst_pred.get('liquidity_shock_prob', 0), 2)
                    # High shock probability = reduce confidence
                    if ptst_pred.get('liquidity_shock_prob', 0) > 0.6:
                        entry_score -= 2
                        score_reasons.append(f"ptst_shock={ptst_pred['liquidity_shock_prob']:.0%}")
                        math_filter_warnings.append(f"PatchTST: {ptst_pred['liquidity_shock_prob']:.0%} liquidity shock probability")
                    # Direction agreement / conflict
                    signal_dir = 1 if signal == "BUY" else -1
                    ptst_dir = ptst_pred.get('prediction', 0)
                    ptst_conf = ptst_pred.get('confidence', 0)
                    if ptst_dir == signal_dir and ptst_conf > 0.2:
                        entry_score += 1
                        score_reasons.append(f"ptst_{ml_context['patchtst_direction']}")
                    elif ptst_dir != 0 and ptst_dir != signal_dir and ptst_conf > 0.3:
                        # PatchTST DISAGREES with signal direction
                        if ptst_conf > 0.6:
                            # High confidence disagreement = hard block
                            print(f"  [{self._ex_tag}:{asset}] PatchTST CONFLICT: predicts {ml_context['patchtst_direction']} (conf={ptst_conf:.0%}) vs signal {'LONG' if signal=='BUY' else 'SHORT'}")
                            return
                        else:
                            entry_score -= 2
                            score_reasons.append(f"ptst_CONFLICT({ml_context['patchtst_direction']}_{ptst_conf:.0%})")
                            math_filter_warnings.append(f"PatchTST: predicts {ml_context['patchtst_direction']} ({ptst_conf:.0%}) — OPPOSITE to signal")
            except Exception as e:
                logger.debug(f"PatchTST prediction error: {e}")

        # ── ALPHA DECAY — signal freshness & optimal hold ──
        if self._alpha_decay and self._alpha_decay._fitted:
            try:
                ad_feats = self._alpha_decay.compute_features(bars_held=0)
                ml_context['alpha_freshness'] = round(ad_feats.get('alpha_freshness', 1.0), 2)
                ml_context['alpha_optimal_hold'] = int(ad_feats.get('alpha_optimal_hold', 6))
                ml_context['alpha_half_life'] = round(ad_feats.get('alpha_half_life', 7.0), 1)
            except Exception as e:
                logger.debug(f"Alpha decay error: {e}")

        # ── GARCH VOLATILITY FORECAST (pre-loaded) ──
        if len(closes) >= 100:
            try:
                _garch = self._garch_per_asset.get(asset)
                if _garch:
                    garch_forecast = _garch.forecast(list(closes[-100:]))
                    if garch_forecast:
                        current_vol = garch_forecast[-1]
                        avg_vol = sum(garch_forecast) / len(garch_forecast)
                        ml_context['garch_vol'] = round(current_vol, 6)
                        ml_context['garch_vol_expanding'] = current_vol > avg_vol * 1.2
                        if current_vol > avg_vol * 1.5:
                            math_filter_warnings.append(f"GARCH: volatility 50%+ above average -> expect wild swings")
            except Exception as e:
                logger.debug(f"GARCH forecast error: {e}")

        # ── RL AGENT (Q-learning optimizer) ──
        # Advises: enter/skip, size multiplier, SL buffer, patience
        rl_decision = None
        _rl = self._rl_per_asset.get(asset) if hasattr(self, '_rl_per_asset') else None
        if _rl and signal in ('BUY', 'SELL'):
            try:
                from src.ai.reinforcement_learning import EMATradeState
                _ema_slope = (ema_vals[-1] - ema_vals[-2]) / ema_vals[-2] * 100 if len(ema_vals) >= 2 else 0
                _atr_pctile = 0.5
                if len(atr_vals) >= 100:
                    _sorted = sorted(atr_vals[-100:])
                    _rank = sum(1 for v in _sorted if v <= current_atr)
                    _atr_pctile = _rank / len(_sorted)
                _vol_ratio = volumes[-1] / (sum(volumes[-20:]) / 20) if len(volumes) >= 20 and sum(volumes[-20:]) > 0 else 1.0
                # Compute Robinhood-specific state features
                _spread_cost_pct = 1.69 if self._paper_mode else 0.1  # Round-trip spread
                _atr_tp_mult = self.config.get('risk', {}).get('atr_tp_mult', 10.0)
                _filter_atr = current_atr
                if hasattr(self, '_last_chosen_tf_atr') and self._last_chosen_tf_atr.get(asset, 0) > 0:
                    _filter_atr = self._last_chosen_tf_atr[asset]
                _expected_move_pct = (_filter_atr * _atr_tp_mult / price) * 100 if price > 0 and _filter_atr > 0 else 0
                _move_to_spread = _expected_move_pct / _spread_cost_pct if _spread_cost_pct > 0 else 10.0
                _sniper_confl = self._last_sniper_confluence.get(asset, 0) if hasattr(self, '_last_sniper_confluence') else 0
                _entry_sc = entry_score if 'entry_score' in dir() and isinstance(entry_score, int) else 0
                # Timeframe rank: higher = better for spot trading
                _tf_rank_map = {'5m': 0, '15m': 1, '1h': 2, '4h': 3, '1d': 4}
                _chosen_tf_for_rl = ml_context.get('chosen_tf', self.SIGNAL_TIMEFRAMES[0] if self.SIGNAL_TIMEFRAMES else '4h')
                _tf_rank = _tf_rank_map.get(_chosen_tf_for_rl, 2)
                # Consecutive losses
                _consec_losses = self.asset_loss_streak.get(asset, 0)

                _state = EMATradeState(
                    ema_slope=_ema_slope,
                    ema_slope_bars=int(ml_context.get('new_line_bars', 1)),
                    price_ema_distance_atr=(price - ema_vals[-1]) / current_atr if current_atr > 0 else 0,
                    ema_acceleration=0.0,
                    trend_bars_since_flip=int(ml_context.get('prior_trend_bars', 3)),
                    trend_consistency=0.8,
                    higher_tf_alignment=ml_context.get('htf_alignment', 0),
                    atr_percentile=_atr_pctile,
                    volume_ratio=_vol_ratio,
                    spread_atr_ratio=_spread_cost_pct / (_filter_atr / price * 100) if _filter_atr > 0 and price > 0 else 0.1,
                    recent_win_rate=self.edge_stats.get(asset, {}).get('win_rate', 0.5),
                    daily_pnl_pct=self.daily_realized_pnl / max(self.equity, 1) * 100,
                    open_positions=len(self.positions),
                    consecutive_losses=_consec_losses,
                    hour_of_day=datetime.utcnow().hour,
                    day_of_week=datetime.utcnow().weekday(),
                    # NEW Robinhood-specific features
                    spread_cost_pct=_spread_cost_pct,
                    expected_move_pct=_expected_move_pct,
                    move_to_spread_ratio=_move_to_spread,
                    is_spot=self._paper_mode,
                    confluence_count=_sniper_confl,
                    entry_score=_entry_sc,
                    timeframe_rank=_tf_rank,
                )
                rl_decision = _rl.decide(_state)
                ml_context['rl_action'] = rl_decision.reasoning
                ml_context['rl_enter'] = rl_decision.enter_trade
                ml_context['rl_size_mult'] = rl_decision.position_size_mult
                ml_context['rl_sl_mult'] = rl_decision.sl_buffer_mult
                ml_context['rl_quality'] = round(rl_decision.quality_score, 2)
                # Store RL state for feedback loop on trade close
                ml_context['_rl_state'] = _state
                ml_context['_rl_action_idx'] = rl_decision.action_idx if hasattr(rl_decision, 'action_idx') else 0
                # Handle WAIT actions: RL says "not now, try again in N bars"
                _rl_wait_bars = 0
                if not rl_decision.enter_trade:
                    _rl_action = self.actions[rl_decision.action_idx] if hasattr(self, 'actions') else {}
                    # Check if it's a wait action from the RL actions list
                    rl_actions_list = _rl.actions if hasattr(_rl, 'actions') else []
                    if rl_decision.action_idx < len(rl_actions_list):
                        _rl_wait_bars = rl_actions_list[rl_decision.action_idx].get('wait_bars', 0)
                    if _rl_wait_bars > 0:
                        math_filter_warnings.append(f"RL: WAIT {_rl_wait_bars} bars ({rl_decision.reasoning})")
                        # Set cooldown so we don't re-evaluate for N poll intervals
                        self.last_trade_time[asset] = time.time() + (_rl_wait_bars * self.poll_interval * 0.5)
                    else:
                        math_filter_warnings.append(f"RL: SKIP recommended ({rl_decision.reasoning})")
            except Exception as e:
                logger.debug(f"RL agent error: {e}")

        # ── LIGHTGBM PRE-FILTER GATE ──
        # Uses 100+ features to predict L3+ runner vs L1 death
        # If LightGBM says HIGH probability of L1 death, skip before wasting LLM call
        lgbm_prediction = None
        lgbm_confidence = 0.0
        # Kill switch — set ACT_DISABLE_ML=1 to skip the entire ML gate (raw + wrapper).
        # Added 2026-04-22 after a backtest showed ML-on losing $330 more than ML-off
        # and flipping Sharpe from +0.09 to -0.29. Keeps the model files on disk for
        # future diagnosis while preventing them from touching live decisions. Soft
        # dep: also silences the Category B ML boosting logic downstream because that
        # branch only fires when lgbm_prediction is set.
        import os as _mlos
        _ml_kill = (_mlos.environ.get('ACT_DISABLE_ML') or '').strip().lower() in ('1', 'true', 'yes', 'on')
        if _ml_kill:
            ml_context['lgbm_direction'] = 'DISABLED'
            ml_context['lgbm_confidence'] = 0.0
            ml_context['lgbm_raw_prob'] = 0.0
            ml_context['lgbm_calibrated'] = False
            # Do not populate _lgbm_raw predictions at all — skip the whole block
            _lgbm_raw = None
        else:
            # Try per-asset binary model first (trained on 30 strategy features)
            _lgbm_raw = self._lgbm_raw.get(asset) if hasattr(self, '_lgbm_raw') else None
        if _lgbm_raw and len(closes) >= 55:
            try:
                from src.scripts.train_all_models import compute_strategy_features
                import numpy as _np
                opens_list = ohlcv.get('opens', closes)
                # Dynamically match feature count to what the trained model expects
                _model_n_features = _lgbm_raw.num_feature()
                X_seq, _ = compute_strategy_features(closes, highs, lows, opens_list, volumes, seq_len=1, n_features=max(_model_n_features, 50))
                if X_seq is not None and len(X_seq) > 0:
                    feat = X_seq[-1].reshape(1, -1)[:, :_model_n_features]  # Match model's expected features
                    trade_prob_raw = float(_lgbm_raw.predict(feat)[0])  # Binary: raw probability of TRADE class

                    # Apply isotonic calibration + data-driven entry_score delta if a
                    # calibration bundle was loaded at init. Falls back to hand-tuned
                    # thresholds when no bundle exists (i.e. first boot after deploy).
                    try:
                        from src.ml import calibration as _calib_mod
                        _bundle = getattr(self, '_lgbm_calibration', {}).get(asset)
                        score_delta, trade_conf = _calib_mod.score_delta_for(_bundle, trade_prob_raw)
                        _calibrated = _bundle is not None
                    except Exception:
                        _bundle = None
                        _calibrated = False
                        trade_conf = trade_prob_raw
                        score_delta = None

                    is_trade = trade_conf > 0.5
                    ml_context['lgbm_direction'] = 'TRADE' if is_trade else 'SKIP'
                    ml_context['lgbm_confidence'] = round(trade_conf if is_trade else (1 - trade_conf), 2)
                    ml_context['lgbm_calibrated'] = bool(_calibrated)
                    ml_context['lgbm_raw_prob'] = round(trade_prob_raw, 4)
                    lgbm_prediction = 1 if is_trade else 0
                    lgbm_confidence = ml_context['lgbm_confidence']

                    if score_delta is not None and _calibrated:
                        # Data-driven path: delta came from fit_calibration's bucket-winrate map.
                        if score_delta != 0:
                            entry_score += int(score_delta)
                            tag = 'TRADE' if score_delta > 0 else 'SKIP'
                            score_reasons.append(f"lgbm_cal_{tag}({trade_conf:.0%},Δ{score_delta:+d})")
                            if score_delta <= -3:
                                math_filter_warnings.append(
                                    f"LGBM STRONG SKIP (calibrated {trade_conf:.0%}) — L1 death predicted"
                                )
                            elif score_delta <= -2:
                                math_filter_warnings.append(
                                    f"LGBM SKIP (calibrated {trade_conf:.0%}) — L1 death predicted by ML"
                                )
                        else:
                            score_reasons.append(f"lgbm_abstain({trade_conf:.0%})")
                    else:
                        # Fallback: historical hand-tuned thresholds on raw probability.
                        if is_trade and trade_conf > 0.60:
                            entry_score += 2
                            score_reasons.append(f"lgbm_TRADE({trade_conf:.0%})")
                        elif is_trade and trade_conf > 0.45:
                            entry_score += 1
                            score_reasons.append(f"lgbm_trade_weak({trade_conf:.0%})")
                        elif not is_trade and (1 - trade_conf) > 0.75:
                            entry_score -= 3
                            score_reasons.append(f"lgbm_HARD_SKIP({1-trade_conf:.0%})")
                            math_filter_warnings.append(f"LGBM STRONG SKIP: {1-trade_conf:.0%} confidence L1 death — LLM should be very cautious")
                            print(f"  [{self._ex_tag}:{asset}] LGBM ADVISORY: conf={1-trade_conf:.0%} ML predicts L1 death — LLM will decide")
                        elif not is_trade and (1 - trade_conf) > 0.60:
                            entry_score -= 2
                            score_reasons.append(f"lgbm_SKIP({1-trade_conf:.0%})")
                            math_filter_warnings.append(f"LGBM: SKIP signal ({1-trade_conf:.0%}) — L1 death predicted by ML")
                        elif not is_trade and (1 - trade_conf) > 0.45:
                            entry_score -= 1
                            score_reasons.append(f"lgbm_skip_weak({1-trade_conf:.0%})")

                    _cal_tag = 'cal' if _calibrated else 'raw'
                    print(f"  [{self._ex_tag}:{asset}] LGBM[{_cal_tag}]: {ml_context['lgbm_direction']} conf={ml_context['lgbm_confidence']:.2f} | score now={entry_score}")
            except Exception as e:
                logger.debug(f"LightGBM binary prediction error for {asset}: {e}")

        # Fallback: old LightGBM wrapper with generic features
        # (also gated by ACT_DISABLE_ML — the wrapper feeds the same LightGBM model
        # through a different feature path and produces the same problematic boost/
        # penalty deltas)
        elif self._lgbm and len(closes) >= 55 and not _ml_kill:
            try:
                # Build Category B enriched external features
                _catb_external = {
                    'vol_regime_encoded': float({'LOW': 0, 'NORMAL': 1, 'HIGH': 2, 'EXTREME': 3}.get(
                        ml_context.get('vol_regime', 'NORMAL'), 1)),
                    'cycle_phase_encoded': float({'BOTTOM': 0, 'RISING': 1, 'TOP': 2, 'FALLING': 3}.get(
                        ml_context.get('cycle_phase', 'RISING'), 1)),
                    'dominant_period': float(ml_context.get('dominant_cycle', 30)),
                    # Category B risk features → LightGBM
                    'evt_var_99': float(ml_context.get('evt_var_99', 0)),
                    'evt_tail_ratio': float(ml_context.get('evt_tail_ratio', 1.0)),
                    'mc_risk_score': float(ml_context.get('mc_risk_score', 0.5)),
                    'mc_position_scale': float(ml_context.get('mc_position_scale', 1.0)),
                    'hawkes_intensity': float(ml_context.get('hawkes_intensity', 0.05)),
                    'tft_forecast_bps': float(ml_context.get('tft_forecast_bps', 0)),
                    'tft_confidence': float(ml_context.get('tft_confidence', 0)),
                }
                # Build Category B sentiment features
                _catb_sentiment = {
                    'sentiment_mean': float(ml_context.get('sentiment_mean', 0.0)),
                    'sentiment_z_score': float(ml_context.get('sentiment_z_score', 0.0)),
                }
                features = self._lgbm.extract_features(
                    closes=closes, highs=highs, lows=lows, volumes=volumes,
                    external_features=_catb_external,
                    sentiment_features=_catb_sentiment,
                )
                if features and features[-1]:
                    preds = self._lgbm.predict([features[-1]])
                    if preds:
                        lgbm_cls, lgbm_conf = preds[0]
                        lgbm_prediction = lgbm_cls
                        lgbm_confidence = lgbm_conf
                        ml_context['lgbm_direction'] = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(lgbm_cls, 'FLAT')
                        ml_context['lgbm_confidence'] = round(lgbm_conf, 2)
                        signal_dir = 1 if signal == "BUY" else -1
                        if lgbm_cls != 0 and lgbm_cls != signal_dir and lgbm_conf > 0.70:
                            entry_score -= 3
                            score_reasons.append(f"lgbm_disagree={ml_context['lgbm_direction']}@{lgbm_conf:.0%}")
                        elif lgbm_cls == signal_dir and lgbm_conf > 0.65:
                            entry_score += 2
                            score_reasons.append(f"lgbm_agrees={lgbm_conf:.0%}")
                        print(f"  [{self._ex_tag}:{asset}] LGBM(fallback): {ml_context['lgbm_direction']} conf={lgbm_conf:.2f}")
            except Exception as e:
                logger.debug(f"LightGBM prediction error for {asset}: {e}")

        # Store ML context for bear agent and other downstream uses
        self._last_ml_context = ml_context

        # ── META CONTROLLER: arbitrates ML model predictions (LGB + RL + PatchTST) ──
        if self._meta_controller:
            try:
                # Map ml_context values to MetaController.arbitrate() signature
                _mc_lgb_class = {'LONG': 1, 'SHORT': -1, 'TRADE': 1, 'SKIP': -1}.get(
                    ml_context.get('lgbm_direction', 'FLAT'), 0)
                _mc_rl_action = 1 if ml_context.get('rl_enter', False) else -1
                _mc_rl_prob = ml_context.get('rl_quality', 0.5)
                _mc_finbert = ml_context.get('sentiment_mean', 0.0)
                _mc_patch = None
                if 'patchtst_direction' in ml_context:
                    _mc_patch = {
                        'prediction': {'UP': 1, 'DOWN': -1}.get(ml_context['patchtst_direction'], 0),
                        'prob_up': ml_context.get('patchtst_prob_up', 0.5),
                        'liquidity_shock_prob': ml_context.get('patchtst_shock_prob', 0),
                    }
                # Build features dict from available ml_context
                _mc_features = {
                    'hurst': hurst_value,
                    'kalman_snr': ml_context.get('kalman_snr', 0),
                    'kalman_slope': ml_context.get('kalman_slope', 0),
                    'vol_percentile': ml_context.get('vol_percentile', 50),
                    'entry_score': entry_score,
                    'ewma_vol': ml_context.get('realized_vol_annual', 0.02),
                }
                meta_dir, meta_conf, meta_scale = self._meta_controller.arbitrate(
                    lgb_class=_mc_lgb_class,
                    lgb_conf=ml_context.get('lgbm_confidence', 0),
                    rl_action=_mc_rl_action,
                    rl_prob=_mc_rl_prob,
                    features=_mc_features,
                    finbert_score=_mc_finbert,
                    patch_result=_mc_patch,
                    asset_name=asset,
                    hmm_regime=ml_context.get('hmm_regime', 'SIDEWAYS').lower(),
                    hmm_crisis_prob=ml_context.get('crisis_probability', 0),
                    kalman_snr=ml_context.get('kalman_snr', 1.0),
                    mc_risk_score=ml_context.get('mc_risk_score', 0.5),
                    mc_position_scale=ml_context.get('mc_position_scale', 1.0),
                    hawkes_intensity=ml_context.get('hawkes_intensity', 0),
                    alpha_freshness=ml_context.get('alpha_freshness', 1.0),
                    evt_risk_score=ml_context.get('evt_var_99', 0.3),
                    evt_position_scale=1.0,
                )
                ml_context['meta_direction'] = meta_dir
                ml_context['meta_confidence'] = meta_conf
                ml_context['meta_position_scale'] = meta_scale
                print(f"  [{self._ex_tag}:{asset}] META-CTRL: dir={meta_dir} conf={meta_conf:.2f} scale={meta_scale:.2f}")
            except Exception as e:
                logger.debug(f"MetaController failed: {e}")

        # ── SIGNAL COMBINER: formal L1+L2+L3 fusion ──
        if self._signal_combiner:
            try:
                # L1 = quantitative consensus signal (entry_score normalized to [-1, +1])
                _sc_l1 = max(-1.0, min(1.0, entry_score / 7.0)) * (1 if signal == 'BUY' else -1)
                # L2 = sentiment dict (matches SentimentPipeline.aggregate_sentiment() format)
                _sc_l2 = {
                    'aggregate_score': ml_context.get('sentiment_mean', 0.0),
                    'confidence': 0.5 if ml_context.get('sentiment_available') else 0.1,
                    'freshness': 1.0,
                }
                # L3 = risk evaluation dict
                _sc_l3 = {'action': 'ALLOW'}
                if ml_context.get('hmm_regime') == 'CRISIS' and ml_context.get('crisis_probability', 0) > 0.7:
                    _sc_l3 = {'action': 'VETO', 'reason': 'HMM CRISIS regime with high probability'}
                elif ml_context.get('anomaly_type', 'NONE') not in ('NONE', 'none', ''):
                    _sc_l3 = {'action': 'REDUCE', 'reason': f"Anomaly: {ml_context.get('anomaly_type')}"}

                combined = self._signal_combiner.combine(_sc_l1, _sc_l2, _sc_l3)
                ml_context['combined_signal'] = combined.get('final_signal', 0)
                ml_context['combined_action'] = combined.get('action', 'hold')
                ml_context['combined_confidence'] = combined.get('confidence', 0)
                if combined.get('action') == 'hold' and abs(_sc_l1) < 0.3:
                    math_filter_warnings.append(f"COMBINER: weak signal ({combined.get('final_signal',0):.3f}) — low conviction")
                print(f"  [{self._ex_tag}:{asset}] COMBINER: signal={combined.get('final_signal',0):.3f} action={combined.get('action','?')} conf={combined.get('confidence',0):.2f}")
            except Exception as e:
                logger.debug(f"SignalCombiner failed: {e}")

        # ── ADVANCED LEARNING: Anomaly Detection (HARD VETO — flash crash / liquidity sweep) ──
        if self._advanced_learning and len(closes) >= 50:
            try:
                import numpy as _np
                _c = _np.array(closes[-100:], dtype=float)
                _v = _np.array(volumes[-100:], dtype=float)
                anomaly = self._advanced_learning.anomaly_detector.detect_anomalies(_c, _v)
                ml_context['anomaly_type'] = anomaly.get('type', 'NONE')
                ml_context['anomaly_severity'] = anomaly.get('severity', 0)
                if anomaly.get('is_anomaly'):
                    print(f"  [{self._ex_tag}:{asset}] ANOMALY VETO: {anomaly['type']} severity={anomaly['severity']} z={anomaly['z_score']} vol_spike={anomaly['vol_spike']}x — BLOCKING ENTRY")
                    return
            except Exception as e:
                logger.debug(f"Anomaly detection error: {e}")

        # ── ADVANCED LEARNING: Regime Classification (enriches ml_context, no score change) ──
        if self._advanced_learning and len(closes) >= 100:
            try:
                import numpy as _np
                _c = _np.array(closes[-200:], dtype=float)
                _h = _np.array(highs[-200:], dtype=float)
                _l = _np.array(lows[-200:], dtype=float)
                _v = _np.array(volumes[-200:], dtype=float)
                regime = self._advanced_learning.regime_classifier.classify_regime(_c, _h, _l, _v)
                ml_context['adv_regime'] = regime.regime_type
                ml_context['adv_regime_confidence'] = round(regime.confidence, 1)
                ml_context['adv_optimal_strategy'] = regime.optimal_strategy
                ml_context['adv_trend_strength'] = round(regime.trend_strength, 3)
            except Exception as e:
                logger.debug(f"Regime classification error: {e}")

        # ── ADVANCED LEARNING: Pipeline Overlay (adjusts SECONDARY knobs only) ──
        # SAFE: Never touches v13/v14 core (min/max_entry_score, short_score_penalty, sr_assets)
        _overlay = None
        if self._advanced_learning and asset in self._advanced_learning.active_overlays:
            _overlay = self._advanced_learning.active_overlays[asset]
            ml_context['meta_regime'] = _overlay.regime
            ml_context['meta_risk_multiplier'] = _overlay.risk_multiplier
            ml_context['meta_reasoning'] = _overlay.reasoning[:100]
            # If overlay says HOLD: block entry entirely
            if _overlay.hold_strategy:
                print(f"  [{self._ex_tag}:{asset}] META HOLD: {_overlay.reasoning[:80]} — blocking entry")
                return

        # ── TEMPORAL TRANSFORMER: attention-based forecast (advisory, feeds ml_context) ──
        if self._temporal_transformer and len(closes) >= 120:
            try:
                import numpy as _np
                # Build relative OHLCV changes as input
                _c = _np.array(closes[-120:], dtype=float)
                _h = _np.array(highs[-120:], dtype=float)
                _l = _np.array(lows[-120:], dtype=float)
                _v = _np.array(volumes[-120:], dtype=float)
                # Normalize to pct changes
                _pct = _np.diff(_c) / _c[:-1]
                _hpct = (_h[1:] - _c[:-1]) / _c[:-1]
                _lpct = (_l[1:] - _c[:-1]) / _c[:-1]
                _vpct = _np.diff(_v) / (_v[:-1] + 1e-12)
                _history = _np.column_stack([_pct, _hpct, _lpct, _vpct])[-self._temporal_transformer.context_len:]
                # Pad to d_model if needed
                if _history.shape[1] < self._temporal_transformer.d_model:
                    _pad = _np.zeros((_history.shape[0], self._temporal_transformer.d_model - _history.shape[1]))
                    _history = _np.hstack([_history, _pad])
                forecast = self._temporal_transformer.forecast_return(_history)
                ml_context['tft_forecast_bps'] = round(forecast.get('forecast_return_bps', 0), 1)
                ml_context['tft_confidence'] = round(forecast.get('confidence', 0), 2)
            except Exception as e:
                logger.debug(f"TemporalTransformer error: {e}")

        # ── HAWKES PROCESS: event clustering intensity (advisory) ──
        if self._hawkes and len(closes) >= 50:
            try:
                import numpy as _np
                # Detect large moves as "events" (returns > 2 std devs)
                _rets = _np.abs(_np.diff(_np.array(closes[-100:], dtype=float)) / _np.array(closes[-100:-1], dtype=float))
                _mean_r = _np.mean(_rets)
                _std_r = _np.std(_rets)
                _threshold = _mean_r + 2.0 * _std_r
                _event_indices = _np.where(_rets > _threshold)[0]
                if len(_event_indices) >= 3:
                    _event_times = _event_indices.astype(float)
                    intensity = self._hawkes.current_intensity(_event_times)
                    ml_context['hawkes_intensity'] = round(float(intensity), 3)
                    ml_context['hawkes_event_count'] = len(_event_indices)
                    # High intensity = event clustering = caution
                    if intensity > 0.5:
                        math_filter_warnings.append(f"HAWKES: intensity={intensity:.2f} (event clustering — caution)")
            except Exception as e:
                logger.debug(f"Hawkes process error: {e}")

        # ── EVT TAIL RISK: fat-tail VaR assessment (advisory, feeds risk context) ──
        if self._evt_risk and len(closes) >= 100:
            try:
                import numpy as _np
                _rets = _np.diff(_np.log(_np.array(closes[-500:], dtype=float) + 1e-12))
                if len(_rets) >= 50:
                    evt_result = self._evt_risk.fit_and_assess(_rets)
                    ml_context['evt_var_99'] = round(evt_result.get('evt_var_99', 0), 4)
                    ml_context['evt_tail_index'] = round(evt_result.get('tail_index', 3.0), 2)
                    ml_context['evt_tail_ratio'] = round(evt_result.get('tail_ratio', 1.0), 2)
            except Exception as e:
                logger.debug(f"EVT risk error: {e}")

        # ── MONTE CARLO RISK: forward VaR simulation (advisory) ──
        if self._mc_risk and len(closes) >= 50 and price > 0:
            try:
                import numpy as _np
                _rets = _np.diff(_np.array(closes[-100:], dtype=float)) / _np.array(closes[-100:-1], dtype=float)
                _vol = float(_np.std(_rets))
                _drift = float(_np.mean(_rets))
                _regime = ml_context.get('adv_regime', 'NEUTRAL').lower()
                _regime_map = {'trending_up': 'bull', 'trending_down': 'bear', 'volatile': 'crisis', 'ranging': 'sideways'}
                mc_regime = _regime_map.get(_regime, 'normal')
                mc_result = self._mc_risk.simulate(current_price=price, volatility=_vol, drift=_drift, regime=mc_regime)
                ml_context['mc_var_95'] = round(mc_result.get('mc_var_95', 0), 4)
                ml_context['mc_risk_score'] = round(mc_result.get('mc_risk_score', 0.5), 2)
                ml_context['mc_position_scale'] = round(mc_result.get('mc_position_scale', 1.0), 2)
            except Exception as e:
                logger.debug(f"Monte Carlo risk error: {e}")

        # ── SENTIMENT: real values from per-bar online computation ──
        if self._sentiment and hasattr(self, '_sentiment_cache'):
            try:
                _sent = self._sentiment_cache.get(asset, {})
                ml_context['sentiment_mean'] = _sent.get('sentiment_mean', 0.0)
                ml_context['sentiment_z_score'] = _sent.get('sentiment_z_score', 0.0)
                ml_context['sentiment_available'] = True
            except Exception:
                ml_context['sentiment_mean'] = 0.0
                ml_context['sentiment_z_score'] = 0.0

        # ── CATEGORY B DIRECT SCORE MODIFIERS ──
        # These quantitative risk signals directly adjust entry_score like LGBM/LSTM/PatchTST do.
        # They are NOT advisory — they are hard quantitative vetoes and boosts.

        # EVT: Extreme tail risk → penalize entry
        _evt_var = ml_context.get('evt_var_99', 0)
        _evt_tail_ratio = ml_context.get('evt_tail_ratio', 1.0)
        if _evt_var < -0.08:  # VaR worse than -8% at 99% confidence
            entry_score -= 2
            score_reasons.append(f"evt_extreme_tail({_evt_var:.3f})")
        elif _evt_tail_ratio > 2.0:  # Fat tails 2x worse than normal
            entry_score -= 1
            score_reasons.append(f"evt_fat_tail({_evt_tail_ratio:.1f}x)")

        # Monte Carlo: High forward risk → penalize
        _mc_risk = ml_context.get('mc_risk_score', 0.5)
        _mc_pos_scale = ml_context.get('mc_position_scale', 1.0)
        if _mc_risk > 0.8:  # VaR exceeds 80% of risk budget
            entry_score -= 2
            score_reasons.append(f"mc_high_risk({_mc_risk:.2f})")
        elif _mc_risk > 0.6:
            entry_score -= 1
            score_reasons.append(f"mc_elevated_risk({_mc_risk:.2f})")

        # Hawkes: Event clustering → penalize (cascade risk)
        _hawkes_intensity = ml_context.get('hawkes_intensity', 0.0)
        if _hawkes_intensity > 0.8:  # Strong event clustering
            entry_score -= 2
            score_reasons.append(f"hawkes_clustering({_hawkes_intensity:.2f})")
        elif _hawkes_intensity > 0.5:
            entry_score -= 1
            score_reasons.append(f"hawkes_elevated({_hawkes_intensity:.2f})")

        # Temporal Transformer: Direction conflict → penalize
        _tft_bps = ml_context.get('tft_forecast_bps', 0)
        _tft_conf = ml_context.get('tft_confidence', 0)
        if _tft_conf > 0.3:  # Only act on confident forecasts
            _tft_bullish = _tft_bps > 0
            _signal_bullish = signal == 'BUY'
            if _tft_bullish != _signal_bullish and abs(_tft_bps) > 5:
                entry_score -= 1
                score_reasons.append(f"tft_disagree({_tft_bps:+.0f}bps)")
            elif _tft_bullish == _signal_bullish and abs(_tft_bps) > 10:
                entry_score += 1
                score_reasons.append(f"tft_confirm({_tft_bps:+.0f}bps)")

        # ── ML ENSEMBLE VOTE: if 2+ models agree SKIP, hard block ──
        # Individual models can be wrong, but when multiple agree the signal is bad, trust them
        ml_skip_votes = []
        if ml_context.get('lgbm_direction') == 'SKIP' and ml_context.get('lgbm_confidence', 0) > 0.55:
            ml_skip_votes.append(f"LGBM({ml_context['lgbm_confidence']:.0%})")
        if ml_context.get('lstm_quality') == 'SKIP' and ml_context.get('lstm_confidence', 0) > 0.55:
            ml_skip_votes.append(f"LSTM({ml_context['lstm_confidence']:.0%})")
        ptst_dir = ml_context.get('patchtst_direction', 'NEUTRAL')
        signal_expected = 'UP' if signal == 'BUY' else 'DOWN'
        if ptst_dir != 'NEUTRAL' and ptst_dir != signal_expected and ml_context.get('patchtst_prob_up', 0.5) not in (0.5,):
            ml_skip_votes.append(f"PatchTST({ptst_dir})")
        if ml_context.get('rl_enter') is False:
            ml_skip_votes.append("RL(SKIP)")
        if ml_context.get('hmm_regime') == 'CRISIS':
            ml_skip_votes.append("HMM(CRISIS)")

        # Category B ensemble voters
        if _hawkes_intensity > 0.8:
            ml_skip_votes.append(f"Hawkes({_hawkes_intensity:.2f})")
        if _mc_risk > 0.8:
            ml_skip_votes.append(f"MC({_mc_risk:.2f})")
        if _evt_var < -0.08:
            ml_skip_votes.append(f"EVT({_evt_var:.3f})")
        if _tft_conf > 0.4 and ((_tft_bps > 0) != (signal == 'BUY')) and abs(_tft_bps) > 15:
            ml_skip_votes.append(f"TFT({_tft_bps:+.0f}bps)")

        if len(ml_skip_votes) >= 4:
            print(f"  [{self._ex_tag}:{asset}] ML CONSENSUS BLOCK: {len(ml_skip_votes)} models vote SKIP — {', '.join(ml_skip_votes)}")
            return
        elif len(ml_skip_votes) >= 3:
            entry_score -= 2
            score_reasons.append(f"ml_consensus_skip({len(ml_skip_votes)})")
            math_filter_warnings.append(f"ML ENSEMBLE: {len(ml_skip_votes)} models vote SKIP — {', '.join(ml_skip_votes)}")

        # ── ENTRY SCORE HARD GATE (v13 optimized) ──
        # After all ML boosting, check score range [min, max]
        # v13 finding: high scores (8+) are momentum traps → cap at 7
        # v13 finding: SHORTs need +3 extra score (LONG bias in crypto)
        min_entry_score = self.config.get('adaptive', {}).get('min_entry_score', 4)
        max_entry_score = self.config.get('adaptive', {}).get('max_entry_score', 7)
        short_score_penalty = self.config.get('adaptive', {}).get('short_score_penalty', 3)
        # SNIPER: raise minimum score for higher-quality entries
        if self.sniper_enabled:
            min_entry_score = max(min_entry_score, self.sniper_min_score)
        effective_min = min_entry_score
        if signal == "SELL":  # SHORT entry needs higher score
            effective_min += short_score_penalty

        # Safe-entries gate (B + D + F): hard veto LLM can't override, spread-aware
        # bump on high-spread venues, consecutive-loss throttle that pauses after N.
        if getattr(self, '_safe_enabled', False):
            try:
                from src.trading import safe_entries as _safe
                rt_spread = float(self.config.get('exchanges', [{}])[0].get('round_trip_spread_pct', 0.1)
                                  if isinstance(self.config.get('exchanges'), list) else 0.1)
                effective_min = _safe.effective_min_score(effective_min, rt_spread, self._safe_config)
                reject, reason = _safe.enforce_hard_score_veto(entry_score, effective_min)
                if reject:
                    print(f"  [{self._ex_tag}:{asset}] SAFE REJECT: {reason}")
                    return
                # Consec-loss throttle — may pause entirely
                mult, throttle_reason = self._safe_state.size_multiplier_for(
                    asset, self._safe_config,
                )
                if mult <= 0.0:
                    print(f"  [{self._ex_tag}:{asset}] SAFE REJECT: {throttle_reason}")
                    # Persist paused_until update
                    try:
                        self._safe_state.save(_safe.default_state_path())
                    except Exception:
                        pass
                    return
                if mult < 1.0:
                    print(f"  [{self._ex_tag}:{asset}] SAFE SIZE x{mult}: {throttle_reason}")
                    self._safe_size_mult = mult  # applied at position sizing
                else:
                    self._safe_size_mult = 1.0
            except Exception as _se:
                print(f"  [{self._ex_tag}:{asset}] SAFE check error: {_se}")

        # ── META-LABEL VETO GATE (rule-conditional) ──
        # Fires ONLY when the rule score already passes effective_min (i.e. the
        # rule strategy wants to enter). The meta model — trained on rule-signaled
        # bars labeled by forward-simulated win/loss — says TAKE or SKIP. It can
        # only SUBTRACT from score (never add), so it can veto rule-approved
        # trades but can never take trades the rules didn't already approve.
        # Honors ACT_DISABLE_ML — set that to bypass this gate too.
        _ml_kill_meta = (os.environ.get('ACT_DISABLE_ML') or '').strip().lower() in ('1', 'true', 'yes', 'on')
        _meta_booster = getattr(self, '_lgbm_meta', {}).get(asset) if not _ml_kill_meta else None
        if (_meta_booster is not None
                and entry_score >= effective_min
                and len(closes) >= 55):
            try:
                from src.scripts.train_all_models import compute_strategy_features as _csf
                import numpy as _np
                _opens = ohlcv.get('opens', closes)
                _nf = _meta_booster.num_feature()
                _X, _ = _csf(closes, highs, lows, _opens, volumes, seq_len=1,
                             n_features=max(_nf, 50))
                if _X is not None and len(_X) > 0:
                    _feat = _X[-1].reshape(1, -1)[:, :_nf]
                    _meta_prob_raw = float(_meta_booster.predict(_feat)[0])
                    # Apply calibration if present so probability reflects actual
                    # per-bucket win rate from the training holdout.
                    _meta_bundle = getattr(self, '_lgbm_meta_calibration', {}).get(asset)
                    try:
                        from src.ml.calibration import apply_calibration as _ac
                        _meta_prob = float(_ac(_meta_bundle, _meta_prob_raw))
                    except Exception:
                        _meta_prob = _meta_prob_raw
                    _take_thresh = float(self._lgbm_meta_threshold.get(asset, 0.5))
                    _veto = _meta_prob < _take_thresh
                    ml_context['meta_prob'] = round(_meta_prob, 3)
                    ml_context['meta_take_threshold'] = _take_thresh
                    ml_context['meta_decision'] = 'SKIP' if _veto else 'TAKE'
                    ml_context['meta_prob_raw'] = round(_meta_prob_raw, 4)
                    ml_context['meta_features_snapshot'] = _feat.flatten().tolist()  # for shadow log
                    # Shadow mode (ACT_META_SHADOW_MODE=1): log prediction, DO NOT veto.
                    # Shadow veto decisions accumulate on disk; retrain uses them as
                    # ground-truth training data once enough trades settle.
                    try:
                        from src.ml.shadow_log import is_enabled as _shadow_on
                        _shadow_mode = _shadow_on()
                    except Exception:
                        _shadow_mode = False
                    if _shadow_mode:
                        ml_context['meta_shadow'] = True
                        score_reasons.append(f"meta_SHADOW({_meta_prob:.2f},would_{'VETO' if _veto else 'TAKE'})")
                        print(f"  [{self._ex_tag}:{asset}] META[shadow]: prob={_meta_prob:.2f} "
                              f"take={_take_thresh:.2f} would_{'VETO' if _veto else 'TAKE'} — NOT applied")
                    elif _veto:
                        _veto_delta = -3  # Strong push below effective_min
                        entry_score += _veto_delta
                        score_reasons.append(f"meta_VETO({_meta_prob:.2f}<{_take_thresh:.2f})")
                        math_filter_warnings.append(
                            f"META VETO: rule-conditional model predicts "
                            f"{_meta_prob:.0%} win prob < {_take_thresh:.0%} threshold"
                        )
                        print(f"  [{self._ex_tag}:{asset}] META VETO: prob={_meta_prob:.2f} < take={_take_thresh:.2f} — score -> {entry_score}")
                    else:
                        score_reasons.append(f"meta_TAKE({_meta_prob:.2f})")
                        print(f"  [{self._ex_tag}:{asset}] META TAKE: prob={_meta_prob:.2f} >= take={_take_thresh:.2f}")
            except Exception as _me:
                logger.debug(f"meta-label veto error for {asset}: {_me}")

        # SCORE: Advisory only (when safe-entries disabled) — LLM + agents make final decision
        if entry_score < effective_min:
            math_filter_warnings.append(f"LOW SCORE: {entry_score}/{effective_min} — weak setup, LLM should be cautious")
            print(f"  [{self._ex_tag}:{asset}] SCORE ADVISORY: {entry_score}/{effective_min} ({', '.join(score_reasons)}) — LOW but LLM decides")
            # DON'T return — let LLM see everything (fallback path only)
        if entry_score > max_entry_score:
            math_filter_warnings.append(f"HIGH SCORE: {entry_score}>{max_entry_score} — possible momentum trap")
            print(f"  [{self._ex_tag}:{asset}] SCORE ADVISORY: {entry_score}>{max_entry_score} (momentum trap warning)")
            # DON'T return — let LLM see everything

        # v13 Volatility: advisory for LLM (not a hard block)
        if len(atr_vals) >= 20:
            avg_atr = sum(atr_vals[-20:]) / 20
            if avg_atr > 0 and current_atr / avg_atr > 2.0:
                math_filter_warnings.append(f"EXTREME VOL: ATR {current_atr/avg_atr:.1f}x average — high risk")
                print(f"  [{self._ex_tag}:{asset}] VOL ADVISORY: ATR {current_atr/avg_atr:.1f}x average — LLM informed")

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
        ema_slope_pct = ((ema_vals[-1] - ema_vals[-4]) / ema_vals[-4] * 100) if len(ema_vals) >= 4 and ema_vals[-4] > 0 else 0
        consecutive_trend = 0
        for i in range(len(ema_vals)-1, max(0, len(ema_vals)-20), -1):
            if i > 0:
                if ema_direction == "RISING" and ema_vals[i] > ema_vals[i-1]:
                    consecutive_trend += 1
                elif ema_direction == "FALLING" and ema_vals[i] < ema_vals[i-1]:
                    consecutive_trend += 1
                else:
                    break

        # ── Loss streak info for LLM (hard block is above, this is informational) ──
        streak = self.asset_loss_streak.get(asset, 0)
        if streak >= 2:
            math_filter_warnings.append(f"LOSS STREAK: {streak} consecutive losses on {asset}")

        forced_action = "SHORT" if signal == "SELL" else "LONG" if signal == "BUY" else "FLAT"

        # L3: ML Inference log
        try:
            _ds = DashboardState()
            ml_models_used = [k for k in ml_context if not k.startswith('_')]
            hmm_r = ml_context.get('hmm_regime', 'N/A')
            _ds.add_layer_log('L3', f"{asset}: {len(ml_models_used)} ML features (HMM={hmm_r}, Hurst={hurst_value:.2f})", "info")
            # L4: RL Agent
            if rl_decision:
                rl_act = 'ENTER' if ml_context.get('rl_enter', False) else 'SKIP'
                _ds.add_layer_log('L4', f"{asset}: RL={rl_act} quality={ml_context.get('rl_quality', 0):.2f} size={ml_context.get('rl_size_mult', 1):.1f}x", "info")
            # L6: Risk Gate
            vpin_v = vpin_status.get('vpin', 0) if vpin_status else 0
            risk_msg = f"{asset}: score={entry_score} VPIN={vpin_v:.3f}"
            if len(math_filter_warnings) > 0:
                risk_msg += f" warnings={len(math_filter_warnings)}"
            _ds.add_layer_log('L6', risk_msg, "warning" if len(math_filter_warnings) > 2 else "info")
        except Exception:
            pass

        # ── Run Agent Orchestrator + Debate Engine (math agents) ──
        orch_result = self._run_orchestrator(
            asset, price, signal, closes, highs, lows, opens, volumes,
            ema_vals=ema_vals, atr_vals=atr_vals, ema_direction=ema_direction
        )

        # ── Push agent data to DashboardState for frontend ──
        try:
            if orch_result and orch_result.get('agent_votes'):
                ds = DashboardState()
                agent_votes = {}
                agent_weights = {}
                for name, v in orch_result['agent_votes'].items():
                    agent_votes[name] = {
                        'direction': v.get('dir', 0),
                        'confidence': v.get('conf', 0.5),
                        'reasoning': v.get('reasoning', ''),
                    }
                    agent_weights[name] = 1.0
                ds.update_agent_overlay({
                    'enabled': True,
                    'agent_votes': agent_votes,
                    'agent_weights': agent_weights,
                    'consensus_level': orch_result.get('consensus', 'N/A'),
                    'data_quality': orch_result.get('data_quality', 1.0),
                    'last_decision': {
                        'asset': asset,
                        'direction': orch_result.get('consensus_dir', 'FLAT'),
                        'confidence': orch_result.get('confidence', 0.5),
                    },
                })
                # Layer logs: L5 (Signal Aggregation)
                ds.add_layer_log('L5', f"{asset}: {orch_result.get('consensus_dir','FLAT')} {orch_result.get('consensus','?')} (net conf={orch_result.get('confidence',0):.2f})", "info")
        except Exception as e:
            logger.debug(f"Agent overlay push failed: {e}")

        # ══════════════════════════════════════════════════════════
        # TRADING BRAIN v2: Multi-model consensus + CoT + Memory + Regime + Kelly + Session
        # Falls back to legacy unified LLM if Brain v2 unavailable
        # ══════════════════════════════════════════════════════════
        if orch_result and orch_result.get('veto'):
            math_filter_warnings.append(f"AGENT VETO: consensus={orch_result.get('consensus','?')} — agents strongly disagree")

        # ── Inject Math Agent analysis into LLM context ──
        # The 10 math agents (momentum, volatility, microstructure, etc.) provide
        # quantitative analysis that the LLM uses for better pattern recognition
        math_agent_context = ""
        if orch_result and orch_result.get('agent_summary'):
            math_lines = ["MATH AGENT ANALYSIS (10 quantitative agents with adversarial debate):"]
            math_lines.append(f"  CONSENSUS: {orch_result.get('consensus', '?')} direction={orch_result.get('consensus_dir', '?')}")
            for summary_line in orch_result.get('agent_summary', [])[:12]:
                math_lines.append(f"  {summary_line}")
            # Include debate results if available
            if orch_result.get('debate_result'):
                debate = orch_result['debate_result']
                math_lines.append(f"  DEBATE: winner={debate.get('winner','?')} bull_score={debate.get('bull_score',0):.1f} bear_score={debate.get('bear_score',0):.1f}")
            math_agent_context = chr(10).join(math_lines)

        # Build P&L history for Brain v2 Kelly sizing
        edge = self.edge_stats.get(asset, {})
        pnl_history_str = f"{edge.get('wins',0)}W/{edge.get('losses',0)}L rate={edge.get('win_rate',0.5):.0%}"

        # Collect recent PnL values for Kelly criterion
        pnl_history_values = []
        try:
            recent_trades = self.journal.load_trades(asset=asset, exchange=self._ex_tag.lower())
            for t in (recent_trades or [])[-50:]:
                pnl_val = t.get('pnl_usd')
                if pnl_val is not None:
                    pnl_history_values.append(float(pnl_val))
        except Exception:
            pass

        # Tell LLM if this is a reversal (counter-trend) trade
        if is_reversal_signal:
            math_filter_warnings.append(f"REVERSAL TRADE: counter-trend entry — price exhausted, expect mean reversion back toward EMA")

        # Build candle text for brain
        n_candles_brain = min(10, len(candle_lines))
        candle_text = chr(10).join(candle_lines[-n_candles_brain:]) if candle_lines else "N/A"

        # Append multi-timeframe candle data (1m, 15m, 1h) for full pattern analysis
        if mtf_candle_summary:
            candle_text = candle_text + chr(10) + chr(10) + "MULTI-TIMEFRAME DATA:" + chr(10) + mtf_candle_summary

        # ══════════════════════════════════════════════════════════
        # INSTITUTIONAL DATA: Price Action + Market Structure + Profit Protector
        # All computed here and injected into LLM context (no hard blocks)
        # ══════════════════════════════════════════════════════════
        institutional_context = ""

        # ── Price Action: FVG + Order Blocks ──
        if self._price_action and len(closes) >= 20:
            try:
                fvgs = self._price_action.get_fvg(highs, lows)
                obs = self._price_action.get_order_blocks(opens, highs, lows, closes, volumes)
                supports_pa, resistances_pa = self._price_action.get_support_resistance(highs, lows, closes)

                pa_lines = []
                # Recent FVGs (last 5)
                recent_fvgs = [f for f in fvgs if f.get('index', 0) >= len(closes) - 30][-5:]
                if recent_fvgs:
                    for fvg in recent_fvgs:
                        filled = "FILLED" if (fvg['type'] == 'bullish' and price < fvg['top']) or (fvg['type'] == 'bearish' and price > fvg['bottom']) else "OPEN"
                        dist = abs(price - (fvg['top'] + fvg['bottom']) / 2) / price * 100
                        pa_lines.append(f"  FVG: {fvg['type'].upper()} gap ${fvg['bottom']:.2f}-${fvg['top']:.2f} (size=${fvg['size']:.2f}) [{filled}] {dist:.1f}% away")

                # Recent Order Blocks (last 3)
                recent_obs = [ob for ob in obs if ob.get('index', 0) >= len(closes) - 30][-3:]
                if recent_obs:
                    for ob in recent_obs:
                        dist = abs(price - (ob['top'] + ob['bottom']) / 2) / price * 100
                        pa_lines.append(f"  ORDER BLOCK: {ob['type'].upper()} zone ${ob['bottom']:.2f}-${ob['top']:.2f} (vol={ob['volume']:.0f}) {dist:.1f}% away")

                # S/R levels
                if supports_pa:
                    nearest_sup = max([s for s in supports_pa if s < price], default=None)
                    if nearest_sup:
                        pa_lines.append(f"  SUPPORT CLUSTER: ${nearest_sup:.2f} ({(price - nearest_sup)/price*100:.2f}% below)")
                if resistances_pa:
                    nearest_res = min([r for r in resistances_pa if r > price], default=None)
                    if nearest_res:
                        pa_lines.append(f"  RESISTANCE CLUSTER: ${nearest_res:.2f} ({(nearest_res - price)/price*100:.2f}% above)")

                if pa_lines:
                    institutional_context += chr(10) + "INSTITUTIONAL LIQUIDITY ZONES (FVG + Order Blocks):" + chr(10) + chr(10).join(pa_lines)
                    # Add warnings for dangerous proximity
                    for ob in recent_obs:
                        dist = abs(price - (ob['top'] + ob['bottom']) / 2) / price * 100
                        if dist < 0.5:
                            if (signal == "BUY" and ob['type'] == 'bearish') or (signal == "SELL" and ob['type'] == 'bullish'):
                                math_filter_warnings.append(f"ORDER BLOCK: {ob['type']} OB at ${ob['bottom']:.2f}-${ob['top']:.2f} — price entering opposing liquidity zone")
            except Exception as e:
                logger.debug(f"Price Action analysis error for {asset}: {e}")

        # ── Market Structure: BOS/CHoCH + Pivot Points ──
        if self._market_structure and len(closes) >= 20:
            try:
                pivots = self._market_structure.find_pivots(highs, lows)
                breaks = self._market_structure.detect_structure_breaks(pivots, price)

                ms_lines = []
                # Last 4 pivots
                recent_pivots = pivots[-4:] if pivots else []
                if recent_pivots:
                    pivot_str = " -> ".join([f"{p.type}(${p.price:.2f})" for p in recent_pivots])
                    ms_lines.append(f"  PIVOTS: {pivot_str}")

                    # Determine structural trend
                    last = recent_pivots[-1]
                    if last.type in ('HH', 'HL'):
                        ms_lines.append(f"  STRUCTURE: BULLISH (last pivot={last.type})")
                    elif last.type in ('LH', 'LL'):
                        ms_lines.append(f"  STRUCTURE: BEARISH (last pivot={last.type})")

                if breaks.get('bos'):
                    ms_lines.append(f"  ** BREAK OF STRUCTURE (BOS) ** — significant level violated")
                if breaks.get('choch'):
                    ms_lines.append(f"  ** CHANGE OF CHARACTER (CHoCH) ** — potential trend reversal")

                if ms_lines:
                    institutional_context += chr(10) + "MARKET STRUCTURE (BOS/CHoCH):" + chr(10) + chr(10).join(ms_lines)

                    # Structure warnings
                    if breaks.get('choch'):
                        math_filter_warnings.append("CHoCH DETECTED: Change of Character — trend may be reversing")
                    if breaks.get('bos') and signal == "BUY" and recent_pivots and recent_pivots[-1].type in ('LH', 'LL'):
                        math_filter_warnings.append("BEARISH BOS: structure broke down but signal is BUY — fighting structure")
                    if breaks.get('bos') and signal == "SELL" and recent_pivots and recent_pivots[-1].type in ('HH', 'HL'):
                        math_filter_warnings.append("BULLISH BOS: structure broke up but signal is SELL — fighting structure")
            except Exception as e:
                logger.debug(f"Market Structure analysis error for {asset}: {e}")

        # ── Profit Protector: Trade Quality Rating (context for LLM) ──
        profit_protector_context = ""
        if self._profit_protector:
            try:
                self._profit_protector.update_balance(self.equity)
                profit_status = self._profit_protector.get_profit_status()

                # Rate this potential trade
                sl_dist = current_atr * 1.5
                if signal == "BUY":
                    sl_price = price - sl_dist
                    tp_price = price + sl_dist * 3.0  # 3:1 R:R target
                else:
                    sl_price = price + sl_dist
                    tp_price = price - sl_dist * 3.0

                win_rate = edge.get('win_rate', 0.5)
                quality = self._profit_protector.rate_trade_quality(
                    signal_confidence=0.6,  # placeholder — LLM sets real conf
                    model_win_rate=win_rate,
                    entry_price=price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    position_size=1.0,
                    current_balance=self.equity,
                )

                pp_lines = []
                pp_lines.append(f"  P&L Status: ${profit_status['total_pnl']:+.2f} ({profit_status['total_pnl_pct']:+.1f}%) | Peak Drawdown: {profit_status['underwater_pct']:.1f}%")
                pp_lines.append(f"  Trade Quality: {quality.quality_score:.0f}/100 | Recommendation: {quality.recommendation}")
                pp_lines.append(f"  Win Probability: {quality.win_probability:.1%} | P(loss): {1-quality.win_probability:.1%}")
                pp_lines.append(f"  Risk/Reward: {quality.risk_reward_ratio:.1f}:1 | Expected Value: ${quality.profit_expectancy:.2f}")
                profit_protector_context = chr(10) + "PROFIT PROTECTOR ANALYSIS:" + chr(10) + chr(10).join(pp_lines)

                # Only add warnings for extreme cases (not routine — LLM sees full context in candle_text)
                if profit_status['underwater_pct'] > 8:
                    math_filter_warnings.append(f"DRAWDOWN NOTE: {profit_status['underwater_pct']:.1f}% below peak")
            except Exception as e:
                logger.debug(f"Profit Protector error for {asset}: {e}")

        # ── Hurst + VPIN regime context for LLM ──
        regime_context = ""
        regime_lines = []
        if hurst_regime != 'unknown':
            regime_lines.append(f"  Hurst exponent: H={hurst_value:.2f} ({hurst_regime})")
            if hurst_regime == 'trending':
                regime_lines.append(f"  -> Market IS trending — EMA crossover signals are reliable")
            elif hurst_regime == 'random':
                regime_lines.append(f"  -> Market is random walk — trend may not persist, tighter targets")
        if vpin_status:
            regime_lines.append(f"  VPIN: {vpin_status['vpin']:.2f} (threshold: {vpin_status['threshold']}) action: {vpin_status['risk_action']}")
            if vpin_status['is_toxic']:
                regime_lines.append(f"  -> TOXIC flow detected — informed traders active, reduce size or skip")
        # Add ML model predictions to regime context — INTERPRETED for LLM
        if ml_context:
            regime_lines.append("  --- ML PATTERN ANALYSIS (trained on real market data) ---")
            # HMM Regime — tell LLM what it means for the strategy
            if 'hmm_regime' in ml_context:
                r = ml_context['hmm_regime']
                c = ml_context.get('hmm_confidence', 0)
                cp = ml_context.get('crisis_probability', 0)
                if r == 'BULL':
                    regime_lines.append(f"  HMM: BULL regime ({c:.0%} conf) -> CALL trades favored, ride trend with wider trailing SL")
                elif r == 'BEAR':
                    regime_lines.append(f"  HMM: BEAR regime ({c:.0%} conf) -> PUT trades favored, CALL trades risky (trend against you)")
                elif r == 'SIDEWAYS':
                    regime_lines.append(f"  HMM: SIDEWAYS regime ({c:.0%} conf) -> BOTH directions risky, expect L1-L2 deaths, SKIP or reduce size")
                elif r == 'CRISIS':
                    regime_lines.append(f"  HMM: CRISIS regime ({c:.0%} conf, crisis_prob={cp:.2f}) -> EXTREME volatility, only trade with trend, tight SL")
            # Kalman — trend direction and signal-noise
            if 'kalman_slope' in ml_context:
                ks = ml_context['kalman_slope']
                ksnr = ml_context.get('kalman_snr', 0)
                kt = ml_context.get('kalman_trend', 'FLAT')
                if ksnr > 2.0:
                    regime_lines.append(f"  KALMAN: Strong {kt} trend (slope={ks:+.4f}, SNR={ksnr:.1f}) -> trend is clean, go WITH it")
                elif ksnr > 1.0:
                    regime_lines.append(f"  KALMAN: Moderate {kt} trend (slope={ks:+.4f}, SNR={ksnr:.1f}) -> trend exists but noisy")
                else:
                    regime_lines.append(f"  KALMAN: No clear trend (slope={ks:+.4f}, SNR={ksnr:.1f}) -> choppy, high L1 death risk")
            # GARCH volatility
            if 'vol_regime' in ml_context:
                vr = ml_context['vol_regime']
                vp = ml_context.get('vol_percentile', 50)
                if vr == 'HIGH' or vr == 'EXTREME':
                    regime_lines.append(f"  GARCH: {vr} volatility (p{vp:.0f}) -> wider stops needed, bigger moves possible but also bigger fakeouts")
                elif vr == 'LOW':
                    regime_lines.append(f"  GARCH: LOW volatility (p{vp:.0f}) -> breakout likely coming, tight range = explosive move ahead")
                else:
                    regime_lines.append(f"  GARCH: NORMAL volatility (p{vp:.0f})")
            # LSTM Ensemble consensus
            if 'lstm_signal' in ml_context:
                ls = ml_context['lstm_signal']
                lc = ml_context.get('lstm_confidence', 0)
                if ls == 'TRADE':
                    regime_lines.append(f"  LSTM ENSEMBLE (3 neural nets): TRADE SIGNAL ({lc:.0%} conf) -> setup matches profitable patterns (trailing SL will lock profit)")
                elif ls == 'SKIP':
                    regime_lines.append(f"  LSTM ENSEMBLE (3 neural nets): SKIP SIGNAL ({lc:.0%} conf) -> setup matches L1 death patterns (likely stopped at loss)")
                else:
                    regime_lines.append(f"  LSTM ENSEMBLE (3 neural nets): {ls} ({lc:.0%} conf)")
            # PatchTST transformer
            if 'patchtst_direction' in ml_context:
                pd_dir = ml_context['patchtst_direction']
                pu = ml_context.get('patchtst_prob_up', 0.5)
                ps = ml_context.get('patchtst_shock_prob', 0)
                regime_lines.append(f"  PatchTST TRANSFORMER: {pd_dir} (prob_up={pu:.0%}, shock_risk={ps:.0%})")
                if ps > 0.3:
                    regime_lines.append(f"    WARNING: {ps:.0%} liquidity shock probability -> reduce size, expect sudden reversal")
            # LightGBM pattern classifier
            if 'lgbm_direction' in ml_context:
                ld = ml_context['lgbm_direction']
                lc = ml_context.get('lgbm_confidence', 0)
                if ld == 'TRADE':
                    regime_lines.append(f"  LIGHTGBM (30 strategy features): TRADE ({lc:.0%} conf) -> setup matches profitable L2+ patterns, trailing SL will lock profits")
                elif ld == 'SKIP':
                    regime_lines.append(f"  LIGHTGBM (30 strategy features): SKIP ({lc:.0%} conf) -> setup matches L1 death patterns, likely stopped at loss")
                else:
                    regime_lines.append(f"  LIGHTGBM: predicts {ld} ({lc:.0%} conf) -> {'pattern matches L3+ runners' if ld not in ('FLAT', 'SKIP') else 'pattern matches L1 deaths, SKIP'}")
            # Alpha Decay — how fresh is the signal?
            if 'alpha_freshness' in ml_context:
                af = ml_context['alpha_freshness']
                ah = ml_context.get('alpha_optimal_hold', 0)
                regime_lines.append(f"  ALPHA DECAY: signal freshness={af:.0%}, optimal hold={ah} bars -> {'signal is fresh, good entry' if af > 0.7 else 'signal is stale, late entry risk'}")
            # Cycle phase
            if 'cycle_phase' in ml_context:
                cp = ml_context['cycle_phase']
                cd = ml_context.get('dominant_cycle', 0)
                regime_lines.append(f"  CYCLE: {cp} phase (period={cd} bars) -> {'RISING/BOTTOM=buy dips' if cp in ('BOTTOM', 'RISING') else 'TOP/FALLING=sell rallies'}")
            # ML consensus summary
            ml_agree = 0
            ml_disagree = 0
            signal_is_buy = signal == "BUY"
            # LSTM now outputs TRADE/SKIP (not direction)
            if 'lstm_signal' in ml_context:
                lstm_val = ml_context['lstm_signal']
                if lstm_val == 'TRADE':
                    ml_agree += 1  # TRADE = setup looks profitable
                elif lstm_val == 'SKIP':
                    ml_disagree += 1  # SKIP = L1 death predicted
            # LightGBM still outputs direction
            for key in ['lgbm_direction', 'patchtst_direction']:
                if key in ml_context:
                    val = ml_context[key]
                    if (signal_is_buy and val in ('LONG', 'UP')) or (not signal_is_buy and val in ('SHORT', 'DOWN')):
                        ml_agree += 1
                    elif val not in ('FLAT', 'NEUTRAL'):
                        ml_disagree += 1
            if ml_agree + ml_disagree > 0:
                regime_lines.append(f"  ML CONSENSUS: {ml_agree} models AGREE, {ml_disagree} DISAGREE -> {'strong ML confirmation — trailing SL will lock profits' if ml_agree > ml_disagree else 'ML models conflict - HIGH RISK, likely L1 death'}")
            # Meta Controller summary for LLM
            if 'meta_direction' in ml_context:
                _md = ml_context['meta_direction']
                _mc = ml_context.get('meta_confidence', 0)
                _ms = ml_context.get('meta_position_scale', 1.0)
                _md_label = {1: 'LONG', -1: 'SHORT', 0: 'FLAT'}.get(_md, 'FLAT')
                regime_lines.append(f"  META-CTRL ARBITRATION: {_md_label} (conf={_mc:.0%}, position_scale={_ms:.2f}) -> LGB+RL+PatchTST fused decision")
            # Signal Combiner summary for LLM
            if 'combined_signal' in ml_context:
                _cs = ml_context['combined_signal']
                _ca = ml_context.get('combined_action', 'hold')
                _cc = ml_context.get('combined_confidence', 0)
                regime_lines.append(f"  SIGNAL COMBINER: L1+L2+L3 fusion -> signal={_cs:+.3f} action={_ca.upper()} (conf={_cc:.0%})")

        if regime_lines:
            regime_context = chr(10) + "REGIME ANALYSIS:" + chr(10) + chr(10).join(regime_lines)

        # Append institutional + protector + regime + math agents data to candle text for LLM
        if institutional_context:
            candle_text = candle_text + chr(10) + institutional_context
        if profit_protector_context:
            candle_text = candle_text + chr(10) + profit_protector_context
        if regime_context:
            candle_text = candle_text + chr(10) + regime_context
        if math_agent_context:
            candle_text = candle_text + chr(10) + math_agent_context

        # ── BTC-ETH Pairs Trading Context for LLM (informational) ──
        if pairs_signal and pairs_signal.get('context_line'):
            pairs_context = chr(10) + "PAIRS TRADING (BTC-ETH Cointegration):" + chr(10) + "  " + pairs_signal['context_line']
            if pairs_signal.get('signal') in ('BUY_BTC_REL_ETH', 'SELL_BTC_REL_ETH'):
                pairs_context += chr(10) + "  -> Use this as ADDITIONAL context for trade direction, not a direct trade signal"
            candle_text = candle_text + pairs_context

        # ── Indicator Suite Context for LLM ──
        if indicator_context:
            ind_lines = ["TECHNICAL INDICATORS (confirmed candle):"]
            for key, val in indicator_context.items():
                ind_lines.append(f"  {key}: {val}")
            candle_text = candle_text + chr(10) + chr(10).join(ind_lines)

        # ── Adaptive Feedback Context: feed learned patterns into LLM ──
        _adaptive_ctx = {}
        if self._adaptive:
            try:
                _regime_hint = ml_context.get('hmm_regime', 'UNKNOWN') if 'ml_context' in dir() else 'UNKNOWN'
                _adaptive_ctx = self._adaptive.get_adaptive_context(asset, _regime_hint)
                # Inject winner/loser DNA into LLM prompt
                _dna_lines = []
                if _adaptive_ctx.get('winner_dna'):
                    _dna_lines.append(f"WINNING PATTERNS (from {_adaptive_ctx['total_trades']} trades):")
                    _dna_lines.append(f"  {_adaptive_ctx['winner_dna']}")
                if _adaptive_ctx.get('loser_dna'):
                    _dna_lines.append(f"LOSING PATTERNS (avoid these):")
                    _dna_lines.append(f"  {_adaptive_ctx['loser_dna']}")
                if _adaptive_ctx.get('rolling_win_rate', 0.5) != 0.5:
                    _dna_lines.append(f"ADAPTIVE STATS: WR={_adaptive_ctx['rolling_win_rate']:.0%} "
                                      f"asset_WR={_adaptive_ctx['asset_win_rate']:.0%} "
                                      f"regime_WR={_adaptive_ctx['regime_win_rate']:.0%}")
                if _dna_lines:
                    candle_text = candle_text + chr(10) + chr(10).join(_dna_lines)
            except Exception:
                pass

        # ── Self-Evolving Overlay: apply adaptive risk/sizing multipliers ──
        _evo_overrides = {}
        if self._evolution_overlay:
            try:
                _evo_overrides = self._evolution_overlay.get_overrides()
            except Exception:
                pass

        if self._brain:
            # ── BRAIN v2: full 7-layer evaluation ──
            print(f"  [{self._ex_tag}:{asset}] >>> BRAIN v2 evaluating (2 models + CoT + regime + memory)...")
            unified = self._brain.evaluate_trade(
                asset=asset, signal=signal, price=price,
                entry_score=entry_score, slope_pct=slope_pct,
                min_trend_bars=min_trend_bars, ema_separation=ema_separation,
                ema_direction=ema_direction, htf_alignment=htf_alignment,
                closes=closes, volumes=volumes,
                current_atr=current_atr, current_ema=current_ema,
                support=support, resistance=resistance,
                candle_lines=candle_text,
                orch_result=orch_result,
                pnl_history=pnl_history_values,
                math_filter_warnings=math_filter_warnings,
                score_reasons=score_reasons,
                htf_1h_direction=htf_1h_direction,
                htf_4h_direction=htf_4h_direction,
                is_reversal_signal=is_reversal_signal,
                equity=self.equity,
                exchange_client=self._exchange_client,
                edge_stats=edge,
            )
            # Log brain details
            bd = unified.get('brain_details', {})
            print(f"  [{self._ex_tag}:{asset}] BRAIN: regime={bd.get('regime','?')} consensus={bd.get('consensus','?')} kelly={bd.get('kelly_size',0):.1f}% session={bd.get('session_multiplier',1.0):.1f}x")
            print(f"  [{self._ex_tag}:{asset}] BRAIN: funding={bd.get('funding_signal','?')} memory={str(bd.get('memory_summary',''))[:80]}")
            votes = bd.get('model_votes', {})
            for model_name, vote in votes.items():
                print(f"  [{self._ex_tag}:{asset}] MODEL {model_name}: proceed={vote.get('proceed')} conf={vote.get('confidence',0):.2f} risk={vote.get('risk_score','?')}")
        else:
            # ── LEGACY: single unified LLM call ──
            unified = self._query_unified_llm(
                asset=asset, signal=signal, price=price,
                current_ema=current_ema, current_atr=current_atr,
                ema_direction=ema_direction, ema_slope_pct=ema_slope_pct,
                consecutive_trend=consecutive_trend, candle_lines=candle_lines,
                support=support, resistance=resistance,
                closes=closes, highs=highs, lows=lows,
                volumes=volumes, orch_result=orch_result,
                pnl_history=pnl_history_str,
                math_filter_warnings=math_filter_warnings,
                entry_score=entry_score,
                score_reasons=score_reasons,
                min_trend_bars=min_trend_bars,
                slope_pct=slope_pct,
                ema_separation=ema_separation,
                htf_1h_direction=htf_1h_direction,
                htf_4h_direction=htf_4h_direction,
                htf_alignment=htf_alignment,
                mtf_signal_block=mtf_signal_block,
                active_tf_signals=active_tf_signals,
            )

        # Extract unified decision
        _quality_override_active = False
        if not unified.get('proceed', False):
            risk_score = unified.get('risk_score', 5)
            tq = unified.get('trade_quality', 3)
            pred_l = unified.get('predicted_l_level', '?')
            fac = str(unified.get('facilitator_verdict', ''))[:80]

            # ── QUALITY OVERRIDE: LLM rejected but setup has exceptional merit ──
            # Only fires when LLM reasoning does NOT contain explicit skip/conflict keywords
            # AND the setup clears much higher quality/confidence bars.
            # This preserves the LLM's autonomy — if it says "skip", we skip.
            _conf = unified.get('confidence', 0.3)
            _reasoning_text = (
                str(unified.get('facilitator_verdict', '')) + ' ' +
                str(unified.get('reasoning', '')) + ' ' +
                str(unified.get('bull_case', '')) + ' ' +
                str(unified.get('bear_case', ''))
            ).lower()
            # Keywords that signal the LLM explicitly wants to skip — respect these always
            _llm_skip_keywords = [
                'skip', 'conflicting', 'not recommended', 'avoid', 'do not enter',
                'should be skipped', 'caution', 'proceed=false', 'conflicting signals',
                'lack of', 'no clear', 'not a clear', 'bearish', 'risk of reversal',
                'own risk', 'not confident',
            ]
            _llm_explicitly_skipping = any(kw in _reasoning_text for kw in _llm_skip_keywords)

            # Only override if: LLM didn't explicitly say skip AND quality is genuinely high
            if not _llm_explicitly_skipping and tq >= 7 and risk_score <= 5 and _conf >= 0.60:
                _quality_override_active = True
                unified['proceed'] = True
                unified['position_size_pct'] = max(1.0, unified.get('position_size_pct', 3) * 0.3)
                unified['confidence'] = max(_conf, 0.85)
                if self._longs_only:
                    unified['chosen_direction'] = 'CALL'
                print(f"  [{self._ex_tag}:{asset}] QUALITY OVERRIDE: LLM rejected but quality={tq}/10 risk={risk_score}/10 conf={_conf:.2f} → entering {'LONG (forced)' if self._longs_only else ''} conf={unified['confidence']:.2f} size={unified['position_size_pct']:.0f}%")
            else:
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
        size_pct = unified.get('position_size_pct', 3)
        reasoning = str(unified.get('facilitator_verdict', ''))[:120]
        risk_score = unified.get('risk_score', 5)

        # ── Apply Adaptive Feedback Multipliers ──
        if _adaptive_ctx:
            _conf_mult = _adaptive_ctx.get('confidence_multiplier', 1.0)
            _size_mult = _adaptive_ctx.get('size_multiplier', 1.0)
            if _conf_mult != 1.0:
                confidence = max(0.1, min(1.0, confidence * _conf_mult))
            if _size_mult != 1.0:
                size_pct = max(1.0, size_pct * _size_mult)
                print(f"  [{self._ex_tag}:{asset}] ADAPTIVE: size *{_size_mult:.2f} conf *{_conf_mult:.2f} (WR={_adaptive_ctx.get('rolling_win_rate', 0.5):.0%})")

        # ── Apply Self-Evolving Overlay Risk Adjustments ──
        if _evo_overrides:
            _evo_size_mult = _evo_overrides.get('risk_params', {}).get('size_mult', 1.0)
            if _evo_size_mult != 1.0:
                size_pct = max(1.0, size_pct * _evo_size_mult)
                print(f"  [{self._ex_tag}:{asset}] EVOLVE: size *{_evo_size_mult:.2f}")

        # ── EVENT GUARD: calendar-based risk advisory (warns during known high-risk events) ──
        # NOTE: Changed from hard-block to advisory — recurring events use midnight as
        # event time which is inaccurate (FOMC is afternoon ET, not midnight).
        # The LLM already sees market conditions; halving position size is safer than
        # blocking a 0.94-conf trade at the wrong time.
        if self._event_guard:
            try:
                if self._event_guard.is_risk_high():
                    size_pct = max(1.0, size_pct * 0.5)  # Halve position size
                    print(f"  [{self._ex_tag}:{asset}] EVENT GUARD ADVISORY: high-risk calendar event — position halved to {size_pct:.1f}%")
                if self._event_guard.paused:
                    print(f"  [{self._ex_tag}:{asset}] EVENT GUARD: manually paused — blocking entry")
                    return
            except Exception as e:
                logger.debug(f"EventGuard error: {e}")

        # ── META OVERLAY: use adaptive thresholds if available (SECONDARY knobs only) ──
        # v13/v14 core (min/max_entry_score, short_score_penalty, sr_assets) is NEVER changed
        _active_bear_veto = self.bear_veto_threshold
        _active_bear_reduce = self.bear_reduce_threshold
        _active_min_confidence = self.min_confidence
        _meta_risk_mult = 1.0
        if _overlay:
            _active_bear_veto = _overlay.bear_veto_threshold
            _active_bear_reduce = _overlay.bear_reduce_threshold
            _active_min_confidence = _overlay.min_confidence
            _meta_risk_mult = _overlay.risk_multiplier
            if _active_bear_veto != self.bear_veto_threshold or _active_min_confidence != self.min_confidence:
                print(f"  [{self._ex_tag}:{asset}] META OVERLAY: bear_veto={_active_bear_veto} bear_reduce={_active_bear_reduce} min_conf={_active_min_confidence:.2f} risk_mult={_meta_risk_mult:.2f} | {_overlay.reasoning[:60]}")

        # ══════════════════════════════════════════════════════════
        # BEAR VETO AGENT — Separate LLM call, contrarian prompt
        # Same model, different perspective: "What could go WRONG?"
        # Thresholds dynamically adjusted by meta-optimizer overlay
        # ══════════════════════════════════════════════════════════
        if self.bear_enabled and not _quality_override_active:
            try:
                bear_result = self._query_bear_agent(
                    asset=asset, signal=signal, price=price,
                    current_ema=current_ema, current_atr=current_atr,
                    ema_direction=ema_direction, ema_slope_pct=ema_slope_pct,
                    consecutive_trend=consecutive_trend, candle_lines=candle_lines,
                    support=support, resistance=resistance,
                    closes=closes, highs=highs, lows=lows,
                    volumes=volumes, bull_confidence=confidence,
                    bull_reasoning=reasoning,
                )
                bear_risk = bear_result.get('risk_score', 5)
                bear_reason = bear_result.get('reasoning', '')[:100]
                print(f"  [{self._ex_tag}:{asset}] BEAR AGENT: risk={bear_risk}/10 | {bear_reason}")

                # Use the HIGHER risk score between unified LLM and bear agent
                if bear_risk > risk_score:
                    print(f"  [{self._ex_tag}:{asset}] BEAR OVERRIDE: risk {risk_score} -> {bear_risk} (bear sees more danger)")
                    risk_score = bear_risk

                # VETO: risk >= threshold (adaptive from meta overlay)
                if bear_risk >= _active_bear_veto:
                    if asset not in self.bear_veto_stats:
                        self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
                    self.bear_veto_stats[asset]['vetoed'] += 1
                    print(f"  [{self._ex_tag}:{asset}] BEAR VETO: risk={bear_risk}/10 >= {_active_bear_veto} | {bear_reason}")
                    return
            except Exception as e:
                logger.warning(f"[{asset}] Bear agent error (proceeding without veto): {e}")
        elif self.bear_enabled and _quality_override_active:
            print(f"  [{self._ex_tag}:{asset}] BEAR AGENT SKIPPED: quality override active (would veto everything in ranging market)")

        # ── Bear REDUCE: risk between reduce and veto thresholds (adaptive) ──
        if risk_score >= _active_bear_reduce and risk_score < _active_bear_veto:
            old_size = size_pct
            size_pct = max(1.0, size_pct * 0.7)  # Floor at 1% — never reduce to 0
            if asset not in self.bear_veto_stats:
                self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
            self.bear_veto_stats[asset]['reduced'] += 1
            print(f"  [{self._ex_tag}:{asset}] BEAR REDUCE: risk={risk_score}/10 (>={_active_bear_reduce}) — size {old_size:.0f}% -> {size_pct:.0f}%")

        # ── VPIN toxic flow → warn but don't reduce further (bear already handles risk) ──
        if vpin_status and vpin_status['is_toxic']:
            print(f"  [{self._ex_tag}:{asset}] VPIN WARNING: toxic flow {vpin_status['vpin']:.2f} — LLM informed (no additional size cut)")

        # LLM IS THE BRAIN — its confidence is the final confidence
        # No more Python-side blending or overriding
        print(f"  [{self._ex_tag}:{asset}] LLM DECISION: conf={confidence:.2f} size={size_pct:.0f}% risk={risk_score}/10 quality={unified.get('trade_quality', 0)}/10 hurst={hurst_value:.2f}")

        # Direction from LLM's chosen direction (multi-TF aware) or fallback to 5m signal
        chosen_dir = unified.get('chosen_direction', 'CALL' if signal == 'BUY' else 'PUT')
        action = "LONG" if chosen_dir == "CALL" else "SHORT"
        direction_label = chosen_dir
        _default_signal_tf = self.SIGNAL_TIMEFRAMES[0] if self.SIGNAL_TIMEFRAMES else '5m'
        chosen_tf = unified.get('chosen_timeframe', _default_signal_tf)
        if chosen_tf not in self.SIGNAL_TIMEFRAMES:
            chosen_tf = _default_signal_tf

        # ── POST-LLM DIRECTION GATE ──
        # SHORT penalty must ALWAYS apply when final action is SHORT,
        # regardless of original 5m signal (BUY/SELL/NEUTRAL).
        # The pre-LLM gate only checks signal=="SELL", but the LLM can flip
        # a BUY or NEUTRAL signal to SHORT, bypassing the penalty.
        #
        # ROBINHOOD SPOT: Shorts are extremely dangerous with 3.3% spread
        # Pre-training data: shorts lose ~70% after spread on both BTC and ETH
        # Block shorts entirely on paper/Robinhood unless score is exceptional
        if action == "SHORT":
            short_penalty = self.config.get('adaptive', {}).get('short_score_penalty', 4)
            min_score = self.config.get('adaptive', {}).get('min_entry_score', 4)
            if self.sniper_enabled:
                min_score = max(min_score, self.sniper_min_score)
            required = min_score + short_penalty  # sniper: 8 + 4 = 12 (nearly impossible)
            if entry_score < required:
                print(f"  [{self._ex_tag}:{asset}] SHORT SCORE BLOCK: score {entry_score} < {required} required for SHORT (LONG bias) -- skipping")
                return

            # ── Robinhood SHORT hard-block ──
            # Robinhood spot has ~0.845% spread. SHORTs lose ~70% after spread.
            # Block ALL shorts on Robinhood/paper unless 1d EMA is confirmed FALLING.
            _htf_1d_dir = 'FLAT'
            if hasattr(self, '_last_tf_signals'):
                _htf_1d_dir = self._last_tf_signals.get(asset, {}).get('1d', {}).get('ema_direction', 'FLAT')
            if self._paper_mode:
                if _htf_1d_dir != 'FALLING':
                    print(f"  [{self._ex_tag}:{asset}] SHORT BLOCKED: 1d EMA not FALLING ({_htf_1d_dir}) -- LONG-only market on Robinhood")
                    return
                # Additional Robinhood SHORT guard: require LLM confidence >= 0.90 for shorts
                if confidence < 0.90:
                    print(f"  [{self._ex_tag}:{asset}] SHORT BLOCKED: confidence {confidence:.2f} < 0.90 required for SHORT on Robinhood")
                    return

        # Quality gate: confidence (adaptive via meta overlay)
        # Skip when quality override is active — override already boosted confidence
        if confidence < _active_min_confidence and not _quality_override_active:
            print(f"  [{self._ex_tag}:{asset}] SKIP: confidence {confidence:.2f} < {_active_min_confidence:.2f}")
            return

        # ── Compute ml_confidence once, reused for the hard gate AND paper record below.
        # Prefer calibrated meta_prob; fall back to raw lgbm_confidence; else 0.0.
        # 0.0 means "ML signal unavailable" (not "ML says no") — gate skips ML checks at 0.0.
        ml_confidence = float(
            ml_context.get('meta_prob')
            or ml_context.get('lgbm_confidence')
            or 0.0
        )

        # ── ROBINHOOD HARD GATE — LLM cannot override these constraints ──
        # Skip when quality override is active (it already verified minimum thresholds,
        # and LGBM entry_score penalties would block every ranging-market override)
        if not _quality_override_active:
            _rh_ok, _rh_reason = self._robinhood_hard_gate(
                asset=asset, action=action, confidence=confidence,
                risk_score=risk_score, trade_quality=unified.get('trade_quality', 5),
                entry_score=entry_score, price=price, atr=current_atr,
                ml_conf=ml_confidence,
            )
            if not _rh_ok:
                print(f"  [{self._ex_tag}:{asset}] ROBINHOOD GATE BLOCKED: {_rh_reason}")
                return
        else:
            # Quality override active — only check ATR move and SHORT block
            if action == "SHORT":
                print(f"  [{self._ex_tag}:{asset}] QUALITY OVERRIDE SHORT BLOCKED: longs only on Robinhood")
                return
            print(f"  [{self._ex_tag}:{asset}] ROBINHOOD GATE SKIPPED: quality override active")

        # Track bear stats
        if asset not in self.bear_veto_stats:
            self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
        self.bear_veto_stats[asset]['passed'] += 1

        # ── Off-hour size reduction (journal-learned) — DISABLED for Robinhood ──
        # Crypto is 24/7, and Robinhood spread demands full position sizes to overcome cost
        if not self._paper_mode and not getattr(self, '_is_profitable_hour', True):
            reduce_factor = self.config.get('filters', {}).get('reduce_size_off_hours', 0.5)
            if reduce_factor < 1.0:
                old_size = size_pct
                size_pct = size_pct * reduce_factor
                print(f"  [{self._ex_tag}:{asset}] OFF-HOUR: UTC {datetime.datetime.utcnow().hour}:00 -- size {old_size:.0f}% -> {size_pct:.0f}% (journal pattern)")

        # ── Edge Positioning: adjust size by historical win rate ──
        if self.edge_enabled and asset in self.edge_stats:
            edge = self.edge_stats[asset]
            if edge['total'] >= 5:
                mult = edge['edge_multiplier']
                old_size = size_pct
                size_pct = size_pct * mult
                if abs(mult - 1.0) > 0.05:
                    print(f"  [{self._ex_tag}:{asset}] EDGE: {mult:.2f}x ({edge['wins']}W/{edge['losses']}L) size {old_size:.0f}% -> {size_pct:.0f}%")

        # ── SNIPER: Profit Compounding — risk profits, protect principal ──
        # After first win, increase position size using accumulated profits
        # Principal stays safe; only profits are used for bigger bets
        _sniper_bonus_equity = 0.0
        if self.sniper_enabled and self.sniper_profit_pool > 0:
            _compound_amount = self.sniper_profit_pool * (self.sniper_compound_pct / 100.0)
            if self.sniper_protect_principal:
                # Only risk profits — don't touch original capital beyond base risk_per_trade
                _sniper_bonus_equity = _compound_amount
                print(f"  [{self._ex_tag}:{asset}] SNIPER COMPOUND: +${_compound_amount:,.2f} from profit pool (${self.sniper_profit_pool:,.2f} total)")
            else:
                _sniper_bonus_equity = _compound_amount

        # ── Portfolio Allocator (cross-asset Kelly sizing) ──
        if self._allocator:
            try:
                alloc_result = self._allocator.calculate_allocation(
                    asset=asset, confidence=confidence,
                    win_rate=0.5, avg_win=0.05, avg_loss=0.03,
                )
                if hasattr(alloc_result, 'position_size_pct') and alloc_result.position_size_pct > 0:
                    alloc_size = alloc_result.position_size_pct
                    if alloc_size < size_pct:
                        print(f"  [{self._ex_tag}:{asset}] ALLOCATOR: reduced size {size_pct:.0f}%→{alloc_size:.1f}% (Kelly constraint)")
                        size_pct = alloc_size
            except Exception:
                pass

        # ── Adaptive Engine feedback (update strategy performance on close) ──
        if self._adaptive_engine:
            try:
                adaptive_signal = self._adaptive_engine.generate_adaptive_signal(
                    prices=closes, highs=highs, lows=lows, volumes=volumes,
                    sentiment_score=ml_context.get('sentiment_score', 0),
                    hmm_regime=ml_context.get('hmm_regime', 'UNKNOWN'),
                )
                if adaptive_signal and adaptive_signal.get('strategy_selected'):
                    ml_context['adaptive_strategy'] = adaptive_signal['strategy_selected']
                    ml_context['adaptive_signal'] = adaptive_signal.get('signal', 0)
            except Exception:
                pass

        # ── Caution-marker size reduction (LLM-reasoning text parser) ──
        # When the LLM's prose flags caution/risk/reversal concerns but its
        # confidence number is still high, trust the prose. Strictly size-reducing:
        # 2+ markers → halve, 1 marker → 0.75x. Never increases size, never blocks.
        _caution_markers = ('caution', 'risk', 'careful', 'reversal',
                            'macro', 'uncertain', 'concern', 'however', 'beware')
        _reasoning_lc = (reasoning or '').lower()
        _caution_hits = sum(1 for m in _caution_markers if m in _reasoning_lc)
        if _caution_hits >= 2:
            _old_sz = size_pct
            size_pct = max(1.0, size_pct * 0.5)
            print(f"  [{self._ex_tag}:{asset}] CAUTION MARKERS ({_caution_hits}): size {_old_sz:.0f}% -> {size_pct:.0f}%")
        elif _caution_hits == 1:
            _old_sz = size_pct
            size_pct = max(1.0, size_pct * 0.75)
            print(f"  [{self._ex_tag}:{asset}] CAUTION MARKER (1): size {_old_sz:.0f}% -> {size_pct:.0f}%")

        # Calculate position size — ATR-based dynamic sizing when available, fixed % fallback
        max_size_pct = 20 if self.equity < 500 else 5
        size_pct = max(1, min(max_size_pct, size_pct))
        if self.equity <= 0:
            print(f"  [{self._ex_tag}:{asset}] SKIP: no equity available")
            return

        # ── ATR-based position sizing (risk-calibrated) ──
        # Uses ATR to compute position size such that a 2xATR adverse move = risk_per_trade% loss
        # This replaces LLM's flat position_size_pct with mathematically correct sizing
        atr_sized = False
        if POSITION_SIZING_AVAILABLE and current_atr > 0 and price > 0:
            try:
                risk_pct = self.config.get('risk', {}).get('risk_per_trade_pct', 1.0)
                # Match ATR multiplier to ACTUAL SL distance from config (atr_stop_mult)
                # Old bug: sized for 2xATR but SL placed at 3xATR = 50% more risk than intended
                _atr_sl_mult = self.config.get('risk', {}).get('atr_stop_mult', 3.0)
                atr_qty = atr_position_size(
                    account_balance=self.equity,
                    atr_value=current_atr,
                    risk_pct=risk_pct,
                    atr_multiplier=_atr_sl_mult,  # Must match actual SL distance
                )
                atr_notional = atr_qty * price
                # Clamp ATR-sized notional to max_size_pct of equity
                max_notional = self.equity * (max_size_pct / 100.0)
                atr_notional = min(atr_notional, max_notional)

                # ── MetaSizer: Half-Kelly multiplier based on win probability ──
                kelly_mult = 1.0
                if self._meta_sizer:
                    try:
                        # Use edge stats for win probability, or LightGBM confidence as proxy
                        win_prob = 0.5
                        win_loss_ratio = 2.0
                        if asset in self.edge_stats and self.edge_stats[asset].get('total', 0) >= 5:
                            win_prob = self.edge_stats[asset].get('win_rate', 0.5)
                            wins = self.edge_stats[asset].get('wins', 1)
                            losses = self.edge_stats[asset].get('losses', 1)
                            # Approximate win/loss ratio from session performance
                            if losses > 0 and wins > 0:
                                win_loss_ratio = max(0.5, min(5.0, wins / losses))
                        kelly_mult = self._meta_sizer.size({}, win_prob=win_prob, win_loss_ratio=win_loss_ratio)
                        atr_notional *= kelly_mult
                    except Exception:
                        kelly_mult = 1.0

                # Use the SMALLER of LLM-suggested and ATR-sized (conservative)
                llm_notional = self.equity * (size_pct / 100.0)
                notional = min(llm_notional, atr_notional)
                atr_size_pct = (atr_notional / self.equity) * 100 if self.equity > 0 else 0
                kelly_tag = f" x Kelly={kelly_mult:.2f}" if kelly_mult < 0.99 else ""
                print(f"  [{self._ex_tag}:{asset}] ATR SIZING: risk={risk_pct}% x {_atr_sl_mult}xATR=${current_atr:,.2f}{kelly_tag} -> ${atr_notional:,.0f} ({atr_size_pct:.1f}%) | LLM={size_pct:.0f}% -> using {'ATR' if atr_notional < llm_notional else 'LLM'}")
                atr_sized = True
            except Exception as sz_err:
                logger.debug(f"ATR sizing error: {sz_err}")

        if not atr_sized:
            notional = self.equity * (size_pct / 100.0)

        # ── SNIPER: Add compound bonus to notional (profits risked on top of base position) ──
        # Capped at 50% of base notional to prevent oversized positions
        if _sniper_bonus_equity > 0:
            old_notional = notional
            _max_compound = notional * 0.5  # Never more than 50% bonus
            _sniper_bonus_equity = min(_sniper_bonus_equity, _max_compound)
            notional += _sniper_bonus_equity
            print(f"  [{self._ex_tag}:{asset}] SNIPER SIZE: ${old_notional:,.0f} + ${_sniper_bonus_equity:,.0f} compound = ${notional:,.0f}")

        # ── META OVERLAY: apply risk multiplier to position size (SECONDARY knob) ──
        if _meta_risk_mult != 1.0:
            old_notional = notional
            notional *= _meta_risk_mult
            if abs(_meta_risk_mult - 1.0) > 0.05:
                print(f"  [{self._ex_tag}:{asset}] META RISK: {_meta_risk_mult:.2f}x — notional ${old_notional:,.0f} -> ${notional:,.0f}")

        # ── ROBINHOOD-HARDENING sizing multipliers (B + C) ──
        # Conviction tier: sniper=3x, normal=1x. Cached by _evaluate_entry's
        # conviction gate. Macro bias size multiplier: 0.5-1.5x based on layer
        # alignment; further faded if direction opposes bias sign.
        _conv_tier = getattr(self, '_last_conviction_tier', {}).get(asset)
        _macro = getattr(self, '_last_macro_bias', {}).get(asset)
        if _conv_tier in ('sniper', 'normal') and _macro is not None:
            try:
                from src.trading.macro_bias import apply_direction_alignment
                _conv_mult = 3.0 if _conv_tier == 'sniper' else 1.0
                _macro_mult = apply_direction_alignment(_macro, action, _macro.size_multiplier)
                _combined = _conv_mult * _macro_mult
                if abs(_combined - 1.0) > 0.05:
                    old_notional = notional
                    notional *= _combined
                    print(f"  [{self._ex_tag}:{asset}] HARDEN SIZE: tier={_conv_tier}({_conv_mult}x) "
                          f"macro({_macro_mult:.2f}x) -> combined {_combined:.2f}x  "
                          f"${old_notional:,.0f} -> ${notional:,.0f}")
            except Exception as _hs:
                logger.debug(f"[HARDEN] sizing multiplier failed: {_hs}")

        # Hard cap: max 5% of equity — prevents catastrophic position sizing
        # March 31: $270K-$764K positions on $30K equity = instant blowup
        max_trade = min(2000.0, self.equity * 0.05)
        notional = min(notional, max_trade)

        print(f"  [{self._ex_tag}:{asset}] SIZING: ${notional:,.0f} of ${self.equity:,.0f} (max ${max_trade:,.0f})")

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
            # Safety: don't exceed 50% of equity (except minimum 1 contract)
            if actual_notional > self.equity * 0.50 and qty > 1:
                qty = max(1, int(self.equity * 0.50 / contract_value))
                actual_notional = qty * contract_value
                print(f"  [{self._ex_tag}:{asset}] CAPPED to {qty} contracts (${actual_notional:,.2f})")
            min_qty = {'BTC': 1, 'ETH': 1}
        else:
            # Bybit: coin-based sizing
            min_qty = {'BTC': 0.001, 'ETH': 0.01}

        asset_min = min_qty.get(asset, 1 if self._exchange_name == 'delta' else 0.001)
        if qty < asset_min:
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

        # Execution type from config — LIMIT preferred (better fills, no spread cost)
        entry_type = self.config.get('execution', {}).get('entry_type', 'limit')

        if entry_type == 'limit':
            # Smart limit: place slightly inside the spread for fast fill
            # BUY → place at best ask (or slightly below) to get filled as taker
            # SELL → place at best bid (or slightly above) to get filled as taker
            try:
                ob_snap = self.price_source.fetch_order_book(self._get_symbol(asset), limit=5)
                best_ask = float(ob_snap['asks'][0][0]) if ob_snap.get('asks') else price * 1.001
                best_bid = float(ob_snap['bids'][0][0]) if ob_snap.get('bids') else price * 0.999
                spread_pct = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

                if side == 'buy':
                    # Place at best ask — guarantees immediate fill like market order
                    # but protects against slippage beyond this level
                    order_price = best_ask
                else:
                    # Place at best bid — same logic for sells
                    order_price = best_bid

                print(f"  [{self._ex_tag}:{asset}] {direction_label}: {side.upper()} {qty:.6f} LIMIT@${order_price:,.2f} spread={spread_pct:.3f}% (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")
            except Exception:
                # Fallback to market if OB fetch fails
                entry_type = 'market'
                order_price = None
                print(f"  [{self._ex_tag}:{asset}] {direction_label}: {side.upper()} {qty:.6f} MARKET-fallback (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")
        else:
            order_price = None  # CRITICAL: market orders must NOT have a price
            print(f"  [{self._ex_tag}:{asset}] {direction_label}: {side.upper()} {qty:.6f} MARKET (${notional:,.0f} = {size_pct:.0f}% of ${self.equity:,.0f})")

        # ── TRADE PROTECTIONS: Pre-entry check (SL guard, drawdown, pair lock, spread, drift) ──
        if self._protections:
            try:
                open_count = len(self.positions)
                prot_check = self._protections.pre_entry_check(
                    asset=asset, signal_price=price, current_price=price,
                    bid=0, ask=0,  # Will be filled from OB below
                    open_trade_count=open_count, equity=self.equity,
                )
                if not prot_check.get('allowed', True):
                    reasons = '; '.join(prot_check.get('reasons', []))
                    print(f"  [{self._ex_tag}:{asset}] PROTECTION BLOCK: {reasons}")
                    return
            except Exception as e:
                logger.warning(f"[{asset}] Protection pre-entry error (proceeding): {e}")

        # PRE-CHECK: Liquidity sanity check
        ob_check = self.price_source.fetch_order_book(self._get_symbol(asset), limit=10)
        max_dev = price * 0.05  # 5% deviation max

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

        # ── SPREAD vs EXPECTED MOVE CHECK ──
        # Advisory: warn if expected move is tight relative to spread, but don't hard-block
        # The LLM already evaluates spread economics and the TP is 25x ATR for swing trades
        if self._round_trip_spread > 1.0 and price > 0:
            atr_tp = self.config.get('risk', {}).get('atr_tp_mult', 4.5)
            _filter_atr = current_atr
            if hasattr(self, '_last_chosen_tf_atr') and self._last_chosen_tf_atr.get(asset, 0) > 0:
                _filter_atr = self._last_chosen_tf_atr[asset]
            if _filter_atr > 0:
                expected_move_pct = (_filter_atr * atr_tp / price) * 100
                round_trip_spread = spread_pct * 2
                _edge_ratio = expected_move_pct / round_trip_spread if round_trip_spread > 0 else 99
                if expected_move_pct < round_trip_spread * 1.2:
                    print(f"  [{self._ex_tag}:{asset}] SPREAD WARNING: expected move {expected_move_pct:.2f}% barely clears spread {round_trip_spread:.2f}% (edge={_edge_ratio:.1f}x)")
                else:
                    print(f"  [{self._ex_tag}:{asset}] SPREAD OK: expected move {expected_move_pct:.2f}% vs spread cost {round_trip_spread:.2f}% (edge={_edge_ratio:.1f}x)")

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

        symbol = self._get_symbol(asset)

        # ── Paper Mode (Robinhood): simulate fill at real bid/ask ──
        if self._paper_mode:
            # Fill at ask for LONG, bid for SHORT (realistic paper fills)
            if side == 'buy':
                fill_price = best_ask
            else:
                fill_price = best_bid
            order_id = f"paper_{asset}_{int(time.time())}"
            price = fill_price
            print(f"  [{self._ex_tag}:{asset}] PAPER FILL: {side.upper()} {qty:.6f} @ ${fill_price:,.2f} (spread={spread_pct:.2f}%)")
        else:
            # ── Live Mode: Place real order (circuit breaker protected) ──
            result = self._api_call(
                self.price_source.place_order,
                symbol=symbol,
                side=side,
                amount=qty,
                order_type=entry_type,
                price=order_price,
            )

            # If market order rejected, retry as limit at best ask/bid
            if result.get('status') != 'success' and '30208' in str(result.get('message', '')):
                try:
                    ob = self._api_call(self.price_source.fetch_order_book, symbol, limit=10)
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
                        result = self._api_call(
                            self.price_source.place_order,
                            symbol=symbol, side=side, amount=qty,
                            order_type='limit', price=limit_price,
                        )
                    else:
                        print(f"  [{self._ex_tag}:{asset}] Market rejected & no reasonable limit price within 5% -- SKIP")
                except Exception as e:
                    print(f"  [{self._ex_tag}:{asset}] Limit fallback failed: {e}")

            if result.get('status') != 'success':
                print(f"  [{self._ex_tag}:{asset}] ORDER FAILED: {result.get('message', result)}")
                return

            order_id = result.get('order_id', 'unknown')

            # Get ACTUAL fill price from exchange (not signal price)
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

        # Compute initial stop-loss (L1) using ORDER BOOK + ATR
        # Scale SL bounds by chosen timeframe (higher TF = wider SL)
        tf_min_pct = self.TF_SL_MIN_PCT.get(chosen_tf, 0.005)
        tf_max_pct = self.TF_SL_MAX_PCT.get(chosen_tf, 0.02)

        # Use chosen TF's ATR if available
        init_atr = current_atr
        try:
            tf_sig = (active_tf_signals or {}).get(chosen_tf)
            if tf_sig and tf_sig.get('current_atr', 0) > 0:
                init_atr = tf_sig['current_atr']
        except Exception:
            pass

        # ════════════════════════════════════════════════════════════
        # L1 INITIAL SL: EMA LINE as PRIMARY, ATR as EMERGENCY FALLBACK
        # ════════════════════════════════════════════════════════════
        sl_distance_emergency = init_atr * 3.0
        sl_distance_emergency = max(sl_distance_emergency, price * tf_min_pct)
        sl_distance_emergency = min(sl_distance_emergency, price * tf_max_pct * 1.5)

        ema_sl_buffer = init_atr * 1.0
        current_ema_val = ema_vals[-2] if len(ema_vals) >= 2 else None

        sl_source = "ATR_EMERGENCY"
        if action == 'LONG':
            sl_price = price - sl_distance_emergency
            if current_ema_val and current_ema_val > 0:
                ema_sl = current_ema_val - ema_sl_buffer
                ema_dist_pct = (price - ema_sl) / price * 100
                if 0.1 <= ema_dist_pct <= 3.0 and ema_sl < price:
                    sl_price = ema_sl
                    sl_source = f"EMA_LINE@${current_ema_val:,.2f}"
            bid_wall = ob_levels.get('bid_wall', 0)
            if bid_wall > 0 and bid_wall < price:
                ob_sl = bid_wall * 0.999
                ob_dist_pct = (price - ob_sl) / price * 100
                if 0.3 <= ob_dist_pct <= 2.0 and ob_sl > sl_price:
                    sl_price = ob_sl
                    sl_source = f"OB_BID_WALL@${bid_wall:,.2f}"
        else:  # SHORT
            sl_price = price + sl_distance_emergency
            if current_ema_val and current_ema_val > 0:
                ema_sl = current_ema_val + ema_sl_buffer
                ema_dist_pct = (ema_sl - price) / price * 100
                if 0.1 <= ema_dist_pct <= 3.0 and ema_sl > price:
                    sl_price = ema_sl
                    sl_source = f"EMA_LINE@${current_ema_val:,.2f}"
            ask_wall = ob_levels.get('ask_wall', 0)
            if ask_wall > 0 and ask_wall > price:
                ob_sl = ask_wall * 1.001
                ob_dist_pct = (ob_sl - price) / price * 100
                if 0.3 <= ob_dist_pct <= 2.0 and ob_sl < sl_price:
                    sl_price = ob_sl
                    sl_source = f"OB_ASK_WALL@${ask_wall:,.2f}"

        # Safe-entries gate (A + C): widen SL to floor + enforce min R:R.
        if getattr(self, '_safe_enabled', False):
            try:
                from src.trading import safe_entries as _safe
                rt_spread = float(self.config.get('exchanges', [{}])[0].get('round_trip_spread_pct', 0.1)
                                  if isinstance(self.config.get('exchanges'), list) else 0.1)
                # A: widen if inside spread
                new_sl, floor_reason = _safe.apply_stop_floor(
                    entry=price,
                    sl_price=sl_price,
                    direction=action,
                    atr=float(init_atr),
                    rt_spread_pct=rt_spread,
                    config=self._safe_config,
                )
                if floor_reason != "floor_ok":
                    print(f"  [{self._ex_tag}:{asset}] SAFE SL {floor_reason}: ${sl_price:.2f} -> ${new_sl:.2f}")
                    sl_price = new_sl
                    sl_source = f"SAFE_FLOOR({floor_reason})"
                # C: enforce min R:R — synthesize TP at min_rr × risk
                min_rr = float(self._safe_config.get('min_rr', 2.0))
                tp_target = _safe.synthesize_tp(price, sl_price, action, min_rr)
                ok, rr, rr_reason = _safe.check_rr(price, sl_price, tp_target, action, min_rr)
                if not ok:
                    # synthesize_tp guarantees rr==min_rr, so this branch should only
                    # fire on a pathological SL (NaN, wrong-side). In LIVE mode the
                    # entry order has already been placed — we must unwind it via a
                    # reduce_only close to avoid an orphan position. In paper mode
                    # no real order exists so a plain return is safe.
                    print(f"  [{self._ex_tag}:{asset}] SAFE REJECT: {rr_reason} (tp=${tp_target:.2f})")
                    if not self._paper_mode:
                        try:
                            symbol = self._get_symbol(asset)
                            close_side = 'sell' if action == 'LONG' else 'buy'
                            self._api_call(
                                self.price_source.place_order,
                                symbol=symbol, side=close_side, amount=qty,
                                order_type='market', price=None, reduce_only=True,
                            )
                            print(f"  [{self._ex_tag}:{asset}] SAFE UNWIND: reduce_only close emitted on SAFE REJECT")
                        except Exception as _ue:
                            logger.warning(f"[{asset}] SAFE unwind failed: {_ue} (manual intervention may be needed)")
                    return
                print(f"  [{self._ex_tag}:{asset}] SAFE R:R={rr:.2f} tp_target=${tp_target:.2f}")
                self._safe_tp_target = tp_target  # stored for partial-take trigger
            except Exception as _se:
                print(f"  [{self._ex_tag}:{asset}] SAFE SL check error: {_se}")

        imb = ob_levels.get('imbalance', 0)
        print(f"  [{self._ex_tag}:{asset}] ORDER OK [{chosen_tf}]: {order_id} | SL L1=${sl_price:,.2f} ({sl_source}) | OB imbalance={imb:+.2f}")

        # Track sniper entry
        if self.sniper_enabled:
            self.sniper_stats['entered'] += 1

        sl_order_id = None

        # ── Capture ML model predictions for Model Accuracy telemetry ──
        # Convert each model's prediction to "predicts-profitable" axis (+1/-1/0):
        #   +1 = model expects this trade to work out
        #   -1 = model expects this trade to fail
        #    0 = model didn't run or was NEUTRAL/FLAT (skipped in stats)
        _mlp_lgbm_raw = ml_context.get('lgbm_direction', 'FLAT')
        if _mlp_lgbm_raw == 'TRADE':
            _mlp_lgbm = 1
        elif _mlp_lgbm_raw == 'SKIP':
            _mlp_lgbm = -1
        elif _mlp_lgbm_raw in ('LONG', 'SHORT'):
            _mlp_lgbm = (1 if _mlp_lgbm_raw == 'LONG' else -1) * (1 if action == 'LONG' else -1)
        else:
            _mlp_lgbm = 0

        _mlp_patch_raw = ml_context.get('patchtst_direction', 'NEUTRAL')
        if _mlp_patch_raw == 'UP':
            _mlp_patch = 1 if action == 'LONG' else -1
        elif _mlp_patch_raw == 'DOWN':
            _mlp_patch = -1 if action == 'LONG' else 1
        else:
            _mlp_patch = 0

        _mlp_rl_enter = ml_context.get('rl_enter')
        if _mlp_rl_enter is True:
            _mlp_rl = 1
        elif _mlp_rl_enter is False:
            _mlp_rl = -1
        else:
            _mlp_rl = 0

        _ml_predictions_snapshot = {
            'lightgbm': _mlp_lgbm,
            'patchtst': _mlp_patch,
            'rl_agent': _mlp_rl,
        }

        # ml_confidence is already computed earlier (before the Robinhood hard gate)
        # and reused here for the paper record. See block above the gate call.

        # Record position (with chosen timeframe for scaled SL management)
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
            'predicted_l_level': unified.get('predicted_l_level', '?'),
            'risk_score': risk_score,
            'bear_risk': risk_score if self.bear_enabled else 0,
            'hurst': hurst_value,
            'hurst_regime': hurst_regime,
            'breakeven_moved': False,
            # Safe-entries partial-take state: True once we've closed 50% at +1R
            # and moved SL to breakeven. Prevents retriggers on the same position.
            'partialled': False,
            'is_reversal': is_reversal_signal,
            'trade_timeframe': chosen_tf,   # LLM chose this TF
            'agent_votes': orch_result.get('agent_votes', {}) if orch_result else {},
            'entry_tag': self._protections.tagger.tag_entry(
                signal=signal, entry_score=entry_score,
                regime=unified.get('brain_details', {}).get('regime', ''),
                htf_alignment=htf_alignment, is_reversal=is_reversal_signal,
                ema_slope=slope_pct, consensus=unified.get('brain_details', {}).get('consensus', '')
            ) if self._protections else 'untagged',
            'dca_count': 0,
            'rl_state': ml_context.get('_rl_state', None),        # RL state at entry (for learning)
            'rl_action_idx': ml_context.get('_rl_action_idx', 0),  # RL action chosen at entry
            'ml_predictions': _ml_predictions_snapshot,           # Model Accuracy telemetry
        }
        self.last_trade_time[asset] = time.time()

        # Shadow-log the prediction alongside this new position. Guarded internally
        # by is_enabled() so it's a no-op unless ACT_META_SHADOW_MODE=1.
        try:
            from src.ml import shadow_log as _slog
            _feats = ml_context.get('meta_features_snapshot') or []
            _p_raw = ml_context.get('meta_prob_raw')
            _p_cal = ml_context.get('meta_prob')
            _thr = ml_context.get('meta_take_threshold')
            if _feats and _p_raw is not None and _p_cal is not None and _thr is not None:
                _slog.log_predict(
                    trade_id=str(order_id), asset=asset, direction=action,
                    entry_price=float(price), entry_score=int(entry_score),
                    meta_prob_raw=float(_p_raw), meta_prob_cal=float(_p_cal),
                    take_threshold=float(_thr), features=_feats,
                )
        except Exception:
            pass
        print(f"  [{self._ex_tag}:{asset}] POSITION STORED: {action} {qty:.6f} @ ${price:,.2f} | positions={list(self.positions.keys())}")

        # ── Persist position for crash recovery ──
        if self._sl_persist:
            try:
                self._sl_persist.save_position(asset, self.positions[asset])
            except Exception:
                pass

        # ── Paper Trade: Record entry with real Robinhood price ──
        if self._paper:
            try:
                self._paper.record_entry(
                    asset=asset, direction=action, price=price,
                    score=entry_score, quantity=qty,
                    sl_price=sl_price, tp_price=0,
                    ml_confidence=ml_confidence,
                    llm_confidence=confidence,
                    size_pct=size_pct, reasoning=reasoning[:200] if reasoning else '',
                )
                self._paper.save_state()  # Immediate save for dashboard
            except Exception as _pe:
                logger.warning(f"[PAPER] Entry record failed: {_pe}")

        # L7+L8: Order Generation + Execution log
        try:
            _ds = DashboardState()
            _ds.add_layer_log('L7', f"{asset}: {action} order generated (size={size_pct:.0f}%, SL=${sl_price:,.2f})", "info")
            _ds.add_layer_log('L8', f"{asset}: {action} executed @ ${price:,.2f} qty={qty:.6f}", "info")
            _ds.add_layer_log('L9', f"{asset}: trade recorded (score={entry_score}, conf={confidence:.2f}, TF={chosen_tf})", "info")
        except Exception:
            pass

        # ── Alert: Trade Entry ──
        self._send_alert('INFO', f'{self._ex_tag} ENTRY {action} {asset}',
            f'{action} {asset} @ ${price:,.2f} | size={size_pct:.0f}% | conf={confidence:.2f} | risk={risk_score}/10 | TF={chosen_tf}',
            {'asset': asset, 'direction': action, 'price': price, 'size_pct': size_pct,
             'confidence': confidence, 'risk_score': risk_score, 'timeframe': chosen_tf})

        # ── Dynamic Risk: Register trade for monitoring (NO stop management) ──
        if self._dynamic_risk:
            try:
                heat = sum(1 for _ in self.positions) / max(len(self.assets), 1)
                allowed, reason = self._dynamic_risk.check_trade_allowed(
                    asset, size_pct / 100.0, heat)
                if not allowed:
                    print(f"  [{self._ex_tag}:{asset}] DRM WARNING: {reason} (trade already placed, monitoring)")
                self._dynamic_risk.update_pnl(0)  # Register activity
            except Exception as drm_err:
                logger.debug(f"DRM register error: {drm_err}")

        # ── MT5: Mirror/execute trade on MetaTrader 5 (skip in paper mode) ──
        if self._mt5 and self._mt5.connected and not self._paper_mode:
            try:
                if self._mt5.mode == 'execute':
                    mt5_result = self._mt5.open_position(
                        asset=asset, direction=action, price=price,
                        qty=qty, sl=sl_price,
                        comment=f"S{entry_score}_C{confidence:.0%}_{chosen_tf}"
                    )
                    if mt5_result.get('status') == 'success':
                        self.positions[asset]['mt5_ticket'] = mt5_result.get('order_id')
                else:
                    self._mt5.mirror_open(
                        asset=asset, direction=action, price=price,
                        qty=qty, sl=sl_price,
                        entry_score=entry_score, confidence=confidence
                    )
            except Exception as e:
                print(f"  [MT5:{asset}] Bridge entry failed: {e}")

    # ------------------------------------------------------------------
    # Position management — Aggressive Trailing SL (L1→L2→...→L38+)
    # Core idea: profit becomes investment, investment becomes safe
    # On every favorable tick, push SL forward so losses come from profits only
    # ------------------------------------------------------------------
    def _robinhood_hard_gate(self, asset: str, action: str, confidence: float,
                              risk_score: int, trade_quality: int, entry_score: int,
                              price: float, atr: float, ml_conf: float = 0.0) -> tuple:
        """
        HARD CONSTRAINTS for Robinhood — LLM cannot override these.
        These are math-verified gates based on spread economics.

        ml_conf: calibrated ML confidence (meta_prob or lgbm_confidence). 0.0 means
        "unavailable" — divergence checks below skip when ml_conf == 0.0 to preserve
        existing behavior on code paths where ML signal isn't computed.

        Returns:
            (proceed: bool, reason: str)
        """
        # Only apply on high-spread exchanges
        if not hasattr(self, '_round_trip_spread') or self._round_trip_spread <= 1.0:
            return True, "low-spread exchange — no Robinhood gates"

        # 1. LONGS ONLY (belt-and-suspenders — also checked earlier)
        if action == "SHORT":
            return False, "SHORT blocked on Robinhood spot"

        # 2. ENTRY SCORE >= 0 (lowered from 5→3→0 — ML models at 51% conf give -2/-3 penalties
        #    too aggressively; real spread protection is gate #3 ATR check below)
        # The multi-strategy engine + LLM + genetic engine provide additional conviction
        if entry_score < 0:
            return False, f"entry_score {entry_score} < 0 required for Robinhood LONG"

        # 3. EXPECTED MOVE > 1.5× SPREAD (ATR-based projected move must justify spread)
        # With 25x ATR TP, BTC/ETH typically get 5-8% expected moves
        # At 1.69% round-trip spread, 1.5x = 5.01% — reasonable for swing trades
        atr_tp_mult = self.config.get('risk', {}).get('atr_tp_mult', 25.0)
        if price > 0 and atr > 0:
            atr_move_pct = (atr * atr_tp_mult / price) * 100
            min_move = self._round_trip_spread * 1.5  # Need 1.5× spread for viable trade
            if atr_move_pct < min_move:
                return False, f"ATR move {atr_move_pct:.1f}% < {min_move:.1f}% (1.5x spread needed)"

        # 4. CONFIDENCE >= 0.50 (LLM returns 0.45-0.65 in ranging markets; 0.65 blocked everything)
        if confidence < 0.50:
            return False, f"confidence {confidence:.2f} < 0.50 required on Robinhood"

        # 5. TRADE QUALITY >= 4 (no garbage setups on high-spread exchange)
        if trade_quality < 4:
            return False, f"quality {trade_quality} < 4 required on Robinhood"

        # 6. RISK SCORE <= 7 (market ranges naturally push risk to 6; only block extreme risk)
        if risk_score > 7:
            return False, f"risk {risk_score} > 7 max on Robinhood"

        # 7. LLM/ML DIVERGENCE FILTER (H1) — only apply when ML signal is available.
        # ml_conf == 0.0 means "ML model didn't run / unavailable" → skip ML checks.
        if ml_conf > 0.0:
            # 7a. Hard floor: if ML actively says "no" (calibrated prob < 0.10), don't enter
            #     even on a high-LLM-confidence signal. Catastrophic-disagreement guard.
            if ml_conf < 0.10:
                return False, f"ml_conf {ml_conf:.2f} < 0.10 (ML model says no, vetoes LLM-only signal)"
            # 7b. Joint floor: both LLM and ML must clear 0.35.
            if min(confidence, ml_conf) < 0.35:
                return False, f"min(llm_conf={confidence:.2f}, ml_conf={ml_conf:.2f}) < 0.35 (joint-floor)"
            # 7c. Strong-divergence guard: if the two disagree by > 0.45, refuse the entry.
            #     Smaller divergence is OK — sizing is reduced upstream by caution-marker parser.
            if abs(confidence - ml_conf) > 0.45:
                return False, f"|llm_conf - ml_conf| = {abs(confidence - ml_conf):.2f} > 0.45 (strong divergence)"

        return True, "passed all Robinhood gates"

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

        # ── Spread-adjusted PnL for ratchet decisions ──
        # On high-spread exchanges, ratchet levels must reflect TRUE profit after spread
        _spread_adj = self._round_trip_spread if hasattr(self, '_round_trip_spread') and self._round_trip_spread > 0.5 else 0
        ratchet_pnl = pnl_pct - _spread_adj  # True PnL after spread cost

        # ── Safe-entries partial-take at +1R (intervention G) ──
        # Close 50% of the position when reward equals the initial risk, and move
        # SL to breakeven on the remainder. Raises realized Sharpe by capping the
        # left tail on trades that move then reverse, while leaving the runner.
        # Fires at most once per position via pos['partialled'].
        if getattr(self, '_safe_enabled', False) and not pos.get('partialled', False):
            try:
                from src.trading import safe_entries as _safe
                take = _safe.maybe_partial_take(
                    entry=entry, current_price=price, sl=sl,
                    direction=direction,
                    already_partialled=False,
                    config=self._safe_config,
                )
                if take is not None:
                    new_sl_be, fraction, take_reason = take
                    partial_qty = pos['qty'] * float(fraction)
                    close_side = 'sell' if direction == 'LONG' else 'buy'
                    # Paper mode: adjust pos['qty'] in-memory; paper fill tracker will
                    # record the realized PnL on next tick. Live mode: reduce_only order.
                    ok = True
                    if not self._paper_mode:
                        try:
                            symbol = self._get_symbol(asset)
                            result = self._api_call(
                                self.price_source.place_order,
                                symbol=symbol, side=close_side, amount=partial_qty,
                                order_type='market', price=None, reduce_only=True,
                            )
                            ok = (result.get('status') == 'success')
                            if not ok:
                                logger.warning(f"[{asset}] SAFE partial-take order failed: {result.get('message','')}")
                        except Exception as _pe:
                            ok = False
                            logger.warning(f"[{asset}] SAFE partial-take exception: {_pe}")
                    if ok:
                        realized_pnl_pct = ((price - entry) / entry * 100.0) if direction == 'LONG' \
                                           else ((entry - price) / entry * 100.0)
                        realized_usd = partial_qty * abs(price - entry) * (1 if realized_pnl_pct > 0 else -1)
                        pos['qty'] -= partial_qty
                        pos['partialled'] = True
                        if new_sl_be > 0:
                            # Move SL to breakeven; ratchet will keep tightening from here
                            pos['sl'] = new_sl_be
                            sl = new_sl_be  # local variable used below must reflect the new SL
                            pos['breakeven_moved'] = True
                        print(f"  [{self._ex_tag}:{asset}] SAFE PARTIAL-TAKE ({take_reason}): "
                              f"closed {fraction*100:.0f}% ({partial_qty:.6f}) @ ${price:,.2f} "
                              f"PnL={realized_pnl_pct:+.2f}% ~${realized_usd:+.2f}, SL->BE=${new_sl_be:,.2f}")
                        # Log to paper tracker if in paper mode so it shows up in the journal
                        if self._paper_mode and self._paper:
                            try:
                                self._paper.record_partial_exit(
                                    asset=asset, direction=direction,
                                    entry_price=entry, exit_price=price,
                                    qty_closed=partial_qty, pnl_pct=realized_pnl_pct,
                                    reason=f"safe_{take_reason}",
                                ) if hasattr(self._paper, 'record_partial_exit') else None
                            except Exception:
                                pass
            except Exception as _se:
                logger.debug(f"[{asset}] safe-entries partial-take error: {_se}")

        # OB imbalance — tracked for display
        ob_imbalance = ob_levels.get('imbalance', 0)

        # ══════════════════════════════════════════════════════════════
        # ROBINHOOD EXIT RULES (Fixes 2, 5, 6)
        # On high-spread exchanges, DIFFERENT exit logic applies:
        # - NO exit before minimum profit (Fix 2)
        # - NO exit before minimum hold time (Fix 5)
        # - Trail at 50% of max profit after clearing spread (Fix 6)
        # ══════════════════════════════════════════════════════════════
        position_age_min = (time.time() - pos.get('entry_time', time.time())) / 60.0

        # Hard stop must be defined before Robinhood exit rules (they reference it)
        _default_hard_stop = -5.0 if self._paper_mode else -1.8
        hard_stop_pct = self.config.get('risk', {}).get('hard_stop_pct', _default_hard_stop)

        if self._rh_min_profit_exit > 0 or self._rh_min_hold_minutes > 0:
            _rh_net_pnl = pnl_pct - self._round_trip_spread  # True P&L after spread

            # Fix 5: Minimum hold time (24h on Robinhood)
            if position_age_min < self._rh_min_hold_minutes:
                # Only allow hard stop exit, block everything else
                if pnl_pct > hard_stop_pct:
                    # Not at hard stop — keep holding
                    if int(position_age_min) % 60 == 0:  # Log every hour
                        print(f"  [{self._ex_tag}:{asset}] HOLD: {position_age_min:.0f}/{self._rh_min_hold_minutes}min | P&L={pnl_pct:+.2f}% (net={_rh_net_pnl:+.2f}%) | min hold not reached")
                    # Skip ALL other exit logic — only hard stop can fire
                    # (fall through to hard stop check below)

            # Fix 2: Minimum profit before exit
            if _rh_net_pnl < self._rh_min_profit_exit and _rh_net_pnl > -abs(hard_stop_pct):
                # Not yet at minimum profit AND not at hard stop — keep holding
                pass  # Let it fall through to hard stop only

            # Fix 6: Robinhood trailing after clearing spread
            if _rh_net_pnl >= self._rh_min_profit_exit:
                # We've cleared the spread + minimum profit — now trail
                _peak_net = ((peak - entry) / entry * 100) - self._round_trip_spread if direction == 'LONG' else ((entry - peak) / entry * 100) - self._round_trip_spread
                _trail_level = _peak_net * (self._rh_trailing_lock_pct / 100.0)
                if _rh_net_pnl < _trail_level and _trail_level > 0:
                    print(f"  [{self._ex_tag}:{asset}] RH TRAILING EXIT: net P&L {_rh_net_pnl:+.2f}% < trail {_trail_level:+.2f}% (50% of peak {_peak_net:+.2f}%)")
                    self._close_position(asset, price, f"RH trailing exit (net={_rh_net_pnl:+.2f}% < trail={_trail_level:+.2f}%)")
                    return

            # Authority-mandated per-asset max hold (BTC=10d swing ceiling,
            # ETH/alts=48h intraday ceiling). Overrides the legacy global
            # _rh_max_hold_days when the authority cap is tighter — ETH can
            # NEVER swing per the authority PDF, so this takes precedence.
            try:
                from src.ai.authority_rules import get_max_hold_hours
                _authority_cap_hours = get_max_hold_hours(asset)
            except Exception:
                _authority_cap_hours = self._rh_max_hold_days * 24.0
            _legacy_cap_hours = self._rh_max_hold_days * 24.0
            _effective_cap_hours = min(_authority_cap_hours, _legacy_cap_hours)
            _cap_source = "AUTHORITY" if _authority_cap_hours <= _legacy_cap_hours else "LEGACY"

            if position_age_min > _effective_cap_hours * 60.0:
                print(
                    f"  [{self._ex_tag}:{asset}] MAX HOLD EXIT ({_cap_source}): "
                    f"held {position_age_min/60.0:.1f}h > {_effective_cap_hours:.0f}h cap "
                    f"| P&L={pnl_pct:+.2f}% (net={_rh_net_pnl:+.2f}%)"
                )
                self._close_position(
                    asset, price,
                    f"{_cap_source} max hold {_effective_cap_hours:.0f}h (net={_rh_net_pnl:+.2f}%)",
                )
                return

        # ── 2. HARD STOP: max loss — non-negotiable ──
        # Robinhood: spread alone is ~0.845%, so -1.8% hard stop kills every trade.
        # Use configurable hard stop: paper mode with wide spread needs -5%+
        _default_hard_stop = -5.0 if self._paper_mode else -1.8
        hard_stop_pct = self.config.get('risk', {}).get('hard_stop_pct', _default_hard_stop)
        # But if asset is blacklisted (stuck, can't close), retry close if profitable or at hard stop
        is_stuck = asset in self.failed_close_assets
        if is_stuck and pnl_pct >= 1.0:
            # Profitable stuck position — retry close aggressively
            print(f"  [{self._ex_tag}:{asset}] STUCK but PROFITABLE ({pnl_pct:+.2f}%) — retrying close")
            self._close_position(asset, price, f"Stuck profitable retry (P&L={pnl_pct:+.2f}%)")
            if asset not in self.positions:  # close succeeded
                self.failed_close_assets.pop(asset, None)
            return
        if pnl_pct <= hard_stop_pct:
            if is_stuck:
                # Retry close even for stuck positions at hard stop — can't just ignore losses
                elapsed = time.time() - self.failed_close_assets.get(asset, 0)
                if elapsed >= 300:  # Retry every 5 min at hard stop (more urgent than normal 10 min)
                    print(f"  [{self._ex_tag}:{asset}] STUCK HARD STOP {pnl_pct:+.2f}% — retrying close")
                    self.failed_close_assets[asset] = time.time()
                    self._close_position(asset, price, f"Stuck hard stop retry ({pnl_pct:+.2f}%)")
                    if asset not in self.positions:
                        self.failed_close_assets.pop(asset, None)
                else:
                    print(f"  [{self._ex_tag}:{asset}] STUCK {pnl_pct:+.2f}% — next retry in {int(300-elapsed)}s")
                return
            print(f"  [{self._ex_tag}:{asset}] HARD STOP at ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
            self._close_position(asset, price, f"Hard stop {pnl_pct:+.1f}%")
            return

        # ── 2b. TIME-BASED EXIT: close zombie positions (Freqtrade pattern) ──
        duration_min = (time.time() - pos.get('entry_time', time.time())) / 60.0
        # TIME EXIT: Only close if held too long AND losing/flat.
        # If profitable, ALWAYS let EMA line exit handle it (ride the trend).
        # Scale loser hold time: paper mode with wide spreads = hold longer (give time to recover from spread)
        max_hold_losers = 1440 if self._paper_mode else 180  # 24hr for Robinhood losers, 3hr for futures
        max_hold_winners = max(self.max_hold_minutes, 720)  # config max or 12hr minimum
        if not is_stuck:
            if pnl_pct <= -0.5 and duration_min >= max_hold_losers:
                print(f"  [{self._ex_tag}:{asset}] TIME EXIT: held {duration_min:.0f}min (max {max_hold_losers:.0f}) P&L={pnl_pct:+.2f}% — closing stale loser")
                self._close_position(asset, price, f"Time exit ({duration_min:.0f}min, P&L={pnl_pct:+.2f}%)")
                return
            elif pnl_pct < 0.5 and duration_min >= max_hold_losers * 1.5:
                print(f"  [{self._ex_tag}:{asset}] TIME EXIT: held {duration_min:.0f}min with tiny P&L={pnl_pct:+.2f}% — closing")
                self._close_position(asset, price, f"Time exit ({duration_min:.0f}min, P&L={pnl_pct:+.2f}%)")
                return
            elif pnl_pct >= 0.5 and duration_min >= max_hold_winners:
                print(f"  [{self._ex_tag}:{asset}] TIME EXIT WINNER: held {duration_min:.0f}min P&L={pnl_pct:+.2f}% — closing")
                self._close_position(asset, price, f"Time exit winner ({duration_min:.0f}min, P&L={pnl_pct:+.2f}%)")
                return
            # If profitable >=0.5% and under max_hold_winners, let EMA line/trailing SL handle it

        # ── 2c. PROTECTION EXITS: ROI table + partial profit taking ──
        if self._protections and not is_stuck:
            try:
                exit_check = self._protections.check_exit(
                    asset=asset, current_pnl_pct=pnl_pct,
                    trade_duration_min=duration_min, sl_level=len(sl_levels)
                )
                if exit_check.get('exit', False):
                    if exit_check.get('partial', False):
                        # Partial exit — reduce position size
                        frac = exit_check['fraction']
                        partial_qty = pos['qty'] * frac
                        reason = exit_check['reason']
                        print(f"  [{self._ex_tag}:{asset}] PARTIAL EXIT: {frac*100:.0f}% ({reason})")
                        # Record the level was taken
                        self._protections.adjuster.record_partial_exit(asset, exit_check.get('level', pnl_pct))
                        # Close partial — reduce qty in position, close fraction on exchange
                        try:
                            symbol = self._get_symbol(asset)
                            close_side = 'sell' if direction == 'LONG' else 'buy'
                            result = self._api_call(
                                self.price_source.place_order,
                                symbol=symbol, side=close_side, amount=partial_qty,
                                order_type='market', price=None, reduce_only=True,
                            )
                            if result.get('status') == 'success':
                                pos['qty'] -= partial_qty
                                print(f"  [{self._ex_tag}:{asset}] PARTIAL OK: closed {partial_qty:.6f}, remaining {pos['qty']:.6f}")
                            else:
                                print(f"  [{self._ex_tag}:{asset}] PARTIAL FAILED: {result.get('message','')}")
                        except Exception as pe:
                            logger.warning(f"[{asset}] Partial exit order failed: {pe}")
                    else:
                        # Full ROI exit
                        reason = exit_check['reason']
                        print(f"  [{self._ex_tag}:{asset}] ROI EXIT: {reason}")
                        self._close_position(asset, price, f"ROI table ({reason})")
                        return
            except Exception as e:
                logger.warning(f"[{asset}] Protection exit check error: {e}")

        # ── 3. Check if current SL is hit ──
        # Use CONFIRMED candle close for SL checks (not live tick)
        # This prevents 10-second wicks from triggering exits in a trending market
        # Hard stop (step 2) still uses live price for safety
        #
        # MINIMUM HOLD: 180s grace period — don't check SL in first 3 minutes
        # Backtest shows L1 SL kills 95% of trades at 30% WR. Trades reaching
        # EMA new-line exit have 68-74% WR. Need to let entries breathe.
        # Hard stop (-2%) still active during grace for emergency protection.
        grace_period_s = 180  # 3 minutes = ~3 bars on 5m (1 bar on 15m)
        position_age_s = time.time() - pos.get('entry_time', time.time())
        if position_age_s < grace_period_s:
            # Still show status but don't check SL
            print(f"  [{self._ex_tag}:{asset}] HOLD {direction} @ ${entry:,.2f} | Now: ${price:,.2f} | P&L: {pnl_pct:+.2f}% | GRACE {grace_period_s - int(position_age_s)}s")
            return

        confirmed_close = closes[-2] if len(closes) >= 2 else price
        sl_hit = False
        if direction == 'LONG' and confirmed_close <= sl:
            sl_hit = True
        elif direction == 'SHORT' and confirmed_close >= sl:
            sl_hit = True

        if sl_hit:
            if is_stuck:
                elapsed = time.time() - self.failed_close_assets.get(asset, 0)
                if elapsed >= 300:
                    print(f"  [{self._ex_tag}:{asset}] STUCK SL {pnl_pct:+.2f}% — retrying close")
                    self.failed_close_assets[asset] = time.time()
                    self._close_position(asset, price, f"Stuck SL retry ({pnl_pct:+.2f}%)")
                    if asset not in self.positions:
                        self.failed_close_assets.pop(asset, None)
                return
            print(f"  [{self._ex_tag}:{asset}] SL {sl_levels[-1]} HIT at ${price:,.2f} (candle closed ${confirmed_close:,.2f} vs SL ${sl:,.2f}) | P&L: {pnl_pct:+.2f}%")
            self._close_position(asset, price, f"SL {sl_levels[-1]} hit (candle close ${confirmed_close:,.2f})")
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
        position_age = time.time() - pos.get('entry_time', time.time())
        is_reversal = pos.get('is_reversal', False)
        trade_tf = pos.get('trade_timeframe', '5m')

        # ── Use chosen timeframe's ATR for SL scaling ──
        # Re-fetch chosen TF candles to get fresh ATR (higher TF = wider ATR)
        current_atr = atr_vals[-1] if atr_vals else price * 0.01  # 5m fallback
        try:
            if trade_tf != '5m':
                symbol = self._get_symbol(asset)
                tf_ohlcv = self._fetch_tf_ohlcv(symbol, asset, trade_tf,
                                                 self.TF_FETCH_LIMITS.get(trade_tf, 30))
                if tf_ohlcv and len(tf_ohlcv.get('closes', [])) >= 15:
                    tf_atr = atr(tf_ohlcv['highs'], tf_ohlcv['lows'], tf_ohlcv['closes'], 14)
                    if tf_atr:
                        current_atr = tf_atr[-1]
        except Exception:
            pass  # Fall back to 5m ATR

        # Ratchet scaling factor — higher TFs need wider thresholds
        ratchet_scale = self.TF_RATCHET_SCALE.get(trade_tf, 1.0)

        # ══════════════════════════════════════════════════════════════
        # RATCHETING BASELINE SL — progressively move line to make
        # investment safe, then lock profit as new investment baseline
        #
        # CALL (LONG): SL line ratchets UPWARD:
        #   Entry → breakeven → lock 15% → lock 30% → lock 50% → lock 60% → lock 70%
        #   Each level makes the locked profit the NEW "safe investment"
        #
        # PUT (SHORT): SL line ratchets DOWNWARD (mirror image):
        #   Entry → breakeven → lock 15% → lock 30% → lock 50% → lock 60% → lock 70%
        #
        # Reversal trades: tighter protection (faster ratchet, lower thresholds)
        #   because reversals are inherently riskier
        # ══════════════════════════════════════════════════════════════

        # ── GARCH-ADJUSTED ATR: adapt trailing SL to volatility regime ──
        # High vol = wider SL (avoid noise stops), Low vol = tighter SL (protect faster)
        garch_atr_factor = 1.0
        try:
            if self._last_ml_context.get('garch_vol_expanding'):
                garch_atr_factor = 1.15  # 15% wider during high vol
            elif self._last_ml_context.get('vol_regime') == 'LOW':
                garch_atr_factor = 0.85  # 15% tighter during low vol
        except Exception:
            pass

        # Define ratchet levels: (min_pnl_pct, protect_pct, atr_mult, label)
        # Scaled by timeframe: higher TF = wider thresholds (bigger moves expected)
        # ratchet_scale: 1m=0.5x, 5m=1.0x, 15m=1.5x, 1h=2.5x, 4h=5.0x
        rs = ratchet_scale
        if is_reversal:
            ratchet_levels = [
                (0.5*rs,  0.0,  2.0, "BREAKEVEN"),
                (1.0*rs,  0.15, 1.8, "LOCK-15%"),
                (1.5*rs,  0.25, 1.5, "LOCK-25%"),
                (2.0*rs,  0.35, 1.3, "LOCK-35%"),
                (3.0*rs,  0.50, 1.0, "LOCK-50%"),
                (5.0*rs,  0.60, 0.8, "LOCK-60%"),
                (10.0*rs, 0.70, 0.6, "LOCK-70%"),
            ]
            min_age_for_breakeven = 300
        else:
            # WIDENED ratchet: EMA line-following is the PRIMARY SL now.
            # Ratchet only kicks in at higher profit levels to lock gains.
            # Old breakeven at 0.3% was killing trades during normal pullbacks.
            ratchet_levels = [
                (1.0*rs,  0.0,  2.0, "BREAKEVEN"),
                (1.5*rs,  0.10, 1.8, "LOCK-10%"),
                (2.0*rs,  0.20, 1.5, "LOCK-20%"),
                (3.0*rs,  0.30, 1.3, "LOCK-30%"),
                (4.0*rs,  0.40, 1.2, "LOCK-40%"),
                (5.0*rs,  0.50, 1.0, "LOCK-50%"),
                (7.0*rs,  0.55, 0.9, "LOCK-55%"),
                (10.0*rs, 0.60, 0.8, "LOCK-60%"),
                (12.0*rs, 0.65, 0.7, "LOCK-65%"),
                (15.0*rs, 0.70, 0.6, "LOCK-70%"),
            ]
            min_age_for_breakeven = 600  # 10 min — give EMA line time to prove trend

        if direction == 'LONG':
            # Walk through ratchet levels from highest to lowest
            # Use spread-adjusted PnL so ratchet levels reflect TRUE profit
            for min_pnl, protect, atr_m, label in reversed(ratchet_levels):
                if ratchet_pnl >= min_pnl:
                    if protect == 0.0:
                        # BREAKEVEN level — move SL to entry
                        if position_age >= min_age_for_breakeven and sl < entry:
                            new_sl = entry
                    else:
                        # PROFIT LOCK level — SL = entry + protect% of (peak - entry)
                        profit_range = peak - entry
                        floor_sl = entry + (profit_range * protect)

                        # ATR trail: peak - multiplier * ATR (gives room for pullbacks)
                        # GARCH adjusts: wider during high vol, tighter during low vol
                        atr_trail_sl = peak - (current_atr * atr_m * garch_atr_factor)

                        # Use whichever is TIGHTER (higher = more protection for LONG)
                        best_sl = max(floor_sl, atr_trail_sl)
                        if best_sl > new_sl and best_sl < price:
                            new_sl = best_sl
                    break  # Only apply highest matching level

            # Additional tightening: swing lows and order book walls
            if pnl_pct >= 1.5:
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

        else:  # SHORT (PUT) — mirror: SL ratchets DOWNWARD
            for min_pnl, protect, atr_m, label in reversed(ratchet_levels):
                if ratchet_pnl >= min_pnl:
                    if protect == 0.0:
                        # BREAKEVEN level — move SL to entry
                        if position_age >= min_age_for_breakeven and sl > entry:
                            new_sl = entry
                    else:
                        # PROFIT LOCK level — SL = entry - protect% of (entry - peak)
                        profit_range = entry - peak
                        floor_sl = entry - (profit_range * protect)

                        # ATR trail: peak + multiplier * ATR
                        # GARCH adjusts: wider during high vol, tighter during low vol
                        atr_trail_sl = peak + (current_atr * atr_m * garch_atr_factor)

                        # Use whichever is TIGHTER (lower = more protection for SHORT)
                        best_sl = min(floor_sl, atr_trail_sl)
                        if best_sl < new_sl and best_sl > price:
                            new_sl = best_sl
                    break

            # Additional tightening: swing highs and order book walls
            if pnl_pct >= 1.5:
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

        # ══════════════════════════════════════════════════════════
        # EMA LINE-FOLLOWING SL — The EMA line IS the trend line
        # Activates AFTER the EMA has had time to build a trend (5+ minutes).
        # Before that, let the trade breathe with only hard stop protection.
        #
        # Once active: SL tracks just below/above the EMA line.
        # Only tightens when EMA has moved in our favor (confirms trend).
        # This is the core of the trend-following strategy:
        #   ENTER on new EMA line → RIDE while price stays on correct side
        #   → EXIT when price crosses back through EMA (SL catches it)
        # ══════════════════════════════════════════════════════════
        ema_follow_min_age = 300  # 5 minutes — EMA needs time to build trend
        if len(ema_vals) >= 3 and position_age >= ema_follow_min_age:
            ema_now = ema_vals[-2]  # Confirmed bar's EMA
            ema_buffer = current_atr * 0.5  # Buffer for wick tolerance

            if direction == 'LONG':
                ema_sl = ema_now - ema_buffer
                # Only tighten if EMA has moved in our favor (trend confirmed)
                if ema_sl > new_sl and ema_sl < price and ema_now > entry:
                    new_sl = ema_sl
            else:
                ema_sl = ema_now + ema_buffer
                if ema_sl < new_sl and ema_sl > price and ema_now < entry:
                    new_sl = ema_sl

        # ══════════════════════════════════════════════════════════
        # ML-DRIVEN TRAILING SL TIGHTENING
        # If ML models detect regime shift or trend exhaustion,
        # tighten SL faster to protect profits (especially for L3+ runners)
        # ══════════════════════════════════════════════════════════
        ml_tighten_reason = None
        try:
            # Kalman slope reversal: if slope direction conflicts with position
            if self._kalman and len(closes) >= 30 and pnl_pct > 0.5:
                import numpy as _np
                k_result = self._kalman.filter(_np.array(closes[-100:]))
                if k_result:
                    k_slope = k_result.get('slope', 0)
                    k_snr = k_result.get('snr', 0)
                    # LONG but Kalman turning down, or SHORT but turning up
                    slope_against = (direction == 'LONG' and k_slope < -0.001 and k_snr > 1.5) or \
                                    (direction == 'SHORT' and k_slope > 0.001 and k_snr > 1.5)
                    if slope_against:
                        # Tighten: move SL to protect 60% of current profit
                        if direction == 'LONG':
                            kalman_sl = entry + (price - entry) * 0.60
                            if kalman_sl > new_sl and kalman_sl < price:
                                new_sl = kalman_sl
                                ml_tighten_reason = f"Kalman slope={k_slope:.4f} reversing"
                        else:
                            kalman_sl = entry - (entry - price) * 0.60
                            if kalman_sl < new_sl and kalman_sl > price:
                                new_sl = kalman_sl
                                ml_tighten_reason = f"Kalman slope={k_slope:.4f} reversing"

            # HMM regime shift: if regime goes BEAR during LONG (or BULL during SHORT)
            if self._hmm and len(closes) >= 50 and pnl_pct > 1.0:
                import numpy as _np
                log_ret = _np.diff(_np.log(_np.array(closes[-100:]) + 1e-12))
                vol_20 = _np.array([_np.std(log_ret[max(0,j-20):j]) for j in range(1, len(log_ret)+1)])
                vol_change = _np.zeros(len(log_ret))
                min_len = min(len(log_ret), len(vol_20), len(vol_change))
                obs = _np.column_stack([log_ret[-min_len:], vol_20[-min_len:], vol_change[-min_len:]])
                regime_result = self._hmm.detect(obs)
                if regime_result:
                    hmm_regime = regime_result.get('regime', 'UNKNOWN')
                    hmm_conf = regime_result.get('confidence', 0)
                    regime_against = (direction == 'LONG' and hmm_regime in ('BEAR', 'CRISIS') and hmm_conf > 0.6) or \
                                     (direction == 'SHORT' and hmm_regime == 'BULL' and hmm_conf > 0.6)
                    if regime_against:
                        # Tighten: move SL to protect 70% of profit (regime shift is serious)
                        if direction == 'LONG':
                            hmm_sl = entry + (price - entry) * 0.70
                            if hmm_sl > new_sl and hmm_sl < price:
                                new_sl = hmm_sl
                                ml_tighten_reason = f"HMM regime={hmm_regime} ({hmm_conf:.0%}) against LONG"
                        else:
                            hmm_sl = entry - (entry - price) * 0.70
                            if hmm_sl < new_sl and hmm_sl > price:
                                new_sl = hmm_sl
                                ml_tighten_reason = f"HMM regime={hmm_regime} ({hmm_conf:.0%}) against SHORT"

            # Hurst: if regime shifted to mean-reverting, trend is exhausting
            if self._hurst and len(closes) >= 50 and pnl_pct > 2.0:
                import numpy as _np
                h_result = self._hurst.compute(_np.array(closes), window=min(200, len(closes)))
                if h_result and h_result['regime'] == 'mean_reverting' and h_result['confidence'] > 0.6:
                    # Trend is exhausted — protect 65% of profit
                    if direction == 'LONG':
                        hurst_sl = entry + (price - entry) * 0.65
                        if hurst_sl > new_sl and hurst_sl < price:
                            new_sl = hurst_sl
                            ml_tighten_reason = f"Hurst H={h_result['hurst']:.2f} mean-reverting"
                    else:
                        hurst_sl = entry - (entry - price) * 0.65
                        if hurst_sl < new_sl and hurst_sl > price:
                            new_sl = hurst_sl
                            ml_tighten_reason = f"Hurst H={h_result['hurst']:.2f} mean-reverting"
        except Exception as e:
            logger.debug(f"ML trailing SL error for {asset}: {e}")

        if ml_tighten_reason:
            print(f"  [{self._ex_tag}:{asset}] ML SL TIGHTEN: {ml_tighten_reason} → SL=${new_sl:,.2f}")

        # SL can only move FORWARD (tighter), never backward
        # Minimum move: $0.50 or 0.01% of entry (whichever is larger) to avoid noise
        new_sl = round(new_sl, 2)
        min_move = max(0.50, entry * 0.0001)
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

        # ── Persist SL state to disk after every update (crash recovery) ──
        if sl_moved and self._sl_persist:
            try:
                self._sl_persist.save_position(asset, pos)
            except Exception:
                pass

        # SL managed by polling (10s check) — no exchange stop orders
        # This avoids orphan positions from exchange SL fills

        # ── MT5: Sync SL ratchet to MetaTrader 5 ──
        if sl_moved and self._mt5 and self._mt5.connected:
            try:
                self._mt5.mirror_sl_update(asset, new_sl) if self._mt5.mode == 'mirror' else self._mt5.update_sl(asset, new_sl)
            except Exception:
                pass  # Don't block main loop for MT5 SL sync

        # ── 5. EMA NEW LINE EXIT — Exit when EMA forms NEW OPPOSITE line ──
        # This is the core exit rule: we entered on a "new line" forming,
        # we exit when a NEW OPPOSITE line forms (EMA direction flips after 3+ bars)
        #
        # From reference images:
        #   CALL exit: EMA was rising (our trade), now falling for 2+ bars + price below EMA
        #   PUT exit:  EMA was falling (our trade), now rising for 2+ bars + price above EMA
        #
        # When in profit (breakeven+): exit immediately on new opposite line
        # When in loss: still exit but give 1 extra bar to confirm (avoid noise)
        if not is_stuck and len(ema_vals) >= 5:
            confirmed_ema = ema_vals[-2]
            prev_confirmed_ema = ema_vals[-3]
            prev2_confirmed_ema = ema_vals[-4]

            # Count consecutive bars of EMA reversal against our position
            reversal_bars = 0
            if direction == 'LONG':
                # Check for NEW DOWN line forming (EMA falling consecutively)
                for ri in range(2, min(8, len(ema_vals))):
                    if ema_vals[-ri] < ema_vals[-ri - 1]:
                        reversal_bars += 1
                    else:
                        break
            else:
                # Check for NEW UP line forming (EMA rising consecutively)
                for ri in range(2, min(8, len(ema_vals))):
                    if ema_vals[-ri] > ema_vals[-ri - 1]:
                        reversal_bars += 1
                    else:
                        break

            # Exit conditions: new opposite line forming
            # ONLY when in MEANINGFUL profit — protects from premature exits on tiny gains
            # With Robinhood 1.69% round-trip spread, exiting at 0.04% "profit" is a real loss
            # min_exit_pnl_pct: minimum profit % required before EMA reversal exit triggers
            min_reversal = 2
            min_exit_pnl_pct = self.config.get('risk', {}).get('min_exit_pnl_pct', 1.5)

            if pnl_pct > min_exit_pnl_pct:
                if direction == 'LONG' and reversal_bars >= min_reversal and confirmed_close < confirmed_ema:
                    trade_tf_label = pos.get('trade_timeframe', '5m')
                    print(f"  [{self._ex_tag}:{asset}] NEW DOWN LINE [{trade_tf_label}]: EMA falling {reversal_bars} bars, price below EMA | P&L: {pnl_pct:+.2f}%")
                    self._close_position(asset, price, f"EMA new down line ({reversal_bars} bars)")
                    self.last_trade_time.pop(asset, None)
                    self.last_close_time.pop(asset, None)
                    return
                elif direction == 'SHORT' and reversal_bars >= min_reversal and confirmed_close > confirmed_ema:
                    trade_tf_label = pos.get('trade_timeframe', '5m')
                    print(f"  [{self._ex_tag}:{asset}] NEW UP LINE [{trade_tf_label}]: EMA rising {reversal_bars} bars, price above EMA | P&L: {pnl_pct:+.2f}%")
                    self._close_position(asset, price, f"EMA new up line ({reversal_bars} bars)")
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
            # Find current ratchet phase from the matching level
            if pnl_pct >= 15: phase = "LOCK-70%"
            elif pnl_pct >= 10: phase = "LOCK-65%"
            elif pnl_pct >= 7: phase = "LOCK-60%"
            elif pnl_pct >= 5: phase = "LOCK-55%"
            elif pnl_pct >= 4: phase = "LOCK-50%"
            elif pnl_pct >= 3: phase = "LOCK-40%"
            elif pnl_pct >= 2.5: phase = "LOCK-30%"
            elif pnl_pct >= 2: phase = "LOCK-25%"
            elif pnl_pct >= 1.5: phase = "LOCK-15%"
            elif pnl_pct >= 1.2: phase = "LOCK-10%"
            elif pnl_pct >= 0.8: phase = "BREAKEVEN"
            else: phase = "INITIAL"
            status = f"SAFE=${sl_pnl_pct * entry / 100:,.2f}({sl_pnl_pct:+.1f}%) AT-RISK={risk_pct:.1f}% [{phase}]"
        elif sl_pnl_pct >= 0:
            status = "BREAKEVEN (investment safe)"

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

        # Prevent re-entrant close (multiple exit conditions in same cycle)
        if not hasattr(self, '_closing_in_progress'):
            self._closing_in_progress = set()
        if asset in self._closing_in_progress:
            return
        self._closing_in_progress.add(asset)

        pos = self.positions[asset]
        direction = pos['direction']
        entry = pos['entry_price']
        qty = pos['qty']
        symbol = self._get_symbol(asset)

        close_side = 'sell' if direction == 'LONG' else 'buy'

        # ── Paper Mode (Robinhood): simulate exit fill at real bid/ask ──
        if self._paper_mode:
            try:
                ob = self.price_source.fetch_order_book(symbol, limit=5)
                if close_side == 'sell' and ob.get('bids'):
                    price = float(ob['bids'][0][0])  # LONG exit at bid
                elif close_side == 'buy' and ob.get('asks'):
                    price = float(ob['asks'][0][0])  # SHORT exit at ask
                else:
                    price = self.price_source.fetch_latest_price(symbol) or price
            except Exception:
                price = self.price_source.fetch_latest_price(symbol) or price
            print(f"  [{self._ex_tag}:{asset}] PAPER CLOSE: {close_side.upper()} {qty:.6f} @ ${price:,.2f} | reason={reason}")
        else:
            # ── Live Mode: Place real close orders ──
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

            remaining_qty = actual_qty
            for close_attempt in range(3):
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
                    result = self._api_call(
                        self.price_source.place_order,
                        symbol=symbol,
                        side=close_side,
                        amount=remaining_qty,
                        order_type='limit',
                        price=close_price,
                        reduce_only=True,
                    )
                    print(f"  [{self._ex_tag}:{asset}] CLOSE LIMIT @ ${close_price:,.2f}")
                    time.sleep(2)
                    try:
                        ex_positions = self._exchange_client.get_positions()
                        still_has = any(asset in p.get('symbol','') and float(p.get('qty',0)) > 0 for p in ex_positions)
                        if not still_has:
                            price = close_price
                            break
                        open_orders = self._exchange_client.exchange.fetch_open_orders(symbol)
                        for o in open_orders:
                            try:
                                self._exchange_client.exchange.cancel_order(o['id'], symbol)
                            except Exception:
                                pass
                    except Exception:
                        pass

                result = self._api_call(
                    self.price_source.place_order,
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
                        self._closing_in_progress.discard(asset)
                        return
                    else:
                        print(f"  [{self._ex_tag}:{asset}] CLOSE WARNING: {err}")

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
                            break
                        if close_attempt < 2:
                            print(f"  [{self._ex_tag}:{asset}] Partial fill — {remaining_qty} remaining, retrying...")
                        else:
                            print(f"  [{self._ex_tag}:{asset}] CLOSE INCOMPLETE — {remaining_qty} still open after 3 attempts")
                            self.failed_close_assets[asset] = time.time()
                            self._closing_in_progress.discard(asset)
                            return
                    else:
                        break
                except Exception:
                    break

            # Fetch actual fill price from exchange
            actual_exit = price
            try:
                if self._exchange_client:
                    time.sleep(0.5)
                    ex_positions = self._exchange_client.get_positions()
                    still_open = any(
                        asset in p.get('symbol', '') and float(p.get('qty', 0)) > 0
                        for p in ex_positions
                    )
                    if not still_open:
                        try:
                            recent = self._exchange_client.exchange.fetch_my_trades(symbol, limit=3)
                            if recent:
                                last_fill = recent[-1]
                                actual_exit = float(last_fill.get('price', price))
                                if abs(actual_exit - price) / price > 0.002:
                                    print(f"  [{self._ex_tag}:{asset}] EXIT SLIPPAGE: expected ${price:,.2f} filled ${actual_exit:,.2f} ({(actual_exit-price)/price*100:+.2f}%)")
                        except Exception:
                            pass
            except Exception:
                pass
            price = actual_exit

        # ── Calculate P&L (correct for contract-based exchanges) ──
        # Delta: qty = number of contracts. 1 BTC contract = 0.001 BTC, 1 ETH contract = 0.01 ETH
        # Bybit: qty = coin amount (e.g., 0.001 BTC)
        # P&L formula: (exit - entry) * qty * contract_size
        if self._exchange_name == 'delta':
            contract_sizes = {'BTC': 0.001, 'ETH': 0.01}
            cs = contract_sizes.get(asset, 0.001)
        else:
            cs = 1.0  # Bybit qty is already in coin units

        if direction == 'LONG':
            pnl_pct = ((price - entry) / entry) * 100.0
            pnl_usd = (price - entry) * qty * cs
        else:
            pnl_pct = ((entry - price) / entry) * 100.0
            pnl_usd = (entry - price) * qty * cs

        # ── Deduct round-trip spread from P&L (Robinhood ~1.69%) ──
        _spread_cost_pct = self._round_trip_spread if hasattr(self, '_round_trip_spread') else 0
        if _spread_cost_pct > 0.5:
            _spread_cost_usd = entry * qty * cs * (_spread_cost_pct / 100.0)
            pnl_pct -= _spread_cost_pct
            pnl_usd -= _spread_cost_usd

        # ── Model Accuracy Telemetry: compare predictions vs actual profitability ──
        try:
            _ml_preds = pos.get('ml_predictions') or {}
            if _ml_preds:
                from src.api.state import DashboardState as _DashboardState
                _mlp_actual = 1 if pnl_pct > 0 else -1
                _mlp_state = _DashboardState()
                for _mlp_name, _mlp_pred in _ml_preds.items():
                    if _mlp_pred != 0:
                        _mlp_state.record_model_prediction(_mlp_name, int(_mlp_pred), _mlp_actual)
        except Exception as _mlp_err:
            try:
                logger.debug(f"record_model_prediction failed: {_mlp_err}")
            except Exception:
                pass

        sl_chain = '->'.join(pos.get('sl_levels', ['L1']))
        actual_l_count = len(pos.get('sl_levels', ['L1']))
        actual_l_level = f"L{actual_l_count}"
        predicted_l = pos.get('predicted_l_level', '?')
        duration_min = (time.time() - pos.get('entry_time', time.time())) / 60.0

        # L-Level prediction accuracy tracking
        pred_hit = "?"
        if predicted_l != '?':
            try:
                pred_num = int(''.join(c for c in predicted_l if c.isdigit()) or '0')
                if pred_num > 0:
                    pred_hit = "HIT" if actual_l_count >= pred_num else "MISS"
            except Exception:
                pass

        trade_tf_label = pos.get('trade_timeframe', '5m')
        print(f"  [{self._ex_tag}:{asset}] CLOSED [{trade_tf_label}]: P&L {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | {reason} | predicted={predicted_l} actual={actual_l_level} [{pred_hit}]")

        # ── Alert: Trade Close ──
        alert_level = 'WARNING' if pnl_usd < -5 else 'INFO'
        self._send_alert(alert_level, f'{self._ex_tag} CLOSE {asset}',
            f'{direction} {asset} closed: P&L {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | {reason} | {pred_hit}',
            {'asset': asset, 'direction': direction, 'pnl_pct': pnl_pct, 'pnl_usd': pnl_usd,
             'reason': reason, 'predicted': predicted_l, 'actual': actual_l_level})

        # ── Dynamic Risk: Feed closed trade P&L ──
        if self._dynamic_risk:
            try:
                self._dynamic_risk.update_pnl(pnl_usd)
            except Exception:
                pass

        # ── LEARNING HOOK 1: Label LLM decision with real trade outcome ──
        # This feeds the fine-tuning pipeline so the LLM learns from every trade
        if self._training_collector:
            try:
                entry_time = pos.get('entry_time', time.time())
                self._training_collector.label_outcome(
                    asset=asset,
                    entry_time=entry_time,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                    sl_level=actual_l_level or 'L1',
                    duration_min=duration_min,
                )
                print(f"  [LEARN] LLM outcome labeled: {asset} {pnl_pct:+.2f}% ({reason[:30]})")
            except Exception as e:
                logger.debug(f"[LEARN] label_outcome failed: {e}")

        # ── LEARNING HOOK 2: Feed live outcome to Genetic Strategy Engine ──
        # This updates strategy fitness scores with real P&L, not just backtests
        if self._genetic_engine:
            try:
                active_dna = pos.get('active_dna_name')
                if active_dna:
                    self._genetic_engine.record_live_outcome(
                        dna_name=active_dna,
                        pnl_pct=pnl_pct,
                        win=(pnl_pct > 0),
                        duration_min=duration_min,
                        regime=pos.get('regime', 'unknown'),
                    )
                    print(f"  [LEARN] Genetic outcome: {active_dna} {pnl_pct:+.2f}%")
            except Exception as e:
                logger.debug(f"[LEARN] genetic record failed: {e}")

        # ── LEARNING HOOK 3 (v8.0): Feed outcome to ALL memory systems ──
        _regime = pos.get('regime', 'unknown')
        _won = pnl_pct > 0
        _label = 'WIN' if _won else 'LOSS'
        _confidence = pos.get('confidence', 0.5)
        # Accuracy Engine: streak tracking + model outcomes
        if self._accuracy_engine:
            try:
                self._accuracy_engine.record_trade_outcome(pnl_pct, pnl_usd)
                print(f"  [v8.0] AccuracyEngine updated: {self._accuracy_engine._streak_type} x{self._accuracy_engine._streak_length}")
            except Exception as e:
                logger.debug(f"[v8.0] accuracy record failed: {e}")
        # Sharpe Optimizer: rolling returns
        if self._sharpe_optimizer:
            try:
                self._sharpe_optimizer.record_trade(pnl_pct, regime=_regime)
                print(f"  [v8.0] Sharpe: mode={self._sharpe_optimizer.mode} rolling={self._sharpe_optimizer.get_rolling_sharpe()}")
            except Exception as e:
                logger.debug(f"[v8.0] sharpe record failed: {e}")
        # LLM Memory: record decision outcome
        if self._llm_memory:
            try:
                self._llm_memory.record_decision(
                    prompt_hash=f"{asset}_{pos.get('entry_time', 0):.0f}",
                    parsed_output={'proceed': True, 'confidence': _confidence},
                    trade_outcome_pnl=pnl_pct, trade_outcome_label=_label,
                    bear_veto_fired=False, actual_move_pct=pnl_pct, predicted_move_pct=0,
                )
            except Exception as e:
                logger.debug(f"[v8.0] llm_memory record failed: {e}")
        # Quant Memories: per-model outcomes
        if hasattr(self, '_quant_memories') and self._quant_memories:
            try:
                for _model_name, _qm in self._quant_memories.items():
                    _qm.record_prediction(asset, 'LONG', _confidence, [],
                                          _regime, 0.5, 0.15, 'live', pnl_pct, _label)
            except Exception as e:
                logger.debug(f"[v8.0] quant_memory record failed: {e}")

        # v8.0: Dynamic position limits — learn from this trade
        if self._dynamic_limits:
            try:
                self._dynamic_limits.update_equity(self.equity)
                self._dynamic_limits.record_trade(
                    asset=asset, size_pct=pos.get('size_pct', 1.0),
                    leverage=pos.get('leverage', 1.0), pnl_pct=pnl_pct,
                    regime=_regime,
                )
            except Exception as e:
                logger.debug(f"[v8.0] dynamic_limits record failed: {e}")

        # Phase 2: publish the outcome to the trade.outcome stream so the
        # Phase 4.5a meta-coordinator can enrich it into an Experience. Fields
        # match what credit_assigner.py will expect — don't rename without
        # updating that consumer too.
        entry_time = float(pos.get('entry_time', 0.0) or 0.0)
        exit_time = time.time()
        exit_reason = {
            'sl': 'SL', 'stop': 'SL', 'tp': 'TP', 'target': 'TP',
            'timeout': 'TIMEOUT', 'hold': 'TIMEOUT', 'manual': 'MANUAL',
        }.get(str(reason).lower().split()[0] if reason else '', 'MANUAL')
        _outcome_row = {
            "asset": asset,
            "symbol": asset,
            "direction": pos.get('direction'),
            "entry_price": float(pos.get('entry_price', 0.0) or 0.0),
            "exit_price": float(price or 0.0),
            "pnl_pct": float(pnl_pct),
            "pnl_usd": float(pnl_usd),
            "duration_s": max(0.0, exit_time - entry_time) if entry_time else 0.0,
            "exit_reason": exit_reason,
            "regime": _regime,
            "won": bool(_won),
            "confidence": float(_confidence),
            "decision_id": pos.get('decision_id'),
            "trace_id": pos.get('trace_id'),
            "entry_ts": entry_time,
            "exit_ts": exit_time,
        }
        try:
            from src.orchestration import STREAM_TRADE_OUTCOME, stream_publish
            stream_publish(STREAM_TRADE_OUTCOME, _outcome_row)
        except Exception as e:
            logger.debug(f"[PHASE2] trade.outcome publish failed: {e}")

        # Phase 3: warm-tier durable outcome write.
        try:
            from src.orchestration.warm_store import get_store
            get_store().write_outcome(_outcome_row)
        except Exception as e:
            logger.debug(f"[WARM] outcome write failed: {e}")

        # ── MT5: Close mirrored/executed position (skip in paper mode) ──
        if self._mt5 and self._mt5.connected and not self._paper_mode:
            try:
                if self._mt5.mode == 'execute':
                    self._mt5.close_position(asset, price, reason)
                else:
                    self._mt5.mirror_close(asset, price, reason)
            except Exception as e:
                print(f"  [MT5:{asset}] Bridge close failed: {e}")

        # Track realized PnL for drawdown limits
        self.session_realized_pnl += pnl_usd
        self.daily_realized_pnl += pnl_usd

        # ── SNIPER: Feed profits into compound pool ──
        if self.sniper_enabled:
            if pnl_usd > 0:
                self.sniper_profit_pool += pnl_usd
                self.sniper_stats['wins'] += 1
                print(f"  [{self._ex_tag}:{asset}] SNIPER WIN: +${pnl_usd:,.2f} → profit pool: ${self.sniper_profit_pool:,.2f}")
            else:
                # Losses reduce profit pool first (protect principal)
                if self.sniper_protect_principal and self.sniper_profit_pool > 0:
                    absorbed = min(self.sniper_profit_pool, abs(pnl_usd))
                    self.sniper_profit_pool -= absorbed
                    print(f"  [{self._ex_tag}:{asset}] SNIPER LOSS: ${pnl_usd:,.2f} | pool absorbed ${absorbed:,.2f} → pool: ${self.sniper_profit_pool:,.2f}")
                self.sniper_stats['losses'] += 1

        # ── Trade Log: record + print formatted trade row ──
        notional = entry * qty * (1.0 if self._exchange_name != 'delta' else {'BTC': 0.001, 'ETH': 0.01}.get(asset, 0.001))
        spread_fee = notional * 0.0085 if self._paper_mode else 0  # ~0.85% Robinhood spread per side
        trade_record = {
            'market': f"{asset}/USD",
            'entry_price': entry,
            'exit_price': price,
            'qty': qty,
            'direction': direction,
            'realized_pnl': pnl_usd,
            'spread_cost': spread_fee * 2,  # entry + exit
            'duration_min': duration_min,
            'reason': reason,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        self._trade_log.append(trade_record)

        # Print trade row
        n = len(self._trade_log)
        pnl_symbol = "+" if pnl_usd >= 0 else ""
        total_pnl = sum(t['realized_pnl'] for t in self._trade_log)
        wins = sum(1 for t in self._trade_log if t['realized_pnl'] > 0)
        losses = sum(1 for t in self._trade_log if t['realized_pnl'] <= 0)
        wr = wins / n * 100 if n > 0 else 0
        print(f"  +-----------------------------------------------------------------------")
        print(f"  | TRADE #{n}: {direction} {asset}/USD")
        print(f"  | Entry: ${entry:,.2f}  ->  Exit: ${price:,.2f}  |  Qty: {qty:.6f}")
        print(f"  | Realized P&L: {pnl_symbol}${pnl_usd:,.2f} ({pnl_pct:+.2f}%)  |  Reason: {reason}")
        if self._paper_mode:
            print(f"  | Spread Cost: ~${spread_fee * 2:,.2f}  |  Duration: {duration_min:.1f}min")
        else:
            print(f"  | Duration: {duration_min:.1f}min")
        print(f"  | -- Session: Total P&L: ${total_pnl:+,.2f}  |  Trades: {n} (W:{wins} L:{losses})  |  WR: {wr:.0f}%")
        print(f"  +-----------------------------------------------------------------------")

        # Update agent orchestrator weights (learn from outcome)
        if self._orchestrator and pos.get('agent_votes'):
            try:
                self._orchestrator.record_outcome(
                    asset=asset,
                    direction=1 if direction == 'LONG' else -1,
                    pnl=pnl_usd,
                )
            except Exception:
                pass  # Don't block close on feedback error

        # Track consecutive loss streaks per asset
        if pnl_usd < 0:
            self.asset_loss_streak[asset] = self.asset_loss_streak.get(asset, 0) + 1
            streak = self.asset_loss_streak[asset]
            if streak >= 3:
                # Reduced cooldown: 3 losses=5min, 4=10min, 5+=15min max
                # Was exponential up to 60min — too aggressive, missed many good patterns
                cooldown_min = min(15, 5 * (streak - 2))
                self.asset_cooldown_until[asset] = time.time() + cooldown_min * 60
                print(f"  [{self._ex_tag}:{asset}] LOSS STREAK {streak} — cooling down {cooldown_min}min (until next good setup)")
        else:
            self.asset_loss_streak[asset] = 0  # Reset on any win/breakeven

        # Update edge stats
        if asset in self.edge_stats:
            self.edge_stats[asset]['total'] += 1
            if pnl_pct > 0:
                self.edge_stats[asset]['wins'] += 1
            else:
                self.edge_stats[asset]['losses'] += 1
            s = self.edge_stats[asset]
            s['win_rate'] = s['wins'] / s['total'] if s['total'] > 0 else 0.5

        # Safe-entries: record outcome for consecutive-loss throttle + rolling Sharpe.
        # Always records even when the gate is disabled, so turning it on later has
        # historical trade distribution to reason about.
        try:
            if getattr(self, '_safe_state', None) is not None:
                self._safe_state.record_outcome(asset, pnl_pct=float(pnl_pct), won=(pnl_pct > 0))
                from src.trading.safe_entries import default_state_path as _ssp
                self._safe_state.save(_ssp())
        except Exception:
            pass

        # Shadow-mode: append outcome to logs/meta_shadow.jsonl so shadow_retrain
        # can join it against the entry-time prediction by trade_id. Guarded by
        # is_enabled() — no-op unless ACT_META_SHADOW_MODE=1.
        try:
            from src.ml import shadow_log as _slog
            _trade_id = str(pos.get('order_id') or '')
            if _trade_id:
                _entry_t = float(pos.get('entry_time', 0) or 0)
                _now_t = time.time()
                _bars_held = int(max(0, (_now_t - _entry_t) / 60.0)) if _entry_t else 0
                _slog.log_outcome(
                    trade_id=_trade_id,
                    pnl_pct=float(pnl_pct),
                    pnl_usd=float(pnl_usd),
                    exit_price=float(price),
                    bars_held=_bars_held,
                    exit_reason=str(reason)[:100] if reason else "",
                )
        except Exception:
            pass

        # Update per-timeframe performance (LLM learns which TFs profit)
        trade_tf = pos.get('trade_timeframe', '5m')
        if trade_tf in self.tf_performance:
            self.tf_performance[trade_tf]['total_pnl'] += pnl_usd
            if pnl_usd > 0:
                self.tf_performance[trade_tf]['wins'] += 1
            else:
                self.tf_performance[trade_tf]['losses'] += 1

        # Log to journal (with prediction tracking)
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
                exchange=self._ex_tag.lower(),
                extra={
                    'predicted_l_level': predicted_l,
                    'actual_l_level': actual_l_level,
                    'prediction_hit': pred_hit,
                    'risk_score': pos.get('risk_score', 0),
                    'bear_risk': pos.get('bear_risk', 0),
                    'hurst': pos.get('hurst', 0.5),
                    'hurst_regime': pos.get('hurst_regime', 'unknown'),
                    'trade_timeframe': pos.get('trade_timeframe', '5m'),
                },
            )
        except Exception as e:
            logger.warning(f"Journal log failed: {e}")

        # ── TradeTrace: structured record for memory/audit ──
        if TRADE_TRACE_AVAILABLE:
            try:
                trace = TradeTrace(
                    timestamp=datetime.utcnow(),
                    asset=asset,
                    market_regime=self._last_ml_context.get('vol_regime_adv',
                                  self._last_ml_context.get('vol_regime', 'UNKNOWN')),
                    funding_rate=0.0,
                    sentiment={'bullish': 0.5, 'bearish': 0.5, 'neutral': 0.0},
                    agent_bias=0.0,
                    proposed_signal=1 if direction == 'LONG' else -1,
                    signal_confidence=pos.get('confidence', 0.5),
                    price={'open': entry, 'high': pos.get('peak_price', entry),
                           'low': min(entry, price), 'close': price},
                    volume=0.0,
                    entry_price=entry,
                    exit_price=price,
                    holding_bars=max(1, int(duration_min / 5)),
                    pnl=pnl_usd,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                    reasoning_trace=pos.get('reasoning', '')[:200],
                )
                # Store for memory vault if available
                if self._memory:
                    try:
                        self._memory.store(trace.to_embedding_text(), trace.to_metadata())
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"TradeTrace error: {e}")

        # ── ADVANCED LEARNING: Online training from trade outcome ──
        # Updates: strategy generator EWMA stats, alpha decay tracker, meta-learner
        if self._advanced_learning:
            try:
                _pred_l = pos.get('predicted_l_level', 0)
                if isinstance(_pred_l, str):
                    try:
                        _pred_l = int(_pred_l.replace('L', '').replace('+', ''))
                    except (ValueError, AttributeError):
                        _pred_l = 0
                _actual_l = 0
                if pnl_pct > 0:
                    # Approximate actual L-level from PnL %
                    _actual_l = max(1, int(pnl_pct / 0.5))  # ~0.5% per L-level
                self._advanced_learning.on_trade_close(
                    asset=asset, pnl_usd=pnl_usd, pnl_pct=pnl_pct,
                    predicted_l_level=_pred_l, actual_l_level=_actual_l,
                )
                _stats = self._advanced_learning.strategy_generator._running_stats.get(asset, {})
                _edge = self._advanced_learning.edge_retention.get(asset, 1.0)
                print(f"  [{self._ex_tag}:{asset}] META-LEARN: WR={_stats.get('win_rate', 0):.0%} edge={_edge:.2f} streak={'W' if _stats.get('win_streak', 0) > 0 else 'L'}{max(_stats.get('win_streak', 0), _stats.get('loss_streak', 0))}")
            except Exception as e:
                logger.debug(f"Advanced learning trade update error: {e}")

        # ── Log to LightGBM classifier for incremental learning ──
        # Fetch fresh 5m candles for feature extraction (not passed to _close_position)
        try:
            _symbol = self._get_symbol(asset)
            _raw = self.price_source.fetch_ohlcv(_symbol, timeframe='5m', limit=60)
            _ohlcv = PriceFetcher.extract_ohlcv(_raw)
            closes = _ohlcv['closes']; highs = _ohlcv['highs']
            lows = _ohlcv['lows']; volumes = _ohlcv['volumes']
        except Exception:
            closes = []; highs = []; lows = []; volumes = []

        if self._lgbm and len(closes) >= 55:
            try:
                lgbm_features = self._lgbm.extract_features(
                    closes=closes, highs=highs, lows=lows, volumes=volumes,
                )
                if lgbm_features and lgbm_features[-1]:
                    dir_int = 1 if direction == 'LONG' else -1
                    bars_held = max(1, int(duration_min / 5))  # 5min candles
                    self._lgbm.log_trade(
                        features=lgbm_features[-1],
                        direction=dir_int,
                        net_pnl=pnl_usd,
                        entry_price=entry,
                        exit_price=price,
                        bars_held=bars_held,
                    )
                # Auto-retrain after N trades
                self._lgbm_trades_since_retrain += 1
                if self._lgbm_trades_since_retrain >= self._lgbm_retrain_interval:
                    try:
                        print(f"  [{self._ex_tag}] LGBM AUTO-RETRAIN: {self._lgbm_trades_since_retrain} trades accumulated")
                        self._lgbm.retrain_from_log(max_examples=500)
                        self._lgbm_trades_since_retrain = 0
                        # Save updated model
                        import os as _os
                        model_dir = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'models')
                        _os.makedirs(model_dir, exist_ok=True)
                        if self._lgbm._lgb_model:
                            self._lgbm._lgb_model.save_model(_os.path.join(model_dir, 'lgbm_latest.txt'))
                            print(f"  [{self._ex_tag}] LGBM: Retrained model saved")
                    except Exception as re:
                        logger.warning(f"LightGBM retrain error: {re}")
            except Exception as e:
                logger.debug(f"LightGBM log_trade error: {e}")

        # ── Record close in Trade Protections (SL guard, pair lock, drawdown) ──
        if self._protections:
            try:
                sl_hit = 'stop' in reason.lower() or 'sl' in reason.lower() or 'hard' in reason.lower()
                exit_tag = self._protections.tagger.tag_exit(
                    exit_reason=reason, sl_level=len(pos.get('sl_levels', ['L1'])),
                    pnl_pct=pnl_pct, duration_min=duration_min,
                    roi_exit='roi' in reason.lower(),
                )
                self._protections.record_close(
                    asset=asset, pnl_pct=pnl_pct, pnl_usd=pnl_usd,
                    exit_reason=reason, sl_hit=sl_hit, equity=self.equity,
                )
                print(f"  [{self._ex_tag}:{asset}] TAG: entry={pos.get('entry_tag','?')} exit={exit_tag}")
            except Exception as e:
                logger.warning(f"[{asset}] Protection record_close error: {e}")

        # ── Update Profit Protector with trade result (learns win rate + adapts sizing) ──
        if self._profit_protector:
            try:
                self._profit_protector.log_trade_result(
                    entry_price=entry, exit_price=price,
                    confidence=pos.get('confidence', 0.5), position_size=qty,
                )
                self._profit_protector.update_balance(self.equity)
            except Exception as e:
                logger.debug(f"Profit Protector log error: {e}")

        # ── RL FEEDBACK LOOP: Learn from trade outcome (spread-aware) ──
        # This is the CRITICAL missing piece: RL agent now learns from every closed trade
        # With spread cost included, it learns to avoid trades where move < spread
        _rl = self._rl_per_asset.get(asset) if hasattr(self, '_rl_per_asset') else None
        if _rl and pos.get('rl_state') is not None:
            try:
                # Compute spread cost for this trade
                _spread_cost_pct = 0.0
                if self._paper_mode:
                    # Robinhood: ~0.845% per side = ~1.69% round-trip
                    _spread_cost_pct = 1.69
                else:
                    # Futures: typical spread ~0.05-0.1% round-trip
                    _spread_cost_pct = 0.1

                # Map exit reason to exit_type
                _exit_type = 'unknown'
                reason_lower = reason.lower()
                if 'ema' in reason_lower or 'line' in reason_lower:
                    _exit_type = 'ema_exit'
                elif 'ratchet' in reason_lower or 'trailing' in reason_lower:
                    _exit_type = 'ratchet'
                elif 'hard stop' in reason_lower:
                    _exit_type = 'hard_stop'
                elif 'sl' in reason_lower or 'stop' in reason_lower:
                    _exit_type = 'sl'
                elif 'time' in reason_lower:
                    _exit_type = 'time'

                _trade_result = {
                    'pnl_pct': pnl_pct,
                    'exit_type': _exit_type,
                    'hold_bars': int(duration_min),
                    'was_skipped': False,
                    'spread_cost_pct': _spread_cost_pct,
                    'is_spot': self._paper_mode,  # Robinhood = spot
                }
                _rl.record_trade_result(
                    state=pos['rl_state'],
                    action_idx=pos.get('rl_action_idx', 0),
                    trade_result=_trade_result,
                )
                print(f"  [{self._ex_tag}:{asset}] RL LEARNED: {_exit_type} net_pnl={pnl_pct - _spread_cost_pct:+.2f}% (raw={pnl_pct:+.2f}% - spread={_spread_cost_pct:.2f}%)")
            except Exception as _rl_err:
                logger.debug(f"RL feedback error: {_rl_err}")

        # ── Paper Trade: Record exit with real Robinhood price ──
        if self._paper:
            try:
                self._paper.record_exit(asset, reason=reason)
                self._paper.save_state()  # Immediate save for dashboard
            except Exception as _pe:
                logger.warning(f"[PAPER] Exit record failed: {_pe}")

        # ── Fine-tuning: label this trade outcome for LLM training ──
        if hasattr(self, '_training_collector') and self._training_collector:
            try:
                self._training_collector.label_outcome(
                    asset=asset,
                    entry_time=pos.get('entry_time', 0),
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                    sl_level=pos.get('sl_levels', ['L1'])[-1] if pos.get('sl_levels') else 'L1',
                    duration_min=(time.time() - pos.get('entry_time', time.time())) / 60,
                )
            except Exception:
                pass

        # Remove from positions
        del self.positions[asset]
        self._closing_in_progress.discard(asset)

        # Clear SL persistence (position is closed — no crash recovery needed)
        if self._sl_persist:
            try:
                self._sl_persist.clear_position(asset)
            except Exception:
                pass
        # ── Adaptive feedback — teach the system from this outcome ──
        if self._adaptive:
            try:
                self._adaptive.record_outcome(TradeOutcome(
                    asset=asset,
                    direction=direction,
                    entry_price=entry,
                    exit_price=price,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl_usd,
                    duration_min=(time.time() - pos.get('entry_time', time.time())) / 60,
                    entry_score=pos.get('entry_tag', '').count('-') + 5,  # rough score from tag
                    strategy_used=pos.get('trade_timeframe', 'ema_trend'),
                    strategy_signals=pos.get('_multi_strategy_details', {}),
                    agent_votes=pos.get('agent_votes', {}),
                    llm_confidence=pos.get('confidence', 0),
                    risk_score=pos.get('risk_score', 0),
                    trade_quality=pos.get('bear_risk', 0),
                    regime=pos.get('hurst_regime', 'UNKNOWN'),
                    hurst=pos.get('hurst', 0.5),
                    sl_level=len(pos.get('sl_levels', ['L1'])),
                    exit_reason=reason,
                    timeframe=pos.get('trade_timeframe', '4h'),
                    spread_cost_pct=self._round_trip_spread if hasattr(self, '_round_trip_spread') else 0,
                    multi_strategy_details=pos.get('_multi_strategy_details', {}),
                ))
            except Exception as _af_err:
                logger.debug(f"[ADAPTIVE] Feedback failed: {_af_err}")

        # ── Self-Evolving Overlay — learn from this trade ──
        if self._evolution_overlay:
            try:
                _evo_trade = {
                    'won': pnl_pct > 0,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd,
                    'asset': asset,
                    'strategy': pos.get('trade_timeframe', 'ema_trend'),
                    'llm_confidence': pos.get('confidence', 0),
                    'entry_score': pos.get('entry_tag', '').count('-') + 5,
                    'exit_reason': reason,
                    'direction': 1 if direction == 'LONG' else -1,
                }
                _evo_votes = pos.get('agent_votes', {})
                _evo_regime = pos.get('hurst_regime', 'UNKNOWN')
                self._evolution_overlay.update_all(
                    trade_outcome=_evo_trade,
                    agent_votes=_evo_votes,
                    regime=_evo_regime,
                )
            except Exception as _evo_err:
                logger.debug(f"[EVOLVE] Feedback failed: {_evo_err}")

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
                "num_predict": 1024,
            },
        }

        try:
            resp = requests.post(url, json=payload, timeout=60)
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
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )

            text = message.content[0].text.strip()
            return self._extract_json(text)
        except Exception as e:
            logger.warning(f"Claude query failed: {e} -- falling back to local Ollama")
            print(f"  [AI] Claude failed ({type(e).__name__}): {str(e)[:80]} -- using Ollama fallback")
            return self._query_llm(prompt)

    # ------------------------------------------------------------------
    # Unified LLM router — picks Claude or Ollama based on exchange
    # ------------------------------------------------------------------
    def _query_llm_auto(self, prompt: str) -> str:
        """Route LLM query through LLMRouter (multi-provider with fallback).
        Falls back to legacy direct calls if router unavailable."""
        # ── NEW: Use LLMRouter for robust multi-provider routing ──
        if self._llm_router:
            try:
                # Build fallback chain: primary provider first, then others
                if self._use_claude and 'claude' in self._llm_router.list_providers():
                    chain = ['claude', 'ollama']
                else:
                    chain = ['ollama', 'claude'] if 'claude' in self._llm_router.list_providers() else ['ollama']
                result = self._llm_router.query(prompt, fallback_chain=chain, cache=False)
                if result:
                    return json.dumps(result)
            except Exception as e:
                logger.warning(f"LLMRouter failed ({e}) — falling back to legacy calls")

        # ── LEGACY FALLBACK: direct Ollama/Claude calls ──
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
                            math_filter_warnings: list = None,
                            entry_score: int = 0,
                            score_reasons: list = None,
                            min_trend_bars: int = 0,
                            slope_pct: float = 0,
                            ema_separation: float = 0,
                            htf_1h_direction: str = "FLAT",
                            htf_4h_direction: str = "FLAT",
                            htf_alignment: int = 2,
                            mtf_signal_block: str = "",
                            active_tf_signals: dict = None) -> dict:
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
        agent_data = "\n".join(agent_lines[:10])
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

        # Order book imbalance for LLM prompt (from caller's ob_levels via closure)
        ob_imbalance = 0

        # Last N candles
        n_candles = min(10, len(candle_lines))
        candle_block = chr(10).join(candle_lines[-n_candles:]) if candle_lines else "N/A"

        # Account context
        equity = self.equity
        max_position_pct = 20 if equity < 500 else 5

        # Historical L-level patterns from journal (teaches LLM what works)
        historical_patterns = self._build_historical_pattern_context(asset)
        if len(historical_patterns) > 800:
            historical_patterns = historical_patterns[:800] + "\n[truncated]"

        # Build warnings string
        warnings_str = chr(10).join(f'  - {w}' for w in (math_filter_warnings or [])) or '  - No warnings — all clear'

        prompt = f"""You are a PROFIT-FOCUSED trading brain for {asset}/USDT perpetual futures. ONLY enter trades that MAKE MONEY.

═══ PROVEN STRATEGY (backtested 6 months: 72% WR, PF 1.19) ═══
The EMA(8) TREND LINE strategy works as follows — follow EXACTLY:

ENTRY: Enter when EMA(8) forms a NEW LINE (direction changes after 3+ bars trending opposite).
- Price must be on correct side of EMA (above for CALL, below for PUT)
- Entry score >= 7 (indicators + multi-TF alignment)
- The EMA line IS the trend. We ride it until it flips.

EXIT RULES (in priority order — this is CRITICAL):
1. HARD STOP -2%: Emergency protection. Always active.
2. EMA NEW LINE EXIT: When EMA reverses 2+ bars AND price crosses EMA — BUT ONLY WHEN IN PROFIT.
   - This exit has 100% win rate. Never use it when losing (18% WR = terrible).
3. EMA LINE-FOLLOWING SL: After 5 min, SL tracks the EMA line with buffer.
   - Ratchet breakeven at 1.0% profit. Lock gains from 1.5%+.

KEY LESSONS FROM BACKTEST DATA:
- Grace period 3 min minimum — early SL checks kill 37% of trades at 0% WR
- L2+ ratchet levels = 100% WR. Let trades breathe to reach them.
- EMA exits on LOSING trades = 18% WR. Let SL handle losses instead.
- Winners ride 1-3 hours avg. Losers die in <40 min. Don't cut winners early.

YOUR KEY ROLE: Choose the BEST TIMEFRAME to trade on.
Pick the timeframe with the CLEANEST EMA new-line setup.

TIMEFRAME SELECTION GUIDE:
- 5m: Standard trades. Best signal/noise ratio. Most common choice.
- 15m: Intermediate. Wider SL, stronger trend confirmation needed.
- 1h: Swing trades. Wide SL but powerful trends. High conviction only.
- 4h: Major moves only. Very wide SL. Crystal-clear setups only.
- HIGHER TF signals that AGREE with lower TF = strongest setups.
- If TFs CONFLICT = dangerous. REJECT or pick the dominant trend TF.

ACTIVE SIGNALS ON ALL TIMEFRAMES:
{mtf_signal_block}

Price: ${price:,.2f} | Equity: ${equity:,.0f}
Entry Score (5m): {entry_score}/10 ({', '.join(score_reasons or []) or 'none'})
Support: ${support:.2f} | Resistance: ${resistance:.2f}

ML PATTERN RECOGNITION (trained on real {asset} data):
USE ML SIGNALS to judge if this setup matches historical L3+ runners or L1 deaths.

Warnings: {'; '.join(math_filter_warnings or []) or 'None'}

CANDLES (5m, newest last):
{candle_block}

QUANT AGENTS:
{agent_data}

{historical_patterns}

DECISION FRAMEWORK:
1. TIMEFRAME: Which TF has the cleanest crossover? Prefer TFs with history of profit.
2. DIRECTION: CALL or PUT based on chosen TF's signal.
3. ML CONSENSUS: Do ML models agree? If 2+ disagree -> REJECT.
4. ENTRY TIMING: Gap from EMA. <2%=excellent, >3%=late, >5%=REJECT.
5. VOLUME: Rising=trend continues. Declining=dying.
6. CANDLE QUALITY: Big bodies + small wicks = conviction. Dojis = SKIP.
7. RISK: What kills this trade? Score 0-10.
8. PROFIT PATH: Can this reach L4+ on the chosen timeframe?

POSITION SIZING (investment protection):
- trade_quality >= 8 AND risk <= 3: full size ({max_position_pct}% equity)
- trade_quality 6-7: half size ({max_position_pct//2}% equity)
- trade_quality <= 5: REJECT

RESPOND WITH ONLY JSON:
{{"proceed": <true/false>, "chosen_timeframe": "<1m/5m/15m/1h/4h>", "chosen_direction": "<CALL/PUT>", "confidence": <0.0-1.0 probability of L3+>, "position_size_pct": <1-{max_position_pct}>, "risk_score": <0-10>, "trade_quality": <0-10>, "predicted_l_level": "<L1/L2/L3/L4/L6+/L10+>", "bull_case": "<specific reason for L4+ on chosen TF>", "bear_case": "<specific risk that kills this trade>", "facilitator_verdict": "<honest final decision — which TF and why>"}}

CRITICAL: If no timeframe has a clean setup, set proceed=false. Capital preservation > missed trades."""

        try:
            raw = self._query_llm_auto(prompt)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                raw = self._extract_json(raw)
                result = json.loads(raw)

            # ── Prompt Constraint Validation (safety layer) ──
            if self._prompt_constraints:
                try:
                    result, violations = self._prompt_constraints.validate_response(result)
                    if violations:
                        print(f"  [{self._ex_tag}:{asset}] LLM VIOLATIONS: {'; '.join(violations[:3])}")
                except Exception as vc_err:
                    logger.debug(f"Prompt constraint validation error: {vc_err}")

            # Validate and clamp all fields
            result['proceed'] = bool(result.get('proceed', False))
            result['confidence'] = max(0.0, min(1.0, float(result.get('confidence', 0.5))))
            result['position_size_pct'] = max(1, min(max_position_pct, float(result.get('position_size_pct', 3))))
            result['risk_score'] = max(0, min(10, int(result.get('risk_score', 5))))
            result['trade_quality'] = max(0, min(10, int(result.get('trade_quality', 5))))

            # Validate chosen_timeframe — must be one of the active signals
            active_tf_signals = active_tf_signals or {}
            _default_tf = self.SIGNAL_TIMEFRAMES[0] if self.SIGNAL_TIMEFRAMES else '5m'
            chosen_tf = str(result.get('chosen_timeframe', _default_tf))
            if chosen_tf not in self.SIGNAL_TIMEFRAMES:
                chosen_tf = _default_tf
            # If LLM chose a TF with no active signal, pick the best active one
            if active_tf_signals and chosen_tf not in active_tf_signals:
                # Fall back to highest entry_score active TF
                chosen_tf = max(active_tf_signals.keys(),
                               key=lambda t: active_tf_signals[t].get('trend_bars', 0))
            result['chosen_timeframe'] = chosen_tf

            # Validate chosen_direction
            chosen_dir = str(result.get('chosen_direction', 'CALL')).upper()
            if chosen_dir not in ('CALL', 'PUT'):
                chosen_dir = 'CALL' if signal == 'BUY' else 'PUT'
            result['chosen_direction'] = chosen_dir

            # Server-side enforcement — override LLM if it ignores rules
            tq = result['trade_quality']
            rs = result['risk_score']
            if tq <= 3:
                result['proceed'] = False
                result['confidence'] = min(result['confidence'], 0.3)
            if rs >= self.bear_veto_threshold:
                result['proceed'] = False
            if tq <= 5:
                result['position_size_pct'] = min(result['position_size_pct'], max_position_pct * 0.5)

            # ── Robinhood-specific response clamping ──
            if hasattr(self, '_round_trip_spread') and self._round_trip_spread > 1.0:
                # Floor risk awareness — Robinhood trades always have base risk from spread
                if result.get('risk_score', 0) < 3:
                    result['risk_score'] = 3
                # Cap position size at 5% (spread makes large positions dangerous)
                result['position_size_pct'] = min(result.get('position_size_pct', 5), 5)
                # Block low-quality trades even if LLM says proceed
                if result.get('trade_quality', 5) < 6 and result.get('proceed', False):
                    result['proceed'] = False
                    result['facilitator_verdict'] = 'BLOCKED: quality too low for high-spread exchange'
                # Force longs-only
                if hasattr(self, '_longs_only') and self._longs_only:
                    if result.get('chosen_direction', 'CALL') == 'PUT':
                        result['chosen_direction'] = 'CALL'
                        result['proceed'] = False
                        result['facilitator_verdict'] = 'BLOCKED: SHORT not allowed on Robinhood'

            # Print compact summary
            bull = str(result.get('bull_case', ''))[:60]
            bear = str(result.get('bear_case', ''))[:60]
            fac = str(result.get('facilitator_verdict', ''))[:80]
            pred_l = str(result.get('predicted_l_level', '?'))

            print(f"  [{self._ex_tag}:{asset}] UNIFIED LLM (multi-TF):")
            print(f"    CHOSEN TF: {chosen_tf} | {chosen_dir} | quality={tq}/10")
            print(f"    BULL: {bull}")
            print(f"    BEAR: risk={rs}/10 | {bear}")
            print(f"    L-LEVEL: {pred_l} | {'ENTER' if result['proceed'] else 'REJECT'} conf={result['confidence']:.2f} size={result['position_size_pct']:.0f}%")
            print(f"    VERDICT: {fac}")

            return result

        except Exception as e:
            logger.warning(f"[{asset}] Unified LLM failed: {e}")
            # Safe fallback — don't trade if LLM fails
            return {
                'proceed': False,
                'confidence': 0.3,
                'position_size_pct': 2,
                'risk_score': 5,
                'trade_quality': 3,
                'bull_case': f'LLM unavailable ({type(e).__name__})',
                'bear_case': 'Cannot assess risk — skipping',
                'facilitator_verdict': 'No LLM analysis — reject for safety',
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

        # Build ML warning block for bear agent
        ml_bear_warnings = []
        try:
            if hasattr(self, '_last_ml_context') and self._last_ml_context:
                mc = self._last_ml_context
                # HMM regime conflict
                hmm_r = mc.get('hmm_regime', '')
                if (signal == 'BUY' and hmm_r in ('BEAR', 'CRISIS')) or (signal == 'SELL' and hmm_r == 'BULL'):
                    ml_bear_warnings.append(f"HMM regime={hmm_r} CONFLICTS with {signal}")
                # LSTM says SKIP (binary model: setup predicts L1 death)
                lstm_s = mc.get('lstm_signal', '')
                lstm_c = mc.get('lstm_confidence', 0)
                if lstm_s == 'SKIP' and lstm_c > 0.50:
                    ml_bear_warnings.append(f"LSTM ensemble says SKIP ({lstm_c:.0%} conf) — setup matches L1 death patterns, trailing SL likely stops at loss")
                # LightGBM disagrees
                lgbm_d = mc.get('lgbm_direction', '')
                if (signal == 'BUY' and lgbm_d == 'SHORT') or (signal == 'SELL' and lgbm_d == 'LONG'):
                    ml_bear_warnings.append(f"LightGBM pattern model predicts {lgbm_d} (opposite)")
                # High shock probability
                shock = mc.get('patchtst_shock_prob', 0)
                if shock > 0.3:
                    ml_bear_warnings.append(f"PatchTST: {shock:.0%} liquidity shock risk")
                # Stale signal
                af = mc.get('alpha_freshness', 1.0)
                if af < 0.5:
                    ml_bear_warnings.append(f"Alpha Decay: signal is stale (freshness={af:.0%})")
                # GARCH expanding vol
                if mc.get('garch_vol_expanding', False):
                    ml_bear_warnings.append("GARCH: volatility expanding above average")
        except Exception:
            pass

        ml_warning_text = chr(10).join(f"  - {w}" for w in ml_bear_warnings) if ml_bear_warnings else "  - No ML model conflicts detected"

        bear_prompt = f"""You are a RISK ANALYST for {asset}/USDT perpetual futures.
Your job is to find reasons this trade SHOULD NOT be taken.
You are the last defense before capital is risked. Investment safety is #1 priority.

STRATEGY CONTEXT (backtested 6 months: 72% WR, PF 1.19):
- We enter on EMA(8) NEW LINE (direction change after 3+ bars). Exit via EMA reversal or SL.
- Hard stop -2%. EMA line-following SL activates after 5 min. Breakeven at 1.0% profit.
- 68-78% of SL exits are winners. The system is profitable WHEN we enter quality setups.
- YOUR JOB: Identify the 28% of trades that will hit hard stop (-2%). These share patterns:
  late entries (price far from EMA), declining volume, reversal wicks, fighting higher TF trend.

A bull analyst recommends: {signal} with confidence {bull_confidence:.2f}
Bull says: "{bull_reasoning}"

MARKET DATA:
Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction})
ATR: ${current_atr:.2f} | Support: ${support:.2f} | Resistance: ${resistance:.2f}
EMA slope: {ema_slope_pct:+.3f}%/bar | Trend bars: {consecutive_trend}
Distance from EMA: {separation_pct:.2f}%
Volume trend: {vol_trend}
{wick_warning}

ML MODEL WARNINGS:
{ml_warning_text}

LAST {n_candles} CANDLES (5m):
{candle_data}

SCORE EACH RISK 0-2 POINTS:

1. LATE ENTRY: Price {separation_pct:.2f}% from EMA.
   - 15%+ from EMA = clearly late (2pts), 8-15% = somewhat extended (1pt), <8% = normal (0pts)

2. REVERSAL SIGNS: {wick_warning or "Check candles above."}
   - Clear reversal pattern (multiple wicks against trend) = 2pts, single wick = 1pt, none = 0pts

3. VOLUME: Volume is {vol_trend}.
   - Declining on strong price move = 2pts, flat = 1pt, increasing = 0pts

4. KEY LEVEL: {"LONG near resistance $"+f"{resistance:,.2f} ({level_dist_pct:.1f}% away)" if signal=="BUY" else "SHORT near support $"+f"{support:,.2f} ({level_dist_pct:.1f}% away)"}
   - Within 0.5% = 2pts, 0.5-2% = 1pt, >2% = 0pts

5. ML MODELS + LOSS STREAK: {len(ml_bear_warnings)} ML conflicts + {recent_losses}/10 recent losses.
   - 2+ ML conflicts OR 5+ losses = 2pts, 1 conflict OR 3-4 losses = 1pt, none = 0pts

TOTAL risk_score = sum (0-10)
0-4: Low risk (proceed), 5-6: Medium (reduce position 50%), 7-8: High (reduce 75%), 9-10: VETO

IMPORTANT: Only score 9-10 if MULTIPLE serious risks combine.
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
- If bull_conf < 0.80 → reject
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

    # ── Agentic shadow loop (C4d) ───────────────────────────────────────
    #
    # Runs the agentic LLM trade-plan compiler in parallel with the
    # existing decision path. Observation-only: the compiled plan is
    # logged to warm_store for A/B analysis against what the executor
    # actually traded. No order is placed through this path yet — that
    # flip happens after a paper soak shows shadow plans are sensible.
    #
    # Cost: one LLM cycle per asset per tick when enabled. Acceptable on
    # the 60s/180s poll schedule; set ACT_AGENTIC_LOOP=0 to disable if
    # the quota/rate-limit tightens.

    def submit_trade_plan(self, plan, mode: str = "paper") -> Dict[str, Any]:
        """Submit a compiled TradePlan through the full gate stack to
        the paper-order path. C26 Step 3 — makes the agentic brain's
        structured output actually fire paper trades.

        Gates run in order: authority → conviction → cost → readiness
        (real only) → operator pre_trade_submit hook. Any rejection
        returns {submitted: False, reason}. On acceptance, calls
        self._paper.record_entry() and writes a NON-shadow decision to
        warm_store, fires post_trade_open hook.

        Kill switch: ACT_DISABLE_AGENTIC_SUBMIT=1 → returns
        {submitted: False, reason: 'disabled'} without running gates.

        Real-capital mode (mode='real') additionally requires:
          - readiness_gate.evaluate().open_ == True
          - os.environ['ACT_REAL_CAPITAL_ENABLED'] == '1'
        """
        if os.environ.get("ACT_DISABLE_AGENTIC_SUBMIT", "").strip() == "1":
            return {"submitted": False, "reason": "disabled_by_env"}

        if plan is None or getattr(plan, "direction", "") in ("SKIP", "FLAT", ""):
            return {"submitted": False, "reason": "no_actionable_plan"}

        asset = str(getattr(plan, "asset", "") or "").upper()
        direction = str(getattr(plan, "direction", "")).upper()
        if asset not in ("BTC", "ETH"):
            return {"submitted": False, "reason": f"unsupported_asset:{asset}"}
        if direction not in ("LONG", "SHORT"):
            return {"submitted": False, "reason": f"unsupported_direction:{direction}"}

        # --- Gate 1: Authority rules (hard-coded operator PDF rules) ---
        # Pass only the minimum context the plan shape carries; fields we
        # don't know (trade_type, htf_trend_direction, etc.) are omitted
        # so per-rule soft-skip applies rather than false-positive veto.
        try:
            from src.ai.authority_rules import validate_authority_entry
            _raw = 1 if direction == "LONG" else -1
            _ctx = {"raw_signal": _raw, "asset": asset}
            # Map TradePlan tier -> authority trade_type taxonomy if
            # possible. Authority expects {scalp|intraday|swing} — plan
            # tier is {sniper|normal|skip}; treat them as orthogonal and
            # only pass trade_type when caller set an explicit one.
            plan_trade_type = getattr(plan, "trade_type", None)
            if plan_trade_type:
                _ctx["trade_type"] = str(plan_trade_type).lower()
            ok, violations = validate_authority_entry({}, _ctx)
            if not ok:
                return {
                    "submitted": False, "reason": "authority",
                    "violations": list(violations),
                }
        except Exception as _e:
            logger.debug("submit_trade_plan: authority check skipped: %s", _e)

        # --- Gate 2: Real-capital guard (mode='real' only) ---
        if mode == "real":
            if os.environ.get("ACT_REAL_CAPITAL_ENABLED", "").strip() != "1":
                return {"submitted": False, "reason": "real_capital_flag_unset"}
            try:
                from src.orchestration.readiness_gate import evaluate as _rg_eval
                rg = _rg_eval()
                if not getattr(rg, "open_", False):
                    return {
                        "submitted": False, "reason": "readiness_closed",
                        "failing": list(getattr(rg, "failing_conditions", [])),
                    }
            except Exception as _e:
                return {"submitted": False, "reason": f"readiness_check_failed:{_e}"}

        # --- Gate 3: Cost awareness (reject if margin < threshold) ---
        try:
            from src.trading import cost_gate
            expected_pct = 0.0
            pnl_range = getattr(plan, "expected_pnl_pct_range", None)
            if pnl_range and len(pnl_range) >= 2:
                expected_pct = float(pnl_range[1])
            cg = cost_gate.evaluate(
                expected_return_pct=expected_pct,
                venue="robinhood",
                size_pct=float(getattr(plan, "size_pct", 1.0)),
                direction=direction,
            )
            if not cg.passed:
                return {"submitted": False, "reason": cg.reason}
        except Exception as _e:
            logger.debug("submit_trade_plan: cost gate skipped: %s", _e)

        # --- Gate 4: Pre-trade hook (blocking — operator veto) ---
        try:
            from src.orchestration.hooks import fire as fire_event
            result = fire_event(
                "pre_trade_submit",
                {"plan": plan.to_dict() if hasattr(plan, "to_dict") else {},
                 "mode": mode, "asset": asset, "direction": direction},
            )
            if isinstance(result, dict) and result.get("veto"):
                return {"submitted": False, "reason": f"hook_veto:{result.get('reason', '')}"}
        except Exception as _e:
            logger.debug("submit_trade_plan: pre_trade_submit hook skipped: %s", _e)

        # --- All gates passed: submit paper order ---
        entry_price = float(getattr(plan, "entry_price", 0.0) or 0.0)
        sl_price = float(getattr(plan, "sl_price", 0.0) or 0.0)
        size_pct = float(getattr(plan, "size_pct", 1.0) or 1.0)
        qty = 0.0
        try:
            if self._paper and entry_price > 0:
                # Equity-scaled quantity
                equity = float(getattr(self._paper, "equity", 100000.0))
                notional = equity * (size_pct / 100.0)
                qty = notional / entry_price
                tp_price = 0.0
                tp_levels = getattr(plan, "tp_levels", None) or []
                if tp_levels:
                    first = tp_levels[0]
                    tp_price = float(getattr(first, "price", 0.0) or
                                      (first.get("price", 0.0) if isinstance(first, dict) else 0.0))
                self._paper.record_entry(
                    asset=asset, direction=direction, price=entry_price,
                    score=int(getattr(plan, "conviction_score", 5) or 5),
                    quantity=qty, sl_price=sl_price, tp_price=tp_price,
                    ml_confidence=float(getattr(plan, "confidence", 0.5) or 0.5),
                    llm_confidence=float(getattr(plan, "confidence", 0.5) or 0.5),
                    size_pct=size_pct,
                    reasoning=str(getattr(plan, "thesis", ""))[:200],
                )
                self._paper.save_state()
        except Exception as _pe:
            logger.warning(f"[SUBMIT_TRADE_PLAN] paper record failed: {_pe}")
            return {"submitted": False, "reason": f"paper_record_failed:{_pe}"}

        # --- Write non-shadow decision to warm_store ---
        try:
            from src.orchestration.warm_store import get_store
            import uuid as _uuid
            store = get_store()
            store.write_decision({
                "decision_id": f"agentic-{_uuid.uuid4().hex}",   # NOT "shadow-*"
                "symbol": asset,
                "direction": {"LONG": 1, "SHORT": -1}.get(direction, 0),
                "confidence": float(getattr(plan, "confidence", 0.5) or 0.5),
                "final_action": direction,
                "plan": plan.to_dict() if hasattr(plan, "to_dict") else {},
                "component_signals": {
                    "source": "agentic_brain_submit",
                    "mode": mode,
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "size_pct": size_pct,
                    "qty": qty,
                },
            })
        except Exception as _e:
            logger.warning("submit_trade_plan: warm_store write failed: %s", _e)

        # --- Fire post-trade open hook ---
        try:
            from src.orchestration.hooks import fire as fire_event
            fire_event("post_trade_open", {
                "asset": asset, "direction": direction,
                "entry_price": entry_price, "size_pct": size_pct,
                "mode": mode,
            })
        except Exception:
            pass

        try:
            print(
                f"  [{self._ex_tag}:{asset}] AGENTIC SUBMIT {direction} "
                f"@ ${entry_price:,.2f} size={size_pct:.2f}% qty={qty:.6f} "
                f"(mode={mode})"
            )
        except Exception:
            pass

        return {
            "submitted": True, "asset": asset, "direction": direction,
            "entry_price": entry_price, "size_pct": size_pct,
            "qty": qty, "mode": mode,
        }

    def _run_agentic_shadow(self, asset: str, closes) -> None:
        """Fire the agentic loop in shadow mode. Never raises.

        C17 — delegates to src.ai.shadow_tick.run_tick which orchestrates
        web-bundle fetch → graph ingest → scanner brain publish →
        analyst plan compile → persona refresh, all throttled per-asset.

        C26 Step 3 — when the analyst returns a valid non-skip plan AND
        ACT_DISABLE_AGENTIC_SUBMIT is not set AND paper mode is active,
        we additionally route the plan through submit_trade_plan() so
        it reaches the paper-order path. Shadow audit still writes.
        """
        try:
            from src.ai.agentic_bridge import agentic_loop_enabled
            from src.ai.shadow_tick import run_tick
        except Exception:
            return

        if not agentic_loop_enabled(getattr(self, 'config', None)):
            return

        try:
            last_close = float(closes[-1]) if closes is not None and len(closes) > 0 else 0.0
            regime = getattr(self, '_last_regime', 'UNKNOWN') or 'UNKNOWN'
            quant_data = f"[ASSET={asset}] [LAST_CLOSE={last_close:.2f}] [REGIME={regime}]"
            tick_summary = run_tick(asset=asset, quant_digest=quant_data)
        except Exception as e:
            logger.debug("[agentic_shadow] %s tick failed: %s", asset, e)
            return

        # Unpack the LoopResult that run_tick stashed for us.
        result = tick_summary.get("_loop_result")
        if result is None:
            return

        # Log to warm_store for A/B analysis. Uses a shadow decision_id
        # so it doesn't collide with the executor's real decisions.
        try:
            from src.orchestration.warm_store import get_store
            import uuid as _uuid
            store = get_store()
            # Tag the row with the active brain profile + models so
            # the operator can A/B weekly profiles by querying
            # component_signals.analyst_model / scanner_model /
            # brain_profile in warm_store.
            try:
                from src.ai.dual_brain import (
                    ANALYST, SCANNER, _resolve, _resolve_profile,
                )
                _cfg = getattr(self, 'config', None)
                _prof = _resolve_profile(_cfg)
                _analyst_cfg = _resolve(_cfg, ANALYST)
                _scanner_cfg = _resolve(_cfg, SCANNER)
                _brain_tag = {
                    "brain_profile": next(
                        (name for name, p in __import__('src.ai.dual_brain',
                                                         fromlist=['BRAIN_PROFILES']
                                                         ).BRAIN_PROFILES.items()
                         if p is _prof),
                        "unknown",
                    ),
                    "analyst_model": _analyst_cfg.model,
                    "scanner_model": _scanner_cfg.model,
                }
            except Exception:
                _brain_tag = {}
            store.write_decision({
                "decision_id": f"shadow-{_uuid.uuid4().hex}",
                "symbol": asset,
                "direction": {"LONG": 1, "SHORT": -1}.get(result.plan.direction, 0),
                "confidence": float(getattr(result.plan, 'confidence', 0.0)),
                "final_action": f"SHADOW_{result.plan.direction}",
                "plan": result.plan.to_dict(),
                "component_signals": {
                    "source": "agentic_shadow",
                    "steps_taken": result.steps_taken,
                    "terminated_reason": result.terminated_reason,
                    "tool_calls": [t.get("name") for t in result.tool_calls],
                    # C17 tick telemetry:
                    "scanner_published": bool(tick_summary.get("scanner_published")),
                    "ingest_counts": tick_summary.get("ingest_counts") or {},
                    "personas_refreshed": bool(tick_summary.get("personas_refreshed")),
                    "persona_report": tick_summary.get("persona_report") or {},
                    **_brain_tag,
                },
            })
        except Exception as e:
            logger.debug("[agentic_shadow] %s warm_store write failed: %s", asset, e)

        # Compact log line for the dashboard / tail-f watchers.
        try:
            print(
                f"  [{self._ex_tag}:{asset}] SHADOW PLAN {result.plan.direction} "
                f"tier={result.plan.entry_tier} size={result.plan.size_pct}% "
                f"({result.terminated_reason}, {result.steps_taken} steps)"
            )
        except Exception:
            pass

        # C26 Step 3 — if the plan is actionable (not SKIP/FLAT) and the
        # operator hasn't disabled agentic submit, route through the
        # full gate stack into the paper-order path. Every gate still
        # runs; real-capital flag still required for mode='real'.
        try:
            plan_direction = str(getattr(result.plan, "direction", "")).upper()
            if plan_direction in ("LONG", "SHORT"):
                submit_result = self.submit_trade_plan(result.plan, mode="paper")
                if submit_result.get("submitted"):
                    logger.info(
                        "[agentic_submit] %s %s submitted via paper path",
                        asset, plan_direction,
                    )
                elif submit_result.get("reason") not in ("disabled_by_env", "no_actionable_plan"):
                    logger.debug(
                        "[agentic_submit] %s %s rejected: %s",
                        asset, plan_direction, submit_result.get("reason"),
                    )
        except Exception as _submit_e:
            logger.debug("[agentic_submit] %s call failed: %s", asset, _submit_e)
