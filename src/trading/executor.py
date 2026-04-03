"""
Trading Executor — EMA(8) Crossover with LLM Confirmation
==========================================================
Bybit USDT perpetual futures. LONG (CALL) and SHORT (PUT).
Dynamic trailing stop-loss L1 -> L2 -> L3 -> L4 ...
"""

import os
import re
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any

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

        # Quality gates — bear veto handles dangerous trades, so confidence bar can be lower
        self.min_confidence: float = 0.60  # Data: conf >= 0.7 is profitable. 0.6 gives brain some room.
        self.min_atr_ratio: float = 0.0003
        self.trade_cooldown: float = 300.0       # 5 min between any trades (was 120 — caused churning)
        self.post_close_cooldown: float = 600.0   # 10 min after closing before new entry (was 180 — too short)
        self.asset_loss_streak: Dict[str, int] = {}    # consecutive losses per asset
        self.asset_cooldown_until: Dict[str, float] = {}  # timestamp when asset can trade again

        # Exchange tag for output (prevents interleaved confusion in multi-exchange mode)
        # MUST be set early — brain, edge stats, journal all use it
        self._ex_tag: str = config.get('exchange', {}).get('name', 'bybit').upper()

        # Exchange
        exchange_name = config.get('exchange', {}).get('name', 'bybit')
        testnet = config.get('mode', 'testnet') in ('testnet', 'paper')
        self.price_source = PriceFetcher(exchange_name=exchange_name, testnet=testnet)

        # LLM strategist (used as fallback / for deeper analysis)
        provider = ai_cfg.get('reasoning_provider', 'auto')
        model = ai_cfg.get('reasoning_model', 'mistral:latest')
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

        # Trading Brain v2 — multi-model consensus (mistral + llama3.2), CoT, memory, regime, Kelly, session
        self._brain: Optional[TradingBrainV2] = None
        if BRAIN_V2_AVAILABLE:
            try:
                journal_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs', 'trading_journal.jsonl')
                self._brain = TradingBrainV2(
                    ollama_base_url=self.ollama_base_url,
                    journal_path=journal_path,
                    exchange=self._ex_tag.lower(),
                )
                print(f"  [AI] Trading Brain v2 ACTIVE — multi-model (mistral+llama3.2), CoT, memory, regime, Kelly, session")
            except Exception as e:
                print(f"  [AI] Trading Brain v2 init failed ({e}) — using legacy LLM")
                self._brain = None

        # ── Trade Protections (freqtrade-inspired) ──
        self._protections = None
        if PROTECTIONS_AVAILABLE:
            try:
                prot_cfg = {
                    "sl_guard": {"trade_limit": 3, "lookback_minutes": 60, "cooldown_minutes": 30},
                    "max_drawdown": {"max_drawdown_pct": risk.get('max_drawdown_pct', 10.0),
                                     "lookback_minutes": 120, "cooldown_minutes": 60},
                    "pair_lock": {"min_profit_pct": -3.0, "lookback_trades": 8, "lock_hours": 2},
                    "roi_table": {0: 0.10, 30: 0.05, 60: 0.025, 120: 0.01, 240: 0.0},
                    "confirm": {"max_spread_pct": 1.0, "max_price_drift_pct": 0.5,
                                "max_concurrent_trades": len(self.assets) * 2},
                    "position_adjust": {
                        "max_dca_entries": 1, "dca_threshold_pct": -3.0,
                        "dca_multiplier": 0.3,
                        "partial_exit_levels": [(4.0, 0.25), (7.0, 0.25), (12.0, 0.25)],
                    },
                }
                self._protections = TradeProtections(prot_cfg)
                print(f"  [PROTECT] Trade Protections ACTIVE — SL guard, drawdown, pair lock, ROI table, partial exits")
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

        # Set leverage to 1x (prevent amplified losses)
        try:
            if self._exchange_client:
                for asset in self.assets:
                    sym = self._get_symbol(asset)  # Uses correct format per exchange
                    try:
                        self._exchange_client.exchange.set_leverage(1, sym)
                        print(f"  [{self._ex_tag}:{asset}] Leverage set to 1x on {self._exchange_name}")
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

        # Bear/Risk veto agent
        self.bear_enabled: bool = ai_cfg.get('bear_agent_enabled', True)
        self.bear_veto_threshold: int = ai_cfg.get('bear_veto_threshold', 7)
        self.bear_reduce_threshold: int = ai_cfg.get('bear_reduce_threshold', 5)
        self.bear_veto_stats: Dict[str, Dict[str, int]] = {}
        for a in self.assets:
            self.bear_veto_stats[a] = {'vetoed': 0, 'reduced': 0, 'passed': 0}

        # ── Portfolio-level drawdown limit (Freqtrade pattern) ──
        # Halt ALL trading if cumulative realized losses exceed threshold
        self.max_drawdown_pct: float = risk.get('max_drawdown_pct', 10.0)
        self.daily_loss_limit_pct: float = risk.get('daily_loss_limit_pct', 3.0)
        self.session_start_equity: float = self.initial_capital
        self.session_realized_pnl: float = 0.0
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

            # Basic consensus for logging
            directions = [v.direction * v.confidence for v in votes.values()]
            net = sum(directions) / len(directions) if directions else 0
            avg_conf = sum(v.confidence for v in votes.values()) / len(votes) if votes else 0.5

            consensus_dir = "CALL" if net > 0.15 else "PUT" if net < -0.15 else "FLAT"
            consensus = "STRONG" if abs(net) > 0.6 else "MODERATE" if abs(net) > 0.3 else "WEAK"

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
                    result = self.price_source.place_order(
                        symbol=symbol,
                        side=close_side,
                        amount=qty,
                        order_type='limit',
                        price=close_price,
                        reduce_only=True,
                    )
                else:
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
                    'sandbox': True,
                    'options': {'defaultType': 'future'},
                    'enableRateLimit': True,
                })
                if True:  # testnet
                    new_ex.urls['api'] = {
                        'public': 'https://cdn-ind.testnet.deltaex.org',
                        'private': 'https://cdn-ind.testnet.deltaex.org',
                    }
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
        print(f"  EMA(8) Crossover + LLM | {ex_name} Futures")
        print(f"  Assets: {self.assets} | Poll: {self.poll_interval}s")
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

                # Fetch REAL account equity (wallet balance + unrealized PnL)
                # Previously used wallet balance which hid -$31K in open losses
                unrealized_pnl = 0.0
                wallet_balance = 0.0
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
        # 5m CANDLES ONLY — analyze 5m, trade 5m, check every 10s
        # ══════════════════════════════════════════════════════════════

        try:
            raw_5m = self.price_source.fetch_ohlcv(symbol, timeframe='5m', limit=100)
        except Exception as e:
            print(f"  [{self._ex_tag}:{asset}] OHLCV fetch failed: {e}")
            # Try reconnecting the exchange client
            self._try_reconnect(asset)
            return
        ohlcv = PriceFetcher.extract_ohlcv(raw_5m)

        closes = ohlcv['closes']
        highs = ohlcv['highs']
        lows = ohlcv['lows']
        opens = ohlcv['opens']
        volumes = ohlcv['volumes']

        if len(closes) < 20:
            print(f"  [{self._ex_tag}:{asset}] Not enough 5m data ({len(closes)} candles)")
            return

        # ── STALENESS DETECTOR: catch frozen/stale OHLCV connections ──
        # If last 10 candles have identical close prices, the data is frozen
        # This happens when CCXT connection goes stale after hours of running
        if len(closes) >= 10:
            last_10 = closes[-10:]
            if len(set(round(c, 2) for c in last_10)) <= 1:
                # All 10 candles identical = FROZEN DATA
                print(f"  [{self._ex_tag}:{asset}] STALE DATA: last 10 candles all ${last_10[-1]:,.2f} - reconnecting")
                self._try_reconnect(asset)
                return

        # Also check candle timestamps — if newest candle is >15 min old, data is stale
        if raw_5m and len(raw_5m) > 0:
            newest_ts = raw_5m[-1][0] / 1000.0  # ms → seconds
            age_minutes = (time.time() - newest_ts) / 60
            if age_minutes > 15:
                print(f"  [{self._ex_tag}:{asset}] STALE CANDLES: newest is {age_minutes:.0f}min old - reconnecting")
                self._try_reconnect(asset)
                return

        # ── TESTNET PRICE SANITY: reject insane candle jumps (>8% between candles) ──
        # Bybit testnet BTC swings from $110K to $922K — these aren't real prices
        # Real BTC moves ~0.1% per 5min candle, not 50%
        if len(closes) >= 3:
            for i in range(-2, 0):
                prev_c = closes[i - 1]
                curr_c = closes[i]
                if prev_c > 0:
                    jump_pct = abs(curr_c - prev_c) / prev_c * 100
                    if jump_pct > 8.0:
                        print(f"  [{self._ex_tag}:{asset}] PRICE JUMP: {jump_pct:.1f}% between candles (${prev_c:,.0f} -> ${curr_c:,.0f}) — skipping unreliable data")
                        return

        # ══════════════════════════════════════════════════════════════
        # HIGHER TIMEFRAME TREND ALIGNMENT (1h + 4h)
        # Fetch 1h and 4h candles, compute EMA(8) direction on each
        # Used to filter entries: only trade when HTFs agree with 5m
        # ══════════════════════════════════════════════════════════════
        htf_1h_direction = "FLAT"
        htf_4h_direction = "FLAT"
        htf_alignment = 2  # default neutral (don't block if fetch fails)
        ohlcv_1h = None  # Pre-init for MTF candle builder later
        try:
            raw_1h = self.price_source.fetch_ohlcv(symbol, timeframe='1h', limit=50)
            ohlcv_1h = PriceFetcher.extract_ohlcv(raw_1h)
            closes_1h = ohlcv_1h['closes']
            if len(closes_1h) >= 10:
                ema_1h = ema(closes_1h, self.ema_period)
                if len(ema_1h) >= 3:
                    if ema_1h[-1] > ema_1h[-2] > ema_1h[-3]:
                        htf_1h_direction = "RISING"
                    elif ema_1h[-1] < ema_1h[-2] < ema_1h[-3]:
                        htf_1h_direction = "FALLING"
        except Exception as e:
            logger.debug(f"1h OHLCV fetch failed for {asset}: {e}")

        try:
            raw_4h = self.price_source.fetch_ohlcv(symbol, timeframe='4h', limit=30)
            ohlcv_4h = PriceFetcher.extract_ohlcv(raw_4h)
            closes_4h = ohlcv_4h['closes']
            if len(closes_4h) >= 10:
                ema_4h = ema(closes_4h, self.ema_period)
                if len(ema_4h) >= 3:
                    if ema_4h[-1] > ema_4h[-2] > ema_4h[-3]:
                        htf_4h_direction = "RISING"
                    elif ema_4h[-1] < ema_4h[-2] < ema_4h[-3]:
                        htf_4h_direction = "FALLING"
        except Exception as e:
            logger.debug(f"4h OHLCV fetch failed for {asset}: {e}")

        # Note: htf_alignment is computed after ema_direction is determined below

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

        # ── Compute HTF alignment score now that 5m ema_direction is known ──
        htf_alignment = 0
        # Point for 5m direction (always counts as 1)
        htf_alignment += 1
        # Point if 1h agrees with 5m
        if htf_1h_direction == ema_direction:
            htf_alignment += 1
        elif htf_1h_direction == "FLAT":
            pass  # neutral, no penalty
        # Point if 4h agrees with 5m
        if htf_4h_direction == ema_direction:
            htf_alignment += 1
        elif htf_4h_direction == "FLAT":
            pass  # neutral, no penalty

        # ══════════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME CANDLE DATA for LLM brain (1m, 15m, 1h)
        # LLMs analyze ALL timeframes to find patterns humans miss
        # ══════════════════════════════════════════════════════════════
        mtf_candle_summary = ""
        try:
            mtf_parts = []

            # 1m candles (last 10 = last 10 minutes of micro-structure)
            try:
                raw_1m = self.price_source.fetch_ohlcv(symbol, timeframe='1m', limit=20)
                ohlcv_1m = PriceFetcher.extract_ohlcv(raw_1m)
                if len(ohlcv_1m['closes']) >= 5:
                    c1m = ohlcv_1m['closes']
                    h1m = ohlcv_1m['highs']
                    l1m = ohlcv_1m['lows']
                    o1m = ohlcv_1m['opens']
                    v1m = ohlcv_1m['volumes']
                    lines_1m = []
                    for i in range(-min(10, len(c1m)), 0):
                        idx = len(c1m) + i
                        lines_1m.append(f"  O={o1m[idx]:.2f} H={h1m[idx]:.2f} L={l1m[idx]:.2f} C={c1m[idx]:.2f} V={v1m[idx]:.0f}")
                    mtf_parts.append(f"1-MINUTE CANDLES (last {len(lines_1m)}, microstructure):" + chr(10) + chr(10).join(lines_1m))
            except Exception:
                pass

            # 15m candles (last 8 = last 2 hours of intermediate structure)
            try:
                raw_15m = self.price_source.fetch_ohlcv(symbol, timeframe='15m', limit=20)
                ohlcv_15m = PriceFetcher.extract_ohlcv(raw_15m)
                if len(ohlcv_15m['closes']) >= 5:
                    c15 = ohlcv_15m['closes']
                    h15 = ohlcv_15m['highs']
                    l15 = ohlcv_15m['lows']
                    o15 = ohlcv_15m['opens']
                    v15 = ohlcv_15m['volumes']
                    ema_15m = ema(c15, self.ema_period)
                    dir_15m = "RISING" if len(ema_15m) >= 3 and ema_15m[-1] > ema_15m[-2] > ema_15m[-3] else \
                              "FALLING" if len(ema_15m) >= 3 and ema_15m[-1] < ema_15m[-2] < ema_15m[-3] else "FLAT"
                    lines_15m = []
                    for i in range(-min(8, len(c15)), 0):
                        idx = len(c15) + i
                        e15 = ema_15m[idx] if idx < len(ema_15m) else 0
                        lines_15m.append(f"  O={o15[idx]:.2f} H={h15[idx]:.2f} L={l15[idx]:.2f} C={c15[idx]:.2f} V={v15[idx]:.0f} EMA={e15:.2f}")
                    mtf_parts.append(f"15-MINUTE CANDLES (EMA direction: {dir_15m}):" + chr(10) + chr(10).join(lines_15m))
            except Exception:
                pass

            # 1h candles (already fetched — build text for LLM)
            try:
                if ohlcv_1h and len(ohlcv_1h.get('closes', [])) >= 5:
                    c1h_c = ohlcv_1h['closes']
                    h1h = ohlcv_1h['highs']
                    l1h = ohlcv_1h['lows']
                    o1h = ohlcv_1h['opens']
                    v1h = ohlcv_1h['volumes']
                    ema_1h_vals = ema(c1h_c, self.ema_period) if len(c1h_c) >= self.ema_period else []
                    lines_1h = []
                    for i in range(-min(6, len(c1h_c)), 0):
                        idx = len(c1h_c) + i
                        e1h = ema_1h_vals[idx] if idx < len(ema_1h_vals) else 0
                        lines_1h.append(f"  O={o1h[idx]:.2f} H={h1h[idx]:.2f} L={l1h[idx]:.2f} C={c1h_c[idx]:.2f} V={v1h[idx]:.0f} EMA={e1h:.2f}")
                    mtf_parts.append(f"1-HOUR CANDLES (EMA direction: {htf_1h_direction}):" + chr(10) + chr(10).join(lines_1h))
            except Exception:
                pass

            if mtf_parts:
                mtf_candle_summary = chr(10) + chr(10).join(mtf_parts)
        except Exception as e:
            logger.debug(f"MTF candle fetch error for {asset}: {e}")

        # Fetch L2 order book for support/resistance walls
        try:
            order_book = self.price_source.fetch_order_book(symbol, limit=25)
            ob_levels = self._extract_ob_levels(order_book, tick_price)
        except Exception:
            ob_levels = {'bid_wall': 0, 'ask_wall': 0, 'bid_walls': [], 'ask_walls': [], 'imbalance': 0, 'bid_depth_usd': 0, 'ask_depth_usd': 0}

        # ── Feed VPIN guard with recent candle volume ──
        if asset in self._vpin_guards and len(closes) >= 3:
            try:
                for i in range(-3, -1):  # Last 2 confirmed candles
                    idx = len(closes) + i
                    c, o, v = closes[idx], opens[idx], volumes[idx]
                    side = 'buy' if c >= o else 'sell'
                    self._vpin_guards[asset].add_trade(c, v, side)
            except Exception:
                pass

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

        # PRICE MOMENTUM CHECK — detect reversals before EMA catches up
        # If last 3 confirmed candles are making lower closes, price is falling
        # even if EMA still says RISING (EMA lags on fast reversals)
        price_falling = False
        price_rising = False
        if len(closes) >= 5:
            c1, c2, c3 = closes[-2], closes[-3], closes[-4]  # last 3 confirmed
            # Use <= to handle stale testnet prices (same close repeated)
            if c1 < c2 and c1 < c3:
                price_falling = True  # latest confirmed is lower than previous 2
            elif c1 > c2 and c1 > c3:
                price_rising = True   # latest confirmed is higher than previous 2

        # OVEREXTENSION — computed and passed to LLM, NOT a hard block
        ema_separation = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
        if ema_separation > 10.0:
            print(f"  [{self._ex_tag}:{asset}] OVEREXTENDED NOTE: {ema_separation:.1f}% from EMA (LLM will decide)")

        # ── REVERSAL DETECTION: Counter-trend when exhaustion signals appear ──
        # Volume declining = trend losing steam
        vol_declining = False
        if len(volumes) >= 10:
            recent_vol = sum(volumes[-5:]) / 5
            prev_vol = sum(volumes[-10:-5]) / 5
            vol_declining = recent_vol < prev_vol * 0.7

        # Rejection wicks = trend being fought
        has_rejection_wicks = False
        if len(highs) >= 3 and len(lows) >= 3 and len(closes) >= 3:
            for i in range(-3, 0):
                total_range = highs[i] - lows[i]
                if total_range > 0:
                    upper_wick = highs[i] - max(closes[i], opens[i] if i + len(opens) >= 0 else closes[i])
                    lower_wick = min(closes[i], opens[i] if i + len(opens) >= 0 else closes[i]) - lows[i]
                    if upper_wick > total_range * 0.5 or lower_wick > total_range * 0.5:
                        has_rejection_wicks = True
                        break

        is_reversal_signal = False

        # CASE 1: Normal trend-following signals (EMA crossover)
        if ema_direction == "RISING" and price > current_ema and ema_crossed:
            if price_falling:
                signal = "NEUTRAL"  # EMA says buy but price is reversing down
            else:
                signal = "BUY"
        elif ema_direction == "FALLING" and price < current_ema and ema_crossed:
            if price_rising:
                signal = "NEUTRAL"  # EMA says sell but price is reversing up
            else:
                signal = "SELL"
        # Strong trend: price >1.5% from EMA — but still check momentum
        # Was 1% — too tight, triggered in ranges. 1.5% ensures real trend.
        elif ema_direction == "FALLING" and price < current_ema * 0.985:
            if not price_rising:
                signal = "SELL"
        elif ema_direction == "RISING" and price > current_ema * 1.015:
            if not price_falling:
                signal = "BUY"

        # CASE 2: REVERSAL signals — DISABLED
        # Data analysis: ALL profitable trades were trend-following (L3→L46 runners).
        # Counter-trend reversals fight the market and die at L1.
        # Keeping the code for future use but not generating reversal signals.
        # if signal == "NEUTRAL" and ema_direction == "RISING" and price > current_ema:
        #     ... reversal PUT logic disabled ...
        # if signal == "NEUTRAL" and ema_direction == "FALLING" and price < current_ema:
        #     ... reversal CALL logic disabled ...

        ob_imb = ob_levels.get('imbalance', 0)
        ob_bid = ob_levels.get('bid_wall', 0)
        ob_ask = ob_levels.get('ask_wall', 0)
        ob_info = f"OB[imb={ob_imb:+.2f}"
        if ob_bid > 0:
            ob_info += f" sup=${ob_bid:,.2f}"
        if ob_ask > 0:
            ob_info += f" res=${ob_ask:,.2f}"
        ob_info += "]"
        print(f"  [{self._ex_tag}:{asset}] ${tick_price:,.2f} (sig=${price:,.2f}) | EMA(5m): ${current_ema:.2f} {ema_direction} | Signal: {signal} | ATR: ${current_atr:.2f} | {ob_info}")

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

        # ── Position management uses LIVE price, entry uses CONFIRMED price ──
        if asset in self.positions:
            self._manage_position(asset, tick_price, ohlcv, ema_vals, atr_vals, ema_direction, signal, ob_levels)
        else:
            self._evaluate_entry(asset, tick_price, ohlcv, ema_vals, atr_vals,
                                 ema_direction, signal, closes, highs, lows,
                                 opens, volumes, current_ema, current_atr, ob_levels,
                                 htf_1h_direction=htf_1h_direction,
                                 htf_4h_direction=htf_4h_direction,
                                 htf_alignment=htf_alignment,
                                 is_reversal_signal=is_reversal_signal,
                                 mtf_candle_summary=mtf_candle_summary)

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
                        mtf_candle_summary: str = ""):

        ob_levels = ob_levels or {}

        # Only trade when there's a signal (BUY/SELL from crossover or strong trend)
        if signal == "NEUTRAL":
            return

        # Initialize math filter warnings list (used throughout evaluation)
        math_filter_warnings = []

        # ── TIME-OF-DAY CONTEXT (informational, not a block) ──
        # Data analysis: 06:00-17:00 UTC = profitable, overnight = losing
        # Passed to LLM as context but NOT blocking — user wants 24/7 trading
        import datetime
        utc_hour = datetime.datetime.utcnow().hour
        if utc_hour < 6 or utc_hour >= 18:
            math_filter_warnings.append(f"TIME: overnight session (UTC {utc_hour}:00) — historically weaker")

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
                        if age_s > 120:  # older than 2 min
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
                        # Reject if entry is >30% away from current price (testnet price artifact)
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
                            result = self.price_source.place_order(
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
                if range_pct < 0.3:
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
                    # Market is actively mean-reverting — trend signals will fail
                    print(f"  [{self._ex_tag}:{asset}] HURST BLOCK: H={hurst_value:.2f} ({hurst_regime}) R2={hurst_conf:.2f} — trend signals unreliable")
                    return
                elif hurst_regime == 'random' and hurst_conf > 0.7:
                    # Random walk — add warning but don't block (weaker signal)
                    math_filter_warnings.append(f"HURST: H={hurst_value:.2f} random walk — trend may not persist")
            except Exception:
                pass

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
            except Exception:
                pass

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

        # ── Loss streak info for LLM (hard block is above, this is informational) ──
        streak = self.asset_loss_streak.get(asset, 0)
        if streak >= 2:
            math_filter_warnings.append(f"LOSS STREAK: {streak} consecutive losses on {asset}")

        forced_action = "SHORT" if signal == "SELL" else "LONG" if signal == "BUY" else "FLAT"

        # ── Run Agent Orchestrator + Debate Engine (math agents) ──
        orch_result = self._run_orchestrator(
            asset, price, signal, closes, highs, lows, opens, volumes,
            ema_vals=ema_vals, atr_vals=atr_vals, ema_direction=ema_direction
        )

        # ══════════════════════════════════════════════════════════
        # TRADING BRAIN v2: Multi-model consensus + CoT + Memory + Regime + Kelly + Session
        # Falls back to legacy unified LLM if Brain v2 unavailable
        # ══════════════════════════════════════════════════════════
        if orch_result and orch_result.get('veto'):
            math_filter_warnings.append(f"AGENT VETO: consensus={orch_result.get('consensus','?')} — agents strongly disagree")

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
        if regime_lines:
            regime_context = chr(10) + "REGIME ANALYSIS:" + chr(10) + chr(10).join(regime_lines)

        # Append institutional + protector + regime data to candle text for LLM
        if institutional_context:
            candle_text = candle_text + chr(10) + institutional_context
        if profit_protector_context:
            candle_text = candle_text + chr(10) + profit_protector_context
        if regime_context:
            candle_text = candle_text + chr(10) + regime_context

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
            )

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
        size_pct = unified.get('position_size_pct', 3)
        reasoning = str(unified.get('facilitator_verdict', ''))[:120]
        risk_score = unified.get('risk_score', 5)

        # ══════════════════════════════════════════════════════════
        # BEAR VETO AGENT — Separate LLM call, contrarian prompt
        # Same model, different perspective: "What could go WRONG?"
        # ══════════════════════════════════════════════════════════
        if self.bear_enabled:
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

                # VETO: risk >= threshold → don't enter
                if bear_risk >= self.bear_veto_threshold:
                    if asset not in self.bear_veto_stats:
                        self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
                    self.bear_veto_stats[asset]['vetoed'] += 1
                    print(f"  [{self._ex_tag}:{asset}] BEAR VETO: risk={bear_risk}/10 >= {self.bear_veto_threshold} | {bear_reason}")
                    return
            except Exception as e:
                logger.warning(f"[{asset}] Bear agent error (proceeding without veto): {e}")

        # ── Bear REDUCE: risk between reduce_threshold and veto_threshold → halve position ──
        if risk_score >= self.bear_reduce_threshold and risk_score < self.bear_veto_threshold:
            old_size = size_pct
            size_pct = size_pct * 0.5
            if asset not in self.bear_veto_stats:
                self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
            self.bear_veto_stats[asset]['reduced'] += 1
            print(f"  [{self._ex_tag}:{asset}] BEAR REDUCE: risk={risk_score}/10 (>={self.bear_reduce_threshold}) — size {old_size:.0f}% -> {size_pct:.0f}%")

        # ── VPIN toxic flow → reduce position ──
        if vpin_status and vpin_status['is_toxic']:
            old_size = size_pct
            size_pct = size_pct * 0.5
            print(f"  [{self._ex_tag}:{asset}] VPIN REDUCE: toxic flow {vpin_status['vpin']:.2f} — size {old_size:.0f}% -> {size_pct:.0f}%")

        # LLM IS THE BRAIN — its confidence is the final confidence
        # No more Python-side blending or overriding
        print(f"  [{self._ex_tag}:{asset}] LLM DECISION: conf={confidence:.2f} size={size_pct:.0f}% risk={risk_score}/10 quality={unified.get('trade_quality', 0)}/10 hurst={hurst_value:.2f}")

        # Direction from EMA signal (always)
        action = "LONG" if signal == "BUY" else "SHORT"
        direction_label = "CALL" if signal == "BUY" else "PUT"

        # Quality gate: confidence
        if confidence < self.min_confidence:
            print(f"  [{self._ex_tag}:{asset}] SKIP: confidence {confidence:.2f} < {self.min_confidence}")
            return

        # Track bear stats
        if asset not in self.bear_veto_stats:
            self.bear_veto_stats[asset] = {'vetoed': 0, 'reduced': 0, 'passed': 0}
        self.bear_veto_stats[asset]['passed'] += 1

        # ── Edge Positioning: adjust size by historical win rate ──
        if self.edge_enabled and asset in self.edge_stats:
            edge = self.edge_stats[asset]
            if edge['total'] >= 5:
                mult = edge['edge_multiplier']
                old_size = size_pct
                size_pct = size_pct * mult
                if abs(mult - 1.0) > 0.05:
                    print(f"  [{self._ex_tag}:{asset}] EDGE: {mult:.2f}x ({edge['wins']}W/{edge['losses']}L) size {old_size:.0f}% -> {size_pct:.0f}%")

        # Calculate position size — use ACTUAL account equity (not config default)
        max_size_pct = 20 if self.equity < 500 else 5
        size_pct = max(1, min(max_size_pct, size_pct))
        if self.equity <= 0:
            print(f"  [{self._ex_tag}:{asset}] SKIP: no equity available")
            return

        notional = self.equity * (size_pct / 100.0)

        # Hard cap: proportional to account size, max $2K
        max_trade = min(2000.0, self.equity * 0.25)
        notional = min(notional, max_trade)

        print(f"  [{self._ex_tag}:{asset}] SIZING: {size_pct:.0f}% of ${self.equity:,.0f} = ${notional:,.0f} (max ${max_trade:,.0f})")

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

            # Compute initial stop-loss (L1) using ORDER BOOK + ATR
            # Priority: order book wall > ATR-based > percentage fallback
            sl_distance = current_atr * 1.5  # ATR-based SL (was 2.5 — too wide, L1 deaths lost $243K)
            sl_distance = max(sl_distance, price * 0.005)  # minimum 0.5% (was 1.0%)
            sl_distance = min(sl_distance, price * 0.02)   # maximum 2.0% (was 3.5% — way too wide)

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
            print(f"  [{self._ex_tag}:{asset}] ORDER OK: {order_id} | SL L1=${sl_price:,.2f} ({sl_source}) | OB imbalance={imb:+.2f}")

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
                'predicted_l_level': unified.get('predicted_l_level', '?'),
                'risk_score': risk_score,
                'bear_risk': risk_score if self.bear_enabled else 0,
                'hurst': hurst_value,
                'hurst_regime': hurst_regime,
                'breakeven_moved': False,
                'is_reversal': is_reversal_signal,  # reversal trades get tighter ratchet
                'agent_votes': orch_result.get('agent_votes', {}) if orch_result else {},
                'entry_tag': self._protections.tagger.tag_entry(
                    signal=signal, entry_score=entry_score,
                    regime=unified.get('brain_details', {}).get('regime', ''),
                    htf_alignment=htf_alignment, is_reversal=is_reversal_signal,
                    ema_slope=slope_pct, consensus=unified.get('brain_details', {}).get('consensus', '')
                ) if self._protections else 'untagged',
                'dca_count': 0,
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

        # ── 2. HARD STOP: max loss — non-negotiable ──
        # Bybit testnet has inflated prices → use wider stop to avoid noise hits
        hard_stop_pct = -2.5 if self._exchange_name == 'bybit' else -2.0  # Was -4/-3 — too much capital destroyed per trade
        # But if asset is blacklisted (stuck, can't close), just log and skip
        is_stuck = asset in self.failed_close_assets
        if pnl_pct <= hard_stop_pct:
            if is_stuck:
                print(f"  [{self._ex_tag}:{asset}] STUCK {pnl_pct:+.2f}% (can't close — no liquidity)")
                return
            print(f"  [{self._ex_tag}:{asset}] HARD STOP at ${price:,.2f} | P&L: {pnl_pct:+.2f}%")
            self._close_position(asset, price, f"Hard stop {pnl_pct:+.1f}%")
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
                            result = self.price_source.place_order(
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
        confirmed_close = closes[-2] if len(closes) >= 2 else price
        sl_hit = False
        if direction == 'LONG' and confirmed_close <= sl:
            sl_hit = True
        elif direction == 'SHORT' and confirmed_close >= sl:
            sl_hit = True

        if sl_hit:
            if is_stuck:
                print(f"  [{self._ex_tag}:{asset}] STUCK SL {pnl_pct:+.2f}% (can't close — no liquidity)")
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
        current_atr = atr_vals[-1] if atr_vals else price * 0.01
        position_age = time.time() - pos.get('entry_time', time.time())
        is_reversal = pos.get('is_reversal', False)

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

        # Define ratchet levels: (min_pnl_pct, protect_pct, atr_mult, label)
        # More granular than before — smooth progression
        if is_reversal:
            # Reversals: tighter, faster protection (they can snap back quickly)
            ratchet_levels = [
                (0.5,  0.0,  2.0, "BREAKEVEN"),   # +0.5% → move to entry (after 5min for reversals)
                (1.0,  0.15, 1.8, "LOCK-15%"),     # +1.0% → lock 15% of profit
                (1.5,  0.25, 1.5, "LOCK-25%"),     # +1.5% → lock 25%
                (2.0,  0.35, 1.3, "LOCK-35%"),     # +2.0% → lock 35%
                (3.0,  0.50, 1.0, "LOCK-50%"),     # +3.0% → lock 50%
                (5.0,  0.60, 0.8, "LOCK-60%"),     # +5.0% → lock 60%
                (10.0, 0.70, 0.6, "LOCK-70%"),     # +10%  → lock 70%
            ]
            min_age_for_breakeven = 300  # 5min for reversals (faster)
        else:
            # Trend-following: move to breakeven FAST, then progressive ratchet
            # Data shows: trades that survive past L2 are 64%+ winners
            # Key: protect capital early, then let runners run
            ratchet_levels = [
                (0.3,  0.0,  1.5, "BREAKEVEN"),    # +0.3% → move to entry FAST (was 0.8% — too slow)
                (0.6,  0.10, 1.4, "LOCK-10%"),     # +0.6% → lock 10% of profit
                (1.0,  0.20, 1.3, "LOCK-20%"),     # +1.0% → lock 20%
                (1.5,  0.30, 1.2, "LOCK-30%"),     # +1.5% → lock 30%
                (2.0,  0.40, 1.1, "LOCK-40%"),     # +2.0% → lock 40%
                (3.0,  0.50, 1.0, "LOCK-50%"),     # +3.0% → lock 50%
                (4.0,  0.55, 0.9, "LOCK-55%"),     # +4.0% → lock 55%
                (5.0,  0.60, 0.8, "LOCK-60%"),     # +5.0% → lock 60%
                (7.0,  0.65, 0.7, "LOCK-65%"),     # +7.0% → lock 65%
                (10.0, 0.65, 0.7, "LOCK-65%"),     # +10%  → lock 65%
                (15.0, 0.70, 0.6, "LOCK-70%"),     # +15%  → lock 70%
            ]
            min_age_for_breakeven = 480  # 8min for trend-following

        if direction == 'LONG':
            # Walk through ratchet levels from highest to lowest
            for min_pnl, protect, atr_m, label in reversed(ratchet_levels):
                if pnl_pct >= min_pnl:
                    if protect == 0.0:
                        # BREAKEVEN level — move SL to entry
                        if position_age >= min_age_for_breakeven and sl < entry:
                            new_sl = entry
                    else:
                        # PROFIT LOCK level — SL = entry + protect% of (peak - entry)
                        profit_range = peak - entry
                        floor_sl = entry + (profit_range * protect)

                        # ATR trail: peak - multiplier * ATR (gives room for pullbacks)
                        atr_trail_sl = peak - (current_atr * atr_m)

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
                if pnl_pct >= min_pnl:
                    if protect == 0.0:
                        # BREAKEVEN level — move SL to entry
                        if position_age >= min_age_for_breakeven and sl > entry:
                            new_sl = entry
                    else:
                        # PROFIT LOCK level — SL = entry - protect% of (entry - peak)
                        profit_range = entry - peak
                        floor_sl = entry - (profit_range * protect)

                        # ATR trail: peak + multiplier * ATR
                        atr_trail_sl = peak + (current_atr * atr_m)

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

        print(f"  [{self._ex_tag}:{asset}] CLOSED: P&L {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | {reason} | predicted={predicted_l} actual={actual_l_level} [{pred_hit}]")

        # Track realized PnL for drawdown limits
        self.session_realized_pnl += pnl_usd
        self.daily_realized_pnl += pnl_usd

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
            if streak >= 2:
                # Exponential cooldown: 2 losses=10min, 3=20min, 4=40min, 5+=60min
                cooldown_min = min(60, 10 * (2 ** (streak - 2)))
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
                },
            )
        except Exception as e:
            logger.warning(f"Journal log failed: {e}")

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
            logger.warning(f"Claude query failed: {e} -- falling back to Ollama mistral")
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
                            math_filter_warnings: list = None,
                            entry_score: int = 0,
                            score_reasons: list = None,
                            min_trend_bars: int = 0,
                            slope_pct: float = 0,
                            ema_separation: float = 0,
                            htf_1h_direction: str = "FLAT",
                            htf_4h_direction: str = "FLAT",
                            htf_alignment: int = 2) -> dict:
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

        prompt = f"""You are a PROFIT-FOCUSED trading brain for {asset}/USDT. Your job: ONLY enter trades that MAKE MONEY.

STRATEGY: EMA(8) crossover + trailing stop-loss (L1->Ln).
- CALL: price crosses ABOVE EMA(8) -> ride uptrend. PUT: crosses BELOW -> ride downtrend.
- Trailing SL locks profit: L1=breakeven, L2=lock 40%, L3=50%, L4=55%, L5+=60-70%.
- WE ONLY MAKE MONEY on trades reaching L4+ (strong sustained trend).
- L1-L2 exits = small loss or breakeven. Hard stop = big loss (-2%).
- Our win rate is LOW ({pnl_history}). Most trades die at L1. ONLY enter A+ setups.

TWO TRADE TYPES:
1. TREND-FOLLOWING: Enter WITH the trend after EMA crossover.
   - CALL in uptrend (price above rising EMA) or PUT in downtrend (price below falling EMA)
   - Need: steep slope >0.3%/bar + 3+ trend bars + volume confirms + early entry near EMA
2. REVERSAL (counter-trend): Enter AGAINST exhausted trend expecting mean reversion.
   - PUT in exhausted uptrend (price far above EMA + declining volume + rejection wicks + price turning)
   - CALL in exhausted downtrend (price far below EMA + selling dried up + wicks + price bouncing)
   - Need: 3%+ from EMA + declining volume + rejection wicks + price already turning

SKIP when: flat/weak slope + choppy candles + no clear trend or reversal + near S/R wall.
If entry score < 5/10 AND no reversal signals, the trade almost always dies at L1. SKIP IT.

Signal: {"CALL" if signal == "BUY" else "PUT"} | Equity: ${equity:,.0f}

MARKET DATA:
Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction}) | Gap: {ema_separation:.2f}%
ATR: ${current_atr:.2f} | Slope: {ema_slope_pct:+.3f}%/bar | Trend bars: {min_trend_bars}
Entry Score: {entry_score}/10 ({', '.join(score_reasons or []) or 'none'})
Support: ${support:.2f} | Resistance: ${resistance:.2f} | Volume: {vol_trend}
HTF Trend: 1h EMA(8)={htf_1h_direction} | 4h EMA(8)={htf_4h_direction} | Alignment: {htf_alignment}/3 (3=all agree)

Warnings: {'; '.join(math_filter_warnings or []) or 'None'}

CANDLES (5m, newest last):
{candle_block}

QUANT AGENTS:
{agent_data}

{historical_patterns}

DECISION CHECKLIST — answer each honestly:
1. TREND STRENGTH: Slope={ema_slope_pct:+.3f}%, {min_trend_bars} bars. Is this strong enough for L4+? (Need slope>0.3% AND 3+ bars)
2. CANDLE QUALITY: Are candles clean (big bodies, small wicks, same direction)? Or messy (dojis, wicks, mixed)?
3. VOLUME: {vol_trend}. Rising volume = trend continues. Flat/declining = dying trend, SKIP.
4. ENTRY TIMING: {ema_separation:.1f}% from EMA. Under 3% = early/good. Over 5% = late, likely L1-L2 max.
5. ROOM TO RUN: Distance to {"resistance $" + f"{resistance:.2f}" if signal == "BUY" else "support $" + f"{support:.2f}"}. Need 2%+ room for L4.
6. PATTERN MATCH: Check historical data above. What L-level did similar setups reach? If mostly L1-L2, SKIP.
7. RISK: Score 0-10. What specifically could go wrong?
8. FINAL CALL: Does this trade have a realistic path to L4+ profit? Be brutally honest.

RESPOND WITH ONLY JSON (fill in YOUR analysis of THIS specific setup):
{{"proceed": <true or false>, "confidence": <0.0 to 1.0>, "position_size_pct": <1 to {max_position_pct}>, "risk_score": <0 to 10>, "trade_quality": <0 to 10>, "predicted_l_level": "<L1/L2/L3/L4/L6+/L10+>", "bull_case": "<why this {asset} trade reaches L4+>", "bear_case": "<what kills this trade — be specific>", "facilitator_verdict": "<your honest final decision>"}}

CRITICAL: confidence = probability of reaching L3+. If you predict L1-L2, set proceed=false. We lose money on L1-L2 exits."""

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

LAST {n_candles} CANDLES (5m):
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
