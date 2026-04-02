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

        # Risk settings
        risk = config.get('risk', {})
        self.daily_loss_limit_pct: float = risk.get('daily_loss_limit_pct', 3.0)

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
    def _tf_snapshot(
        self, closes: list, highs: list, lows: list
    ) -> Optional[Dict[str, Any]]:
        """Per-timeframe trend stats aligned with signal TF semantics (confirmed bars)."""
        need = max(self.ema_period + 5, 20)
        if len(closes) < need:
            return None
        ema_vals = ema(closes, self.ema_period)
        if len(ema_vals) < 5:
            return None
        cur_e = ema_vals[-2]
        prev_e = ema_vals[-3]
        direction = "RISING" if cur_e > prev_e else "FALLING"
        base = ema_vals[-5] if abs(ema_vals[-5]) > 1e-12 else cur_e
        slope_pct = ((ema_vals[-2] - base) / base * 100) if base else 0.0
        sig_close = closes[-2]
        ema_term = ema_vals[-1]
        sep_pct = (
            abs(sig_close - ema_term) / ema_term * 100 if ema_term > 0 else 0.0
        )
        c1, c2, c3 = closes[-2], closes[-3], closes[-4]
        if c1 > c2 and c1 > c3:
            mom = "up"
        elif c1 < c2 and c1 < c3:
            mom = "down"
        else:
            mom = "flat"
        return {
            "direction": direction,
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
                lines.append(
                    f"  {tf}: EMA{self.ema_period} {s['direction']} "
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

        # Compute EMA(8) and ATR(14) on signal-timeframe candles (incl. forming bar for freshness)
        ema_vals = ema(closes, self.ema_period)
        atr_vals = atr(highs, lows, closes, 14)

        # Use confirmed candle EMA for signal direction
        current_ema = ema_vals[-2]  # EMA at last confirmed candle
        prev_ema = ema_vals[-3] if len(ema_vals) >= 3 else current_ema
        current_atr = atr_vals[-1] if atr_vals else 0
        ema_direction = "RISING" if current_ema > prev_ema else "FALLING"

        mtf_block, mtf_aligned, mtf_n, mtf_snaps = self._collect_mtf_analysis(
            symbol, closes, highs, lows, ema_direction
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
            micro_ema = "R" if ctx_snap['direction'] == "RISING" else "F"
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

        # ── Signal from signal_timeframe EMA crossover (CONFIRMED candles only) ──
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

        # OVEREXTENSION CHECK — if price is >10% from EMA, it's parabolic
        # Don't enter LONG at the top of a parabolic move or SHORT at the bottom
        ema_separation = abs(price - current_ema) / current_ema * 100 if current_ema > 0 else 0
        if ema_separation > 10.0:
            signal = "NEUTRAL"
            print(f"  [{self._ex_tag}:{asset}] OVEREXTENDED: {ema_separation:.1f}% from EMA — too far, waiting for pullback")
        elif ema_direction == "RISING" and price > current_ema and ema_crossed:
            if price_falling:
                signal = "NEUTRAL"  # EMA says buy but price is reversing down
            else:
                signal = "BUY"
        elif ema_direction == "FALLING" and price < current_ema and ema_crossed:
            if price_rising:
                signal = "NEUTRAL"  # EMA says sell but price is reversing up
            else:
                signal = "SELL"
        # Strong trend: price >1% from EMA — but still check momentum
        elif ema_direction == "FALLING" and price < current_ema * 0.99:
            if not price_rising:
                signal = "SELL"
        elif ema_direction == "RISING" and price > current_ema * 1.01:
            if not price_falling:
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
        print(
            f"  [{self._ex_tag}:{asset}] ${tick_price:,.2f} (sig=${price:,.2f}) | "
            f"EMA({self.signal_timeframe}): ${current_ema:.2f} {ema_direction} | "
            f"Signal: {signal} | ATR: ${current_atr:.2f} | {ob_info}{ctx_suffix}"
        )
        if mtf_n > 0:
            compact = " ".join(
                f"{tf}:{('R' if mtf_snaps[tf]['direction'] == 'RISING' else 'F')}"
                for tf in self.analysis_timeframes
                if tf in mtf_snaps
            )
            print(
                f"  [{self._ex_tag}:{asset}] MTF EMA{self.ema_period} "
                f"align {mtf_aligned}/{mtf_n} vs {ema_direction} | {compact}"
            )

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

            # 3. PRICE vs EMA SEPARATION (0-2 points) — spec §4.1 / §2.4: price = closes[-2], EMA term = ema_vals[-1]
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
        elif min_trend_bars < min_trend_need:
            print(f"  [{self._ex_tag}:{asset}] TOO EARLY: only {min_trend_bars} trend bars -- need {min_trend_need}+")
            return

        # ── HARD GATE: EMA slope must agree with signal direction ──
        # Prevents LLM from confirming LONG when EMA is falling (or SHORT when rising)
        if len(ema_vals) >= 4:
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
                                # Max position: 5% of equity (or 20% for small accounts)
                                max_pct = 20 if self.equity < 500 else 5
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
                                print(f"  [{self._ex_tag}:{asset}] SYNCED {pos_side} position ({contracts}) ${pos_notional:,.0f} | SL=${sync_sl:,.2f}")
                            # Don't return — let _manage_position handle it
                            return
        except Exception as e:
            logger.debug(f"Exchange position check failed: {e}")

        # CANDLE DEDUP: only one entry attempt per confirmed signal-timeframe candle
        timestamps = ohlcv.get('timestamps', [])
        if len(timestamps) >= 2:
            confirmed_candle_ts = timestamps[-2]  # Last confirmed candle
            if asset in self.last_signal_candle and self.last_signal_candle[asset] == confirmed_candle_ts:
                print(
                    f"  [{self._ex_tag}:{asset}] DEDUP: waiting for next "
                    f"{self.signal_timeframe} candle"
                )
                return
            self.last_signal_candle[asset] = confirmed_candle_ts

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
            return

        # Build P&L history string for context
        edge = self.edge_stats.get(asset, {})
        pnl_history = f"{edge.get('wins',0)}W/{edge.get('losses',0)}L rate={edge.get('win_rate',0.5):.0%}"

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

        # spec §6.2: elevated risk → halve size (veto already handled via proceed=false)
        if self.bear_reduce_threshold <= risk_score < self.bear_veto_threshold:
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
                'breakeven_moved': False,
                'agent_votes': orch_result.get('agent_votes', {}) if orch_result else {},
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

        # ── 2. HARD STOP: max -2% loss — non-negotiable ──
        # But if asset is blacklisted (stuck, can't close), just log and skip
        is_stuck = asset in self.failed_close_assets
        if pnl_pct <= -2.0:
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
        duration_min = (time.time() - pos.get('entry_time', time.time())) / 60.0

        print(f"  [{self._ex_tag}:{asset}] CLOSED: P&L {pnl_pct:+.2f}% (${pnl_usd:+,.2f}) | {reason}")

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

        # Update edge stats
        if asset in self.edge_stats:
            self.edge_stats[asset]['total'] += 1
            if pnl_pct > 0:
                self.edge_stats[asset]['wins'] += 1
            else:
                self.edge_stats[asset]['losses'] += 1
            s = self.edge_stats[asset]
            s['win_rate'] = s['wins'] / s['total'] if s['total'] > 0 else 0.5

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
                            mtf_n: int = 0) -> dict:
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
        max_position_pct = 20 if equity < 500 else 5

        # Historical L-level patterns from journal (teaches LLM what works)
        historical_patterns = self._build_historical_pattern_context(asset)

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
Price: ${price:,.2f} | EMA(8): ${current_ema:.2f} ({ema_direction}) | EMA-price gap: {separation_pct:.2f}%
ATR: ${current_atr:.2f} | Slope: {ema_slope_pct:+.3f}%/bar | Trend bars: {consecutive_trend}
Support: ${support:.2f} | Resistance: ${resistance:.2f} | Volume: {vol_trend}

=== MULTI-TIMEFRAME STACK (EMA{self.ema_period} on each TF; last closed bar; refreshed every engine poll) ===
Signal timeframe is {self.signal_timeframe} — that is the bar size for ENTRY rules above.
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
- predicted_l_level: Your best estimate of where trailing SL exits (L1, L2, L4, L10+, etc.)
- risk_score: 0=safe, 5=moderate — veto entry if >={self.bear_veto_threshold}; size reduction zone if >={self.bear_reduce_threshold}
- trade_quality: 0=terrible, 5=marginal, 8+=excellent (measures L-level potential)
  quality 8+: Strong trend indicators → likely L4+ → ENTER with full size
  quality 5-7: Moderate setup → might reach L3-L4 → enter with reduced size
  quality ≤4: Weak/choppy → likely L1-L2 → SKIP (save capital for better setup)
- confidence: 0.0-1.0 (your HONEST estimate based on historical patterns above)
- If agents conflict OR risk>={self.bear_veto_threshold} OR quality<=3: set proceed=false
- position_size_pct: 1-{max_position_pct}% (scale with trade_quality and predicted L-level)
- REMEMBER: It's BETTER to skip a marginal trade than to lose at L1. We profit from PATIENCE.

Respond with ONLY this JSON (no other text):
{{"proceed": true, "confidence": 0.5, "position_size_pct": 3, "risk_score": 5, "trade_quality": 5, "predicted_l_level": "L4", "bull_case": "why this reaches L4+", "bear_case": "why this dies at L1-L2", "facilitator_verdict": "final L-level prediction and reasoning", "agent_conflicts": "which agents disagree"}}"""

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
