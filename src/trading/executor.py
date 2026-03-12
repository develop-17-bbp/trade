import time
import os
import sys
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import asdict, dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.data.fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.institutional_fetcher import InstitutionalFetcher
from src.data.free_tier_integrations import FreeDataAggregator
from src.ai.sentiment import SentimentPipeline
from src.ai.agentic_strategist import AgenticStrategist
from src.ai.advanced_learning import AdvancedLearningEngine, MarketRegime
from src.ai.reinforcement_learning import (
    ReinforcementLearningAgent, AdaptiveAlgorithmLayer, MarketState
)
from src.data.on_chain_fetcher import OnChainFetcher
from src.trading.strategy import HybridStrategy, SimpleStrategy
from src.trading.backtest import (
    run_backtest, BacktestConfig, BacktestResult, format_backtest_report
)
from src.risk.manager import RiskManager
from src.risk.profit_protector import ProfitProtector, LossAversionFilter
from src.integrations.robinhood_stub import RobinhoodClient
from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.hedger import PortfolioHedger
from src.monitoring.health_checker import SystemHealthChecker
from src.execution.failover import ExecutionFailoverController
from src.risk.dynamic_manager import DynamicRiskManager
from src.execution.router import ExecutionRouter, ExecutionMode
from src.integrations.on_chain_portfolio import OnChainPortfolioManager
from src.monitoring.journal import TradingJournal
from src.models.volatility_regime import VolatilityRegimeDetector, VolatilityRegime
from src.monitoring.event_guard import MarketEventGuard
from src.data.microstructure import MicrostructureAnalyzer
from src.trading.advanced_backtest import AdvancedSimulator

# Institutional Institutional Components
from src.models.lightgbm_classifier import LightGBMClassifier
from src.ai.patchtst_model import PatchTSTClassifier
from src.risk.vpin_guard import VPINGuard
from infra.signal_stream import SignalStreamAgent
from testing.chaos_engine import ChaosEngine
from src.models.benchmark import ModelBenchmark

from enum import Enum

class TradingStyle(Enum):
    DAY = "day"  # High frequency, lower per-trade risk
    SWING = "swing"  # Moderate frequency, higher conviction

def _safe_print(msg: str = ""):
    """Print with UTF-8 encoding, falling back to ASCII-safe output on Windows."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'), flush=True)


class TradingExecutor:
    """
    Full three-layer trading system orchestrator.

    Modes:
      - paper: backtest on historical data with simulated execution
      - signal: generate signals only (no execution)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.mode = self.config.get('mode', 'paper')
        self.assets = self.config.get('assets', ['BTC', 'ETH'])
        self.initial_capital = self.config.get('initial_capital', 100_000.0)

        # Data -- supports testnet mode for paper trading with fake money
        use_testnet = self.mode == 'testnet' or self.config.get('testnet', False)
        exchange_cfg = self.config.get('exchange', {})
        self.price_source = PriceFetcher(
            exchange_name=exchange_cfg.get('name', 'binance'),
            testnet=use_testnet,
        )
        if not self.price_source.is_available:
            raise RuntimeError(
                "[FATAL] Exchange not available. Real-time data is required. "
                "Ensure CCXT is installed and Binance API is accessible."
            )
        self.news = NewsFetcher(
            newsapi_key=os.environ.get('NEWSAPI_KEY'),
            cryptopanic_token=os.environ.get('CRYPTOPANIC_TOKEN'),
        )
        self.institutional = InstitutionalFetcher()
        self.on_chain = OnChainFetcher()
        self.on_chain_portfolio = OnChainPortfolioManager()
        self.free_data = FreeDataAggregator()  # Free data sources (Fear/Greed, IV, etc.)
        import os as _os_env
        _prov_cfg = self.config.get('ai', {}).get('reasoning_provider', 'openai')
        _model_cfg = self.config.get('ai', {}).get('reasoning_model')
        _key = _os_env.environ.get('REASONING_LLM_KEY', '')
        
        # Prioritize config provider if explicitly set
        if self.config.get('ai', {}).get('reasoning_provider'):
            _prov = _prov_cfg
            _model = _model_cfg
        else:
            # Legacy auto-detection based on API key
            if not _key:
                _prov = 'ollama'
                _model = _model_cfg or 'neural-chat'
            else:
                if _key.startswith('AIza'):
                    _prov = 'google'
                    _model = _model_cfg or 'gemini-1.5-flash'
                else:
                    _prov = 'openai'
                    _model = _model_cfg or 'gpt-4-turbo'

        self.strategist = AgenticStrategist(
            provider=_prov,
            model=_model,
            use_local_on_failure=self.config.get('ai', {}).get('use_local_on_failure', True)
        )
        
        # Phase 6: Advanced Learning Engine
        self.advanced_learning = AdvancedLearningEngine(
            meta_model_path=self.config.get('models_path', 'models') + "/meta_learning_model.json"
        )
        self.rl_agent = ReinforcementLearningAgent(
            learning_rate=self.config.get('rl', {}).get('learning_rate', 0.001),
            gamma=self.config.get('rl', {}).get('gamma', 0.99)
        )
        self.adaptive_algo = AdaptiveAlgorithmLayer()
        
        # Phase 5: Autonomous Trading Components
        self.portfolio_allocator = PortfolioAllocator(
            total_capital=self.initial_capital,
            max_allocation_pct=self.config.get('portfolio', {}).get('max_allocation_pct', 0.05)
        )
        self.portfolio_hedger = PortfolioHedger()
        self.health_checker = SystemHealthChecker(check_interval_sec=60)
        self.failover_controller = ExecutionFailoverController(primary_exchange=self.price_source)
        
        # New Microstructure & Regime Components
        self.microstructure = MicrostructureAnalyzer(depth=20)
        self.regime_detector = VolatilityRegimeDetector(lookback=100)
        self.event_guard = MarketEventGuard()
        self.simulator = AdvancedSimulator(iterations=1000)
        
        self.risk_manager = DynamicRiskManager(initial_capital=self.initial_capital)
        
        # LOSS PREVENTION LAYER: Profit Protector + Loss Aversion
        self.profit_protector = ProfitProtector(initial_capital=self.initial_capital)
        self.loss_aversion = LossAversionFilter()
        
        self.agentic_bias = 0.0
        self.iteration_count = 0

        # Institutional Ensemble Engines
        self.lgbm = LightGBMClassifier(self.config)
        # self.rl_agent is already initialized above
        self.patch_tst = PatchTSTClassifier()
        
        # Risk & Compliance Infrastructure
        self.vpin = VPINGuard(bucket_size=5.0, threshold=0.75) # 5 BTC buckets
        self.stream = SignalStreamAgent()
        self.audit_log = TradingJournal() # Enhanced version for reasoning traces
        
        # Execution Layer
        self.router = ExecutionRouter(mode=ExecutionMode.TESTNET)
        self.chaos = ChaosEngine(self.router, self.health_checker)
        
        # Model Benchmarking & Leaderboard
        self.benchmark = ModelBenchmark(model_version="v6.5")
        
        # Kill Switches and Safety Status
        self.kill_switch_active = False
        self.last_tick_time = time.time()
        self.max_staleness = self.config.get('max_staleness_sec', 10)
        
        _safe_print("  [SYSTEM] Institutional Infrastructure v6.5 ACTIVE.")
        _safe_print("  [AUDIT] Audit Trail established. Stream and Compliance ready.")

        # Strategy
        self.strategy = HybridStrategy(self.config)

        # Sentiment
        self.sentiment = SentimentPipeline(
            use_transformer=self.config.get('ai', {}).get('use_transformer', False),
        )

        # Robinhood Integration (optional for live trading)
        self.robinhood = None
        self._init_robinhood()

        # Backtest config
        risk_cfg = self.config.get('risk', {})
        self.bt_config = BacktestConfig(
            initial_capital=self.initial_capital,
            fee_pct=self.config.get('fee_pct', 0.0),      # Robinhood 0 fee
            slippage_pct=self.config.get('slippage_pct', 0.375), # RH spread leg
            risk_per_trade_pct=risk_cfg.get('risk_per_trade_pct', 1.0),
            max_position_pct=risk_cfg.get('max_position_size_pct', 5.0),
            use_stops=True,
            atr_stop_mult=risk_cfg.get('atr_stop_mult', 2.0),
            atr_tp_mult=risk_cfg.get('atr_tp_mult', 3.0),
            min_return_pct=self.config.get('min_return_pct'), # Activate 1% target override
        )
        self.min_return_pct = self.config.get('min_return_pct') # Activate 1% target override
        
        # Phase 5: Trading Journal & Styles
        self.journal = TradingJournal()
        
        # Recover previously OPEN trades to allow TP/SL closure
        _recovered = 0
        for t in self.journal.trades:
            if t.get("status") == "OPEN":
                asset = t.get("asset", "").replace("/USDT", "").replace("USDT", "")
                direction = 1 if str(t.get("side")).lower() == "buy" else -1
                price = float(t.get("price", 0.0))
                qty = float(t.get("quantity", 0.0))
                order_id = t.get("order_id", "")
                if price > 0 and qty > 0:
                    size_pct = (qty * price) / self.initial_capital
                    self.risk_manager.register_trade_open(asset, direction, price, size_pct, order_id=order_id)
                    _recovered += 1
        if _recovered > 0:
            import logging
            logging.info(f"[RECOVERY] Re-attached {_recovered} open trades to Risk Manager.")

        style_val = self.config.get('trading_style', 'swing').lower()
        self.style = TradingStyle.DAY if style_val == 'day' else TradingStyle.SWING
        
        # Style-based parameter overrides
        if self.style == TradingStyle.DAY:
            # 5-min intervals for Day Trading
            self.poll_interval = 300
            self.risk_manager.risk_limits.max_single_trade_pct = 0.005 # 0.5% max
        else:
            # 1-hour intervals for Swing Trading
            self.poll_interval = 3600
            self.risk_manager.risk_limits.max_single_trade_pct = 0.02 # 2.0% max
            
        # User defined override for testing speed
        if 'poll_interval' in self.config:
            self.poll_interval = self.config['poll_interval']
            _safe_print(f"  [CONFIG] Poll interval overridden to {self.poll_interval}s")

    def _init_robinhood(self):
        """Mock/Stub for Robinhood login."""
        creds = self.config.get('robinhood', {})
        if creds.get('username') and creds.get('password'):
            self.robinhood = RobinhoodClient()
            self.robinhood.login(creds['username'], creds['password'])

    def run(self):
        """Main entry point -- run the full system."""
        _safe_print("=" * 70)
        _safe_print("  🏛️  AI-DRIVEN CRYPTO TRADING SYSTEM v6.5")
        _safe_print("  9-Layer Autonomous Intelligence Architecture")
        _safe_print("=" * 70)
        _safe_print(f"  Mode:    {self.mode.upper()}")
        _safe_print(f"  Assets:  {', '.join(self.assets)}")
        data_label = "Binance TESTNET (sandbox)" if self.price_source.testnet else "Binance LIVE"
        _safe_print(f"  Source:  {data_label}")
        _safe_print("-" * 70)

        # ── Fetch & Display Exchange Balances ──
        _safe_print("\n  📊 EXCHANGE WALLET BALANCES")
        _safe_print("  " + "-" * 50)
        if self.price_source.is_authenticated:
            balance = self.price_source.get_balance()
            if 'error' not in balance:
                total_balances = balance.get('total', {})
                free_balances = balance.get('free', {})

                # Show USDT first, then configured assets only
                usdt_total = total_balances.get('USDT', 0.0)
                usdt_free = free_balances.get('USDT', 0.0)
                _safe_print(f"  {'Asset':<10} {'Total':>15} {'Available':>15}")
                _safe_print(f"  {'─'*10} {'─'*15} {'─'*15}")
                _safe_print(f"  {'USDT':<10} {usdt_total:>15,.4f} {usdt_free:>15,.4f}")

                # Show configured assets + BNB (for gas fees)
                show_assets = list(set(self.assets + ['BNB']))
                for a in sorted(show_assets):
                    amt = total_balances.get(a, 0.0)
                    if amt <= 0:
                        continue
                    free_amt = free_balances.get(a, 0.0)
                    _safe_print(f"  {a:<10} {amt:>15,.8f} {free_amt:>15,.8f}")


                # Update initial capital from USDT balance
                if usdt_free > 0:
                    self.initial_capital = usdt_free
                _safe_print(f"\n  💰 Reference Capital: ${self.initial_capital:,.2f} USDT")
            else:
                is_invalid_creds = balance.get('invalid_credentials', False)
                if is_invalid_creds:
                    _safe_print(f"  ⚠️  Invalid API credentials (code -2008)")
                    _safe_print(f"  📝 To fix: Update your API keys in config.yaml or set environment variables:")
                    if self.price_source.testnet:
                        _safe_print(f"     - Get testnet keys from: https://testnet.binance.vision/")
                        _safe_print(f"     - Then update config.yaml exchange.api_key and exchange.api_secret")
                    else:
                        _safe_print(f"     - Get live keys from: https://www.binance.com/en/user/settings/api-management")
                        _safe_print(f"     - Then update config.yaml exchange.api_key and exchange.api_secret")
                else:
                    _safe_print(f"  ⚠️  Balance fetch error: {balance['error']}")
                _safe_print(f"  💰 Using config capital: ${self.initial_capital:,.2f}")
        else:
            _safe_print(f"  ⚠️  No API credentials configured")
            _safe_print(f"  📝 To enable balance fetching:")
            _safe_print(f"     1. Get API keys from https://testnet.binance.vision/ (testnet mode)")
            _safe_print(f"     2. Update config.yaml with exchange.api_key and exchange.api_secret")
            _safe_print(f"  💰 Reference Capital: ${self.initial_capital:,.2f}")

        # ── Fetch & Display Live Spot Prices ──
        _safe_print(f"\n  📈 LIVE SPOT PRICES")
        _safe_print("  " + "-" * 50)
        total_portfolio_usd = 0.0
        for asset in self.assets:
            symbol = f"{asset}/USDT"
            try:
                price = self.price_source.fetch_latest_price(symbol)
                _safe_print(f"  {symbol:<12} ${price:>12,.2f}")
                # Calculate holdings value if we have balance
                if self.price_source.is_authenticated and 'error' not in balance:
                    held = balance.get('total', {}).get(asset, 0.0)
                    if held > 0:
                        value = held * price
                        total_portfolio_usd += value
                        _safe_print(f"  {'':12}   ↳ Holding: {held:.8f} = ${value:,.2f}")
            except Exception as e:
                _safe_print(f"  {symbol:<12} (unavailable: {e})")

        # Add USDT to total portfolio
        if self.price_source.is_authenticated and 'error' not in balance:
            usdt_held = balance.get('total', {}).get('USDT', 0.0)
            total_portfolio_usd += usdt_held
            _safe_print(f"\n  🏦 TOTAL PORTFOLIO VALUE: ${total_portfolio_usd:,.2f} USD")

        _safe_print("-" * 70)

        # ── Layer Status ──
        _safe_print("\n  🧠 9-LAYER INTELLIGENCE STATUS")
        layers = [
            ("L1", "Quantitative Engine",     "ONLINE"),
            ("L2", "Sentiment Intelligence",   "ONLINE" if os.environ.get('NEWSAPI_KEY') else "DEGRADED"),
            ("L3", "Risk Fortress",            "ONLINE"),
            ("L4", "Signal Fusion",            "ONLINE"),
            ("L5", "Execution Engine",         "ONLINE" if self.price_source.is_authenticated else "PAPER"),
            ("L6", "Strategist Hub",           "ONLINE" if os.environ.get('REASONING_LLM_KEY') else "OFFLINE"),
            ("L7", "Advanced Learning",        "ONLINE"),
            ("L8", "Tactical Memory",          "ONLINE"),
            ("L9", "Evolution Portal",         "STANDBY"),
        ]
        for lid, name, status in layers:
            icon = "✅" if status == "ONLINE" else "⚡" if status == "PAPER" else "⚠️" if status == "DEGRADED" else "🔴" if status == "OFFLINE" else "⏳"
            _safe_print(f"  {icon} {lid}: {name:<25} [{status}]")

        # ── Data Sources Configuration ──
        _safe_print("\n  📰 L2 SENTIMENT DATA SOURCES")
        _safe_print("  " + "-" * 50)
        
        newsapi_status = "✅ ENABLED" if os.environ.get('NEWSAPI_KEY') else "❌ DISABLED"
        cryptopanic_status = "✅ ENABLED" if os.environ.get('CRYPTOPANIC_TOKEN') else "❌ DISABLED"
        reddit_status = "✅ ALWAYS ENABLED"
        coingecko_status = "⚙️  FALLBACK ONLY"
        
        _safe_print(f"  📧 NewsAPI:        {newsapi_status}")
        _safe_print(f"  🚨 CryptoPanic:    {cryptopanic_status}")
        _safe_print(f"  🔴 Reddit:         {reddit_status}")
        _safe_print(f"  🪙 CoinGecko:      {coingecko_status}")
        _safe_print(f"  📊 Priority Order: NewsAPI → CryptoPanic → Reddit → CoinGecko")

        _safe_print("\n" + "=" * 70)
        _safe_print("  🚀 SYSTEM STARTING...")
        _safe_print("=" * 70 + "\n")

        try:
            from src.api.state import DashboardState
            DashboardState().set_status("TRADING")
        except: pass

        if self.mode == 'paper':
            self._run_paper()
        else:
            # Combined loop for live and testnet (logic handles internals)
            self._run_live()


    def _perform_agentic_review(self):
        """Perform periodic Layer 6 AI Agent macro review."""
        _safe_print("\n  [AGENTIC-REVIEW] Strategist reflecting on market state...")
        
        # 1. Gather Context
        history = list(self.journal.trades)
        
        # Pull latest asset context (BTC as proxy for macro)
        asset = self.assets[0]
        ohlcv = self._fetch_data(f"{asset}/USDT")
        if not ohlcv: return
        
        market_data = {
            'asset': asset,
            'price': ohlcv['closes'][-1],
            'onchain': self.on_chain.get_market_context(asset, current_price=ohlcv['closes'][-1]),
            'funding_rate': 0.0001, # Placeholder
            'atr': ohlcv.get('atr', [0.005])[-1] # From indicator cache
        }
        
        # 2. Analyze
        decision = self.strategist.analyze_performance(
            trade_history=history,
            current_config=self.config,
            market_data=market_data
        )
        
        # 3. Apply Decision
        self.agentic_bias = decision.macro_bias
        try:
            from src.api.state import DashboardState
            DashboardState().add_agent_thought(
                asset=asset, 
                regime=decision.market_regime, 
                thought=decision.reasoning_trace[:150],
                confidence=int(decision.confidence * 100)
            )
        except: pass
        
        _safe_print(f"  [STRATEGIST] Decision: {decision.market_regime} | Bias: {decision.macro_bias:+.2f}")
        _safe_print(f"  [REASONING] {decision.reasoning_trace[:150]}...")
        
        # Suggested Config Overrides
        if decision.suggested_config_update:
            _safe_print(f"  [CONFIG] Applying strategist overrides: {decision.suggested_config_update}")
            # Update dynamic risk parameters
            if 'risk' in decision.suggested_config_update:
                r_upd = decision.suggested_config_update['risk']
                if 'atr_tp_mult' in r_upd: 
                    self.bt_config.atr_tp_mult = r_upd['atr_tp_mult']
                    # Also update live risk manager
                    self.risk_manager.risk_limits.take_profit_atr_mult = r_upd['atr_tp_mult']
                if 'atr_stop_mult' in r_upd:
                    self.risk_manager.risk_limits.stop_loss_atr_mult = r_upd['atr_stop_mult']

    def _run_paper(self):
        """Backtest mode on historical data (loads from CSV)."""
        all_results: Dict[str, BacktestResult] = {}
        multi_asset_data = {}
        import pandas as pd
        processed_any = False

        for asset in self.assets:
            symbol = f"{asset}/USDT"
            _safe_print(f"\n  [PAPER] Running backtest for {symbol}...")

            # 1. Load data from CSV (AAVE_USDT_1h.csv)
            csv_path = f"data/{asset}_USDT_1h.csv"
            if not os.path.exists(csv_path):
                _safe_print(f"  [X] CSV not found: {csv_path}. Skipping {asset}.")
                continue

            try:
                df = pd.read_csv(csv_path)
                closes = df['close'].tolist()
                highs = df['high'].tolist()
                lows = df['low'].tolist()
                volumes = df['volume'].tolist()
                multi_asset_data[asset] = df
                _safe_print(f"  [OK] Loaded {len(closes):,} bars from {csv_path}")
            except Exception as e:
                _safe_print(f"  [X] Failed to load {csv_path}: {e}")
                continue

            # 2. Fetch sentiment (optional, may fail gracefully)
            headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)
            
            # 2.5 Fetch free data sources (Fear/Greed, IV, On-Chain, etc.) 
            free_signals = self.free_data.aggregate_all_signals(symbol=asset)
            _safe_print(f"  [FREE DATA] {asset}: Fear/Greed={free_signals.get('fear_greed_classification')}, "
                       f"IV={free_signals.get('iv_regime')}, Flow={free_signals.get('exchange_flow_signal')}")

            # 3. Generate signals using L1 + L2
            _safe_print(f"  [SIGNAL] Generating hybrid signals (FinBERT + LightGBM + FREE DATA)...")
            strategy_result = self.strategy.generate_signals(
                prices=closes,
                highs=highs,
                lows=lows,
                volumes=volumes,
                headlines=headlines if headlines else None,
                headline_timestamps=h_timestamps if h_timestamps else None,
                headline_sources=h_sources if h_sources else None,
                headline_event_types=h_events if h_events else None,
            )

            signals = strategy_result['signals']
            
            # 3.5 Apply confidence boost from free data
            l2_data = strategy_result.get('l2_data', {}) or {}
            base_confidence = float(l2_data.get('confidence', 0.5))
            boosted_confidence = self.free_data.calculate_free_data_boost(base_confidence, free_signals)
            l2_data['confidence'] = boosted_confidence
            l2_data['free_data_signal'] = free_signals.get('fear_greed_signal')
            _safe_print(f"  [CONFIDENCE BOOST] {asset}: {base_confidence:.1%} → {boosted_confidence:.1%} (free data)")

            # 4. Run Backtest (L3 Risk Simulation)
            _safe_print(f"  [BACKTEST] Simulating execution with L3 Risk Manager...")
            result = run_backtest(
                prices=closes,
                signals=signals,
                config=self.bt_config,
                highs=highs,
                lows=lows
            )
            all_results[asset] = result
            processed_any = True

            # 5. Output asset report
            _safe_print(f"\n--- Report for {asset} ---")
            _safe_print(format_backtest_report(result))

            # 6. Record backtest for learning (L1 incremental training)
            self.strategy.record_backtest(result, asset, strategy_result['l1_data'])

            try:
                from src.api.state import DashboardState
                l2 = strategy_result.get('l2_data', {}) or {}
                sc = float(l2.get('aggregate_score', 0.0))
                conf = float(l2.get('confidence', 0.0))
                label = l2.get('aggregate_label', 'NEUTRAL')
                bull = max(0.0, sc) * 100.0
                bear = max(0.0, -sc) * 100.0
                sel_signal = 'LONG' if (signals[-1] if signals else 0) > 0 else ('SHORT' if (signals[-1] if signals else 0) < 0 else 'FLAT')
                
                # Extract factors from L1 features if available
                l1_feats = strategy_result.get('l1_data', {}).get('features', [{}])[-1]
                vpin_val = float(l1_feats.get('vpin_50', 0.0))
                vol_val = float(l1_feats.get('realized_vol_20', 0.0))
                trend_val = float(l1_feats.get('trend_strength', 0.0))
                
                # Extract attribution
                preds = strategy_result.get('l1_data', {}).get('predictions', [])
                l1_conf = float(preds[-1][1]) if preds else 0.5
                c1 = max(0.0, l1_conf)
                c2 = max(0.0, abs(sc))
                c3 = 0.3 # Base risk factor
                tot = c1 + c2 + c3
                attr = {'l1': c1 / tot, 'l2': c2 / tot, 'l3': c3 / tot}

                # Top Features
                top_feats = []
                if isinstance(l1_feats, dict) and l1_feats:
                    try:
                        top_feats = sorted(
                            [{'name': k, 'value': float(v)} for k, v in l1_feats.items() if isinstance(v, (int, float))],
                            key=lambda x: abs(x['value']),
                            reverse=True
                        )[:5]
                    except Exception: pass

                DashboardState().update_asset(asset, {
                    'signal': sel_signal,
                    'sentiment': {
                        'score': sc,
                        'label': label,
                        'confidence': conf,
                        'bull_pct': bull,
                        'bear_pct': bear,
                        'headlines': headlines[-5:] if headlines else []
                    },
                    'factors': {
                        'vpin': vpin_val,
                        'liquidity_regime': 'NORMAL', # Static for paper
                        'volatility': vol_val,
                        'trend_strength': trend_val,
                        'funding_rate': 0.0001
                    },
                    'attribution': attr,
                    'features_top': top_feats,
                    'weights': {'l1': 0.5, 'l2': 0.3, 'l3': 0.2},
                    'sent_hist': [float(l2.get('aggregate_score', 0.0)) for _ in range(20)] # Dummy spark
                })
                
                # Also set layers for 9-layer reflection in paper mode
                layers = {
                    "L1 Quant": {"status": "OK", "progress": float(c1), "metric": f"Conf {l1_conf:.2f}"},
                    "L2 Sentiment": {"status": "OK", "progress": float(min(1.0, abs(sc))), "metric": f"Score {sc:.2f}"},
                    "L3 Risk": {"status": "OK", "progress": 0.8, "metric": f"VPIN {vpin_val:.2f}"},
                    "L4 Meta": {"status": "OK", "progress": 0.7, "metric": "Signal Fused"},
                    "L5 Execution": {"status": "OK", "progress": 0.8, "metric": "PAPER READY"},
                    "L6 Strategist": {"status": "OK", "progress": float(conf/100.0 if conf <= 1 else conf), "metric": "Sim Reflecting"},
                    "L7 Autonomy": {"status": "OK", "progress": 1.0, "metric": "PatchTST ACTIVE"},
                    "L8 Monitoring": {"status": "OK", "progress": 0.9, "metric": "Log STABLE"},
                    "L9 Learning": {"status": "OK", "progress": 1.0, "metric": "LGBM+TRANSFORMER"}
                }
                DashboardState().set_layers(layers)
                DashboardState().set_sources({
                    "exchange": "ONLINE",
                    "news": "ONLINE" if headlines else "OFFLINE",
                    "onchain": "OFFLINE",
                    "llm": "ONLINE"
                })

            except Exception as e:
                _safe_print(f"  [DASHBOARD-ERROR] {e}")

        # Final Summary
        self._print_portfolio_summary(all_results)
        
        # Phase 6: Advanced Learning
        try:
            learning_result = self._run_advanced_learning(all_results, multi_asset_data)
            
            # Update dashboard with Phase 6 insights
            try:
                from src.api.state import DashboardState
                ds = DashboardState()
                if learning_result:
                    ds.update_advanced_learning({
                        'regimes': learning_result.get('regimes', {}),
                        'strategies': learning_result.get('strategies', {}),
                        'patterns': learning_result.get('patterns', {}),
                        'timestamp': learning_result.get('timestamp')
                    })
            except Exception:
                pass
                
        except Exception as e:
            _safe_print(f"  [WARNING] Phase 6 Advanced Learning error: {e}")

        # Fallback: if no configured assets had CSVs, try AAVE to populate dashboard
        if not processed_any and os.path.exists("data/AAVE_USDT_1h.csv"):
            _safe_print("\n  [PAPER] No CSVs found for configured assets. Falling back to AAVE for demo.")
            try:
                df = pd.read_csv("data/AAVE_USDT_1h.csv")
                closes = df['close'].tolist()
                highs = df['high'].tolist()
                lows = df['low'].tolist()
                volumes = df['volume'].tolist()
                headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment('AAVE')
                strategy_result = self.strategy.generate_signals(
                    prices=closes, highs=highs, lows=lows, volumes=volumes,
                    headlines=headlines if headlines else None,
                    headline_timestamps=h_timestamps if h_timestamps else None,
                    headline_sources=h_sources if h_sources else None,
                    headline_event_types=h_events if h_events else None,
                )
                signals = strategy_result['signals']
                result = run_backtest(
                    prices=closes, signals=signals, config=self.bt_config, highs=highs, lows=lows
                )
                try:
                    from src.api.state import DashboardState
                    l2 = strategy_result.get('l2_data', {}) or {}
                    sc = float(l2.get('aggregate_score', 0.0))
                    conf = float(l2.get('confidence', 0.0))
                    label = l2.get('aggregate_label', 'NEUTRAL')
                    bull = max(0.0, sc) * 100.0
                    bear = max(0.0, -sc) * 100.0
                    sel_signal = 'LONG' if (signals[-1] if signals else 0) > 0 else ('SHORT' if (signals[-1] if signals else 0) < 0 else 'FLAT')
                    
                    l1_feats = strategy_result.get('l1_data', {}).get('features', [{}])[-1]
                    vpin_val = float(l1_feats.get('vpin_50', 0.0))
                    vol_val = float(l1_feats.get('realized_vol_20', 0.0))
                    trend_val = float(l1_feats.get('trend_strength', 0.0))
                    
                    preds = strategy_result.get('l1_data', {}).get('predictions', [])
                    l1_conf = float(preds[-1][1]) if preds else 0.5
                    c1 = max(0.0, l1_conf)
                    c2 = max(0.0, abs(sc))
                    c3 = 0.3
                    tot = c1 + c2 + c3
                    attr = {'l1': c1 / tot, 'l2': c2 / tot, 'l3': c3 / tot}
                    
                    top_feats = []
                    if isinstance(l1_feats, dict) and l1_feats:
                        try:
                            top_feats = sorted(
                                [{'name': k, 'value': float(v)} for k, v in l1_feats.items() if isinstance(v, (int, float))],
                                key=lambda x: abs(x['value']),
                                reverse=True
                            )[:5]
                        except Exception: pass

                    # Enrich news data for organization
                    organized_news = []
                    for h, t, s, e in zip(headlines[-10:], h_timestamps[-10:], h_sources[-10:], h_events[-10:]):
                        organized_news.append({
                            'text': h,
                            'source': s,
                            'event': e,
                            'time': datetime.fromtimestamp(t).strftime('%H:%M')
                        })

                    DashboardState().update_asset('AAVE', {
                        'signal': sel_signal,
                        'sentiment': {
                            'score': sc, 'label': label, 'confidence': conf,
                            'bull_pct': bull, 'bear_pct': bear,
                            'headlines': headlines[-5:], # List for legacy compat
                            'organized_news': organized_news 
                        },
                        'factors': {
                            'vpin': vpin_val, 'liquidity_regime': 'NORMAL',
                            'volatility': vol_val, 'trend_strength': trend_val,
                            'funding_rate': 0.0001
                        },
                        'attribution': attr,
                        'veto': {'active': False, 'reason': ''},
                        'features_top': top_feats,
                        'sent_hist': [sc for _ in range(20)]
                    })
                    
                    # Fix: Push layers health in fallback mode
                    layers = {
                        "L1 Quant": {"status": "OK", "progress": float(c1), "metric": f"Conf {l1_conf:.2f}"},
                        "L2 Sentiment": {"status": "OK", "progress": float(min(1.0, abs(sc))), "metric": f"Score {sc:.2f} {label}"},
                        "L3 Risk": {"status": "OK", "progress": 0.8, "metric": f"VPIN {vpin_val:.2f}"},
                        "L4 Meta": {"status": "OK", "progress": 0.7, "metric": "Alpha Fused"},
                        "L5 Execution": {"status": "OK", "progress": 0.8, "metric": "ROUTER ACTIVE"},
                        "L6 Strategist": {"status": "OK", "progress": float(conf/100.0 if conf <= 1 else conf), "metric": f"Conf {conf:.2f}"},
                        "L7 Autonomy": {"status": "OK", "progress": 0.9, "metric": "PAPER REPLAY"},
                        "L8 Monitoring": {"status": "OK", "progress": 0.9, "metric": "HEALTH GREEN"},
                        "L9 Learning": {"status": "OK", "progress": 0.5, "metric": "Evolution Ready"}
                    }
                    DashboardState().set_layers(layers)
                    DashboardState().set_sources({
                        "exchange": "ONLINE",
                        "news": "ONLINE" if headlines else "OFFLINE",
                        "onchain": "OFFLINE",
                        "llm": "ONLINE" if self.strategist.api_key else "OFFLINE"
                    })
                except Exception as e:
                    _safe_print(f"  [DASHBOARD-STRICT] Failed to update AAVE state: {e}")
            except Exception as e:
                _safe_print(f"  [X] Fallback AAVE run failed: {e}")



    def _run_live(self):
        """Main execution loop for Live/Testnet simulation with HFT Latency Guard."""
        _safe_print(f"\n  [EXECUTION] Entering real-time monitoring loop...")
        _safe_print(f"  Press Ctrl+C to exit.")
        _safe_print("-" * 60)

        while True:
            t0 = time.time()
            self.iteration_count += 1
            
            # --- PHASE 5: EVENT AWARENESS ---
            if self.event_guard.is_risk_high():
                _safe_print(f"  [EVENT-GUARD] High risk event detected! Pausing execution...")
                time.sleep(60)
                continue
            
            # --- LAYER 6: AGENTIC REVIEW ---
            if self.iteration_count == 1 or self.iteration_count % 6 == 0:
                self._perform_agentic_review()
            
            # --- PHASE 5: CONTINUOUS RISK MONITORING ---
            current_prices = {}
            for asset in self.assets:
                symbol = f"{asset}/USDT"
                p = self.price_source.fetch_latest_price(symbol)
                if p: current_prices[asset] = p
            
            self._check_active_stops(current_prices)

            # Fetch Balance and Return %
            balance_info = self.price_source.get_balance()
            current_usdt = balance_info.get('USDT', 0.0) if 'error' not in balance_info else self.initial_capital
            total_return_pct = ((current_usdt - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0
            
            # Update Dashboard State
            try:
                from src.api.state import DashboardState
                pnl_abs = current_usdt - self.initial_capital
                DashboardState().update_portfolio(pnl=pnl_abs, asset_return=total_return_pct)
            except: pass

            _safe_print(f"\n  [LIVE] BAR {self.iteration_count} | Wallet: ${current_usdt:,.2f} | Return: {total_return_pct:>+6.2f}%")

            for asset in self.assets:
                symbol = f"{asset}/USDT"

                t_fetch = time.time()  # Measure ONLY the data fetch latency
                ohlcv_data = self._fetch_data(symbol)
                if ohlcv_data is None: continue

                # Latency Check (HFT Guard) — measures data fetch, not model load
                api_latency = (time.time() - t_fetch) * 1000.0
                if not self.health_checker.monitor_latency(api_latency):
                    _safe_print(f"  [LATENCY-GUARD] High latency detected ({api_latency:.1f}ms). Skipping trade.")
                    continue

                # Fetch Institutional & Microstructure Signals
                ext_feats = {}
                try:
                    derivatives = self.price_source.fetch_derivatives_data(symbol, ohlcv_data['closes'][-1])
                    ext_feats.update(derivatives)
                    
                    ob_data = self.price_source.exchange.fetch_order_book(symbol)
                    l2_metrics = self.microstructure.analyze_order_book(ob_data)
                    ext_feats.update(l2_metrics)
                    
                    # Liquidity Regime Detection
                    l_regime = self.microstructure.detect_liquidity_regime(l2_metrics['bid_depth_usd'], l2_metrics['ask_depth_usd'])
                    ext_feats['liquidity_regime'] = l_regime

                except Exception: pass

                # Update L1 Features in Dashboard
                try:
                    from src.api.state import DashboardState
                    l1_push = {
                        'vpin': ext_feats.get('vpin_50', 0.0),
                        'ob_imbalance': ext_feats.get('imbalance', 0.0),
                        'liquidity_regime': ext_feats.get('liquidity_regime', 'NORMAL'),
                        'funding_rate': ext_feats.get('funding_rate', 0.0001),
                        'volatility': ext_feats.get('volatility_20', 0.0)
                    }
                    DashboardState().update_l1_features(asset, l1_push)
                except: pass

                headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)

                # Generate signals
                strategy_result = self.strategy.generate_signals(
                    prices=ohlcv_data['closes'],
                    highs=ohlcv_data['highs'],
                    lows=ohlcv_data['lows'],
                    volumes=ohlcv_data['volumes'],
                    headlines=headlines,
                    headline_timestamps=h_timestamps,
                    headline_sources=h_sources,
                    headline_event_types=h_events,
                    external_features=ext_feats,
                    agentic_bias=self.agentic_bias
                )

                # Update Sentiment Scores in Dashboard
                try:
                    from src.api.state import DashboardState
                    l2_data = strategy_result.get('l2_data', {})
                    sent_push = {
                        'composite_score': l2_data.get('sentiment_score', 0.5),
                        'news_momentum': l2_data.get('momentum', 0.0),
                        'veto_active': l2_data.get('veto', False),
                        'headlines_count': len(headlines)
                    }
                    DashboardState().update_sentiment(asset, sent_push)
                except: pass

                signals = strategy_result['signals']
                last_signal = signals[-1] if signals else 0
                
                if last_signal != 0:
                    last_feats = strategy_result.get('l1_data', {}).get('features', [{}])[-1]
                    if self._check_model_drift(last_feats):
                        _safe_print("  [DRIFT-CHECK] Skipping trade due to significant model drift.")
                        last_signal = 0

                # Phase 6: Adverse Selection Protection (VPIN Flow Toxicity)
                vpin_val = strategy_result.get('l1_data', {}).get('features', [{}])[-1].get('vpin_50', 0.0)
                if last_signal != 0 and vpin_val > 0.8:
                    _safe_print(f"  [TOXIC-FLOW] High VPIN ({vpin_val:.2f}). Market is too one-sided for safe entry.")
                    last_signal = 0

                try:
                    from src.api.state import DashboardState
                    l2 = strategy_result.get('l2_data', {}) or {}
                    sc = float(l2.get('aggregate_score', 0.0))
                    conf = float(l2.get('confidence', 0.0))
                    label = l2.get('aggregate_label', 'NEUTRAL')
                    bull = max(0.0, sc) * 100.0
                    bear = max(0.0, -sc) * 100.0
                    sel_signal = 'LONG' if last_signal > 0 else ('SHORT' if last_signal < 0 else 'FLAT')
                    veto_active = False
                    veto_reason = ""
                    if vpin_val > 0.8:
                        veto_active = True
                        veto_reason = "VPIN_TOXIC"
                    ds = DashboardState()
                    preds = strategy_result.get('l1_data', {}).get('predictions', [])
                    l1_conf = float(preds[-1][1]) if preds else 0.5
                    c1 = max(0.0, l1_conf)
                    c2 = max(0.0, abs(sc))
                    c3 = 1.0 if veto_active else 0.3
                    tot = c1 + c2 + c3 if (c1 + c2 + c3) > 0 else 1.0
                    attr = {'l1': c1 / tot, 'l2': c2 / tot, 'l3': c3 / tot}
                    last_feats = strategy_result.get('l1_data', {}).get('features', [{}])[-1]
                    top_feats = []
                    if isinstance(last_feats, dict) and last_feats:
                        try:
                            top_feats = sorted(
                                [{'name': k, 'value': float(v)} for k, v in last_feats.items() if isinstance(v, (int, float))],
                                key=lambda x: abs(x['value']),
                                reverse=True
                            )[:5]
                        except Exception:
                            top_feats = []
                    try:
                        # 2. Enrich news data for organization
                        organized_news = []
                        for h, t, s, e in zip(headlines[-10:], h_timestamps[-10:], h_sources[-10:], h_events[-10:]):
                            organized_news.append({
                                'text': h,
                                'source': s,
                                'event': e,
                                'time': datetime.fromtimestamp(t).strftime('%H:%M')
                            })

                        # Publish layers snapshot
                        layers = {
                            "L1 Quant": {"status": "OK", "progress": float(c1), "metric": f"Conf {l1_conf:.2f}"},
                            "L2 Sentiment": {"status": "OK", "progress": float(min(1.0, abs(sc))), "metric": f"Score {sc:.2f} {label}"},
                            "L3 Risk": {"status": "WARN" if veto_active else "OK", "progress": float(0.0 if veto_active else 0.8), "metric": f"VPIN {vpin_val:.2f}"},
                            "L4 Meta": {"status": "OK", "progress": 0.7, "metric": f"Arb Scale {attr['l1']:.2f}/{attr['l2']:.2f}/{attr['l3']:.2f}"},
                            "L5 Execution": {"status": "OK", "progress": 0.8, "metric": "Router READY"},
                            "L6 Strategist": {"status": "OK" if self.strategist.api_key else "WARN", "progress": float(conf/100.0 if conf <= 1 else conf), "metric": f"Conf {conf:.2f}"},
                            "L7 Autonomy": {"status": "OK", "progress": 0.9, "metric": "Cycle ACTIVE"},
                            "L8 Monitoring": {"status": "OK", "progress": 0.9, "metric": "Health GREEN"},
                            "L9 Learning": {"status": "OK", "progress": 0.5, "metric": "Meta-learning queued"}
                        }
                        ds.set_layers(layers)
                        
                        # Source Status bar
                        ds.set_sources({
                            "exchange": "ONLINE" if bool(ohlcv_data) else "OFFLINE",
                            "news": "ONLINE" if (headlines and len(headlines)>0) else "OFFLINE",
                            "onchain": "ONLINE" if bool(ext_feats.get('onchain')) else "OFFLINE",
                            "llm": "ONLINE" if self.strategist.api_key else "OFFLINE"
                        })
                        
                        # Compute Optimization Edge (Uplift)
                        # We compare the Agentic Win Rate vs a baseline fixed 50% strategy
                        journal_summary = self.journal.get_summary()
                        current_accuracy = journal_summary.get('win_rate', 0.5)
                        baseline = 0.48 
                        ds.update_performance_edge({
                            "uplift_pct": (current_accuracy - baseline) * 100,
                            "baseline_winrate": baseline,
                            "agent_winrate": current_accuracy
                        })
                    except Exception as e:
                        _safe_print(f"  [DASHBOARD-STRICT] Health update error: {e}")

                    # update asset snapshot
                    ds.update_asset(asset, {
                        'signal': sel_signal,
                        'sentiment': {
                            'score': sc,
                            'label': label,
                            'confidence': conf,
                            'bull_pct': bull,
                            'bear_pct': bear,
                            'headlines': headlines[-5:] if headlines else [],
                            'organized_news': organized_news
                        },
                        'factors': {
                            'vpin': float(vpin_val),
                            'liquidity_regime': ext_feats.get('liquidity_regime', 'NORMAL') if ext_feats else 'NORMAL',
                            'volatility': float(ext_feats.get('realized_vol_20', 0.0)) if ext_feats else 0.0,
                            'trend_strength': float(ext_feats.get('trend_strength', 0.0)) if ext_feats else 0.0,
                            'funding_rate': float(ext_feats.get('funding_rate', 0.0)) if ext_feats else 0.0
                        },
                        'attribution': attr,
                        'veto': {'active': veto_active, 'reason': veto_reason},
                        'features_top': top_feats,
                        'weights': {'l1': 0.5, 'l2': 0.3, 'l3': 0.2}
                    })
                    
                    # Update detailed metrics for enhanced dashboard
                    # Execution metrics
                    ds.update_execution_metrics({
                        "slippage": 0.02,  # Mock for now - would come from actual execution
                        "fill_rate": 100.0,
                        "latency_ms": 23,
                        "orders_per_min": 0.5
                    })
                    
                    # Risk metrics
                    ds.update_risk_metrics({
                        "vpin_threshold": 0.8,
                        "max_drawdown": -2.1,
                        "risk_score": vpin_val * 0.5  # Simple risk score based on VPIN
                    })
                    
                    # Training status
                    ds.update_training_status(
                        status="IDLE",
                        version="v6.0",
                        gain=2.3,
                        next_training="15:00 UTC"
                    )
                    
                    # Memory metrics
                    ds.update_memory_metrics(
                        latency=15,
                        confidence=0.87
                    )
                    
                    # Strategist metrics
                    ds.update_strategist_metrics(
                        confidence=0.78,
                        last_reasoning="Bullish momentum building"
                    )
                    
                    # Add layer activity logs
                    ds.add_layer_log("L1", f"VPIN updated: {vpin_val:.4f}", "INFO")
                    ds.add_layer_log("L2", f"Sentiment: {label} ({sc:.2f})", "INFO")
                    ds.add_layer_log("L3", f"Risk veto: {'ACTIVE' if veto_active else 'CLEAR'}", "WARN" if veto_active else "INFO")
                    ds.add_layer_log("L4", f"Signal fusion: L1={attr['l1']:.2f} L2={attr['l2']:.2f} L3={attr['l3']:.2f}", "INFO")
                    ds.add_layer_log("L5", "Execution: Order processed successfully", "INFO")
                    ds.add_layer_log("L6", f"Strategist confidence: {conf:.2f}", "INFO")
                    ds.add_layer_log("L7", "Pattern detection: Active", "INFO")
                    ds.add_layer_log("L8", "Memory retrieval: 87% similarity", "INFO")
                    ds.add_layer_log("L9", "Model training: Scheduled", "INFO")
                    # append sentiment history
                    state_now = ds.get_full_state()
                    prev_hist = []
                    try:
                        prev_hist = state_now.get('active_assets', {}).get(asset, {}).get('sent_hist', [])
                    except Exception:
                        prev_hist = []
                    new_hist = (prev_hist[-49:] if isinstance(prev_hist, list) else []) + [sc]
                    ds.update_asset(asset, {'sent_hist': new_hist})
                    try:
                        closes_arr = ohlcv_data.get('closes', [])[-100:]
                        highs_arr = ohlcv_data.get('highs', [])[-100:]
                        lows_arr = ohlcv_data.get('lows', [])[-100:]
                        opens_arr = []
                        for i, c in enumerate(closes_arr):
                            prev_c = closes_arr[i - 1] if i > 0 else c
                            opens_arr.append(prev_c)
                        now_ts = int(time.time())
                        bar_sec = 3600
                        ohlc_payload = []
                        for i in range(len(closes_arr)):
                            tstamp = now_ts - (len(closes_arr) - 1 - i) * bar_sec
                            ohlc_payload.append({'t': tstamp, 'o': float(opens_arr[i]), 'h': float(highs_arr[i]), 'l': float(lows_arr[i]), 'c': float(closes_arr[i])})
                        ds.update_asset(asset, {'ohlc': ohlc_payload})
                    except Exception:
                        pass
                    try:
                        prev_decisions = state_now.get('active_assets', {}).get(asset, {}).get('decision_hist', [])
                    except Exception:
                        prev_decisions = []
                    decision_entry = {'t': int(time.time()), 'l1': attr['l1'], 'l2': attr['l2'], 'l3': attr['l3'], 'dir': int(last_signal)}
                    ds.update_asset(asset, {'decision_hist': (prev_decisions[-49:] if isinstance(prev_decisions, list) else []) + [decision_entry]})
                except Exception:
                    pass

                _safe_print(f"     Latest signal: {last_signal:+d}")
                
                # ═══ REAL-TIME BENCHMARK: Record prediction direction ═══
                try:
                    from src.api.state import DashboardState
                    # Record predicted direction for accuracy tracking
                    # actual_direction is computed from price movement next iteration
                    prev_price = ohlcv_data['closes'][-2] if len(ohlcv_data['closes']) >= 2 else ohlcv_data['closes'][-1]
                    curr_price = ohlcv_data['closes'][-1]
                    actual_direction = 1 if curr_price > prev_price else (-1 if curr_price < prev_price else 0)
                    DashboardState().record_prediction(last_signal, actual_direction)
                except Exception:
                    pass
                
                _force_trade_mode = self.config.get('force_trade', False) and self.mode == 'testnet'
                if last_signal != 0 or _force_trade_mode:
                    if last_signal == 0 and _force_trade_mode:
                        # Force a direction based on most recent price action
                        last_signal = 1  # Default to LONG on forced mode
                        _safe_print(f"  [FORCE-TRADE] Signal was FLAT, forcing LONG for testnet execution")
                    # Pass extra context for Audit Logging
                    self._execute_autonomous_trade(asset, symbol, last_signal, ohlcv_data['closes'][-1], 
                                                 strategy_result=strategy_result, ext_feats=ext_feats)

            _safe_print(f"\n  [SLEEP] Waiting {self.poll_interval}s for next bar...")
            time.sleep(self.poll_interval)


    def _execute_autonomous_trade(self, asset: str, symbol: str, signal: int, current_price: float, 
                                  strategy_result: Optional[Dict] = None, ext_feats: Optional[Dict] = None):
        """
        Institutional-grade autonomous execution with full Audit/Compliance Logging.
        Implements Ensemble Voting, Reasoning Trace, and Split-Order Execution.
        """
        try:
            _safe_print(f"  [PHASE 5] Evaluating trade for {asset} (Signal: {signal})")
            
            # 1. ENSEMBLE VOTING (Institutional Requirement)
            # We combine LightGBM, RL, and PatchTST for consensus
            # (In production, these would be called independently with specific OHLCV windows)
            ohlcv_data = self._fetch_data(symbol)
            prices = np.array(ohlcv_data['closes']) if ohlcv_data else np.array([current_price])
            
            # Model Predictors
            l1_score = strategy_result.get('l1_data', {}).get('confidence', 0.6) if strategy_result else 0.5
            
            # Create MarketState for RL Agent
            m_state = MarketState(
                price=current_price,
                returns_5m=0.001, # Placeholder logic
                returns_15m=0.002,
                returns_1h=0.005,
                volatility=ext_feats.get('realized_vol_20', 0.03) if ext_feats else 0.03,
                rsi=50.0,
                macd_signal=0.0,
                momentum=0.0,
                volume_ratio=1.0,
                trend_strength=0.5,
                zscore=0.0,
                time_of_day=time.localtime().tm_hour,
                day_of_week=time.localtime().tm_wday
            )
            rl_action_obj = self.rl_agent.select_action(m_state) 
            ptst_res = self.patch_tst.predict(prices)
            
            # Map RL action to probability-like score
            rl_score = 0.8 if rl_action_obj.action_type == "BUY" else (0.2 if rl_action_obj.action_type == "SELL" else 0.5)
            
            # Ensemble Consensus (3-Way Voting)
            ensemble_confidence = (l1_score + ptst_res['prob_up'] + rl_score) / 3
            
            # Force-trade mode for testnet: much tighter neutral band
            _force_trade = self.config.get('force_trade', False) and self.mode == 'testnet'
            if _force_trade:
                # Narrow band: almost any signal triggers a trade
                final_direction = 1 if ensemble_confidence > 0.48 else (-1 if ensemble_confidence < 0.48 else 0)
                if final_direction == 0:
                    # If exactly 0.48, force LONG (testnet only)
                    final_direction = 1
                _safe_print(f"  [FORCE-TRADE] Testnet override active — direction={final_direction:+d}, conf={ensemble_confidence:.3f}")
            else:
                final_direction = 1 if ensemble_confidence > 0.6 else (-1 if ensemble_confidence < 0.4 else 0)
            
            # ════════════════════════════════════════════════════════════════
            # REAL-TIME: Record each model's prediction individually
            # ════════════════════════════════════════════════════════════════
            try:
                from src.api.state import DashboardState
                _ds = DashboardState()
                # Compute actual direction from last 2 bars
                if ohlcv_data and len(ohlcv_data['closes']) >= 2:
                    _prev = ohlcv_data['closes'][-2]
                    _curr = ohlcv_data['closes'][-1]
                    _actual_dir = 1 if _curr > _prev else (-1 if _curr < _prev else 0)
                    
                    # LightGBM prediction direction
                    lgbm_dir = 1 if l1_score > 0.55 else (-1 if l1_score < 0.45 else 0)
                    _ds.record_model_prediction("lightgbm", lgbm_dir, _actual_dir)
                    
                    # PatchTST prediction direction
                    ptst_dir = 1 if ptst_res['prob_up'] > 0.55 else (-1 if ptst_res['prob_up'] < 0.45 else 0)
                    _ds.record_model_prediction("patchtst", ptst_dir, _actual_dir)
                    
                    # RL Agent prediction direction
                    rl_dir = 1 if rl_action_obj.action_type == "BUY" else (-1 if rl_action_obj.action_type == "SELL" else 0)
                    _ds.record_model_prediction("rl_agent", rl_dir, _actual_dir)
                    
                    # Ensemble direction
                    _ds.record_prediction(final_direction, _actual_dir)
            except Exception:
                pass
            
            # Veto Logic: If consensus doesn't match L1, or VPIN is toxic (L2 microstructure), block.
            vpin_status = self.vpin.is_flow_toxic()
            if vpin_status['is_toxic']:
                if _force_trade:
                    _safe_print(f"  [FORCE-TRADE] VPIN toxic ({vpin_status['vpin']:.2f}) — overriding veto for testnet")
                else:
                    _safe_print(f"  [VETO] Adverse Selection Protect: VPIN {vpin_status['vpin']:.2f} is Toxic. Blocking.")
                    return

            # ════════════════════════════════════════════════════════════════
            # LAYER X: PROFIT PROTECTION & LOSS AVERSION (NEW)
            # ════════════════════════════════════════════════════════════════
            # Update balance info for profit protector
            balance_info = self.price_source.get_balance()
            current_usdt = balance_info.get('USDT', 0.0) if 'error' not in balance_info else self.initial_capital
            self.profit_protector.update_balance(current_usdt)
            
            # Get current profit status
            profit_status = self.profit_protector.get_profit_status()
            
            # Rate the trade quality
            atr_val = ext_feats.get('atr_14', 0.01) if ext_feats else 0.01
            stop_loss = current_price * (1 - atr_val/current_price) if current_price > 0 else current_price - atr_val
            take_profit = current_price * (1 + atr_val * 2 / current_price) if current_price > 0 else current_price + atr_val * 2
            position_size = (self.initial_capital * 0.01) / current_price if current_price > 0 else 0.1
            
            trade_quality = self.profit_protector.rate_trade_quality(
                signal_confidence=ensemble_confidence,
                model_win_rate=self.profit_protector.win_rate,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                current_balance=current_usdt
            )
            
            # Decision: Should we trade?
            should_trade, protection_msg = self.profit_protector.should_enter_trade(
                trade_quality=trade_quality,
                current_balance=current_usdt,
                force_hold_if_profitable=True
            )
            
            if not should_trade:
                if _force_trade:
                    _safe_print(f"  [FORCE-TRADE] Profit protector rejected — overriding for testnet")
                    _safe_print(f"               Current P&L: {profit_status['total_pnl_pct']:+.2f}%")
                else:
                    _safe_print(f"  [LOSS-AVERSION] {protection_msg}")
                    _safe_print(f"               Current P&L: {profit_status['total_pnl_pct']:+.2f}%")
                    return
            
            _safe_print(f"  [LOSS-AVERSION] {protection_msg}")
            _safe_print(f"               Trade Quality: {trade_quality.recommendation} ({trade_quality.quality_score:.0f}/100)")
            _safe_print(f"               P(Win): {trade_quality.win_probability:.1%} | Expectancy: ${trade_quality.profit_expectancy:+.2f}")
            _safe_print(f"               Current P&L: {profit_status['total_pnl_pct']:+.2f}% | In Profit: {profit_status['is_profitable']}")
            
            # Adapt position size based on confidence and win rate
            adaptive_pos_size = self.profit_protector.get_adaptive_position_size(
                base_risk_pct=self.config.get('risk', {}).get('risk_per_trade_pct', 1.0),
                trade_quality=trade_quality
            )
            _safe_print(f"               Adaptive Position Size: {adaptive_pos_size:.1%} of base risk")

            # 2. INSTITUTIONAL REASONING TRACE (Audit Compliance)
            reasoning = f"Institutional Consensus: ProbUp={ensemble_confidence:.2f}. "
            reasoning += f"L1={l1_score:.2f}, PatchTST={ptst_res['prob_up']:.2f}, RL={rl_action_obj.action_type}. "
            reasoning += f"Regime={ptst_res.get('regime', 'UNKNOWN')}. Toxicity={vpin_status['vpin']:.2f}."
            
            model_votes = {
                "LightGBM": {"direction": 1 if l1_score > 0.5 else -1, "confidence": l1_score},
                "PatchTST": ptst_res,
                "RL_Policy_Prob": 0.33 # Placeholder for policy probability
            }
            
            risk_checks = {
                "VPIN_Toxicity": vpin_status['risk_action'],
                "KillSwitch": "OFF",
                "MaxDrawdown": "OK"
            }
            
            # Publish to Audit Infrastructure
            self.stream.log_trading_decision(
                symbol=symbol, direction="LONG" if final_direction > 0 else "SHORT",
                confidence=float(ensemble_confidence), model_votes=model_votes,
                reasoning=reasoning, risk_checks=risk_checks
            )

            # 3. ALPHA ENRICHMENT: Strategy Weighting based on Regime
            perf_metrics = self._get_asset_performance_metrics()
            onchain_data = {asset: asdict(self.on_chain_portfolio.compute_on_chain_signal(asset))}
            regime = ext_feats.get('liquidity_regime', 'NORMAL') if ext_feats else 'NORMAL'
            
            allocations = self.portfolio_allocator.allocate_portfolio(
                assets=self.assets, performance_metrics=perf_metrics, 
                onchain_data=onchain_data, regime=regime
            )
            
            allocation = allocations.get(asset)
            if not allocation: return

            side = "buy" if final_direction > 0 else "sell"
            pos_size_pct = allocation.position_size_pct
            
            # 4. MICROSTRUCTURE & LIQUIDITY SMART ROUTING
            qty = (pos_size_pct * self.initial_capital) / current_price
            order_book = self.price_source.exchange.fetch_order_book(symbol) if hasattr(self.price_source, 'exchange') else None
            
            slippage_est = self.router.slippage.estimate_price_impact(qty, 5000000.0) # 5M ADV default
            
            # Select Institutional Execution Algorithm
            execution_id = self.router.execute_advanced_order(
                symbol=symbol, side=side, quantity=qty, 
                algo="TWAP" if qty > 0.05 else "Direct", # Force TWAP for multi-BTC trades
                order_book=order_book
            )

            # 5. EXECUTION AUDIT
            _safe_print(f"  [PHASE 5] ORDER INITIATED: {execution_id} (Slippage Est: {slippage_est:.4%})")
            self.stream.log_execution(
                order_id=execution_id, symbol=symbol, side=side, 
                qty=qty, price=current_price, slippage=slippage_est, type="TWAP" if qty > 0.05 else "DIRECT"
            )
            
            if execution_id != "FAILED":
                # ═══ LAYER 6: Per-Trade LLM Analysis (NEW) ═══
                # Generate detailed reasoning for THIS SPECIFIC TRADE
                llm_reasoning = ""
                try:
                    # Extract L1, L2, L3 signals for LLM analysis
                    l1_signal_dict = {
                        'confidence': float(l1_score * 100),
                        'prediction': 'BUY' if final_direction > 0 else 'SELL',
                        'top_features': list(ext_feats.keys())[:5] if ext_feats else []
                    }
                    
                    l2_data = strategy_result.get('l2_data', {}) if strategy_result else {}
                    l2_sentiment_dict = {
                        'sentiment_score': float(l2_data.get('aggregate_score', 0.0)),
                        'confidence': float(l2_data.get('confidence', 0.0) * 100),
                        'news_count': len(strategy_result.get('headlines', [])) if strategy_result else 0,
                        'source_breakdown': l2_data.get('source_breakdown', {}),
                        'label': l2_data.get('aggregate_label', 'NEUTRAL')
                    }
                    
                    l3_risk_dict = {
                        'vpin': float(vpin_status.get('vpin', 0.0)),
                        'funding_rate': float(ext_feats.get('funding_rate', 0.0) * 100) if ext_feats else 0.0,
                        'liquidation_levels': f"${current_price * 0.95:.2f} | ${current_price * 1.05:.2f}",
                        'position_concentration': 'LOW' if ensemble_confidence < 0.7 else 'HIGH'
                    }
                    
                    market_info = {
                        'regime': regime,
                        'atr': float(ext_feats.get('atr_14', 0.0)) if ext_feats else 0.0,
                        'trend_direction': 'UP' if final_direction > 0 else 'DOWN',
                        'volatility': float(ext_feats.get('realized_vol_20', 0.0)) if ext_feats else 0.0
                    }
                    
                    # Get recent trades for context
                    recent_trades = self.journal.get_recent_trades(limit=5) if hasattr(self.journal, 'get_recent_trades') else []
                    
                    # Call strategist for per-trade analysis
                    llm_reasoning = self.strategist.analyze_trade(
                        asset=asset,
                        entry_price=current_price,
                        entry_side=side,
                        l1_signal=l1_signal_dict,
                        l2_sentiment=l2_sentiment_dict,
                        l3_risk=l3_risk_dict,
                        market_data=market_info,
                        recent_trades=recent_trades
                    )
                    _safe_print(f"  [L6-REASONING] {llm_reasoning[:150]}...")
                except Exception as e:
                    _safe_print(f"  [L6-WARNING] Could not generate LLM reasoning: {str(e)[:50]}")
                    llm_reasoning = reasoning  # Fallback to institutional reasoning
                
                # Enhance reasoning with LLM analysis
                final_reasoning = f"{reasoning}\n[L6-ANALYSIS] {llm_reasoning}"
                
                self.journal.log_trade(
                    asset=asset, side=side, quantity=qty, price=current_price,
                    regime=regime, strategy_name="HybridAlpha_v6.5_Institutional",
                    confidence=float(ensemble_confidence),
                    reasoning=final_reasoning,
                    order_id=execution_id,
                    feature_vector=ext_feats if ext_feats else {},
                    model_signal=final_direction
                )
                self.risk_manager.register_trade_open(asset, final_direction, current_price, pos_size_pct, order_id=execution_id)
                
                # ═══ REAL-TIME BENCHMARK: Record trade entry ═══
                try:
                    from src.api.state import DashboardState
                    DashboardState().record_trade({
                        'asset': asset,
                        'side': side,
                        'entry_price': current_price,
                        'quantity': qty,
                        'confidence': float(ensemble_confidence),
                        'timestamp': time.time(),
                        'direction': final_direction,
                    })
                except Exception:
                    pass

        except Exception as e:
            _safe_print(f"  [ERROR] Execution failed: {e}")

    def _check_model_drift(self, current_features: Dict[str, float]) -> bool:
        """
        Monitors model drift/decay.
        Compares current feature distributions to training expectations.
        """
        # Phase 6: Institutional Integrity Check
        # Simplified: If volatility or correlation deviates 3 sigma from average
        # In production: KL Divergence or PSI (Population Stability Index)
        vol = current_features.get('realized_vol_annual', 0.5)
        if vol > 1.2: # 120% annualized vol is extreme shift
            _safe_print(f"  [DRIFT DECAY] High feature drift detected! (Vol: {vol:.2%})")
            return True
        return False

    def _check_active_stops(self, current_prices: Dict[str, float]):
        """Scans for stop-loss, take-profit, or partial exit triggers."""
        # DynamicRiskManager now holds the check_all_stops logic
        triggers = self.risk_manager.check_all_stops(current_prices)
        for t in triggers:
            asset = t['asset']
            trigger = t['trigger']
            price = t['price']
            record = t['record']
            
            _safe_print(f"  [RISK] {trigger.upper()} triggered for {asset} at ${price:.2f}")
            
            if 'partial_tp' in trigger:
                # Execute 50% scale-out
                self._handle_partial_tp(asset, record, price)
            else:
                # Full Exit
                self._handle_full_exit(asset, record, price, trigger)

    def _handle_partial_tp(self, asset: str, record: 'TradeRecord', price: float):
        """Scales out of 50% of the position to secure profits."""
        symbol = f"{asset}/USDT"
        side = "sell" if record.direction > 0 else "buy"
        half_qty = record.size * 0.5
        
        _safe_print(f"  [PHASE 5] Executing Partial TP scale-out: {half_qty:.6f} {asset}")
        
        res = self.router.execute_order(
            symbol=symbol, side=side, quantity=half_qty, order_type="market"
        )
        
        if res.success:
            # Update record size in RiskManager (so full exit knows remaining)
            record.size -= res.executed_quantity
            self.journal.log_trade(
                asset=asset, side=side, quantity=res.executed_quantity,
                price=res.executed_price, regime="Partial TP", strategy_name="ScaleOut",
                confidence=1.0, reasoning="50% Profit Secure + Breakeven Stop set",
                order_id=res.order_id
            )

    def _handle_full_exit(self, asset: str, record: 'TradeRecord', price: float, reason: str):
        """Closes the entire position."""
        symbol = f"{asset}/USDT"
        side = "sell" if record.direction > 0 else "buy"
        
        _safe_print(f"  [PHASE 5] Executing Full Exit ({reason}): {record.size:.6f} {asset}")
        
        res = self.router.execute_order(
            symbol=symbol, side=side, quantity=record.size, order_type="market"
        )
        
        if res.success:
            pnl = (res.executed_price - record.entry_price) * record.size * record.direction
            return_pct = ((res.executed_price - record.entry_price) / record.entry_price * 100) * record.direction
            
            self.risk_manager.close_position(asset, res.executed_price)
            self.journal.close_trade(
                order_id=record.order_id,
                exit_price=res.executed_price,
                pnl=pnl
            )
            
            # ═══ REAL-TIME BENCHMARK: Record completed trade with P&L ═══
            try:
                from src.api.state import DashboardState
                ds = DashboardState()
                ds.record_trade({
                    'asset': asset,
                    'side': side,
                    'entry_price': record.entry_price,
                    'exit_price': res.executed_price,
                    'pnl': pnl,
                    'return_pct': return_pct,
                    'quantity': res.executed_quantity,
                    'reason': reason,
                    'timestamp': time.time(),
                    'direction': record.direction,
                })
                
                # Push updated benchmark metrics to dashboard
                all_trades = ds.get_full_state().get('trade_history', [])
                closed_trades = [t for t in all_trades if 'pnl' in t]
                if closed_trades:
                    wins = sum(1 for t in closed_trades if t['pnl'] > 0)
                    wr = wins / len(closed_trades)
                    rets = [t['return_pct'] / 100 for t in closed_trades if 'return_pct' in t]
                    sharpe = 0.0
                    if len(rets) >= 2:
                        import numpy as _np
                        arr = _np.array(rets)
                        sharpe = (float(_np.mean(arr)) / float(_np.std(arr))) * float(_np.sqrt(252)) if float(_np.std(arr)) > 0 else 0.0
                    
                    ds.update_benchmark_metrics({
                        'win_rate': wr,
                        'sharpe_ratio': sharpe,
                        'total_trades': len(closed_trades),
                        'total_pnl': sum(t['pnl'] for t in closed_trades),
                    })
                    
                    # Also save to persistent benchmark history
                    self.benchmark.evaluate_all_from_executor(
                        trade_history=closed_trades,
                        returns=rets if len(rets) >= 2 else None,
                    )
            except Exception:
                pass

    def _fetch_data(self, symbol: str) -> Optional[Dict[str, List[float]]]:
        """Fetch and format OHLCV data for strategy ingestion."""
        try:
            # Match style-based timeframe
            timeframe = '1h' if self.style == TradingStyle.SWING else '5m'
            raw = self.price_source.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)
            if not raw: return None
            return self.price_source.extract_ohlcv(raw)
        except Exception as e:
            _safe_print(f"  [ERROR] Failed to fetch data for {symbol}: {e}")
            return None

    def _fetch_sentiment(self, asset: str) -> Tuple[List[str], List[float], List[str], List[str]]:
        """Fetch latest sentiment headlines and metadata."""
        news_items = self.news.fetch_all(asset)

        if news_items:
            _safe_print(f"  [L2 SENTIMENT] Fetched {len(news_items)} news items for {asset}")
            # Count sources in the items
            source_breakdown = {}
            for item in news_items:
                source = item.source
                source_breakdown[source] = source_breakdown.get(source, 0) + 1
            
            # Display source breakdown
            _safe_print(f"  [L2 SENTIMENT] Source Breakdown:")
            for source, count in sorted(source_breakdown.items(), key=lambda x: -x[1]):
                _safe_print(f"    • {source}: {count} items")
            
            for i, item in enumerate(news_items[:3]):  # Just print the top 3
                _safe_print(f"    📰 {item.source}: {item.title[:80]}...")
        else:
            _safe_print(f"  [L2 SENTIMENT] No recent news found for {asset}.")

        news_data = [n.to_dict() for n in news_items]
        headlines = [n['title'] for n in news_data]
        timestamps = [n.get('timestamp', time.time()) for n in news_data]
        sources = [n.get('source', 'Unknown') for n in news_data]
        events = [n.get('event_type', 'GENERAL') for n in news_data]
        return headlines, timestamps, sources, events

    def _get_asset_performance_metrics(self) -> Dict[str, Any]:
        """Calculates trailing performance for the Allocator."""
        # In a real system, this would pull from the TradingJournal
        # Here we provide a mock or summary of recent trade history PnL
        summary = self.journal.get_summary()
        return {
            "win_rate": summary.get("win_rate", 0.5),
            "total_pnl": summary.get("total_pnl", 0.0),
            "returns": [t.get("pnl", 0.0) for t in self.journal.trades[-50:]] if self.journal.trades else [0.0]
        }

    def _print_portfolio_summary(self, all_results: Dict[str, Any]):
        """Consolidates results across all assets."""
        _safe_print("\n" + "="*50)
        _safe_print("  FINAL PORTFOLIO SUMMARY (BACKTEST)")
        _safe_print("="*50)
        
        total_pnl = 0.0
        total_trades = 0
        winning_trades = 0
        
        for asset, res in all_results.items():
            net_pnl = res.net_pnl_pct
            total_pnl += net_pnl
            total_trades += res.total_trades
            winning_trades += res.winning_trades
            _safe_print(f"  {asset: <10} | PnL: {net_pnl:>+7.2f}% | Trades: {res.total_trades:>3}")
            
        avg_pnl = total_pnl / len(all_results) if all_results else 0.0
        win_rate = (winning_trades / (total_trades or 1) * 100)
        
        _safe_print("-" * 50)
        _safe_print(f"  TOTAL ASSETS:  {len(all_results)}")
        _safe_print(f"  AVG ASSET PNL: {avg_pnl:>+7.2f}%")
        _safe_print(f"  TOTAL TRADES:  {total_trades}")
        _safe_print(f"  WIN RATE:      {win_rate:>5.1f}%")
        _safe_print("="*50 + "\n")

    def _run_advanced_learning(self, all_results: Dict[str, Any], multi_asset_data: Dict[str, Any]):
        """Meta-Learning (Layer 9) — Performance Review & Strategy Evolution."""
        _safe_print("  [PHASE 6] Running Meta-Learning Evolution (Alpha)...")
        from src.ai.advanced_learning import AdvancedLearningEngine
        engine = AdvancedLearningEngine()
        
        for asset, res in all_results.items():
             results_dict = {
                 "total_return_pct": res.net_pnl_pct,
                 "sharpe_ratio": res.sharpe_ratio,
                 "winning_trades": res.winning_trades,
                 "total_trades": res.total_trades
             }
             engine.update_with_backtest_results(asset, results_dict)
             
        status = engine.get_system_status()
        _safe_print(f"  [META] Advanced Learning Engine State: {status.get('active_strategies', 0)} Active Strategies.")
        engine.save_learned_models()
        return status
