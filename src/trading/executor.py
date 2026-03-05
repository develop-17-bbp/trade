"""
Trading Executor -- Full System Orchestrator
=============================================
Main entry point that orchestrates the three-layer hybrid signal architecture:

  1. Fetch real-time OHLCV data from Binance (via CCXT)
  2. Run L1 quantitative engine (indicators, models, cycles)
  3. Fetch and process L2 sentiment (news, social media)
  4. Apply L3 risk evaluation (stops, limits, volatility gating)
  5. Combine signals (50/30/20 weights + VETO)
  6. Run enhanced backtest with fees, slippage, and full metrics
  7. Output comprehensive performance report

Supports paper mode (backtest) and simulation mode.
"""

import time
import os
import sys
from typing import Dict, Optional, List
from dataclasses import asdict

from src.data.fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.institutional_fetcher import InstitutionalFetcher
from src.ai.sentiment import SentimentPipeline
from src.ai.agentic_strategist import AgenticStrategist
from src.ai.advanced_learning import AdvancedLearningEngine, MarketRegime
from src.ai.reinforcement_learning import ReinforcementLearningAgent, AdaptiveAlgorithmLayer
from src.data.on_chain_fetcher import OnChainFetcher
from src.trading.strategy import HybridStrategy, SimpleStrategy
from src.trading.backtest import (
    run_backtest, BacktestConfig, BacktestResult, format_backtest_report
)
from src.risk.manager import RiskManager
from src.integrations.robinhood_stub import RobinhoodClient
from src.portfolio.allocator import PortfolioAllocator
from src.portfolio.hedger import PortfolioHedger
from src.monitoring.health_checker import SystemHealthChecker
from src.execution.failover import ExecutionFailoverController
from src.risk.dynamic_manager import DynamicRiskManager
from src.execution.router import ExecutionRouter, ExecutionMode
from src.integrations.on_chain_portfolio import OnChainPortfolioManager

def _safe_print(msg: str = ""):
    """Print with UTF-8 encoding, falling back to ASCII-safe output on Windows."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('ascii', errors='replace').decode('ascii'))


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
            api_key=exchange_cfg.get('api_key'),
            api_secret=exchange_cfg.get('api_secret'),
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
        self.strategist = AgenticStrategist(
            provider=self.config.get('ai', {}).get('reasoning_provider', 'openai'),
            model=self.config.get('ai', {}).get('reasoning_model', 'gpt-4-turbo')
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
        
        self.risk_manager = DynamicRiskManager(initial_capital=self.initial_capital)
        self.execution_router = ExecutionRouter(
            mode=ExecutionMode.TESTNET if self.mode == 'testnet' else ExecutionMode.LIVE
        )
        
        self.agentic_bias = 0.0
        self.iteration_count = 0

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

    def _init_robinhood(self):
        """Mock/Stub for Robinhood login."""
        creds = self.config.get('robinhood', {})
        if creds.get('username') and creds.get('password'):
            self.robinhood = RobinhoodClient()
            self.robinhood.login(creds['username'], creds['password'])

    def run(self):
        """Main entry point -- run the full system."""
        _safe_print("=" * 60)
        _safe_print("  AI-DRIVEN CRYPTO TRADING SYSTEM")
        _safe_print("  Three-Layer Hybrid Signal Architecture")
        _safe_print("=" * 60)
        _safe_print(f"  Mode:    {self.mode}")
        _safe_print(f"  Assets:  {', '.join(self.assets)}")
        _safe_print(f"  Capital: ${self.initial_capital:,.2f}")
        data_label = "Binance TESTNET (sandbox)" if self.price_source.testnet else "Live (Binance CCXT)"
        _safe_print(f"  Data Source: {data_label}")
        _safe_print("-" * 60)

        try:
            from src.api.state import DashboardState
            DashboardState().set_status("TRADING")
        except: pass

        if self.mode == 'paper':
            self._run_paper()
        elif self.mode == 'testnet':
            self._run_testnet()
        else:
            self._run_live()

    def _run_paper(self):
        """Backtest mode on historical data (loads from CSV)."""
        all_results: Dict[str, BacktestResult] = {}
        multi_asset_data = {}
        import pandas as pd

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

            # 3. Generate signals using L1 + L2
            _safe_print(f"  [SIGNAL] Generating hybrid signals (FinBERT + LightGBM)...")
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

            # 5. Output asset report
            _safe_print(f"\n--- Report for {asset} ---")
            _safe_print(format_backtest_report(result))

            # 6. Record backtest for learning (L1 incremental training)
            self.strategy.record_backtest(result, asset, strategy_result['l1_data'])

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

    def _run_live(self):
        """Main execution loop for Live/Testnet simulation."""
        _safe_print(f"\n  [EXECUTION] Entering real-time monitoring loop...")
        _safe_print(f"  Press Ctrl+C to exit.")
        _safe_print("-" * 60)

        while True:
            self.iteration_count += 1
            
            # --- LAYER 6: AGENTIC REVIEW (Initial + Every 6 cycles) ---
            if self.iteration_count == 1 or self.iteration_count % 6 == 0:
                self._perform_agentic_review()

            for asset in self.assets:
                symbol = f"{asset}/USDT"
                _safe_print(f"\n  [LIVE] Scanning {symbol}...")

                # Fetch price data
                ohlcv_data = self._fetch_data(symbol)
                if ohlcv_data is None:
                    continue

                closes = ohlcv_data['closes']
                highs = ohlcv_data['highs']
                lows = ohlcv_data['lows']
                volumes = ohlcv_data['volumes']

                # Fetch sentiment
                headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)

                # Fetch Institutional Data
                ext_feats = {}
                try:
                    ccxt_derivatives = self.price_source.fetch_derivatives_data(symbol)
                    ext_feats.update(ccxt_derivatives)
                    ext_scraped = self.institutional.get_all_institutional(asset)
                    ext_feats.update(ext_scraped)
                except Exception:
                    pass

                # Check account balance
                acct_info = self.price_source.get_balance()
                if acct_info and 'error' not in acct_info:
                    _safe_print(f"     USDT: {acct_info.get('USDT', 0):,.2f}")

                # Generate signals
                strategy_result = self.strategy.generate_signals(
                    prices=closes,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    headlines=headlines if headlines else None,
                    headline_timestamps=h_timestamps if h_timestamps else None,
                    headline_sources=h_sources if h_sources else None,
                    headline_event_types=h_events if h_events else None,
                    external_features=ext_feats if ext_feats else None,
                    agentic_bias=self.agentic_bias
                )

                signals = strategy_result['signals']
                last_signal = signals[-1] if signals else 0
                _safe_print(f"     Latest signal: {last_signal:+d}")

                # Execute order if signal changed
                if last_signal != 0:
                    # Phase 5: Autonomous execution with portfolio allocation and risk management
                    self._execute_autonomous_trade(asset, symbol, last_signal, closes[-1])

            # Wait for next cycle (e.g. 5 min)
            _safe_print(f"\n  [SLEEP] Waiting 300s for next bar...")
            time.sleep(300)

    def _run_testnet(self):
        """Testnet execution mode - real-time signals with simulated orders."""
        _safe_print(f"\n  [TESTNET] Entering testnet execution mode...")
        _safe_print(f"  Press Ctrl+C to exit.")
        _safe_print("-" * 60)

        while True:
            self.iteration_count += 1

            # --- LAYER 6: AGENTIC REVIEW (Initial + Every 6 cycles) ---
            if self.iteration_count == 1 or self.iteration_count % 6 == 0:
                self._perform_agentic_review()

            for asset in self.assets:
                symbol = f"{asset}/USDT"
                _safe_print(f"\n  [TESTNET] Scanning {symbol}...")

                # Fetch price data
                ohlcv_data = self._fetch_data(symbol)
                if ohlcv_data is None:
                    continue

                closes = ohlcv_data['closes']
                highs = ohlcv_data['highs']
                lows = ohlcv_data['lows']
                volumes = ohlcv_data['volumes']

                # Fetch sentiment
                headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)

                # Fetch Institutional Data
                ext_feats = {}
                try:
                    ccxt_derivatives = self.price_source.fetch_derivatives_data(symbol)
                    ext_feats.update(ccxt_derivatives)
                    ext_scraped = self.institutional.get_all_institutional(asset)
                    ext_feats.update(ext_scraped)
                except Exception:
                    pass

                # Check account balance (testnet)
                acct_info = self.price_source.get_balance()
                if acct_info and 'error' not in acct_info:
                    _safe_print(f"     USDT: {acct_info.get('USDT', 0):,.2f}")

                # Generate signals
                strategy_result = self.strategy.generate_signals(
                    prices=closes,
                    highs=highs,
                    lows=lows,
                    volumes=volumes,
                    headlines=headlines if headlines else None,
                    headline_timestamps=h_timestamps if h_timestamps else None,
                    headline_sources=h_sources if h_sources else None,
                    headline_event_types=h_events if h_events else None,
                    external_features=ext_feats if ext_feats else None,
                    agentic_bias=self.agentic_bias
                )

                signals = strategy_result['signals']
                last_signal = signals[-1] if signals else 0
                _safe_print(f"     Latest signal: {last_signal:+d}")

                # Execute order if signal changed (using Phase 5 autonomous execution)
                if last_signal != 0:
                    self._execute_autonomous_trade(asset, symbol, last_signal, closes[-1])

            # Wait for next cycle (e.g. 5 min)
            _safe_print(f"\n  [SLEEP] Waiting 300s for next bar...")
            time.sleep(300)

    def _execute_autonomous_trade(self, asset: str, symbol: str, signal: int, current_price: float):
        """
        Phase 5: Execute autonomous trade with portfolio allocation and risk management.
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
            symbol: Trading pair (e.g., 'BTC/USDT')
            signal: Trading signal (-1, 0, 1)
            current_price: Current market price
        """
        try:
            _safe_print(f"  [PHASE 5] Evaluating autonomous trade for {asset}...")
            
            # Get current portfolio allocations
            performance_metrics = self._get_asset_performance_metrics()
            onchain_data = {asset: asdict(self.on_chain_portfolio.compute_on_chain_signal(asset))}
            
            allocations = self.portfolio_allocator.allocate_portfolio(
                assets=self.assets,
                performance_metrics=performance_metrics,
                onchain_data=onchain_data
            )
            
            # Get allocation for this asset
            allocation = allocations.get(asset)
            if not allocation:
                _safe_print(f"  [PHASE 5] No allocation calculated for {asset}")
                return
            
            # Determine trade direction and size
            side = "buy" if signal > 0 else "sell"
            position_size_pct = allocation.position_size_pct
            
            # Adjust for signal strength (signal can be -1, 0, 1, but we'll use absolute value)
            signal_strength = abs(signal)
            position_size_pct *= signal_strength
            
            # Check risk limits
            current_portfolio_heat = sum(a.position_size_pct for a in allocations.values())
            allowed, reason = self.risk_manager.check_trade_allowed(
                asset=asset,
                proposed_size_pct=position_size_pct,
                current_portfolio_heat=current_portfolio_heat
            )
            
            if not allowed:
                _safe_print(f"  [PHASE 5] Trade blocked by risk management: {reason}")
                return
            
            # Calculate quantity
            position_size_usd = position_size_pct * self.initial_capital
            quantity = position_size_usd / current_price
            
            _safe_print(f"  [PHASE 5] Executing {side} order: {quantity:.6f} {asset} (${position_size_usd:.2f})")
            _safe_print(f"  [PHASE 5] Kelly fraction: {allocation.kelly_fraction:.3f}, On-chain confidence: {allocation.onchain_confidence:.3f}")
            
            # Execute the order
            execution_result = self.execution_router.execute_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type="market"  # Use market orders for now
            )
            
            if execution_result.success:
                _safe_print(f"  [PHASE 5] ✓ Order executed successfully")
                _safe_print(f"  [PHASE 5]   Order ID: {execution_result.order_id}")
                _safe_print(f"  [PHASE 5]   Executed: {execution_result.executed_quantity:.6f} @ ${execution_result.executed_price:.2f}")
                _safe_print(f"  [PHASE 5]   Fee: ${execution_result.fee:.4f}")
                
                # Update risk manager with P&L (simplified - would need actual fill tracking)
                # For now, assume immediate execution at requested price
                pnl_change = 0.0  # Would calculate based on actual entry/exit
                self.risk_manager.update_pnl(pnl_change)
                
            else:
                _safe_print(f"  [PHASE 5] ✗ Order execution failed: {execution_result.error_message}")
                
        except Exception as e:
            _safe_print(f"  [PHASE 5] Error in autonomous execution: {e}")
            import traceback
            traceback.print_exc()

    def _get_asset_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for all assets (simplified version).
        In production, this would pull from actual trade history.
        """
        # Simplified metrics - in production would calculate from actual performance
        metrics = {}
        for asset in self.assets:
            # Mock performance data - replace with real calculations
            metrics[asset] = {
                "win_rate": 0.55,  # 55% win rate
                "avg_win_loss_ratio": 1.8,  # 1.8:1 reward/risk
                "sharpe_ratio": 1.2,  # Decent risk-adjusted returns
                "volatility": 0.03,  # 3% daily volatility
                "max_drawdown": 0.08  # 8% max drawdown
            }
        return metrics
        # 0. Initial Agentic Sync (Instant Telemetry)
        try:
            from src.api.state import DashboardState
            DashboardState().set_status("SCANNING")
        except: pass
        self._perform_agentic_review()

        # 1. Fetch and show balance
        balance = self.price_source.get_balance()
        if 'error' not in balance:
            _safe_print(f"  [BALANCE] USDT: ${balance.get('USDT', 0.0):,.2f} | BTC: {balance.get('BTC', 0.0):.6f} | ETH: {balance.get('ETH', 0.0):.6f}")
        else:
            _safe_print(f"  [BALANCE] Notice: {balance['error']} (Read-only mode)")

        all_results: Dict[str, BacktestResult] = {}

        for asset in self.assets:
            symbol = f"{asset}/USDT"
            _safe_print(f"\n{'-' * 80}")
            _safe_print(f"  Asset Analysis: {symbol}")
            _safe_print(f"{'-' * 80}")
            _safe_print(f"  [DYNAMIC OPTIMIZATION] Evaluating timeframes...")
            _safe_print(f"  {'TF':<8} | {'P&L %':<12} | {'PF':<8} | {'Trades':<8}")
            _safe_print(f"  {'-' * 45}")

            best_tf = None
            best_score = -float('inf')
            best_res = None
            
            timeframes_to_test = ['15m', '1h', '4h', '1d', '1w', '1M']
            for tf in timeframes_to_test:
                ohlcv_data = self._fetch_data(symbol, timeframe=tf)
                if ohlcv_data is None:
                    continue

                closes = ohlcv_data['closes']
                highs = ohlcv_data['highs']
                lows = ohlcv_data['lows']
                volumes = ohlcv_data['volumes']

                # Fetch sentiment
                headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)

                # Generate signals
                strategy_result = self.strategy.generate_signals(
                    prices=closes, highs=highs, lows=lows, volumes=volumes,
                    headlines=headlines if headlines else None,
                    headline_timestamps=h_timestamps if h_timestamps else None,
                    headline_sources=h_sources if h_sources else None,
                    headline_event_types=h_events if h_events else None,
                )

                signals = strategy_result['signals']

                # Run backtest for analysis
                result = run_backtest(
                    prices=closes,
                    signals=signals,
                    config=self.bt_config,
                    highs=highs,
                    lows=lows
                )

                _safe_print(f"  {tf:<8} | {result.total_return_pct:>10.2f}% | {result.profit_factor:>8.2f} | {len(result.trades):>8}")
                
                # Selection score: prioritize Return, but favor higher PF
                score = result.total_return_pct + (result.profit_factor * 0.1)
                
                if score > best_score:
                    best_score = score
                    best_tf = tf
                    best_res = result

            if best_res:
                all_results[asset] = best_res
                _safe_print(f"\n  [OPTIMIZED] Best timeframe for {asset} is {best_tf} ({best_res.total_return_pct:.2f}%)")

                # Bootstrap memory vault with optimized trades
                if len(best_res.trades) > 0:
                    _safe_print(f"  [MEMORY] Archiving {len(best_res.trades)} trade experiences for {asset}...")
                    for t in best_res.trades:
                        self.strategist.record_trade_outcome({
                            'asset': asset,
                            'signal': t.direction,
                            'entry_price': t.entry_price,
                            'exit_price': t.exit_price,
                            'size': t.size,
                            'duration': t.holding_bars,
                            'exit_reason': t.exit_reason,
                            'confidence': 75.0
                        }, t.net_pnl)
                
                # Reasoning
                if best_res.total_return_pct > 2.0 and best_res.profit_factor > 1.5:
                    reason = "Strong trend capture with high profit factor."
                elif best_res.profit_factor > 2.0:
                    reason = "Exceptional win/loss ratio, prioritizing capital safety."
                elif best_res.total_return_pct > 0:
                    reason = "Highest absolute profitability in this regime."
                else:
                    reason = "Least defensive posture among tested timeframes."
                
                _safe_print(f"  [REASON] {reason}")
                
                if best_res.total_return_pct < -2.0:
                    _safe_print(f"  [VETO] Returns catastrophic (< -2%). Skipping orders for risk prevention.")
                else:
                    _safe_print(f"  [GO] Signal quality acceptable. Ready for orders.")

        self._print_portfolio_summary(all_results)

        # 3. Enter real-time loop using the best parameters found
        _safe_print("\n  [TESTNET] Entering continuous monitoring loop...")
        try:
            from src.api.state import DashboardState
            DashboardState().set_status("TRADING")
        except: pass

        while True:
            self.iteration_count += 1
            
            # Phase 6 & Layer 6: Meta-Learning + Agentic Review
            if self.iteration_count == 1 or self.iteration_count % 6 == 0:
                self._perform_agentic_review()
                self._run_system_learning()
            
            for asset in self.assets:
                # Periodic sync to keep dashboard alive
                pass
            
            _safe_print(f"\n  [SLEEP] Waiting 300s for next bar...")
            time.sleep(300)

    def _perform_agentic_review(self):
        """Perform agentic review and sync to dashboard."""
        _safe_print("\n  [AGENTIC] Performing reasoning-based system review...")
        try:
            import pandas as pd
            hist_path = "logs/trade_history.csv"
            trades = []
            if os.path.exists(hist_path):
                df = pd.read_csv(hist_path)
                trades = df.tail(20).to_dict('records')
            
            sample_symbol = f"{self.assets[0]}/USDT"
            raw_data = self.price_source.fetch_ohlcv(sample_symbol, limit=20)
            processed = PriceFetcher.extract_ohlcv(raw_data)
            
            from src.indicators.indicators import atr as compute_atr
            atr_val = compute_atr(processed['highs'], processed['lows'], processed['closes'])[-1]
            
            m_data = {
                "atr": float(atr_val),
                "funding_rate": self.price_source.fetch_derivatives_data(sample_symbol).get('funding_rate', 0.0),
                "onchain": asdict(self.on_chain_portfolio.compute_on_chain_signal(self.assets[0])),
                "asset": self.assets[0],
                "sentiment": {"bullish": 0.0, "bearish": 0.0}  # Neutral default
            }
            
            try:
                from src.api.state import DashboardState
                ds = DashboardState()
                ds.update_onchain(m_data['onchain'])
            except: pass

            decision = self.strategist.analyze_performance(trades, self.config, m_data)
            _safe_print(f"     [!] Reason:      {decision.reasoning_trace}")
            _safe_print(f"     [!] Regime:      {decision.market_regime}")
            _safe_print(f"     [!] Confidence:  {decision.confidence_score}%")
            self.agentic_bias = decision.macro_bias

        except Exception as e:
            _safe_print(f"  [!] Agentic review failed: {e}")

    def _run_system_learning(self):
        """Phase 6: Run meta-learning on recent data."""
        _safe_print("\n  [META-LEARN] Running Phase 6 adaptive learning...")
        try:
            multi_asset_data = {}
            for asset in self.assets:
                symbol = f"{asset}/USDT"
                ohlcv = self._fetch_data(symbol)
                if ohlcv:
                    import pandas as pd
                    # Rename executor keys to match AdvancedLearningEngine expectations
                    df = pd.DataFrame({
                        'close': ohlcv['closes'],
                        'high': ohlcv['highs'],
                        'low': ohlcv['lows'],
                        'volume': ohlcv['volumes']
                    })
                    multi_asset_data[asset] = df
            
            if multi_asset_data:
                learning_result = self._run_advanced_learning({}, multi_asset_data)
                # Sync results to dashboard (convert dataclasses to dicts)
                try:
                    from src.api.state import DashboardState
                    ds = DashboardState()
                    # MarketRegime is a dataclass — serialize to dict for JSON
                    regimes_serialized = {}
                    for asset_key, regime_obj in learning_result.get('regimes', {}).items():
                        if hasattr(regime_obj, 'regime_type'):
                            regimes_serialized[asset_key] = {
                                'regime_type': regime_obj.regime_type,
                                'confidence': regime_obj.confidence,
                                'volatility': regime_obj.volatility,
                                'trend_strength': regime_obj.trend_strength,
                                'optimal_strategy': regime_obj.optimal_strategy
                            }
                        else:
                            regimes_serialized[asset_key] = regime_obj
                    
                    ds.update_advanced_learning({
                        'regimes': regimes_serialized,
                        'strategies': learning_result.get('strategies', {}),
                        'patterns': learning_result.get('patterns', {}),
                        'timestamp': learning_result.get('timestamp')
                    })
                    _safe_print(f"  [META-LEARN] Dashboard synced with {len(regimes_serialized)} regime classifications")
                except Exception as e:
                    _safe_print(f"  [META-LEARN] Dashboard sync failed: {e}")
        except Exception as e:
            _safe_print(f"  [!] Meta-learning cycle failed: {e}")

    def _print_portfolio_summary(self, results: Dict[str, BacktestResult]):
        """Aggregate and print global portfolio performance."""
        _safe_print("\n" + "=" * 80)
        _safe_print("  FINAL PORTFOLIO SUMMARY")
        _safe_print("=" * 80)

        total_pnl = 0.0
        total_ret = 0.0

        for asset, res in results.items():
            _safe_print(f"\n  {asset}:")
            _safe_print(f"    Net P&L:    $ {res.net_pnl:>12,.2f}")
            _safe_print(f"    Return:     {res.total_return_pct:>10.2f}%")
            _safe_print(f"    Profit Fact: {res.profit_factor:>10.2f}")
            _safe_print(f"    Sharpe:     {res.sharpe_ratio:>10.3f}")
            _safe_print(f"    Max DD:     {res.max_drawdown_pct:>10.2f}%")
            _safe_print(f"    Trades:     {res.total_trades:>12}")
            _safe_print(f"    Win Rate:   {res.win_rate * 100:>10.1f}%")
            
            total_pnl += res.net_pnl
            total_ret += res.total_return_pct

        avg_ret = total_ret / len(results) if results else 0
        _safe_print("-" * 50)
        _safe_print(f"  Total Portfolio P&L: $ {total_pnl:>12,.2f}")
        _safe_print(f"  Avg Asset Return:      {avg_ret:>10.2f}%")
        _safe_print("=" * 80 + "\n")

        # Dashboard Update
        try:
            from src.api.state import DashboardState
            ds = DashboardState()
            ds.update_portfolio(total_pnl, avg_ret)
            for asset, res in results.items():
                ds.update_asset(asset, {
                    "pnl": res.net_pnl,
                    "return": res.total_return_pct,
                    "trades": res.total_trades,
                    "win_rate": res.win_rate * 100
                })
        except ImportError:
            pass

    def _fetch_data(self, symbol: str, timeframe: Optional[str] = None) -> Optional[Dict]:
        """Fetch real-time data."""
        if timeframe is None:
            timeframe = '1h' if self.mode == 'testnet' else '1d'

        try:
            raw = self.price_source.fetch_ohlcv(symbol, timeframe=timeframe, limit=2000)
            if not raw:
                raise ValueError(f"No data returned for {symbol}")
            return PriceFetcher.extract_ohlcv(raw)
        except Exception as e:
            _safe_print(f"  [X] Failed to fetch {symbol}: {e}")
            raise

    def _fetch_sentiment(self, asset: str):
        """Fetch headlines from primary sources and return metadata."""
        try:
            items = self.news.fetch_all(asset, limit=50)
            headlines = [item.title for item in items]
            timestamps = [item.timestamp for item in items]
            sources = [item.source for item in items]
            events = [item.event_type for item in items]
            return headlines, timestamps, sources, events
        except Exception:
            return [], [], [], []

    def _run_advanced_learning(self, all_results: Dict[str, BacktestResult], 
                               price_data: Dict[str, Dict]) -> Dict:
        """
        PHASE 6: Run advanced learning to generate adaptive strategies.
        Analyzes performance across assets and learns optimal hyperparameters.
        """
        _safe_print("\n" + "=" * 80)
        _safe_print("  PHASE 6: ADVANCED LEARNING (Meta-Learning Engine)")
        _safe_print("=" * 80)
        
        # Step 1: Process market data through advanced learning
        import pandas as pd
        multi_asset_data = {}
        for asset in self.assets:
            if asset in price_data:
                # Convert to DataFrame
                data = price_data[asset]
                df = pd.DataFrame(data)
                multi_asset_data[asset] = df
        
        if not multi_asset_data:
            _safe_print("  [SKIP] No market data for advanced learning.")
            return {}
        
        # Step 2: Run advanced learning pipeline
        try:
            # Get onchain data for all assets
            onchain_data = {}
            for asset in self.assets:
                try:
                    onchain_data[asset] = asdict(self.on_chain_portfolio.compute_on_chain_signal(asset))
                except Exception as e:
                    _safe_print(f"  [WARNING] Failed to get onchain data for {asset}: {e}")
            
            learning_result = self.advanced_learning.process_market_data(multi_asset_data, onchain_data)
            _safe_print(f"  [LEARNING] Processed {len(multi_asset_data)} assets with onchain integration")
            _safe_print(f"  [PATTERNS] Discovered {len(learning_result['patterns'])} cross-market patterns")
            _safe_print(f"  [REGIMES] Classified {len(learning_result['regimes'])} market regimes")
            _safe_print(f"  [STRATEGIES] Generated {len(learning_result['strategies'])} adaptive strategies")
        except Exception as e:
            _safe_print(f"  [WARNING] Advanced learning failed: {e}")
            return {}
        
        # Step 3: Update adaptive algorithm layer with backtest results
        for asset, result in all_results.items():
            try:
                self.adaptive_algo.record_performance(
                    'neutral',  # Could be tracked per variant
                    {
                        "pnl_pct": result.total_return_pct,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown_pct / 100.0
                    }
                )
            except:
                pass
        
        # Step 4: Adapt RL agent to current market conditions with onchain data
        if onchain_data:
            for asset in self.assets:
                if asset in onchain_data:
                    try:
                        market_metrics = {
                            "volatility": learning_result.get('regimes', {}).get(asset, MarketRegime("UNKNOWN", 0, 0.03, 0, 0, "HOLD")).volatility,
                            "trend_strength": learning_result.get('regimes', {}).get(asset, MarketRegime("UNKNOWN", 0, 0.03, 0, 0, "HOLD")).trend_strength,
                            "momentum": onchain_data[asset].get('on_chain_momentum', 0.0)
                        }
                        self.adaptive_algo.adapt_to_market_conditions(market_metrics, onchain_data[asset])
                    except Exception as e:
                        _safe_print(f"  [WARNING] Failed to adapt RL agent for {asset}: {e}")
        
        # Step 4: Display regime-specific strategies
        _safe_print("\n  [REGIMES & STRATEGIES]:")
        for asset, regime in learning_result.get('regimes', {}).items():
            _safe_print(f"    {asset}: {regime.regime_type} (confidence: {regime.confidence:.1f}%)")
            if asset in learning_result.get('strategies', {}):
                strategy = learning_result['strategies'][asset]
                _safe_print(f"      Strategy: {strategy['strategy_name']}")
                _safe_print(f"      Expected Performance: {strategy['predicted_performance']:.3f}")
        
        # Step 5: Display highest-confidence patterns
        _safe_print("\n  [CROSS-MARKET PATTERNS]:")
        patterns = learning_result.get('patterns', {})
        if patterns.get('momentum_breakout'):
            _safe_print(f"    Momentum Breakouts: {len(patterns['momentum_breakout'])} assets")
        if patterns.get('mean_reversion'):
            _safe_print(f"    Mean Reversion Signals: {len(patterns['mean_reversion'])} assets")
        if patterns.get('volatility_expansion'):
            _safe_print(f"    Volatility Expansion: {len(patterns['volatility_expansion'])} assets")
        
        # Step 6: Save learned models
        try:
            self.advanced_learning.save_learned_models()
            _safe_print("\n  [SAVED] Advanced learning models persisted")
        except Exception as e:
            _safe_print(f"  [WARNING] Could not save models: {e}")
        
        return learning_result
    
    def _run_reinforcement_learning(self, asset: str, price_data: Optional[Dict] = None):
        """
        Run reinforcement learning on a single asset.
        Trains policy for optimal trading actions.
        """
        if not price_data:
            return None
        
        _safe_print(f"  [RL-LEARNING] Training RL agent for {asset}...")
        
        # Convert price data to market states and train
        # This is a simplified version - in production would use actual trade trajectory
        try:
            policy = self.rl_agent.get_policy_summary()
            _safe_print(f"    Action preferences: {policy['action_preferences']}")
            return policy
        except Exception as e:
            _safe_print(f"    [WARNING] RL learning failed: {e}")
            return None

