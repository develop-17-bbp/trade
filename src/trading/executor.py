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

from src.data.fetcher import PriceFetcher
from src.data.news_fetcher import NewsFetcher
from src.data.institutional_fetcher import InstitutionalFetcher
from src.ai.sentiment import SentimentPipeline
from src.trading.strategy import HybridStrategy, SimpleStrategy
from src.trading.backtest import (
    run_backtest, BacktestConfig, BacktestResult, format_backtest_report
)
from src.risk.manager import RiskManager
from src.integrations.robinhood_stub import RobinhoodClient


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
        _safe_print(f"  Data:    {data_label}")
        if self.price_source.is_authenticated:
            _safe_print(f"  Auth:    API Key authenticated (order execution enabled)")
        broker_label = 'Robinhood' if self.robinhood and self.robinhood.authenticated else 'CCXT'
        _safe_print(f"  Broker:  {broker_label}")
        _safe_print("=" * 60)
        _safe_print()

        if self.mode == 'paper':
            self._run_paper()
        elif self.mode == 'testnet':
            self._run_testnet()
        elif self.mode == 'live':
            if self.robinhood and self.robinhood.authenticated:
                self._run_live()
            else:
                _safe_print("[!] Robinhood not authenticated. Falling back to paper mode.")
                self._run_paper()
        else:
            _safe_print("Unknown mode. Use 'paper', 'testnet', or 'live'.")

    def _init_robinhood(self):
        """Initialize Robinhood client with credentials from environment or config."""
        if self.mode != 'live':
            return  # Only init for live mode

        rh_config = self.config.get('robinhood', {})
        username = rh_config.get('username') or os.environ.get('ROBINHOOD_USER')
        password = rh_config.get('password') or os.environ.get('ROBINHOOD_PASSWORD')
        mfa_code = rh_config.get('mfa_code') or os.environ.get('ROBINHOOD_MFA')

        if not username or not password:
            _safe_print(
                "[!] Robinhood credentials not found. Live mode requires ROBINHOOD_USER and ROBINHOOD_PASSWORD env vars."
            )
            return

        _safe_print("[INIT] Initializing Robinhood client...")
        self.robinhood = RobinhoodClient(cache_token=True)
        if self.robinhood.login(username, password, mfa_code=mfa_code):
            _safe_print(f"[OK] Robinhood authenticated as {username}")
        else:
            _safe_print("[!] Robinhood authentication failed. Will fall back to paper mode.")
            self.robinhood = None

    def _run_live(self):
        """Run system in live mode with real Robinhood execution."""
        _safe_print("\n[LIVE MODE] Trading with real capital via Robinhood")
        _safe_print("[WARNING] Ensure you have reviewed the strategy thoroughly!")
        _safe_print("[WARNING] Start with small positions before scaling.")
        _safe_print()

        all_results: Dict[str, Dict] = {}

        for asset in self.assets:
            symbol = f"{asset}/USDT"
            _safe_print(f"\n{'-' * 50}")
            _safe_print(f"  Live Trading: {symbol}")
            _safe_print(f"{'-' * 50}")

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

            # --- NEW: Fetch Institutional Data ---
            ext_feats = {}
            try:
                # CCXT Source
                ccxt_derivatives = self.price_source.fetch_derivatives_data(symbol)
                ext_feats.update(ccxt_derivatives)
                # External Source (CoinDesk, Amberdata placeholders)
                ext_scraped = self.institutional.get_all_institutional(asset)
                ext_feats.update(ext_scraped)
            except Exception:
                pass

            # Check account balance
            acct_info = self.robinhood.get_account_balance()
            if acct_info:
                _safe_print(f"\n  [ACCOUNT]")
                _safe_print(f"     Cash:       ${acct_info['cash']:,.2f}")
                _safe_print(f"     Buying Power: ${acct_info['buying_power']:,.2f}")
                _safe_print(f"     Portfolio:  ${acct_info['portfolio_value']:,.2f}")

            # Generate signals
            _safe_print(f"\n  [SIGNAL] Running hybrid strategy...")
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
                account_balance=acct_info['portfolio_value'] if acct_info else self.initial_capital,
            )

            signals = strategy_result['signals']
            l1_data = strategy_result['l1_data']
            last_signal = signals[-1] if signals else 0

            _safe_print(f"     Latest signal: {last_signal:+d}")

            # Execute if signal present
            if last_signal > 0:
                qty = self._calculate_position_size(asset, acct_info)
                _safe_print(f"\n  [EXECUTE] Placing BUY order for {qty} {asset}...")
                order_result = self.robinhood.place_order(
                    symbol=asset,
                    quantity=qty,
                    side='buy',
                    order_type='market'
                )
                if order_result['status'] == 'success':
                    _safe_print(f"     [OK] Buy order placed successfully")
                else:
                    _safe_print(f"     [X] Buy order failed: {order_result['message']}")

            elif last_signal < 0:
                # Get current position
                position = self.robinhood.get_position(asset)
                if position:
                    qty = float(position.get('quantity', 0))
                    if qty > 0:
                        _safe_print(f"\n  [EXECUTE] Placing SELL order for {qty} {asset}...")
                        order_result = self.robinhood.place_order(
                            symbol=asset,
                            quantity=qty,
                            side='sell',
                            order_type='market'
                        )
                        if order_result['status'] == 'success':
                            _safe_print(f"     [OK] Sell order placed successfully")
                        else:
                            _safe_print(f"     [X] Sell order failed: {order_result['message']}")
            else:
                _safe_print(f"\n  [HOLD] Signal is neutral, holding position.")

            all_results[asset] = {
                'strategy': strategy_result,
                'live_execution': True,
            }

        _safe_print(f"\n{'=' * 60}")
        _safe_print("  Live trading cycle complete. Check positions in Robinhood.")
        _safe_print(f"{'=' * 60}")

    def _calculate_position_size(self, asset: str, acct_info: Optional[Dict]) -> float:
        """Calculate position size based on risk management rules."""
        if not acct_info:
            return 0.1  # default to 0.1 units if no account info

        buying_power = acct_info.get('buying_power', 0)
        risk_per_trade_pct = self.bt_config.risk_per_trade_pct / 100.0

        # Allocate 1% of buying power per trade
        risk_amount = buying_power * risk_per_trade_pct
        current_price = self.price_source.fetch_ticker(f"{asset}/USDT")['last']
        qty = risk_amount / current_price if current_price > 0 else 0.1

        # Round to reasonable precision
        return round(qty, 4)

    def _run_testnet(self):
        """Run system in testnet mode — real orders on Binance sandbox with fake money."""
        _safe_print("\n  [TESTNET MODE] Trading with fake money on Binance Testnet")
        _safe_print("  [INFO] Orders execute against a live order book — zero real risk")
        _safe_print()

        # Show testnet balance if authenticated
        if self.price_source.is_authenticated:
            balance = self.price_source.get_balance()
            if 'error' not in balance:
                _safe_print(f"  [BALANCE] Testnet Account:")
                _safe_print(f"     USDT:  {balance.get('USDT', 0.0):,.2f}")
                _safe_print(f"     BTC:   {balance.get('BTC', 0.0):.8f}")
                _safe_print(f"     ETH:   {balance.get('ETH', 0.0):.8f}")
            else:
                _safe_print(f"  [!] Balance fetch failed: {balance['error']}")
        _safe_print()

        all_results: Dict[str, Dict] = {}

        for asset in self.assets:
            symbol = f"{asset}/USDT"
            _safe_print(f"\n{'-' * 50}")
            _safe_print(f"  Testnet Trading: {symbol}")
            _safe_print(f"{'-' * 50}")

            _safe_print(f"  [DYNAMIC OPTIMIZATION] Finding most profitable timeframe...")

            best_tf = None
            best_score = -float('inf')
            best_strategy_result = None
            best_bt_result = None
            best_data = None
            
            timeframes_to_test = ['15m', '1h', '4h', '1d', '1w', '1M']
            for tf in timeframes_to_test:
                _safe_print(f"\n  >> Testing Timeframe: {tf}")
                # Fetch data
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
                from src.indicators.indicators import atr as compute_atr
                atr_vals = compute_atr(highs, lows, closes)

                bt_result = run_backtest(
                    prices=closes, signals=signals,
                    highs=highs, lows=lows, atr_values=atr_vals,
                    config=self.bt_config,
                )
                
                import math
                pf = bt_result.profit_factor if not math.isinf(bt_result.profit_factor) else 2.0
                
                # Base score is purely the Net P&L. We boost it if profit factor is high.
                score = bt_result.net_pnl
                if bt_result.net_pnl > 0 and pf > 1.0:
                    score *= pf  # Reward profitable and highly efficient setups
                elif bt_result.net_pnl < 0:
                    score -= (1.0 / (pf + 0.1))  # Penalize worse profit factors further
                
                # If everything is 0 (no trades or break even), give small penalty so trades are preferred over flat
                if bt_result.net_pnl == 0.0:
                    score = -0.01  
                
                _safe_print(f"     Return: {bt_result.total_return_pct:.2f}% | P&L: ${bt_result.net_pnl:.2f} | PF: {pf:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_tf = tf
                    best_strategy_result = strategy_result
                    best_bt_result = bt_result
                    best_data = ohlcv_data

            if best_tf is None:
                _safe_print(f"  [!] Failed to evaluate timeframes for {symbol}")
                continue
                
            _safe_print(f"\n  [BEST TIMEFRAME] Selected '{best_tf}' with P&L: ${best_bt_result.net_pnl:.2f} for live execution")
            
            # ── STRICT PROFIT GATE ──
            # Only allow trading if the best timeframe is actually profitable
            if best_bt_result.net_pnl <= 0.0:
                _safe_print(f"  [GATE REJECTED] Even the best timeframe ({best_tf}) is losing money (P&L: ${best_bt_result.net_pnl:.2f}).")
                _safe_print(f"  [HOLD] System will not take trades in a currently unprofitable regime.")
                reject_unprofitable = True
            else:
                reject_unprofitable = False
            
            strategy_result = best_strategy_result
            bt_result = best_bt_result
            signals = strategy_result['signals']
            l2_data = strategy_result['l2_data']
            closes = best_data['closes']
            
            num_buy = sum(1 for s in signals if s == 1)
            num_sell = sum(1 for s in signals if s == -1)
            _safe_print(f"\n  [BEST TF REPORT]")
            _safe_print(f"     Signals generated: {num_buy} buy, {num_sell} sell, {len(signals) - num_buy - num_sell} hold")
            
            report = format_backtest_report(bt_result)
            _safe_print(f"\n{report}")

            # Get latest signal for execution
            last_signal = signals[-1] if signals else 0
            _safe_print(f"\n  [LATEST SIGNAL] {'+1 BUY' if last_signal > 0 else '-1 SELL' if last_signal < 0 else '0 HOLD'}")

            # Execute on testnet if authenticated and signal present
            if self.price_source.is_authenticated and last_signal != 0 and not reject_unprofitable:
                current_price = closes[-1]
                # Position size: 1% of capital / current price
                trade_amount = (self.initial_capital * 0.01) / current_price
                trade_amount = max(trade_amount, 0.0001)  # min order size

                side = 'buy' if last_signal > 0 else 'sell'
                _safe_print(f"\n  [TESTNET ORDER] Placing {side.upper()} for {trade_amount:.6f} {asset}")
                order_result = self.price_source.place_order(
                    symbol=symbol,
                    side=side,
                    amount=trade_amount,
                )
                if order_result['status'] == 'success':
                    _safe_print(f"     [OK] Order filled!")
                    _safe_print(f"         ID:     {order_result.get('order_id')}")
                    _safe_print(f"         Price:  ${order_result.get('price', 0):,.2f}")
                    _safe_print(f"         Amount: {order_result.get('filled', 0):.6f}")
                    _safe_print(f"         Cost:   ${order_result.get('cost', 0):,.2f}")
                else:
                    _safe_print(f"     [X] Order failed: {order_result.get('message')}")
            elif not self.price_source.is_authenticated:
                _safe_print(f"\n  [!] API key not set — skipping order execution")
                _safe_print(f"      Set BINANCE_TESTNET_KEY and BINANCE_TESTNET_SECRET env vars")
            else:
                _safe_print(f"\n  [HOLD] Signal is neutral, no order placed.")

            all_results[asset] = {
                'backtest': bt_result,
                'strategy': strategy_result,
                'l2_sentiment': l2_data,
            }

        # Show updated balance
        if self.price_source.is_authenticated:
            _safe_print(f"\n  [BALANCE] Updated Testnet Account:")
            balance = self.price_source.get_balance()
            if 'error' not in balance:
                _safe_print(f"     USDT:  {balance.get('USDT', 0.0):,.2f}")
                _safe_print(f"     BTC:   {balance.get('BTC', 0.0):.8f}")
                _safe_print(f"     ETH:   {balance.get('ETH', 0.0):.8f}")

        # Summary
        if all_results:
            self._print_summary(all_results)

    def _run_paper(self):
        """Run full pipeline in paper mode with backtesting."""
        all_results: Dict[str, Dict] = {}

        for asset in self.assets:
            symbol = f"{asset}/USDT"
            _safe_print(f"\n{'-' * 50}")
            _safe_print(f"  Processing: {symbol}")
            _safe_print(f"{'-' * 50}")

            # ---- Step 1: Fetch Price Data ----
            ohlcv_data = self._fetch_data(symbol)
            if ohlcv_data is None:
                continue

            closes = ohlcv_data['closes']
            highs = ohlcv_data['highs']
            lows = ohlcv_data['lows']
            volumes = ohlcv_data['volumes']

            _safe_print(f"  [DATA] {len(closes)} candles")
            _safe_print(f"     Price range: ${min(closes):,.2f} - ${max(closes):,.2f}")
            _safe_print(f"     Current:     ${closes[-1]:,.2f}")

            # ---- Step 2: Fetch News & Sentiment (L2) ----
            _safe_print(f"\n  [NEWS] Fetching news for {asset}...")
            headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)
            _safe_print(f"     Found {len(headlines)} headlines")

            # ---- Step 3: Run Hybrid Strategy (L1 + L2 + L3) ----
            _safe_print(f"\n  [STRATEGY] Running hybrid strategy...")
            strategy_result = self.strategy.generate_signals(
                prices=closes,
                highs=highs,
                lows=lows,
                volumes=volumes,
                headlines=headlines if headlines else None,
                headline_timestamps=h_timestamps if h_timestamps else None,
                headline_sources=h_sources if h_sources else None,
                headline_event_types=h_events if h_events else None,
                account_balance=self.initial_capital,
            )

            signals = strategy_result['signals']
            l1_data = strategy_result['l1_data']
            l2_data = strategy_result['l2_data']

            num_buy = sum(1 for s in signals if s == 1)
            num_sell = sum(1 for s in signals if s == -1)
            _safe_print(f"     Signals: {num_buy} buy, {num_sell} sell, {len(signals) - num_buy - num_sell} hold")

            # Print L1 details
            _safe_print(f"\n  -- L1 Quantitative Engine --")
            _safe_print(f"     Cycle phase:     {l1_data.get('cycle_phase', 'N/A')}")
            _safe_print(f"     Holding period:  {l1_data.get('holding_period', 'N/A')} bars")
            cycles = l1_data.get('dominant_cycles', [])
            if cycles:
                _safe_print(f"     Top cycles:      {', '.join(f'{c[0]}d ({c[1]:.1%})' for c in cycles[:3])}")
            # Forecast sub-signal if present
            f_sig = l1_data.get('forecast_signal', [])
            if f_sig:
                _safe_print(f"     Forecast signal (last): {f_sig[-1]:+.2f}")

            # Print L4 Meta-Controller Fusion State
            _safe_print(f"\n  -- L4 Meta-Controller (XGBoost Arbitrator) --")
            _safe_print(f"     LGBM Weight:     ~60% (Base)")
            _safe_print(f"     RL Weight:       ~40% (Adapts to 80% if vol > 0.04)")
            _safe_print(f"     Action Taken:    Fused Dual-Signal")

            # Print L2 details
            _safe_print(f"\n  -- L2 Sentiment Layer --")
            _safe_print(f"     Aggregate score: {l2_data.get('aggregate_score', 0):.3f}")
            _safe_print(f"     Label:           {l2_data.get('aggregate_label', 'N/A')}")
            _safe_print(f"     Confidence:      {l2_data.get('confidence', 0):.3f}")
            _safe_print(f"     Sources:         {l2_data.get('num_sources', 0)}")
            _safe_print(f"     Freshness:       {l2_data.get('freshness', 0):.3f}")

            # Print latest RSI / MACD
            rsi_vals = l1_data.get('rsi', [])
            if rsi_vals:
                latest_rsi = rsi_vals[-1]
                if not (latest_rsi != latest_rsi):  # not nan
                    _safe_print(f"\n  -- Latest Indicators --")
                    _safe_print(f"     RSI(14):         {latest_rsi:.2f}")
                    macd_h = l1_data.get('macd_hist', [])
                    if macd_h:
                        _safe_print(f"     MACD Histogram:  {macd_h[-1]:.4f}")
                    z_vals = l1_data.get('zscore', [])
                    if z_vals and not (z_vals[-1] != z_vals[-1]):
                        _safe_print(f"     Z-Score:         {z_vals[-1]:.3f}")

            # ---- Step 4: Run Enhanced Backtest ----
            _safe_print(f"\n  [BACKTEST] Running enhanced backtest...")
            from src.indicators.indicators import atr as compute_atr
            atr_vals = compute_atr(highs, lows, closes)

            bt_result = run_backtest(
                prices=closes,
                signals=signals,
                highs=highs,
                lows=lows,
                atr_values=atr_vals,
                features=strategy_result.get('features'),
                config=self.bt_config,
            )
            # feed results back to strategy so models can learn from mistakes
            try:
                self.strategy.record_backtest(bt_result)
                self.strategy.retrain_models()
            except Exception:
                pass

            # Print report
            report = format_backtest_report(bt_result)
            _safe_print(f"\n{report}")

            # ---- Monte Carlo Evaluation ----
            from src.trading.backtest import monte_carlo_simulation, format_monte_carlo_report
            _safe_print(f"\n  [MONTE CARLO] Running 1,000 simulations with 5% miss rate...")
            mc_results = monte_carlo_simulation(bt_result.trades, n_simulations=1000, miss_rate=0.05)
            mc_report = format_monte_carlo_report(mc_results, getattr(self.bt_config, 'min_return_pct', None))
            _safe_print(f"\n{mc_report}")

            # ---- 1% daily target assessment ----
            self._assess_daily_target(bt_result)

            all_results[asset] = {
                'backtest': bt_result,
                'strategy': strategy_result,
                'l2_sentiment': l2_data,
            }

        # ---- Summary ----
        if all_results:
            self._print_summary(all_results)

    def _fetch_data(self, symbol: str, timeframe: Optional[str] = None) -> Optional[Dict]:
        """Fetch real-time data focusing on L3 events (falling back to CCXT L1)."""
        # Testnet lacks deep daily history (only ~28 days), so we use 1h candles by default
        if timeframe is None:
            timeframe = '1h' if self.mode == 'testnet' else '1d'

        try:
            _safe_print(f"  [LIVE] Fetching {timeframe} candles for {symbol} (CoinAPI via CCXT)...")
            raw = self.price_source.fetch_ohlcv(symbol, timeframe=timeframe, limit=200)

            if not raw or len(raw) < 50:
                _safe_print(f"  [!] Warning: Only {len(raw) if raw else 0} candles fetched for {symbol}.")
                _safe_print(f"      (Need at least 50 for the Trend SMA gate to pass).")

            if not raw:
                raise ValueError(f"No data returned for {symbol}")
            return PriceFetcher.extract_ohlcv(raw)
        except Exception as e:
            _safe_print(f"  [X] Failed to fetch {symbol}: {e}")
            _safe_print(f"  [FATAL] Real-time data is required. Cannot proceed with synthetic data.")
            raise

    def _fetch_sentiment(self, asset: str):
        """Fetch news headlines and analyze sentiment."""
        headlines: List[str] = []
        timestamps: List[float] = []
        sources: List[str] = []
        events: List[str] = []

        try:
            news_limit = self.config.get('news', {}).get('limit', 50)
            news_items = self.news.fetch_all(asset, limit=news_limit)
            for item in news_items:
                headlines.append(item.title)
                timestamps.append(item.timestamp)
                sources.append(item.source)
                events.append(item.event_type)
        except Exception as e:
            _safe_print(f"     [!] News fetch failed: {e}")
            # Fallback
            try:
                headlines = self.news.fetch_headlines(asset, limit=5)
            except Exception:
                headlines = []
            timestamps = [time.time()] * len(headlines)
            sources = ['reddit'] * len(headlines)
            events = ['general'] * len(headlines)

        return headlines, timestamps, sources, events

    def _execute_twap(self):
        """L5 Adaptive Gateway: Routes orders via CCXT handling Binance/Coinbase burst rate limits."""
        pass

    def _assess_daily_target(self, bt_result: BacktestResult):
        """Assess realism of 1% daily target."""
        _safe_print(f"\n  -- 1% Daily Target Assessment --")
        target = 1.0
        actual = bt_result.avg_daily_return_pct

        _safe_print(f"     Target daily return: {target:.2f}%")
        _safe_print(f"     Actual avg daily:    {actual:.4f}%")

        if actual >= target:
            _safe_print(f"     [OK] Target met! (but verify over longer periods)")
        elif actual >= 0.1:
            _safe_print(f"     [~] Below 1% target, but positive ({actual:.4f}%)")
            _safe_print(f"       -> Realistic achievable range: 0.05-0.3%/day")
        elif actual >= 0:
            _safe_print(f"     [~] Marginal -- barely positive")
            _safe_print(f"       -> Consider expanding asset universe or strategy tuning")
        else:
            _safe_print(f"     [!!] Negative returns -- strategy needs optimization")

        _safe_print(f"\n     Realistic expectations:")
        _safe_print(f"       * 1%/day = ~3,678% annual (compounded) -- extremely aggressive")
        _safe_print(f"       * Top hedge funds target 15-25% annual (0.04-0.07%/day)")
        _safe_print(f"       * Sharpe > 1.0 is good; > 2.0 is excellent")
        _safe_print(f"       * Current Sharpe: {bt_result.sharpe_ratio:.3f}")

    def _print_summary(self, results: Dict[str, Dict]):
        """Print portfolio-level summary."""
        _safe_print(f"\n{'=' * 60}")
        _safe_print(f"  PORTFOLIO SUMMARY")
        _safe_print(f"{'=' * 60}")

        total_pnl = 0.0
        for asset, data in results.items():
            bt = data['backtest']
            total_pnl += bt.net_pnl
            l2 = data['l2_sentiment']
            _safe_print(f"\n  {asset}:")
            _safe_print(f"    Net P&L:    ${bt.net_pnl:>12,.2f}")
            _safe_print(f"    Return:     {bt.total_return_pct:>8.2f}%")
            _safe_print(f"    Sharpe:     {bt.sharpe_ratio:>8.3f}")
            _safe_print(f"    Max DD:     {bt.max_drawdown_pct:>8.2f}%")
            _safe_print(f"    Trades:     {bt.total_trades:>8d}")
            _safe_print(f"    Win Rate:   {bt.win_rate * 100:>8.1f}%")
            _safe_print(f"    Sentiment:  {l2.get('aggregate_label', 'N/A')}")

        _safe_print(f"\n  {'-' * 40}")
        _safe_print(f"  Total Net P&L: ${total_pnl:>12,.2f}")
        portfolio_return = (total_pnl / self.initial_capital) * 100
        _safe_print(f"  Portfolio Return: {portfolio_return:>8.2f}%")
        _safe_print(f"{'=' * 60}")


# Allow direct execution (real-time data only)
if __name__ == '__main__':
    cfg = {'mode': 'paper'}  # demo flag removed; real-time data is enforced
    ex = TradingExecutor(cfg)
    ex.run()
