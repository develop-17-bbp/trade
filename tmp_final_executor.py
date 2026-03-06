import sys

file_path = r'c:\Users\convo\trade\src\trading\executor.py'

new_run_live = r'''
    def _run_live(self):
        """Main execution loop for Live/Testnet simulation."""
        _safe_print(f"\n  [EXECUTION] Entering real-time monitoring loop...")
        _safe_print(f"  Press Ctrl+C to exit.")
        _safe_print("-" * 60)

        while True:
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

            for asset in self.assets:
                symbol = f"{asset}/USDT"
                _safe_print(f"\n  [LIVE] Starting bar {self.iteration_count} (Style: {self.style.value})")

                ohlcv_data = self._fetch_data(symbol)
                if ohlcv_data is None: continue

                # Fetch Institutional & Microstructure Signals (Top 30 Institutional Signals)
                ext_feats = {}
                try:
                    # 1. Derivatives (Funding, OI Change/Divergence)
                    derivatives = self.price_source.fetch_derivatives_data(symbol, ohlcv_data['closes'][-1])
                    ext_feats.update(derivatives)
                    
                    # 2. Microstructure (Imbalance, Slope, Walls, Iceberg/Spoofing)
                    ob_data = self.price_source.exchange.fetch_order_book(symbol)
                    l2_metrics = self.microstructure.analyze_order_book(ob_data)
                    ext_feats.update(l2_metrics)

                    # 3. Institutional & On-Chain (Macro, Flows, Liquidations)
                    ext_inst = self.institutional.get_all_institutional(asset)
                    ext_feats.update(ext_inst)

                    # 4. Volatility Regime
                    vol_metrics = self.regime_detector.detect_regime(ohlcv_data['closes'], ohlcv_data['highs'], ohlcv_data['lows'])
                    ext_feats.update(vol_metrics)
                except Exception:
                    pass

                # Fetch sentiment
                headlines, h_timestamps, h_sources, h_events = self._fetch_sentiment(asset)

                # Generate signals
                strategy_result = self.strategy.generate_signals(
                    prices=ohlcv_data['closes'],
                    highs=ohlcv_data['highs'],
                    lows=ohlcv_data['lows'],
                    volumes=ohlcv_data['volumes'],
                    headlines=headlines if headlines else None,
                    headline_timestamps=h_timestamps if h_timestamps else None,
                    headline_sources=h_sources if h_sources else None,
                    headline_event_types=h_events if h_events else None,
                    external_features=ext_feats if ext_feats else None,
                    agentic_bias=self.agentic_bias
                )

                signals = strategy_result['signals']
                last_signal = signals[-1] if signals else 0
                
                # Model Drift Check
                if last_signal != 0:
                    last_feats = strategy_result.get('l1_data', {}).get('features', [{}])[-1]
                    if self._check_model_drift(last_feats):
                        _safe_print("  [DRIFT-CHECK] Skipping trade due to significant model drift.")
                        last_signal = 0

                _safe_print(f"     Latest signal: {last_signal:+d}")

                # Execute order
                if last_signal != 0:
                    self._execute_autonomous_trade(asset, symbol, last_signal, ohlcv_data['closes'][-1])

            _safe_print(f"\n  [SLEEP] Waiting {self.poll_interval}s for next bar...")
            time.sleep(self.poll_interval)
'''

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "def _run_live(self):" in line:
        start_idx = i
    if start_idx != -1 and "def _execute_autonomous_trade" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    content = "".join(lines[:start_idx]) + new_run_live + "\n" + "".join(lines[end_idx:])
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated _run_live in TradingExecutor")
else:
    print(f"Error: Could not find _run_live boundaries ({start_idx}, {end_idx})")
