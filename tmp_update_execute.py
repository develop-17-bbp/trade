import sys

file_path = r'c:\Users\convo\trade\src\trading\executor.py'

new_execute = r'''
    def _execute_autonomous_trade(self, asset: str, symbol: str, signal: int, current_price: float, 
                                  strategy_result: Optional[Dict] = None, ext_feats: Optional[Dict] = None):
        """
        Institutional-grade autonomous execution with full Audit/Compliance Logging.
        """
        try:
            _safe_print(f"  [PHASE 5] Evaluating trade for {asset} (Signal: {signal})")
            
            # 1. ALPHA ENRICHMENT: Strategy Weighting based on Regime
            perf_metrics = self._get_asset_performance_metrics()
            onchain_data = {asset: asdict(self.on_chain_portfolio.compute_on_chain_signal(asset))}
            regime = ext_feats.get('liquidity_regime', 'NORMAL') if ext_feats else 'NORMAL'
            
            allocations = self.portfolio_allocator.allocate_portfolio(
                assets=self.assets, performance_metrics=perf_metrics, 
                onchain_data=onchain_data, regime=regime
            )
            
            allocation = allocations.get(asset)
            if not allocation: return

            side = "buy" if signal > 0 else "sell"
            pos_size_pct = allocation.position_size_pct
            
            # 2. RISK MANAGEMENT: Update VaR & Tail Risk
            import numpy as np
            rets = perf_metrics.get("returns", [0.0]*50)
            vol = float(np.std(rets)) * np.sqrt(252) if rets else 0.05
            skew = float(np.mean((np.array(rets) - np.mean(rets))**3) / (np.std(rets)**3)) if len(rets)>30 and np.std(rets)>0 else 0.0
            
            self.risk_manager.update_market_conditions(
                volatility=vol, correlations={}, 
                liquidation_risk=onchain_data[asset].get("liquidation_risk", 0.0),
                realized_skewness=skew
            )
            
            heat = sum(a.position_size_pct for a in allocations.values())
            allowed, reason = self.risk_manager.check_trade_allowed(asset, pos_size_pct, heat)
            
            if not allowed:
                _safe_print(f"  [PHASE 5] Risk block: {reason}")
                return

            # 3. MICROSTRUCTURE & SLIPPAGE HURDLE
            qty = (pos_size_pct * self.initial_capital) / current_price
            try:
                ob = self.price_source.exchange.fetch_order_book(symbol)
                slippage = self.microstructure.estimate_slippage(ob, qty, side)
                if slippage > 0.003: # 30 bps
                    _safe_print(f"  [PHASE 5] SKIPPED: Slippage {slippage:.2%} exceeds HFT hurdle.")
                    return
            except Exception: pass

            # 4. EXECUTION & AUDIT LOGGING
            _safe_print(f"  [PHASE 5] Executing {side} {qty:.6f} {asset}. (Kelly: {allocation.kelly_fraction:.3f})")
            
            optimal_type = "limit" if pos_size_pct > 0.05 else "market"
            execution_res = self.execution_router.execute_order(
                symbol=symbol, side=side, quantity=qty, 
                price=current_price if optimal_type == "limit" else None, 
                order_type=optimal_type
            )

            # --- COMPLIANCE / AUDIT LOGGING ---
            if execution_res and 'id' in execution_res:
                feat_snap = strategy_result.get('l1_data', {}).get('features', [{}])[-1] if strategy_result else {}
                self.journal.log_trade(
                    asset=asset, side=side, quantity=qty, price=current_price,
                    regime=regime, strategy_name="HybridAlpha_v5",
                    confidence=allocation.onchain_confidence,
                    reasoning=f"Autonomous Execution Mode. Regime={regime}",
                    order_id=execution_res['id'],
                    feature_vector=feat_snap,
                    model_signal=signal
                )
                self.risk_manager.register_trade_open(asset, signal, current_price, pos_size_pct)

        except Exception as e:
            _safe_print(f"  [ERROR] Execution failed: {e}")
'''

with open(file_path, 'r', encoding='utf-8') as f:
    orig_lines = f.readlines()

start_idx = -1
end_idx = -1
for i, line in enumerate(orig_lines):
    if "def _execute_autonomous_trade(self" in line:
        start_idx = i
    if start_idx != -1 and "def _check_model_drift(" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    content = "".join(orig_lines[:start_idx]) + new_execute + "\n" + "".join(orig_lines[end_idx:])
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated _execute_autonomous_trade in TradingExecutor")
else:
    print(f"Error: Could not find method boundaries")
