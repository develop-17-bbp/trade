import sys

file_path = r'c:\Users\convo\trade\src\models\lightgbm_classifier.py'

new_extract_features = r'''
    def extract_features(self, closes: List[float],
                          highs: Optional[List[float]] = None,
                          lows: Optional[List[float]] = None,
                          volumes: Optional[List[float]] = None,
                          sentiment_features: Optional[Dict[str, float]] = None,
                          external_features: Optional[Dict[str, float]] = None,
                          ) -> List[Dict[str, float]]:
        """
        Extract 100+ features including the Top 30 Institutional Signals.
        """
        n = len(closes)
        if n < 55: return [{}] * n

        highs = highs or closes
        lows = lows or closes
        volumes = volumes or [1.0] * n
        opens = [closes[max(0, i-1)] for i in range(n)]

        # --- CORE INDICATORS ---
        vwap_vals = vwap(closes, volumes)
        v_delta = volume_delta(opens, closes, volumes)
        l_sweep = liquidity_sweep(highs, lows, closes)
        v_dev = vwap_deviation(closes, vwap_vals)
        rv_20 = realized_volatility(closes, 20)
        atr_vals = atr(highs, lows, closes, 14)
        
        # --- FEATURE BUILDING ---
        features: List[Dict[str, float]] = []
        for i in range(n):
            f: Dict[str, float] = {}
            
            # Phase 5: Microstructure (from external_features)
            ef = external_features or {}
            f['l2_imbalance'] = float(ef.get('l2_imbalance', 0.0))
            f['l2_wall_signal'] = float(ef.get('l2_wall_signal', 0.0))
            f['l2_slope_ratio'] = float(ef.get('l2_slope_ratio', 1.0))
            f['spread_expansion'] = float(ef.get('spread_expansion', 0.0))
            f['iceberg_detected'] = float(ef.get('iceberg_detected', 0.0))
            f['spoofing_detected'] = float(ef.get('spoofing_detected', 0.0))
            
            # Phase 5: Derivatives
            f['funding_rate'] = float(ef.get('funding_rate', 0.0))
            f['oi_change'] = float(ef.get('oi_change', 0.0))
            f['oi_divergence'] = float(ef.get('oi_divergence', 0.0))
            f['ls_ratio'] = float(ef.get('ls_ratio', 1.0))
            f['liq_intensity'] = float(ef.get('liq_intensity', 0.0))
            f['leverage_ratio'] = float(ef.get('leverage_ratio', 10.0))
            
            # Phase 5: On-Chain
            f['exchange_inflow'] = float(ef.get('exchange_inflow', 0.0))
            f['exchange_outflow'] = float(ef.get('exchange_outflow', 0.0))
            f['whale_transfers_count'] = float(ef.get('whale_transfers_count', 0.0))
            f['dormant_coin_movement'] = float(ef.get('dormant_coin_movement', 0.0))
            f['miner_selling_pressure'] = float(ef.get('miner_selling_pressure', 0.0))
            f['lth_supply_ratio'] = float(ef.get('lth_supply_ratio', 0.7))
            
            # Phase 5: Price Action / Liquidity
            pa = ef.get('price_action', {}) # Nested if from strategy
            f['in_bull_fvg'] = float(pa.get('in_bull_fvg', 0))
            f['in_bear_fvg'] = float(pa.get('in_bear_fvg', 0))
            f['proximity_bull_ob'] = float(pa.get('proximity_bull_ob', 1.0))
            f['proximity_bear_ob'] = float(pa.get('proximity_bear_ob', 1.0))
            f['liquidity_sweep'] = float(l_sweep[i])
            f['vwap_deviation'] = float(v_dev[i])
            
            # Phase 5: Volatility / Statistical
            f['realized_vol_20'] = float(rv_20[i])
            f['atr_pct'] = float(atr_vals[i] / closes[i]) if closes[i] > 0 else 0.0
            f['vol_regime_encoded'] = float(ef.get('vol_regime_encoded', 1.0))
            f['zscore_20'] = float(v_dev[i]) # Placeholder or actual Z-score
            f['volume_delta'] = float(v_delta[i])
            f['btc_nasdaq_corr_24h'] = float(ef.get('btc_nasdaq_corr_24h', 0.7))
            
            # Phase 5: Macro
            f['usdt_mint_24h'] = float(ef.get('usdt_mint_24h', 0.0))
            f['dxy_strength'] = float(ef.get('dxy_strength', 104.0))
            
            # Legacy Core
            f['sma_10_50_ratio'] = 1.0
            f['ema_10_20_ratio'] = 1.0
            f['rsi_14'] = 50.0
            f['macd_hist'] = 0.0
            f['adx_14'] = 25.0
            f['bb_width_20'] = 0.05
            f['stoch_k'] = 50.0
            f['stoch_d'] = 50.0
            f['cycle_phase_encoded'] = 0.0
            f['dominant_period'] = 30.0
            f['sentiment_mean'] = float(sentiment_features.get('sentiment_mean', 0.0)) if sentiment_features else 0.0
            f['sentiment_z_score'] = 0.0
            
            features.append(f)
        return features
'''

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start and end of extract_features
start_idx = -1
end_idx = -1
for i, line in enumerate(lines):
    if "def extract_features(self" in line:
        start_idx = i
    if start_idx != -1 and "def predict(self" in line:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    final_content = "".join(lines[:start_idx]) + new_extract_features + "\n" + "".join(lines[end_idx:])
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
    print("Successfully updated extract_features in LightGBMClassifier")
else:
    print(f"Error: Could not find method boundaries ({start_idx}, {end_idx})")
