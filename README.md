# AI-Driven Institutional Crypto Trading System v6.5

An autonomous, multi-layer trading intelligence platform for BTC/ETH/AAVE. Combines LightGBM, PatchTST Transformers, Reinforcement Learning, FinBERT NLP, On-Chain Analytics, a 12-Agent Bayesian Consensus Engine, and LLM-driven Agentic Reasoning into a unified 9-layer decision architecture with real-time dashboards.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Complete Data Flow: One Trading Iteration](#complete-data-flow-one-trading-iteration)
- [Layer-by-Layer Deep Dive](#layer-by-layer-deep-dive)
  - [L1: Quantitative Engine](#l1-quantitative-engine)
  - [L2: Sentiment Intelligence](#l2-sentiment-intelligence)
  - [L3: Risk Fortress](#l3-risk-fortress)
  - [L4: Signal Fusion (Meta-Controller)](#l4-signal-fusion-meta-controller)
  - [L5: Smart Execution](#l5-smart-execution)
  - [L6: Multi-Agent Intelligence + LLM Strategist](#l6-multi-agent-intelligence--llm-strategist)
  - [L7-L9: Learning, Memory, Evolution](#l7-l9-learning-memory-evolution)
- [12-Agent Consensus System](#12-agent-consensus-system)
- [Risk Management System](#risk-management-system)
- [Model Retraining Pipeline](#model-retraining-pipeline)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [Environment Variables](#environment-variables)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Disclaimer](#disclaimer)

---

## System Architecture

The system uses a **9-layer cascading intelligence pipeline**. Each layer processes a unique signal dimension, and all layers feed into a unified meta-controller for trade execution.

| Layer | Name | Function | Key Files |
|:------|:-----|:---------|:----------|
| **L1** | Quantitative Engine | 80+ feature extraction, LightGBM/PatchTST/RL ensemble | `models/lightgbm_classifier.py`, `ai/patchtst_model.py`, `ai/reinforcement_learning.py` |
| **L2** | Sentiment Intelligence | FinBERT NLP + multi-source news aggregation | `ai/sentiment.py`, `ai/finbert_service.py`, `data/news_fetcher.py` |
| **L3** | Risk Fortress | Position sizing, drawdown protection, VPIN toxicity guard | `risk/manager.py`, `risk/vpin_guard.py`, `risk/profit_protector.py` |
| **L4** | Signal Fusion | Meta-controller combining L1-L3 with Bayesian weights | `trading/meta_controller.py`, `trading/signal_combiner.py` |
| **L5** | Smart Execution | TWAP/VWAP order routing, slippage estimation | `execution/router.py`, `execution/bridge.py` |
| **L6** | Agent Intelligence | 12-agent Bayesian consensus + LLM macro reasoning | `agents/orchestrator.py`, `agents/combiner.py`, `ai/agentic_strategist.py` |
| **L7** | Advanced Learning | Regime detection, pattern recognition, adaptive tuning | `ai/advanced_learning.py`, `ai/math_injection.py` |
| **L8** | Tactical Memory | Episodic trade memory for contextual pattern recall | `ai/memory_vault.py` |
| **L9** | Evolution Portal | Model versioning, scheduled retraining, self-improvement | `models/scheduled_retrain.py`, `models/auto_retrain.py` |

```
                        BINANCE (CCXT)          NewsAPI / CryptoPanic / Reddit
                             |                            |
                    +--------v--------+          +--------v--------+
                    | L1: Quantitative |          | L2: Sentiment   |
                    | - LightGBM      |          | - FinBERT NLP   |
                    | - PatchTST      |          | - Keyword Rules |
                    | - RL Agent      |          | - Time Decay    |
                    | - 80+ Features  |          | - Source Weights |
                    +--------+--------+          +--------+--------+
                             |                            |
                    +--------v----------------------------v--------+
                    |          L4: Meta-Controller                  |
                    |   Bayesian weighted ensemble of L1+L2+L3     |
                    +--------+-------------------------------------+
                             |
            +----------------v----------------+
            |   L3: Risk Fortress (VETO)      |
            |   - Daily loss limit (3%)       |
            |   - Max drawdown (10%)          |
            |   - VPIN toxicity gate          |
            |   - Volatility regime filter    |
            |   - Profit protection           |
            +----------------+----------------+
                             |
          +------------------v------------------+
          |  L6: Multi-Agent Overlay (12 Agents)|
          |  1. Data Integrity Pre-Gate         |
          |  2. 10 Parallel Analysis Agents     |
          |  3. Bayesian Consensus (Combiner)   |
          |  4. Decision Auditor Post-Gate      |
          +------------------+------------------+
                             |
                    +--------v--------+
                    | L5: Execution   |
                    | - TWAP/VWAP     |
                    | - Slippage Est. |
                    | - Order Routing |
                    +--------+--------+
                             |
                    +--------v--------+
                    | EXCHANGE        |
                    | (Paper/Testnet/ |
                    |  Live)          |
                    +-----------------+
```

---

## Complete Data Flow: One Trading Iteration

Every `poll_interval` seconds (default: 30s), the system executes one full cycle through all 9 layers. Here is exactly what happens in each iteration, in order:

### Phase 1: Timing and Event Guard

```
executor.run() -> _run_live() while True loop
  |
  +-> iteration_count += 1
  +-> event_guard.is_risk_high()  -- check for macro risk events
  |     If high risk -> sleep 60s, skip iteration
  |
  +-> Every 6 iterations: _perform_agentic_review()
        L6 LLM reflects on recent market state and adjusts config
```

### Phase 2: Data Acquisition (Per Asset)

```
For each asset in [BTC, ETH, AAVE]:
  |
  +-> 2a. Fetch OHLCV
  |     price_source.fetch_ohlcv(symbol, timeframe='1h', limit=200)
  |     -> Extract: closes[], highs[], lows[], volumes[]
  |     -> OHLCVValidator checks data integrity
  |     -> Latency guard: skip if API response > threshold
  |
  +-> 2b. Fetch Derivatives & Microstructure
  |     price_source.fetch_derivatives_data(symbol)
  |       -> funding_rate, open_interest, cross_exchange_dislocation
  |     microstructure.analyze_order_book(order_book)
  |       -> bid_depth, ask_depth, imbalance, vpin_50, l2_wall_signal
  |     microstructure.detect_liquidity_regime(bid, ask)
  |       -> NORMAL | DRY | FLASH | TSUNAMI
  |
  +-> 2c. Fetch Sentiment & News
        news.fetch_all(asset)
          -> NewsAPI -> CryptoPanic -> Reddit -> CoinGecko (fallback)
          -> Returns: headlines[], timestamps[], sources[], event_types[]
```

### Phase 3: Signal Generation (L1 + L2 + L3 Fusion)

```
strategy.generate_signals(prices, highs, lows, volumes, headlines, ext_feats)
  |
  +-> L2: FinBERT Sentiment
  |     finbert.get_sentiment_features(headlines)
  |       -> sentiment_mean, sentiment_z_score, bullish_ratio, bearish_ratio
  |     sentiment.analyze(headlines, timestamps, sources, event_types)
  |       -> aggregate_score in [-1, +1], aggregate_label, confidence
  |
  +-> L1: Feature Engineering (80+ features)
  |     classifier.extract_features(closes, highs, lows, volumes, sentiment_features, ext_feats)
  |       -> Microstructure: VPIN, order book imbalance, spoofing detection
  |       -> Derivatives: funding rate, OI change, liquidation cascade prob
  |       -> On-Chain: exchange flows, whale clusters, LTH ratios
  |       -> Technical: RSI, MACD, Bollinger, ATR, ADX, Stochastic
  |       -> Volatility: realized vol, EWMA vol, GARCH vol, regime encoding
  |       -> Trend: SMA ratios, EMA slopes, momentum
  |
  +-> L1: Model Ensemble Predictions
  |     classifier.predict(features)       -> (class, probability) per bar
  |     rl_agent.predict(features)         -> (action, action_prob) per bar
  |     patch_tst.predict(prices)          -> {prob_up, prob_shock, regime}
  |
  +-> L4: Meta-Controller Arbitration
  |     meta_controller.arbitrate(lgb, rl, patchtst, finbert, agentic_bias)
  |       -> Weighted ensemble vote
  |       -> Output: final_direction in {-1, 0, +1}, final_confidence in [0, 1]
  |
  +-> L3: Risk Gating
  |     risk_manager.is_trade_safe(price, direction, atr, balance)
  |       -> If unsafe: direction = 0 (suppress signal)
  |
  +-> Post-checks:
        - Model drift detection -> suppress if features have shifted
        - VPIN toxicity gate -> suppress if vpin > 0.8
```

### Phase 4: Multi-Agent Intelligence Overlay

```
agent_orchestrator.run_cycle(quant_state, raw_signal, sentiment, ext_feats, ...)
  |
  +-> Step 1: Data Integrity Validator (Pre-Gate)
  |     Sanitize all inputs, check NaN/Inf, range violations
  |     quality_score < 0.3 -> VETO (skip all agents)
  |
  +-> Step 2: 10 Analysis Agents in Parallel (ThreadPoolExecutor)
  |     Each agent returns AgentVote(direction, confidence, reasoning)
  |     Confidence scaled by data quality score
  |
  +-> Step 3: Bayesian Weighted Consensus (AgentCombiner)
  |     S_long  = SUM(weight_i * reliability_i * confidence_i) for LONG votes
  |     S_short = SUM(weight_i * reliability_i * confidence_i) for SHORT votes
  |     P_long  = S_long / total -> compare to consensus_threshold (0.55)
  |     Regime-adaptive weight multipliers applied
  |     -> Consensus levels: STRONG (>75%) | MODERATE (>60%) | WEAK (>55%) | CONFLICT
  |
  +-> Step 4: Decision Auditor (Post-Gate)
  |     Validates: indicator alignment, agent contradictions, crisis checks
  |     Can BLOCK (veto) or DEFER (set to flat)
  |
  +-> Apply overlay to raw signal:
        - If VETO -> direction = 0 (unless testnet force_trade)
        - If agent direction != raw AND confidence > 0.6 -> override
        - Blend: (1 - blend_weight) * existing + blend_weight * agent = final
```

### Phase 5: Trade Execution

```
_execute_autonomous_trade(asset, signal, price, strategy_result, enhanced_decision)
  |
  +-> 5a. Ensemble Confidence Score
  |     ensemble = (lgb_score + patchtst_score + rl_score) / 3
  |
  +-> 5b. Agent Overlay Blending
  |     blended = (1 - 0.60) * pipeline_score + 0.60 * agent_score
  |
  +-> 5c. Final Direction
  |     Testnet: direction = 1 if ensemble > 0.48 (aggressive)
  |     Live:    direction = 1 if ensemble > 0.6 (conservative)
  |
  +-> 5d. VPIN Toxicity Veto (last gate)
  |     vpin.is_flow_toxic() -> block if toxic (unless force_trade)
  |
  +-> 5e. Profit Protection Gate
  |     profit_protector.should_enter_trade(trade_quality, balance)
  |       -> In profit: require >= 75% confidence
  |       -> Max P(loss) < 35%
  |       -> Expected value must be positive
  |
  +-> 5f. Portfolio Allocation
  |     portfolio_allocator.allocate_portfolio(assets, performance, on_chain, regime)
  |       -> position_size_pct per asset
  |
  +-> 5g. Smart Order Routing
  |     router.execute_advanced_order(symbol, side, qty, algo="TWAP"|"Direct")
  |     slippage_model.estimate_price_impact(qty, adv)
  |
  +-> 5h. LLM Per-Trade Analysis (L6)
  |     strategist.analyze_trade(asset, price, l1_signal, l2_sentiment, l3_risk)
  |       -> Natural language reasoning trace
  |
  +-> 5i. Journal & Risk Registration
        journal.log_trade(asset, side, qty, price, reasoning, ...)
        risk_manager.register_trade_open(asset, direction, price, size_pct)
        DashboardState().record_trade(...)
```

### Phase 6: Dashboard Telemetry and Sleep

```
Push to DashboardState:
  - Asset snapshot (signal, sentiment, factors, attribution, veto)
  - Layer status (L1-L9 health, progress, metrics)
  - Performance metrics (uplift, win rate, edge)
  - Agent overlay state (votes, consensus, quality)
  - Model prediction accuracy tracking
  - OHLC history, sentiment history, decision history

Sleep(poll_interval) -> loop
```

---

## Layer-by-Layer Deep Dive

### L1: Quantitative Engine

**Files:** `src/models/lightgbm_classifier.py`, `src/ai/patchtst_model.py`, `src/ai/reinforcement_learning.py`

Three models vote on trade direction:

| Model | Architecture | Output | Weight |
|:------|:------------|:-------|:-------|
| **LightGBM** | Gradient-boosted 3-class classifier (1200 trees) | LONG / FLAT / SHORT + confidence | ~33% |
| **PatchTST** | Transformer with overlapping patches (seq_len=400) | prob_up, prob_shock, regime | ~33% |
| **RL Agent** | PPO policy optimization | BUY / SELL / HOLD + action_prob | ~33% |

**Feature Vector (80+ features across 7 categories):**

| Category | Count | Examples |
|:---------|:------|:--------|
| Microstructure | 7 | l2_imbalance, l2_wall_signal, spoofing_detected, vpin_50 |
| Derivatives | 9 | funding_rate, funding_momentum, oi_change, liq_cascade_prob |
| On-Chain | 10 | exchange_inflow/outflow, whale_cluster, lth_spent_ratio |
| Price Action | 6 | bull/bear_fvg, liquidity_sweep, vwap_deviation |
| Volatility | 7 | realized_vol_20, atr_pct, vol_regime_encoded, zscore_20 |
| Core Technical | 12 | rsi_14, macd_hist, adx_14, bb_width_20, stoch_k/d |
| Sentiment | 6 | sentiment_mean, sentiment_z_score, bullish_ratio |

**3-Class Prediction:**
- Class 2 (LONG): confidence > 0.65 AND score > 0.15
- Class 1 (FLAT): confidence < 0.65 OR score in neutral band
- Class 0 (SHORT): confidence > 0.65 AND score < -0.15

**Confidence Calibration:** Raw LightGBM probabilities are calibrated via Platt Scaling to produce realistic [0, 1] confidence scores.

**Rule-Based Fallback (when no model loaded):**
When `_lgb_model is None`, the classifier uses a weighted scoring system:
- Trend signals (30%): price vs SMA50, SMA cross, EMA slope x ADX
- Momentum (25%): RSI, Stochastic, MACD, ROC
- Mean-Reversion (15%): Bollinger Band position, Z-score
- Volatility (10%): regime conditioning (low = bullish, high = bearish)
- Sentiment (15%): sentiment_mean x confidence
- Cycle (5%): cycle momentum

---

### L2: Sentiment Intelligence

**Files:** `src/ai/sentiment.py`, `src/ai/finbert_service.py`, `src/data/news_fetcher.py`

**Two-Tier Architecture:**

| Tier | Model | Latency | Accuracy | Fallback |
|:-----|:------|:--------|:---------|:---------|
| Tier 1 | Rule-based keyword scoring (60+ pos, 50+ neg terms) | <1ms | Moderate | Always available |
| Tier 2 | ProsusAI/FinBERT transformer | ~10s/batch | High | Falls back to Tier 1 |

**News Source Priority:**

| Priority | Source | Auth Required | Notes |
|:---------|:-------|:-------------|:------|
| 1 | NewsAPI | API key | High quality general + crypto |
| 2 | CryptoPanic | Token | Real-time crypto-specific |
| 3 | Reddit | None | r/cryptocurrency, r/Bitcoin, r/ethereum |
| 4 | CoinGecko | None | Trending fallback (used if <50% fill) |

**Time-Decayed Aggregation:**
```
S_final = SUM(confidence_i * source_weight_i * score_i * decay_i) / SUM(confidence_i * source_weight_i)
where decay_i = e^(-0.001 * age_seconds)  -- 50% decay in ~11 minutes
```

**Source Credibility Weights:** NewsAPI (0.9) > CryptoPanic (0.8) > CoinGecko (0.7) > Reddit (0.5)

**Event Multipliers:** regulatory (1.8x), hack (2.0x), etf (1.5x), macro (1.3x), exchange (1.4x), adoption (1.2x)

**Output Features (injected into L1 feature vector):**
`sentiment_mean`, `sentiment_std`, `sentiment_z_score`, `bullish_ratio`, `bearish_ratio`, `avg_confidence`, `max_negative_score`, `sentiment_momentum`

---

### L3: Risk Fortress

**Files:** `src/risk/manager.py`, `src/risk/dynamic_manager.py`, `src/risk/profit_protector.py`, `src/risk/vpin_guard.py`

**10 Sequential Risk Checks (in order):**

| # | Check | Threshold | Action |
|:--|:------|:----------|:-------|
| 1 | System halted? | `_halted == True` | VETO |
| 2 | Daily loss limit | daily_loss >= 3% | HALT + VETO |
| 3 | Max drawdown | drawdown >= 10% | HALT + VETO |
| 4 | Extreme volatility | vol_regime == EXTREME | VETO |
| 5 | Trade frequency cap | daily_trades >= 20 | BLOCK |
| 6 | Post-loss cooldown | < 30 min since last loss | BLOCK |
| 7 | Position size limit | > 2% of account balance | REDUCE |
| 8 | Portfolio exposure | total > 20% of balance | REDUCE or BLOCK |
| 9 | Volatility-scaled sizing | LOW:1.2x, MED:1.0x, HIGH:0.5x, EXTREME:0.0x | REDUCE |
| 10 | Signal strength scaling | <0.3:0.5x, 0.3-0.7:0.75x, >0.7:1.0x | REDUCE |

**Circuit Breaker:** Triggers when `(peak_equity - current_equity) / peak_equity >= max_drawdown_pct`. Halts all trading. Manual `unhalt()` required.

**ATR-Based Stop/TP:**
```
Stop Loss   = entry +/- atr_stop_mult * ATR  (default: 3.5x ATR)
Take Profit = entry +/- atr_tp_mult * ATR    (default: 2.5x ATR)
```
Features trailing stops and breakeven locks after partial TP.

**Profit Protector:** When account is in profit, tightens entry requirements:
- Requires >= 75% confidence (vs 60% at breakeven)
- Max P(loss) must be < 35%
- Expected value must be positive
- Position size scales with confidence x win_rate x risk_reward_ratio

---

### L4: Signal Fusion (Meta-Controller)

**Files:** `src/trading/meta_controller.py`, `src/trading/signal_combiner.py`

The meta-controller combines L1 (LightGBM + PatchTST + RL), L2 (FinBERT sentiment), and L3 (risk assessment) into a single direction + confidence output.

```
meta_controller.arbitrate(
    lgb_class, lgb_conf,      # LightGBM vote
    rl_action, rl_prob,       # RL Agent vote
    features,                 # Raw feature dict
    finbert_score,            # L2 sentiment [-1, +1]
    patch_result,             # PatchTST {prob_up, regime}
    agentic_bias              # L6 prior influence
) -> (direction, confidence, position_scale)
```

Attribution weights are dynamically computed and published to the dashboard.

---

### L5: Smart Execution

**Files:** `src/execution/router.py`, `src/execution/bridge.py`, `src/execution/failover.py`

| Algorithm | When Used | Mechanism |
|:----------|:----------|:----------|
| **Direct** | Small orders (qty < 0.05) | Single market order |
| **TWAP** | Large orders | Time-weighted splits over N intervals |
| **VWAP** | Volume-sensitive | Volume-weighted execution schedule |

**Slippage Model:** Square-root price impact estimation using order book depth and estimated ADV.

**Execution Flow:**
1. Calculate position size from portfolio allocation
2. Fetch real-time order book
3. Estimate slippage via impact model
4. Select execution algorithm
5. Submit via exchange bridge (CCXT)
6. Log execution event with order ID

---

### L6: Multi-Agent Intelligence + LLM Strategist

**Files:** `src/agents/orchestrator.py`, `src/agents/combiner.py`, `src/ai/agentic_strategist.py`

See [12-Agent Consensus System](#12-agent-consensus-system) below for the full agent breakdown.

**LLM Strategist (Agentic Reasoner):**

Provider auto-detection chain: Google Gemini -> Ollama local -> Rule-based fallback.

The strategist provides two functions:
1. **Periodic Review** (every 6 iterations): Analyzes market regime, suggests config overrides (whitelisted keys with clamped ranges), adjusts macro bias
2. **Per-Trade Analysis**: Generates natural language reasoning trace for each executed trade

**Hallucination Safeguards:**
- Tier 1: Pydantic schema validation of LLM output
- Tier 2: Fact-checking (e.g., "TRENDING" claim vs actual ATR)
- Tier 3: Confidence calibration via historical accuracy

**Config Override Whitelist (clamped ranges):**
```
risk.max_position_size_pct:  [0.1, 5.0]
risk.daily_loss_limit_pct:   [0.5, 5.0]
signal.min_confidence:       [0.3, 0.95]
l1.short_window:             [3, 20]
l1.long_window:              [10, 50]
```

---

### L7-L9: Learning, Memory, Evolution

| Layer | File | Function |
|:------|:-----|:---------|
| **L7** | `ai/advanced_learning.py`, `ai/math_injection.py` | HMM regime detection, Kalman filter, OU process analysis, GARCH volatility |
| **L8** | `ai/memory_vault.py` | Episodic trade memory (ChromaDB vector store). Finds similar past scenarios to adjust confidence +/- 10-20% |
| **L9** | `models/scheduled_retrain.py`, `models/auto_retrain.py` | Weekly model retraining with Optuna hyperparameter optimization, walk-forward validation, model versioning |

---

## 12-Agent Consensus System

The multi-agent overlay runs a strict 4-step pipeline on every iteration:

### Pipeline Steps

```
Step 1: Data Integrity Validator (PRE-GATE)
  -> Sanitize inputs, check NaN/Inf, detect anomalies
  -> quality_score < 0.3 = VETO entire cycle
     |
Step 2: 10 Analysis Agents (PARALLEL via ThreadPoolExecutor)
  -> Each returns AgentVote(direction, confidence, reasoning)
  -> Timeout: 5s per agent
     |
Step 3: Bayesian Weighted Consensus (AgentCombiner)
  -> Confidence-weighted direction scoring
  -> Regime-adaptive multipliers
  -> LossPreventionGuardian has priority veto
     |
Step 4: Decision Auditor (POST-GATE)
  -> Validates indicator alignment, flags contradictions
  -> Can BLOCK or DEFER decisions
  -> Can only DOWNGRADE confidence, never upgrade
```

### All 12 Agents

| Agent | File | Purpose | Special Authority |
|:------|:-----|:--------|:-----------------|
| `data_integrity` | `data_integrity_validator.py` | Pre-gate: sanitize data, detect NaN/Inf/anomalies | Veto on quality < 0.3 |
| `market_structure` | `market_structure_agent.py` | Hurst exponent + Kalman filter + order book regime | - |
| `regime_intelligence` | `regime_intelligence_agent.py` | HMM + GARCH macro regime (bull/bear/crisis/sideways) | - |
| `mean_reversion` | `mean_reversion_agent.py` | Ornstein-Uhlenbeck stationarity + RSI/BB extremes | - |
| `trend_momentum` | `trend_momentum_agent.py` | ADX + MACD + EMA/SMA + Kalman + Hurst confirmation | - |
| `risk_guardian` | `risk_guardian_agent.py` | MC VaR + EVT tail + Hawkes + VPIN + drawdown score | - |
| `sentiment_decoder` | `sentiment_decoder_agent.py` | FinBERT + Fear/Greed + whale + funding rate fusion | - |
| `trade_timing` | `trade_timing_agent.py` | Hawkes intensity + alpha freshness + OU z-score timing | WAIT signal defers |
| `portfolio_optimizer` | `portfolio_optimizer_agent.py` | Kelly criterion + correlation + daily PnL targets | - |
| `pattern_matcher` | `pattern_matcher_agent.py` | Trade history streak analysis + LSTM predictions | - |
| `loss_prevention` | `loss_prevention_guardian.py` | Daily PnL modes: NORMAL -> CAUTION -> PRESERVATION -> HALT | **Absolute VETO power** |
| `decision_auditor` | `decision_auditor.py` | Post-gate: validate, flag contradictions, block/defer | Can BLOCK or DEFER |

### Loss Prevention Modes

| Mode | Trigger | Effect |
|:-----|:--------|:-------|
| NORMAL | daily_pnl > -0.5% | Full trading |
| CAUTION | daily_pnl <= -0.5% | Reduced position sizes |
| PRESERVATION | daily_pnl >= +0.8% (protecting gains) | Only high-confidence trades |
| HALT | daily_pnl <= -1.0% | All trading stopped |

### Bayesian Consensus Scoring

```
For each direction (LONG, SHORT, FLAT):
  Score = SUM(agent_weight_i * reliability_i * confidence_i)

P(direction) = Score(direction) / Total_Score
Winner = direction with P > consensus_threshold (0.55)

Consensus Levels:
  STRONG:   P > 75%
  MODERATE: P > 60%
  WEAK:     P > 55%
  CONFLICT: P <= 55%  (no trade)
```

Agent weights are updated via Bayesian EMA (alpha=0.15) after each trade outcome.

---

## Risk Management System

### Veto Hierarchy (Any of These Blocks a Trade)

1. **RiskManager**: daily loss >= 3%, drawdown >= 10%, extreme volatility, system halted
2. **LossPreventionGuardian**: HALT mode, intraday drawdown > 2%, correlated with open position
3. **DecisionAuditor**: crisis + BUY combination, excessive contradictions
4. **VPIN Guard**: order flow toxicity > 0.8
5. **Profit Protector**: in profit + low confidence, P(loss) > 35%, negative expected value

### Position Sizing Flow

```
Base size = risk_per_trade_pct * account_balance / current_price

Adjustments:
  * volatility_scale:  LOW=1.2x, MED=1.0x, HIGH=0.5x, EXTREME=0.0x
  * signal_strength:   weak=0.5x, moderate=0.75x, strong=1.0x
  * profit_protector:  adaptive sizing based on trade quality
  * portfolio_alloc:   Black-Litterman allocation across assets

Final size = base * vol_scale * signal_scale * profit_adj * alloc_pct
```

---

## Model Retraining Pipeline

### Manual Retrain (Optuna Optimization)

```bash
python -m src.main --retrain
# or per-asset:
python -m src.models.auto_retrain --symbol AAVE/USDT --model-out models/lgbm_aave.txt --n-trials 50
```

**Process:** Fetch 10,000 bars -> extract features -> label by next-bar return -> Optuna 50 trials -> train best -> save

**Hyperparameter Search Space:**
- num_leaves: 20-100, learning_rate: 0.01-0.3, max_depth: 3-12
- min_child_samples: 10-100, subsample: 0.6-1.0, colsample: 0.6-1.0
- reg_alpha/lambda: 0.0-1.0

### Scheduled Weekly Retrain

```bash
python -m src.models.scheduled_retrain
# Cron: 0 3 * * 0  (every Sunday 3 AM)
```

**Safety Mechanisms:**
- Walk-forward validation (15% holdout on recent data)
- New model must beat old by >= 0.5% accuracy
- Old model backed up before replacement
- Parallel retraining across BTC/ETH/AAVE

**Config:**
```
lookback_bars: 15000 (~625 days hourly)
optuna_trials: 40
boost_rounds: 500
early_stopping: 30 rounds
min_accuracy_improvement: 0.5%
```

---

## Data Sources

### Exchange Data (CCXT)

| Data | Source | Timeframes |
|:-----|:-------|:-----------|
| OHLCV | Binance (live or testnet) | 1m, 5m, 15m, 1h, 4h, 1d |
| Order Book | Binance L2 | Real-time |
| Funding Rate | Binance Futures | 8h intervals |
| Open Interest | Binance Futures | Real-time |

### Free-Tier APIs (No Key Required)

| Source | Data |
|:-------|:-----|
| Blockchain.com | BTC hashrate, difficulty, tx count, miners revenue |
| Mempool.space | BTC mempool stats, fee rates |
| DefiLlama | DeFi TVL, stablecoin flows, DEX volumes |
| CoinGecko | Market dominance, trending coins, stablecoin supply |
| Alternative.me | Fear & Greed Index |
| Deribit | Options implied volatility |

### API-Key Data Sources (Optional)

| Source | Env Variable | Data | Cost |
|:-------|:-------------|:-----|:-----|
| NewsAPI | `NEWSAPI_KEY` | News headlines | Free tier: 100 req/day |
| CryptoPanic | `CRYPTOPANIC_TOKEN` | Crypto news feed | Free |
| Google Gemini | `REASONING_LLM_KEY` | LLM reasoning (L6) | Free tier available |
| Glassnode | `GLASSNODE_API_KEY` | LTH metrics, SOPR, NUPL | $29+/mo |
| CryptoQuant | `CRYPTOQUANT_API_KEY` | Miner flows, MVRV | $99/mo |
| Whale Alert | `WHALE_ALERT_API_KEY` | Large transfer tracking | Free/Pro |

**Graceful Degradation:** Every optional API has a fallback. Missing keys cause that layer to use defaults, not crash.

---

## Project Structure

```
trade/
├── src/
│   ├── main.py                          # Entry point (paper/testnet/live/retrain/dashboard)
│   ├── ai/
│   │   ├── agentic_strategist.py        # L6: LLM macro reasoning (Gemini/Ollama/rules)
│   │   ├── advanced_learning.py         # L7: Regime detection, pattern recognition
│   │   ├── finbert_service.py           # L2: FinBERT transformer sentiment
│   │   ├── llm_provider.py             # LLM router (cloud -> local -> rule-based)
│   │   ├── math_injection.py           # Mathematical constraint injection
│   │   ├── memory_vault.py             # L8: Episodic trade memory (ChromaDB)
│   │   ├── patchtst_model.py           # L1: PatchTST time-series transformer
│   │   ├── reinforcement_learning.py   # L1: PPO RL agent
│   │   └── sentiment.py                # L2: Multi-source sentiment aggregator
│   ├── agents/
│   │   ├── orchestrator.py             # 4-step agent pipeline coordinator
│   │   ├── combiner.py                 # Bayesian consensus aggregation
│   │   ├── data_integrity_validator.py # Pre-gate: data quality
│   │   ├── market_structure_agent.py   # Order book + Hurst + Kalman
│   │   ├── regime_intelligence_agent.py# HMM + GARCH regime classifier
│   │   ├── mean_reversion_agent.py     # OU process + BB + RSI extremes
│   │   ├── trend_momentum_agent.py     # ADX + MACD + trend confirmation
│   │   ├── risk_guardian_agent.py      # MC VaR + EVT + drawdown
│   │   ├── sentiment_decoder_agent.py  # Sentiment context interpretation
│   │   ├── trade_timing_agent.py       # Entry/exit timing optimization
│   │   ├── portfolio_optimizer_agent.py# Kelly + correlation + allocation
│   │   ├── pattern_matcher_agent.py    # Historical pattern detection
│   │   ├── loss_prevention_guardian.py # Daily PnL modes + absolute VETO
│   │   ├── polymarket_agent.py         # Prediction market signals
│   │   └── decision_auditor.py         # Post-gate: validate + block/defer
│   ├── api/
│   │   ├── dashboard_app.py            # Streamlit dashboard (port 8501)
│   │   ├── state.py                    # Dashboard state management
│   │   └── server.py                   # FastAPI REST server (port 8000)
│   ├── data/
│   │   ├── fetcher.py                  # Binance CCXT data fetcher
│   │   ├── news_fetcher.py             # NewsAPI + CryptoPanic + Reddit aggregator
│   │   ├── on_chain_fetcher.py         # Blockchain/Mempool/DefiLlama metrics
│   │   ├── institutional_fetcher.py    # Institutional-grade data feeds
│   │   ├── microstructure.py           # VPIN, order book analysis
│   │   ├── polymarket_fetcher.py       # Prediction market data
│   │   └── free_tier_integrations.py   # Free API aggregation
│   ├── execution/
│   │   ├── router.py                   # Smart order router (TWAP/VWAP/Direct)
│   │   ├── bridge.py                   # Exchange bridge + shared memory IPC
│   │   ├── failover.py                 # Multi-exchange failover
│   │   ├── twap_engine.py              # Time-weighted average price engine
│   │   └── vwap_engine.py              # Volume-weighted average price engine
│   ├── models/
│   │   ├── lightgbm_classifier.py      # L1: 3-class LightGBM (80+ features)
│   │   ├── auto_retrain.py             # Optuna hyperparameter optimization
│   │   ├── scheduled_retrain.py        # Weekly production retraining
│   │   ├── numerical_models.py         # HMM, Kalman, OU process models
│   │   ├── volatility.py               # EWMA, GARCH, realized vol
│   │   ├── volatility_regime.py        # Regime detection
│   │   └── cycle_detector.py           # Market cycle identification
│   ├── risk/
│   │   ├── manager.py                  # L3: 10-check risk engine + circuit breakers
│   │   ├── dynamic_manager.py          # Dynamic risk adjustment
│   │   ├── profit_protector.py         # Profit protection + loss aversion
│   │   ├── vpin_guard.py               # VPIN adverse selection guard
│   │   ├── monte_carlo_risk.py         # Monte Carlo VaR simulation
│   │   └── evt_risk.py                 # Extreme Value Theory tail risk
│   ├── trading/
│   │   ├── executor.py                 # Core orchestrator (9-layer pipeline, ~98KB)
│   │   ├── strategy.py                 # HybridStrategy (L1+L2+L3+L4 fusion)
│   │   ├── meta_controller.py          # L4: Weighted signal arbitration
│   │   ├── signal_combiner.py          # Multi-signal ensemble combiner
│   │   ├── backtest.py                 # Backtesting engine
│   │   └── adaptive_engine.py          # Adaptive parameter tuning
│   ├── monitoring/
│   │   ├── journal.py                  # Trade journal + decision logging
│   │   ├── alerting.py                 # Telegram/webhook alerts
│   │   ├── health_checker.py           # System health monitoring
│   │   └── event_guard.py              # Macro event detection
│   ├── persistence/
│   │   └── state_store.py              # SQLite state persistence + recovery
│   ├── security/
│   │   └── model_integrity.py          # Cryptographic model checksums
│   ├── indicators/                     # RSI, MACD, BB, ATR, ADX, OBV, etc.
│   ├── portfolio/                      # Black-Litterman allocation, hedging
│   ├── dashboard_server.py             # Flask dashboard (port 5000)
│   └── reporting/                      # Performance reports
│
├── models/                             # Trained model artifacts (~237 files)
│   ├── lgbm_{asset}.txt               # LightGBM models per asset
│   ├── lgbm_{asset}_{tf}.txt          # Multi-timeframe models
│   ├── lgbm_{asset}_{tf}_{year}.txt   # Year-specific models
│   ├── patchtst_v1.pt                 # PatchTST transformer weights
│   ├── meta_learning_model.json       # Meta-learning state
│   ├── checksums.json                 # Model integrity checksums
│   └── feature_importance_*.csv       # Feature importance per model
│
├── tests/                              # Test suite (15 files)
├── config.yaml                         # System configuration
├── .env                                # API keys (never committed)
├── requirements.txt                    # Python dependencies
└── logs/                               # Runtime logs + trade journal
```

---

## Quick Start

**Supported platforms:** Windows, macOS (Intel & Apple Silicon), and Linux. **Python 3.11+** is required (CI uses 3.12). The same commands work everywhere; only the venv activation step differs by OS.

### 1. Environment Setup

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API Keys

Create `.env` from the example:
```bash
cp .env.example .env
```

**Minimum required for testnet:**
```env
BINANCE_TESTNET_KEY=your_testnet_api_key
BINANCE_TESTNET_SECRET=your_testnet_api_secret
```

Get testnet keys (free, sandbox money): https://testnet.binance.vision/

**Recommended (optional, for full functionality):**
```env
REASONING_LLM_KEY=your_gemini_api_key      # L6 LLM reasoning
NEWSAPI_KEY=your_newsapi_key               # L2 news sentiment
CRYPTOPANIC_TOKEN=your_cryptopanic_token   # L2 crypto news
```

### 3. Verify API Keys

```bash
python verify_api_keys.py
```

### 4. Run the System

```bash
# Paper trading (live data, simulated orders)
python -m src.main --mode paper

# Testnet trading (real orders on sandbox exchange)
python -m src.main

# Testnet + dashboard
python -m src.main --dashboard

# Retrain models
python -m src.main --retrain

# Dashboard only
streamlit run src/api/dashboard_app.py --server.port 8501

# Flask dashboard only
python -m src.dashboard_server
```

### 5. Access Dashboards

| Dashboard | URL | Port |
|:----------|:----|:-----|
| Streamlit (primary) | http://localhost:8501 | 8501 |
| Flask (secondary) | http://localhost:5000 | 5000 |
| FastAPI (REST) | http://localhost:8000 | 8000 |

---

## Configuration Reference

**File:** `config.yaml`

```yaml
# Execution mode
mode: testnet                    # paper | testnet | live
poll_interval: 30                # seconds between iterations

# Assets
assets: [BTC, ETH, AAVE]
initial_capital: 100000.0

# L1 Quantitative Engine
l1:
  short_window: 5                # Fast MA period
  long_window: 20                # Slow MA period
  vol_threshold: 1.5             # Volatility filter

# L2 Sentiment Thresholds
sentiment:
  fear_threshold: 0.3            # Below = bearish bias
  greed_threshold: 0.7           # Above = bullish bias

# L3 Risk Parameters
risk:
  max_position_size_pct: 1.0     # Max 1% of equity per position
  daily_loss_limit_pct: 3.0      # Halt trading at -3% daily
  max_drawdown_pct: 10.0         # Circuit breaker at -10%
  risk_per_trade_pct: 0.5        # Risk 0.5% per trade
  atr_stop_mult: 3.5             # Stop = 3.5x ATR
  atr_tp_mult: 2.5               # Take profit = 2.5x ATR

# Signal Gating
signal:
  min_confidence: 0.65           # Only trade above 65% confidence
  use_ensemble: true             # Use L1+L2+L3 ensemble
  neutral_threshold: 0.40

# Testnet Overrides
force_trade: true                # Force trades on testnet
max_trades_per_hour: 5

# AI Reasoning (L6)
ai:
  reasoning_provider: "auto"     # auto = cloud -> local -> rules
  rate_limit_calls_per_minute: 15
  fallback_on_quota_error: true

# Multi-Agent Overlay
agents:
  enabled: true
  blend_weight: 0.60             # 60% agent, 40% pipeline
  consensus_threshold: 0.55
  loss_prevention:
    daily_target_pct: 1.0        # Target +1% daily
    preservation_threshold_pct: 0.8
    halt_threshold_pct: -1.0     # HALT at -1%

# Polymarket Integration
polymarket:
  enabled: true
  divergence_threshold: 0.15
```

---

## Environment Variables

**File:** `.env`

| Variable | Purpose | Required |
|:---------|:--------|:---------|
| `BINANCE_API_KEY` | Live Binance trading | For live mode |
| `BINANCE_API_SECRET` | Live Binance secret | For live mode |
| `BINANCE_TESTNET_KEY` | Testnet Binance trading | For testnet mode |
| `BINANCE_TESTNET_SECRET` | Testnet Binance secret | For testnet mode |
| `REASONING_LLM_KEY` | Google Gemini API (L6 reasoning) | Recommended |
| `NEWSAPI_KEY` | NewsAPI (L2 sentiment) | Optional |
| `CRYPTOPANIC_TOKEN` | CryptoPanic (L2 sentiment) | Optional |
| `HUGGINGFACE_TOKEN` | HuggingFace (FinBERT download) | Optional |
| `GLASSNODE_API_KEY` | Glassnode on-chain metrics | Optional |
| `CRYPTOQUANT_API_KEY` | CryptoQuant exchange flows | Optional |
| `TELEGRAM_BOT_TOKEN` | Telegram alert notifications | Optional |
| `TELEGRAM_CHAT_ID` | Telegram chat for alerts | Optional |
| `ALERT_WEBHOOK_URL` | Webhook notifications | Optional |

All optional APIs have graceful fallbacks. The system runs with zero optional keys.

---

## Dashboard

### Streamlit Dashboard (port 8501)

**Header:** Portfolio P&L, Agent Winrate, Baseline Uplift, Active Layers

**Sentiment Heatmap:** Real-time Fear/Greed gradient bar from L2 composite scores

**TradingView Widgets:** Embedded candlestick chart + technical analysis gauge

**Expandable Layer Cards (L1-L9):**
- L1: VPIN toxicity, liquidity regime, flow imbalance, top features
- L2: Composite score, sentiment bias, bull/bear %, news velocity
- L3: Whale sentiment, net exchange flow, risk status
- L4: L1/L2/L3 attribution weights, final signal
- L5: Slippage estimate, fill rate, execution latency
- L6-L9: LLM reasoning trace, agent consensus, learning status

**News Feed:** Source-tagged, impact-classified (HIGH/MED), timestamped

**Sidebar:** 9-layer health bars, asset selector, sentiment veto toggle, emergency halt

---

## Testing

```bash
# All tests
pytest tests/ -v

# Specific areas
pytest tests/test_train_lgbm.py -v           # LightGBM features
pytest tests/test_sentiment_transformer.py -v # Sentiment pipeline
pytest tests/test_end_to_end.py -v           # Full pipeline smoke test
pytest tests/test_flash_crash_kill_switch.py -v # Circuit breakers
pytest tests/test_indicators.py -v           # Technical indicators
pytest tests/test_risk.py -v                 # Risk guardrails
pytest tests/test_meta_controller.py -v      # Agent ensemble voting
```

**Test Coverage:**
- Feature extraction shape validation
- Sentiment analysis (FinBERT + rule-based)
- End-to-end pipeline smoke test
- Circuit breaker behavior (flash crash, daily loss halt)
- Technical indicator accuracy
- Risk guardrail enforcement
- Meta-controller consensus detection
- Backtesting framework validation

---

## Disclaimer

This system is for professional and educational use. Automated cryptocurrency trading carries significant capital risk.

- Past performance does not guarantee future results
- Always use testnet mode first before deploying real capital
- The system provides tools, not financial advice
- Consult a financial advisor before live deployment

**System Version:** v6.5
**Last Updated:** 2026-03-18
