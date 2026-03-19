# Engineering Proposal: AI-Driven Automated Crypto Trading System (v5.5)

**Project Focus:** BTC/ETH  
**Platform:** Robinhood API (paper first, live later)  
**Target Benchmark:** 1% Daily **Stretch Goal** (realistic goal 0.2-0.5%)

---

## 1. Executive Summary & The "1% Daily" Engineering Challenge

This document outlines a hybrid ensemble AI trading platform built for Bitcoin and Ethereum.  
The stated business target of 1% daily return is treated as an *engineering challenge* rather than a financial guarantee.  

- **Compounding math reminder:** 1% daily → ~3,678% annual, far beyond historical hedge fund results.  
- **Realistic operating zone:** 0.2–0.5% daily (≈40–100% annual) during trending markets.  
- **Engineering stretch:** Activate a high‑performance regime in extreme momentum/volatility windows; otherwise conserve capital.

The system architecture prioritizes safety by default, with an on‑demand performance mode for favorable conditions.

---

## 2. Technical Architecture: Hybrid Ensemble Integration

### 2.1 The Five-Layer System Design

1. **L1 – Data Ingestion**  
   • Real-time 1‑second ticks from Robinhood (primary) and Binance (failover)  
   • News and social feeds via FinBERT‑ready pipelines.
2. **L2 – Feature Engineering**  
   • 80+ technical indicators (RSI, MACD, SMA, EMA, Bollinger, etc.)  
   • GARCH(1,1) volatility, ATR, log returns, cycle detection.
3. **L3 – Model Inference**  
   • Temporal Fusion Transformer (TFT) and N‑Beats for multi‑horizon forecasting.  
   • FinBERT/FinGPT performs narrative sentiment and event classification.
4. **L4 – Decision & Orchestration**  
   • LightGBM classifier consuming fused features (quant + sentiment).  
   • Output: ternary signal {Long / Short / Flat}.  
5. **L5 – Execution & Audit**  
   • ExchangeAdapter abstracts Robinhood orders; 5‑minute fallback to Coinbase.  
   • Full audit trail and FIFO cost‑basis tracking for compliance.

### 2.2 Feature Fusion Pipeline

- **Sentiment Stream:** FinBERT scores each article [-1,+1] with confidence.  
- **Quantitative Stream:** LightGBM inputs price action and on‑chain metrics (MVRV, SOPR, fee indexes).  
- **Fusion:** Append sentiment score to LightGBM feature vector; model learns blended relationships.

---

## 3. Model Strategy: Quantitative & Narrative Intelligence

### 3.1 Quantitative Precision

- **TFT & N‑BEATS:** Forecast 1–30 bar horizons; detect seasonal/weekend cycles.  
- **Prophet (Meta):** Macro trend decomposition; weekly crypto rhythm detection.  
- **GARCH(1,1):** Conditional variance model to scale position sizes inversely to risk.

### 3.2 Narrative Sentiment Layer

- **Llama 3.1 8B (Ollama):** Deep event classification (regulatory, hack, ETF, etc.).  
- **Celery Queue:** Asynchronous processing to handle news bursts without blocking.
- **Veto Logic:** LightGBM long + FinBERT strong bearish → veto or reduce size.

---

## 4. Risk Management Framework: The "Safety First" Protocol

### 4.1 Capital Allocation & Stop-Loss Suite

- **Fractional Kelly (0.25×):** Limits variance while using edge.  
- **Exposure Caps:** 5% max per trade, 20% total crypto exposure (80% cash).  
- **Five Stop Types:** ATR (1.5×), hard % (3%), trailing (3%), time (>48h), sentiment flip.

### 4.2 Tiered Drawdown Halt System

- **Daily:** 3% loss ⇒ 24‑hour halt.  
- **Weekly:** 7% loss ⇒ Conservative Mode (50% sizing).  
- **Monthly/Peak:** 15–20% loss ⇒ Permanent halt + liquidation.

---

## 5. Performance & Continuous Learning

- **KPIs:** Sharpe Ratio >1.5, Calmar Ratio >2.0.  
- **Retraining:** LightGBM weekly on 6‑month rolling window; FinBERT quarterly fine-tune.  
- **Drift Detection:** Evidently AI monitors sentiment-price correlation.  
- **Tax Compliance:** FIFO cost‑basis tracker for automated 1099‑DA.

---

## 6. Implementation Timeline (22 Weeks)

1. **Phase 1 (Wks 1–7):**  Infrastructure, data ingestion, backtesting; simulate 0.75% Robinhood spread.  
2. **Phase 2 (Wks 8–17):**  Sentiment integration + 6‑week paper trading live.  
3. **Phase 3 (Wks 18–19):**  Live pilot with $1k capital, manual supervision.  
4. **Phase 4 (Wk 20+):**  Full capital scaling; weekly retraining and monitoring cycle.

---

*Note: Robinhood API integration remains as a stub in current repo pending compliance review; execution module will be replaced during Phase 1.*

---

*This engineering proposal is responsive to your earlier requests and aligns with the existing repository structure.  Additional components (FinGPT, LightGBM, TFT, N‑Beats) have been scaffolded and documented.  Let me know if you would like template code for the execution adapter or detailed risk‑engine pseudocode.*
