# ACT — Autonomous Crypto Trader v8.0

An autonomous, self-evolving crypto trading system for BTC and ETH on **Robinhood Crypto**. Combines LightGBM, PatchTST Transformers, Reinforcement Learning, FinBERT NLP, a **14-Agent Bayesian Consensus Engine** (including an Authority Compliance Guardian with veto power), Genetic Strategy Evolution seeded from authority strategies, multi-source RSS news intelligence, live Fear&Greed + derivatives data, and a two-pass LLM brain (Mistral + Llama3.2) — running 24/7 on RTX 5090 GPU with continuous self-improvement.

---

## Table of Contents

- [What's new in v8.0](#whats-new-in-v80)
- [System Architecture](#system-architecture)
- [9-Layer Decision Pipeline](#9-layer-decision-pipeline)
- [14-Agent Consensus System](#14-agent-consensus-system)
- [Authority Rule Enforcement](#authority-rule-enforcement)
- [Real-time Data Sources](#real-time-data-sources)
- [Self-Evolution Loop](#self-evolution-loop)
- [Risk Management](#risk-management)
- [GPU Setup (RTX 5090)](#gpu-setup-rtx-5090)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Dashboard](#dashboard)
- [Audit Trail](#audit-trail)
- [Disclaimer](#disclaimer)

---

## What's new in v8.0

- **Authority Compliance Guardian** — 14th agent with absolute VETO power. Enforces the non-negotiable authority-PDF rules (asset trade-type permissions, higher-TF trend agreement, wick entries, small-body candles, stop widening, adding to losers, news blackouts, regime-gated mean reversion, all 4 fakeout filters on 5m/15m). Deterministic — runs alongside the LLM rather than relying on it.
- **Authority system prompt injection** — every Mistral/Llama3.2 call opens with the authority directives, so the LLM cites the official strategy by name (S1/S2/S3). Lean-mode toggle (`LLM_LEAN_PROMPT=1`) drops redundant strategy explanations once LoRA internalizes them, saving ~4.5k chars/call.
- **Multi-source RSS news intelligence** — replaces the dead CryptoPanic API with 9 free RSS feeds (CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, CryptoSlate, Bitcoinist, U.Today, NewsBTC, CryptoPotato). Parallel fetch, dedupe, time-decay weighting, asset-keyword filtering. ~180 unique headlines per refresh.
- **Live sentiment + macro wired into every agent** — `sentiment_data`, `ext_feats` (Fear&Greed, funding rate, open interest, put/call ratio), and `economic_data` (macro composite) now flow into every orchestrator cycle. Sentiment decoder sees real `[FINBERT=-0.91]` scores instead of neutral defaults.
- **Dynamic GPU-aware process orchestration** — `START_ALL.ps1` auto-detects GPU and scales adaptation/genetic/autonomous loop intervals from the formula `compute_score = VRAM*2 + cores + RAM/8`. Works from CPU-only laptops to A100/H100 class hardware without config changes.
- **Single universal launcher** — one `START_ALL.ps1` (PowerShell) is the only entry point (runs from cmd or PowerShell). Animated ANSI banner with red/white/green color flow, persistent VT enablement via registry + Win32 API.
- **Decision audit log** — every orchestrator cycle is appended to `logs/trade_decisions.jsonl` with direction/confidence, all agent votes, the 5 most recent headlines that drove sentiment, Fear&Greed, funding rate, and macro composite. Queryable via `GET /api/v1/decisions/recent`.
- **Live Intelligence dashboard panel** — React panel showing the sentiment score, headlines, Fear&Greed, funding rate, OI, put/call ratio that the agents actually saw on the most recent cycle.

---

## System Architecture

ACT runs a **9-layer cascading intelligence pipeline** with four parallel self-improvement loops running alongside the trading bot.

```
Market Data (BTC/ETH)  +  RSS News (9 feeds)  +  Fear&Greed  +  Derivatives  +  FRED macro
        │
        ▼
┌─────────────────────────────────────────────┐
│           9-LAYER DECISION PIPELINE          │
│                                             │
│  L1  Quantitative Engine (LightGBM+RL+TST) │
│  L2  Sentiment Intelligence (RSS+FinBERT)  │
│  L3  Risk Fortress (Kelly+ATR+VPIN)        │
│  L4  Signal Fusion (Meta-Controller)       │
│  L5  Smart Execution (Limit Orders)        │
│  L6  14-Agent Bayesian Consensus           │
│       └─ Authority Guardian (VETO)         │
│  L7  Two-Pass LLM Brain (Mistral+Llama)   │
│       └─ Authority-rule system prompt      │
│  L8  Tactical Memory (Trade History)      │
│  L9  Genetic Evolution (DNA Strategies)   │
│       └─ Seeded from authority strategies  │
└─────────────────────────────────────────────┘
        │
        ▼
  Robinhood Crypto API
  (BTC/ETH — Paper + Live)
        │
        ▼
┌─────────────────────────────────────────────┐
│         SELF-EVOLUTION LOOPS (parallel)      │
│                                             │
│  Adaptation Loop   dynamic — retrain       │
│  Autonomous Loop   dynamic — self-heal     │
│  Genetic Loop      dynamic — evolve DNA    │
│  LLM Fine-Tune     on outcomes — LoRA      │
└─────────────────────────────────────────────┘
```

Loop intervals scale with hardware: compute_score = VRAM×2 + cores + RAM/8, so RTX 5090 runs adaptation ~15min while a CPU-only machine runs it every 6h.

---

## 9-Layer Decision Pipeline

| Layer | Name | Function | Key Files |
|:------|:-----|:---------|:----------|
| **L1** | Quantitative Engine | 80+ features, LightGBM/PatchTST/RL ensemble | `src/models/lightgbm_classifier.py`, `src/ai/patchtst_model.py` |
| **L2** | Sentiment Intelligence | RSS news (9 feeds) → FinBERT/rule-based → time-decayed aggregate | `src/ai/rss_news_aggregator.py`, `src/ai/sentiment.py` |
| **L3** | Risk Fortress | Kelly sizing, ATR stops, VPIN toxicity guard | `src/risk/manager.py`, `src/risk/position_sizing.py` |
| **L4** | Signal Fusion | L1+L2+L3 weighted combiner (50/30/20%) | `src/trading/signal_combiner.py` |
| **L5** | Smart Execution | Limit order routing, spread-aware entry | `src/trading/executor.py` |
| **L6** | Agent Intelligence | 14-agent Bayesian consensus + authority + loss-prevention vetoes | `src/agents/orchestrator.py` |
| **L7** | LLM Brain | Two-pass Mistral (scanner) + Llama3.2 (analyst), authority-prompt injection | `src/ai/trading_brain.py`, `src/ai/authority_rules.py` |
| **L8** | Tactical Memory | Episodic trade memory for pattern recall | `src/ai/memory_vault.py` |
| **L9** | Genetic Evolution | DNA strategy population seeded from authority strategies | `src/trading/genetic_strategy_engine.py` |

---

## 14-Agent Consensus System

All agents run in parallel and vote on every trade. Authority Guardian and Loss Prevention Guardian can VETO the trade entirely; authority takes highest priority (rules come from superiors).

| # | Agent | Role | Veto? |
|---|-------|------|:-----:|
| 1 | Authority Compliance Guardian | Enforces authority-PDF rules | ✅ highest |
| 2 | Loss Prevention Guardian | Daily PnL modes, drawdown, correlation | ✅ |
| 3 | Pattern Matcher | Candlestick + price action patterns | |
| 4 | Portfolio Optimizer | Correlation-aware position sizing | |
| 5 | Mean Reversion | RSI extremes, Bollinger reversals | |
| 6 | Market Structure | HH/HL/LH/LL, BOS/CHoCH | |
| 7 | Trend Momentum | MACD, ADX, slope acceleration | |
| 8 | Risk Guardian | Drawdown risk, sizing guard | |
| 9 | Sentiment Decoder | RSS/FinBERT + Fear&Greed + whale | |
| 10 | Trade Timing | Session awareness, volatility regimes | |
| 11 | Regime Intelligence | Trending/Ranging/Volatile/Choppy | |
| 12 | Polymarket Arbitrage | Prediction-market mispricing | |
| 13 | Data Integrity Validator | Pre-gate: validate quant data quality | |
| 14 | Decision Auditor | Post-gate: audit for contradictions | |

---

## Authority Rule Enforcement

Every decision is deterministically validated against the authority-PDF directives:

**Asset permissions** — BTC: scalp+intraday+swing allowed. ETH + alts: day trades ONLY (never swing).

**Multi-timeframe hierarchy** — Trend TF always wins conflicts:

| Trade Type | Trend TF | Entry TF | Hold |
|---|---|---|---|
| Swing | 1D | 1H | 3-10 days |
| Intraday | 4H | 15m | 12-48h |
| Scalp | 1H | 5m | 2-8h |

**Official strategies** — seeded into the genetic DNA pool:
- **S1** — 400 EMA Two-Candle Closure
- **S2** — Three-Candle Formation
- **S3** — Regime-Gated Mean Reversion (CHOP/LOW_VOL only)

**Fakeout filters** (all 4 required on 5m/15m): unusual candle, liquidity sweep, double top/bottom, back-to-zone re-entry.

**Universal rules** (VETO on any violation): higher-TF trend must agree, never enter on wick, never enter small-body candle, never widen stop, never add to loser, news blackout active, mean reversion outside CHOP.

Rules live in `src/ai/authority_rules.py` and are both (a) injected into every LLM prompt and (b) checked deterministically by `AuthorityComplianceGuardian` agent.

---

## Real-time Data Sources

All data sources are live — no synthetic/default fallbacks when the feed is up.

| Source | Purpose | Endpoint | Auth |
|---|---|---|---|
| **9 RSS feeds** (CoinDesk, Cointelegraph, Decrypt, Bitcoin Magazine, CryptoSlate, Bitcoinist, U.Today, NewsBTC, CryptoPotato) | News sentiment | feed URLs | none |
| **alternative.me** | Fear & Greed Index | `api.alternative.me/fng/` | none |
| **Bybit** | Funding rate | `api.bybit.com/v5/market/tickers` | none |
| **Binance Futures** | Open interest | `fapi.binance.com/fapi/v1/openInterest` | none |
| **Deribit** | Put/Call ratio | `www.deribit.com/api/v2/public/get_book_summary_by_currency` | none |
| **yfinance + FRED** | Treasury yields, yield curve | yfinance ticker + FRED API | optional FRED key |
| **blockchain.info** | Whale flows, exchange in/outflow | public API | none |
| **CCXT (Kraken)** | OHLCV candles | via ccxt | none |
| **Robinhood Crypto** | Real-time bid/ask, account | `trading.robinhood.com` | ED25519 key |

Diagnostic: `GET /api/v1/signals/live_intelligence` returns the snapshot the agents just saw.

---

## Self-Evolution Loop

ACT improves itself continuously without human intervention:

```
Trade executed
     │
     ▼
Outcome labeled (win/loss/pnl%)
     │
     ├── LLM Fine-Tune (LoRA) — Mistral + Llama learn from real outcomes
     │
     ├── Genetic Engine — DNA fitness updated (80% backtest / 20% live)
     │   seeded from 400 EMA, 3-candle, regime-reversion
     │
     ├── LGBM Retrain — new labels with spread-aware threshold (1.2x spread)
     │
     └── Confidence Calibrator — adjusts LLM confidence weights
```

**Session Filter** — blocks new entries during:
- Weekends (volume -40%)
- UTC 00:00-07:00 (dead hours, stop hunts)
- News blackout windows (15min before major scheduled events, 2 bars after)

**ROI Table** (minimum profit targets):

| Hold Time | Min Profit |
|---|---|
| 0-30 min | 6.0% |
| 30-120 min | 4.0% |
| 2-6 hours | 3.0% |
| 6-12 hours | 2.5% |
| 12-24 hours | 2.0% |

---

## Risk Management

- **Kelly Criterion** position sizing (half-Kelly for safety)
- **ATR-based stops** — `atr_stop_mult: 3.0` (3x ATR)
- **SL Progression** — L1 to L4 trailing stop (never moves down)
- **Daily loss limit** — 5% of equity halts trading
- **Max drawdown** — 15% triggers full shutdown
- **Hard stop** — -6% emergency exit per trade
- **Min confluence** — 5 independent signals required
- **Min entry score** — 6/15
- **Min expected move** — 5% (covers 1.69% spread + profit margin)
- **Longs only** — No short selling on Robinhood
- **Authority veto** — any authority-rule violation blocks the trade before it reaches the risk engine
- **Correlation veto** — Loss Prevention Guardian blocks new trade correlated with existing open position

---

## GPU Setup (RTX 5090)

ACT is optimized for RTX 5090 (Blackwell architecture, 32GB VRAM):

| Component | Status |
|---|---|
| PyTorch | cu128 build (CUDA 13.1) |
| Ollama | GPU-accelerated inference |
| VRAM Usage | ~11GB (Mistral + Llama3.2) |
| Fine-Tuning | LoRA in under 10 min per model |

Install CUDA PyTorch for RTX 5090:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA (auto-checked by `START_ALL.ps1`):
```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Works on any GPU** — the launcher computes `compute_score = VRAM*2 + cores + RAM/8` and adapts adaptation/autonomous/genetic loop intervals automatically. A100/H100 auto-loads additional models (neural-chat, codellama); smaller GPUs load just Mistral+Llama.

---

## Quick Start

### Prerequisites
- Windows 10/11 with NVIDIA GPU
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

### 1. Clone and install
```cmd
git clone https://github.com/develop-17-bbp/trade.git
cd trade
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Configure environment
```cmd
notepad .env
```

Add your credentials:
```env
ROBINHOOD_API_KEY=rh-api-xxxxxxxx
ROBINHOOD_PRIVATE_KEY=your-base64-ed25519-private-key
OLLAMA_BASE_URL=http://localhost:11434
# Optional
ANTHROPIC_API_KEY=sk-ant-xxxxx          # Falls back to Ollama if missing
FRED_API_KEY=xxxxxxxxx                  # Richer macro signals
LLM_LEAN_PROMPT=0                        # Set to 1 after LoRA maturity
```

### 3. Start all 7 systems
```powershell
powershell -ExecutionPolicy Bypass -File START_ALL.ps1
```

This launches (dynamic intervals based on detected hardware):

| # | Process | Details |
|---|---|---|
| 1 | API Server | `http://localhost:11007` |
| 2 | Trading Bot | Robinhood Paper (BTC/ETH), GPU priority |
| 3 | Adaptation Loop | Dynamic: retrain + fine-tune LLMs |
| 4 | Autonomous Loop | Dynamic: self-heal + monitor |
| 5 | Genetic Loop | Dynamic: evolve strategy DNA |
| 6 | Frontend | `http://localhost:5173` |
| 7 | Cloudflare Tunnel | Public remote access URL |

### 4. View dashboard
Open `http://localhost:5173` in your browser. Expand the bottom panel to see Live Intelligence: real sentiment, Fear&Greed, funding rate, headlines.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ROBINHOOD_API_KEY` | Yes | Robinhood Crypto API key |
| `ROBINHOOD_PRIVATE_KEY` | Yes | Base64-encoded ED25519 private key |
| `OLLAMA_BASE_URL` | Yes | Ollama server URL (default: localhost:11434) |
| `OLLAMA_REMOTE_MODEL` | No | Model name (default: mistral) |
| `ANTHROPIC_API_KEY` | No | Claude API (falls back to Ollama if missing) |
| `FRED_API_KEY` | No | Richer central-bank data (10y-2y curve) |
| `LLM_LEAN_PROMPT` | No | `1` drops base prompt after LoRA maturity, keeps authority |
| `DASHBOARD_API_KEY` | No | Protect API (auto dev-mode if unset) |

---

## Project Structure

```
trade/
├── START_ALL.ps1                  # Single universal launcher (dynamic GPU-aware)
├── STOP_ALL.ps1                   # Shutdown all ACT processes
├── config.yaml                    # Main configuration
├── .env                           # API credentials (never commit)
│
├── src/
│   ├── main.py                    # Entry point
│   ├── ai/
│   │   ├── trading_brain.py       # Two-pass LLM brain (Mistral+Llama)
│   │   ├── authority_rules.py     # Authority directives + validator + system prompt
│   │   ├── authority_context.py   # Live indicator → authority field mapper
│   │   ├── rss_news_aggregator.py # 9-source RSS news fetcher
│   │   ├── sentiment.py           # FinBERT + rule-based sentiment
│   │   ├── prompt_constraints.py  # Safety/authority prompt engine (lean mode)
│   │   ├── lora_trainer.py        # LoRA/QLoRA fine-tuning pipeline
│   │   ├── memory_vault.py        # Episodic trade memory
│   │   └── training_data_collector.py  # Captures decisions for fine-tuning
│   ├── trading/
│   │   ├── executor.py            # Main trading loop (wires sentiment, macro, audit)
│   │   ├── signal_combiner.py     # L1/L2/L3 signal fusion
│   │   └── genetic_strategy_engine.py  # DNA seeded from authority strategies
│   ├── agents/
│   │   ├── orchestrator.py        # 14-agent consensus engine
│   │   ├── combiner.py            # Authority > Loss-Prevention > Bayesian
│   │   ├── authority_compliance_guardian.py  # VETO on authority violations
│   │   ├── loss_prevention_guardian.py       # VETO on capital protection
│   │   └── ... (12 other agents)
│   ├── risk/
│   │   ├── position_sizing.py     # Kelly + ATR + volatility sizing
│   │   └── manager.py             # Dynamic risk allocation
│   ├── scripts/
│   │   ├── continuous_adapt.py    # Adaptation loop
│   │   ├── autonomous_loop.py     # Self-healing loop
│   │   └── genetic_loop.py        # Genetic evolution loop
│   ├── models/
│   │   ├── lightgbm_classifier.py
│   │   └── patchtst_model.py
│   ├── data/
│   │   ├── economic_intelligence.py  # Aggregates 12 macro layers
│   │   ├── layers/                # Fear&Greed, derivatives, FRED, etc.
│   │   ├── on_chain_fetcher.py    # Free-tier whale/exchange flow
│   │   └── ... (free-tier fetchers)
│   └── api/
│       └── production_server.py   # FastAPI REST API (port 11007)
│
├── frontend/                      # React/Vite dashboard (port 5173)
│   └── src/components/ai/
│       └── LiveIntelligencePanel.tsx  # Live sentiment + F&G + headlines
├── models/                        # Trained LGBM + LoRA model files
├── logs/                          # Trading journals, decisions, metrics
│   ├── trade_decisions.jsonl      # Per-cycle audit (sentiment + macro + agents)
│   └── trading_journal.jsonl      # LLM decisions with outcomes
├── data/                          # OHLCV parquet files (BTC/ETH 1h/4h/1d)
└── tests/
    └── test_core_systems.py       # 68 tests — all passing
```

---

## Testing

```cmd
pytest tests/test_core_systems.py -v
```

68 tests covering:

| Test Class | Coverage |
|---|---|
| `TestLLMResponseParsing` | Markdown fences, trailing commas, prose-embedded JSON, garbage input |
| `TestPositionSizing` | Kelly math, ATR sizing, volatility cap, hard cap at 2x risk |
| `TestSignalCombiner` | VETO override, BLOCK, REDUCE, staleness decay, agreement bonus |
| `TestConfluenceGating` | Entry score gate, min signals, expected move gate |
| `TestOrderExecution` | PnL math, SL monotonicity, daily loss limit, longs-only guard |
| `TestMultiAssetCombiner` | Signal ranking, max positions cap, per-asset caching |

---

## Dashboard

React/Vite frontend at `http://localhost:5173`:

- **Full-page candlestick chart** with ACT trade markers + position overlays
- **Live Intelligence panel** — aggregate news sentiment score, 5 most recent headlines, source breakdown, Fear&Greed gauge, funding rate, open interest, put/call ratio, macro composite (updates every 15s)
- **AI Consensus panel** — all 14 agent votes in real time
- **Performance metrics** — win rate, profit factor, Sharpe ratio
- **Model accuracy** — LightGBM, PatchTST, RL Agent, Strategist
- **Risk panel** — drawdown, open positions, SL progression levels
- **Trade history** — full decision log with LLM reasoning

Remote access via Cloudflare tunnel (auto-started by `START_ALL.ps1`).

---

## Audit Trail

Every orchestrator cycle is logged to `logs/trade_decisions.jsonl` so you can audit trades against the news + macro context that drove them:

```json
{
  "ts": "2026-04-18T01:23:45Z",
  "asset": "BTC",
  "raw_signal": 1,
  "decision": {"direction": 0, "veto": true, "violations": ["WICK_ENTRY: entry based on wick"]},
  "sentiment": {"score": -0.91, "label": "STRONG_NEGATIVE", "headline_count": 20, "recent_headlines": ["Russia introduces bill to criminalize unregistered crypto services", ...], "sources": ["coindesk", "cointelegraph", ...]},
  "macro": {"fear_greed": 21, "funding_rate": -0.000079, "open_interest_usd": 8.31e9, "put_call_ratio": 0.697, "composite": "CRISIS"},
  "agents": {"authority_compliance": {"direction": 0, "veto": true, "reasoning": "[AUTHORITY_VETO] WICK_ENTRY"}, ...}
}
```

Query via `GET /api/v1/decisions/recent?limit=50` or tail the file directly.

---

## Disclaimer

This software is for **educational and research purposes only**. Crypto trading involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this system. Always test with paper trading before using real funds.

---

*Built with Python 3.14 · FastAPI · React/Vite · LightGBM · PyTorch cu128 · Ollama · RTX 5090*
