# ACT — Autonomous Crypto Trader v7.0

An autonomous, self-evolving crypto trading system for BTC and ETH on **Robinhood Crypto**. Combines LightGBM, PatchTST Transformers, Reinforcement Learning, FinBERT NLP, a 12-Agent Bayesian Consensus Engine, Genetic Strategy Evolution, and a two-pass LLM brain (Mistral + Llama3.2) into a unified 9-layer decision architecture — running 24/7 on RTX 5090 GPU with continuous self-improvement.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [9-Layer Decision Pipeline](#9-layer-decision-pipeline)
- [12-Agent Consensus System](#12-agent-consensus-system)
- [Self-Evolution Loop](#self-evolution-loop)
- [Risk Management](#risk-management)
- [GPU Setup (RTX 5090)](#gpu-setup-rtx-5090)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Dashboard](#dashboard)
- [Disclaimer](#disclaimer)

---

## System Architecture

ACT runs a **9-layer cascading intelligence pipeline** with four parallel self-improvement loops running alongside the trading bot.

```
Market Data (BTC/ETH)
        │
        ▼
┌─────────────────────────────────────────────┐
│           9-LAYER DECISION PIPELINE          │
│                                             │
│  L1  Quantitative Engine (LightGBM+RL+TST) │
│  L2  Sentiment Intelligence (FinBERT+News) │
│  L3  Risk Fortress (Kelly+ATR+VPIN)        │
│  L4  Signal Fusion (Meta-Controller)       │
│  L5  Smart Execution (Limit Orders)        │
│  L6  12-Agent Bayesian Consensus           │
│  L7  Two-Pass LLM Brain (Mistral+Llama)   │
│  L8  Tactical Memory (Trade History)      │
│  L9  Genetic Evolution (DNA Strategies)   │
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
│  Adaptation Loop   every 1h  — retrain     │
│  Autonomous Loop   every 30m — self-heal   │
│  Genetic Loop      every 2h  — evolve DNA  │
│  LLM Fine-Tune     on outcomes — LoRA      │
└─────────────────────────────────────────────┘
```

---

## 9-Layer Decision Pipeline

| Layer | Name | Function | Key Files |
|:------|:-----|:---------|:----------|
| **L1** | Quantitative Engine | 80+ features, LightGBM/PatchTST/RL ensemble | `src/models/lightgbm_classifier.py`, `src/ai/patchtst_model.py` |
| **L2** | Sentiment Intelligence | FinBERT NLP + multi-source news | `src/ai/sentiment.py`, `src/ai/finbert_service.py` |
| **L3** | Risk Fortress | Kelly sizing, ATR stops, VPIN toxicity guard | `src/risk/manager.py`, `src/risk/position_sizing.py` |
| **L4** | Signal Fusion | L1+L2+L3 weighted combiner (50/30/20%) | `src/trading/signal_combiner.py` |
| **L5** | Smart Execution | Limit order routing, spread-aware entry | `src/trading/executor.py` |
| **L6** | Agent Intelligence | 12-agent Bayesian consensus + veto authority | `src/agents/orchestrator.py` |
| **L7** | LLM Brain | Two-pass Mistral (scanner) + Llama3.2 (analyst) | `src/ai/trading_brain.py` |
| **L8** | Tactical Memory | Episodic trade memory for pattern recall | `src/ai/memory_vault.py` |
| **L9** | Genetic Evolution | DNA strategy population, live fitness feedback | `src/trading/genetic_strategy_engine.py` |

---

## 12-Agent Consensus System

All 12 agents run in parallel and vote on every trade. Any agent can issue a **VETO** to block the trade entirely.

| Agent | Role |
|-------|------|
| Loss Prevention Guardian | Detects loss streaks — VETO authority |
| Pattern Matcher | Candlestick + price action patterns |
| Portfolio Optimizer | Correlation-aware position sizing |
| Mean Reversion | RSI extremes, Bollinger Band reversals |
| Market Structure | HH/HL/LH/LL, BOS/CHoCH detection |
| Trend Momentum | MACD, ADX, slope acceleration |
| Risk Guardian | Drawdown risk, position sizing guard |
| Sentiment Decoder | FinBERT + social signal aggregate |
| Trade Timing | Session awareness, volatility regimes |
| Regime Intelligence | Trending/Ranging/Volatile/Choppy |
| Data Integrity Validator | Pre-gate: validate quant data quality |
| Decision Auditor | Post-gate: audit for contradictions |

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
     │
     ├── LGBM Retrain — new labels with spread-aware threshold (1.2x spread)
     │
     └── Confidence Calibrator — adjusts LLM confidence weights
```

**Session Filter** — Blocks new entries during:
- Weekends (volume -40%)
- UTC 00:00-07:00 (dead hours, stop hunts)

**ROI Table** (minimum profit targets):

| Hold Time | Min Profit |
|-----------|------------|
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

---

## GPU Setup (RTX 5090)

ACT is optimized for RTX 5090 (Blackwell architecture, 32GB VRAM):

| Component | Status |
|-----------|--------|
| PyTorch | cu128 build (CUDA 13.1) |
| Ollama | GPU-accelerated inference |
| VRAM Usage | ~11GB (Mistral + Llama3.2) |
| Fine-Tuning | LoRA in under 10 min per model |

Install CUDA PyTorch for RTX 5090:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA:
```cmd
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

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
ANTHROPIC_API_KEY=sk-ant-xxxxx
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Start all 7 systems
```powershell
powershell -ExecutionPolicy Bypass -File START_ALL.ps1
```

This launches:

| # | Process | Details |
|---|---------|---------|
| 1 | API Server | `http://localhost:11007` |
| 2 | Trading Bot | Robinhood Paper (BTC/ETH) |
| 3 | Adaptation Loop | Every 1h: retrain + fine-tune LLMs |
| 4 | Autonomous Loop | Every 30min: self-heal + monitor |
| 5 | Genetic Loop | Every 2h: evolve 100 DNA strategies |
| 6 | Frontend | `http://localhost:5173` |
| 7 | Cloudflare Tunnel | Public remote access URL |

### 4. View dashboard
Open `http://localhost:5173` in your browser.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ROBINHOOD_API_KEY` | Yes | Robinhood Crypto API key |
| `ROBINHOOD_PRIVATE_KEY` | Yes | Base64-encoded ED25519 private key |
| `ANTHROPIC_API_KEY` | No | Claude API (falls back to Ollama if missing) |
| `OLLAMA_BASE_URL` | Yes | Ollama server URL (default: localhost:11434) |
| `OLLAMA_REMOTE_MODEL` | No | Model name (default: mistral) |

---

## Project Structure

```
trade/
├── START_ALL.ps1                  # Launch all 7 systems (universal — run from cmd or PowerShell)
├── STOP_ALL.ps1                   # Shutdown all ACT processes
├── config.yaml                    # Main configuration
├── .env                           # API credentials (never commit)
│
├── src/
│   ├── main.py                    # Entry point
│   ├── ai/
│   │   ├── trading_brain.py       # Two-pass LLM brain (Mistral+Llama)
│   │   ├── lora_trainer.py        # LoRA/QLoRA fine-tuning pipeline
│   │   ├── memory_vault.py        # Episodic trade memory
│   │   └── training_data_collector.py  # Captures decisions for fine-tuning
│   ├── trading/
│   │   ├── executor.py            # Main trading loop
│   │   ├── signal_combiner.py     # L1/L2/L3 signal fusion
│   │   └── genetic_strategy_engine.py  # DNA strategy evolution
│   ├── agents/
│   │   ├── orchestrator.py        # 12-agent consensus engine
│   │   └── combiner.py            # Bayesian vote combining
│   ├── risk/
│   │   ├── position_sizing.py     # Kelly + ATR + volatility sizing
│   │   └── manager.py             # Dynamic risk allocation
│   ├── scripts/
│   │   ├── continuous_adapt.py    # 1h adaptation loop
│   │   ├── autonomous_loop.py     # 30min self-healing loop
│   │   └── genetic_loop.py        # 2h genetic evolution loop
│   ├── models/
│   │   ├── lightgbm_classifier.py
│   │   └── patchtst_model.py
│   └── api/
│       └── production_server.py   # FastAPI REST API (port 11007)
│
├── frontend/                      # React/Vite dashboard (port 5173)
├── models/                        # Trained LGBM + LoRA model files
├── logs/                          # Trading journals, decisions, metrics
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
|------------|----------|
| `TestLLMResponseParsing` | Markdown fences, trailing commas, prose-embedded JSON, garbage input |
| `TestPositionSizing` | Kelly math, ATR sizing, volatility cap, hard cap at 2x risk |
| `TestSignalCombiner` | VETO override, BLOCK, REDUCE, staleness decay, agreement bonus |
| `TestConfluenceGating` | Entry score gate, min signals, expected move gate |
| `TestOrderExecution` | PnL math, SL monotonicity, daily loss limit, longs-only guard |
| `TestMultiAssetCombiner` | Signal ranking, max positions cap, per-asset caching |

---

## Dashboard

React/Vite frontend at `http://localhost:5173`:

- **Live equity curve** with entry/exit trade markers
- **AI Consensus panel** — all 12 agent votes in real time
- **Performance metrics** — win rate, profit factor, Sharpe ratio
- **Model accuracy** — LightGBM, PatchTST, RL Agent, Strategist
- **Risk panel** — drawdown, open positions, SL progression levels
- **Trade history** — full decision log with LLM reasoning

Remote access via Cloudflare tunnel (auto-started by `START_ALL.ps1`).

---

## Disclaimer

This software is for **educational and research purposes only**. Crypto trading involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred through use of this system. Always test with paper trading before using real funds.

---

*Built with Python 3.14 · FastAPI · React/Vite · LightGBM · PyTorch cu128 · Ollama · RTX 5090*
