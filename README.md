# ACT — Autonomous Crypto Trader

**ACT is a trading system that runs itself.** It watches the crypto
market 24/7, reads the news, thinks through trades like a human analyst
would, places orders on Robinhood, learns from its mistakes, and gets
better every day.

It trades BTC and ETH. Two AI "brains" run locally on your GPU — no
cloud LLM bills, no data leaving the machine. It follows a fixed rule
book that can't be overridden, so it won't blow up your account chasing
a bad trade.

---

## The big picture — what actually happens

```
        ┌──────────────────────────────────────────────────┐
        │   1. EXPLORE     — read the market + news        │
        │   2. PLAN        — think through a trade         │
        │   3. CODE        — place the order (or skip)     │
        │   4. VERIFY      — was I right? What did I miss? │
        └──────────────────────────────────────────────────┘
                          ▲                │
                          │                ▼
                          └──── LEARN ─────┘
```

Every 60–180 seconds this loop runs. Each trip around the loop, the
system gets a little smarter. This is the same pattern Claude Code uses
to write software — applied to trading.

**Wide loop** (every few hours, and nightly): old strategies die,
new ones evolve, the AI brains get fine-tuned on yesterday's trades.

---

## The two brains

ACT uses **two local AI models** that work together — one fast, one
careful. They're both just open-source models running on your GPU
through Ollama.

```
 Market data, news, whale flows, sentiment, prices
                         │
                         ▼
            ┌──────────────────────────┐
            │   SCANNER BRAIN (right)  │   Qwen3 32B
            │   Quick pattern-spotting │   (every tick)
            │   "Something's brewing"  │
            └──────────────────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │   ANALYST BRAIN (left)   │   DeepSeek-R1 32B
            │   Deep reasoning         │   (on-demand)
            │   Compiles a trade plan  │
            └──────────────────────────┘
                         │
                         ▼
                ┌──────────────────┐
                │   SAFETY GATES   │
                │   (see below)    │
                └──────────────────┘
                         │
                         ▼
                   Trade placed
                   (or skipped)
```

**Scanner** is fast and runs every tick. Its job is to say *"the market
looks interesting"* or *"nothing's happening."* If nothing's happening,
no analyst call, no trade, no cost.

**Analyst** only wakes up when the scanner sees something. It has access
to 30+ tools (see below), thinks through the trade over multiple rounds,
writes out a full **Trade Plan** (asset, entry price, stop loss, take
profit, expected hold time, reasoning), and submits it.

You can A/B-test different model pairs via the `ACT_BRAIN_PROFILE` env:

| Profile | Scanner | Analyst | When to use |
|---|---|---|---|
| `qwen3_r1` *(default)* | qwen3:32b | deepseek-r1:32b | Recommended for trading |
| `dense_r1` | deepseek-r1:7b | deepseek-r1:32b | Lighter VRAM footprint |
| `moe_agentic` | qwen2.5-coder:7b | qwen3-coder:30b | Best for strict JSON output |
| `devstral_qwen3coder` | devstral:24b | qwen3-coder:30b | Agentic tool-use pair |

---

## Why it (probably) won't blow up your account

Seven independent "gates" stand between the AI and your money. The AI
cannot disable them. Any one of them says no → no trade:

```
        AI wants to trade
                │
                ▼
   ┌──────────────────────────────┐
   │ 1. Authority rules           │ ← Hard-coded rules from the
   │    (7 non-negotiable rules)  │    operator's authority PDF
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ 2. Conviction gate           │ ← Sniper / normal / reject tier
   │    (is the setup strong?)    │
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ 3. Readiness gate            │ ← 500 trades + 14-day paper soak
   │    (has it proved itself?)   │    required before real money
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ 4. Champion gate             │ ← New AI model beats old by 2%
   │    (is the new brain better?)│    before it gets to trade
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ 5. Credit assigner           │ ← Losing components automatically
   │    (who's been screwing up?) │    get less say in future trades
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ 6. Quarantine manager        │ ← Anything that goes weird gets
   │    (is the signal sane?)     │    isolated for 5+ samples
   └──────────────────────────────┘
                │
                ▼
   ┌──────────────────────────────┐
   │ 7. Pre-trade hook            │ ← Your custom policy / Slack
   │    (operator-defined)        │    approval / whatever you want
   └──────────────────────────────┘
                │
                ▼
        Trade actually placed
```

**Default mode is paper.** Real money requires you to explicitly set
`ACT_REAL_CAPITAL_ENABLED=1`. Until then every trade is simulated and
logged so the soak counter can climb.

---

## The 30+ tools the analyst can use

When the analyst wakes up, it doesn't hallucinate — it calls tools to
get real data. These are grouped by what they do:

**In-process data**
- `get_recent_bars` — price history
- `get_support_resistance` — S/R levels
- `get_orderbook_imbalance` — bid/ask pressure
- `get_macro_bias` — is macro favoring longs or shorts?
- `get_regime` — trending / ranging / volatile / choppy
- `query_recent_trades` — what did I just do?
- `get_readiness_state` — are the gates open?
- `estimate_impact` — Monte-Carlo what-if for a trade size
- `submit_trade_plan` — **the only tool that causes a trade**

**Quantitative math** (wraps `src/models/`)
- `fit_ou_process` — mean-reversion half-life
- `hurst_exponent` — trending or mean-reverting?
- `kalman_trend` — smoothed trend estimate
- `hmm_regime` — 4-state regime classifier
- `hawkes_clustering` — volatility cluster intensity
- `test_cointegration` — pair-trade opportunity

**Web tools**
- `get_web_context` — news/sentiment/institutional bundle
- `get_news_digest` — recent headlines
- `get_fear_greed` — crypto fear/greed index
- MCP-server tools (whatever you wire in — CoinGecko, TradingView, etc.)

**Agent tools** (each wraps a specialist agent — see below)
- `ask_risk_guardian`, `ask_loss_prevention`, `ask_trend_momentum`,
  `ask_mean_reversion`, etc.

**Knowledge graph**
- `query_knowledge_graph` — real-time graph of news, whale flows,
  institutional activity, polymarket odds, correlations

Every tool returns a short digest (≤500 chars) so the analyst's
thinking-budget isn't flooded with raw data.

---

## Skills — operator-facing commands

Type these from your terminal (or build automation around them):

```bash
python -m src.skills.cli run <skill-name>
```

| Skill | What it does |
|---|---|
| `status` | Traffic-light view of all subsystems (ollama, warm_store, brain_memory, graph, personas, readiness) |
| `diagnose-noop` | "Why hasn't ACT traded yet?" — audits every gate in order |
| `readiness` | Is the gate open? How close are we? |
| `regime-check` | What regime are we in? Which strategy is champion? |
| `agent-post-mortem` | Replay a past decision — each agent explains its vote |
| `fine-tune-brain` | Run the QLoRA fine-tune cycle (confirm required) |
| `polymarket-hunt` | Scan Polymarket for binary-option opportunities |
| `emergency-flatten` | Kill switch — stops all trading immediately |

---

## The agents (13 specialists + transient personas)

These are smaller rule-based experts that the analyst brain can
consult. Each has an opinion; each keeps a rolling accuracy score
that earns or loses them influence over time.

**Fixed 13:**
`data_integrity_validator` · `market_structure` · `regime_intelligence` ·
`mean_reversion` · `trend_momentum` · `risk_guardian` ·
`sentiment_decoder` · `trade_timing` · `portfolio_optimizer` ·
`pattern_matcher` · `loss_prevention_guardian` · `decision_auditor` ·
`polymarket_agent`

**Transient personas** spawn automatically when the knowledge graph sees
a hot cluster (e.g. *"FOMC-announcement-in-18h"*, *"whale-accumulation-
phase"*) and dissolve when the event passes. Max 6 concurrent.

**Debate engine:** a 3-round adversarial process where paired opposites
(bull vs bear, timing vs portfolio, etc.) challenge each other and can
flip their votes.

---

## How it learns — the wide loop

```
Every tick  →  trade closes  →  self-critique writes a row
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Was the thesis right?   │
                         │ What did I miss?        │
                         │ Next-time-do-this.      │
                         └─────────────────────────┘
                                      │
                                      ▼
                    last 5 critiques feed NEXT cycle's seed context
                                      │
                                      ▼
Every 6 hours → genetic engine evolves 50-100 new strategy candidates
Every 6 hours → Thompson bandit picks which candidate to try
Every 20 trades → credit-assigner re-weights who gets credit / blame
Every night    → /fine-tune-brain runs QLoRA on filtered winning trades
                 → champion gate: new adapter must beat old by 2%
```

**Learning mesh (C18)** closes the loop: a **differential Sharpe ratio
(DSR)** is computed per component per asset, and feeds back into the
bandit + credit assigner. Components sustaining negative DSR get
quarantined automatically.

---

## Data it watches, in real time

- Price bars (BTC + ETH, 1m / 5m / 15m / 1h / 4h / 1d)
- 9 crypto news feeds (CoinDesk, Cointelegraph, Decrypt, etc.)
- FinBERT sentiment scores on every headline
- On-chain whale flows, exchange inflows/outflows
- Institutional data (L/S ratio, options sentiment, stablecoin flows)
- 12 macro indicators (CPI, FOMC, DXY, VIX, etc.)
- Polymarket prediction-market odds (as signal)
- Fear & Greed index, funding rates, open interest
- Orderbook state (L2 from Robinhood when available)

All of it flows into a **real-time knowledge graph** with time-decayed
edges — so an old correlation is worth less than a fresh one.

---

## Quick start

**Requirements:** RTX 5090 (or similar 32GB VRAM), Windows 11, Python
3.14, Ollama installed.

```cmd
# On the GPU box:
cd C:\Users\admin\trade
git pull

# Optional — pick a brain profile (default is qwen3_r1):
setx ACT_BRAIN_PROFILE qwen3_r1
setx ACT_AGENTIC_LOOP 1

# Open a fresh terminal, then:
powershell -ExecutionPolicy Bypass -File START_ALL.ps1
```

This starts 7 processes:

1. API server (port 11007 — dashboard backend)
2. Trading bot (the main loop)
3. Adaptation loop (continuous model retraining)
4. Autonomous loop (self-improvement every 20 min)
5. Genetic loop (strategy evolution every 6 hours)
6. Frontend (React dashboard)
7. Cloudflare tunnel (remote access, optional)

Verify it's working:

```cmd
python -m src.skills.cli run status
python -m src.skills.cli run diagnose-noop
```

---

## Honest performance expectations

Three ways to read *"1%/day"*:

| Interpretation | What that means | Is it realistic? |
|---|---|---|
| Compounded 1%/day | 3,778%/year | ❌ Not on any current strategy class |
| Simple 1%/day on seed | 365%/year simple | ❌ ~10× above Robinhood ceiling |
| Individual days ≥ 1% | "Good days" | ✅ Regularly reachable on sniper setups |

**The honest ceiling on Robinhood is ~0.15%/day simple (~37%/year)**
because Robinhood charges a 1.69% round-trip spread. That's the limit
of this venue; no amount of code tweaking gets past it.

To get meaningfully above that ceiling, the path is **venue migration:**
- Polymarket (already wired, shadow mode) — binary-option markets
  bypass the spread tax entirely.
- Bybit spot + perp (not yet connected) — 30× cheaper spread
  (0.055% vs 1.69%), unlocks dormant funding-arbitrage.

ACT **will** hunt outlier 1%+ days on Robinhood. It **won't** average
them. The dashboard shows both — today's %, rolling 7-day average,
best-day-of-month, and the gap to 1%/day — so you see the honest
number every day.

---

## Project structure (simplified)

```
trade/
├─ src/
│   ├─ ai/              ← LLM routing, dual-brain, tools, fine-tune
│   ├─ agents/          ← 13 specialist agents + persona spawner
│   ├─ trading/         ← executor, conviction gate, plan model, backtest
│   ├─ learning/        ← DSR reward, bandit, credit, safety, co-evolution
│   ├─ orchestration/   ← hooks, warm_store, readiness gate, streams
│   ├─ models/          ← quant models (OU, HMM, Kalman, Hurst, Hawkes)
│   ├─ backtesting/     ← 4 engines (vectorized, event-driven, walk-forward, MC)
│   ├─ scripts/         ← autonomous_loop, adaptation, daily_ops
│   └─ skills/          ← CLI dispatcher
├─ skills/              ← <name>/skill.yaml + action.py
├─ tests/               ← 819 passing tests
├─ config.yaml          ← all knobs
├─ START_ALL.ps1        ← single-command boot
├─ CLAUDE.md            ← capability map for AI sessions
└─ README.md            ← this file
```

---

## Testing

```bash
pytest tests/ -q
# → 819 passed, 2 skipped (3m01s typical)
```

Tests cover: plan compile, context manager, strategy repo, tool dispatch,
agentic loop, web tools, dual-brain routing, brain-memory round-trip,
Thompson bandit, knowledge graph, persona lifecycle, quant tools,
episodic memory, Polymarket gate, champion gate, training-data filter,
fine-tune orchestrator, DSR reward, learning mesh, brain-to-body
controller, and 30+ existing regression tests.

---

## Kill switches (if things go wrong)

| Env var | Effect |
|---|---|
| `ACT_DISABLE_AGENTIC_LOOP=1` | Reverts to non-agentic executor path |
| `ACT_DISABLE_DUAL_BRAIN=1` | Disables both LLM brains |
| `ACT_DISABLE_HOOKS=1` | Disables operator hooks |
| `ACT_DISABLE_BODY_CONTROLLER=1` | Neutralizes brain-to-body pressure signals |
| `ACT_REAL_CAPITAL_ENABLED=0` (or unset) | Paper mode only |
| `python -m src.skills.cli run emergency-flatten` | Nuclear option |

---

## Audit trail

Every decision ACT makes is logged:

- `data/warm_store.sqlite` — decisions + outcomes + plan JSON +
  component signals + self-critique
- `data/brain_memory.sqlite` — scanner↔analyst shared memory
- `data/strategy_repo.sqlite` — versioned strategies with live stats
- `data/knowledge_graph.sqlite` — real-time entity graph
- `memory/agent_<name>_state.json` — per-agent weights + episode buffers
- `logs/trade_decisions.jsonl` — cycle-level decision trace
- `logs/autonomous_cycles.jsonl` — self-improvement cycle history

---

## Known limits (read these honestly)

- Sustained 1%/day average on Robinhood is **structurally impossible**
  because of the 1.69% round-trip spread. The math caps at ~0.15%/day
  average.
- New LLM adapters need 100+ quality-filtered trades before fine-tune
  is worth running.
- GPU can only fine-tune one 30B-class model at a time (sequential,
  not parallel).
- The knowledge graph's ingest throttles at ~5 nodes/s to stay under
  SQLite contention — surprising news spikes may take a few ticks to
  fully absorb.

---

## Disclaimer

This software makes automated trading decisions. **Trading involves
real risk of loss.** Run in paper mode until you've seen enough ticks
on your own machine to trust it. The readiness gate exists specifically
to stop you from turning on real capital too early — don't override it.

ACT's AI brains are open-source models running locally. They are
capable but not infallible. The safety gates are what keep small
mistakes from becoming account-destroying ones. Don't disable them.

No warranty. No guaranteed returns. Use at your own risk.

---

*For a deeper capability map aimed at AI-engineering sessions, see
[CLAUDE.md](./CLAUDE.md). For the complete engineering activity log
of the session that built the current system, see
[docs/activity_log_2026_04_23_to_24.md](./docs/activity_log_2026_04_23_to_24.md).*
