# CLAUDE.md — ACT capability map for future sessions

This file is the single reference for what ACT can do today, what
pieces are wired, and what still gates profitable trades on Robinhood.
Read this first when starting work on ACT.

---

## 1. Target framing (read before anything else)

**What operator asks for:** "1%/day consistently."

**The three interpretations** — always use all three on the dashboard:
- **Compounded 1%/day** = 3,778% annual. Infeasible on any current
  strategy class on any venue. Don't promise this.
- **Simple 1%/day on seed capital** = 365%/year simple = operator's
  stated 7%/week + 30%/month numbers. ~10× the measured Robinhood
  spot-only ceiling (~37%/year simple). Not sustainable on RH alone.
  Path is venue migration: Polymarket (C13) + Bybit spot/perp + funding
  arb.
- **Individual days hitting 1%+ equity** — absolutely reachable on
  sniper-tier setups (e.g. 5% move × 15% position = +0.75% equity;
  both BTC+ETH resolving = +1-1.5% day). Welcome these, don't dismiss.

**Dashboard commitment:** every day show `today_pct`, `rolling_7d_avg`,
`best_day_this_month`, `gap_to_1pct_day` — so the honest number is
visible without burying upside.

Memory reference: `memory/project_realistic_targets.md`,
`memory/feedback_upside_framing.md`.

---

## 2. System architecture — the Claude-Code pattern for trading

ACT mirrors Claude Code's loop (Explore → Plan → Code → Verify) at
two timescales:

### Tight loop (60-180s per tick)

```
1. EXPLORE  — web_context.fetch_bundle (news/sentiment/institutional/
              polymarket/macro/fear_greed). Ingest into knowledge_graph.
2. PLAN     — Scanner brain (Qwen2.5-Coder-7B or DeepSeek-R1-7B)
              publishes ScanReport to brain_memory.
              Analyst brain (Qwen3-Coder-30B or DeepSeek-R1-32B) reads
              the scan + recent critiques + quant tools + MCP tools, runs
              multi-turn ReAct loop, compiles a TradePlan.
3. CODE     — conviction_gate + authority_rules validate the plan.
              In shadow mode (default), plan is logged to warm_store
              as `shadow-*`. In live mode (post-soak), executor
              places the order.
4. VERIFY   — On trade close, trade_verifier runs a second-LLM pass
              comparing predicted EV vs realized PnL, writes
              SelfCritique to warm_store. Next cycle's Scanner +
              Analyst see the last 5 critiques in their seed context.
```

### Wide loop (hours-to-days)

```
* Every 6h — Genetic strategy engine evolves new candidate strategies
             → strategy_repository (challenger status)
* Every 6h — Thompson-sampling bandit picks which challenger to test
             on the next allocation window
* Nightly  — /fine-tune-brain runs QLoRA on quality-filtered
             experience; champion-gate promotes new adapters
* Per 20 trades — credit_assigner reweights component contributions
                  (multi_strategy / LGBM / LoRA / LLM / RL)
```

---

## 3. Dual-brain model configuration (C5-C5d)

Three named profiles in `config.yaml:ai.dual_brain.profile`, switchable
via `ACT_BRAIN_PROFILE` env:

| Profile | Analyst (orchestrator) | Scanner (worker) | VRAM | When to use |
|---|---|---|---|---|
| **dense_r1** (default) | deepseek-r1:32b | deepseek-r1:7b | ~25 GB | Paper soak — most consistent output, fewer variance sources |
| **moe_agentic** | qwen3-coder:30b (MoE) | qwen2.5-coder:7b | ~24 GB | Best tool-use speed post-soak; 2026 agentic gold standard |
| **hybrid** | qwen3-coder:30b (MoE) | deepseek-r1:7b | ~24 GB | Post-soak recommendation — MoE speed on analyst + dense reasoning consistency on scanner |

Env overrides: `ACT_SCANNER_MODEL`, `ACT_ANALYST_MODEL` win over profile.
Kill switch: `ACT_DISABLE_DUAL_BRAIN=1`.

`<think>` tag stripping: scanner output is auto-stripped of
`<think>...</think>` traces (C5c) so brain_memory stays compact; analyst
keeps the full trace for audit.

Docs: `docs/dual_brain_setup.md`.

---

## 4. Skills — operator-facing named workflows

CLI: `python -m src.skills.cli {list|describe|run} [name]`

| Skill | Confirm? | Destructive? | Purpose |
|---|---|---|---|
| `/status` | — | — | Subsystem traffic lights (env, ollama, warm_store, brain_memory, graph, personas, readiness). Use after every pull. |
| `/diagnose-noop` | — | — | Answer "why hasn't ACT traded yet?" — audits every gate in priority order. |
| `/readiness` | — | — | Readiness gate state + emergency-mode flag + progress to target. |
| `/regime-check` | — | — | Current regime + champion strategy + top-3 posterior-mean picks from Thompson bandit. |
| `/agent-post-mortem` | — | — | Counterfactual chat about one decision — each agent persona explains its vote + `what_if_seconds` rewind. |
| `/fine-tune-brain` | ✅ | ✅ | QLoRA dual-brain fine-tune cycle (analyst 30-45min + scanner 5-10min sequential on RTX 5090). `dry_run=true` uses stub backend safely. |
| `/polymarket-hunt` | ✅ | — (shadow default) | Scan crypto markets, run conviction gate, submit top candidate. Shadow unless all 4 live gates open. |
| `/emergency-flatten` | ✅ | ✅ non-reversible | Kill switch: sets `ACT_DISABLE_AGENTIC_LOOP=1` + raises emergency mode + logs incident to warm_store. |

Skills live in `skills/<name>/{skill.yaml, action.py}`. Add new ones
by dropping a folder; the registry auto-discovers.

---

## 5. Tools (LLM-callable via ToolRegistry) — what the Analyst can ask for

The Analyst brain sees all of these inside its multi-turn ReAct loop.
Every tool returns a ≤500-char digest — no raw payloads reach the parent
context.

### In-process
- `get_recent_bars(asset, timeframe, n)` — OHLCV bars
- `get_support_resistance(asset)` — S/R levels
- `get_orderbook_imbalance(asset)` — bid/ask imbalance
- `get_macro_bias()` — signed macro tilt from `macro_bias.py`
- `get_regime()` — current regime classification
- `ask_risk_guardian(task_description, ...)` — sub-agent query
- `ask_loss_prevention(task_description, ...)` — sub-agent query
- `query_recent_trades(asset, limit)` — decisions+outcomes join
- `get_readiness_state()` — gate evaluation
- `estimate_impact(action, size_pct)` — Monte-Carlo over last 100 outcomes
- `search_strategy_repo(status, regime, min_sharpe)` — versioned store
- `submit_trade_plan(plan_json)` — **the only 'write' tool**

### Quant (C11) — wraps `src/models/`
- `fit_ou_process(asset, timeframe, bars)` — OU half-life + z-score
- `hurst_exponent(asset, timeframe, bars)` — trending/mean-rev/random-walk
- `kalman_trend(asset, timeframe, bars)` — level + slope
- `hmm_regime(asset, timeframe, bars)` — 4-state HMM regime + confidence
- `hawkes_clustering(asset, timeframe, bars)` — event intensity
- `test_cointegration(asset_a, asset_b, timeframe, bars)` — Engle-Granger

### Web (C3 + C7)
- `get_web_context(asset, include=[...])` — parallel Tier-1 bundle
- `get_news_digest(asset, hours)` — recent headlines
- `get_fear_greed()` — F&G index
- (when config.yaml `mcp_clients` has entries) — MCP-sourced tools mirrored
  as `{tag}_{remote_name}` (e.g. `cg_get_price`, `tv_get_pine_alerts`)

### Knowledge graph (C12)
- `query_knowledge_graph(asset, since_s, max_chars)` — compact digest
  over real-time graph (news/sentiment/institutional/polymarket/correlation
  edges with time-decayed weights)

Add new tools via `src/ai/trade_tools.py::build_default_registry` or
register external MCP servers in `config.yaml:mcp_clients`.

---

## 6. Agents — the 13 specialists + transient personas

**Fixed roster (`src/agents/`):**
`data_integrity_validator`, `market_structure`, `regime_intelligence`,
`mean_reversion`, `trend_momentum`, `risk_guardian`, `sentiment_decoder`,
`trade_timing`, `portfolio_optimizer`, `pattern_matcher`,
`loss_prevention_guardian`, `decision_auditor`, `polymarket_agent`.

All inherit `BaseAgent` with:
- Bayesian accuracy tracker (200-outcome rolling deque)
- Dynamic weight (clamped [0.3, 3.0]) updated per trade close
- JSON state persistence (survives restart)
- **C12b — per-agent episodic memory**: each agent keeps its own
  `(state_fingerprint, vote, outcome)` buffer. Query via
  `agent.get_similar_episodes(current_state, k=5)`.

**Transient personas (C14):** PersonaManager spawns LLM-backed personas
from hot clusters in the real-time knowledge graph (e.g. "FOMC-in-18h",
"whale-accumulation"), dissolves them when clusters cool. Max 6
concurrent, min cluster heat 0.6. They vote alongside the fixed 13 in
`debate_engine.py`.

**Debate engine (`src/agents/debate_engine.py`):** 3-round adversarial
deliberation. Round 1 parallel votes → Round 2 challengers critique
citing metrics → Round 3 defenders respond. Combiner weights survivors
with conviction multipliers. Paired opposition: bull vs bear, timing
vs portfolio, sentiment vs quant, etc.

---

## 7. Hooks — operator extension points (C8)

Event catalog (register via `config.yaml:hooks`):

| Event | Firing site | Blocking? |
|---|---|---|
| `pre_trade_submit` | before executor places order | ✅ non-zero exit vetoes |
| `post_trade_open` | after fill | — |
| `pre_exit` | before close | ✅ vetoes |
| `post_trade_close` | after close | — |
| `on_authority_violation` | authority rule triggered | — |
| `on_emergency_mode_enter` / `_exit` | Sharpe/target ratio crosses 0.7 | — |
| `on_strategy_promote` | challenger → champion | — |
| `on_startup` / `on_shutdown` | lifecycle | — |

Hook spec: `cmd` (shell, supports `${ENV}` + `{{context.key}}`) or
`python` (`module:attr`), `blocking: bool`, `timeout_s: float`, `name`.
Kill switch: `ACT_DISABLE_HOOKS=1`.

---

## 8. Safety gates (everything that can refuse a trade)

Stacked — any one says no and the trade doesn't fire:

1. **Authority rules** (`src/ai/authority_rules.py`) — 7 universal rules
   from the operator's authority PDF. Hard-coded; LLM cannot bypass.
2. **Conviction gate** (`src/trading/conviction_gate.py`) — sniper /
   normal / reject tiering based on TF alignment + Hurst + multi-strategy
   consensus + macro bias.
3. **Polymarket conviction** (`src/trading/polymarket_conviction.py`) —
   parallel gate for binary-option markets (different sizing math).
4. **Readiness gate** (`src/orchestration/readiness_gate.py`) — blocks
   real-capital trades until: 500 trades + 14d soak + credit R² ≥ 0.4 +
   authority violation rate < 2% + rolling Sharpe ≥ 1.0 + no quarantined
   learners + `ACT_REAL_CAPITAL_ENABLED=1`. Paper mode bypasses this.
5. **Champion gate** (`src/ai/champion_gate.py`) — new LoRA adapters
   must beat incumbent by ≥ 2% before hot-swap.
6. **Credit assigner** (`src/learning/credit_assigner.py`) — consistently-
   losing components get downweighted. Quarantined after 5 consecutive
   breaches of safety bounds.
7. **Pre-trade hook** (C8) — `pre_trade_submit` with `blocking: true`
   can veto per operator-defined policy.

---

## 9. Four backtest engines (C3 discovery)

1. **Vectorized** (`src/backtesting/engine.py`) — pure-numpy EMA(8)
   crossover, fast, used for hypothesis pre-checks.
2. **Event-driven** (`src/backtesting/full_engine.py`) — replicates
   live executor logic bar-by-bar. Used for candidate strategy
   validation before promotion.
3. **Walk-forward** (`src/trading/backtest.py`) — 80/15/5 train/val/test.
4. **Monte Carlo** (`src/backtesting/monte_carlo_bt.py`) — 10k trade
   sequences, produces probability-of-ruin + VaR + Kelly.

The Analyst can call vectorized backtest as a tool (`backtest_hypothesis`)
for <2s sanity checks pre-commit.

---

## 10. Data streams + storage

**Live fetchers (all in `src/data/` or `src/agents/`):**
news_fetcher, sentiment_decoder_agent, polymarket_fetcher,
on_chain_fetcher, institutional_fetcher, economic_intelligence (12
macro layers), robinhood_fetcher.

**Persistent stores:**
- `data/warm_store.sqlite` — decisions + outcomes + plan_json +
  component_signals + self_critique (C1 migration added last three)
- `data/brain_memory.sqlite` — scanner↔analyst corpus callosum (C7b)
- `data/strategy_repo.sqlite` — versioned strategies (C2)
- `data/knowledge_graph.sqlite` — real-time graph (C12)
- `memory/agent_<name>_state.json` — per-agent weight + accuracy +
  episode buffer (C12b)

---

## 11. What "git pull + START_ALL.ps1" actually does now (C17)

1. Detects GPU VRAM → picks `dense_r1` profile unless
   `ACT_BRAIN_PROFILE` overrides.
2. Pulls the active pair's models via `ollama pull`.
3. Starts `ollama serve` with `OLLAMA_NUM_PARALLEL=4` so scanner +
   analyst run concurrently.
4. Exports `ACT_AGENTIC_LOOP=1` so the shadow hook fires from the first
   tick.
5. Launches 7 parallel processes (API, trading bot, adaptation,
   autonomous, genetic, frontend, tunnel).
6. Every tick per asset: `shadow_tick.run_tick` fires which does
   web-bundle → graph ingest → scanner publish → analyst compile →
   persona refresh → warm_store log.

**After START_ALL completes:**
```powershell
python -m src.skills.cli run status
```
Should show:
- `[✓] env         = green`       (ACT_AGENTIC_LOOP=1)
- `[✓] ollama      = green`       (both models pulled)
- `[✓] warm_store  = green`       (shadow rows accumulating)
- `[✓] brain_memory = green`     (fresh scan <10min)
- `[…] graph        = yellow/green` (ingest throttled, takes a few ticks)
- `[…] personas     = yellow`     (takes cluster accumulation to spawn)
- `[…] readiness    = yellow`     (expected — paper soak in progress)

---

## 12. What still blocks profitable trades on Robinhood

Listed in order of likelihood (run `python scripts/diagnose_llm_silence.py
--hours 6 --tail 20` first to narrow — section 4's skip-reason histogram
is the smoking gun):

1. **paper_exploration cooldown reset by shadow rows** — FIXED 2026-04-30.
   `_last_decision_age_h()` now excludes `shadow-%` rows. If you see this
   regress, the fix is in `scripts/paper_exploration_tick.py`.
2. **Technical-lane direct fires bypassing LLM** — FIXED 2026-04-30 via
   `ACT_LLM_SOLE_AUTHOR=1` (default-on). `_evaluate_entry` short-circuits
   at the top, leaving `submit_trade_plan` as the only order path.
3. **LLM analyst silent (parse_failure / max_steps / model unreachable)** —
   surface via skip-reason histogram. If `parse_failure` dominates on the
   4060, the analyst is qwen2.5-coder:7b (a CODER model bad at trading
   JSON) — switch to `qwen3:8b` via `ACT_BRAIN_PROFILE=local_8gb`.
   Tech-blended escalation (`ACT_LLM_TECH_BLENDED=1`) catches this case
   automatically when tick_state has a directional consensus.
4. **Conviction gate refusing low-quality setups on RH 1.69% spread** —
   by design. paper_exploration `--relaxed` fires anyway as the soak
   safety net. Not a bug.
5. **Readiness gate closed** — expected during 14-day paper soak.
   Real capital waits for `ACT_REAL_CAPITAL_ENABLED=1` + 500+ trades.
6. **Authority min-hold forcing no day-trading** — by design; 24h hold
   on Robinhood. Not a bug.
7. **Ollama down / model not pulled** — `/status` shows red on `ollama`.
   Fix: `ollama pull qwen3:8b qwen2.5-coder:7b deepseek-r1:32b`.

---

## 13. Path to 1%/day (the honest plan)

**Phase 1 — Robinhood paper soak (now):**
- Let readiness gate accumulate 500 trades + 14 days.
- Watch `/status` + `/diagnose-noop` daily.
- Ceiling: ~0.15%/day simple average = ~37%/year.

**Phase 2 — Fine-tune on own experience:**
- After 100+ filtered positive samples accumulate, `/fine-tune-brain
  confirm=true dry_run=false` (needs Unsloth install on GPU box).
- Champion-gate ensures only better adapters swap in.
- Expected improvement: +0.1 to +0.3 Sharpe, translating to
  ~0.18-0.22%/day average.

**Phase 3 — Venue migration (the math-changing step):**
- Polymarket (C13) is already wired in shadow mode. Set
  `ACT_POLYMARKET_LIVE=1` + `POLYMARKET_API_KEY` + `pip install
  py-clob-client` when ready.
- Bybit spot + perp (not yet wired) cuts spread from 1.69% → 0.055%
  (30× reduction). Funding-arb dormant module
  (`src/trading/funding_arbitrage.py`) activates automatically when
  a perp venue is added to `config.yaml:exchanges`.
- Post-migration ceiling: 40-80% simple annual on Bybit spot alone;
  100%+ stacked with funding-arb + market-making.

**Phase 4 — Sustained 1%/day:**
- Requires Phase 3 venue migration + at least 2 months of accumulated
  fine-tune cycles + >1.5 rolling Sharpe.
- On Robinhood alone: structurally impossible. Document this honestly
  every time the question comes up.

---

## 13b. LLM-sole-author architecture (2026-04-30)

The operator decided ACT must run as **agentic_primary**: every trade is
authored by the LLM analyst. The 13 fixed agents + 36 strategies +
genetic engine + ML ensemble remain VOTE INPUTS to the LLM, not
parallel writers to the executor. Three env flags carry this:

| Env (default) | Effect |
|---|---|
| `ACT_AGENTIC_LOOP=1` | Runs `agentic_trade_loop` per asset per tick |
| `ACT_LLM_SOLE_AUTHOR=1` | `_evaluate_entry` short-circuits at the top — technical lane stops calling `_paper.record_entry` / `alpaca_exec.submit_order` directly. Only `submit_trade_plan()` fires orders |
| `ACT_LLM_TECH_BLENDED=1` | When LLM emits `skip` AND tick_state has `agents_consensus + conviction_tier in {sniper, normal} + |agents_net|≥0.15`, agentic_trade_loop auto-promotes the technical signal to a TradePlan. Plan still passes EVERY gate in submit_trade_plan |

All three default-on in both `START_ALL.ps1` (5090 Robinhood) and
`START_ALL_4060.ps1` (4060 Alpaca). To disable any one: `setx ACT_LLM_*
0` then restart.

**Routing matrix after the architecture commit:**

| Asset class | Active venue | Path |
|---|---|---|
| US stock (SPY, QQQ, NVDA, etc.) | alpaca | `submit_trade_plan` → `AlpacaExecutor.submit_order` (real Alpaca paper) |
| BTC/ETH | alpaca / alpaca_crypto | `submit_trade_plan` → `price_source.place_order` (real Alpaca crypto, **NOT** RH paper-sim) |
| BTC/ETH | robinhood | `submit_trade_plan` → `RobinhoodPaperFetcher.record_entry` (RH paper-sim) |
| Other crypto | any | `unsupported_asset` reject |

**4060 brain profile when no remote**: when `OLLAMA_REMOTE_URL` is
unset, `START_ALL_4060.ps1` defaults `ACT_BRAIN_PROFILE=local_8gb`
(qwen3:8b analyst + qwen2.5-coder:7b scanner, ~5.5GB total VRAM). The
`local_8gb` profile is the auto-downgrade tail; it sacrifices analyst
depth for guaranteed local availability. Pulls qwen3:8b in pre-flight.

---

## 13c. Soak-data safety net (2026-04-30)

`scripts/paper_exploration_tick.py` is the OS-scheduler-loop fallback
that fires a small momentum trade when the LLM lane is silent.
Critical wiring details:

- `_last_decision_age_h()` excludes `decision_id LIKE 'shadow-%'` rows.
  Without that, the agentic loop's per-minute shadow writes
  permanently reset the 4-hour quiet-hours gate and exploration never
  fires.
- First call after START_ALL launch passes `--force` so an end-to-end
  trade lands within ~30 seconds (proves the path; subsequent calls in
  the 15-min loop use the normal cooldown).
- RH path (`_submit_robinhood`) writes a `paper_explore_*` warm_store
  row so the MAX_PER_DAY cap and audits work for both venues.

Diagnostic: `python scripts/diagnose_llm_silence.py --hours 6 --tail 20`
prints (1) action histogram, (2) NON-shadow LONG/SHORT count, (3)
exploration firings, (4) skip-reason histogram, (5) latest decisions.
Section 4 is the smoking gun for "LLM is silent — why?".

---

## 13d. MCP server connection (Acer ↔ 5090)

Stable URL: `http://100.127.155.36:9100/mcp` (Tailscale IP, NOT
MagicDNS hostname `act5090` which intermittently 5s-timeouts on the
Acer).

5090 binding: `scripts/start_mcp.ps1` defaults to `BindHost="0.0.0.0"`
so Tailscale-routed peers reach the server. One-time firewall rule:

```
netsh advfirewall firewall add rule name="ACT-MCP-9100" dir=in `
  action=allow protocol=TCP localport=9100 profile=any
```

Acer-side: `.mcp.json` is read at session-launch only. Editing the
file mid-session does NOT reload — operator must `/exit` and run
`claude` (NOT `--continue` or `--resume`) for new URL to take effect.

---

## 14. Recent commits (read this on session start)

```
LLM-AUTHOR-2026-04-30
     - feat(llm-author): ACT_LLM_SOLE_AUTHOR + RH paper-sim unblocked
       (paper_exploration shadow-row filter; technical-lane gate)
     - feat(4060-llm): local_8gb brain profile (qwen3:8b analyst)
     - feat(llm-trades): tech-blended escalation
       (LLM-skip + strong tech signal -> auto-promote TradePlan)
     - fix(routing): BTC/ETH on Alpaca venue routes to alpaca_crypto
       REST (was silently logging to RH paper-sim file)
     - fix(soak): force first paper_exploration trade on launch
     - fix(mcp): Tailscale IP + BindHost=0.0.0.0 default
     - feat(diag): scripts/diagnose_llm_silence.py
C17  — end-to-end wiring + START_ALL alignment
C14  — transient persona agents from knowledge graph
C12  — real-time knowledge graph over live data streams
C15  — /agent-post-mortem counterfactual chat
C13  — Polymarket venue integration (shadow default)
C12b — per-agent episodic memory
C11  — 6 quant models exposed as LLM tools
C10  — dual-brain fine-tune orchestration (QLoRA + champion gate)
C9b  — /diagnose-noop urgent skill
C8   — hooks system
C7b  — brain memory (scanner↔analyst corpus callosum)
C7   — MCP client integration
C6   — skills system
C5d  — three brain profiles, env-switchable
C5c  — DeepSeek-R1 reasoning pair + <think>-stripper
C5b  — research-pair correction
C5   — dual-brain architecture
C4a-d — emergency mode + bandit + bridge + executor shadow hook
C3   — web_context + trade_tools + agentic loop + verifier
C2   — strategy_repository + agentic_context
C1   — warm_store migration + TradePlan model
```

---

## 15. Work NOT yet done (pending roadmap)

- **C9 brain-to-body controller** — scanner score + analyst verdicts
  steer bandit exploration + genetic loop cadence + agent orchestrator
  priority. Next commit.
- **C10b Unsloth backend** — real QLoRA training call (the orchestrator
  + quality filter + champion gate all work now with StubBackend).
  Requires `pip install unsloth` on the GPU box.
- **C16 market-participant simulation** — multi-week; deferred until
  the above stabilize.
- **Bybit venue integration** — needed for sustained 1%/day. Not
  started; funding-arb skeleton already in repo awaiting venue
  connector.

---

## 16. Golden rules for future Claude sessions

1. **LLM is the SOLE author of every trade** (operator directive
   2026-04-30). The 13 fixed agents + math + genetics feed the LLM
   via tick_state / orchestrator output, never as parallel writers.
   Maintained by `ACT_LLM_SOLE_AUTHOR=1` (default-on). If asked to
   change this, push back: it's a settled architecture, not a tweak.
2. **Trades MUST fire automatically once START_ALL runs** — no manual
   force_test_trade required. paper_exploration `--force` on first
   launch + tech-blended escalation guarantee this. If "no trades"
   is reported, run `scripts/diagnose_llm_silence.py` BEFORE
   suggesting code changes.
3. **Never promise 1%/day on Robinhood** — ceiling is ~37%/year simple.
   Venue migration is the load-bearing move, not code tweaks. But
   1%/day is the operator's non-negotiable target — don't lower it,
   show the realistic path.
4. **Always check `memory/` on session start** — `project_realistic_targets`,
   `feedback_upside_framing`, `feedback_realtime_paradigm`, etc.
5. **Disk-side audit before MCP queries** — when MCP is unreachable,
   `scripts/diagnose_llm_silence.py` answers "is the LLM trading?"
   from warm_store.sqlite alone. NON-shadow LONG/SHORT count > 0 means
   yes; only SHADOW_SKIP means no.
6. **Adapt external tools to real-time paradigm** — ACT is streaming,
   not batch. Don't port static-batch patterns.
7. **Feature-by-feature audit before dismissing external tools** —
   MiroFish looked "wrong paradigm" at first glance; rigorous audit
   found 5 genuine wins after reframing.
8. **Default to shadow mode for new venues** — real orders require
   multiple independent gates.
9. **Use the Tailscale IP for MCP, not MagicDNS hostname** —
   `http://100.127.155.36:9100/mcp` is reliable; `http://act5090:9100/mcp`
   intermittently 5s-timeouts on the Acer due to local resolver chain
   stalls. Memory: this session debugged it for an hour; don't repeat.

---

*Last updated: after C17 ship. When you add a commit that changes this
picture, update section 14 at minimum, and the affected numbered
section(s).*
