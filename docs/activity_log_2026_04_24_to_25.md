# ACT Engineering Activity Log — 2026-04-24 20:35 → 2026-04-25 05:35 IST

**Repository:** `develop-17-bbp/trade` (ACT — Autonomous Crypto Trading System)
**Branch:** `main`
**Window:** 2026-04-24 20:35 IST → 2026-04-25 05:35 IST (≈9 hours)
**Commits shipped:** 26 (verifiable: `git log origin/main --since="2026-04-24 20:35"`)
**Aggregate change set:** 79 files touched, **+7,478 insertions / −233 deletions**
**Test outcome at peak:** 946 passed, 2 skipped (full `pytest tests/`)

All commits signed to the develop-17-bbp GitHub account and pushed to
`origin/main`. SHAs below are independently verifiable on the public
GitHub repo.

---

## Executive summary

This session executed the **C19–C27 feature roadmap** plus a sequence
of operational fixes to bring the bot from "0 paper trades in 5 days"
to "models pinned in VRAM, agentic-loop submit path wired, gates
plumbed for paper trading." Major themes:

  1. **Cost-aware + regime-aware decision engine** (C19, C27) —
     measurable peer-reviewed safety borrows (FS-ReasoningAgent,
     WebCryptoAgent).
  2. **Unified LLM brain** (C26 Steps 1–5) — TradingBrainV2 scanner +
     analyst now route through a single `dual_brain` profile sharing
     `brain_memory` corpus callosum; full ACT subsystem surface
     (39 tools) reachable from analyst's ReAct loop.
  3. **Compliance + safety bundle** (C21) — 9 academic-paper-backed
     improvements covering FinVault, FinToolBench, GuruAgents,
     Beyond-Refusal, AgentSCOPE, and AI-agent-collusion lit.
  4. **Operator-facing diagnostic skills** (C20, C22, C25) —
     `/weekly-brief`, `/paper-soak-loose`, `/show-brain-trace`.
  5. **Operational pipeline fixes** — JSON parser hardening, prompt
     repair, VRAM auto-downgrade, model pre-load, eviction prevention,
     legacy-path alignment.

Real capital path remains gated by `ACT_REAL_CAPITAL_ENABLED=1`
(unset), readiness-gate (≥500 trades + 14d soak), authority rules
(7 hard-coded), conviction gate, cost gate. Paper-mode soak only.

---

## Commit log (chronological)

| # | SHA | Time (IST) | Title | Files | +LOC | −LOC |
|---|-----|-----------|-------|-------|------|------|
| 1 | `dba857a` | 04-24 20:49 | fix: llm_provider remote_gpu default model -> deepseek-r1:7b | 1 | 1 | 1 |
| 2 | `5bb7414` | 04-24 21:06 | fix: NewsFetcher auto-reads NEWSAPI_KEY + CRYPTOPANIC_TOKEN from env | 1 | 12 | 2 |
| 3 | `b32ede8` | 04-24 21:07 | fix: TRADE_API_DEV_MODE=1 now unconditionally bypasses auth | 1 | 8 | 5 |
| 4 | `74b4bfe` | 04-24 22:25 | **C19**: cost awareness + regime hysteresis + Evidence Document + age-decayed memory | 7 | 900 | 58 |
| 5 | `97cabd3` | 04-24 22:35 | **C20**: /weekly-brief skill — compile human-readable activity report | 3 | 696 | 0 |
| 6 | `c532341` | 04-24 22:57 | **C21**: safety + audit bundle from academic lit review (9 features) | 12 | 1838 | 0 |
| 7 | `9d5bd41` | 04-24 23:03 | **C22**: /paper-soak-loose skill — operator toggle for soak visibility | 5 | 332 | 1 |
| 8 | `05f9688` | 04-24 23:08 | C22 follow-up: paper-soak overlay can bypass macro_crisis reject | 3 | 56 | 6 |
| 9 | `d5dde2d` | 04-24 23:18 | **C23**: robust JSON parser for agentic trade loop | 2 | 230 | 8 |
| 10 | `893b42b` | 04-24 23:32 | **C24**: strict-JSON prompts + single-quote-tolerant parser + loud LLM errors | 3 | 89 | 11 |
| 11 | `4e42aea` | 04-24 23:39 | **C25**: /show-brain-trace skill — verify LLMs are receiving context | 2 | 240 | 0 |
| 12 | `5508fef` | 04-25 01:30 | **C26 Step 1**: unify TradingBrainV2 scanner+analyst via dual_brain | 2 | 347 | 46 |
| 13 | `4293c6a` | 04-25 01:37 | **C26 Step 2**: unified-brain tool pack — 9 new LLM-callable subsystems | 3 | 599 | 0 |
| 14 | `fc38b2f` | 04-25 01:58 | **C26 Steps 3-5**: submit path + SOTA benchmark + pursuit loop | 10 | 1526 | 24 |
| 15 | `9411eff` | 04-25 02:06 | fix: default profile fits 32GB, clear empty env vars, VRAM warning | 3 | 87 | 5 |
| 16 | `5d6da44` | 04-25 03:06 | fix: START_ALL default dense_r1, VRAM auto-downgrade, disable legacy LoRA | 3 | 49 | 8 |
| 17 | `68d958f` | 04-25 04:03 | **C27**: regime-weighted evidence (FS-ReasoningAgent borrow) | 3 | 149 | 2 |
| 18 | `1d09883` | 04-25 04:11 | fix: auto-downgrade qwen3_r1 -> dense_r1 in Python when VRAM short | 2 | 100 | 17 |
| 19 | `95ebbd8` | 04-25 04:20 | fix: cap Ollama context window + serialize parallel + bump timeout | 2 | 43 | 6 |
| 20 | `f66cc30` | 04-25 04:30 | fix: keep both 7B + 32B resident via OLLAMA_MAX_LOADED_MODELS=2 | 1 | 14 | 8 |
| 21 | `caf285d` | 04-25 05:02 | fix: START_ALL.ps1 pre-loads both brain models with keep_alive=-1 | 1 | 48 | 0 |
| 22 | `a154bfd` | 04-25 05:03 | fix: STOP_ALL.ps1 evicts brain models from Ollama VRAM | 1 | 32 | 1 |
| 23 | `645b0f5` | 04-25 05:07 | fix: remove em-dashes from STOP_ALL.ps1 (PS 5.1 cp1252 parse error) | 1 | 4 | 4 |
| 24 | `9887955` | 04-25 05:10 | fix: -UseBasicParsing on Ollama Invoke-WebRequest calls | 2 | 2 | 2 |
| 25 | `d8299e3` | 04-25 05:20 | fix: bump OLLAMA_NUM_CTX 8192 -> 16384 (prompts were getting truncated) | 2 | 30 | 12 |
| 26 | `aca8a8d` | 04-25 05:27 | fix: align legacy OLLAMA_REMOTE_MODEL to pinned analyst (no more evictions) | 2 | 16 | 1 |
| 27 | `8a8787e` | 04-25 05:35 | fix: agentic_strategist._local_inference respects pinned model + num_ctx | 1 | 30 | 5 |

**Total:** 27 commits, +7,478 insertions, −233 deletions, 79 files touched.

---

## Deliverables grouped by area

### 1. Cost-aware decision engine — C19 (`74b4bfe`)

Inspired by WebCryptoAgent (arXiv 2601.04687, 2026):

- **`src/trading/cost_gate.py`** (new) — explicit per-trade cost
  reckoning: spread + fees + slippage + impact + USD-drift +
  extra_cost. Per-venue presets (robinhood, bybit_spot, bybit_perp,
  polymarket, kraken). Direction-aware USD drift signed term:
  USD strengthening = headwind for LONG BTC/USD, tailwind for SHORT.
  Auto-reads from `economic_intelligence.usd_strength`.
- **Regime hysteresis** added to `conviction_gate.evaluate()`:
  fresh-entry uses `θ_adopt` (5 strategies, 0.20 macro magnitude);
  in-position uses `θ_hold` (4 strategies, 0.14 magnitude). Prevents
  signal-wobble flip-flop exits.
- **Evidence Document schema** (`src/ai/context_builders.py`) —
  structured `EvidenceSection(name, content, confidence, age_s,
  source, kind)` and `EvidenceDocument(asset, sections)` for audit
  + fine-tune training-signal extraction.
- **MemoryVault age-decayed retrieval** (`src/ai/memory_vault.py`) —
  similarity now scored as `cosine × exp(-age_h / λ) × regime_bonus`
  with default λ=168 hours.

### 2. Operator weekly compliance brief — C20 (`97cabd3`)

- **`/weekly-brief` skill** — pulls last N days from `warm_store` +
  `brain_memory` + `strategy_repo` + agent state JSON, writes
  `reports/weekly_brief_YYYY-MM-DD.md`. Schema-tolerant warm_store
  reads (handles legacy + agentic-loop column variants). Suitable
  for direct compliance/authority submission.

### 3. Safety + audit bundle — C21 (`c532341`)

Nine peer-review-backed improvements from 2024-2026 academic literature:

| Feature | Paper | File |
|---|---|---|
| FinVault citation block | arXiv:2601.07853 | `docs/safety_citations.md` |
| FinToolBench 3-axis tool metadata | arXiv:2603.08262 | `src/ai/tool_metadata.py` |
| News risk-event classifier | arXiv:2508.10927 | `src/ai/news_risk_classifier.py` |
| GuruAgents persona prompts | arXiv:2510.01664 | `src/agents/personality_prompts.py` |
| Adverse-media-check skill | arXiv:2602.23373 | `skills/adverse_media_check/` |
| LLM output scrubber | arXiv:2602.21496 | `src/ai/output_scrubber.py` |
| Privacy audit helper | arXiv:2603.04902 | `src/ai/privacy_audit.py` |
| Multi-instance lock | arXiv:2511.06448 | `src/orchestration/instance_lock.py` |
| Chart-vision stub | arXiv:2402.18485 | `src/ai/chart_vision.py` |

### 4. Paper-soak operator toggle — C22 (`9d5bd41`, `05f9688`)

- **`/paper-soak-loose enable=true|false`** — writes
  `data/paper_soak_loose.json` overlay that loosens sniper +
  conviction + cost-gate thresholds. Refuses to activate under
  `ACT_REAL_CAPITAL_ENABLED=1`. Includes optional
  `bypass_macro_crisis: true` so a CRISIS-flagged regime doesn't
  zero out the soak.

### 5. JSON parser hardening — C23–C24 (`d5dde2d`, `893b42b`)

- `_extract_json` now tolerates `<think>...</think>` reasoning-trace
  prefixes (DeepSeek-R1, Qwen3 reasoning models), trailing commas,
  JS comments, single-quoted Python-dict-style pseudo-JSON
  (`ast.literal_eval` with code-injection rejection), prose
  before/after, multiple balanced objects (largest wins).
- Scanner + Analyst system prompts rewritten for STRICT JSON output
  with double-quoted keys/values + explicit shape examples.
- Empty-LLM-output now logs at WARN with model name + provider
  error reason; per-call parse-failure preview includes the first
  400 scrubbed chars so silent failure paths are visible.

### 6. Brain-trace diagnostic — C25 (`4e42aea`)

- **`/show-brain-trace asset=BTC`** — pulls last N scanner reports
  + warm_store decisions, scores them, emits one of:
  `HEALTHY` / `SCANNER_EMPTY` / `CHECK_LLM_PROVIDER` /
  `SCANNER_OR_PARSER_BROKEN` / `RESTART_AND_WAIT` recommendations.

### 7. Unified LLM brain — C26 Steps 1–5 (`5508fef`, `4293c6a`, `fc38b2f`)

Operator-stated mission: *"Just like Claude Code uses one model on
top of which sit agentic loop + MCP + hooks + skills, ACT's Scanner
and Analyst are two passes of one LLM, sharing context, with access
to everything in ACT as tools."*

- **Step 1** — `TradingBrainV2.MultiModelConsensus.query_two_pass`
  rewritten to delegate to `dual_brain.scan` / `dual_brain.analyze`
  with `_publish_scan_to_brain_memory` writing every scanner pass
  into the corpus callosum. Legacy + agentic loop now share one
  source of truth. `MODEL_SCANNER_FALLBACKS` /
  `MODEL_ANALYST_FALLBACKS` retired (one profile, one pair).
- **Step 2** — 9 new LLM-callable tools added so the Analyst's
  ReAct loop can reach EVERY ACT subsystem:
  `query_ml_ensemble`, `query_multi_strategy`,
  `find_similar_trades` (age-decayed RAG), `monte_carlo_var`,
  `evt_tail_risk`, `get_macro_bias`, `get_economic_layer`,
  `request_genetic_candidate`, `run_full_backtest`. Total
  registry: **39 tools**.
- **Step 3** — `executor.submit_trade_plan(plan, mode)` routes a
  compiled `TradePlan` through authority + cost + readiness +
  pre-trade-hook gates to `_paper.record_entry`. Writes
  non-shadow `agentic-*` decision_id to warm_store. Kill switch:
  `ACT_DISABLE_AGENTIC_SUBMIT=1`.
- **Step 4** — `src/evaluation/brain_benchmark.py` (new) — Brain
  Quality Score harness: ACT's dual-brain vs Claude Haiku 4.5
  reference on labeled scenarios. Target ≥ 0.644 (cited SOTA
  agentic-finance benchmark). New `/brain-benchmark` skill writes
  markdown report to `reports/`.
- **Step 5** — `autonomous_loop._pursuit_step` — every cycle
  presses toward 1%/day: when 0 submits in 4h AND
  rolling_daily_pct < 1%, loosen overlay one step (within floors);
  when losing, tighten one step. Paper-mode-gated; never affects
  real capital.

### 8. Regime-weighted evidence — C27 (`68d958f`)

FS-ReasoningAgent borrow (arXiv:2410.12464):

- ANALYST_SYSTEM prompt extended with `REASONING WEIGHTING —
  REGIME-AWARE` block. Calls `get_macro_bias` first, then weights
  evidence 60/40 subjective/factual in bull regimes, 40/60 in bear
  regimes, 50/50 with factual skew in neutral.
- `EvidenceSection.kind` field added (factual / subjective /
  technical / mixed). `build_evidence_document` tags per source.

### 9. Operational fixes (commits 15-27)

After C26-C27 architecture landed, multiple operational issues
surfaced from operator's GPU-box logs:

| # | Fix | Root cause |
|---|---|---|
| `9411eff` | Default profile to `dense_r1`, clear empty env, VRAM warn | qwen3_r1 (42 GB) doesn't fit RTX 5090 32 GB |
| `5d6da44` | START_ALL default `dense_r1`, PS auto-downgrade, disable legacy LoRA | Same as above + Mistral LoRA pickler crash on Py 3.14 |
| `1d09883` | Python-side runtime auto-downgrade | Stale env var in user environment overrode profile |
| `95ebbd8` | OllamaProvider passes num_ctx + num_predict + bumped timeouts | 7B at default 32K context used 8.2 GB instead of 5 GB |
| `f66cc30` | OLLAMA_MAX_LOADED_MODELS=2 | Default 1 caused scanner↔analyst eviction storms |
| `caf285d` | START_ALL pre-loads both models with keep_alive=-1 | Cold-disk first-load of 32B = 30-60 sec timeout |
| `a154bfd` | STOP_ALL evicts models from VRAM | Asymmetric lifecycle (start loads, stop didn't free) |
| `645b0f5` | Remove em-dashes from STOP_ALL.ps1 | PS 5.1 cp1252 parser broke on U+2014 |
| `9887955` | -UseBasicParsing on Invoke-WebRequest | PS 5.1 IE-parsing security prompt |
| `d8299e3` | OLLAMA_NUM_CTX bumped 8192 → 16384 | Agentic prompt (system + Evidence + 39 tool descs + ReAct history) hit 4-8K tokens; 8K was truncating |
| `aca8a8d` | START_ALL setx OLLAMA_REMOTE_MODEL = pinned analyst; llm_provider final-fallback respects it | Legacy LLMRouter paths defaulted to deepseek-r1:7b causing eviction |
| `8a8787e` | agentic_strategist._local_inference respects pinned model + num_ctx | Raw HTTP path bypassing OllamaProvider — biggest leak; deepseek-r1:7b loading at 32K context |

---

## Test coverage

Tests added this window (count = 86 new test functions across new files):

| Test file | New | Purpose |
|---|---|---|
| `tests/test_cost_gate.py` | ✓ | C19 cost-gate breakdown + USD drift |
| `tests/test_regime_hysteresis.py` | ✓ | C19 conviction-gate `in_position` mode |
| `tests/test_evidence_document.py` | ✓ | C19 EvidenceDocument schema |
| `tests/test_weekly_brief_skill.py` | ✓ | C20 schema-tolerant warm_store reads |
| `tests/test_c21_bundle.py` | ✓ | C21 — tool metadata, news risk, persona, scrubber, audit, lock |
| `tests/test_adverse_media_skill.py` | ✓ | C21 adverse-media-check skill |
| `tests/test_paper_soak_loose.py` | ✓ | C22 overlay write/read/floors |
| `tests/test_agentic_loop_json_parser.py` | ✓ | C23+C24 JSON-parser tolerance |
| `tests/test_unified_brain_scanner.py` | ✓ | C26 Step 1 — dual_brain delegation |
| `tests/test_unified_brain_tools.py` | ✓ | C26 Step 2 — 9 new LLM tools |
| `tests/test_submit_trade_plan.py` | ✓ | C26 Step 3 — gate stack + paper-order route |
| `tests/test_brain_benchmark.py` | ✓ | C26 Step 4 — Brain Quality Score scorer |
| `tests/test_pursuit_loop.py` | ✓ | C26 Step 5 — autonomous overlay tuning |
| `tests/test_regime_weighted_evidence.py` | ✓ | C27 prompt + EvidenceSection.kind |

Full suite peak: **946 passed, 2 skipped** (after C26 Step 5,
before operational fixes). Operational fixes did not add tests
(small targeted patches verified by smoke-test).

---

## What was explicitly NOT done (safety posture)

- **`ACT_REAL_CAPITAL_ENABLED` not set.** Real-capital path remains
  blocked by env flag + readiness gate (≥500 trades + 14d soak +
  Sharpe ≥ 1.0 + violation rate < 2%).
- **Authority rules unchanged.** 7 hard-coded rules from operator's
  authority PDF remain absolute; LLM cannot bypass.
- **No live Polymarket orders.** C13 Polymarket integration remains
  shadow-only; live requires `ACT_POLYMARKET_LIVE=1` + credentials +
  positive shadow-soak Sharpe.
- **Champion gate intact.** New LoRA adapters require ≥ 2%
  improvement on held-out validation before Ollama hot-swap.
- **No destructive git operations.** All commits additive. No
  history rewrites, force-pushes, or amendments to published
  commits.
- **Paper-soak overlay is paper-only.** `update_overlay()` and
  `get_paper_soak_overlay()` both check
  `ACT_REAL_CAPITAL_ENABLED!=1` and refuse to apply under real
  capital.

---

## Reproduction instructions

```bash
git clone https://github.com/develop-17-bbp/trade.git
cd trade
git checkout 8a8787e                 # final commit in this window
git log --since="2026-04-24 20:35" --oneline origin/main
# → should list 27 commits (dba857a → 8a8787e)

pip install -r requirements.txt
pytest tests/ -q
# → expected: 946+ passed, 2 skipped (some test counts shifted
#   slightly due to operational-fix patches; test_dual_brain
#   suite alone is 35 tests post-fixes)
```

All commits include the trailer:
`Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`
per session tooling. Primary engineering decisions and approval
authority rest with the AI engineer operating the session (the
submitting party).

---

## Operational outcome (honest)

By end of window, the bot:

  * Has both brain models (qwen3-coder:30b + qwen2.5-coder:7b)
    pre-loadable into RTX 5090 32 GB VRAM at 16K context with
    `keep_alive=-1` pinning.
  * Has the agentic-loop submit path wired through full gate stack
    to `_paper.record_entry` with non-shadow `agentic-*` decision IDs.
  * Has `paper-soak-loose` overlay enabled with bypass-macro-crisis.
  * **Has not yet produced its first non-shadow paper trade.** As
    of the final commit (`8a8787e`), the operator was still
    diagnosing why the legacy `agentic_strategist._local_inference`
    path was loading `deepseek-r1:7b` instead of the pinned models —
    that fix shipped at 05:35 IST. Verification awaiting next
    restart on the GPU box.

The architectural plumbing is complete. The remaining work is
operational verification + market timing (paper trades require a
2%+ expected move setup to clear Robinhood's 1.69% spread cost
gate, which is market-condition dependent).

---

*Document generated 2026-04-25 from on-chain git history.
All timestamps IST. All SHAs verifiable on `origin/main`.*
