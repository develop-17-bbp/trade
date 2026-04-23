# Strategy: push ACT paper equity to +1% (≥ $16,610)

Working notes for the /loop task that self-observes every 30 min and evolves the strategy.
Do NOT mutate live config or restart the bot from this doc. Propose, operator approves, then change.

## Baseline (iteration 1 — 2026-04-22 23:41 UTC)

- Initial capital: $16,445.79
- Current equity: $16,430.91 (-0.09%)
- Peak equity: $16,445.79
- Trades closed: 1 (the MCP `recent_trades` view shows 1; paper stats show 2 entries / 2 exits — one is still being joined)
- Win rate: 0/1
- combined_rolling_sharpe_30: 0.0
- Readiness gate: CLOSED (trades 0 < 500, soak 0.0d < 14d, ACT_REAL_CAPITAL_ENABLED unset)
- Alert band: trip at equity ≥ $16,610 (+1%) or ≤ $15,952 (-3%) or sharpe crossing 1.0

## The single closed trade (the only evidence we have)

| field | value |
|---|---|
| asset / dir | ETH / LONG |
| entry / exit | 2336.43 → 2325.47 |
| window | 2026-04-20 18:51 → 21:15 UTC (144 bars held) |
| pnl | -2.16% / -$10.79 |
| score | 12 (top bucket [10,99)) |
| ml_conf | **0.00** (lowest bucket) |
| llm_conf | 1.00 (highest bucket) |
| spread_pct | 0.00 |
| size_pct | 1.00 (full size) |
| exit reason | SL L2 hit (close $2,326.37) |
| llm reasoning | "Enter with caution due to CHoCH and macro risks. Monitor for potential reversals." |

## Hypotheses (ordered by expected lift × confidence)

### H1 — LLM-ML divergence filter (highest priority)

The one observed loss had `llm_conf=1.0` AND `ml_conf=0.0`. That is a structural disagreement: the language model is maximally confident while the gradient-boosted model says no. Allowing those to trade at full size is the most obvious hole.

Proposal (for operator review, not live change):
- Gate: require `min(llm_conf, ml_conf) ≥ 0.35` OR `|llm_conf − ml_conf| ≤ 0.4`
- Half-size when the two disagree by > 0.3 even if both exceed their floor
- Kill entries entirely when `ml_conf < 0.1` regardless of LLM signal — the ML model is explicitly saying "no"

### H2 — "Enter with caution" → full size is wrong

LLM reasoning contained "enter with caution" and "macro risks" yet size_pct=1.0 went through. There is either no caution parser wired in, or it is ignored. Either re-parse the LLM text for caution-markers (regex: "caution", "risk", "monitor", "reversal", "careful") and halve size when present, or teach the prompt to return a structured `size_suggestion` alongside `confidence` so the runtime does not have to guess.

### H3 — 144-bar holding time is too long for a 2.16% SL

144 bars × 1h = 6 days held to eat -2.16%. Either the SL is too far (risk/reward broken) or the exit logic does not re-evaluate when the setup decays. Authority-PDF rules state ETH is day-trade-only — a 6-day hold is itself an authority violation. Verify in `src/executor/` that the day-trade-only enforcement for ETH is actually wired.

### H4 — Too few entries to learn anything

n=1 in 3 days is far below what the readiness gate (500 trades in 14 days = 35/day) needs. Either the Safe-entries gate is over-restrictive (rolling-Sharpe min=1.0 with 0 samples blocks every entry), or the signal emitters are silent. If Safe-entries requires Sharpe ≥ 1.0 but the sample window has zero trades, it can deadlock — check the bootstrap behavior in `src/gates/safe_entries.py` (or equivalent).

### H5 — Observability is off; we can't debug what we can't see

`ACT_METRICS_ENABLED` and `ACT_TRACING_ENABLED` are both unset. Without Prometheus + OTel we are blind to decision-path rejects (how many signals → how many pass each gate → how many get executed). Turn these on so we can attribute where the funnel is collapsing.

## Proposed experiment queue (operator picks order)

1. Enable observability: `setx ACT_METRICS_ENABLED 1 && setx ACT_TRACING_ENABLED 1` + restart. Watch the decision-funnel dashboard for a cycle.
2. Apply H1's strict gate (`min(llm_conf, ml_conf) ≥ 0.35`, size halved on disagreement). Measure entries/day and WR over next 72h.
3. Verify H3 — ETH 144-bar hold either is or is not an authority violation. If it is, the authority enforcement layer is broken and must be fixed before any other experiment.
4. Diagnose H4 — if Safe-entries gate is rejecting everything because of the sharpe bootstrap, relax the floor only until N=30 trades exist.

## What we are NOT doing until operator approves

- Not touching any `setx` env var
- Not restarting the bot
- Not editing strategy code
- Not forcing retrain

## Code locations identified (for when operator says "go")

Mapped by grep across `src/`:

| Hypothesis | File(s) | Key line / note |
|---|---|---|
| H1 — LLM/ML divergence | `src/trading/executor.py:382` | `self.llm_conf_threshold = ai_cfg.get('llm_trade_conf_threshold', 0.40)` — there is an LLM floor (0.4) but no paired ML floor and no divergence check |
| H1 — ml_conf plumbing | `src/trading/executor.py:6878` | `ml_confidence if 'ml_confidence' in dir() else 0` — defaults to 0 when the variable wasn't defined in scope. This is why the one closed trade shows `ml_conf=0.0`: the ML gate output may never be propagated. Worth verifying before adding the divergence filter — the fix may be "wire ml_confidence through" rather than "add a new gate." |
| H2 — "caution" parser | `src/trading/executor.py` LLM reasoning path (lines 6876-6881) | `reasoning[:200]` is stored but not parsed for caution-markers before sizing |
| H3 — ETH day-trade-only | `src/ai/authority_rules.py`, `src/agents/authority_compliance_guardian.py`, `src/ai/authority_context.py` | Authority enforcement layer — check ETH hold-time check here; the observed 144-bar (6-day) hold suggests this rule is either missing or bypassed |
| H4 — Safe-entries bootstrap | `src/trading/safe_entries.py` | Gate needs inspection for sharpe-floor behavior when samples < window |
| H5 — observability | `ACT_METRICS_ENABLED`, `ACT_TRACING_ENABLED` env flags (both unset per `component_state`) | No code change needed — one-line setx each + restart |

## Iteration log

| # | timestamp (UTC) | equity | sharpe | notes |
|---|---|---|---|---|
| 1 | 2026-04-22 23:41 | 16430.91 | 0.00 | baseline seeded; no alerts |
| 2 | 2026-04-23 00:11 | 16430.91 | 0.00 | no change; no alerts. Mapped code locations for H1–H5. Key finding: `executor.py:6878` defaults `ml_confidence` to 0 when the variable isn't in scope — explains the `ml_conf=0.0` on the only recorded trade. Pending operator "go" on the standing-authorization reframe before any edits. |
| 3 | 2026-04-23 00:18 | 16430.91 | 0.00 | Operator said "go" (-1% fence @ $16,281 active). Two edits shipped — both strictly risk-reducing, no trading-behavior increase: (A) `executor.py` — fix `ml_confidence` plumbing: now populated from `ml_context['meta_prob']` or fallback `lgbm_confidence`, was always 0.0 due to scope bug. (B) `executor.py` — caution-marker parser before final size clamp: 2+ caution keywords halve size, 1 keyword → 0.75x. Never blocks, never increases. Would have halved the ETH losing trade's size (its reasoning had 4 markers: caution/risk/reversal/macro). Next: commit + restart bot. |
