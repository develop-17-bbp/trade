# Dual-Brain Setup on RTX 5090 (32 GB VRAM)

## Three profiles, A/B switchable (C5d)

Three named brain profiles in `config.yaml`. Pick via
`ai.dual_brain.profile: <name>` or the `ACT_BRAIN_PROFILE` env var.
Per-role env overrides (`ACT_SCANNER_MODEL` / `ACT_ANALYST_MODEL`)
still win over profile defaults.

| Profile | Analyst | Scanner | Combined VRAM | When to use |
|---|---|---|---|---|
| **`dense_r1`** (default) | deepseek-r1:32b | deepseek-r1:7b | ~25 GB | Paper soak. Most consistent output; fewest variance sources while you validate the system. |
| **`moe_agentic`** | qwen3-coder:30b (MoE) | qwen2.5-coder:7b | ~24 GB | Tool-use heavy workflows + MCP. Fastest; 2026 agentic gold standard. |
| **`hybrid`** | qwen3-coder:30b (MoE) | deepseek-r1:7b | ~24 GB | Post-soak recommendation. MoE speed for agentic tool use + dense reasoning scanner for tick-cadence consistency. |

All three:
* Fit on RTX 5090 32 GB with KV cache headroom.
* Require `OLLAMA_NUM_PARALLEL=4` for concurrent scanner+analyst inference.
* Auto-strip `<think>...</think>` tags from scanner output (via
  `src/ai/dual_brain.py::strip_reasoning_tags`) so reasoning traces don't
  bloat brain_memory. Analyst keeps its full trace for audit.

### A/B workflow on the GPU box

```powershell
# Week 1 — default (dense reasoning):
# do nothing; config.yaml profile: dense_r1 is the default
# restart bot, run paper soak for 7 days

# Week 2 — MoE agentic:
setx ACT_BRAIN_PROFILE moe_agentic
# restart bot + ollama

# Week 3 — hybrid:
setx ACT_BRAIN_PROFILE hybrid
# restart bot
```

Compare profiles in warm_store afterward:

```sql
SELECT
  json_extract(component_signals, '$.analyst_model') AS profile_analyst,
  COUNT(*) AS shadow_plans,
  AVG(CASE WHEN final_action LIKE 'SHADOW_LONG' THEN 1 ELSE 0 END) AS long_rate
FROM decisions
WHERE decision_id LIKE 'shadow-%'
  AND ts_ns >= (strftime('%s', 'now', '-7 days') * 1000000000)
GROUP BY profile_analyst;
```

## Pull on the GPU box

```bash
ollama pull deepseek-r1:32b
ollama pull deepseek-r1:7b
```

## Parallelism (important)

Ollama by default serializes requests to a single model. ACT's dual-brain
wants scanner and analyst to run concurrently when needed. Set:

```bash
# Windows (PowerShell)
setx OLLAMA_NUM_PARALLEL 4
# or per-session: $env:OLLAMA_NUM_PARALLEL="4"

# Linux / WSL
export OLLAMA_NUM_PARALLEL=4
```

Then restart `ollama serve`. The 5090's 1792 GB/s bandwidth handles
this without a tok/s hit.

## Long-context Modelfiles

Default Ollama context is small. Crank it up so the Analyst can reason
over hours of market data + brain_memory traces:

```Modelfile
FROM deepseek-r1:32b
PARAMETER num_ctx 32000
PARAMETER temperature 0.4
```

Save as `Modelfile.deepseek-r1-32b-act` and build:

```bash
ollama create deepseek-r1:32b-act -f Modelfile.deepseek-r1-32b-act
```

Then set in `config.yaml` (or env):
```yaml
ai:
  dual_brain:
    analyst_model: deepseek-r1:32b-act
```

Same pattern for the scanner (usually unnecessary — scanner takes a
tight snapshot per tick, 4-8k context is plenty).

## Alternative pairs

Switch via `config.yaml` or env (`ACT_SCANNER_MODEL` / `ACT_ANALYST_MODEL`):

**Agentic-tool-use specialized (maximum MCP compatibility):**
```yaml
ai:
  dual_brain:
    analyst_model: qwen3-coder:30b      # gold standard for JSON tool calls
    scanner_model: qwen2.5-coder:7b     # coder-tuned worker
```
Combined ~27 GB. Less reasoning depth than DeepSeek-R1 but better JSON.

**Originally specified (Devstral 24B + Qwen 3 32B):**
```yaml
ai:
  dual_brain:
    analyst_model: qwen3:32b       # ~20 GB @ Q4_K_M
    scanner_model: devstral:24b    # ~14 GB @ Q4_K_M
```
Combined ~34 GB — oversubscribed on 32 GB, Ollama will model-swap.

## MCP wired in

`src/ai/mcp_client_registry.py` (C7) lets the Analyst call external MCP
servers. qwen3-coder is MCP-optimized out of the box. Add entries in
`config.yaml:mcp_clients` to connect CoinGecko / Perplexity / Chroma /
community TradingView servers.

## Verify it's working

```bash
# Confirm both models are pulled and reachable:
ollama list | grep -E "deepseek-r1"

# Dry-run the agentic loop with the live models:
export ACT_AGENTIC_LOOP=1
python -m src.ai.agentic_bridge --asset BTC --max-steps 4
# Expect JSON LoopResult; terminated_reason in {"plan","skip","max_steps"}

# Tail warm_store for shadow plans produced during a paper run:
sqlite3 data/warm_store.sqlite \
  "SELECT decision_id, symbol, final_action, plan_json \
   FROM decisions WHERE decision_id LIKE 'shadow-%' \
   ORDER BY ts_ns DESC LIMIT 5;"
```
