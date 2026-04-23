# Dual-Brain Setup on RTX 5090 (32 GB VRAM)

## Current default pair — reasoning on both sides (C5c)

```
Analyst (orchestrator)  deepseek-r1:32b   Q4_K_M   ~20 GB   reasoning-focused
Scanner (worker)        deepseek-r1:7b    Q4_K_M    ~5 GB   reasoning + fast
────────────────────────────────────────────────────────────────────────
Combined                                           ~25 GB   (fits w/ KV cache)
```

Both are **DeepSeek-R1 distills** — they preserve R1's `<think>`
chain-of-thought trace into a smaller Qwen backbone. Full DeepSeek-V3 /
R1 (671B MoE) doesn't fit on a 5090.

**Scanner-side `<think>` stripping:** the scanner runs every tick (60-180s).
Its raw output can include a 1000+ token reasoning trace that would bloat
`brain_memory` and the analyst's seed context. `dual_brain.py` automatically
strips `<think>...</think>` from scanner output (controlled by
`DEFAULT_STRIP_THINK_TAGS_FROM_SCANNER`), keeping only the final JSON.

Analyst keeps its full reasoning trace — it's only called on-demand when
the scanner flags a setup, and the trace is valuable audit data.

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
