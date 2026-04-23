# Dual-Brain Setup on RTX 5090 (32 GB VRAM)

## Model pair (2026 research-default)

```
Analyst (orchestrator)  qwen3-coder:30b   Q5_K_M   ~22 GB   ~100 tok/s
Scanner (worker)        qwen2.5-coder:7b  Q5_K_M    ~5 GB   ~150 tok/s
────────────────────────────────────────────────────────────────────────
Combined                                           ~27 GB   (fits w/ KV cache)
```

Both Qwen-family → same tokenizer → cleaner context sharing through
`brain_memory.py`.

## Pull on the GPU box

```bash
ollama pull qwen3-coder:30b
ollama pull qwen2.5-coder:7b
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
FROM qwen3-coder:30b
PARAMETER num_ctx 32000
PARAMETER temperature 0.3
```

Save as `Modelfile.qwen3-coder-act` and build:

```bash
ollama create qwen3-coder:act -f Modelfile.qwen3-coder-act
```

Then set in `config.yaml` (or env):
```yaml
ai:
  dual_brain:
    analyst_model: qwen3-coder:act
```

Same for the scanner if you want extended context (usually unnecessary —
scanner takes a tight snapshot per tick).

## Alternative pair (originally specified)

If you want to stay on Devstral 24B + Qwen 3 32B:

```yaml
ai:
  dual_brain:
    analyst_model: qwen3:32b       # ~20 GB @ Q4_K_M
    scanner_model: devstral:24b    # ~14 GB @ Q4_K_M
```

Combined ~34 GB — oversubscribed on 32 GB, so Ollama will model-swap.
Works but slower than the research-default pair.

## MCP wired in

`src/ai/mcp_client_registry.py` (C7) lets the Analyst call external MCP
servers. qwen3-coder is MCP-optimized out of the box. Add entries in
`config.yaml:mcp_clients` to connect CoinGecko / Perplexity / Chroma /
community TradingView servers.

## Verify it's working

```bash
# Confirm both models are pulled and reachable:
ollama list | grep -E "qwen3-coder|qwen2.5-coder"

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
