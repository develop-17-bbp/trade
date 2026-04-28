# Fine-Tune VRAM Strategy on RTX 5090 (32 GB)

**Status:** Operational guide. The `/fine-tune-brain` skill exists with a stub backend; this doc explains how to flip it to the real Unsloth backend without OOM-ing the live serving stack.

---

## VRAM accounting

### Current live serving (running ACT)

| Component | VRAM (Q4 quantization) |
|---|---|
| qwen3-coder:30b analyst (`keep_alive=-1`) | ~18-20 GB |
| qwen2.5-coder:7b scanner | ~5-6 GB |
| PyTorch ML models (LSTM, PatchTST, RL) | ~1-2 GB |
| MiniLM embedder | 0 GB (CPU-pinned) |
| **Live total** | **~25 GB / 32 GB** |

### QLoRA fine-tune of a 30B model

| Component | VRAM |
|---|---|
| 4-bit base weights | ~16 GB |
| LoRA adapter (rank 8-16) | ~50 MB |
| 8-bit AdamW optimizer state | ~1-2 GB |
| Activations (batch=1, seq=2048) | ~4-8 GB |
| **QLoRA-30B total** | **~22-26 GB** |

### QLoRA fine-tune of a 7B model

| Component | VRAM |
|---|---|
| 4-bit base weights | ~4 GB |
| LoRA adapter | ~50 MB |
| Optimizer state | ~500 MB |
| Activations | ~1-2 GB |
| **QLoRA-7B total** | **~6 GB** |

---

## Verdict

| Scenario | Fits in 32 GB? |
|---|---|
| Concurrent: 30B serving + 30B QLoRA | ❌ ~50 GB needed |
| Concurrent: 30B serving + 7B QLoRA | ✅ ~25 GB + 6 GB = 31 GB (tight, viable) |
| Sequential: pause 30B serving → run 30B QLoRA → restart | ✅ Bot offline ~30-45 min |
| Concurrent: 7B serving + 30B QLoRA | ❌ Same OOM problem (30B QLoRA dominates) |

---

## Recommended path

### Phase 1: Concurrent 7B scanner fine-tune (no downtime)

The scanner (qwen2.5-coder:7b) is the model most exposed to data contamination — it sees every tick's market state and emits a ScanReport. Fine-tuning it on filtered winning ScanReports is high-value AND fits concurrently with 30B serving.

**Steps:**
1. `pip install unsloth` on the GPU box (Windows-compatible build).
2. Confirm 100+ quality-filtered scanner outputs accumulate in warm_store (already automatic via training_data_collector).
3. Run the existing `/fine-tune-brain` skill with target=`scanner`:
   ```cmd
   python -m src.skills.cli run fine-tune-brain target=scanner confirm=true dry_run=false
   ```
4. Champion gate validates the new adapter beats the old by ≥2% before hot-swap.
5. **No bot downtime** — 30B analyst keeps serving the whole time.

**Expected effect:** Better scan-quality → fewer false-positive ScanReports → less wasted analyst calls. Estimated +0.05-0.10 Sharpe on a 100-trade window.

### Phase 2: Nightly 30B analyst fine-tune (sequential, bot pauses)

Once the scanner pipeline is proven, flip the analyst's nightly cycle on:
1. Operator (or scheduler) calls `/emergency-flatten` to stop bot.
2. ollama unloads serving models (`ollama stop qwen3-coder:30b qwen2.5-coder:7b`).
3. Fine-tune skill runs the 30B analyst QLoRA cycle (~30-45 min).
4. Champion gate validates.
5. ollama loads the new (or kept) model.
6. Bot restarts via `START_ALL.ps1`.

**Bot downtime per cycle:** ~30-45 min. Schedule during low-activity window (Asia early hours UTC, ~02:00-04:00 UTC = ~07:30-09:30 IST).

**Expected effect:** Compounding improvement — the analyst learns from THIS operator's THIS venue's THIS market regime's actual outcomes, not just generic pretraining. +0.1-0.3 Sharpe per cycle, then plateaus.

### Phase 3 (optional): When real-capital is enabled

Real-capital readiness requires Sharpe ≥ 1.0 over 500-trade rolling window. If fine-tuning is what unlocks Sharpe ≥ 1.0, gate Phase 2 to run only when the rolling Sharpe is stable enough that the soak counter clears the readiness gate.

---

## Operational guards (already built, just need flipping)

| Guard | Existing module | Status |
|---|---|---|
| Champion gate (≥2% improvement required) | `src/ai/champion_gate.py` | ✓ wired |
| Quality-filtered training data | `src/ai/training_data_filter.py` + `training_data_collector.py` | ✓ wired |
| QLoRA orchestrator | `src/ai/dual_brain_trainer.py` | ✓ stub backend, needs Unsloth real backend |
| Skill: `/fine-tune-brain` | `skills/fine_tune_brain/` | ✓ wired, `dry_run=true` default |
| LoRA adapter promotion (hot-swap) | `src/ai/champion_gate.py::promote_adapter` | ✓ wired |

The infrastructure is built. All that's missing is the actual Unsloth library install + flipping `dry_run=false`.

---

## Risks / mitigations

| Risk | Mitigation |
|---|---|
| Unsloth Windows install fails | Fall back to `transformers + peft` (slower but works). Operator can document which path succeeded. |
| Phase 1 7B fine-tune OOMs anyway (5090 VRAM fragmentation) | Reduce QLoRA `per_device_train_batch_size` to 1; gradient_accumulation_steps to 16 (slower, less memory). |
| Phase 2 30B fine-tune produces worse adapter | Champion gate rejects → old adapter stays. No effect on live trading. |
| Soak counter resets after model swap | The readiness gate is conservative by design; treat each major adapter swap as a potential reset and let the soak rebuild. |

---

## Confirmation checklist before flipping `dry_run=false`

- [ ] Unsloth installed and import-tested on GPU box
- [ ] 100+ quality-filtered scanner outputs in warm_store
- [ ] champion_gate test passes on a synthetic adapter (already in test suite)
- [ ] Disk space: ~5-10 GB free for adapter checkpoints
- [ ] Backup of current scanner adapter (operator manual copy of `models/`)

After Phase 1 validates, repeat for Phase 2 with the analyst.
