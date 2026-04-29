"""
Unsloth QLoRA backend for the dual-brain trainer (C10b).

Implements the TrainerBackend protocol from dual_brain_trainer.py with
real 4-bit QLoRA training via Unsloth + inference via Ollama's HTTP API.

Usage:
    from src.ai.dual_brain_trainer import run_cycle
    from src.ai.unsloth_backend import UnslothQLoRABackend
    backend = UnslothQLoRABackend()
    report = run_cycle(backend, asset="BTC", min_samples=100)

Install gate — this module imports cleanly without unsloth, but any call
to `train()` raises a RuntimeError until the operator runs:

    pip install unsloth
    pip install --no-deps xformers trl peft accelerate bitsandbytes

On RTX 5090 (32 GB VRAM) with default Modelfile-derived base models:
    deepseek-r1:32b Q4_K_M  — ~25-28 GB with grad-checkpointing
    deepseek-r1:7b  Q4_K_M  — ~8 GB easy

Training is SEQUENTIAL per the dual_brain_trainer's VRAM discipline —
the orchestrator pauses the agentic loop before calling train(),
restores it after. No inference competes for VRAM during training.

Hot-swap workflow (inside train()):
    1. Load base model via unsloth.FastLanguageModel.from_pretrained
       with load_in_4bit=True.
    2. Attach LoRA adapters (r=16 for 32B, r=32 for 7B typical).
    3. Train on sft_rows (prompt/completion pairs) via SFTTrainer.
    4. Merge LoRA into base → save_pretrained_merged.
    5. Build a GGUF via unsloth's save_pretrained_gguf (Q4_K_M).
    6. Register in Ollama: `ollama create <out_tag> -f Modelfile`.
    7. Now `ollama run <out_tag>` serves the new adapter. Champion
       gate then A/Bs it vs the incumbent; promotion flips
       ACT_ANALYST_MODEL / ACT_SCANNER_MODEL env.

If any step fails the backend returns train=False so the champion
gate rejects cleanly and the incumbent keeps running.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = os.getenv(
    "ACT_UNSLOTH_OUTPUT_ROOT",
    str(Path(__file__).resolve().parents[2] / "models" / "unsloth_adapters"),
)
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_REMOTE_URL") or "http://localhost:11434"
DEFAULT_MAX_SEQ_LEN = int(os.getenv("ACT_UNSLOTH_MAX_SEQ", "2048"))
DEFAULT_EPOCHS = int(os.getenv("ACT_UNSLOTH_EPOCHS", "2"))
DEFAULT_BATCH_SIZE = int(os.getenv("ACT_UNSLOTH_BATCH", "1"))
DEFAULT_GRAD_ACCUM = int(os.getenv("ACT_UNSLOTH_GRAD_ACCUM", "4"))
DEFAULT_LEARNING_RATE = float(os.getenv("ACT_UNSLOTH_LR", "2e-4"))
DEFAULT_LORA_R = int(os.getenv("ACT_UNSLOTH_LORA_R", "16"))


class UnslothUnavailable(RuntimeError):
    """Raised when a train() is attempted without unsloth installed."""


def _require_unsloth():
    try:
        import unsloth  # noqa: F401
    except Exception as e:
        raise UnslothUnavailable(
            "unsloth not installed — `pip install unsloth` on the GPU box "
            f"to enable real training. Original import error: {e}"
        ) from e


def _ollama_has_model(name: str, host: str = DEFAULT_OLLAMA_HOST,
                     timeout_s: float = 5.0) -> bool:
    """Check the Ollama /api/tags endpoint for a model name prefix."""
    try:
        import urllib.request
        url = host.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout_s) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        names = {m.get("name", "") for m in (data.get("models") or [])}
        # Match exact + prefix (ollama shows `deepseek-r1:7b` etc.).
        head = name.split(":")[0]
        return any(n == name or n.startswith(head + ":") for n in names)
    except Exception as e:
        logger.debug("unsloth_backend: ollama /api/tags failed: %s", e)
        return False


def _ollama_create(out_tag: str, modelfile_path: Path,
                   timeout_s: float = 600.0) -> bool:
    """Run `ollama create <tag> -f <Modelfile>`. Returns True on success."""
    try:
        r = subprocess.run(
            ["ollama", "create", out_tag, "-f", str(modelfile_path)],
            check=False, capture_output=True, text=True, timeout=timeout_s,
        )
        if r.returncode != 0:
            logger.warning("ollama create %s failed: %s", out_tag, r.stderr[:500])
            return False
        return True
    except Exception as e:
        logger.debug("ollama create %s exception: %s", out_tag, e)
        return False


class UnslothQLoRABackend:
    """Concrete TrainerBackend backed by unsloth + Ollama.

    Attributes:
        output_root: filesystem root for per-training-run artifacts.
        ollama_host: HTTP endpoint used by infer() for validation calls.

    Configuration (all env-overridable, see defaults at module top):
        ACT_UNSLOTH_MAX_SEQ          — sequence length (default 2048)
        ACT_UNSLOTH_EPOCHS           — training epochs (default 2)
        ACT_UNSLOTH_BATCH            — per-device batch (default 1)
        ACT_UNSLOTH_GRAD_ACCUM       — gradient accumulation (default 4)
        ACT_UNSLOTH_LR               — learning rate (default 2e-4)
        ACT_UNSLOTH_LORA_R           — LoRA rank (default 16)
    """

    def __init__(
        self,
        *,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        export: str = "gguf",
        lora_r: Optional[int] = None,
    ):
        """`export` = 'gguf' (default) does the full merge + GGUF + Ollama
        register pipeline (5090 path). 'lora_only' stops after step 4
        (LoRA training) and writes adapter_model.safetensors only — used
        by the 4060 box which uploads the LoRA delta to the 5090 over
        SSH; the 5090 watcher merges + GGUF-quantizes locally.

        `lora_r` overrides DEFAULT_LORA_R (env-driven default = 16).
        Pass 8 from the 4060 to fit the smaller VRAM budget.
        """
        self.output_root = Path(output_root)
        self.ollama_host = ollama_host
        self.export = export
        self.lora_r = int(lora_r) if lora_r is not None else DEFAULT_LORA_R
        self.output_root.mkdir(parents=True, exist_ok=True)

    # ── TrainerBackend.train ────────────────────────────────────────────

    def train(
        self,
        base_model: str,
        sft_rows: List[Dict[str, str]],
        out_tag: str,
    ) -> bool:
        """Fine-tune `base_model` on `sft_rows` and register as `out_tag`
        in Ollama. Returns True on success.

        Never raises on "expected" failures (unsloth missing, disk full,
        ollama unreachable). Unexpected exceptions are caught + logged;
        the champion gate will then reject the `out_tag` for
        'train_failed' which is the correct behavior.
        """
        t0 = time.time()
        try:
            _require_unsloth()
        except UnslothUnavailable as e:
            logger.warning("train() aborted: %s", e)
            return False

        try:
            from unsloth import FastLanguageModel
            from datasets import Dataset
            from trl import SFTTrainer
            from transformers import TrainingArguments
        except Exception as e:
            logger.warning("train() import chain failed: %s", e)
            return False

        # ── 1. Base model load (lazy; happens only when operator triggers) ──
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=_resolve_hf_id(base_model),
                max_seq_length=DEFAULT_MAX_SEQ_LEN,
                load_in_4bit=True,
                dtype=None,
            )
        except Exception as e:
            logger.warning("base-model load failed for %s: %s", base_model, e)
            return False

        # ── 2. Attach LoRA ─────────────────────────────────────────────
        try:
            model = FastLanguageModel.get_peft_model(
                model, r=self.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.lora_r,
                lora_dropout=0.0, bias="none",
                use_gradient_checkpointing=True,
                random_state=3407,
            )
        except Exception as e:
            logger.warning("LoRA attach failed: %s", e)
            return False

        # ── 3. Dataset: {prompt, completion} → chat-formatted text ─────
        try:
            rows = _format_rows_for_sft(sft_rows, tokenizer)
            if len(rows) < 5:
                logger.warning("train() rejected — only %d rows after format", len(rows))
                return False
            ds = Dataset.from_list(rows)
        except Exception as e:
            logger.warning("dataset build failed: %s", e)
            return False

        # ── 4. Train ───────────────────────────────────────────────────
        run_dir = self.output_root / out_tag.replace(":", "_").replace("/", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            trainer = SFTTrainer(
                model=model, tokenizer=tokenizer, train_dataset=ds,
                dataset_text_field="text", max_seq_length=DEFAULT_MAX_SEQ_LEN,
                dataset_num_proc=2, packing=False,
                args=TrainingArguments(
                    per_device_train_batch_size=DEFAULT_BATCH_SIZE,
                    gradient_accumulation_steps=DEFAULT_GRAD_ACCUM,
                    warmup_steps=5,
                    num_train_epochs=DEFAULT_EPOCHS,
                    learning_rate=DEFAULT_LEARNING_RATE,
                    fp16=False, bf16=True,
                    logging_steps=10,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="cosine",
                    seed=3407,
                    output_dir=str(run_dir / "checkpoints"),
                    report_to="none",
                ),
            )
            trainer.train()
        except Exception as e:
            logger.warning("training step failed: %s", e)
            return False

        # ── 4b. Save metadata sidecar for cross-box deploy ─────────────
        try:
            (run_dir / "metadata.json").write_text(json.dumps({
                "out_tag": out_tag,
                "base_model": base_model,
                "hf_repo": _resolve_hf_id(base_model),
                "lora_r": self.lora_r,
                "epochs": DEFAULT_EPOCHS,
                "batch": DEFAULT_BATCH_SIZE,
                "grad_accum": DEFAULT_GRAD_ACCUM,
                "lr": DEFAULT_LEARNING_RATE,
                "max_seq_len": DEFAULT_MAX_SEQ_LEN,
                "n_train_rows": len(rows),
                "trained_at": int(time.time()),
                "export_mode": self.export,
            }, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug("metadata write failed: %s", e)

        # ── lora_only path: persist adapter only, skip GGUF + Ollama ───
        if self.export == "lora_only":
            try:
                model.save_pretrained(str(run_dir))   # writes adapter_model.safetensors
                tokenizer.save_pretrained(str(run_dir))
                logger.info(
                    "train(lora_only) succeeded: %s in %.1fs (%d rows) → %s",
                    out_tag, time.time() - t0, len(sft_rows), run_dir,
                )
                return True
            except Exception as e:
                logger.warning("LoRA save_pretrained failed: %s", e)
                return False

        # ── 5. Merge LoRA, save as GGUF, register in Ollama ────────────
        gguf_path = run_dir / "model.gguf"
        modelfile_path = run_dir / "Modelfile"
        try:
            # Unsloth's one-shot merge + GGUF export (Q4_K_M).
            model.save_pretrained_gguf(
                str(run_dir), tokenizer, quantization_method="q4_k_m",
            )
            # Unsloth names the output <dir>/unsloth.Q4_K_M.gguf or
            # <dir>/<model>-Q4_K_M.gguf depending on version. Find + symlink
            # to model.gguf for a stable Modelfile reference.
            found = sorted(run_dir.glob("*Q4_K_M*.gguf"))
            if not found:
                found = sorted(run_dir.glob("*.gguf"))
            if not found:
                logger.warning("no GGUF produced in %s", run_dir)
                return False
            try:
                gguf_path.unlink(missing_ok=True)
            except Exception:
                pass
            gguf_path.write_bytes(found[0].read_bytes())
        except Exception as e:
            logger.warning("GGUF export failed: %s", e)
            return False

        try:
            # Honor OLLAMA_NUM_CTX (set by START_ALL to 16384) so a
            # fine-tune output doesn't bake in a 32K ctx that would
            # evict the resident base models on its first request.
            try:
                _num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
            except (ValueError, TypeError):
                _num_ctx = 16384
            modelfile_path.write_text(
                f"FROM {gguf_path.as_posix()}\n"
                f"PARAMETER num_ctx {_num_ctx}\n"
                f"PARAMETER temperature 0.3\n",
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("Modelfile write failed: %s", e)
            return False

        if not _ollama_create(out_tag, modelfile_path):
            return False

        if not _ollama_has_model(out_tag, self.ollama_host):
            logger.warning("ollama reports %s not registered after create", out_tag)
            return False

        logger.info("train() succeeded: %s in %.1fs (%d rows)",
                    out_tag, time.time() - t0, len(sft_rows))
        return True

    # ── TrainerBackend.infer ────────────────────────────────────────────

    def infer(self, model_id: str, sample: Dict[str, Any]) -> str:
        """Call Ollama /api/generate to produce the model's raw output
        for one validation sample. Never raises — empty string on error."""
        prompt = _sample_to_prompt(sample)
        try:
            import urllib.request
            payload = json.dumps({
                "model": model_id, "prompt": prompt, "stream": False,
                "options": {"temperature": 0.2, "num_predict": 512},
            }).encode("utf-8")
            req = urllib.request.Request(
                self.ollama_host.rstrip("/") + "/api/generate",
                data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60.0) as r:
                body = json.loads(r.read().decode("utf-8", errors="replace"))
            return str(body.get("response") or "")
        except Exception as e:
            logger.debug("infer(%s) failed: %s", model_id, e)
            return ""


# ── Helpers ─────────────────────────────────────────────────────────────


# Map Ollama-style tags → HuggingFace repo IDs for unsloth.from_pretrained.
# Operator can override via ACT_UNSLOTH_HF_MAP env (json dict) if they
# pin specific weights.
_DEFAULT_HF_MAP = {
    "deepseek-r1:32b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek-r1:14b":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:7b":    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "qwen3:32b":         "Qwen/Qwen3-32B",
    "qwen3-coder:30b":   "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "qwen2.5-coder:7b":  "Qwen/Qwen2.5-Coder-7B-Instruct",
    "devstral:24b":      "mistralai/Devstral-Small-2507",
}


def _resolve_hf_id(base_model: str) -> str:
    """Turn an Ollama tag (`deepseek-r1:32b`) into a HuggingFace repo ID
    that unsloth.FastLanguageModel.from_pretrained understands."""
    env_json = os.environ.get("ACT_UNSLOTH_HF_MAP")
    if env_json:
        try:
            override = json.loads(env_json) or {}
            if base_model in override:
                return str(override[base_model])
        except Exception:
            pass
    # Strip Ollama-specific suffixes like :act-1712345 (our own hot-swapped
    # tags); fall back to the base family's HF repo.
    bare = base_model.split(":act-", 1)[0]
    if bare in _DEFAULT_HF_MAP:
        return _DEFAULT_HF_MAP[bare]
    # Bare family head (deepseek-r1:7b-custom → deepseek-r1:7b base).
    head = bare.split(":")[0]
    for k, v in _DEFAULT_HF_MAP.items():
        if k.split(":")[0] == head:
            return v
    # No map entry → pass the tag through; unsloth will complain loudly if
    # it doesn't resolve to a real repo.
    return base_model


def _format_rows_for_sft(
    rows: List[Dict[str, str]], tokenizer
) -> List[Dict[str, str]]:
    """Convert [{prompt, completion}, ...] → [{text: full chat-formatted}, ...].

    Uses the tokenizer's chat template when available (newer Qwen/R1
    tokenizers ship one). Falls back to a plain "USER:/ASSISTANT:"
    format otherwise.
    """
    out: List[Dict[str, str]] = []
    has_template = bool(getattr(tokenizer, "chat_template", None))
    for r in rows:
        prompt = str(r.get("prompt") or "")
        completion = str(r.get("completion") or "")
        if not prompt or not completion:
            continue
        if has_template:
            try:
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": completion},
                    ],
                    tokenize=False, add_generation_prompt=False,
                )
            except Exception:
                text = f"USER: {prompt}\nASSISTANT: {completion}"
        else:
            text = f"USER: {prompt}\nASSISTANT: {completion}"
        out.append({"text": text})
    return out


def _sample_to_prompt(sample: Dict[str, Any]) -> str:
    """Shape a validation sample into a prompt the model can answer.

    Validation samples are ExperienceSample.to_dict() outputs — they
    carry the plan that was compiled. For analyst inference we ask the
    model to re-compile given the scanner tag + asset; for scanner we
    ask for a scan report given the asset. Either path works as long as
    the model produces JSON the champion_gate.score_* parsers can read.
    """
    if not isinstance(sample, dict):
        return str(sample)[:1000]
    asset = sample.get("asset", "BTC")
    scanner_tag = sample.get("scanner_tag") or {}
    return (
        f"Asset: {asset}\n"
        f"Scanner tag: {json.dumps(scanner_tag, default=str)[:400]}\n"
        "Task: compile a TradePlan OR emit a scan report. Respond with "
        "ONE JSON object containing `direction` and optionally `size_pct`."
    )
