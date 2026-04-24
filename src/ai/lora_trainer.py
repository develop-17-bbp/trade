"""
LoRA Fine-Tuning Pipeline for ACT Trading System
===================================================
Legacy Brain v2 LoRA pipeline (7B/8B class models). Superseded on the
RTX 5090 box by `src/ai/dual_brain_trainer.py` which drives Unsloth
QLoRA over the `devstral:24b` + `qwen3:32b` pair (C10b).

Kept here for two reasons:
  1. Operators still running the 8GB-class workflow (laptop builds,
     CI pipelines) can fine-tune the 7B scanner + 8B analyst locally.
  2. The JSONL data-prep + GGUF export + Ollama `create` plumbing is
     reused by `dual_brain_trainer.py` via import.

If you're on the 5090, prefer C10b's dual_brain_trainer — it pulls
from `training_data_filter.py` (quality-filtered samples) and goes
through the `champion_gate` before swapping adapters in Ollama.

Requirements (on the GPU server):
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes

Usage:
  # Train scanner model
  python -m src.ai.lora_trainer --model scanner --epochs 3

  # Train analyst model
  python -m src.ai.lora_trainer --model analyst --epochs 3

  # Deploy to Ollama
  python -m src.ai.lora_trainer --deploy scanner
  python -m src.ai.lora_trainer --deploy analyst

  # Full pipeline (train + deploy both)
  python -m src.ai.lora_trainer --full
"""

import os
import json
import time
import logging
import subprocess
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Model Configs — tuned for 8GB GPU
# ═══════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    'scanner': {
        'base_model': 'unsloth/mistral-7b-instruct-v0.3-bnb-4bit',
        'fallback_model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'ollama_name': 'act-scanner',
        'lora_rank': 16,
        'lora_alpha': 16,
        'max_seq_length': 2048,
        'system_prompt': (
            "You are a crypto market pattern scanner. Analyze candlestick data, "
            "price action, volume, and momentum patterns. Output ONLY JSON with: "
            "pattern_bias (BULLISH/BEARISH/NEUTRAL), pattern_strength (1-10), "
            "strongest_signal, danger_pattern, resembles_winner (true/false)."
        ),
        'chat_template': 'mistral',
    },
    'analyst': {
        'base_model': 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit',
        'fallback_model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'ollama_name': 'act-analyst',
        'lora_rank': 16,
        'lora_alpha': 16,
        'max_seq_length': 4096,
        'system_prompt': (
            "You are a crypto trade analyst for Robinhood with 1.69% round-trip spread. "
            "Given pattern scan results and market context, decide whether to trade. "
            "Output ONLY JSON with: proceed (bool), confidence (0.0-1.0), risk_score (0-10), "
            "trade_quality (0-10), predicted_l_level, bull_case, bear_case, facilitator_verdict."
        ),
        'chat_template': 'llama-3.1',
    },
}


class LoRATrainer:
    """QLoRA fine-tuning pipeline optimized for 8GB GPU."""

    def __init__(self, model_type: str = 'scanner',
                 data_dir: str = 'data/finetune',
                 output_dir: str = 'models/lora'):
        assert model_type in MODEL_CONFIGS, f"Unknown model: {model_type}"
        self.model_type = model_type
        self.config = MODEL_CONFIGS[model_type]
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) / model_type
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_training_data(self) -> str:
        """
        Convert collected training data into ChatML/instruction format.
        Returns path to formatted JSONL ready for training.
        """
        if self.model_type == 'scanner':
            raw_path = self.data_dir / 'scanner_training.jsonl'
        else:
            raw_path = self.data_dir / 'analyst_training.jsonl'

        if not raw_path.exists():
            logger.warning(f"No training data at {raw_path}")
            return ''

        formatted_path = self.output_dir / 'training_formatted.jsonl'
        system_prompt = self.config['system_prompt']
        count = 0

        with open(raw_path) as fin, open(formatted_path, 'w') as fout:
            for line in fin:
                try:
                    rec = json.loads(line.strip())
                    inp = rec.get('input', '')
                    out = rec.get('output', {})

                    if isinstance(out, dict):
                        out_str = json.dumps(out)
                    else:
                        out_str = str(out)

                    # Format as conversation for chat fine-tuning
                    example = {
                        'conversations': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': inp},
                            {'role': 'assistant', 'content': out_str},
                        ]
                    }
                    fout.write(json.dumps(example) + '\n')
                    count += 1
                except Exception:
                    continue

        logger.info(f"[TRAINER] Prepared {count} {self.model_type} examples → {formatted_path}")
        return str(formatted_path)

    def train(self, epochs: int = 3, batch_size: int = 2,
              learning_rate: float = 2e-4,
              gradient_accumulation: int = 4) -> Dict:
        """
        Fine-tune with QLoRA using unsloth for 8GB GPU efficiency.
        Falls back to standard HuggingFace if unsloth unavailable.
        """
        dataset_path = self.prepare_training_data()
        if not dataset_path:
            return {'status': 'error', 'message': 'No training data'}

        # Acquire a P3 GPU lease so LoRA training gets bumped by RL (P2) but
        # beats meta-coordinator (P3.5) and default (P4). wait_s=60 — we'll
        # wait up to a minute for a higher-priority learner to release before
        # giving up on this retrain cycle.
        _leased = False
        _lease_cm = None
        try:
            from src.orchestration import gpu_scheduler as _gs
            _lease_cm = _gs.lease(priority=_gs.P3, wait_s=60.0)
            _lease_cm.__enter__()
            _leased = True
        except Exception as _le:
            logger.info(f"[TRAINER] GPU lease unavailable ({_le}); training without lease")

        try:
            try:
                result = self._train_unsloth(dataset_path, epochs, batch_size,
                                             learning_rate, gradient_accumulation)
            except ImportError:
                logger.info("[TRAINER] unsloth not available, using standard HuggingFace")
                result = self._train_hf(dataset_path, epochs, batch_size,
                                        learning_rate, gradient_accumulation)
        finally:
            if _leased and _lease_cm is not None:
                try:
                    _lease_cm.__exit__(None, None, None)
                except Exception:
                    pass

        # Phase 4.5b: publish LoRA training metrics so the meta-coordinator can
        # factor them into credit / transfer decisions. Soft-fail on any error.
        try:
            from src.learning.signal_bus import publish_lora_logprob
            final_loss = float(result.get('final_loss', result.get('train_loss', 0.0)) or 0.0)
            publish_lora_logprob(
                prompt_hash=f"epoch{epochs}_{self.model_type}",
                direction="TRAIN",
                logprob=-final_loss,  # log-prob approximation: -loss
                loss=final_loss,
            )
        except Exception:
            pass
        return result

    def _train_unsloth(self, dataset_path: str, epochs: int,
                       batch_size: int, lr: float, grad_accum: int) -> Dict:
        """Train using unsloth — 2x faster, 60% less VRAM."""
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from datasets import load_dataset

        max_seq = self.config['max_seq_length']
        rank = self.config['lora_rank']
        alpha = self.config['lora_alpha']

        logger.info(f"[TRAINER] Loading {self.config['base_model']} with unsloth...")

        # Load 4-bit quantized model (fits in ~5.5GB)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config['base_model'],
            max_seq_length=max_seq,
            dtype=None,  # Auto-detect
            load_in_4bit=True,
        )

        # Apply LoRA to ALL linear layers for best quality
        model = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            bias='none',
            use_gradient_checkpointing='unsloth',  # 30% less VRAM
            random_state=42,
        )

        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')

        # Format conversations → text using chat template
        def format_chat(example):
            convos = example['conversations']
            text = tokenizer.apply_chat_template(convos, tokenize=False)
            return {'text': text}

        dataset = dataset.map(format_chat)

        # Training
        output_str = str(self.output_dir / 'checkpoints')
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field='text',
            max_seq_length=max_seq,
            dataset_num_proc=2,
            packing=True,  # Pack short examples for efficiency
            args=TrainingArguments(
                output_dir=output_str,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                fp16=True,
                logging_steps=5,
                warmup_steps=5,
                lr_scheduler_type='linear',
                optim='adamw_8bit',
                save_strategy='epoch',
                report_to='none',
                seed=42,
            ),
        )

        logger.info("[TRAINER] Starting fine-tuning...")
        start = time.time()
        result = trainer.train()
        elapsed = time.time() - start

        # Save LoRA adapter
        adapter_path = str(self.output_dir / 'adapter')
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        metrics = {
            'status': 'success',
            'model_type': self.model_type,
            'base_model': self.config['base_model'],
            'epochs': epochs,
            'training_loss': result.training_loss,
            'training_time_s': elapsed,
            'adapter_path': adapter_path,
            'backend': 'unsloth',
        }

        # Save metrics
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"[TRAINER] Done in {elapsed:.0f}s | loss={result.training_loss:.4f}")
        return metrics

    def _train_hf(self, dataset_path: str, epochs: int,
                  batch_size: int, lr: float, grad_accum: int) -> Dict:
        """Fallback: standard HuggingFace + PEFT training."""
        import torch
        from transformers import (
            AutoTokenizer, AutoModelForCausalLM,
            TrainingArguments, BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
        from datasets import load_dataset

        max_seq = self.config['max_seq_length']
        base = self.config.get('fallback_model', self.config['base_model'])

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb_config, device_map='auto',
        )
        model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=self.config['lora_rank'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=0.05,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, lora_config)

        dataset = load_dataset('json', data_files=dataset_path, split='train')

        def format_chat(example):
            convos = example['conversations']
            text = tokenizer.apply_chat_template(convos, tokenize=False)
            return {'text': text}

        dataset = dataset.map(format_chat)

        output_str = str(self.output_dir / 'checkpoints')
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field='text',
            max_seq_length=max_seq,
            packing=True,
            args=TrainingArguments(
                output_dir=output_str,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                fp16=True,
                logging_steps=5,
                warmup_ratio=0.05,
                lr_scheduler_type='cosine',
                optim='paged_adamw_8bit',
                save_strategy='epoch',
                report_to='none',
            ),
        )

        start = time.time()
        result = trainer.train()
        elapsed = time.time() - start

        adapter_path = str(self.output_dir / 'adapter')
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return {
            'status': 'success',
            'model_type': self.model_type,
            'training_loss': result.training_loss,
            'training_time_s': elapsed,
            'adapter_path': adapter_path,
            'backend': 'huggingface',
        }

    # ═══════════════════════════════════════════════════════
    # GGUF Export + Ollama Deployment
    # ═══════════════════════════════════════════════════════

    def merge_and_export_gguf(self, quantization: str = 'q4_k_m') -> str:
        """
        Merge LoRA adapter into base model and export as GGUF for Ollama.

        Args:
            quantization: GGUF quantization level
                q4_k_m = good quality, fits 8GB (recommended)
                q5_k_m = better quality, ~5.5GB
                q8_0   = best quality, ~8GB (tight fit)

        Returns:
            Path to GGUF file
        """
        adapter_path = self.output_dir / 'adapter'
        if not adapter_path.exists():
            raise FileNotFoundError(f"No adapter at {adapter_path}. Train first.")

        merged_path = self.output_dir / 'merged'
        gguf_path = self.output_dir / f'{self.config["ollama_name"]}.gguf'

        try:
            # Try unsloth merge (fastest)
            return self._merge_unsloth(adapter_path, gguf_path, quantization)
        except ImportError:
            # Fallback: merge with PEFT + llama.cpp convert
            return self._merge_manual(adapter_path, merged_path, gguf_path, quantization)

    def _merge_unsloth(self, adapter_path: Path, gguf_path: Path,
                       quant: str) -> str:
        """Merge and export GGUF using unsloth (handles everything)."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            max_seq_length=self.config['max_seq_length'],
            load_in_4bit=True,
        )

        # Save as GGUF directly (unsloth handles merge + quantize)
        model.save_pretrained_gguf(
            str(gguf_path.parent),
            tokenizer,
            quantization_method=quant,
        )

        # Find the generated GGUF file
        for f in gguf_path.parent.iterdir():
            if f.suffix == '.gguf':
                logger.info(f"[TRAINER] GGUF exported: {f} ({f.stat().st_size / 1e9:.1f}GB)")
                return str(f)

        raise FileNotFoundError("GGUF file not found after export")

    def _merge_manual(self, adapter_path: Path, merged_path: Path,
                      gguf_path: Path, quant: str) -> str:
        """Fallback: merge with PEFT, then use llama.cpp for GGUF conversion."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        base_name = self.config.get('fallback_model', self.config['base_model'])

        logger.info(f"[TRAINER] Loading base model {base_name} for merge...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.float16, device_map='cpu',
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.merge_and_unload()

        merged_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(merged_path))
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(merged_path))

        logger.info(f"[TRAINER] Merged model saved to {merged_path}")

        # Convert to GGUF using llama.cpp's convert script
        # User needs llama.cpp installed (or we can use huggingface_hub)
        try:
            result = subprocess.run(
                ['python', '-m', 'llama_cpp.convert',
                 '--outfile', str(gguf_path),
                 '--outtype', quant,
                 str(merged_path)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info(f"[TRAINER] GGUF: {gguf_path}")
                return str(gguf_path)
        except Exception as e:
            logger.warning(f"[TRAINER] llama.cpp convert failed: {e}")

        # Alternative: use huggingface_hub gguf export
        logger.info("[TRAINER] Merged model ready at {merged_path}. "
                    "Convert to GGUF manually with llama.cpp or upload to HF.")
        return str(merged_path)

    def deploy_to_ollama(self, gguf_path: str = None,
                         ollama_url: str = None) -> bool:
        """
        Create Ollama model from GGUF and register it.

        If ollama_url is provided, deploys to remote server via API.
        Otherwise deploys to local Ollama installation.
        """
        if gguf_path is None:
            # Find GGUF in output dir
            for f in self.output_dir.iterdir():
                if f.suffix == '.gguf':
                    gguf_path = str(f)
                    break

        if not gguf_path or not os.path.exists(gguf_path):
            # Try deploying merged model directly (Ollama can load safetensors)
            merged_path = self.output_dir / 'merged'
            if merged_path.exists():
                gguf_path = str(merged_path)
            else:
                logger.error("[TRAINER] No GGUF or merged model found")
                return False

        model_name = self.config['ollama_name']
        system_prompt = self.config['system_prompt']

        # Create Modelfile
        modelfile_path = self.output_dir / 'Modelfile'
        modelfile_content = f"""FROM {gguf_path}

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER num_predict 1000
PARAMETER stop "</s>"
PARAMETER stop "[/INST]"

SYSTEM \"\"\"{system_prompt}\"\"\"

TEMPLATE \"\"\"[INST] <<SYS>>
{{{{ .System }}}}
<</SYS>>

{{{{ .Prompt }}}} [/INST]\"\"\"
"""

        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)

        logger.info(f"[TRAINER] Created Modelfile at {modelfile_path}")

        if ollama_url:
            # Remote deployment via Ollama API
            return self._deploy_remote(model_name, gguf_path, ollama_url)
        else:
            # Local deployment via CLI
            return self._deploy_local(model_name, modelfile_path)

    def _deploy_local(self, model_name: str, modelfile_path: Path) -> bool:
        """Deploy to local Ollama via CLI."""
        try:
            result = subprocess.run(
                ['ollama', 'create', model_name, '-f', str(modelfile_path)],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info(f"[TRAINER] Deployed {model_name} to local Ollama")
                return True
            else:
                logger.error(f"[TRAINER] ollama create failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("[TRAINER] ollama CLI not found. Install Ollama first.")
            return False

    def _deploy_remote(self, model_name: str, gguf_path: str,
                       ollama_url: str) -> bool:
        """Deploy to remote Ollama server (the 8GB GPU box)."""
        import requests

        # Check if remote Ollama is reachable
        try:
            resp = requests.get(f"{ollama_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                logger.error(f"[TRAINER] Remote Ollama not reachable: {resp.status_code}")
                return False
        except Exception as e:
            logger.error(f"[TRAINER] Remote Ollama connection failed: {e}")
            return False

        # For remote deployment, we need to:
        # 1. Upload the GGUF to the remote server (scp/rsync)
        # 2. Create Modelfile on remote
        # 3. Run ollama create on remote
        # This requires SSH access to the GPU box
        logger.info(f"[TRAINER] Remote deployment to {ollama_url}")
        logger.info(f"[TRAINER] Upload {gguf_path} to GPU server and run:")
        logger.info(f"[TRAINER]   ollama create {model_name} -f Modelfile")

        return True

    def full_pipeline(self, epochs: int = 3, quantization: str = 'q4_k_m',
                      ollama_url: str = None) -> Dict:
        """
        Complete pipeline: train → merge → GGUF → Ollama deploy.

        Returns comprehensive metrics.
        """
        results = {'model_type': self.model_type, 'steps': {}}

        # Step 1: Train
        logger.info(f"[PIPELINE] Step 1/3: Training {self.model_type}...")
        train_result = self.train(epochs=epochs)
        results['steps']['train'] = train_result

        if train_result.get('status') != 'success':
            results['status'] = 'failed_at_training'
            return results

        # Step 2: Export GGUF
        logger.info(f"[PIPELINE] Step 2/3: Exporting GGUF ({quantization})...")
        try:
            gguf_path = self.merge_and_export_gguf(quantization=quantization)
            results['steps']['gguf'] = {'path': gguf_path, 'status': 'success'}
        except Exception as e:
            logger.error(f"[PIPELINE] GGUF export failed: {e}")
            results['steps']['gguf'] = {'status': 'failed', 'error': str(e)}
            results['status'] = 'failed_at_gguf'
            return results

        # Step 3: Deploy to Ollama
        logger.info(f"[PIPELINE] Step 3/3: Deploying to Ollama...")
        deployed = self.deploy_to_ollama(gguf_path, ollama_url)
        results['steps']['deploy'] = {'status': 'success' if deployed else 'failed'}

        results['status'] = 'success' if deployed else 'partial'
        return results


# ═══════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description='ACT LLM Fine-Tuning Pipeline')
    parser.add_argument('--model', choices=['scanner', 'analyst', 'both'],
                        default='both', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--quant', default='q4_k_m',
                        choices=['q4_k_m', 'q5_k_m', 'q8_0'])
    parser.add_argument('--deploy', action='store_true',
                        help='Deploy to Ollama after training')
    parser.add_argument('--ollama-url', type=str, default=None,
                        help='Remote Ollama URL for deployment')
    parser.add_argument('--full', action='store_true',
                        help='Full pipeline: train + GGUF + deploy')
    parser.add_argument('--stats', action='store_true',
                        help='Show training data statistics')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    if args.stats:
        from src.ai.training_data_collector import TrainingDataCollector
        collector = TrainingDataCollector()
        stats = collector.get_stats()
        print(f"\n{'='*50}")
        print(f"  ACT Training Data Statistics")
        print(f"{'='*50}")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print(f"{'='*50}\n")
        return

    models = ['scanner', 'analyst'] if args.model == 'both' else [args.model]

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"  Training: {model_type.upper()} model")
        print(f"{'='*60}")

        trainer = LoRATrainer(model_type=model_type)

        if args.full:
            result = trainer.full_pipeline(
                epochs=args.epochs,
                quantization=args.quant,
                ollama_url=args.ollama_url,
            )
        else:
            result = trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )

            if args.deploy and result.get('status') == 'success':
                gguf_path = trainer.merge_and_export_gguf(args.quant)
                trainer.deploy_to_ollama(gguf_path, args.ollama_url)

        print(f"\n  Result: {json.dumps(result, indent=2, default=str)}")


if __name__ == '__main__':
    main()
