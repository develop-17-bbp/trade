"""
LoRA Fine-Tuning Infrastructure for Trading-Specific LLMs
============================================================
Fine-tunes small language models (Mistral 7B, Llama 3 8B, Phi-3, Qwen2)
on our trading system's historical decisions for:

  1. Better market regime classification from quant data
  2. More accurate trade reasoning that cites our specific indicators
  3. Reduced hallucination via domain-specific supervision
  4. Faster inference (small local model vs large cloud API)

Method: LoRA (Low-Rank Adaptation) — trains only 0.1-1% of parameters.
  W' = W + BA  where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)

Training Data: Generated from our own system's trade history + quant states.
Each training example = (quant_data_prompt → correct_json_response)

Requirements:
  pip install peft transformers datasets bitsandbytes accelerate

Usage:
    from src.ai.lora_trainer import LoRATrainer
    trainer = LoRATrainer(base_model='mistralai/Mistral-7B-Instruct-v0.3')
    trainer.prepare_dataset('logs/trade_decisions.jsonl')
    trainer.train(output_dir='models/lora_trading')
    # After training:
    trainer.load_and_infer(prompt)
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA fine-tuning pipeline for trading-specific language models.
    """

    # Tested base models (any HuggingFace causal LM works)
    RECOMMENDED_MODELS = {
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
        'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'phi3-mini': 'microsoft/Phi-3-mini-4k-instruct',
        'qwen2-7b': 'Qwen/Qwen2-7B-Instruct',
        'gemma2-9b': 'google/gemma-2-9b-it',
    }

    def __init__(self, base_model: str = 'mistralai/Mistral-7B-Instruct-v0.3',
                 lora_rank: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05,
                 quantize_4bit: bool = True):
        """
        Args:
            base_model: HuggingFace model ID or path
            lora_rank: LoRA rank (r) — higher = more capacity, slower
            lora_alpha: LoRA scaling factor (α)
            lora_dropout: Dropout for LoRA layers
            quantize_4bit: Use 4-bit quantization (QLoRA) for memory efficiency
        """
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quantize_4bit = quantize_4bit
        self.model = None
        self.tokenizer = None
        self._trained = False

    def prepare_dataset(self, trade_log_path: str,
                        output_path: str = 'data/lora_training_data.jsonl',
                        min_examples: int = 100) -> str:
        """
        Generate training data from trade decision logs.

        Each example is a (system_prompt + quant_data_prompt) → JSON response pair,
        formatted for instruction tuning.

        Args:
            trade_log_path: Path to trade decision JSONL log
            output_path: Where to save formatted training data
            min_examples: Minimum examples needed

        Returns:
            Path to formatted training data
        """
        from src.ai.prompt_constraints import SYSTEM_PROMPT_BASE

        examples = []

        # Load trade decisions
        if os.path.exists(trade_log_path):
            with open(trade_log_path, 'r') as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        examples.append(record)
                    except json.JSONDecodeError:
                        continue

        if len(examples) < min_examples:
            logger.warning(f"Only {len(examples)} examples (need {min_examples}). "
                          f"Generating synthetic examples...")
            examples.extend(self._generate_synthetic_examples(min_examples - len(examples)))

        # Format for instruction tuning
        formatted = []
        for ex in examples:
            # Build the training example
            quant_data = ex.get('quant_data', 'No quant data available.')
            if isinstance(quant_data, dict):
                from src.ai.math_injection import MathInjector
                mi = MathInjector()
                quant_data = mi.format_for_prompt(quant_data)

            response = ex.get('decision', ex.get('response', {}))
            if isinstance(response, dict):
                response_str = json.dumps(response, indent=2)
            else:
                response_str = str(response)

            formatted.append({
                'instruction': SYSTEM_PROMPT_BASE,
                'input': quant_data,
                'output': response_str,
            })

        # Save
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            for item in formatted:
                f.write(json.dumps(item) + '\n')

        logger.info(f"Prepared {len(formatted)} training examples → {output_path}")
        return output_path

    def train(self, dataset_path: str = 'data/lora_training_data.jsonl',
              output_dir: str = 'models/lora_trading',
              epochs: int = 3, batch_size: int = 4,
              learning_rate: float = 2e-4,
              max_seq_length: int = 2048,
              gradient_accumulation: int = 4) -> Dict:
        """
        Fine-tune the base model with LoRA.

        Args:
            dataset_path: Path to prepared training data
            output_dir: Where to save LoRA adapter weights
            epochs: Training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate for LoRA parameters
            max_seq_length: Maximum sequence length
            gradient_accumulation: Gradient accumulation steps

        Returns:
            Training metrics
        """
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM,
                TrainingArguments, Trainer,
                BitsAndBytesConfig,
            )
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from datasets import load_dataset
        except ImportError as e:
            logger.error(f"Missing dependencies for LoRA training: {e}")
            return {
                'status': 'error',
                'message': f"Install required packages: pip install peft transformers datasets bitsandbytes accelerate\nMissing: {e}",
            }

        logger.info(f"Loading base model: {self.base_model}")

        # Quantization config (QLoRA for memory efficiency)
        bnb_config = None
        if self.quantize_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16 if not self.quantize_4bit else None,
        )

        if self.quantize_4bit:
            model = prepare_model_for_kbit_training(model)

        # LoRA config
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                            'gate_proj', 'up_proj', 'down_proj'],
            bias='none',
            task_type='CAUSAL_LM',
        )

        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Load dataset
        dataset = load_dataset('json', data_files=dataset_path, split='train')

        def format_example(example):
            text = (
                f"<s>[INST] <<SYS>>\n{example['instruction']}\n<</SYS>>\n\n"
                f"{example['input']} [/INST] {example['output']} </s>"
            )
            tokenized = self.tokenizer(
                text, truncation=True, max_length=max_seq_length, padding='max_length'
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized

        tokenized_dataset = dataset.map(format_example, remove_columns=dataset.column_names)

        # Training arguments
        os.makedirs(output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            fp16=True,
            save_strategy='epoch',
            logging_steps=10,
            warmup_ratio=0.05,
            lr_scheduler_type='cosine',
            optim='paged_adamw_8bit' if self.quantize_4bit else 'adamw_torch',
            report_to='none',
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        logger.info("Starting LoRA fine-tuning...")
        train_result = trainer.train()

        # Save adapter weights only (small: 10-50MB)
        model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self._trained = True
        self.model = model

        metrics = {
            'status': 'success',
            'trainable_params': trainable,
            'total_params': total,
            'trainable_pct': f"{100*trainable/total:.2f}%",
            'epochs': epochs,
            'training_loss': train_result.training_loss,
            'output_dir': output_dir,
            'adapter_size_mb': sum(
                os.path.getsize(os.path.join(output_dir, f))
                for f in os.listdir(output_dir)
                if f.endswith('.bin') or f.endswith('.safetensors')
            ) / 1024 / 1024,
        }

        logger.info(f"LoRA training complete: {metrics}")
        return metrics

    def load_adapter(self, adapter_path: str = 'models/lora_trading') -> bool:
        """
        Load trained LoRA adapter for inference.

        Args:
            adapter_path: Path to saved LoRA adapter

        Returns:
            True if loaded successfully
        """
        if not os.path.exists(adapter_path):
            logger.warning(f"Adapter not found at {adapter_path}")
            return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            bnb_config = None
            if self.quantize_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            base = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                quantization_config=bnb_config,
                device_map='auto',
            )
            self.model = PeftModel.from_pretrained(base, adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            self.model.eval()
            self._trained = True

            logger.info(f"Loaded LoRA adapter from {adapter_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            return False

    def infer(self, prompt: str, system_prompt: str = '',
              max_new_tokens: int = 1000) -> Dict:
        """
        Run inference with the fine-tuned model.

        Args:
            prompt: User prompt (with quant data already injected)
            system_prompt: System instructions
            max_new_tokens: Maximum tokens to generate

        Returns:
            Parsed JSON response
        """
        if not self._trained or self.model is None:
            logger.warning("Model not loaded. Call load_adapter() first.")
            return {'error': 'model_not_loaded'}

        try:
            import torch

            # Format prompt (Llama/Mistral chat format)
            full_prompt = (
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
                f"{prompt} [/INST]"
            )

            inputs = self.tokenizer(full_prompt, return_tensors='pt').to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )

            # Decode only the generated part
            generated = outputs[0][inputs['input_ids'].shape[1]:]
            text = self.tokenizer.decode(generated, skip_special_tokens=True)

            # Parse JSON
            text = text.strip()
            if text.startswith('```'):
                lines = text.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                text = '\n'.join(lines)

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                start = text.find('{')
                end = text.rfind('}')
                if start >= 0 and end > start:
                    return json.loads(text[start:end + 1])
                return {'error': 'json_parse_failed', 'raw_text': text[:500]}

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {'error': str(e)}

    def _generate_synthetic_examples(self, n: int) -> List[Dict]:
        """
        Generate synthetic training examples from known market scenarios.
        Used when real trade logs are insufficient.
        """
        import numpy as np

        scenarios = [
            # (regime, rsi, macd, trend, action, confidence)
            ('TRENDING', 55, 'BULLISH', 'UP', 'LONG', 75),
            ('TRENDING', 45, 'BEARISH', 'DOWN', 'SHORT', 70),
            ('RANGING', 30, 'BULLISH', 'FLAT', 'LONG', 60),
            ('RANGING', 70, 'BEARISH', 'FLAT', 'SHORT', 60),
            ('VOLATILE', 50, 'NEUTRAL', 'FLAT', 'FLAT', 40),
            ('CHOPPY', 50, 'NEUTRAL', 'FLAT', 'FLAT', 30),
            ('BULL', 60, 'BULLISH', 'UP', 'LONG', 80),
            ('BEAR', 40, 'BEARISH', 'DOWN', 'SHORT', 75),
            ('CRISIS', 25, 'BEARISH', 'DOWN', 'FLAT', 20),
            ('SIDEWAYS', 50, 'NEUTRAL', 'FLAT', 'FLAT', 50),
        ]

        examples = []
        for i in range(n):
            scenario = scenarios[i % len(scenarios)]
            regime, rsi, macd, trend, action, conf = scenario

            # Add some noise
            rsi += np.random.randint(-5, 6)
            conf += np.random.randint(-10, 11)
            conf = max(0, min(100, conf))

            quant_data = {
                'trend': {
                    'rsi_14': float(rsi),
                    'macd_signal': macd,
                    'trend_direction': trend,
                    'adx': float(np.random.uniform(15, 45)),
                },
                'volatility': {
                    'regime': regime,
                    'ewma_vol': float(np.random.uniform(0.01, 0.08)),
                },
                'sentiment': {
                    'finbert_score': float(np.random.uniform(-0.5, 0.5)),
                },
            }

            decision = {
                'market_regime': regime,
                'action': action,
                'confidence_score': conf,
                'reasoning_trace': (
                    f"[RSI_14={rsi}] is {'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'}. "
                    f"[MACD_SIGNAL={macd}] confirms direction. "
                    f"[TREND_DIRECTION={trend}] with regime [REGIME={regime}]."
                ),
                'macro_bias': float(np.random.uniform(-0.2, 0.2)),
                'suggested_config_update': {},
            }

            examples.append({
                'quant_data': quant_data,
                'decision': decision,
            })

        return examples

    def log_decision(self, quant_state: Dict, decision: Dict,
                     log_path: str = 'logs/trade_decisions.jsonl'):
        """
        Log a trade decision for future fine-tuning data collection.
        Call this after every real trade to build up training data.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        record = {
            'timestamp': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ'),
            'quant_data': quant_state,
            'decision': decision,
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
