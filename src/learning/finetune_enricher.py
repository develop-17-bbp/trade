"""
LLM Fine-Tune Enricher — ACT v8.0
Enriches every LLM training example with:
  - Economic Intelligence macro context (12 layers)
  - Memory-driven few-shot examples (similar past trades)
  - Accuracy engine data (which models/agents to trust)
  - Sharpe optimizer state (current mode, quality score)

The LLM learns to incorporate ALL data sources into its decisions,
not just technical indicators. This is what makes ACT truly self-evolving.

Usage:
    enricher = FinetuneEnricher(econ_intel, accuracy_engine, sharpe_optimizer, llm_memory)
    # Called in training_data_collector.record_decision():
    enriched_context = enricher.enrich(base_context)
    # enriched_context gets baked into the training example prompt
    # so the LLM learns to use macro + memory + accuracy data
"""
import logging
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FinetuneEnricher:
    """Enriches LLM training data with all v8.0 intelligence layers."""

    def __init__(self, econ_intel=None, accuracy_engine=None,
                 sharpe_optimizer=None, llm_memory=None, quant_memories=None):
        self._econ = econ_intel
        self._accuracy = accuracy_engine
        self._sharpe = sharpe_optimizer
        self._llm_memory = llm_memory
        self._quant_memories = quant_memories or {}

    def enrich(self, base_context: dict) -> dict:
        """Enrich a training context with all available intelligence.

        Args:
            base_context: dict with asset, price, regime, indicators, etc.

        Returns:
            Enriched context dict that becomes part of the LLM training prompt.
            The LLM learns to read and act on ALL these signals.
        """
        enriched = dict(base_context)

        # 1. Economic Intelligence — macro signals
        if self._econ:
            try:
                enriched['macro'] = self._econ.get_finetune_context()
                enriched['macro_text'] = self._econ.get_llm_context_block()
            except Exception as e:
                logger.debug(f"[ENRICHER] macro failed: {e}")

        # 2. Accuracy Engine — model and agent reliability
        if self._accuracy:
            try:
                enriched['accuracy'] = self._accuracy.get_finetune_accuracy_context()
                regime = base_context.get('regime', 'unknown')
                enriched['ensemble_weights'] = self._accuracy.get_ensemble_weights(regime)
                enriched['effective_spread'] = self._accuracy.get_effective_spread()
            except Exception as e:
                logger.debug(f"[ENRICHER] accuracy failed: {e}")

        # 3. Sharpe Optimizer — current mode and quality gate
        if self._sharpe:
            try:
                enriched['sharpe_stats'] = self._sharpe.get_stats()
                enriched['sharpe_mode'] = self._sharpe.mode
                enriched['sharpe_filter'] = self._sharpe.get_filter_adjustments()
            except Exception as e:
                logger.debug(f"[ENRICHER] sharpe failed: {e}")

        # 4. LLM Memory — similar past trades as few-shot context
        if self._llm_memory:
            try:
                signal = {
                    'market_regime': base_context.get('regime', 'unknown'),
                    'action_taken': base_context.get('direction', 'LONG'),
                }
                enriched['memory_context'] = self._llm_memory.build_dynamic_prompt_context(signal)
                enriched['confidence_threshold'] = self._llm_memory.calibrate_confidence_threshold()
            except Exception as e:
                logger.debug(f"[ENRICHER] llm_memory failed: {e}")

        # 5. Quant Model Trust — which models to weight in current regime
        if self._quant_memories:
            try:
                regime = base_context.get('regime', 'unknown')
                model_trust = {}
                for model_name, mem in self._quant_memories.items():
                    model_trust[model_name] = {
                        'trusted': mem.should_trust_model(regime, base_context.get('volatility', 0.2)),
                        'regime_perf': mem.get_regime_performance().get(regime, {}),
                    }
                enriched['model_trust'] = model_trust
            except Exception as e:
                logger.debug(f"[ENRICHER] quant_memory failed: {e}")

        return enriched

    def build_enriched_prompt_block(self, base_context: dict) -> str:
        """Build a text block that gets injected into LLM prompts during
        both training AND inference. The LLM learns to read this format.

        This is the KEY innovation: the LLM sees macro data, memory patterns,
        model reliability, and Sharpe state AS PART OF ITS INPUT PROMPT.
        During fine-tuning, it learns to act on these signals correctly.
        """
        enriched = self.enrich(base_context)
        lines = []

        # Macro block
        if 'macro_text' in enriched:
            lines.append(enriched['macro_text'])

        # Memory patterns (few-shot)
        if 'memory_context' in enriched and enriched['memory_context']:
            lines.append("\n=== SIMILAR PAST TRADES (from memory) ===")
            lines.append(enriched['memory_context'])

        # Model trust
        if 'model_trust' in enriched:
            lines.append("\n=== MODEL RELIABILITY (current regime) ===")
            for model, data in enriched['model_trust'].items():
                perf = data.get('regime_perf', {})
                acc = perf.get('accuracy', 'N/A')
                trusted = "TRUSTED" if data['trusted'] else "UNRELIABLE"
                lines.append(f"  {model}: {trusted} (accuracy={acc})")

        # Ensemble weights
        if 'ensemble_weights' in enriched:
            lines.append(f"  Dynamic weights: {enriched['ensemble_weights']}")

        # Sharpe state
        if 'sharpe_mode' in enriched:
            sharpe = enriched.get('sharpe_stats', {}).get('rolling_sharpe', '?')
            lines.append(f"\n=== SYSTEM STATE ===")
            lines.append(f"Sharpe mode: {enriched['sharpe_mode']} (rolling={sharpe})")
            if enriched['sharpe_mode'] == 'RECOVERY':
                lines.append("  *** RECOVERY MODE: only take highest-quality setups ***")

        # Accuracy / consistency
        if 'accuracy' in enriched:
            acc = enriched['accuracy']
            lines.append(f"Current streak: {acc.get('current_streak', '?')}")
            lines.append(f"Weekly PnL: {acc.get('weekly_pnl_pct', 0):.2f}%")
            lines.append(f"Effective spread: {enriched.get('effective_spread', 1.69):.2f}%")

        # Confidence threshold (learned from memory)
        if 'confidence_threshold' in enriched:
            lines.append(f"\nCalibrated confidence threshold: {enriched['confidence_threshold']:.2f}")
            lines.append("(Only proceed if your confidence exceeds this threshold)")

        return '\n'.join(lines)
