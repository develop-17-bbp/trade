"""
Training Data Collector for LLM Fine-Tuning
=============================================
Captures every LLM decision with FULL market context + eventual outcome.
Two training streams matching the 2-pass brain architecture:

  Stream 1 (Scanner): market_data → pattern_report
  Stream 2 (Analyst): patterns + context → trade_decision

Each record is labeled with actual P&L outcome once the trade closes,
creating supervised training data that teaches the model what ACTUALLY
works on THIS exchange with THIS spread cost.

Usage:
    collector = TrainingDataCollector()
    # On every LLM call:
    collector.record_decision(asset, context, scanner_output, analyst_output)
    # On every trade close:
    collector.label_outcome(asset, entry_time, pnl_pct, exit_reason, sl_level)
    # Before fine-tuning:
    scanner_data, analyst_data = collector.export_training_data()
"""

import json
import os
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Outcome labels based on actual P&L (adjusted for spread)
# These teach the model what "good" and "bad" look like on Robinhood
LABEL_THRESHOLDS = {
    'strong_win': 3.0,    # Net P&L > 3% after spread = excellent call
    'win': 1.0,           # Net P&L > 1% = good call
    'breakeven': -1.0,    # Net P&L -1% to 1% = neutral
    'loss': -3.0,         # Net P&L < -1% = bad call
    'hard_stop': -999,    # Hit hard stop = terrible call
}


class TrainingDataCollector:
    """Collects and labels LLM decision data for fine-tuning."""

    def __init__(self, data_dir: str = 'data/finetune',
                 spread_cost_pct: float = 1.69):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.spread_cost = spread_cost_pct

        # Separate streams for 2-pass architecture
        self.decisions_path = self.data_dir / 'raw_decisions.jsonl'
        self.scanner_training_path = self.data_dir / 'scanner_training.jsonl'
        self.analyst_training_path = self.data_dir / 'analyst_training.jsonl'
        self.unlabeled_path = self.data_dir / 'unlabeled_decisions.jsonl'

        # In-memory buffer of open decisions awaiting outcome labels
        self._pending: Dict[str, List[Dict]] = {}  # asset → [decision records]
        self._load_pending()

    def _load_pending(self):
        """Load unlabeled decisions from disk."""
        if self.unlabeled_path.exists():
            try:
                with open(self.unlabeled_path) as f:
                    for line in f:
                        rec = json.loads(line.strip())
                        asset = rec.get('asset', 'UNK')
                        self._pending.setdefault(asset, []).append(rec)
            except Exception as e:
                logger.warning(f"[COLLECTOR] Failed to load pending: {e}")

    def _save_pending(self):
        """Persist unlabeled decisions."""
        try:
            with open(self.unlabeled_path, 'w') as f:
                for records in self._pending.values():
                    for rec in records:
                        f.write(json.dumps(rec, default=str) + '\n')
        except Exception as e:
            logger.warning(f"[COLLECTOR] Failed to save pending: {e}")

    def record_decision(self, asset: str, price: float,
                        scanner_prompt: str, scanner_output: Dict,
                        analyst_prompt: str, analyst_output: Dict,
                        market_context: Dict = None):
        """
        Record a complete LLM decision with full context.
        Called after every Brain v2 evaluation.

        Args:
            asset: BTC, ETH, etc.
            price: Current price at decision time
            scanner_prompt: Full prompt sent to Pass 1 (Mistral scanner)
            scanner_output: JSON response from scanner
            analyst_prompt: Full prompt sent to Pass 2 (Llama analyst)
            analyst_output: JSON response from analyst (proceed, conf, risk, etc.)
            market_context: Additional context (regime, hurst, indicators, etc.)
        """
        record = {
            'timestamp': time.time(),
            'asset': asset,
            'price': price,
            'scanner': {
                'prompt': scanner_prompt,
                'output': scanner_output,
            },
            'analyst': {
                'prompt': analyst_prompt,
                'output': analyst_output,
            },
            'context': market_context or {},
            'proceed': analyst_output.get('proceed', False),
            'confidence': analyst_output.get('confidence', 0),
            'risk_score': analyst_output.get('risk_score', 5),
            'trade_quality': analyst_output.get('trade_quality', 5),
            'predicted_l_level': analyst_output.get('predicted_l_level', '?'),
            'outcome': None,  # Filled in by label_outcome()
        }

        # Save raw decision
        try:
            with open(self.decisions_path, 'a') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            logger.warning(f"[COLLECTOR] Write failed: {e}")

        # Track for outcome labeling if trade was approved
        if record['proceed']:
            self._pending.setdefault(asset, []).append(record)
            self._save_pending()

        return record

    def label_outcome(self, asset: str, entry_time: float,
                      pnl_pct: float, exit_reason: str,
                      sl_level: str = 'L1', duration_min: float = 0):
        """
        Label a pending decision with its actual outcome.
        Called when a trade closes.

        This is the KEY to fine-tuning: teaching the model what works.
        """
        if asset not in self._pending or not self._pending[asset]:
            return

        # Find the decision closest to entry_time
        best_idx = -1
        best_diff = float('inf')
        for i, rec in enumerate(self._pending[asset]):
            diff = abs(rec['timestamp'] - entry_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx < 0 or best_diff > 300:  # 5 min tolerance
            return

        rec = self._pending[asset].pop(best_idx)

        # Net P&L after spread cost
        net_pnl = pnl_pct - self.spread_cost

        # Classify outcome
        if net_pnl >= LABEL_THRESHOLDS['strong_win']:
            label = 'STRONG_WIN'
        elif net_pnl >= LABEL_THRESHOLDS['win']:
            label = 'WIN'
        elif net_pnl >= LABEL_THRESHOLDS['breakeven']:
            label = 'BREAKEVEN'
        elif 'hard stop' in exit_reason.lower():
            label = 'HARD_STOP'
        else:
            label = 'LOSS'

        rec['outcome'] = {
            'pnl_pct': pnl_pct,
            'net_pnl_pct': net_pnl,
            'exit_reason': exit_reason,
            'sl_level': sl_level,
            'duration_min': duration_min,
            'label': label,
        }

        # Generate training examples from this labeled decision
        self._generate_training_examples(rec)
        self._save_pending()

        logger.info(f"[COLLECTOR] Labeled {asset} decision: {label} "
                    f"(net={net_pnl:+.2f}%, {exit_reason})")

    def label_rejection_outcome(self, asset: str, decision_time: float,
                                price_at_decision: float,
                                price_after_1h: float,
                                price_after_4h: float):
        """
        Label a REJECTED trade decision with what WOULD have happened.
        This teaches the model when rejections were correct vs missed opportunities.

        Called periodically to evaluate recent rejections.
        """
        # Find rejected decisions near this time
        all_decisions = []
        if self.decisions_path.exists():
            with open(self.decisions_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line.strip())
                        if (rec['asset'] == asset and
                            not rec['proceed'] and
                            abs(rec['timestamp'] - decision_time) < 120):
                            all_decisions.append(rec)
                    except Exception:
                        continue

        for rec in all_decisions:
            # Would this trade have been profitable?
            hypothetical_pnl_1h = ((price_after_1h - price_at_decision)
                                   / price_at_decision * 100)
            hypothetical_pnl_4h = ((price_after_4h - price_at_decision)
                                   / price_at_decision * 100)
            net_1h = hypothetical_pnl_1h - self.spread_cost
            net_4h = hypothetical_pnl_4h - self.spread_cost

            if net_4h > 2.0:
                rejection_label = 'MISSED_WINNER'  # Should have traded
            elif net_4h < -2.0:
                rejection_label = 'CORRECT_REJECT'  # Good rejection
            else:
                rejection_label = 'NEUTRAL_REJECT'

            rec['outcome'] = {
                'type': 'rejection_eval',
                'hypothetical_pnl_1h': hypothetical_pnl_1h,
                'hypothetical_pnl_4h': hypothetical_pnl_4h,
                'net_pnl_4h': net_4h,
                'label': rejection_label,
            }

            # For missed winners: generate training example with proceed=True
            # For correct rejects: generate training example with proceed=False
            self._generate_training_examples(rec)

    def _generate_training_examples(self, record: Dict):
        """Convert a labeled decision into training format for both models."""
        outcome = record.get('outcome', {})
        label = outcome.get('label', 'UNKNOWN')

        # ── Scanner Training Data ──
        # Input: market data (extracted from scanner prompt)
        # Output: improved pattern report based on whether it led to profit
        scanner_example = {
            'input': self._extract_scanner_input(record),
            'output': self._build_scanner_target(record, label),
            'label': label,
            'asset': record['asset'],
            'timestamp': record['timestamp'],
        }

        try:
            with open(self.scanner_training_path, 'a') as f:
                f.write(json.dumps(scanner_example, default=str) + '\n')
        except Exception:
            pass

        # ── Analyst Training Data ──
        # Input: patterns + context (extracted from analyst prompt)
        # Output: correct decision based on actual outcome
        analyst_example = {
            'input': self._extract_analyst_input(record),
            'output': self._build_analyst_target(record, label),
            'label': label,
            'asset': record['asset'],
            'timestamp': record['timestamp'],
        }

        try:
            with open(self.analyst_training_path, 'a') as f:
                f.write(json.dumps(analyst_example, default=str) + '\n')
        except Exception:
            pass

    def _extract_scanner_input(self, record: Dict) -> str:
        """Extract the market data portion from scanner prompt (without instructions)."""
        prompt = record.get('scanner', {}).get('prompt', '')
        # The scanner prompt has market data between delimiters
        # Return the data section only (not the instructions)
        if 'CANDLE DATA' in prompt:
            start = prompt.find('CANDLE DATA')
            return prompt[start:start + 2000]  # Cap at 2000 chars
        return prompt[-2000:]  # Last 2000 chars (data section)

    def _build_scanner_target(self, record: Dict, label: str) -> Dict:
        """Build the ideal scanner output based on actual outcome."""
        scanner_out = record.get('scanner', {}).get('output', {})

        # Keep the original pattern analysis but adjust bias/strength
        # based on whether the trade actually worked
        target = dict(scanner_out)

        if label in ('STRONG_WIN', 'WIN'):
            # Scanner should have been more bullish
            target['pattern_strength'] = max(7, scanner_out.get('pattern_strength', 5))
            target['resembles_winner'] = True
        elif label in ('LOSS', 'HARD_STOP'):
            # Scanner should have caught danger
            target['pattern_strength'] = min(3, scanner_out.get('pattern_strength', 5))
            if not target.get('danger_pattern'):
                target['danger_pattern'] = 'post-hoc: setup led to loss'
        elif label == 'MISSED_WINNER':
            # Scanner was too bearish — should have identified opportunity
            target['pattern_bias'] = 'BULLISH'
            target['pattern_strength'] = 7

        return target

    def _extract_analyst_input(self, record: Dict) -> str:
        """Extract analyst prompt input (patterns + context, not instructions)."""
        prompt = record.get('analyst', {}).get('prompt', '')
        # Return everything between PATTERN SCAN RESULTS and YOUR ANALYSIS
        start = prompt.find('PATTERN SCAN RESULTS')
        end = prompt.find('YOUR ANALYSIS')
        if start >= 0 and end > start:
            return prompt[start:end]
        return prompt[-3000:]

    def _build_analyst_target(self, record: Dict, label: str) -> Dict:
        """Build the ideal analyst output based on actual outcome."""
        original = record.get('analyst', {}).get('output', {})
        outcome = record.get('outcome', {})

        if label in ('STRONG_WIN', 'WIN'):
            return {
                'proceed': True,
                'confidence': min(0.95, max(0.75, outcome.get('net_pnl_pct', 2) / 10 + 0.7)),
                'position_size_pct': 5,
                'risk_score': max(2, original.get('risk_score', 5) - 2),
                'trade_quality': min(10, max(7, original.get('trade_quality', 5) + 2)),
                'predicted_l_level': outcome.get('sl_level', 'L3'),
                'pattern_alignment': 'CONFIRMS',
                'bull_case': original.get('bull_case', 'Setup confirmed by outcome'),
                'bear_case': original.get('bear_case', 'Manageable risk'),
                'facilitator_verdict': f"ENTER — this setup netted {outcome.get('net_pnl_pct', 0):+.1f}% after spread",
            }
        elif label in ('LOSS', 'HARD_STOP'):
            return {
                'proceed': False,
                'confidence': 0.20,
                'position_size_pct': 0,
                'risk_score': min(10, max(7, original.get('risk_score', 5) + 2)),
                'trade_quality': max(1, min(3, original.get('trade_quality', 5) - 2)),
                'predicted_l_level': 'L1',
                'pattern_alignment': 'CONTRADICTS',
                'bull_case': 'Insufficient — setup failed',
                'bear_case': f"CONFIRMED: {outcome.get('exit_reason', 'loss')}",
                'facilitator_verdict': f"REJECT — this setup lost {outcome.get('net_pnl_pct', 0):+.1f}% after spread",
            }
        elif label == 'MISSED_WINNER':
            return {
                'proceed': True,
                'confidence': 0.70,
                'position_size_pct': 3,
                'risk_score': 5,
                'trade_quality': 7,
                'predicted_l_level': 'L3',
                'pattern_alignment': 'CONFIRMS',
                'bull_case': f"Missed: price moved {outcome.get('hypothetical_pnl_4h', 0):+.1f}% in 4h",
                'bear_case': original.get('bear_case', 'Risk was manageable'),
                'facilitator_verdict': f"ENTER — rejection missed a {outcome.get('net_pnl_4h', 0):+.1f}% net winner",
            }
        elif label == 'CORRECT_REJECT':
            return {
                'proceed': False,
                'confidence': 0.15,
                'position_size_pct': 0,
                'risk_score': 8,
                'trade_quality': 2,
                'predicted_l_level': 'L1',
                'pattern_alignment': 'CONTRADICTS',
                'bull_case': 'None — rejection was correct',
                'bear_case': f"Price dropped {outcome.get('hypothetical_pnl_4h', 0):+.1f}% in 4h",
                'facilitator_verdict': "REJECT — correct call, would have lost money",
            }
        else:
            # BREAKEVEN / NEUTRAL — keep original but slight adjustments
            return original

    def export_training_data(self, min_examples: int = 20) -> Tuple[str, str]:
        """
        Export formatted training data for both models.

        Returns:
            (scanner_path, analyst_path) — paths to training JSONL files
        """
        scanner_count = 0
        analyst_count = 0

        if self.scanner_training_path.exists():
            with open(self.scanner_training_path) as f:
                scanner_count = sum(1 for _ in f)

        if self.analyst_training_path.exists():
            with open(self.analyst_training_path) as f:
                analyst_count = sum(1 for _ in f)

        logger.info(f"[COLLECTOR] Training data: {scanner_count} scanner, "
                    f"{analyst_count} analyst examples")

        if scanner_count < min_examples:
            logger.warning(f"[COLLECTOR] Only {scanner_count} scanner examples "
                          f"(need {min_examples}). Generating synthetic...")
            self._generate_synthetic_data(min_examples - scanner_count)

        return str(self.scanner_training_path), str(self.analyst_training_path)

    def _generate_synthetic_data(self, n: int):
        """Generate synthetic labeled examples from known market patterns."""
        import random

        regimes = ['TRENDING', 'RANGING', 'VOLATILE', 'CHOPPY']
        signals = ['BUY', 'SELL']

        for i in range(n):
            regime = random.choice(regimes)
            signal = random.choice(signals)
            is_winner = random.random() < 0.45  # 45% win rate target

            # Synthetic scanner output
            scanner_out = {
                'pattern_bias': 'BULLISH' if signal == 'BUY' else 'BEARISH',
                'pattern_strength': random.randint(6, 9) if is_winner else random.randint(2, 5),
                'strongest_signal': random.choice(['EMA Cross', 'Hammer', 'Engulfing', 'Doji']),
                'danger_pattern': '' if is_winner else random.choice(['Divergence', 'Exhaustion', 'Double Top']),
                'resembles_winner': is_winner,
            }

            # Synthetic analyst output
            pnl = random.uniform(2, 8) if is_winner else random.uniform(-6, -1)
            net_pnl = pnl - 1.69
            label = 'WIN' if net_pnl > 1 else 'LOSS' if net_pnl < -1 else 'BREAKEVEN'

            analyst_target = self._build_analyst_target(
                {'analyst': {'output': {
                    'risk_score': random.randint(3, 5) if is_winner else random.randint(6, 8),
                    'trade_quality': random.randint(6, 9) if is_winner else random.randint(2, 5),
                    'bull_case': 'Synthetic example',
                    'bear_case': 'Synthetic risk',
                }}},
                label
            )

            # Scanner training example
            scanner_input = (
                f"Asset: {'BTC' if i % 2 == 0 else 'ETH'} | Regime: {regime}\n"
                f"EMA(8): RISING | Signal: {signal} | ATR: ${random.uniform(50, 150):.2f}\n"
                f"RSI: {random.randint(30, 70)} | MACD: {'BULLISH' if signal == 'BUY' else 'BEARISH'}\n"
                f"1h: {'RISING' if is_winner else 'FALLING'} | 4h: {'RISING' if is_winner else 'FLAT'}"
            )

            with open(self.scanner_training_path, 'a') as f:
                f.write(json.dumps({
                    'input': scanner_input,
                    'output': scanner_out,
                    'label': label,
                    'synthetic': True,
                }, default=str) + '\n')

            # Analyst training example
            analyst_input = (
                f"Pattern Bias: {scanner_out['pattern_bias']} Strength: {scanner_out['pattern_strength']}/10\n"
                f"Signal: {signal} | Regime: {regime} | Spread: 1.69%\n"
                f"EMA Direction: RISING | HTF Alignment: {'3/3' if is_winner else '1/3'}\n"
                f"Entry Score: {random.randint(4, 8) if is_winner else random.randint(0, 4)}/10"
            )

            with open(self.analyst_training_path, 'a') as f:
                f.write(json.dumps({
                    'input': analyst_input,
                    'output': analyst_target,
                    'label': label,
                    'synthetic': True,
                }, default=str) + '\n')

    def get_stats(self) -> Dict:
        """Return collection statistics."""
        total = 0
        labeled = 0
        pending = sum(len(v) for v in self._pending.values())

        if self.decisions_path.exists():
            with open(self.decisions_path) as f:
                for line in f:
                    total += 1
                    try:
                        rec = json.loads(line.strip())
                        if rec.get('outcome'):
                            labeled += 1
                    except Exception:
                        pass

        scanner_count = 0
        analyst_count = 0
        if self.scanner_training_path.exists():
            with open(self.scanner_training_path) as f:
                scanner_count = sum(1 for _ in f)
        if self.analyst_training_path.exists():
            with open(self.analyst_training_path) as f:
                analyst_count = sum(1 for _ in f)

        return {
            'total_decisions': total,
            'labeled': labeled,
            'pending_label': pending,
            'scanner_examples': scanner_count,
            'analyst_examples': analyst_count,
        }
