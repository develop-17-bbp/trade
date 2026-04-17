"""
ACT v8.0 Test Suite — Memory, Economic Intelligence, Sharpe, Accuracy, Backtesting
"""
import pytest
import time
import math


# ═══════════════════════════════════════════════════════════════
# 1. MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════

class TestMemorySystem:
    def test_quant_memory_records_and_recalls(self, tmp_path):
        from src.memory.quant_memory import QuantMemory
        m = QuantMemory('t_lgbm', db_dir=str(tmp_path))
        m.record_prediction('BTC', 'LONG', 0.8, ['ema', 'rsi'], 'BULL', 0.6, 0.15, 'US', 3.0, 'WIN')
        m.record_prediction('BTC', 'LONG', 0.7, ['macd', 'atr'], 'BULL', 0.5, 0.20, 'EU', -2.0, 'LOSS')
        stats = m.get_stats()
        assert stats['total_events'] == 2
        assert 0.0 <= stats['win_rate'] <= 1.0

    def test_agent_memory_recalibrates_weights(self, tmp_path):
        from src.memory.agent_memory import AgentMemory
        m = AgentMemory('t_agent', db_dir=str(tmp_path))
        for _ in range(8):
            m.record_vote('LONG', 0.8, 'bullish momentum', False, 2.0, 'WIN')
        for _ in range(2):
            m.record_vote('LONG', 0.6, 'weak signal', True, -1.5, 'LOSS')
        weight = m.recalibrate_weight()
        assert weight > 1.0
        acc = m.get_agent_accuracy(last_n=10)
        assert acc['accuracy'] == pytest.approx(0.8)

    def test_llm_memory_builds_prompt_context(self, tmp_path):
        from src.memory.llm_memory import LLMMemory
        m = LLMMemory('t_scanner', db_dir=str(tmp_path))
        for i in range(5):
            m.record_decision(f'hash_{i}', {'proceed': True, 'confidence': 0.8},
                              2.5, 'WIN', False, 3.0, 2.5)
        ctx = m.build_dynamic_prompt_context({'market_regime': 'BULL', 'action_taken': 'LONG'})
        assert isinstance(ctx, str)

    def test_memory_consolidation_extracts_patterns(self, tmp_path):
        from src.memory.quant_memory import QuantMemory
        m = QuantMemory('t_consolidate', db_dir=str(tmp_path))
        for i in range(10):
            m.record_prediction('BTC', 'LONG', 0.7, ['ema'], 'BULL', 0.5, 0.15, 'US',
                                2.0 if i < 7 else -1.0, 'WIN' if i < 7 else 'LOSS')
        m.consolidate()
        patterns = m.extract_winning_patterns()
        assert isinstance(patterns, dict)

    def test_memory_persists_across_restart(self, tmp_path):
        from src.memory.quant_memory import QuantMemory
        m1 = QuantMemory('t_persist', db_dir=str(tmp_path))
        m1.record_prediction('BTC', 'LONG', 0.8, ['ema'], 'BULL', 0.5, 0.15, 'US', 3.0, 'WIN')
        m1._conn.close()
        m2 = QuantMemory('t_persist', db_dir=str(tmp_path))
        stats = m2.get_stats()
        assert stats['total_events'] == 1


# ═══════════════════════════════════════════════════════════════
# 2. ECONOMIC LAYERS
# ═══════════════════════════════════════════════════════════════

class TestEconomicLayers:
    def test_all_12_layers_initialize(self):
        from src.data.economic_intelligence import EconomicIntelligence
        ei = EconomicIntelligence()
        assert len(ei._layers) == 12

    def test_macro_summary_returns_valid_structure(self):
        from src.data.economic_intelligence import EconomicIntelligence
        ei = EconomicIntelligence()
        summary = ei.get_macro_summary()
        assert 'composite_signal' in summary
        assert summary['composite_signal'] in ['BULLISH', 'BEARISH', 'NEUTRAL', 'CRISIS']
        assert 0 <= summary['macro_risk'] <= 100
        assert isinstance(summary['pre_event_flag'], bool)

    def test_llm_context_block_is_string(self):
        from src.data.economic_intelligence import EconomicIntelligence
        ei = EconomicIntelligence()
        block = ei.get_llm_context_block()
        assert isinstance(block, str)
        assert 'MACRO INTELLIGENCE' in block

    def test_finetune_context_has_all_keys(self):
        from src.data.economic_intelligence import EconomicIntelligence
        ei = EconomicIntelligence()
        ctx = ei.get_finetune_context()
        for key in ['macro_composite', 'macro_risk', 'usd_regime', 'crisis', 'pre_event', 'layer_signals']:
            assert key in ctx

    def test_fear_greed_signal_logic(self):
        from src.data.layers.social_sentiment import SocialSentiment
        ss = SocialSentiment()
        # Test contrarian logic
        ss._last_result = {'value': 15, 'signal': 'BULLISH', 'confidence': 0.7, 'stale': False}
        assert ss.get_retail_vs_institutional_signal() == 'INSTITUTIONAL_ACCUMULATION'
        ss._last_result = {'value': 85, 'signal': 'BEARISH', 'confidence': 0.7, 'stale': False}
        assert ss.get_retail_vs_institutional_signal() == 'RETAIL_FOMO'


# ═══════════════════════════════════════════════════════════════
# 3. BACKTESTING
# ═══════════════════════════════════════════════════════════════

class TestBacktesting:
    def test_monte_carlo_returns_risk_metrics(self):
        from src.backtesting.monte_carlo_bt import MonteCarloBacktest
        mc = MonteCarloBacktest({'monte_carlo_runs': 1000})
        returns = [1.5, -0.8, 2.1, -1.2, 0.5, 3.0, -0.3, 1.8, -2.5, 0.9,
                   1.1, -0.5, 2.3, -1.0, 0.7, 1.6, -0.2, 2.8, -1.8, 0.4]
        result = mc.run(returns, initial_equity=16000)
        assert 'probability_of_ruin' in result
        assert 0.0 <= result['probability_of_ruin'] <= 1.0
        assert result['equity_p50'] > 0
        assert 'var_95' in result

    def test_monte_carlo_needs_minimum_trades(self):
        from src.backtesting.monte_carlo_bt import MonteCarloBacktest
        mc = MonteCarloBacktest()
        result = mc.run([1.0, -0.5])
        assert 'error' in result

    def test_pinescript_export_valid_syntax(self):
        from src.backtesting.pinescript_exporter import PineScriptExporter
        e = PineScriptExporter()
        script = e.generate_strategy('BTCUSD', '5')
        assert '//@version=5' in script
        assert 'strategy(' in script
        assert 'ACT v8.0' in script
        assert 'alertcondition(' in script


# ═══════════════════════════════════════════════════════════════
# 4. SHARPE OPTIMIZER
# ═══════════════════════════════════════════════════════════════

class TestSharpeOptimizer:
    def test_recovery_mode_tightens_filters(self):
        from src.optimization.sharpe_optimizer import SharpeOptimizer
        so = SharpeOptimizer()
        # Simulate losing trades → low Sharpe
        for _ in range(20):
            so.record_trade(-1.5)
        assert so.mode == 'RECOVERY'
        adj = so.get_filter_adjustments()
        assert adj['min_entry_score_add'] > 0
        assert adj['position_size_mult'] < 1.0

    def test_trade_quality_score_range(self):
        from src.optimization.sharpe_optimizer import SharpeOptimizer
        so = SharpeOptimizer()
        # Minimum score with no good signals
        score_low = so.compute_trade_quality_score(0.3, 15, 70, 0.3, 0.3, 80, False)
        assert 0 <= score_low <= 100
        # Maximum score with all good signals
        score_high = so.compute_trade_quality_score(0.8, 40, 30, 0.8, 0.85, 20, True)
        assert score_high > score_low
        assert score_high >= 55  # should pass quality gate

    def test_sharpe_adjusted_sizing(self):
        from src.optimization.sharpe_optimizer import SharpeOptimizer
        so = SharpeOptimizer()
        # Low quality → small size
        size_low = so.sharpe_adjusted_size(100, quality_score=30)
        # High quality → larger size
        size_high = so.sharpe_adjusted_size(100, quality_score=80)
        assert size_high > size_low

    def test_sortino_ratio_computed(self):
        from src.optimization.sharpe_optimizer import SharpeOptimizer
        so = SharpeOptimizer()
        for r in [2.0, -1.0, 1.5, -0.5, 3.0, 0.8, -0.3, 2.5, 1.0, -1.5]:
            so.record_trade(r)
        sortino = so.get_sortino_ratio()
        assert sortino is not None
        assert sortino > 0


# ═══════════════════════════════════════════════════════════════
# 5. CONSISTENCY GUARDIAN
# ═══════════════════════════════════════════════════════════════

class TestConsistencyGuardian:
    def test_3_consecutive_losses_triggers_preservation(self):
        from src.learning.accuracy_engine import AccuracyEngine
        ae = AccuracyEngine()
        ae.record_trade_outcome(-2.0)
        ae.record_trade_outcome(-1.5)
        ae.record_trade_outcome(-1.8)
        skip, reason = ae.should_skip_trade()
        assert skip is True
        assert 'Capital Preservation' in reason or ae._capital_preservation_remaining > 0

    def test_weekly_drawdown_block(self):
        from src.learning.accuracy_engine import AccuracyEngine
        ae = AccuracyEngine({'weekly_drawdown_block_threshold': 0.02})
        # Lose more than 2% in a week
        ae.record_trade_outcome(-1.5)
        ae.record_trade_outcome(-1.5)
        skip, reason = ae.should_skip_trade()
        assert skip is True or ae._weekly_blocked is True

    def test_dynamic_ensemble_weights_sum_to_one(self):
        from src.learning.accuracy_engine import AccuracyEngine
        ae = AccuracyEngine()
        # Record predictions for 3 models
        for _ in range(20):
            ae.record_model_prediction('lgbm', 'BULL', 'LONG', True)
            ae.record_model_prediction('patchtst', 'BULL', 'LONG', False)
            ae.record_model_prediction('rl', 'BULL', 'LONG', True)
        weights = ae.get_ensemble_weights('BULL')
        assert abs(sum(weights.values()) - 1.0) < 0.01
        # lgbm and rl should have higher weights than patchtst
        assert weights.get('lgbm', 0) > weights.get('patchtst', 0)

    def test_agent_dynamic_weight_scales_with_accuracy(self):
        from src.learning.accuracy_engine import AccuracyEngine
        ae = AccuracyEngine()
        # Good agent
        for _ in range(20):
            ae.record_agent_vote('trend_momentum', True)
        # Bad agent
        for _ in range(20):
            ae.record_agent_vote('mean_reversion', False)
        good_w = ae.get_agent_dynamic_weight('trend_momentum')
        bad_w = ae.get_agent_dynamic_weight('mean_reversion')
        assert good_w > bad_w

    def test_robinhood_slippage_profiling(self):
        from src.learning.accuracy_engine import AccuracyEngine
        ae = AccuracyEngine()
        ae.record_fill(80000, 80050, 2.5)
        ae.record_fill(80000, 80030, 3.0)
        avg = ae.get_avg_slippage()
        assert avg > 0
        eff_spread = ae.get_effective_spread(1.69)
        assert eff_spread > 1.69


# ═══════════════════════════════════════════════════════════════
# 6. FINETUNE ENRICHER
# ═══════════════════════════════════════════════════════════════

class TestFinetuneEnricher:
    def test_enricher_builds_complete_prompt(self):
        from src.learning.finetune_enricher import FinetuneEnricher
        from src.data.economic_intelligence import EconomicIntelligence
        from src.learning.accuracy_engine import AccuracyEngine
        from src.optimization.sharpe_optimizer import SharpeOptimizer

        enricher = FinetuneEnricher(
            econ_intel=EconomicIntelligence(),
            accuracy_engine=AccuracyEngine(),
            sharpe_optimizer=SharpeOptimizer(),
        )
        ctx = {'asset': 'BTC', 'regime': 'BULL', 'direction': 'LONG'}
        block = enricher.build_enriched_prompt_block(ctx)
        assert isinstance(block, str)
        assert 'MACRO INTELLIGENCE' in block
        assert len(block) > 100

    def test_enricher_returns_dict(self):
        from src.learning.finetune_enricher import FinetuneEnricher
        enricher = FinetuneEnricher()
        ctx = {'asset': 'ETH', 'regime': 'BEAR'}
        enriched = enricher.enrich(ctx)
        assert isinstance(enriched, dict)
        assert enriched['asset'] == 'ETH'
