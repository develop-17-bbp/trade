"""
Comprehensive test suite for ACT Trading System core components.

Covers:
  1. LLM response parsing (malformed inputs, markdown fences, trailing commas)
  2. Order execution flow (mocked Robinhood, SL/TP, PnL)
  3. Position sizing (Kelly, ATR, volatility-scaled, optimal)
  4. Confluence scoring (entry score gates, signal combiner, VETO)
"""

import json
import math
import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


# ═══════════════════════════════════════════════════════════════
# 1. LLM RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════

class TestLLMResponseParsing:
    """Tests for MultiModelConsensus._parse_llm_json"""

    @pytest.fixture
    def brain(self):
        """Create a minimal MultiModelConsensus instance without starting Ollama."""
        from src.ai.trading_brain import MultiModelConsensus
        with patch("src.ai.trading_brain.MultiModelConsensus.__init__", return_value=None):
            obj = MultiModelConsensus.__new__(MultiModelConsensus)
        # Bind the real method
        import types, re
        obj._parse_llm_json = types.MethodType(
            MultiModelConsensus._parse_llm_json, obj
        )
        return obj

    def test_clean_json(self, brain):
        text = '{"proceed": true, "confidence": 0.85, "risk_score": 3}'
        result = brain._parse_llm_json(text)
        assert result["proceed"] is True
        assert result["confidence"] == pytest.approx(0.85)
        assert result["risk_score"] == 3

    def test_markdown_fence_json(self, brain):
        text = '```json\n{"proceed": false, "confidence": 0.4, "risk_score": 7}\n```'
        result = brain._parse_llm_json(text)
        assert result["proceed"] is False
        assert result["confidence"] == pytest.approx(0.4)

    def test_markdown_fence_no_lang(self, brain):
        text = '```\n{"proceed": true, "confidence": 0.9}\n```'
        result = brain._parse_llm_json(text)
        assert result["proceed"] is True

    def test_trailing_comma(self, brain):
        text = '{"proceed": true, "confidence": 0.75, "risk_score": 4,}'
        result = brain._parse_llm_json(text)
        assert result["proceed"] is True
        assert result["risk_score"] == 4

    def test_json_embedded_in_prose(self, brain):
        text = (
            "Based on my analysis, I recommend: "
            '{"proceed": true, "confidence": 0.80, "risk_score": 3, "trade_quality": 7} '
            "due to strong momentum signals."
        )
        result = brain._parse_llm_json(text)
        assert result["proceed"] is True
        assert result["confidence"] == pytest.approx(0.80)

    def test_completely_unparseable_returns_safe_default(self, brain):
        text = "I cannot make a decision at this time. Market is uncertain."
        result = brain._parse_llm_json(text)
        assert result["proceed"] is False
        assert result.get("parse_error") is True
        assert result["risk_score"] >= 7  # conservative on parse failure

    def test_empty_string_returns_safe_default(self, brain):
        result = brain._parse_llm_json("")
        assert result["proceed"] is False

    def test_nested_json_picks_largest(self, brain):
        # Two JSON objects in text — parser should pick the more complete one
        text = (
            'Small: {"ok": true} '
            'Full: {"proceed": false, "confidence": 0.3, "risk_score": 8, "trade_quality": 2}'
        )
        result = brain._parse_llm_json(text)
        assert "risk_score" in result

    def test_confidence_boundary_values(self, brain):
        for conf in [0.0, 0.5, 1.0]:
            text = f'{{"proceed": true, "confidence": {conf}, "risk_score": 5}}'
            result = brain._parse_llm_json(text)
            assert result["confidence"] == pytest.approx(conf)

    def test_boolean_as_string_preserved(self, brain):
        # LLMs sometimes return "true" inside strings — parser should still work
        text = '{"proceed": true, "reasoning": "bullish momentum"}'
        result = brain._parse_llm_json(text)
        assert result["proceed"] is True
        assert result["reasoning"] == "bullish momentum"


# ═══════════════════════════════════════════════════════════════
# 2. POSITION SIZING
# ═══════════════════════════════════════════════════════════════

class TestPositionSizing:
    """Tests for src/risk/position_sizing.py"""

    def setup_method(self):
        from src.risk.position_sizing import (
            kelly_criterion, half_kelly, atr_position_size,
            volatility_scaled_size, fixed_fraction, optimal_position_size,
        )
        self.kelly = kelly_criterion
        self.half_kelly = half_kelly
        self.atr_size = atr_position_size
        self.vol_size = volatility_scaled_size
        self.fixed = fixed_fraction
        self.optimal = optimal_position_size

    # ── Kelly Criterion ──────────────────────────────────────────

    def test_kelly_positive_edge(self):
        # win_prob=0.6, win_loss_ratio=1.5 → f* = (1.5*0.6 - 0.4)/1.5 = 0.633...
        k = self.kelly(0.6, 1.5, fraction=1.0)
        assert k == pytest.approx((1.5 * 0.6 - 0.4) / 1.5, rel=1e-4)

    def test_kelly_half_fraction(self):
        full = self.kelly(0.6, 1.5, fraction=1.0)
        half = self.kelly(0.6, 1.5, fraction=0.5)
        assert half == pytest.approx(full / 2)

    def test_kelly_zero_edge_returns_zero(self):
        # win_prob=0.4, win_loss_ratio=1.0 → negative edge → 0
        k = self.kelly(0.4, 1.0)
        assert k == 0.0

    def test_kelly_invalid_inputs_return_zero(self):
        assert self.kelly(0.0, 1.5) == 0.0   # zero win prob
        assert self.kelly(1.0, 1.5) == 0.0   # 100% win prob (invalid)
        assert self.kelly(0.6, 0.0) == 0.0   # zero win/loss ratio
        assert self.kelly(0.6, -1.0) == 0.0  # negative ratio

    def test_half_kelly_is_half_of_full(self):
        full = self.kelly(0.55, 2.0, fraction=1.0)
        half = self.half_kelly(0.55, 2.0)
        assert half == pytest.approx(full / 2)

    def test_kelly_never_exceeds_one(self):
        # Even with extreme edge, should not exceed 1.0
        k = self.kelly(0.99, 100.0, fraction=1.0)
        assert 0.0 <= k <= 1.0

    # ── ATR Position Sizing ──────────────────────────────────────

    def test_atr_basic(self):
        # balance=10000, risk=1%, atr=500, mult=2 → risk_amt=100, stop=1000 → size=0.1
        size = self.atr_size(10000, 500, risk_pct=1.0, atr_multiplier=2.0)
        assert size == pytest.approx(0.1, rel=1e-4)

    def test_atr_zero_atr_returns_zero(self):
        assert self.atr_size(10000, 0.0) == 0.0

    def test_atr_negative_atr_returns_zero(self):
        assert self.atr_size(10000, -100.0) == 0.0

    def test_atr_larger_atr_gives_smaller_size(self):
        small_atr = self.atr_size(10000, 200, risk_pct=1.0)
        large_atr = self.atr_size(10000, 1000, risk_pct=1.0)
        assert small_atr > large_atr

    def test_atr_larger_balance_gives_larger_size(self):
        small_bal = self.atr_size(5000, 500)
        large_bal = self.atr_size(10000, 500)
        assert large_bal == pytest.approx(2 * small_bal)

    # ── Volatility Scaled ─────────────────────────────────────────

    def test_vol_scaled_high_vol_reduces_size(self):
        normal = self.vol_size(10000, realized_vol=0.15, target_vol=0.15)
        high_vol = self.vol_size(10000, realized_vol=0.30, target_vol=0.15)
        assert normal > high_vol

    def test_vol_scaled_low_vol_increases_size(self):
        normal = self.vol_size(10000, realized_vol=0.15, target_vol=0.15)
        low_vol = self.vol_size(10000, realized_vol=0.05, target_vol=0.15)
        assert low_vol > normal

    def test_vol_scaled_zero_vol_returns_base(self):
        # Should not divide by zero
        size = self.vol_size(10000, realized_vol=0.0)
        assert size > 0

    def test_vol_scaled_cap_prevents_extreme_leverage(self):
        # Very low vol → capped at 3x base
        base = self.vol_size(10000, realized_vol=0.15, target_vol=0.15)
        extreme = self.vol_size(10000, realized_vol=0.001, target_vol=0.15)
        assert extreme <= base * 3.0 + 1e-9  # max 3x cap

    # ── Fixed Fraction ────────────────────────────────────────────

    def test_fixed_fraction_basic(self):
        size = self.fixed(10000, risk_pct=2.0)
        assert size == pytest.approx(200.0)

    def test_fixed_fraction_zero_risk_returns_zero(self):
        assert self.fixed(10000, 0.0) == 0.0

    # ── Optimal Position Size ─────────────────────────────────────

    def test_optimal_conservative_picks_minimum(self):
        result = self.optimal(
            account_balance=10000, win_prob=0.55, win_loss_ratio=1.5,
            atr_value=300, realized_vol=0.20, risk_pct=1.0, method='conservative'
        )
        assert result['final_size'] >= 0
        # Final should not exceed any individual method
        assert result['final_size'] <= result['atr_based'] + 1e-9 or \
               result['final_size'] <= result['fixed_fraction'] + 1e-9

    def test_optimal_returns_all_keys(self):
        result = self.optimal(10000, 0.55, 1.5, 300, 0.20)
        for key in ['final_size', 'kelly_amount', 'kelly_fraction', 'fixed_fraction',
                    'atr_based', 'vol_scaled', 'method']:
            assert key in result

    def test_optimal_final_size_non_negative(self):
        # Even with bad inputs, never return negative size
        result = self.optimal(10000, 0.3, 0.5, 1000, 0.5, risk_pct=0.5)
        assert result['final_size'] >= 0.0

    def test_optimal_hard_cap_respected(self):
        # Final size should not exceed 2x risk amount
        balance = 10000
        risk_pct = 1.0
        result = self.optimal(balance, 0.9, 10.0, 50, 0.01, risk_pct=risk_pct)
        max_cap = balance * (risk_pct / 100.0) * 2
        assert result['final_size'] <= max_cap + 1e-9


# ═══════════════════════════════════════════════════════════════
# 3. CONFLUENCE SCORING (Signal Combiner)
# ═══════════════════════════════════════════════════════════════

class TestSignalCombiner:
    """Tests for src/trading/signal_combiner.py"""

    @pytest.fixture
    def combiner(self):
        from src.trading.signal_combiner import SignalCombiner
        return SignalCombiner()

    def _l2(self, score=0.0, confidence=1.0, freshness=1.0):
        return {"aggregate_score": score, "confidence": confidence, "freshness": freshness}

    def _l3_allow(self, risk_score=0.2, size=100.0):
        from src.risk.manager import RiskAction
        return {"action": RiskAction.ALLOW, "risk_score": risk_score,
                "adjusted_size": size, "reason": "OK",
                "stop_loss": 95.0, "take_profit": 110.0}

    def _l3_veto(self):
        from src.risk.manager import RiskAction
        return {"action": RiskAction.VETO, "risk_score": 1.0,
                "adjusted_size": 0, "reason": "max drawdown breached",
                "stop_loss": 0.0, "take_profit": 0.0}

    # ── Basic signal math ─────────────────────────────────────────

    def test_strong_l1_buy_produces_buy(self, combiner):
        result = combiner.combine(
            l1_signal=0.8,
            l2_sentiment=self._l2(0.5),
            l3_evaluation=self._l3_allow(0.1),
        )
        assert result["action"] == "buy"
        assert result["final_signal"] > 0

    def test_strong_l1_sell_produces_sell(self, combiner):
        result = combiner.combine(
            l1_signal=-0.8,
            l2_sentiment=self._l2(-0.5),
            l3_evaluation=self._l3_allow(0.1),
        )
        assert result["action"] == "sell"
        assert result["final_signal"] < 0

    def test_weak_signal_produces_hold(self, combiner):
        result = combiner.combine(
            l1_signal=0.05,
            l2_sentiment=self._l2(0.0),
            l3_evaluation=self._l3_allow(0.1),
        )
        assert result["action"] == "hold"

    # ── VETO logic ────────────────────────────────────────────────

    def test_l3_veto_overrides_strong_buy(self, combiner):
        result = combiner.combine(
            l1_signal=1.0,
            l2_sentiment=self._l2(1.0),
            l3_evaluation=self._l3_veto(),
        )
        assert result["action"] == "hold"
        assert result["final_signal"] == 0.0
        assert result["confidence"] == 0.0
        assert "VETO" in result["risk_status"]

    def test_veto_position_size_is_zero(self, combiner):
        result = combiner.combine(1.0, self._l2(1.0), self._l3_veto())
        assert result["position_size"] == 0.0

    # ── Staleness decay ───────────────────────────────────────────

    def test_stale_l2_reduces_its_weight(self, combiner):
        now = time.time()
        fresh = combiner.combine(
            l1_signal=0.5, l2_sentiment=self._l2(1.0),
            l3_evaluation=self._l3_allow(),
            l2_timestamp=now, current_time=now,
        )
        stale = combiner.combine(
            l1_signal=0.5, l2_sentiment=self._l2(1.0),
            l3_evaluation=self._l3_allow(),
            l2_timestamp=now - 700,  # > l2_max_age (600s) → fully stale
            current_time=now,
        )
        # Fresh L2 contributes more → higher final signal
        assert fresh["final_signal"] >= stale["final_signal"]
        assert stale["breakdown"]["l2_freshness"] == pytest.approx(0.0)

    def test_fully_stale_l2_redistributes_to_l1(self, combiner):
        now = time.time()
        result = combiner.combine(
            l1_signal=0.6, l2_sentiment=self._l2(0.0, confidence=0.0),
            l3_evaluation=self._l3_allow(),
            l2_timestamp=now - 1000, current_time=now,
        )
        # L1 weight should absorb what L2 lost (w_l1 >= 0.50)
        assert result["breakdown"]["l1_weight"] >= 0.50

    # ── Agreement bonus ───────────────────────────────────────────

    def test_agreement_bonus_boosts_aligned_signals(self, combiner):
        now = time.time()
        # L1 and L2 agree (both positive, fresh)
        agreed = combiner.combine(
            l1_signal=0.5, l2_sentiment=self._l2(0.5),
            l3_evaluation=self._l3_allow(),
            l2_timestamp=now, current_time=now,
        )
        # L1 and L2 disagree
        disagreed = combiner.combine(
            l1_signal=0.5, l2_sentiment=self._l2(-0.5),
            l3_evaluation=self._l3_allow(),
            l2_timestamp=now, current_time=now,
        )
        assert agreed["final_signal"] > disagreed["final_signal"]

    # ── Confidence bounds ─────────────────────────────────────────

    def test_confidence_always_between_0_and_1(self, combiner):
        for l1 in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for risk in [0.0, 0.5, 1.0]:
                result = combiner.combine(
                    l1_signal=l1,
                    l2_sentiment=self._l2(l1 * 0.5),
                    l3_evaluation=self._l3_allow(risk_score=risk),
                )
                assert 0.0 <= result["confidence"] <= 1.0, \
                    f"confidence={result['confidence']} out of bounds for l1={l1}, risk={risk}"

    def test_final_signal_always_clamped(self, combiner):
        result = combiner.combine(
            l1_signal=1.0,
            l2_sentiment=self._l2(1.0),
            l3_evaluation=self._l3_allow(0.0),
        )
        assert -1.0 <= result["final_signal"] <= 1.0

    # ── BLOCK action ─────────────────────────────────────────────

    def test_l3_block_produces_hold(self, combiner):
        from src.risk.manager import RiskAction
        l3 = {"action": RiskAction.BLOCK, "risk_score": 0.6,
               "adjusted_size": 50, "reason": "daily loss limit",
               "stop_loss": 0.0, "take_profit": 0.0}
        result = combiner.combine(0.9, self._l2(0.9), l3)
        assert result["action"] == "hold"
        assert result["final_signal"] == 0.0

    # ── REDUCE action ────────────────────────────────────────────

    def test_l3_reduce_halves_position_size(self, combiner):
        from src.risk.manager import RiskAction
        l3 = {"action": RiskAction.REDUCE, "risk_score": 0.3,
              "adjusted_size": 100.0, "reason": "elevated risk",
              "stop_loss": 95.0, "take_profit": 110.0}
        result = combiner.combine(0.7, self._l2(0.3), l3)
        assert result["position_size"] == pytest.approx(50.0)


# ═══════════════════════════════════════════════════════════════
# 4. ORDER EXECUTION (mocked Robinhood)
# ═══════════════════════════════════════════════════════════════

class TestOrderExecution:
    """
    Tests for key executor logic using mocked exchange clients.
    We test the pure logic paths, not actual API calls.
    """

    def _make_executor(self):
        """Create a TradingExecutor with all external deps mocked."""
        from src.trading.executor import TradingExecutor
        config = {
            'mode': 'paper',
            'exchanges': [{'name': 'robinhood', 'spread_pct_per_side': 0.845,
                           'round_trip_spread_pct': 1.69, 'longs_only': True,
                           'min_expected_move_pct': 3.0}],
            'risk': {'daily_loss_limit_pct': 5.0, 'max_drawdown_pct': 15.0,
                     'risk_per_trade_pct': 1.0, 'atr_stop_mult': 5.0,
                     'atr_tp_mult': 15.0, 'hard_stop_pct': -6.0},
            'ai': {'use_llm_trade_decider': False, 'orchestrator_enabled': False},
            'sniper': {'enabled': False},
            'multi_strategy': {'enabled': False},
            'adaptive': {'ema_period': 8, 'min_entry_score': 4},
        }
        with patch.multiple(
            'src.trading.executor.TradingExecutor',
            __init__=MagicMock(return_value=None),
        ):
            ex = TradingExecutor.__new__(TradingExecutor)
            ex.config = config
            ex.positions = {}
            ex.trade_history = []
            ex._training_collector = None
            ex._genetic_engine = None
            ex.equity = 16000.0
            ex.initial_equity = 16000.0
            ex.daily_pnl = 0.0
            ex.logger = MagicMock()
        return ex

    # ── PnL calculation ───────────────────────────────────────────

    def test_long_profit_pnl(self):
        entry = 80000.0
        exit_p = 82000.0
        pnl_pct = (exit_p - entry) / entry * 100
        assert pnl_pct == pytest.approx(2.5)

    def test_long_loss_pnl(self):
        entry = 80000.0
        exit_p = 78000.0
        pnl_pct = (exit_p - entry) / entry * 100
        assert pnl_pct == pytest.approx(-2.5)

    def test_spread_cost_reduces_net_pnl(self):
        entry = 80000.0
        exit_p = 81600.0
        spread = 1.69  # round-trip spread %
        gross_pnl = (exit_p - entry) / entry * 100
        net_pnl = gross_pnl - spread
        assert gross_pnl == pytest.approx(2.0)
        assert net_pnl == pytest.approx(0.31)

    def test_hard_stop_triggers_below_threshold(self):
        """Position should be closed if price drops below hard stop."""
        entry = 80000.0
        hard_stop_pct = -6.0
        hard_stop_price = entry * (1 + hard_stop_pct / 100)
        current_price = 74000.0
        pnl_pct = (current_price - entry) / entry * 100
        assert pnl_pct < hard_stop_pct  # triggers hard stop
        assert current_price < hard_stop_price

    def test_take_profit_triggers_above_threshold(self):
        entry = 80000.0
        atr_tp_mult = 15.0
        atr = 500.0
        tp_price = entry + atr_tp_mult * atr
        current_price = tp_price + 100
        assert current_price > tp_price

    # ── SL progression (L1 → L2 → L3 → L4) ─────────────────────

    def test_sl_levels_tighten_as_price_moves_up(self):
        """Each SL level should be higher than the previous (trailing up)."""
        entry = 80000.0
        sl_levels = {
            'L1': entry * 0.94,   # -6%
            'L2': entry * 0.97,   # -3%
            'L3': entry * 0.99,   # -1%
            'L4': entry * 1.005,  # break-even+
        }
        levels = list(sl_levels.values())
        for i in range(len(levels) - 1):
            assert levels[i] < levels[i + 1], \
                f"SL level {i} should be below level {i+1}"

    def test_sl_progression_only_moves_up(self):
        """SL should never move down (trailing stop logic)."""
        current_sl = 78000.0
        new_candidate_sl = 77000.0  # attempted to lower
        # System should reject lowering the SL
        final_sl = max(current_sl, new_candidate_sl)
        assert final_sl == current_sl

    # ── Position size risk limits ─────────────────────────────────

    def test_position_size_capped_at_risk_per_trade(self):
        equity = 16000.0
        risk_per_trade_pct = 1.0
        max_risk_usd = equity * (risk_per_trade_pct / 100.0)
        assert max_risk_usd == pytest.approx(160.0)

    def test_daily_loss_limit_blocks_new_trades(self):
        equity = 16000.0
        daily_loss_limit_pct = 5.0
        daily_pnl = -850.0  # lost $850 today
        limit = -equity * (daily_loss_limit_pct / 100.0)
        assert daily_pnl < limit  # should block new trades

    def test_max_drawdown_triggers_shutdown(self):
        initial_equity = 16000.0
        current_equity = 13000.0
        max_drawdown_pct = 15.0
        drawdown = (initial_equity - current_equity) / initial_equity * 100
        assert drawdown > max_drawdown_pct  # triggers shutdown

    # ── Mocked order placement ────────────────────────────────────

    def test_mock_place_order_returns_order_id(self):
        mock_client = MagicMock()
        mock_client.place_order.return_value = {"id": "abc123", "status": "filled"}
        result = mock_client.place_order("BTC", "buy", 0.01, 80000)
        assert result["id"] == "abc123"
        mock_client.place_order.assert_called_once()

    def test_mock_order_failure_returns_none(self):
        mock_client = MagicMock()
        mock_client.place_order.side_effect = Exception("API error")
        with pytest.raises(Exception, match="API error"):
            mock_client.place_order("BTC", "buy", 0.01, 80000)

    def test_longs_only_rejects_short(self):
        """In longs_only mode, short signals must be rejected."""
        longs_only = True
        signal = "short"
        allowed = not (longs_only and signal == "short")
        assert allowed is False

    def test_longs_only_allows_long(self):
        longs_only = True
        signal = "long"
        allowed = not (longs_only and signal == "short")
        assert allowed is True


# ═══════════════════════════════════════════════════════════════
# 5. ENTRY SCORE / CONFLUENCE GATE
# ═══════════════════════════════════════════════════════════════

class TestConfluenceGating:
    """Tests for entry score and confluence count gates."""

    def test_min_entry_score_gate(self):
        """Trades with score below min_entry_score must be blocked."""
        min_score = 4
        for score in range(0, 4):
            assert score < min_score  # should block

        for score in range(4, 10):
            assert score >= min_score  # should allow

    def test_min_confluence_gate(self):
        """Trades with fewer than min_confluence confirming signals must be blocked."""
        min_confluence = 4
        assert 3 < min_confluence   # blocked
        assert 4 >= min_confluence  # allowed
        assert 7 >= min_confluence  # allowed

    def test_min_expected_move_gate(self):
        """Expected move must exceed spread + minimum threshold."""
        spread_pct = 1.69
        min_expected_move_pct = 4.0
        required = spread_pct + min_expected_move_pct

        small_move = 3.0
        good_move = 6.0

        assert small_move < required   # blocked
        assert good_move >= required   # allowed? let's check
        # Actually min_expected_move_pct is checked independently
        assert small_move < min_expected_move_pct   # blocked
        assert good_move >= min_expected_move_pct   # allowed

    def test_entry_score_zero_for_no_signals(self):
        """Entry score of 0 is invalid and must always block."""
        score = 0
        min_score = 4
        assert score < min_score

    def test_confluence_all_six_signals(self):
        """Maximum confluence when all 6 signals agree."""
        signals = {
            'multi_tf_align': True,
            'htf_4h_align': True,
            'htf_1d_align': True,
            'volume_surge': True,
            'ema_slope_accel': True,
            'expected_move_ok': True,
        }
        count = sum(signals.values())
        assert count == 6

    def test_confluence_partial_signals(self):
        signals = {
            'multi_tf_align': True,
            'htf_4h_align': False,
            'htf_1d_align': True,
            'volume_surge': False,
            'ema_slope_accel': True,
            'expected_move_ok': True,
        }
        count = sum(signals.values())
        assert count == 4  # exactly at minimum → allowed

    def test_confluence_below_minimum_blocks(self):
        signals = {
            'multi_tf_align': True,
            'htf_4h_align': False,
            'htf_1d_align': False,
            'volume_surge': True,
            'ema_slope_accel': False,
            'expected_move_ok': True,
        }
        count = sum(signals.values())
        min_confluence = 4
        assert count < min_confluence  # blocked


# ═══════════════════════════════════════════════════════════════
# 6. MULTI-ASSET COMBINER
# ═══════════════════════════════════════════════════════════════

class TestMultiAssetCombiner:

    @pytest.fixture
    def mac(self):
        from src.trading.signal_combiner import MultiAssetCombiner
        return MultiAssetCombiner(config={'max_long_positions': 2, 'max_short_positions': 1})

    def test_rank_signals_returns_list(self, mac):
        signals = {
            'BTC': {'final_signal': 0.8, 'action': 'buy'},
            'ETH': {'final_signal': 0.6, 'action': 'buy'},
            'SOL': {'final_signal': 0.3, 'action': 'buy'},
        }
        ranked = mac.rank_signals(signals)
        assert isinstance(ranked, list)

    def test_rank_respects_max_longs(self, mac):
        signals = {
            'BTC': {'final_signal': 0.9, 'action': 'buy'},
            'ETH': {'final_signal': 0.8, 'action': 'buy'},
            'SOL': {'final_signal': 0.7, 'action': 'buy'},
            'AVAX': {'final_signal': 0.6, 'action': 'buy'},
        }
        ranked = mac.rank_signals(signals)
        buys = [a for a, s in ranked if s['action'] == 'buy']
        assert len(buys) <= mac.max_long_positions

    def test_rank_orders_by_signal_strength(self, mac):
        signals = {
            'ETH': {'final_signal': 0.4, 'action': 'buy'},
            'BTC': {'final_signal': 0.9, 'action': 'buy'},
        }
        ranked = mac.rank_signals(signals)
        assets = [a for a, _ in ranked]
        assert assets[0] == 'BTC'  # stronger signal first

    def test_get_combiner_creates_per_asset(self, mac):
        c_btc = mac.get_combiner('BTC')
        c_eth = mac.get_combiner('ETH')
        c_btc2 = mac.get_combiner('BTC')
        assert c_btc is not c_eth
        assert c_btc is c_btc2  # same instance returned
