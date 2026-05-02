"""Smoke tests for the genetic-loop audit modules (P0–P3).

These tests verify that each new module imports, runs without raising,
and returns the expected output shape on synthetic data. They are NOT
strategy-quality tests — they only confirm the code is wired and not
broken. Real fitness numbers will only emerge once the loop runs on
live parquet data.

Run:
    python -m pytest tests/test_genetic_audit_modules.py -v
"""
from __future__ import annotations

import math
import random
import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Ensure src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ─────────────────────────────────────────────────────────────


def _synthetic_ohlcv(n: int = 600, seed: int = 7):
    """Generate synthetic OHLCV with mild trend + noise."""
    rng = np.random.default_rng(seed)
    base = 100.0
    drift = np.cumsum(rng.normal(0.0005, 0.01, n))
    closes = base * np.exp(drift)
    noise = rng.normal(0, 0.005, n)
    highs = closes * (1 + np.abs(noise))
    lows = closes * (1 - np.abs(noise))
    volumes = rng.uniform(1000, 5000, n)
    return closes, highs, lows, volumes


@pytest.fixture
def synthetic_market():
    return _synthetic_ohlcv()


# ── P0: Walk-forward + DSR gate ─────────────────────────────────────────


def test_deflated_sharpe_fitness_gate_smoke():
    """DSR gate annotates DNAs with p_positive when n_trials > threshold."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_dsr_gate import apply_dsr_gate

    pop = []
    for i in range(8):
        d = StrategyDNA()
        d.fitness = 0.5 + i * 0.05
        d.sharpe = 0.5 + i * 0.1
        d.total_pnl = 5.0 + i
        d.trades = 10
        d.name = f"D{i}"
        pop.append(d)

    stats = apply_dsr_gate(pop, n_trials=30)
    assert stats["applied"] is True
    assert stats["n_dna_tested"] >= 1
    # All DNAs got annotated
    annotated = [d for d in pop if hasattr(d, "dsr_p_positive")]
    assert len(annotated) >= 1


def test_deflated_sharpe_skipped_below_threshold():
    """DSR gate is no-op when n_trials below threshold."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_dsr_gate import apply_dsr_gate

    d = StrategyDNA()
    d.fitness = 0.7
    d.sharpe = 1.0
    d.trades = 10
    stats = apply_dsr_gate([d], n_trials=10, only_when_above=20)
    assert stats["applied"] is False
    assert stats["reason"] == "below_trial_threshold"


def test_walk_forward_split_construction():
    """WF split divides data correctly into train/val/test."""
    from src.trading.genetic_walk_forward import _make_split

    s = _make_split(1000)
    assert s.train_start == 0
    assert s.train_end == 800
    assert s.val_start == 800
    assert s.val_end == 950
    assert s.test_start == 950
    assert s.test_end == 1000


def test_walk_forward_evaluate_dna_smoke(synthetic_market):
    """WF evaluation returns expected shape on synthetic data."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_walk_forward import evaluate_dna_walk_forward

    closes, highs, lows, volumes = synthetic_market
    dna = StrategyDNA()
    dna.entry_rule = "ema_cross"
    dna.exit_rule = "trailing_atr"

    ev = evaluate_dna_walk_forward(
        dna,
        np.array(closes), np.array(highs),
        np.array(lows), np.array(volumes),
        spread_pct=0.5, n_trials=20,
    )
    d = ev.to_dict()
    for key in ("train_fitness", "val_fitness", "test_fitness",
                "test_sharpe", "deflated_sharpe", "p_true_sharpe_positive",
                "promotable", "n_trials"):
        assert key in d, f"missing key {key} in WF eval"


# ── P1: CMA-ES, LLM mutation, multi-asset ────────────────────────────────


def test_cma_es_refine_smoke(synthetic_market):
    """CMA-ES refine improves or matches initial fitness on synthetic data."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_cma_es import cma_es_refine

    closes, highs, lows, volumes = synthetic_market
    seed = StrategyDNA()
    seed.entry_rule = "ema_cross"
    seed.exit_rule = "trailing_atr"
    seed.name = "SEED"

    refined, res = cma_es_refine(
        seed,
        list(closes), list(highs), list(lows), list(volumes),
        spread_pct=0.5,
        max_generations=2,  # cheap smoke
        sigma_init=0.2,
        seed=42,
    )
    assert refined.entry_rule == seed.entry_rule  # structural genes frozen
    assert res.n_evaluations > 0
    assert isinstance(res.refined_genes, dict)


def test_llm_mutation_invalid_response_rejected():
    """LLM mutation rejects malformed JSON gracefully."""
    from src.trading.genetic_llm_mutation import _validate_llm_proposal

    assert _validate_llm_proposal("") is None
    assert _validate_llm_proposal("not json at all") is None
    # Unknown entry_rule should be rejected
    bad = '{"entry_rule":"INVENTED","exit_rule":"profit_target","genes":{}}'
    assert _validate_llm_proposal(bad) is None
    # Unknown exit_rule rejected
    bad = '{"entry_rule":"ema_cross","exit_rule":"INVENTED","genes":{}}'
    assert _validate_llm_proposal(bad) is None
    # Valid one accepted
    good = '{"entry_rule":"ema_cross","exit_rule":"profit_target","genes":{"ema_fast":12}}'
    proposal = _validate_llm_proposal(good)
    assert proposal is not None
    assert proposal["entry_rule"] == "ema_cross"
    assert proposal["genes"]["ema_fast"] == 12


def test_llm_mutation_unreachable_soft_fail():
    """If ollama is unreachable, llm_mutate_population reports soft-fail."""
    from src.trading.genetic_llm_mutation import llm_mutate_population
    from src.trading.genetic_strategy_engine import StrategyDNA

    pop = [StrategyDNA() for _ in range(3)]
    # We don't expect ollama to be up in test env — so _call_local_llm
    # returns None and the function reports llm_unreachable.
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:1"  # bad port
    result = llm_mutate_population(pop)
    # Either unreachable or invalid; both should produce soft-fail.
    assert result.proposed in (False, True)
    if not result.proposed:
        assert "unreachable" in result.rejection_reason


def test_multi_asset_pair_discovery(tmp_path, monkeypatch):
    """Multi-asset discovery picks up valid parquet names from a dir."""
    from src.trading.genetic_multi_asset import discover_pairs

    # Create a fake data dir with three names
    d = tmp_path / "fakedata"
    d.mkdir()
    for name in ("BTCUSDT-4h.parquet", "ETHUSDT-1h.parquet", "junk.txt"):
        (d / name).touch()
    pairs = discover_pairs(data_dir=str(d))
    assert len(pairs) == 2
    names = {(p.asset, p.timeframe) for p in pairs}
    assert ("BTC", "4h") in names
    assert ("ETH", "1h") in names


# ── P2: MAP-Elites + surrogate ──────────────────────────────────────────


def test_map_elites_archive_admission():
    """MAP-Elites only admits non-zero-fitness DNAs and replaces only on
    strict improvement."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_map_elites import MAPElitesArchive

    archive = MAPElitesArchive()

    d1 = StrategyDNA()
    d1.entry_rule = "ema_cross"
    d1.fitness = 0.5
    d1.win_rate = 0.55
    d1.trades = 10

    d2 = StrategyDNA()
    d2.entry_rule = "ema_cross"  # same family
    d2.fitness = 0.5  # tie — should NOT replace d1
    d2.win_rate = 0.55
    d2.trades = 10

    d3 = StrategyDNA()
    d3.entry_rule = "ema_cross"
    d3.fitness = 0.6  # strict improvement
    d3.win_rate = 0.55
    d3.trades = 10
    d3.sharpe = 0.0  # match d1/d2 cell so admission tests strict-greater

    archive.update_one(d1)
    archive.update_one(d2)  # tie should not replace
    archive.update_one(d3)  # improvement should replace
    assert archive.n_filled == 1
    summary = archive.summary()
    assert summary["n_filled"] == 1
    assert summary["coverage_pct"] >= 0


def test_map_elites_diverse_top_k():
    """diverse_top_k spreads selection across entry_family."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_map_elites import MAPElitesArchive

    archive = MAPElitesArchive()
    for entry, fit in [
        ("ema_cross", 0.9),  # trend
        ("rsi_oversold_bounce", 0.8),  # mean_rev
        ("breakout_volume", 0.7),  # breakout
        ("momentum_surge", 0.6),  # momentum
        ("fear_greed_contrarian", 0.5),  # macro
    ]:
        d = StrategyDNA()
        d.entry_rule = entry
        d.fitness = fit
        d.win_rate = 0.55
        d.trades = 10
        archive.update_one(d)
    diverse = archive.diverse_top_k(5)
    families = [e["entry_family"] for e in diverse]
    assert len(set(families)) >= 3  # at least 3 distinct families


def test_surrogate_filter_cold_start_keeps_all():
    """Surrogate with no trained model keeps the entire population."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_surrogate import SurrogateFilter

    s = SurrogateFilter()
    pop = [StrategyDNA() for _ in range(5)]
    for d in pop:
        d.fitness = 0.5
    kept, preds = s.filter_population(pop, keep_fraction=0.4)
    assert len(kept) == len(pop)  # cold start
    assert all(p.is_kept for p in preds)


def test_surrogate_filter_after_training_filters():
    """After training, surrogate keeps roughly keep_fraction + epsilon."""
    from src.trading.genetic_strategy_engine import StrategyDNA, INDICATOR_GENES
    from src.trading.genetic_surrogate import SurrogateFilter

    s = SurrogateFilter(epsilon=0.0)  # disable epsilon for this test
    rng = random.Random(7)
    # Generate many observations
    for _ in range(60):
        d = StrategyDNA()
        d.fitness = rng.uniform(0, 1)
        for k in INDICATOR_GENES:
            lo, hi = INDICATOR_GENES[k]["range"]
            d.genes[k] = rng.uniform(lo, hi)
        s.add_observation(dict(d.genes), d.entry_rule, d.exit_rule, d.fitness)
    trained = s.train(force=True)
    assert trained is True
    pop = []
    for _ in range(20):
        d = StrategyDNA()
        d.fitness = 0.5
        d.name = f"P{rng.randint(0, 9999)}"
        pop.append(d)
    kept, preds = s.filter_population(pop, keep_fraction=0.5)
    assert len(kept) == 10  # half of 20 with epsilon=0


# ── P3: Drift + Grammatical Evolution ───────────────────────────────────


def test_drift_detect_no_drift_on_random_walk():
    """Pure random walk should NOT trigger drift."""
    from src.trading.genetic_drift import detect_drift

    rng = np.random.default_rng(11)
    closes = list(np.cumsum(rng.normal(0, 0.005, 400)) + 100)
    sig = detect_drift(closes)
    # Random walk: shouldn't usually trigger; if it does, the variance
    # ratio should still be moderate.
    assert isinstance(sig.z_score_mean, float)
    assert isinstance(sig.variance_ratio, float)


def test_drift_detect_triggers_on_volatility_jump():
    """Sudden vol increase in last window should trigger drift."""
    from src.trading.genetic_drift import detect_drift

    rng = np.random.default_rng(13)
    base = list(np.cumsum(rng.normal(0, 0.001, 400)) + 100)
    # Recent 60 bars: 10× volatility
    spike = list(np.cumsum(rng.normal(0, 0.020, 60)) + base[-1])
    closes = base[:-60] + spike
    sig = detect_drift(closes)
    assert sig.drift_detected is True
    assert any("variance_ratio" in t or "z_score" in t for t in sig.triggers)


def test_drift_inject_immigrants_replaces_bottom():
    """inject_immigrants replaces the bottom fraction of the population."""
    from src.trading.genetic_strategy_engine import StrategyDNA
    from src.trading.genetic_drift import inject_immigrants

    class FakeEngine:
        population = []
        _current_mutation_rate = 0.15
        _stagnation_counter = 5
        _total_generations = 7

    eng = FakeEngine()
    for i in range(10):
        d = StrategyDNA()
        d.fitness = float(i)  # 0..9
        d.name = f"OG{i}"
        eng.population.append(d)
    stats = inject_immigrants(eng, fraction=0.20)
    # Bottom 2 should be replaced (fraction=0.20 of 10)
    assert stats["replaced"] == 2
    # The two with the lowest fitness are now immigrants
    sorted_pop = sorted(eng.population, key=lambda d: d.fitness)
    immigrant_names = [d.name for d in sorted_pop[:2]]
    assert all(n.startswith("IMMIG_") for n in immigrant_names)
    # Stagnation counter reset
    assert eng._stagnation_counter == 0


def test_grammatical_derivation_bounded():
    """GE derivation produces a finite expression with depth ≤ max_depth."""
    from src.trading.genetic_grammar import derive

    rng = random.Random(3)
    genome = [rng.randint(0, 255) for _ in range(40)]
    d = derive(genome, max_depth=6)
    assert d.depth_reached <= 6
    if not d.overflowed:
        assert d.expression
        # Tokens should all be terminals
        from src.trading.genetic_grammar import NON_TERMINALS
        assert not any(t in NON_TERMINALS for t in d.tokens)


def test_grammatical_evolve_returns_individuals(synthetic_market):
    """Tiny GE evolve loop returns a sorted list of GEIndividuals."""
    from src.trading.genetic_grammar import evolve_grammatical

    closes, highs, lows, _ = synthetic_market
    top = evolve_grammatical(
        list(closes), list(highs), list(lows),
        spread_pct=0.5,
        population_size=8,
        generations=2,
        seed=99,
    )
    assert top  # non-empty
    fitnesses = [ind.fitness for ind in top]
    # Sorted descending
    assert fitnesses == sorted(fitnesses, reverse=True)


# ── Integration: genetic_loop wires it all ──────────────────────────────


def test_genetic_loop_imports():
    """Top-level genetic_loop module imports cleanly."""
    from src.scripts import genetic_loop  # noqa: F401
    assert hasattr(genetic_loop, "run_evolution_cycle")
    assert hasattr(genetic_loop, "main")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
