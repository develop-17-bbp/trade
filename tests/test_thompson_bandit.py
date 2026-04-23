"""Tests for src/learning/thompson_bandit.py — Thompson sampling over strategies."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import pytest

from src.learning.thompson_bandit import (
    EMERGENCY_EXPLOIT_BIAS,
    BanditDraw,
    sample_from_records,
    top_k_by_posterior_mean,
)


@dataclass
class _FakeRec:
    strategy_id: str
    live_wins: int = 0
    live_losses: int = 0


# ── Pure sampling ───────────────────────────────────────────────────────


def test_sample_empty_returns_none():
    assert sample_from_records([]) is None


def test_sample_single_strategy_returns_it():
    rng = random.Random(0)
    draw = sample_from_records([_FakeRec("a")], rng=rng)
    assert draw is not None
    assert draw.strategy_id == "a"


def test_sample_favours_winning_strategy_over_long_run():
    # Posteriors with meaningful overlap so exploration still fires.
    winner = _FakeRec("good", live_wins=12, live_losses=8)   # ~60% WR, narrow CI
    loser = _FakeRec("bad", live_wins=8, live_losses=12)     # ~40% WR
    rng = random.Random(42)
    counts = {"good": 0, "bad": 0}
    for _ in range(500):
        draw = sample_from_records([winner, loser], rng=rng)
        counts[draw.strategy_id] += 1
    # Winner should dominate but exploration should still pick loser sometimes.
    assert counts["good"] > counts["bad"]
    assert counts["bad"] > 0


def test_sample_equal_priors_roughly_equal_split():
    rng = random.Random(42)
    a = _FakeRec("a")
    b = _FakeRec("b")
    counts = {"a": 0, "b": 0}
    for _ in range(500):
        counts[sample_from_records([a, b], rng=rng).strategy_id] += 1
    # Both should see many draws; nothing should hit 90%+ with uniform priors.
    assert 150 < counts["a"] < 350
    assert 150 < counts["b"] < 350


def test_emergency_mode_sharpens_ranking():
    # Under emergency, stronger strategy should win even more of the time.
    # Use small overlapping posteriors so there's room for sharpening to show.
    winner = _FakeRec("good", live_wins=8, live_losses=6)    # ~57% WR
    loser = _FakeRec("bad", live_wins=6, live_losses=8)      # ~43% WR

    def _split(emergency: bool) -> int:
        rng = random.Random(7)
        wins = 0
        for _ in range(1000):
            draw = sample_from_records([winner, loser], rng=rng, emergency_mode=emergency)
            if draw.strategy_id == "good":
                wins += 1
        return wins

    base = _split(False)
    emerg = _split(True)
    # Emergency sharpens — winner's share should be higher (or at least not lower).
    assert emerg >= base
    # Sanity: winner leads in both modes.
    assert base > 500 and emerg > 500


def test_draw_exposes_posterior_stats():
    rng = random.Random(0)
    draw = sample_from_records([_FakeRec("x", live_wins=10, live_losses=5)], rng=rng)
    d = draw.to_dict()
    assert d["strategy_id"] == "x"
    # α = 1 + 10 = 11, β = 1 + 5 = 6, mean = 11/17
    assert abs(draw.alpha - 11.0) < 1e-9
    assert abs(draw.beta - 6.0) < 1e-9
    assert abs(draw.posterior_mean - 11 / 17) < 1e-9


def test_records_without_strategy_id_are_ignored():
    rng = random.Random(0)

    class _NoId:
        live_wins = 10
        live_losses = 0
    records = [_NoId(), _FakeRec("keeper")]
    draw = sample_from_records(records, rng=rng)
    assert draw is not None and draw.strategy_id == "keeper"


# ── Deterministic top-k ─────────────────────────────────────────────────


def test_top_k_ranks_by_posterior_mean():
    recs = [
        _FakeRec("a", live_wins=10, live_losses=90),  # mean ~0.11
        _FakeRec("b", live_wins=80, live_losses=20),  # mean ~0.79
        _FakeRec("c", live_wins=50, live_losses=50),  # mean ~0.50
    ]
    top = top_k_by_posterior_mean(recs, k=2)
    assert [d.strategy_id for d in top] == ["b", "c"]


def test_top_k_clamps_to_at_least_one():
    recs = [_FakeRec("only", live_wins=1, live_losses=0)]
    top = top_k_by_posterior_mean(recs, k=0)
    assert len(top) == 1


# ── Integration-ish sample_from_repo ────────────────────────────────────


def test_sample_from_repo_uses_real_repository(tmp_path, monkeypatch):
    from src.trading.strategy_repository import StrategyRepository
    import src.trading.strategy_repository as sr_mod

    db = tmp_path / "bandit.sqlite"
    repo = StrategyRepository(str(db))
    a = repo.register(dna={}, regime_tag="any")
    b = repo.register(dna={}, regime_tag="any")
    # Give A a strong win record; B is mediocre.
    for _ in range(20):
        repo.record_outcome(a, 1.0)
    for i in range(20):
        repo.record_outcome(b, 1.0 if i < 2 else -1.0)

    monkeypatch.setattr(sr_mod, "_repo_singleton", repo, raising=False)
    from src.learning.thompson_bandit import sample_from_repo
    rng = random.Random(0)
    counts = {a: 0, b: 0}
    for _ in range(50):
        draw = sample_from_repo(rng=rng, emergency_mode=False)
        if draw is None:
            continue
        counts[draw.strategy_id] += 1
    # Strong winner should draw the vast majority.
    assert counts[a] > counts[b] * 3


def test_sample_from_repo_returns_none_when_empty(tmp_path, monkeypatch):
    from src.trading.strategy_repository import StrategyRepository
    import src.trading.strategy_repository as sr_mod

    db = tmp_path / "empty.sqlite"
    repo = StrategyRepository(str(db))
    monkeypatch.setattr(sr_mod, "_repo_singleton", repo, raising=False)
    from src.learning.thompson_bandit import sample_from_repo
    assert sample_from_repo() is None


# ── Constants sanity ────────────────────────────────────────────────────


def test_emergency_exploit_bias_sane():
    assert EMERGENCY_EXPLOIT_BIAS >= 1.0
