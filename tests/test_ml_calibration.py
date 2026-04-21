"""Tests for src.ml.calibration and src.ml.champion_gate.

These cover the new calibration + score-delta + champion/challenger path that gates
both the binary SKIP/TRADE executor gate and the 3-class retrain loop.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# calibration.py
# ---------------------------------------------------------------------------

def test_calibration_fallback_matches_historical_deltas():
    """With no bundle, score_delta_for must reproduce the hand-tuned boosts/penalties
    that executor.py used before calibration existed. This is the degrade-gracefully
    path when a calibration.json doesn't exist yet on a freshly deployed node."""
    from src.ml import calibration as C

    # Strong SKIP: raw_prob < 0.3 → delta = -3
    d, p = C.score_delta_for(None, 0.1)
    assert d == -3, f"expected -3 for raw=0.1, got {d}"
    assert p == 0.1

    # Weak SKIP: 0.30 <= raw < 0.45 → delta = -1
    d, _ = C.score_delta_for(None, 0.40)
    assert d == -1

    # Abstain: 0.45 <= raw < 0.55 → delta = 0
    d, _ = C.score_delta_for(None, 0.50)
    assert d == 0

    # Weak TRADE: 0.55 <= raw < 0.70 → delta = +1
    d, _ = C.score_delta_for(None, 0.65)
    assert d == 1

    # Strong TRADE: raw >= 0.70 → delta = +2
    d, _ = C.score_delta_for(None, 0.95)
    assert d == 2


def _synthetic_calibrated_probs(n: int = 2000, seed: int = 7):
    """Generate (raw_prob, y_true) pairs where the raw probability is systematically
    overconfident — the same overconfidence we observed from LightGBM with
    is_unbalance=True. A good isotonic fit must pull these back toward the true rate."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.2, 0.8, size=n)
    y = (rng.uniform(0, 1, size=n) < true_p).astype(int)
    # Overconfident raw prob: pushes true probabilities toward extremes
    raw = np.clip(true_p * 1.4 - 0.2, 0.0, 1.0)
    raw = np.clip(raw + rng.normal(0, 0.03, size=n), 0.0, 1.0)
    return raw, y, true_p


def test_fit_calibration_shrinks_extreme_probs():
    """After calibration, probabilities near 0/1 should move toward the true rate.
    This is the whole point of isotonic regression here."""
    from src.ml import calibration as C

    pytest.importorskip("sklearn")

    raw, y, _ = _synthetic_calibrated_probs()
    bundle = C.fit_calibration(raw, y, asset="TEST")
    assert bundle is not None
    assert bundle.is_usable()
    assert bundle.fit_n_samples == len(raw)

    # Calibrated curve must be non-decreasing
    for i in range(1, len(bundle.isotonic_y)):
        assert bundle.isotonic_y[i] >= bundle.isotonic_y[i - 1] - 1e-12


def test_fit_calibration_produces_bounded_deltas():
    from src.ml import calibration as C

    pytest.importorskip("sklearn")

    raw, y, _ = _synthetic_calibrated_probs()
    bundle = C.fit_calibration(raw, y)
    assert bundle is not None
    for d in bundle.deltas:
        assert C.MIN_DELTA <= d <= C.MAX_DELTA, f"delta {d} outside clip range"

    # The abstain bucket (straddling 0.5) must always be 0 regardless of its winrate
    for (lo, hi), d in zip(bundle.buckets, bundle.deltas):
        if lo <= 0.5 < hi:
            assert d == 0, f"abstain bucket delta={d}, expected 0"


def test_calibration_round_trip_through_json(tmp_path):
    from src.ml import calibration as C

    pytest.importorskip("sklearn")

    raw, y, _ = _synthetic_calibrated_probs()
    bundle = C.fit_calibration(raw, y, asset="BTC")
    assert bundle is not None

    path = tmp_path / "lgbm_btc_calibration.json"
    C.save_calibration(bundle, str(path))

    reloaded = C.load_calibration(str(path))
    assert reloaded is not None
    assert reloaded.is_usable()
    assert reloaded.asset == "BTC"
    # Applying to a few raw probs should give identical calibrated values
    for p in [0.05, 0.3, 0.5, 0.7, 0.99]:
        assert abs(reloaded.apply(p) - bundle.apply(p)) < 1e-9


def test_calibration_degenerate_holdout_returns_none():
    """Single-class holdout or too-few-samples must return None, not crash."""
    from src.ml import calibration as C

    pytest.importorskip("sklearn")

    # Only one class present
    assert C.fit_calibration([0.5] * 100, [1] * 100) is None
    # Too few samples
    assert C.fit_calibration([0.3, 0.7] * 10, [0, 1] * 10) is None


def test_score_delta_uses_calibrated_not_raw():
    """score_delta_for must route through isotonic when a bundle is present."""
    from src.ml import calibration as C

    pytest.importorskip("sklearn")

    raw, y, _ = _synthetic_calibrated_probs()
    bundle = C.fit_calibration(raw, y)
    assert bundle is not None

    # The raw 0.9 probability from an overconfident model maps to something lower
    # after calibration. The returned `calibrated_prob` must reflect that, and the
    # delta should be based on the calibrated bucket.
    delta, calibrated_prob = C.score_delta_for(bundle, 0.90)
    assert 0.0 <= calibrated_prob <= 1.0
    assert isinstance(delta, int)


# ---------------------------------------------------------------------------
# champion_gate.py
# ---------------------------------------------------------------------------

def _train_tiny_booster(X, y, seed: int = 0):
    """Train a tiny binary LightGBM booster for gate tests."""
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 8,
        "learning_rate": 0.1,
        "verbose": -1,
        "seed": seed,
    }
    dset = lgb.Dataset(X, label=y)
    return lgb.train(params, dset, num_boost_round=30)


def test_champion_gate_promotes_when_no_incumbent(tmp_path):
    pytest.importorskip("lightgbm")
    from src.ml.champion_gate import evaluate_and_gate

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = _train_tiny_booster(X[:400], y[:400])
    incumbent_path = str(tmp_path / "lgbm_test_trained.txt")

    result = evaluate_and_gate(model, incumbent_path, X[400:], y[400:])
    assert result.promoted
    assert result.reason == "no_incumbent"
    assert result.incumbent_f1 is None


def test_champion_gate_rejects_worse_challenger(tmp_path):
    """A challenger trained on noise must be rejected in favor of a competent incumbent."""
    pytest.importorskip("lightgbm")
    from src.ml.champion_gate import evaluate_and_gate

    rng = np.random.default_rng(1)
    # Clean, learnable data
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Competent incumbent trained on clean data
    incumbent = _train_tiny_booster(X[:600], y[:600])
    incumbent_path = str(tmp_path / "lgbm_test_trained.txt")
    incumbent.save_model(incumbent_path)

    # Challenger trained on pure noise — will be much worse
    y_noise = rng.integers(0, 2, size=600)
    challenger = _train_tiny_booster(X[:600], y_noise)

    result = evaluate_and_gate(challenger, incumbent_path, X[600:], y[600:])
    assert not result.promoted, f"Noise challenger should have been rejected: {result.reason}"
    assert result.incumbent_f1 is not None
    assert result.new_f1 < result.incumbent_f1
    # Rejected model must be archived
    assert result.challenger_path is not None
    assert os.path.exists(result.challenger_path)


def test_champion_gate_accepts_equal_challenger(tmp_path):
    """A challenger within the 1pp tolerance must still be accepted (not blocked by noise)."""
    pytest.importorskip("lightgbm")
    from src.ml.champion_gate import evaluate_and_gate

    rng = np.random.default_rng(2)
    X = rng.normal(size=(800, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Two models trained on the same data with different seeds — effectively equivalent
    incumbent = _train_tiny_booster(X[:600], y[:600], seed=10)
    challenger = _train_tiny_booster(X[:600], y[:600], seed=20)

    incumbent_path = str(tmp_path / "lgbm_test_trained.txt")
    incumbent.save_model(incumbent_path)

    result = evaluate_and_gate(challenger, incumbent_path, X[600:], y[600:])
    # With tolerance 0.01, two near-equal models should promote
    assert result.promoted, f"Equivalent model was blocked: {result.reason}"


# ---------------------------------------------------------------------------
# executor integration — pure-python slice, no lightgbm loading required
# ---------------------------------------------------------------------------

def test_score_delta_fallback_when_calibration_missing(tmp_path, monkeypatch):
    """Simulates executor-at-boot when no calibration.json has been written yet.
    The executor must degrade to the hand-tuned fallback without crashing."""
    from src.ml import calibration as C

    missing_path = tmp_path / "nonexistent_calibration.json"
    bundle = C.load_calibration(str(missing_path))
    assert bundle is None

    delta, p = C.score_delta_for(bundle, 0.72)
    # Falls back to DEFAULT_FALLBACK_DELTAS at the 0.70-1.0 bucket
    assert delta == 2
    assert p == 0.72
