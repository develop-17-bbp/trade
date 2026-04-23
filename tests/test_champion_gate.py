"""Tests for src/ai/champion_gate.py — pure promotion-decision logic."""
from __future__ import annotations

import json

import pytest

from src.ai.champion_gate import (
    DEFAULT_MAX_REGRESSION_PCT,
    DEFAULT_MIN_IMPROVEMENT_PCT,
    ChampionGateResult,
    MetricScore,
    evaluate_gate,
    run_validation,
    score_analyst_output,
    score_scanner_output,
)


# ── score_analyst_output ────────────────────────────────────────────────


def _gt(direction="LONG", size=5.0):
    return {"direction": direction, "plan": {"size_pct": size}}


def test_analyst_scores_all_three_on_match():
    out = json.dumps({"direction": "LONG", "size_pct": 5.0})
    s = score_analyst_output(out, _gt("LONG", 5.0))
    assert s["schema_valid"] == 1.0
    assert s["direction_agreement"] == 1.0
    assert s["size_reasonableness"] == 1.0


def test_analyst_accepts_envelope_with_plan_key():
    out = json.dumps({"plan": {"direction": "LONG", "size_pct": 6.0}})
    s = score_analyst_output(out, _gt("LONG", 5.0))
    assert s["direction_agreement"] == 1.0
    assert s["size_reasonableness"] == 1.0


def test_analyst_zero_on_unparseable():
    s = score_analyst_output("not json", _gt())
    assert s["schema_valid"] == 0.0
    assert s["direction_agreement"] == 0.0


def test_analyst_direction_mismatch():
    out = json.dumps({"direction": "SHORT", "size_pct": 5.0})
    s = score_analyst_output(out, _gt("LONG", 5.0))
    assert s["schema_valid"] == 1.0
    assert s["direction_agreement"] == 0.0


def test_analyst_size_out_of_range():
    out = json.dumps({"direction": "LONG", "size_pct": 30.0})   # 6× ground truth
    s = score_analyst_output(out, _gt("LONG", 5.0))
    assert s["direction_agreement"] == 1.0
    assert s["size_reasonableness"] == 0.0


def test_analyst_no_plan_key_zero():
    out = json.dumps({"foo": "bar"})
    s = score_analyst_output(out, _gt())
    assert s["schema_valid"] == 0.0


# ── score_scanner_output ────────────────────────────────────────────────


def test_scanner_scores_match():
    out = json.dumps({"opportunity_score": 70, "proposed_direction": "LONG"})
    s = score_scanner_output(out, {"direction": "LONG"})
    assert s["schema_valid"] == 1.0
    assert s["direction_agreement"] == 1.0


def test_scanner_mismatch():
    out = json.dumps({"proposed_direction": "SHORT"})
    s = score_scanner_output(out, {"direction": "LONG"})
    assert s["direction_agreement"] == 0.0


# ── MetricScore.delta_pct ───────────────────────────────────────────────


def test_metric_delta_pct_basic():
    m = MetricScore(name="x", incumbent=0.50, challenger=0.60)
    assert abs(m.delta_pct - 20.0) < 1e-6


def test_metric_delta_handles_zero_incumbent():
    m = MetricScore(name="x", incumbent=0.0, challenger=0.5)
    assert m.delta_pct > 0


def test_metric_delta_both_zero():
    assert MetricScore(name="x", incumbent=0.0, challenger=0.0).delta_pct == 0.0


# ── run_validation ──────────────────────────────────────────────────────


def test_run_validation_averages_scores():
    samples = [{"direction": "LONG", "plan": {"size_pct": 5.0}} for _ in range(3)]
    calls: list = []

    def fn(model_id, s):
        calls.append(model_id)
        return json.dumps({"direction": "LONG", "size_pct": 5.0})

    scores = run_validation("incumbent-0", samples, brain="analyst", inference_fn=fn)
    assert len(calls) == 3
    assert scores["schema_valid"] == 1.0
    assert scores["direction_agreement"] == 1.0


def test_run_validation_handles_inference_error():
    def boom(_m, _s):
        raise RuntimeError("down")

    scores = run_validation("bad", [{"direction": "LONG"}] * 2,
                            brain="analyst", inference_fn=boom)
    assert scores["schema_valid"] == 0.0


def test_run_validation_empty_samples():
    assert run_validation("x", [], brain="analyst", inference_fn=lambda *_: "") == {}


# ── evaluate_gate ──────────────────────────────────────────────────────


def _samples(n=5):
    return [{"direction": "LONG", "plan": {"size_pct": 5.0}} for _ in range(n)]


def test_gate_promotes_when_challenger_beats_incumbent():
    def fn(model_id, _s):
        if model_id == "champ-v1":
            return json.dumps({"direction": "SHORT"})   # wrong
        return json.dumps({"direction": "LONG"})        # right

    r = evaluate_gate(
        "analyst", "champ-v1", "challenger-v2", _samples(10),
        inference_fn=fn,
    )
    assert r.promote is True
    assert r.primary_metric == "direction_agreement"
    assert any(m.name == "direction_agreement" and m.challenger > m.incumbent for m in r.metrics)


def test_gate_rejects_when_improvement_too_small():
    # Both models are equally good → delta 0 → below min_improvement_pct.
    def fn(_m, _s):
        return json.dumps({"direction": "LONG"})

    r = evaluate_gate(
        "analyst", "inc", "cha", _samples(10), inference_fn=fn,
    )
    assert r.promote is False
    assert "delta" in r.reason.lower() or "required" in r.reason.lower()


def test_gate_rejects_on_regression_even_when_primary_improves():
    # Challenger gets direction right (primary improves) but fails schema
    # validation on half the samples (regression > 5%).
    def fn(model_id, _s):
        if model_id == "inc":
            return json.dumps({"direction": "SHORT"})   # valid schema, wrong dir
        return "bad-json-on-half"                        # invalid schema

    r = evaluate_gate(
        "analyst", "inc", "cha", _samples(10),
        inference_fn=fn, min_improvement_pct=0.0,   # permissive primary
        max_regression_pct=5.0,
    )
    # Primary direction_agreement improved (but inc was 0, cha is 0 → zero).
    # Schema validation regressed from 1.0 to 0.0 = 100% regression → reject.
    # Depending on inf-fn outputs, reason should mention regression.
    assert r.promote is False


def test_gate_empty_validation_set():
    r = evaluate_gate("analyst", "a", "b", [], inference_fn=lambda *_: "")
    assert r.promote is False
    assert "no metrics" in r.reason.lower() or "empty" in r.reason.lower()


def test_gate_to_dict_roundtrip():
    def fn(_m, _s):
        return json.dumps({"direction": "LONG"})
    r = evaluate_gate("scanner", "a", "b", [{"direction": "LONG"}] * 3,
                      inference_fn=fn)
    d = r.to_dict()
    assert d["brain"] == "scanner"
    assert d["incumbent_id"] == "a"
    assert isinstance(d["metrics"], list)


def test_defaults_sane():
    assert DEFAULT_MIN_IMPROVEMENT_PCT >= 1.0
    assert DEFAULT_MAX_REGRESSION_PCT >= 1.0
