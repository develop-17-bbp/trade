"""Tests for src/trading/strategy_repository.py — versioned strategy store."""
from __future__ import annotations

import pytest

from src.trading.strategy_repository import (
    VALID_STATUSES,
    StrategyRecord,
    StrategyRepository,
)


@pytest.fixture
def repo(tmp_path):
    db = tmp_path / "repo.sqlite"
    r = StrategyRepository(str(db))
    yield r
    r.close()


# ── Registration ────────────────────────────────────────────────────────


def test_register_returns_id_and_persists(repo):
    sid = repo.register(dna={"ema_fast": 8, "ema_slow": 21}, name="EMA-8-21", regime_tag="trending")
    assert isinstance(sid, str) and len(sid) > 0
    rec = repo.get(sid)
    assert rec is not None
    assert rec.status == "candidate"
    assert rec.regime_tag == "trending"
    assert rec.dna["ema_fast"] == 8


def test_register_with_explicit_id(repo):
    repo.register(dna={}, strategy_id="mine-001")
    assert repo.get("mine-001") is not None


def test_get_missing_returns_none(repo):
    assert repo.get("does-not-exist") is None


# ── Status FSM ──────────────────────────────────────────────────────────


def test_set_status_valid(repo):
    sid = repo.register(dna={})
    for status in ("challenger", "champion", "retired", "quarantine", "candidate"):
        repo.set_status(sid, status)
        assert repo.get(sid).status == status


def test_set_status_rejects_unknown(repo):
    sid = repo.register(dna={})
    with pytest.raises(ValueError):
        repo.set_status(sid, "bogus")


def test_promote_retires_previous_champion_same_regime(repo):
    old = repo.register(dna={"x": 1}, regime_tag="trending")
    repo.promote(old)
    new = repo.register(dna={"x": 2}, regime_tag="trending")
    repo.promote(new)
    assert repo.get(old).status == "retired"
    assert repo.get(new).status == "champion"


def test_promote_does_not_retire_other_regime_champion(repo):
    a = repo.register(dna={}, regime_tag="trending")
    b = repo.register(dna={}, regime_tag="choppy")
    repo.promote(a)
    repo.promote(b)
    assert repo.get(a).status == "champion"
    assert repo.get(b).status == "champion"


def test_quarantine_records_reason(repo):
    sid = repo.register(dna={})
    repo.quarantine(sid, reason="z-score breach x5")
    rec = repo.get(sid)
    assert rec.status == "quarantine"
    assert rec.backtest_summary.get("quarantine_reason") == "z-score breach x5"
    assert rec.backtest_summary.get("quarantine_ts") is not None


# ── Outcome updates ─────────────────────────────────────────────────────


def test_record_outcome_increments_counters(repo):
    sid = repo.register(dna={})
    repo.record_outcome(sid, 1.5)
    repo.record_outcome(sid, -0.8)
    repo.record_outcome(sid, 2.0)
    rec = repo.get(sid)
    assert rec.live_trades == 3
    assert rec.live_wins == 2
    assert rec.live_losses == 1
    assert abs(rec.live_wr - 2 / 3) < 1e-6


def test_record_outcome_computes_per_trade_sharpe(repo):
    sid = repo.register(dna={})
    # Deterministic sequence: mean 0.5, std 0.5, sharpe 1.0
    for pnl in (1.0, 0.0, 1.0, 0.0):
        repo.record_outcome(sid, pnl)
    rec = repo.get(sid)
    assert abs(rec.live_sharpe - 1.0) < 1e-6


def test_record_outcome_on_unknown_id_is_noop(repo):
    repo.record_outcome("no-such-id", 1.0)  # must not raise


# ── Search ──────────────────────────────────────────────────────────────


def test_search_by_status(repo):
    a = repo.register(dna={})
    b = repo.register(dna={})
    c = repo.register(dna={})
    repo.set_status(a, "champion")
    repo.set_status(b, "challenger")
    assert {r.strategy_id for r in repo.search(status="champion")} == {a}
    assert {r.strategy_id for r in repo.search(status="challenger")} == {b}
    assert {r.strategy_id for r in repo.search(status="candidate")} == {c}


def test_search_by_regime_matches_any(repo):
    a = repo.register(dna={}, regime_tag="trending")
    b = repo.register(dna={}, regime_tag="any")
    c = repo.register(dna={}, regime_tag="choppy")
    ids = {r.strategy_id for r in repo.search(regime="trending")}
    assert a in ids and b in ids and c not in ids


def test_search_by_min_sharpe(repo):
    a = repo.register(dna={})
    b = repo.register(dna={})
    # a -> sharpe ~ 1.0 (mean 0.5 / std 0.5), b -> sharpe ~ 0 (constant)
    for pnl in (1.0, 0.0, 1.0, 0.0):
        repo.record_outcome(a, pnl)
    for pnl in (0.0, 0.0, 0.0, 0.0):
        repo.record_outcome(b, pnl)
    ids = {r.strategy_id for r in repo.search(min_sharpe=0.5)}
    assert a in ids and b not in ids


def test_current_champion_returns_latest(repo):
    old = repo.register(dna={}, regime_tag="trending")
    new = repo.register(dna={}, regime_tag="trending")
    repo.promote(old)
    repo.promote(new)
    champ = repo.current_champion(regime="trending")
    assert champ is not None and champ.strategy_id == new


def test_count_by_status(repo):
    for _ in range(3):
        repo.register(dna={})
    x = repo.register(dna={})
    repo.set_status(x, "champion")
    counts = repo.count_by_status()
    assert counts.get("candidate") == 3
    assert counts.get("champion") == 1


# ── A/B tests ───────────────────────────────────────────────────────────


def test_ab_test_lifecycle(repo):
    champ = repo.register(dna={})
    chal = repo.register(dna={})
    repo.promote(champ)
    repo.set_status(chal, "challenger")

    ab = repo.start_ab_test(chal, champ)
    active = repo.active_ab_tests()
    assert len(active) == 1
    assert active[0]["ab_id"] == ab

    repo.finish_ab_test(ab, "challenger_promoted", {"sharpe_delta": 0.4})
    assert repo.active_ab_tests() == []


# ── Sanity on constants ─────────────────────────────────────────────────


def test_valid_statuses_has_full_fsm():
    assert {"candidate", "challenger", "champion", "quarantine", "retired"} <= set(VALID_STATUSES)


def test_record_to_dict_roundtrip(repo):
    sid = repo.register(dna={"k": 1}, name="n", regime_tag="r")
    rec = repo.get(sid)
    d = rec.to_dict()
    assert d["strategy_id"] == sid
    assert d["name"] == "n"
    assert d["regime_tag"] == "r"
    assert d["dna"] == {"k": 1}
