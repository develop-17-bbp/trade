"""Phase 0 — Decision envelope + decision_id propagation tests.

Three assertions:
  (a) every JSONL row post-deploy has a ULID decision_id
  (b) the ULID is unique across 100 consecutive decisions
  (c) the Pydantic model round-trips without loss

Synthetic IDs (prefix `synth-`) are excluded from the uniqueness test —
they come from error-path audit callers that don't sit inside a real
decision cycle. Phase 0 inspection confirmed the only current audit
call site (executor.py:1509) always passes a real decision_id.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pytest


ULID_PATTERN = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")  # Crockford Base32


class TestPhase0Envelope:
    """Phase 0 decision envelope + audit-log propagation."""

    # ── (a) + (b): ULID presence and uniqueness ──

    def test_new_decision_id_is_valid_ulid(self):
        from src.orchestration import new_decision_id
        for _ in range(10):
            did = new_decision_id()
            assert isinstance(did, str)
            assert ULID_PATTERN.match(did), f"Not a Crockford-Base32 ULID: {did}"

    def test_decision_ids_unique_across_100_draws(self):
        from src.orchestration import new_decision_id
        ids = [new_decision_id() for _ in range(100)]
        assert len(set(ids)) == 100, "ULIDs collided — RNG is broken"

    def test_decision_ids_are_time_sortable(self):
        """ULID is time-ordered — later IDs must sort after earlier ones."""
        import time
        from src.orchestration import new_decision_id
        first = new_decision_id()
        time.sleep(0.005)
        second = new_decision_id()
        assert first < second, f"ULIDs not monotonic: {first} >= {second}"

    # ── (c): Pydantic round-trip ──

    def test_decision_model_roundtrip(self):
        from src.orchestration.envelope import Decision, AgentVoteRecord
        d = Decision(
            symbol="BTC",
            l1_signal=1,
            l1_confidence=0.72,
            l4_fused_direction=1,
            l4_fused_confidence=0.65,
            agent_votes={
                "authority_compliance": AgentVoteRecord(
                    direction=1, confidence=0.9, veto=False, reasoning="COMPLIANT"
                ),
            },
            final_action="LONG",
            final_confidence=0.65,
            final_position_scale=0.5,
        )
        payload = d.model_dump(mode="json")
        back = Decision(**payload)
        assert back.decision_id == d.decision_id
        assert back.trace_id == d.trace_id == d.decision_id  # Phase 0: mirrored
        assert back.agent_votes["authority_compliance"].direction == 1
        assert back.final_action == "LONG"

    def test_trace_id_defaults_to_decision_id(self):
        from src.orchestration.envelope import Decision
        d = Decision(symbol="BTC")
        assert d.trace_id == d.decision_id

    def test_trace_id_explicit_override_preserved(self):
        from src.orchestration.envelope import Decision
        d = Decision(symbol="BTC", trace_id="explicit-trace-xyz")
        assert d.trace_id == "explicit-trace-xyz"
        assert d.decision_id != "explicit-trace-xyz"

    # ── Validators ──

    def test_direction_must_be_minus1_zero_one(self):
        from src.orchestration.envelope import Decision
        with pytest.raises(Exception):
            Decision(l1_signal=2)
        with pytest.raises(Exception):
            Decision(l4_fused_direction=-2)

    def test_sl_price_positive_for_longs(self):
        from src.orchestration.envelope import Decision
        with pytest.raises(Exception):
            Decision(final_action="LONG", sl_price=-1.0)
        # SL=None is fine for LONG
        Decision(final_action="LONG", sl_price=None)

    def test_agent_vote_direction_validator(self):
        from src.orchestration.envelope import AgentVoteRecord
        with pytest.raises(Exception):
            AgentVoteRecord(direction=7)
        with pytest.raises(Exception):
            AgentVoteRecord(confidence=1.5)

    # ── JSONL uniqueness across 100 synthetic audit rows ──

    def test_jsonl_audit_rows_unique_decision_ids(self, tmp_path):
        """Simulate 100 audit rows; every non-synthetic decision_id must be unique."""
        from src.orchestration import new_decision_id
        log_path = tmp_path / "trade_decisions.jsonl"
        for i in range(100):
            row = {
                "decision_id": new_decision_id(),
                "trace_id": None,
                "asset": "BTC",
                "decision": {"direction": 0, "confidence": 0.0, "veto": False},
            }
            row["trace_id"] = row["decision_id"]
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        rows: List[dict] = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        real_ids = [
            r["decision_id"] for r in rows
            if not r["decision_id"].startswith("synth-")
        ]
        assert len(real_ids) == 100
        assert len(set(real_ids)) == 100, "collisions in audit log"
        for did in real_ids:
            assert ULID_PATTERN.match(did), f"Bad ULID in audit: {did}"

    # ── Synthetic decision_id is labeled and excluded ──

    def test_synthetic_decision_id_prefix(self):
        from src.orchestration.envelope import synthetic_decision_id
        sid = synthetic_decision_id("shutdown_flatten")
        assert sid.startswith("synth-shutdown_flatten-")
        assert not ULID_PATTERN.match(sid), "synth IDs must NOT match raw ULID pattern"

    # ── Live audit-log sanity check (skipped if no audit log yet) ──

    def test_live_audit_log_has_decision_id(self):
        """If the deployed bot has written any rows, verify they all carry a decision_id.

        This is the Phase 0 go/no-go check: one week of live paper-trading should
        leave every row with a unique ULID decision_id.
        """
        path = Path(__file__).resolve().parent.parent / "logs" / "trade_decisions.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            pytest.skip("no live audit log yet — run the bot first")

        seen_ids = set()
        bad_rows = []
        real_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    bad_rows.append((lineno, "JSON parse"))
                    continue

                did = row.get("decision_id")
                if did is None:
                    # Pre-Phase-0 historical rows are allowed — the deploy cutover
                    # is not instant. Only fail on rows written after cutover.
                    # Skip rows lacking the 'provenance' key as pre-cutover.
                    if "provenance" in row:
                        bad_rows.append((lineno, "post-cutover row missing decision_id"))
                    continue

                if did.startswith("synth-"):
                    continue  # error-path rows excluded

                if not ULID_PATTERN.match(did):
                    bad_rows.append((lineno, f"malformed ULID: {did}"))
                    continue

                if did in seen_ids:
                    bad_rows.append((lineno, f"duplicate decision_id: {did}"))
                    continue
                seen_ids.add(did)
                real_count += 1

        assert not bad_rows, f"{len(bad_rows)} bad audit rows: {bad_rows[:5]}"
        # If we saw any post-cutover rows, we saw at least one. If zero,
        # either cutover hasn't happened or no decisions ran — tolerate.
