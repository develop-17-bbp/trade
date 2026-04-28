"""Tests for persistent_context — Ollama KV-thread reuse per asset.

Default OFF; existing dynamic prompting path is unaffected when env unset.
"""
from __future__ import annotations

import pytest


def test_persistent_context_dormant_when_unset(monkeypatch):
    monkeypatch.delenv("ACT_PERSISTENT_CONTEXT", raising=False)
    from src.ai.persistent_context import is_enabled
    assert is_enabled() is False


def test_persistent_context_enabled_with_one(monkeypatch):
    monkeypatch.setenv("ACT_PERSISTENT_CONTEXT", "1")
    from src.ai.persistent_context import is_enabled
    assert is_enabled() is True


def test_persistent_context_consolidate_every_default():
    from src.ai.persistent_context import (
        _consolidate_every, DEFAULT_CONSOLIDATE_EVERY,
    )
    assert _consolidate_every() == DEFAULT_CONSOLIDATE_EVERY


def test_persistent_context_consolidate_every_env_override(monkeypatch):
    monkeypatch.setenv("ACT_CONTEXT_CONSOLIDATE_EVERY", "30")
    from src.ai.persistent_context import _consolidate_every
    assert _consolidate_every() == 30


def test_manager_singleton():
    from src.ai.persistent_context import get_manager
    m1 = get_manager()
    m2 = get_manager()
    assert m1 is m2


def test_thread_starts_uninitialized():
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    t = m.get("BTC")
    assert t.context_array is None
    assert t.n_calls_since_seed == 0


def test_needs_seeding_first_call():
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    assert m.needs_seeding("BTC") is True


def test_needs_seeding_after_consolidate(monkeypatch):
    monkeypatch.setenv("ACT_CONTEXT_CONSOLIDATE_EVERY", "5")
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    # Manually set state past consolidation threshold
    t = m.get("BTC")
    t.context_array = [1, 2, 3]
    t.n_calls_since_seed = 6
    assert m.needs_seeding("BTC") is True


def test_update_increments_call_count():
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    m.update("BTC", [10, 20, 30], token_delta=100)
    t = m.get("BTC")
    assert t.context_array == [10, 20, 30]
    assert t.n_calls_since_seed == 1
    assert t.estimated_token_count == 100


def test_update_with_none_does_nothing():
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    m.update("BTC", None, token_delta=100)
    t = m.get("BTC")
    assert t.context_array is None


def test_reset_clears_state():
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    m.update("BTC", [1, 2, 3], token_delta=500)
    m.reset("BTC")
    t = m.get("BTC")
    assert t.context_array is None
    assert t.n_calls_since_seed == 0


def test_per_asset_isolation():
    from src.ai.persistent_context import PersistentContextManager
    m = PersistentContextManager()
    m.update("BTC", [1, 2, 3])
    m.update("ETH", [4, 5, 6])
    assert m.get("BTC").context_array == [1, 2, 3]
    assert m.get("ETH").context_array == [4, 5, 6]


def test_build_evidence_delta_returns_full_on_first_call():
    from src.ai.persistent_context import build_evidence_delta
    full = "GOAL: ...\nPORTFOLIO: ..."
    d = build_evidence_delta("BTC", full, "")
    assert d == full


def test_build_evidence_delta_returns_only_new_lines():
    from src.ai.persistent_context import build_evidence_delta
    last = "GOAL: today=+0.1%\nPORTFOLIO: same_asset_open=2"
    full = "GOAL: today=+0.3%\nPORTFOLIO: same_asset_open=2"
    d = build_evidence_delta("BTC", full, last)
    # Only the GOAL line changed
    assert "today=+0.3%" in d
    assert "PORTFOLIO" not in d


def test_build_evidence_delta_no_change():
    from src.ai.persistent_context import build_evidence_delta
    full = "GOAL: ...\nPORTFOLIO: ..."
    d = build_evidence_delta("BTC", full, full)
    assert d == "(no material change since last tick)"


def test_persistent_context_tool_registered():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    assert "query_persistent_context_stats" in set(r.list_names())


def test_persistent_context_tool_dispatches():
    from src.ai.trade_tools import build_default_registry
    r = build_default_registry()
    res = r.dispatch("query_persistent_context_stats", {})
    s = str(res).lower()
    assert "enabled" in s
