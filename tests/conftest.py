import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _isolate_runtime_overlays(monkeypatch, tmp_path):
    """Tests must not see the operator's local data/paper_soak_loose.json
    or local env state. Without this, conviction_gate / cost_gate tests
    see whatever overlay happens to be on disk and assert against
    moving thresholds. Redirect the overlay to a tmp path (won't exist,
    so reader returns None) and clear paper/real mode env vars so each
    test sets exactly what it needs."""
    overlay_path = tmp_path / "paper_soak_loose.json"
    try:
        from skills.paper_soak_loose import action as _psl_action
        monkeypatch.setattr(_psl_action, "OVERLAY_FILE", overlay_path,
                            raising=False)
    except Exception:
        pass
    monkeypatch.delenv("ACT_REAL_CAPITAL_ENABLED", raising=False)
    monkeypatch.delenv("ACT_FORBID_MODELS", raising=False)
    yield


@pytest.fixture
def stub_dual_brain(monkeypatch):
    """Shared fixture — tests that stub src.ai.dual_brain.analyze /
    scan via sys.modules were re-implementing the same 3-line setup
    (test_polymarket_analyst, test_persona_from_graph, test_shadow_tick,
    test_trade_verifier). Centralised here.

    Returns the injected mock module so callers can program specific
    responses:
        def test_x(stub_dual_brain):
            stub_dual_brain.analyze.return_value = MagicMock(ok=True, text="...")
    """
    import sys
    from unittest import mock
    fake_module = mock.MagicMock()
    fake_module.analyze.return_value = mock.MagicMock(
        ok=True, text="", model="stub", fallback_used=False,
    )
    fake_module.scan.return_value = mock.MagicMock(
        ok=True, text="", model="stub", fallback_used=False,
    )
    monkeypatch.setitem(sys.modules, "src.ai.dual_brain", fake_module)
    yield fake_module


# Shared fixture — several tests (test_agent_post_mortem, test_diagnose_noop,
# test_dual_brain_trainer, test_shadow_tick) redirect warm_store to a tmp
# sqlite. Keep ONE canonical builder here so updates to the schema
# propagate automatically.
@pytest.fixture
def tmp_warm_store(tmp_path, monkeypatch):
    """Redirect warm_store singleton + ACT_WARM_DB_PATH at a tmp sqlite.

    Yields the WarmStore instance. Callers that need to seed rows can
    use `warm_store.write_decision(...)` directly.
    """
    from src.orchestration.warm_store import WarmStore
    import src.orchestration.warm_store as ws_mod
    db_path = tmp_path / "warm_store.sqlite"
    store = WarmStore(str(db_path))
    monkeypatch.setenv("ACT_WARM_DB_PATH", str(db_path))
    monkeypatch.setattr(ws_mod, "_store_singleton", store, raising=False)
    yield store
    try:
        store.close()
    except Exception:
        pass
