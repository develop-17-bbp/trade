import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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
