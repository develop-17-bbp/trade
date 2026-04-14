import pytest
from src.trading.executor import TradingExecutor


def test_paper_run_smoke():
    """Smoke test: executor initializes correctly in paper mode with all subsystems."""
    cfg = {
        "mode": "paper",
        "assets": ["BTC"],
        "risk": {"max_position_size_pct": 1.0}
    }
    ex = TradingExecutor(cfg)
    # Executor should initialize all subsystems without crashing
    assert ex is not None
    assert ex.config.get("mode") == "paper"
    assert "BTC" in ex.config.get("assets", [])
    # Key subsystems should be initialized
    assert hasattr(ex, '_lgbm')       # LightGBM classifier
    assert hasattr(ex, '_multi_engine')  # Multi-strategy engine
    assert hasattr(ex, '_adaptive')   # Adaptive feedback loop
