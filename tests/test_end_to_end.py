import pytest
from src.trading.executor import TradingExecutor


def test_paper_run_smoke():
    """Smoke test with real-time data (no demo/synthetic)."""
    cfg = {
        "mode": "paper",
        "assets": ["BTC"],
        "risk": {"max_position_size_pct": 1.0}
    }
    ex = TradingExecutor(cfg)
    # run() will fetch real-time data and execute the full pipeline
    ex.run()
    # If we reach here without exception, executor initialized and ran successfully
    assert ex.mode == "paper"
    assert "BTC" in ex.assets
