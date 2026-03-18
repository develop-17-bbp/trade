"""
Centralized path management for the trading system.
All logs, models cache, and runtime data live OUTSIDE the repo by default.
Future: swap LOG_DIR backend for AWS RDS / S3.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# -- Project root (where this code lives) --
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -- Runtime data root — defaults to ~/.trade, overridable via env var --
_DATA_ROOT_ENV = os.environ.get("TRADE_DATA_ROOT", "")
if _DATA_ROOT_ENV:
    DATA_ROOT = Path(_DATA_ROOT_ENV).expanduser().resolve()
else:
    DATA_ROOT = Path.home() / ".trade"

# -- Sub-directories --
LOG_DIR    = DATA_ROOT / "logs"
MODEL_DIR  = DATA_ROOT / "models"        # trained model files (.txt, .pt, .pkl)
MEMORY_DIR = DATA_ROOT / "memory"        # LLM memory / experience vault
DB_DIR     = DATA_ROOT / "db"            # SQLite databases

# -- Specific files --
DASHBOARD_STATE_FILE = LOG_DIR / "dashboard_state.json"
TRADING_JOURNAL_FILE = LOG_DIR / "trading_journal.enc"
AUDIT_LOG_FILE       = LOG_DIR / "audit_failover.jsonl"
ALERTS_LOG_FILE      = LOG_DIR / "alerts.jsonl"
TRAINING_STATE_FILE  = LOG_DIR / "training_state.json"
BENCHMARK_FILE       = LOG_DIR / "benchmark_history.json"
TRADING_DB_FILE      = DB_DIR  / "trading_state.db"


def ensure_dirs() -> None:
    """Create all runtime directories if they don't exist."""
    for d in [LOG_DIR, MODEL_DIR, MEMORY_DIR, DB_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logger.debug(f"[PATHS] Runtime data root: {DATA_ROOT}")


def get_log_path(filename: str) -> Path:
    """Get a path inside LOG_DIR, creating LOG_DIR if needed."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / filename


def get_model_path(filename: str) -> Path:
    """Get a path inside MODEL_DIR."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / filename


# Auto-ensure dirs on import
ensure_dirs()
