import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time

STATE_FILE = "logs/dashboard_state.json"

# Default state template (never written directly — only merged)
_DEFAULT_STATE = {
    "active_assets": {},
    "portfolio": {
        "pnl": 0.0,
        "return": 0.0,
        "equity_curve": []
    },
    "agentic_log": [],
    "memory_hits": [],
    "onchain_metrics": {},
    "advanced_learning": {
        "regimes": {},
        "strategies": {},
        "patterns": {},
        "timestamp": ""
    },
    "status": "INITIALIZING",
    "last_update": "",
    "accuracy": 0.5
}


class DashboardState:
    """Cross-process state manager using JSON file as shared bus.
    
    Design:
      - Every read reloads from disk (cross-process safe)
      - Every write does read-modify-write atomically (within the lock)
      - Default keys are merged non-destructively on load
    """
    _instance: Optional['DashboardState'] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DashboardState, cls).__new__(cls)
                cls._instance.state: Dict[str, Any] = {}
                cls._instance._ensure_file()
        return cls._instance

    def _ensure_file(self):
        """Create the state file with defaults if it doesn't exist."""
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        if not os.path.exists(STATE_FILE):
            self._write_file(_DEFAULT_STATE.copy())

    def _read_file(self) -> Dict[str, Any]:
        """Read state from disk, merging with defaults for missing keys."""
        try:
            with open(STATE_FILE, 'r') as f:
                file_state = json.load(f)
            # Merge defaults for any missing keys (schema migration)
            merged = _DEFAULT_STATE.copy()
            merged.update(file_state)
            return merged
        except (json.JSONDecodeError, FileNotFoundError, Exception):
            return _DEFAULT_STATE.copy()

    def _write_file(self, state: Dict[str, Any]):
        """Atomically write state to disk."""
        try:
            tmp_path = STATE_FILE + ".tmp"
            with open(tmp_path, 'w') as f:
                json.dump(state, f)
            # Atomic rename (as close as Windows allows)
            if os.path.exists(STATE_FILE):
                os.replace(tmp_path, STATE_FILE)
            else:
                os.rename(tmp_path, STATE_FILE)
        except Exception:
            pass

    def _read_modify_write(self, modifier_fn):
        """Thread-safe read-modify-write cycle."""
        with self._lock:
            state = self._read_file()
            modifier_fn(state)
            self._write_file(state)
            self.state = state

    # --- Public API ---

    def update_asset(self, asset: str, data: Dict[str, Any]):
        def _mod(s):
            if asset not in s["active_assets"]:
                s["active_assets"][asset] = {}
            s["active_assets"][asset].update(data)
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def add_agent_thought(self, asset: str, regime: str, thought: str, confidence: int):
        def _mod(s):
            entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "asset": asset,
                "regime": regime,
                "thought": thought,
                "confidence": confidence
            }
            s["agentic_log"].insert(0, entry)
            s["agentic_log"] = s["agentic_log"][:50]
        self._read_modify_write(_mod)

    def update_portfolio(self, pnl: float, asset_return: float):
        def _mod(s):
            s["portfolio"]["pnl"] = pnl
            s["portfolio"]["return"] = asset_return
            s["portfolio"]["equity_curve"].append({
                "t": datetime.now().isoformat(),
                "v": pnl
            })
        self._read_modify_write(_mod)

    def set_memory_hits(self, hits: List[Dict]):
        def _mod(s):
            s["memory_hits"] = hits
        self._read_modify_write(_mod)

    def update_onchain(self, metrics: Dict[str, Any]):
        def _mod(s):
            s["onchain_metrics"] = metrics
        self._read_modify_write(_mod)

    def update_advanced_learning(self, learning_data: Dict[str, Any]):
        def _mod(s):
            s["advanced_learning"] = learning_data
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def set_status(self, status: str):
        def _mod(s):
            s["status"] = status
        self._read_modify_write(_mod)

    def get_full_state(self) -> Dict:
        """Read-only: returns latest state from disk."""
        self.state = self._read_file()
        return self.state
