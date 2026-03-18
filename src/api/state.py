import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading
import time

# Resolve state file relative to project root so dashboard and executor always use the same file
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
STATE_FILE = os.path.join(_PROJECT_ROOT, "logs", "dashboard_state.json")

# Default state template (never written directly — only merged)
_DEFAULT_STATE = {
    "active_assets": {},
    "portfolio": {
        "pnl": 0.0,
        "return": 0.0,
        "total": 0.0,
        "equity_curve": []
    },
    "agentic_log": [],
    "memory_hits": [],
    "onchain_metrics": {},
    "sentiment": {},
    "l1_features": {},
    "layers": {},
    "sources": {
        "exchange": "UNKNOWN",
        "news": "UNKNOWN",
        "onchain": "UNKNOWN",
        "llm": "UNKNOWN"
    },
    "advanced_learning": {
        "regimes": {},
        "strategies": {},
        "patterns": {},
        "timestamp": ""
    },
    "performance_edge": {
        "uplift_pct": 0.0,
        "baseline_winrate": 0.5,
        "agent_winrate": 0.5
    },
    "execution": {
        "slippage": 0.0,
        "fill_rate": 100.0,
        "latency_ms": 0,
        "orders_per_min": 0.0
    },
    "risk_metrics": {
        "vpin_threshold": 0.8,
        "max_drawdown": 0.0,
        "current_drawdown": 0.0,
        "risk_score": 0.0
    },
    "training_status": "IDLE",
    "model_version": "v6.0",
    "performance_gain": 0.0,
    "next_training": "",
    "memory_latency": 0,
    "memory_confidence": 0.7,
    "strategist_confidence": 0.7,
    "last_reasoning": "",
    "status": "INITIALIZING",
    "last_update": "",
    "accuracy": 0.5,
    "layer_logs": {
        "L1": [],
        "L2": [],
        "L3": [],
        "L4": [],
        "L5": [],
        "L6": [],
        "L7": [],
        "L8": [],
        "L9": []
    },
    "trade_history": [],
    "open_positions": {},
    "benchmark": {
        "predictions": [],
        "actuals": [],
        "per_model": {
            "lightgbm": {"predictions": [], "actuals": [], "correct": 0, "total": 0},
            "patchtst": {"predictions": [], "actuals": [], "correct": 0, "total": 0},
            "rl_agent": {"predictions": [], "actuals": [], "correct": 0, "total": 0},
            "strategist": {"predictions": [], "actuals": [], "correct": 0, "total": 0}
        }
    },
    "performance": {},
    "agent_overlay": {
        "enabled": False,
        "last_decision": {},
        "agent_votes": {},
        "agent_weights": {},
        "consensus_level": "N/A",
        "data_quality": 0.0,
        "daily_pnl_mode": "NORMAL",
        "cycle_count": 0,
        "last_cycle_time": ""
    },
    "polymarket": {
        "active_markets": 0,
        "liquid_markets": 0,
        "top_markets": [],
        "top_divergences": [],
        "avg_divergence": 0.0,
        "last_fetch": ""
    }
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
            # Deep-merge portfolio so pnl/return/total are always present and visible in cards
            default_portfolio = _DEFAULT_STATE["portfolio"].copy()
            file_portfolio = merged.get("portfolio") or {}
            if isinstance(file_portfolio, dict):
                default_portfolio.update(file_portfolio)
                # Preserve equity_curve from file (list), don't overwrite with empty
                if file_portfolio.get("equity_curve"):
                    default_portfolio["equity_curve"] = file_portfolio["equity_curve"]
            merged["portfolio"] = default_portfolio
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

    def update_portfolio(self, pnl: float, asset_return: float, total_value: float = None):
        def _mod(s):
            port = s.setdefault("portfolio", _DEFAULT_STATE["portfolio"].copy())
            port["pnl"] = pnl
            port["return"] = asset_return
            if total_value is not None:
                port["total"] = total_value
            if "equity_curve" not in port:
                port["equity_curve"] = []
            port["equity_curve"].append({
                "t": datetime.now().isoformat(),
                "v": pnl
            })
            s["last_update"] = datetime.now().isoformat()
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

    def update_sentiment(self, asset: str, data: Dict[str, Any]):
        def _mod(s):
            s["sentiment"][asset] = data
        self._read_modify_write(_mod)

    def update_l1_features(self, asset: str, data: Dict[str, Any]):
        def _mod(s):
            s["l1_features"][asset] = data
        self._read_modify_write(_mod)

    def set_layers(self, layers: Dict[str, Any]):
        def _mod(s):
            s["layers"] = layers
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def set_sources(self, sources: Dict[str, str]):
        def _mod(s):
            s["sources"].update(sources)
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def update_performance_edge(self, edge_data: Dict[str, float]):
        def _mod(s):
            s["performance_edge"].update(edge_data)
        self._read_modify_write(_mod)

    def update_execution_metrics(self, exec_data: Dict[str, float]):
        def _mod(s):
            s["execution"].update(exec_data)
        self._read_modify_write(_mod)

    def update_risk_metrics(self, risk_data: Dict[str, float]):
        def _mod(s):
            s["risk_metrics"].update(risk_data)
        self._read_modify_write(_mod)

    def update_training_status(self, status: str, version: str = None, gain: float = None, next_training: str = None):
        def _mod(s):
            s["training_status"] = status
            if version:
                s["model_version"] = version
            if gain is not None:
                s["performance_gain"] = gain
            if next_training:
                s["next_training"] = next_training
        self._read_modify_write(_mod)

    def update_memory_metrics(self, latency: int, confidence: float):
        def _mod(s):
            s["memory_latency"] = latency
            s["memory_confidence"] = confidence
        self._read_modify_write(_mod)

    def update_strategist_metrics(self, confidence: float, last_reasoning: str):
        def _mod(s):
            s["strategist_confidence"] = confidence
            s["last_reasoning"] = last_reasoning
        self._read_modify_write(_mod)

    def add_layer_log(self, layer: str, message: str, level: str = "INFO"):
        """Add a log entry for a specific layer"""
        def _mod(s):
            log_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "message": message,
                "level": level
            }
            if layer not in s["layer_logs"]:
                s["layer_logs"][layer] = []
            s["layer_logs"][layer].insert(0, log_entry)
            # Keep only last 20 logs per layer
            s["layer_logs"][layer] = s["layer_logs"][layer][:20]
        self._read_modify_write(_mod)

    def get_layer_logs(self, layer: str, limit: int = 5) -> List[Dict]:
        """Get recent logs for a specific layer"""
        state = self.get_full_state()
        return state.get("layer_logs", {}).get(layer, [])[:limit]

    def set_status(self, status: str):
        def _mod(s):
            s["status"] = status
        self._read_modify_write(_mod)

    def get_full_state(self) -> Dict:
        """Read-only: returns latest state from disk."""
        self.state = self._read_file()
        return self.state

    def update_open_positions(self, positions: Dict[str, Any]):
        """Push current open positions with entry price, size, unrealized P&L."""
        def _mod(s):
            s["open_positions"] = positions
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def record_trade(self, trade_data: Dict):
        """Record a completed trade for benchmark tracking."""
        def _mod(s):
            if "trade_history" not in s:
                s["trade_history"] = []
            s["trade_history"].append(trade_data)
            # Keep last 500 trades
            s["trade_history"] = s["trade_history"][-500:]
        self._read_modify_write(_mod)

    def record_prediction(self, predicted_direction: int, actual_direction: int):
        """Record a prediction vs actual for direction accuracy tracking."""
        def _mod(s):
            if "benchmark" not in s:
                s["benchmark"] = {"predictions": [], "actuals": []}
            s["benchmark"]["predictions"].append(predicted_direction)
            s["benchmark"]["actuals"].append(actual_direction)
            # Keep last 1000
            s["benchmark"]["predictions"] = s["benchmark"]["predictions"][-1000:]
            s["benchmark"]["actuals"] = s["benchmark"]["actuals"][-1000:]
        self._read_modify_write(_mod)

    def update_benchmark_metrics(self, metrics: Dict):
        """Push computed benchmark metrics (win_rate, sharpe, etc)."""
        def _mod(s):
            s["performance"] = metrics
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def update_agent_overlay(self, agent_data: Dict[str, Any]):
        """Push full agent overlay status (decision, votes, weights, consensus)."""
        def _mod(s):
            if "agent_overlay" not in s:
                s["agent_overlay"] = {}
            s["agent_overlay"].update(agent_data)
            s["agent_overlay"]["last_cycle_time"] = datetime.now().isoformat()
            s["agent_overlay"]["cycle_count"] = s["agent_overlay"].get("cycle_count", 0) + 1
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def update_agent_decision(self, decision: Dict[str, Any]):
        """Push the latest enhanced decision from the agent overlay."""
        def _mod(s):
            if "agent_overlay" not in s:
                s["agent_overlay"] = {}
            s["agent_overlay"]["last_decision"] = decision
            s["agent_overlay"]["last_cycle_time"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def update_polymarket(self, data: Dict[str, Any]):
        """Push Polymarket prediction market data."""
        def _mod(s):
            s["polymarket"] = data
            s["last_update"] = datetime.now().isoformat()
        self._read_modify_write(_mod)

    def record_model_prediction(self, model_name: str, predicted: int, actual: int):
        """Record a single model's prediction vs actual for per-model accuracy."""
        def _mod(s):
            if "benchmark" not in s:
                s["benchmark"] = {"predictions": [], "actuals": [], "per_model": {}}
            pm = s["benchmark"].setdefault("per_model", {})
            m = pm.setdefault(model_name, {"predictions": [], "actuals": [], "correct": 0, "total": 0})
            m["predictions"].append(predicted)
            m["actuals"].append(actual)
            m["total"] += 1
            if (predicted > 0) == (actual > 0) and actual != 0:
                m["correct"] += 1
            # Keep last 1000
            m["predictions"] = m["predictions"][-1000:]
            m["actuals"] = m["actuals"][-1000:]
        self._read_modify_write(_mod)
