"""
Shared Context Bus — ACT v8.0
================================
Central nervous system for inter-layer communication.
Every layer, model, agent, and strategy can READ and WRITE to the shared context.
No more waterfall — all components see the same live state simultaneously.

Architecture:
  - Single SharedContext instance per trading cycle
  - Any component can publish data: ctx.publish('macro.usd_regime', 'weak')
  - Any component can read data: ctx.get('agents.trend_momentum.direction')
  - Components subscribe to events: ctx.on('trade.closed', callback)
  - Full state snapshot available for LLM prompt injection

This replaces the waterfall model where L1 -> L2 -> L3 in sequence.
Now ALL layers see ALL data and can react to each other.
"""
import time
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class SharedContext:
    """Central context bus — every ACT component reads/writes here."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: List[Dict] = []  # last N state snapshots

    def publish(self, key: str, value: Any, source: str = ''):
        """Publish data to the shared context. Any component can call this.

        Examples:
            ctx.publish('macro.usd_regime', 'weak', source='usd_strength_layer')
            ctx.publish('agents.risk_guardian.veto', True, source='risk_guardian')
            ctx.publish('models.lgbm.prediction', 0.72, source='lgbm_classifier')
            ctx.publish('market.btc.regime', 'BULL', source='hmm_regime')
        """
        with self._lock:
            self._data[key] = value
            self._timestamps[key] = time.time()

        # Notify subscribers
        for callback in self._subscribers.get(key, []):
            try:
                callback(key, value, source)
            except Exception as e:
                logger.debug(f"[CTX] Subscriber error for {key}: {e}")

        # Wildcard subscribers (e.g., subscribe to 'agents.*')
        parts = key.split('.')
        for i in range(len(parts)):
            wildcard = '.'.join(parts[:i+1]) + '.*'
            for callback in self._subscribers.get(wildcard, []):
                try:
                    callback(key, value, source)
                except Exception:
                    pass

    def get(self, key: str, default: Any = None) -> Any:
        """Read from shared context. Any component can call this."""
        with self._lock:
            return self._data.get(key, default)

    def get_age(self, key: str) -> float:
        """How many seconds ago was this key last updated?"""
        ts = self._timestamps.get(key, 0)
        return time.time() - ts if ts > 0 else float('inf')

    def get_fresh(self, key: str, max_age_sec: float = 300, default: Any = None) -> Any:
        """Get value only if it's fresh enough, else return default."""
        if self.get_age(key) > max_age_sec:
            return default
        return self.get(key, default)

    def on(self, key: str, callback: Callable):
        """Subscribe to changes on a key. Supports wildcards like 'agents.*'."""
        self._subscribers[key].append(callback)

    def get_namespace(self, prefix: str) -> Dict[str, Any]:
        """Get all keys under a namespace.

        Example: ctx.get_namespace('agents') returns all agent data.
        """
        with self._lock:
            return {k: v for k, v in self._data.items() if k.startswith(prefix)}

    def get_full_state(self) -> Dict[str, Any]:
        """Full snapshot of all shared context data."""
        with self._lock:
            return dict(self._data)

    def get_llm_context_block(self, max_keys: int = 50) -> str:
        """Format shared context as text for LLM prompt injection.
        Every layer's latest state visible to the LLM in one block."""
        with self._lock:
            lines = ["=== SHARED CONTEXT (live inter-layer state) ==="]

            # Group by namespace
            namespaces = defaultdict(dict)
            for key, val in sorted(self._data.items()):
                ns = key.split('.')[0] if '.' in key else 'global'
                namespaces[ns][key] = val

            count = 0
            for ns, items in sorted(namespaces.items()):
                lines.append(f"\n[{ns.upper()}]")
                for key, val in sorted(items.items()):
                    age = self.get_age(key)
                    stale_marker = " (STALE)" if age > 600 else ""
                    # Truncate long values
                    val_str = str(val)[:80]
                    lines.append(f"  {key} = {val_str}{stale_marker}")
                    count += 1
                    if count >= max_keys:
                        lines.append(f"  ... ({len(self._data) - count} more keys)")
                        return '\n'.join(lines)

            return '\n'.join(lines)

    def snapshot(self):
        """Save current state to history (called after each trading cycle)."""
        with self._lock:
            snap = {
                'time': time.time(),
                'data': dict(self._data),
            }
            self._history.append(snap)
            # Keep last 100 snapshots
            if len(self._history) > 100:
                self._history = self._history[-100:]

    def clear(self):
        """Clear all context (new trading cycle)."""
        with self._lock:
            self._data.clear()
            self._timestamps.clear()


class ContextPublisher:
    """Mixin for any component that publishes to SharedContext.

    Usage:
        class MyAgent(ContextPublisher):
            def __init__(self, ctx):
                super().__init__(ctx, namespace='agents.my_agent')

            def analyze(self):
                result = ...
                self.pub('direction', 1)       # publishes 'agents.my_agent.direction'
                self.pub('confidence', 0.85)
    """

    def __init__(self, ctx: SharedContext, namespace: str):
        self._ctx = ctx
        self._namespace = namespace

    def pub(self, key: str, value: Any):
        full_key = f"{self._namespace}.{key}"
        self._ctx.publish(full_key, value, source=self._namespace)

    def read(self, key: str, default: Any = None) -> Any:
        return self._ctx.get(key, default)

    def read_namespace(self, prefix: str) -> Dict[str, Any]:
        return self._ctx.get_namespace(prefix)


def create_shared_context() -> SharedContext:
    """Factory function — creates the single shared context instance."""
    ctx = SharedContext()
    logger.info("[CTX] Shared context bus initialized")
    return ctx
