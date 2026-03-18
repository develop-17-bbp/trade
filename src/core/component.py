"""
Component Sandbox — wraps any model/agent so failures never crash the system.
"""
import logging
import traceback
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar('T')


class ComponentSandbox:
    """
    Wraps a component so that:
    - Init failures produce a NullComponent, not a crash
    - Method calls on a failed component return safe defaults
    - Health status is always queryable via .healthy / .status
    """

    def __init__(self, name: str, factory: Callable[[], T], fallback: Any = None,
                 critical: bool = False):
        self.name = name
        self._healthy = False
        self._error: Optional[str] = None
        self._instance: Optional[T] = fallback

        try:
            self._instance = factory()
            self._healthy = True
            logger.info(f"[COMPONENT] + {name} initialized")
        except Exception as e:
            self._error = str(e)
            tb = traceback.format_exc()
            if critical:
                logger.error(f"[COMPONENT] x CRITICAL {name} failed: {e}\n{tb}")
                raise  # re-raise for truly critical components
            else:
                logger.warning(f"[COMPONENT] x {name} failed (degraded): {e}")

    # -- Proxy attribute access to wrapped instance --
    def __getattr__(self, attr: str):
        instance = object.__getattribute__(self, '_instance')
        if instance is not None:
            return getattr(instance, attr)
        # No instance — return a safe no-op callable
        def _noop(*args, **kwargs):
            logger.debug(f"[COMPONENT] {object.__getattribute__(self, 'name')}.{attr} called on degraded component — returning None")
            return None
        return _noop

    @property
    def healthy(self) -> bool:
        return self._healthy

    @property
    def status(self) -> dict:
        return {
            'name': self.name,
            'healthy': self._healthy,
            'error': self._error,
            'type': type(self._instance).__name__ if self._instance else 'NullComponent',
        }

    def __bool__(self):
        return self._healthy and self._instance is not None

    def __repr__(self):
        state = '+' if self._healthy else 'x'
        return f"<ComponentSandbox [{state}] {self.name}>"


class ComponentRegistry:
    """
    Central registry of all sandboxed components.
    Provides health aggregation and status API.
    """
    _instance: Optional['ComponentRegistry'] = None

    def __init__(self):
        self._components: dict[str, ComponentSandbox] = {}

    @classmethod
    def get(cls) -> 'ComponentRegistry':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: str, factory: Callable, fallback=None,
                 critical: bool = False) -> ComponentSandbox:
        sb = ComponentSandbox(name, factory, fallback=fallback, critical=critical)
        self._components[name] = sb
        return sb

    def health_report(self) -> dict:
        report = {}
        for name, sb in self._components.items():
            report[name] = sb.status
        healthy = sum(1 for sb in self._components.values() if sb.healthy)
        total = len(self._components)
        return {
            'healthy': healthy,
            'degraded': total - healthy,
            'total': total,
            'components': report,
        }

    def all_healthy(self) -> bool:
        return all(sb.healthy for sb in self._components.values())
