"""
Hooks system — named-event extension points for operator customization.

Claude Code has PreToolUse / PostToolUse / UserPromptSubmit / SessionStart
etc. hooks. ACT's equivalent: every significant trading-lifecycle event
fires through a central dispatcher. Operators register shell commands or
Python callables in config.yaml; each hook sees a context dict with the
event's details.

Why this matters (from operator audit):
  * Custom policies without code changes — want a Slack alert on
    authority violations? One config line.
  * Pre-trade hooks are a final safety layer — a hook that checks
    `/tmp/ACT_STOP` gives one-keystroke ops shutdown independent of
    the bot's own state.
  * Audit discipline — every emergency-mode toggle, every authority
    violation, every champion promotion emits an event any downstream
    audit tool can tail.

Event catalog (canonical names):
  * pre_trade_submit          — Fires before the executor places an
                                 order. Blocking hooks can VETO by
                                 returning non-zero exit code / ok=False.
  * post_trade_open           — After order fills. Non-blocking.
  * pre_exit                  — Before closing a position. Blocking.
  * post_trade_close          — After close fills. Non-blocking.
  * on_authority_violation    — Authority rule triggered. Non-blocking.
  * on_emergency_mode_enter   — Rolling Sharpe/target dropped below
                                 threshold.
  * on_emergency_mode_exit    — Recovered above threshold.
  * on_strategy_promote       — A challenger became champion.
  * on_startup / on_shutdown  — Lifecycle.

Config shape (config.yaml):

    hooks:
      pre_trade_submit:
        - cmd: "python scripts/killswitch_check.py"
          blocking: true                  # non-zero exit vetoes
          timeout_s: 3
        - python: "src.skills:get_registry"  # python:"<module>:<attr>"
          blocking: false
      on_emergency_mode_enter:
        - cmd: "notify-send 'ACT EMERGENCY'"
          blocking: false

Design constraints:
  * Never raises — a hook error becomes a logged HookError, not a
    dead dispatcher.
  * Blocking hooks have timeouts; a stuck hook does NOT freeze the
    executor.
  * Env-var expansion in cmd strings (${VAR}).
  * Env kill switch ACT_DISABLE_HOOKS=1 short-circuits everything.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


DISABLE_ENV = "ACT_DISABLE_HOOKS"
DEFAULT_TIMEOUT_S = float(os.getenv("ACT_HOOK_TIMEOUT_S", "5.0"))

# Canonical event names — other modules use these as constants rather than
# passing raw strings so typos are a compile-time issue, not a silent miss.
EVT_PRE_TRADE_SUBMIT       = "pre_trade_submit"
EVT_POST_TRADE_OPEN        = "post_trade_open"
EVT_PRE_EXIT               = "pre_exit"
EVT_POST_TRADE_CLOSE       = "post_trade_close"
EVT_ON_AUTHORITY_VIOLATION = "on_authority_violation"
EVT_ON_EMERGENCY_ENTER     = "on_emergency_mode_enter"
EVT_ON_EMERGENCY_EXIT      = "on_emergency_mode_exit"
EVT_ON_STRATEGY_PROMOTE    = "on_strategy_promote"
EVT_ON_STARTUP             = "on_startup"
EVT_ON_SHUTDOWN            = "on_shutdown"

VALID_EVENTS = frozenset({
    EVT_PRE_TRADE_SUBMIT, EVT_POST_TRADE_OPEN, EVT_PRE_EXIT,
    EVT_POST_TRADE_CLOSE, EVT_ON_AUTHORITY_VIOLATION,
    EVT_ON_EMERGENCY_ENTER, EVT_ON_EMERGENCY_EXIT,
    EVT_ON_STRATEGY_PROMOTE, EVT_ON_STARTUP, EVT_ON_SHUTDOWN,
})


# ── Data types ──────────────────────────────────────────────────────────


@dataclass
class HookSpec:
    """One registered hook. Either `cmd` (shell) or `python` (dotted path)."""
    event: str
    cmd: Optional[str] = None
    python: Optional[str] = None          # "module.path:attr"
    blocking: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    name: str = ""                         # optional friendly label

    def is_shell(self) -> bool:
        return bool(self.cmd)

    def is_python(self) -> bool:
        return bool(self.python) and not self.cmd


@dataclass
class HookResult:
    """Outcome of one hook invocation."""
    event: str
    hook_name: str
    ok: bool
    vetoed: bool = False
    output: str = ""
    error: Optional[str] = None
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event, "hook_name": self.hook_name,
            "ok": self.ok, "vetoed": self.vetoed,
            "output": (self.output or "")[:800],
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class FireResult:
    """Aggregate result of firing all hooks for one event."""
    event: str
    any_veto: bool = False
    results: List[HookResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "any_veto": self.any_veto,
            "results": [r.to_dict() for r in self.results],
        }


# ── Registry ────────────────────────────────────────────────────────────


class HookRegistry:
    """Event → list[HookSpec]. Thread-safe."""

    def __init__(self):
        self._hooks: Dict[str, List[HookSpec]] = {}
        self._lock = threading.Lock()

    def register(self, spec: HookSpec) -> None:
        if spec.event not in VALID_EVENTS:
            logger.warning("hooks: unknown event %r — registering anyway", spec.event)
        with self._lock:
            self._hooks.setdefault(spec.event, []).append(spec)

    def clear(self, event: Optional[str] = None) -> None:
        with self._lock:
            if event is None:
                self._hooks.clear()
            else:
                self._hooks.pop(event, None)

    def list_for(self, event: str) -> List[HookSpec]:
        with self._lock:
            return list(self._hooks.get(event, []))

    def events(self) -> List[str]:
        with self._lock:
            return sorted(self._hooks.keys())


# Process-wide singleton.
_registry_singleton: Optional[HookRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> HookRegistry:
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is None:
            _registry_singleton = HookRegistry()
        return _registry_singleton


# ── Dispatcher ──────────────────────────────────────────────────────────


def _expand_env(s: str, context: Dict[str, Any]) -> str:
    """Expand ${ENV_VAR} and {{context.key}} placeholders in a command string."""
    if not s:
        return s
    out = s
    for var in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", out):
        out = out.replace("${" + var + "}", os.environ.get(var, ""))
    for key in re.findall(r"\{\{([^}]+)\}\}", out):
        val = _resolve_dotted(context, key.strip())
        out = out.replace("{{" + key + "}}", str(val if val is not None else ""))
    return out


def _resolve_dotted(d: Any, path: str) -> Any:
    """Walk a dotted key path through a dict/object tree."""
    cur = d
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def _run_shell(spec: HookSpec, context: Dict[str, Any]) -> HookResult:
    import time as _t
    t0 = _t.perf_counter()
    cmd = _expand_env(spec.cmd or "", context)
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=spec.timeout_s,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        ok = proc.returncode == 0
        vetoed = (not ok) and spec.blocking and spec.event.startswith(("pre_", "on_authority"))
        return HookResult(
            event=spec.event, hook_name=spec.name or cmd[:60],
            ok=ok, vetoed=vetoed, output=out,
            error=None if ok else f"exit={proc.returncode}",
            duration_s=_t.perf_counter() - t0,
        )
    except subprocess.TimeoutExpired:
        return HookResult(
            event=spec.event, hook_name=spec.name or (cmd[:60] if cmd else "shell"),
            ok=False, vetoed=spec.blocking, error="timeout",
            duration_s=_t.perf_counter() - t0,
        )
    except Exception as e:
        return HookResult(
            event=spec.event, hook_name=spec.name or "shell",
            ok=False, vetoed=False,
            error=f"{type(e).__name__}: {e}",
            duration_s=_t.perf_counter() - t0,
        )


def _run_python(spec: HookSpec, context: Dict[str, Any]) -> HookResult:
    import time as _t
    t0 = _t.perf_counter()
    try:
        mod_path, _, attr = (spec.python or "").partition(":")
        if not mod_path or not attr:
            return HookResult(
                event=spec.event, hook_name=spec.name or spec.python or "python",
                ok=False, error="bad python spec; expected 'module:attr'",
                duration_s=_t.perf_counter() - t0,
            )
        import importlib
        module = importlib.import_module(mod_path)
        fn = getattr(module, attr, None)
        if not callable(fn):
            return HookResult(
                event=spec.event, hook_name=spec.name or spec.python,
                ok=False, error=f"{spec.python} is not callable",
                duration_s=_t.perf_counter() - t0,
            )
        result = fn(context) if context is not None else fn()
        # Convention: if the hook returns False or {"ok": False}, veto
        # on blocking events; otherwise pass.
        ok = True
        vetoed = False
        if result is False:
            ok, vetoed = False, bool(spec.blocking)
        elif isinstance(result, dict) and result.get("ok") is False:
            ok, vetoed = False, bool(spec.blocking)
        return HookResult(
            event=spec.event, hook_name=spec.name or spec.python,
            ok=ok, vetoed=vetoed, output=str(result)[:400],
            duration_s=_t.perf_counter() - t0,
        )
    except Exception as e:
        return HookResult(
            event=spec.event, hook_name=spec.name or spec.python or "python",
            ok=False, error=f"{type(e).__name__}: {e}",
            duration_s=_t.perf_counter() - t0,
        )


def fire(event: str, context: Optional[Dict[str, Any]] = None,
         registry: Optional[HookRegistry] = None) -> FireResult:
    """Dispatch every registered hook for `event`. Never raises.

    For blocking hooks, a failed hook on a `pre_*` or `on_authority_*`
    event sets `any_veto=True`. Callers should check `any_veto` and
    abort the protected action (place-order, exit, etc.) when true.
    """
    out = FireResult(event=event)
    if os.environ.get(DISABLE_ENV, "0") == "1":
        return out

    reg = registry or get_registry()
    hooks = reg.list_for(event)
    if not hooks:
        return out

    ctx = dict(context or {})
    for spec in hooks:
        if spec.is_shell():
            result = _run_shell(spec, ctx)
        elif spec.is_python():
            result = _run_python(spec, ctx)
        else:
            result = HookResult(event=event, hook_name=spec.name or "<empty>",
                                ok=False, error="hook has neither cmd nor python")
        out.results.append(result)
        if result.vetoed:
            out.any_veto = True
            logger.warning("[HOOK-VETO] %s blocked by %s: %s",
                           event, result.hook_name, result.error or "non-zero exit")

    return out


# ── Config loader ──────────────────────────────────────────────────────


def load_from_config(
    config: Optional[Dict[str, Any]], registry: Optional[HookRegistry] = None,
) -> HookRegistry:
    """Parse `config['hooks']` and register every entry.

    Silently skips malformed entries (warn logs) so a bad hook config
    can't prevent ACT from starting.
    """
    reg = registry or get_registry()
    if not isinstance(config, dict):
        return reg
    hooks_cfg = config.get("hooks") or {}
    if not isinstance(hooks_cfg, dict):
        return reg
    for event, entries in hooks_cfg.items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            if not isinstance(e, dict):
                continue
            try:
                spec = HookSpec(
                    event=str(event),
                    cmd=e.get("cmd"),
                    python=e.get("python"),
                    blocking=bool(e.get("blocking", False)),
                    timeout_s=float(e.get("timeout_s") or DEFAULT_TIMEOUT_S),
                    name=str(e.get("name") or ""),
                )
                reg.register(spec)
            except Exception as ex:
                logger.warning("hooks: failed to register %s entry: %s", event, ex)
    return reg
