"""
Skills registry — operator-facing named workflows (C6 of the plan).

Claude Code has `.claude/skills/<name>/` directories, each containing a
skill.yaml metadata file + an action (shell / python). Users invoke via
`/skill-name`. ACT mirrors that pattern here.

Why skills matter for ACT:
  * An operator shouldn't have to remember the 4-tool-call sequence the
    agentic loop would use to "flatten everything and explain why" —
    they should type `/emergency-flatten` and get the safe, audited
    workflow.
  * Skills encapsulate both the workflow AND the safety policy
    (confirm-before-acting, audit-log required, reversibility flag)
    that shouldn't live in the LLM's hot path.
  * The agentic loop's LLM CAN also invoke skills as a higher-tier
    tool — `run_skill(name, args)` in trade_tools (C7/C8 wiring).

Design:
  * One directory per skill under `skills/`; each has:
      - skill.yaml   — metadata (name, description, policy)
      - action.py    — one function `run(args: dict) -> SkillResult`
  * Registry loads all directories at startup (and can reload on demand).
  * Policy fields enforced at dispatch:
      - requires_confirmation: bool (the operator must type confirm=true
        in args; LLM-driven invocations get blocked unless explicit)
      - reversible: bool (non-reversible skills log to audit at INFO)
      - tags: list[str] (filter, e.g. ["destructive", "ops", "ml"])
  * Skills never raise — errors come back in SkillResult.error.

Not in scope here:
  * Hooks — skills fire events, but the hook dispatcher lives in C8.
  * MCP — skills can be exposed as MCP tools in C7, but this module
    only owns local registration.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import yaml  # ACT already depends on pyyaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


# Default skills directory — configurable via env.
DEFAULT_SKILLS_DIR = os.getenv(
    "ACT_SKILLS_DIR",
    str(Path(__file__).resolve().parents[2] / "skills"),
)


@dataclass
class Skill:
    """One loaded skill — metadata + callable.

    Fields mirror Claude Code's skill.yaml shape where possible.
    """
    name: str
    description: str
    handler: Callable[[Dict[str, Any]], "SkillResult"]
    requires_confirmation: bool = False
    reversible: bool = True
    tags: List[str] = field(default_factory=list)
    source_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "requires_confirmation": self.requires_confirmation,
            "reversible": self.reversible,
            "tags": list(self.tags),
            "source_path": self.source_path,
        }


@dataclass
class SkillResult:
    """Structured output from a skill invocation."""
    ok: bool
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "message": self.message[:1000],
            "data": dict(self.data),
            "error": self.error,
        }


# ── Registry ────────────────────────────────────────────────────────────


class SkillRegistry:
    """Name-keyed store of loaded skills. Thread-safe."""

    def __init__(self):
        self._skills: Dict[str, Skill] = {}
        self._lock = threading.Lock()

    def register(self, skill: Skill) -> None:
        with self._lock:
            # Last-writer wins — hot-reload replaces.
            self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        with self._lock:
            return self._skills.get(name)

    def list_names(self) -> List[str]:
        with self._lock:
            return sorted(self._skills.keys())

    def list_skills(self, tag: Optional[str] = None) -> List[Skill]:
        with self._lock:
            skills = list(self._skills.values())
        if tag:
            skills = [s for s in skills if tag in s.tags]
        return sorted(skills, key=lambda s: s.name)

    def dispatch(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        *,
        invoker: str = "operator",
    ) -> SkillResult:
        """Invoke a skill. Enforces requires_confirmation policy.

        `invoker` tags the caller for audit — 'operator' invocations from
        a CLI / skill-call are trusted; 'llm' invocations from the agentic
        loop must pass `confirm=True` in args to run any skill with
        requires_confirmation=True. This keeps destructive skills out of
        the LLM's reach without an operator seeing and approving them.
        """
        args = dict(args or {})
        skill = self.get(name)
        if skill is None:
            return SkillResult(ok=False, error=f"unknown skill {name!r}")

        if skill.requires_confirmation and invoker == "llm" and not args.get("confirm", False):
            return SkillResult(
                ok=False,
                error=f"skill {name!r} requires confirmation; llm-invoked without confirm=True",
            )

        try:
            result = skill.handler(args)
        except Exception as e:
            logger.exception("skill %s raised", name)
            return SkillResult(ok=False, error=f"{type(e).__name__}: {e}")

        if not isinstance(result, SkillResult):
            # Tolerate skills that forget to wrap output.
            return SkillResult(ok=True, message=str(result)[:1000])

        # Audit log on non-reversible skills.
        if not skill.reversible:
            logger.info("[SKILL-AUDIT] %s invoked by %s, ok=%s", name, invoker, result.ok)

        return result


# ── Directory loader ────────────────────────────────────────────────────


def load_skills_from_dir(
    dir_path: str,
    registry: Optional[SkillRegistry] = None,
) -> SkillRegistry:
    """Walk `dir_path`, load any subdirectory with skill.yaml + action.py.

    action.py must expose a `run(args)` callable that returns a
    SkillResult (or a plain string, which we wrap).
    """
    registry = registry or SkillRegistry()
    base = Path(dir_path)
    if not base.exists():
        logger.debug("skills dir %s does not exist; skipping", dir_path)
        return registry

    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        yaml_path = entry / "skill.yaml"
        action_path = entry / "action.py"
        if not yaml_path.exists() or not action_path.exists():
            continue
        try:
            skill = _load_one(yaml_path, action_path)
        except Exception as e:
            logger.warning("failed to load skill at %s: %s", entry, e)
            continue
        if skill is not None:
            registry.register(skill)
    return registry


def _load_one(yaml_path: Path, action_path: Path) -> Optional[Skill]:
    if not _HAS_YAML:
        logger.warning("pyyaml unavailable — cannot load skill %s", yaml_path.parent.name)
        return None
    with yaml_path.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    name = str(meta.get("name") or yaml_path.parent.name).strip()
    description = str(meta.get("description") or "").strip()
    if not name:
        return None

    # Load action.py as an anonymous module so we don't pollute sys.modules
    # with per-skill names.
    spec = importlib.util.spec_from_file_location(f"act_skill_{name}", str(action_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handler = getattr(module, "run", None)
    if not callable(handler):
        logger.warning("skill %s: action.py has no callable run(args)", name)
        return None

    return Skill(
        name=name,
        description=description,
        handler=handler,
        requires_confirmation=bool(meta.get("requires_confirmation", False)),
        reversible=bool(meta.get("reversible", True)),
        tags=list(meta.get("tags") or []),
        source_path=str(yaml_path.parent),
    )


# ── Process-wide singleton ──────────────────────────────────────────────

_registry_singleton: Optional[SkillRegistry] = None
_registry_lock = threading.Lock()


def get_registry(refresh: bool = False) -> SkillRegistry:
    """Get the process-wide registry, loading `DEFAULT_SKILLS_DIR` on
    first call (or when refresh=True)."""
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is None or refresh:
            _registry_singleton = load_skills_from_dir(DEFAULT_SKILLS_DIR)
        return _registry_singleton
