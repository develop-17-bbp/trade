"""Model guard -- forbid specific Ollama models from being requested.

When the operator wants to fully retire a model (e.g. switch from
`dense_r1` profile's `deepseek-r1:7b` + `deepseek-r1:32b` to
`moe_agentic`'s qwen pair), legacy code paths or stale config can
still emit requests for the old models, causing Ollama to reload
them on top of the pinned new pair and trigger eviction storms.

This module is the single circuit-breaker:

  ACT_FORBID_MODELS=deepseek-r1:7b,deepseek-r1:32b,deepseek-r1

Any call site that asks `is_forbidden(model)` before sending a
request will refuse and surface the rejection -- so a stale
hardcode reaches a hard error rather than silently fighting the
operator's profile choice. The forbid list is read fresh on every
call, so an in-process `os.environ` flip takes effect immediately.

Comparison is case-insensitive substring match against either the
full tag (`deepseek-r1:7b`) or the family head (`deepseek-r1`), so
forbidding the family head also blocks all sized variants.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


FORBID_ENV = "ACT_FORBID_MODELS"


def _read_forbid_list() -> List[str]:
    raw = os.environ.get(FORBID_ENV, "").strip()
    if not raw:
        return []
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


_KNOWN_PROFILE_NAMES = {
    "dense_r1", "moe_agentic", "qwen3_r1", "hybrid", "devstral_qwen3coder",
}


def resolve_pinned_analyst() -> Optional[str]:
    """Last-resort: ask dual_brain what analyst model the active profile
    wants. Useful as the final link in env-driven resolution chains so
    a fully-empty env (no OLLAMA_REMOTE_MODEL, no ACT_*_MODEL) still
    produces a real model name instead of a `no_model_resolved` error.

    Imports lazily to avoid a circular import (dual_brain may use
    model_guard at module-load time).
    """
    try:
        from src.ai.dual_brain import _resolve_profile  # noqa: PLC0415
        profile = _resolve_profile()
        analyst = (profile.get("analyst_model") or "").strip()
        if analyst and is_well_formed_model_tag(analyst) and not is_forbidden(analyst):
            return analyst
        scanner = (profile.get("scanner_model") or "").strip()
        if scanner and is_well_formed_model_tag(scanner) and not is_forbidden(scanner):
            return scanner
    except Exception:
        pass
    return None


def is_well_formed_model_tag(model_id: Optional[str]) -> bool:
    """True iff `model_id` looks like a real Ollama model tag.

    Ollama tags follow the `family:size` shape (e.g. `qwen3-coder:30b`,
    `deepseek-r1:7b`). A bare profile name like `moe_agentic` is a
    common operator typo (`setx OLLAMA_REMOTE_MODEL moe_agentic`
    instead of the actual tag) and Ollama 404s it -- producing
    `All Ollama endpoints failed for model moe_agentic` and silent
    parse_failures downstream.

    Heuristics:
      * has at least one colon (every Ollama tag does)
      * not equal to a known dual_brain profile name
    """
    if not model_id:
        return False
    s = str(model_id).strip()
    if not s:
        return False
    if s.lower() in _KNOWN_PROFILE_NAMES:
        return False
    if ":" not in s:
        return False
    return True


def is_forbidden(model_id: Optional[str]) -> bool:
    """Return True if `model_id` matches any entry in ACT_FORBID_MODELS.

    A forbid entry matches a model_id when either:
      * the entry equals the full tag (`deepseek-r1:7b`),
      * the entry is a substring of the tag's family head
        (entry `deepseek-r1` matches `deepseek-r1:7b` and `deepseek-r1:32b`),
      * the entry is a substring of the full tag (catches arbitrary
        shapes future operators might choose).

    Empty / None / "" input is never forbidden -- the caller is
    presumably about to apply its own resolution chain.
    """
    if not model_id:
        return False
    target = str(model_id).strip().lower()
    if not target:
        return False
    for entry in _read_forbid_list():
        if entry == target:
            return True
        # Family-head match: entry "deepseek-r1" blocks
        # "deepseek-r1:7b", "deepseek-r1:32b", etc.
        head = target.split(":")[0]
        if entry == head:
            return True
        if entry in target:
            return True
    return False


def resolve_safe_model(
    candidates: Sequence[Optional[str]],
    *,
    forbid_list: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Walk `candidates` in order and return the first non-empty,
    non-forbidden model. Returns None if every candidate is empty or
    forbidden -- caller decides how to surface that.

    Used by call sites that have a priority chain (env override →
    config → hardcoded default) and want a single line that picks
    the first viable name.
    """
    if forbid_list is None:
        forbid_set = set(_read_forbid_list())
    else:
        forbid_set = {str(x).strip().lower() for x in forbid_list if str(x).strip()}

    def _is_blocked(name: str) -> bool:
        if not forbid_set:
            return False
        nl = name.strip().lower()
        if nl in forbid_set:
            return True
        head = nl.split(":")[0]
        if head in forbid_set:
            return True
        for entry in forbid_set:
            if entry in nl:
                return True
        return False

    for c in candidates:
        if not c:
            continue
        c_str = str(c).strip()
        if not c_str:
            continue
        if not is_well_formed_model_tag(c_str):
            logger.warning(
                "model_guard: skipping malformed candidate %r "
                "(not a valid Ollama tag -- expected family:size like "
                "'qwen3-coder:30b'; common typo: profile name in env)",
                c_str,
            )
            continue
        if _is_blocked(c_str):
            logger.warning(
                "model_guard: skipping forbidden candidate %r "
                "(matches %s=%r)",
                c_str, FORBID_ENV, os.environ.get(FORBID_ENV, ""),
            )
            continue
        return c_str
    return None
