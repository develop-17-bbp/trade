"""Per-asset persistent context thread (Ollama KV cache reuse).

Today ACT rebuilds the full system prompt every tick (~22K chars +
30+ context streams). The model recomputes attention over all of it
each call. With Ollama's `context` parameter we can thread the KV
cache across calls so subsequent ticks only pay the marginal-token
cost of NEW evidence, not the full prompt re-compute.

Tradeoff:
  * Pro:  ~5-10× faster per-tick LLM call after the first call
          (only new tokens get attended); model "remembers" prior
          ticks within a session
  * Con:  context grows unbounded if not periodically reset; model
          can drift if working memory becomes stale
  * Mitigation: explicit consolidation every N ticks (default 20),
          which compresses history into a summary and reseeds the
          thread

Architecture:

    Tick 1 (cold start):
      send: SYSTEM_PROMPT + EVIDENCE_DOC_FULL → Ollama
      receive: response + context_array (KV state)
      cache: per-asset context_array

    Tick 2..N (warm):
      send: cached_context + EVIDENCE_DELTA_ONLY → Ollama
      receive: response + updated_context_array
      cache: updated context

    Tick consolidation_every (default 20):
      send: cached_context + "summarize the last 20 ticks" → Ollama
      receive: summary + reset_context_array
      cache: reset (next tick is a "cold start" with summary as seed)

Anti-drift design:
  * Hard cap on context size (default 100K tokens) — auto-reset above
  * Consolidation forces the model to compress, preventing
    "remembered" stale facts from corrupting fresh decisions
  * Per-asset isolation — BTC's thread doesn't pollute ETH's
  * Fallback: if Ollama context API errors, fall back to full
    dynamic prompting (the existing path) so brain never blocks

Activation:
  ACT_PERSISTENT_CONTEXT unset / "0"  → dormant; existing dynamic
                                         prompting path runs
  ACT_PERSISTENT_CONTEXT = "1"        → use persistent KV thread
                                         per asset; consolidate every
                                         ACT_CONTEXT_CONSOLIDATE_EVERY
                                         ticks (default 20)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONSOLIDATE_EVERY = 20
DEFAULT_MAX_CONTEXT_TOKENS = 100_000


@dataclass
class AssetContextThread:
    """Per-asset persistent KV thread state."""
    asset: str
    context_array: Optional[List[int]] = None    # Ollama KV cache token IDs
    n_calls_since_seed: int = 0
    last_seeded_at: float = 0.0
    last_evidence_hash: str = ""
    estimated_token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "has_context": self.context_array is not None,
            "n_calls_since_seed": int(self.n_calls_since_seed),
            "last_seeded_age_s": round(time.time() - self.last_seeded_at, 1)
                if self.last_seeded_at else None,
            "estimated_token_count": int(self.estimated_token_count),
        }


def is_enabled() -> bool:
    val = (os.environ.get("ACT_PERSISTENT_CONTEXT") or "").strip().lower()
    return val in ("1", "true", "on")


def _consolidate_every() -> int:
    try:
        return max(5, int(os.environ.get("ACT_CONTEXT_CONSOLIDATE_EVERY") or
                          DEFAULT_CONSOLIDATE_EVERY))
    except Exception:
        return DEFAULT_CONSOLIDATE_EVERY


def _max_tokens() -> int:
    try:
        return max(8_000, int(os.environ.get("ACT_CONTEXT_MAX_TOKENS") or
                              DEFAULT_MAX_CONTEXT_TOKENS))
    except Exception:
        return DEFAULT_MAX_CONTEXT_TOKENS


class PersistentContextManager:
    """Per-asset KV-thread manager. Thread-safe."""

    def __init__(self) -> None:
        self._threads: Dict[str, AssetContextThread] = {}
        self._lock = threading.Lock()

    def get(self, asset: str) -> AssetContextThread:
        asset = str(asset).upper()
        with self._lock:
            if asset not in self._threads:
                self._threads[asset] = AssetContextThread(asset=asset)
            return self._threads[asset]

    def needs_seeding(self, asset: str) -> bool:
        """True when the thread needs a fresh full-prompt seed:
          * first call ever for this asset
          * consolidation interval hit (n_calls >= consolidate_every)
          * context grew past max_tokens
          * stale (last seed > 10 minutes ago)
        """
        t = self.get(asset)
        if t.context_array is None:
            return True
        if t.n_calls_since_seed >= _consolidate_every():
            return True
        if t.estimated_token_count >= _max_tokens():
            return True
        if t.last_seeded_at and (time.time() - t.last_seeded_at) > 600:
            return True
        return False

    def update(self, asset: str, context_array: Optional[List[int]],
               token_delta: int = 0) -> None:
        """Called after an Ollama generate. Updates the thread state."""
        if context_array is None:
            return
        t = self.get(asset)
        with self._lock:
            t.context_array = list(context_array)
            t.n_calls_since_seed += 1
            t.estimated_token_count += max(0, int(token_delta))

    def reset(self, asset: str) -> None:
        """Force consolidation: drops the thread so next call seeds fresh."""
        t = self.get(asset)
        with self._lock:
            t.context_array = None
            t.n_calls_since_seed = 0
            t.last_seeded_at = time.time()
            t.estimated_token_count = 0
            t.last_evidence_hash = ""

    def mark_seeded(self, asset: str, evidence_hash: str = "") -> None:
        """Mark that a fresh full-prompt seed just happened."""
        t = self.get(asset)
        with self._lock:
            t.last_seeded_at = time.time()
            t.n_calls_since_seed = 0
            t.last_evidence_hash = evidence_hash
            t.estimated_token_count = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": is_enabled(),
                "consolidate_every": _consolidate_every(),
                "max_context_tokens": _max_tokens(),
                "threads": [t.to_dict() for t in self._threads.values()],
            }


# Module-level singleton
_singleton: Optional[PersistentContextManager] = None


def get_manager() -> PersistentContextManager:
    global _singleton
    if _singleton is None:
        _singleton = PersistentContextManager()
    return _singleton


def build_evidence_delta(asset: str, full_evidence: str,
                          last_evidence: str) -> str:
    """Compute the minimal new-evidence delta between two evidence
    documents. When seeding, the full evidence is used. After seeding
    only the delta is sent.

    Naive line-level diff — sufficient for tick_state's structured
    output where each section is a single line.
    """
    if not last_evidence:
        return full_evidence
    last_lines = set(last_evidence.split("\n"))
    new_lines = full_evidence.split("\n")
    delta = [ln for ln in new_lines if ln not in last_lines]
    if not delta:
        return "(no material change since last tick)"
    return "\n".join(delta)
