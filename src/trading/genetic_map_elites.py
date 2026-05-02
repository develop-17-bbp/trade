"""
MAP-Elites Archive for Strategy Evolution (P2 of genetic-loop audit)
=====================================================================
MAP-Elites (Mouret & Clune 2015) replaces a flat fitness-ranked
Hall-of-Fame with a multi-dimensional grid keyed by *behavioral
descriptors*. Each cell holds the best-performing strategy with that
behavior. The result: an archive that explicitly preserves diversity
of trading styles, not just a clutter of slight variations on the
single best winner.

For ACT, the descriptors are:

  1. win_rate_bin: 5 bins (0-0.4, 0.4-0.5, 0.5-0.6, 0.6-0.75, 0.75+)
  2. avg_bars_bin: 4 bins (≤5, 6-12, 13-25, 26+)
  3. entry_family: 5 families derived from entry_rule
  4. pnl_volatility_bin: 3 bins (low, med, high — measured from
     metrics.std_pnl if present, else fallback to sharpe inverse)

Total cells: 5 × 4 × 5 × 3 = 300 max niches. Real population
typically fills 20-60 of these, giving the operator a much richer
view than a flat top-10 list.

Anti-overfit design:
  * A new DNA only displaces the cell incumbent if its fitness is
    strictly higher (no ties → keep older entry, more battle-tested).
  * Every entry stores the fitness *that earned it admission* and
    a generation timestamp; archives can be aged-out.
  * Promotion to live trading should require entries from K cells
    (regime-diverse winners), not the global champion.

Usage:
    archive = MAPElitesArchive()
    archive.update(engine.population)
    diverse_top = archive.diverse_top_k(k=10)
    summary = archive.summary()
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


_ENTRY_FAMILIES: Dict[str, str] = {
    # trend
    "ema_cross": "trend", "trend_strength": "trend", "ema_bounce": "trend",
    "multi_ma_align": "trend", "ema400_two_candle": "trend",
    # mean reversion
    "rsi_oversold_bounce": "mean_rev", "bb_lower_touch": "mean_rev",
    "stoch_reversal": "mean_rev", "regime_mean_reversion": "mean_rev",
    "double_bottom_rejection": "mean_rev",
    # breakout
    "breakout_volume": "breakout", "volatility_squeeze": "breakout",
    "three_candle_breakout": "breakout", "three_candle_retracement": "breakout",
    "liquidity_sweep_reversal": "breakout",
    # momentum
    "momentum_surge": "momentum",
    # macro
    "fear_greed_contrarian": "macro", "usd_weakness_long": "macro",
    "macro_risk_off_skip": "macro", "funding_rate_contrarian": "macro",
    "vix_spike_pause": "macro", "etf_inflow_confirmation": "macro",
    "defi_tvl_expansion": "macro", "pre_event_skip": "macro",
}

_FAMILY_LIST = ["trend", "mean_rev", "breakout", "momentum", "macro"]


def _entry_family(entry_rule: str) -> str:
    return _ENTRY_FAMILIES.get(entry_rule, "macro")


def _win_rate_bin(wr: float) -> int:
    if wr < 0.40: return 0
    if wr < 0.50: return 1
    if wr < 0.60: return 2
    if wr < 0.75: return 3
    return 4


def _avg_bars_bin(avg_bars: float) -> int:
    if avg_bars <= 5: return 0
    if avg_bars <= 12: return 1
    if avg_bars <= 25: return 2
    return 3


def _pnl_volatility_bin(metrics: Dict[str, Any], sharpe: float) -> int:
    std_pnl = metrics.get("std_profit_pct") if metrics else None
    if std_pnl is not None:
        try:
            std_pnl = float(std_pnl)
            if std_pnl < 1.5: return 0
            if std_pnl < 4.0: return 1
            return 2
        except (ValueError, TypeError):
            pass
    # Fallback: sharpe-based proxy (high sharpe → low vol).
    if sharpe >= 1.5: return 0
    if sharpe >= 0.5: return 1
    return 2


def behavior_cell(dna: Any) -> Tuple[int, int, str, int]:
    """Compute the behavioral cell key for a StrategyDNA."""
    metrics = getattr(dna, "metrics", {}) or {}
    avg_bars = metrics.get("avg_duration_min", None)
    if avg_bars is not None and avg_bars > 0:
        # Convert minutes to "bars" assuming 4h = 240 min default;
        # the engine logs duration_min = bars * 240 on 4h data.
        avg_bars = float(avg_bars) / 240.0
    else:
        avg_bars = float(getattr(dna, "trades", 1) or 1)
        # rough: total_pnl / trades doesn't give bars; default to 10
        avg_bars = 10.0
    return (
        _win_rate_bin(float(getattr(dna, "win_rate", 0.0) or 0.0)),
        _avg_bars_bin(avg_bars),
        _entry_family(str(getattr(dna, "entry_rule", "ema_cross"))),
        _pnl_volatility_bin(metrics, float(getattr(dna, "sharpe", 0.0) or 0.0)),
    )


@dataclass
class CellEntry:
    dna_dict: Dict[str, Any]
    fitness: float
    generation: int
    win_rate: float
    total_pnl: float
    sharpe: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dna_name": self.dna_dict.get("name"),
            "entry_rule": self.dna_dict.get("entry_rule"),
            "exit_rule": self.dna_dict.get("exit_rule"),
            "fitness": round(float(self.fitness), 4),
            "win_rate": round(float(self.win_rate), 4),
            "total_pnl": round(float(self.total_pnl), 2),
            "sharpe": round(float(self.sharpe), 3),
            "generation": int(self.generation),
        }


class MAPElitesArchive:
    """A behavioral grid of best-by-cell strategies."""

    def __init__(self) -> None:
        self.cells: Dict[Tuple[int, int, str, int], CellEntry] = {}
        self._update_count: int = 0

    @property
    def n_filled(self) -> int:
        return len(self.cells)

    def update_one(self, dna: Any) -> bool:
        """Try inserting `dna` into the archive. Returns True if it
        replaced or filled a cell."""
        try:
            fitness = float(getattr(dna, "fitness", 0.0) or 0.0)
        except (TypeError, ValueError):
            return False
        if fitness <= 0:
            return False
        if int(getattr(dna, "trades", 0) or 0) < 3:
            return False  # ignore non-trading DNAs

        key = behavior_cell(dna)
        existing = self.cells.get(key)
        # Strict > so older entries with equal fitness aren't displaced.
        if existing is None or fitness > existing.fitness:
            self.cells[key] = CellEntry(
                dna_dict=dna.to_dict() if hasattr(dna, "to_dict") else dict(dna.__dict__),
                fitness=fitness,
                generation=int(getattr(dna, "generation", 0) or 0),
                win_rate=float(getattr(dna, "win_rate", 0.0) or 0.0),
                total_pnl=float(getattr(dna, "total_pnl", 0.0) or 0.0),
                sharpe=float(getattr(dna, "sharpe", 0.0) or 0.0),
            )
            self._update_count += 1
            return True
        return False

    def update(self, population: Iterable[Any]) -> Dict[str, Any]:
        """Apply update_one across a population. Returns stats."""
        n_inserted = 0
        n_attempted = 0
        for dna in population:
            n_attempted += 1
            if self.update_one(dna):
                n_inserted += 1
        return {
            "n_attempted": n_attempted,
            "n_inserted": n_inserted,
            "n_filled_cells": self.n_filled,
        }

    def diverse_top_k(self, k: int = 10) -> List[Dict[str, Any]]:
        """Return the top-K cells by fitness, prioritizing diversity:
        no two consecutive entries share the same entry_family.

        Falls back to plain fitness ordering if diversity-aware
        selection runs out of options.
        """
        if not self.cells:
            return []
        sorted_entries = sorted(
            self.cells.items(),
            key=lambda kv: kv[1].fitness,
            reverse=True,
        )
        result: List[Dict[str, Any]] = []
        used_families: List[str] = []
        # First pass: enforce diversity (no two same-family in a row,
        # max half from one family).
        for _, entry in sorted_entries:
            family = entry.dna_dict.get("entry_rule", "ema_cross")
            family = _entry_family(family)
            if used_families and used_families[-1] == family:
                continue
            n_same = sum(1 for f in used_families if f == family)
            if n_same >= max(1, k // 2):
                continue
            result.append({**entry.to_dict(), "entry_family": family})
            used_families.append(family)
            if len(result) >= k:
                return result
        # Second pass: top-up with fitness-only if still short.
        seen = {e["dna_name"] for e in result}
        for _, entry in sorted_entries:
            if entry.dna_dict.get("name") in seen:
                continue
            family = _entry_family(entry.dna_dict.get("entry_rule", "ema_cross"))
            result.append({**entry.to_dict(), "entry_family": family})
            if len(result) >= k:
                break
        return result

    def summary(self) -> Dict[str, Any]:
        """Stats about archive coverage + per-family counts."""
        family_counts: Dict[str, int] = {f: 0 for f in _FAMILY_LIST}
        wr_counts: Dict[int, int] = {i: 0 for i in range(5)}
        for key in self.cells:
            wr_bin, _, family, _ = key
            wr_counts[wr_bin] += 1
            family_counts[family] = family_counts.get(family, 0) + 1
        max_cells = 5 * 4 * 5 * 3
        return {
            "n_filled": self.n_filled,
            "max_cells": max_cells,
            "coverage_pct": round(100 * self.n_filled / max_cells, 1),
            "n_updates": self._update_count,
            "by_entry_family": family_counts,
            "by_win_rate_bin": wr_counts,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializable form (for log persistence)."""
        return {
            "summary": self.summary(),
            "cells": {
                f"{k[0]}/{k[1]}/{k[2]}/{k[3]}": v.to_dict()
                for k, v in self.cells.items()
            },
        }


__all__ = [
    "MAPElitesArchive",
    "CellEntry",
    "behavior_cell",
]
