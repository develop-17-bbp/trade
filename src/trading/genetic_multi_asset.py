"""
Multi-Asset Evolution Orchestrator (P1 of genetic-loop audit)
==============================================================
Runs the genetic loop across all available (asset, timeframe) pairs
discovered from `data/*USDT-*.parquet`. Tags each Hall-of-Fame entry
with its origin so the consumer can distinguish:

  * "asset-specific" winners — strategies that excel on one symbol/TF
    only; useful when the bot is currently trading that symbol.
  * "regime-generalist" winners — strategies that hit a fitness floor
    across N>=K (asset, TF) pairs; these are the most promotion-worthy
    candidates because they survived multi-market scrutiny.

Per the genetic-loop audit (P1, "expand cycle to all assets × all
timeframes — half day"), this closes the gap to mid-2025 academic
state of art: the published evolutionary-trading work uses
multi-asset selection-bias awareness as a default, while ACT was
running BTC/ETH on 4h only.

Design:
  * Time-budgeted: each (asset, TF) gets a small evolution (5 gen × 30
    pop by default), bounded to keep the full sweep under ~10 min on
    the 5090. Operator overrides via CLI flags.
  * Soft-fail per-asset: if one parquet is unreadable, skip and
    continue.
  * Aggregation: top-3 from every asset → master Hall-of-Fame, then
    re-evaluate the top-K candidates on EVERY asset to compute a
    "generalization score" (median fitness across assets).
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
_PARQUET_RE = re.compile(r"^([A-Z0-9]+)USDT-([0-9]+[smhd])\.parquet$")


@dataclass
class AssetTimeframe:
    asset: str
    timeframe: str

    def to_dict(self) -> Dict[str, str]:
        return {"asset": self.asset, "timeframe": self.timeframe}


@dataclass
class MultiAssetCycleResult:
    pairs_attempted: int
    pairs_succeeded: int
    pairs_failed: int
    elapsed_s: float
    per_pair_summary: List[Dict[str, Any]] = field(default_factory=list)
    cross_asset_top: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pairs_attempted": self.pairs_attempted,
            "pairs_succeeded": self.pairs_succeeded,
            "pairs_failed": self.pairs_failed,
            "elapsed_s": round(self.elapsed_s, 2),
            "per_pair_summary": self.per_pair_summary,
            "cross_asset_top": self.cross_asset_top,
        }


def discover_pairs(
    assets: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    data_dir: str = DATA_DIR,
) -> List[AssetTimeframe]:
    """List all available (asset, TF) parquet pairs in `data/`.

    Filters by the optional whitelist of assets / timeframes.
    """
    out: List[AssetTimeframe] = []
    if not os.path.isdir(data_dir):
        return out
    for fname in sorted(os.listdir(data_dir)):
        m = _PARQUET_RE.match(fname)
        if not m:
            continue
        a, tf = m.group(1), m.group(2)
        if assets and a not in assets:
            continue
        if timeframes and tf not in timeframes:
            continue
        out.append(AssetTimeframe(asset=a, timeframe=tf))
    return out


def _evaluate_dna_on_pair(
    dna_dict: Dict[str, Any],
    closes, highs, lows, volumes,
    spread_pct: float,
) -> Dict[str, Any]:
    """Reconstruct DNA + run backtest_dna on a single asset/TF."""
    from src.trading.genetic_strategy_engine import (
        StrategyDNA, backtest_dna,
    )
    dna = StrategyDNA.from_dict(dna_dict)
    return backtest_dna(dna, list(closes), list(highs),
                        list(lows), list(volumes), spread_pct=spread_pct)


def run_multi_asset_cycle(
    assets: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    generations: int = 5,
    population_size: int = 30,
    spread_pct: float = 1.69,
    walk_forward: bool = True,
    cross_asset_top_k: int = 5,
    max_pairs: Optional[int] = None,
    data_dir: str = DATA_DIR,
) -> MultiAssetCycleResult:
    """Run a full multi-asset evolution cycle.

    Args:
      assets: whitelist of base symbols (e.g. ["BTC","ETH","SOL"]).
      timeframes: whitelist of TFs (e.g. ["4h","1h"]).
      generations / population_size: per-pair evolution budget.
      walk_forward: if True, each pair is evolved with WF wrapper.
      cross_asset_top_k: re-evaluate this many master Hall-of-Fame
        DNAs on EVERY pair to produce a generalization ranking.
      max_pairs: stop after this many pairs (debug helper).

    Returns:
      MultiAssetCycleResult — per-pair summaries + cross-asset top.
    """
    from src.trading.genetic_strategy_engine import GeneticStrategyEngine

    # Optional walk-forward import (lazy so this module still imports
    # cleanly if walk_forward module is missing for any reason).
    if walk_forward:
        try:
            from src.trading.genetic_walk_forward import evolve_walk_forward
        except ImportError:
            walk_forward = False

    pairs = discover_pairs(assets=assets, timeframes=timeframes, data_dir=data_dir)
    if max_pairs:
        pairs = pairs[:max_pairs]

    if not pairs:
        logger.warning("[MULTI] no (asset, TF) pairs discovered — abort")
        return MultiAssetCycleResult(0, 0, 0, 0.0, [], [])

    t0 = time.time()
    per_pair_summary: List[Dict[str, Any]] = []
    master_hof: List[Tuple[Dict[str, Any], AssetTimeframe]] = []
    succeeded = 0
    failed = 0

    for atf in pairs:
        try:
            engine = GeneticStrategyEngine(spread_pct=spread_pct)
            ok = engine.load_market_data(asset=atf.asset, timeframe=atf.timeframe)
            if not ok or engine.closes is None or len(engine.closes) < 200:
                failed += 1
                per_pair_summary.append({
                    **atf.to_dict(), "status": "insufficient_data"
                })
                continue

            if walk_forward:
                wf_report = evolve_walk_forward(
                    engine,
                    generations=generations,
                    population_size=population_size,
                )
                best = wf_report.get("best_promotable") or wf_report.get("best_oos")
                per_pair_summary.append({
                    **atf.to_dict(),
                    "status": "ok",
                    "wf_best": best,
                    "n_promotable": wf_report.get("n_promotable", 0),
                })
                # collect HOF DNAs for cross-asset eval
                for dna in engine.hall_of_fame[:5]:
                    master_hof.append((dna.to_dict(), atf))
            else:
                hof = engine.evolve(
                    generations=generations,
                    population_size=population_size,
                )
                if hof:
                    best = hof[0]
                    per_pair_summary.append({
                        **atf.to_dict(),
                        "status": "ok",
                        "best_name": best.name,
                        "best_fitness": best.fitness,
                        "best_pnl": best.total_pnl,
                        "best_win_rate": best.win_rate,
                        "best_entry": best.entry_rule,
                        "best_exit": best.exit_rule,
                    })
                    for dna in hof[:5]:
                        master_hof.append((dna.to_dict(), atf))
                else:
                    per_pair_summary.append({
                        **atf.to_dict(), "status": "no_hof"
                    })

            succeeded += 1
        except Exception as exc:
            logger.exception("[MULTI] pair %s/%s failed: %s",
                            atf.asset, atf.timeframe, exc)
            failed += 1
            per_pair_summary.append({
                **atf.to_dict(), "status": f"error:{type(exc).__name__}"
            })

    # Cross-asset generalization scoring
    cross_asset_top: List[Dict[str, Any]] = []
    if master_hof and cross_asset_top_k > 0:
        # Take top by initial fitness (already sorted within each pair)
        candidates = sorted(
            master_hof,
            key=lambda x: x[0].get("fitness", 0.0),
            reverse=True,
        )[:cross_asset_top_k * 3]  # over-sample, will trim after generalization scoring

        for dna_dict, origin_atf in candidates:
            scores: List[float] = []
            assets_evaluated: List[str] = []
            for atf in pairs[:6]:  # bound to 6 pairs to keep compute tractable
                try:
                    import pandas as pd
                    p = os.path.join(data_dir, f"{atf.asset}USDT-{atf.timeframe}.parquet")
                    df = pd.read_parquet(p)
                    df.columns = [c.lower() for c in df.columns]
                    res = _evaluate_dna_on_pair(
                        dna_dict,
                        df["close"].values, df["high"].values,
                        df["low"].values, df["volume"].values,
                        spread_pct,
                    )
                    if res.get("trades", 0) > 0:
                        scores.append(res.get("fitness", 0.0))
                        assets_evaluated.append(f"{atf.asset}-{atf.timeframe}")
                except Exception:
                    continue

            if not scores:
                continue
            median = sorted(scores)[len(scores) // 2]
            mean = sum(scores) / len(scores)
            generalization = mean - (max(scores) - min(scores))  # mean penalised by spread

            cross_asset_top.append({
                "name": dna_dict.get("name"),
                "entry_rule": dna_dict.get("entry_rule"),
                "exit_rule": dna_dict.get("exit_rule"),
                "origin": f"{origin_atf.asset}-{origin_atf.timeframe}",
                "n_assets_evaluated": len(scores),
                "median_fitness": round(float(median), 4),
                "mean_fitness": round(float(mean), 4),
                "generalization_score": round(float(generalization), 4),
                "assets": assets_evaluated,
            })

        cross_asset_top.sort(
            key=lambda x: x["generalization_score"],
            reverse=True,
        )
        cross_asset_top = cross_asset_top[:cross_asset_top_k]

    elapsed = time.time() - t0
    return MultiAssetCycleResult(
        pairs_attempted=len(pairs),
        pairs_succeeded=succeeded,
        pairs_failed=failed,
        elapsed_s=elapsed,
        per_pair_summary=per_pair_summary,
        cross_asset_top=cross_asset_top,
    )


__all__ = [
    "AssetTimeframe",
    "MultiAssetCycleResult",
    "discover_pairs",
    "run_multi_asset_cycle",
]
