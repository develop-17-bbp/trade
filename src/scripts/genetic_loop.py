"""
Genetic Strategy Evolution Loop
================================
Runs on a schedule to evolve the population of DNA strategies.
Designed for RTX 5090 with large population sizes.

Modes (genetic-loop audit P0-P3 wired in):
  --walk-forward          P0: 80/15/5 train/val/test split, fitness=OOS Sharpe
  --dsr-gate              P0: Deflated Sharpe fitness gate when N>20 trials
  --cma-es                P1: CMA-ES local-search refine on top-K HOF
  --llm-mutate N          P1: N LLM mutation calls per cycle (Analyst proposes)
  --multi-asset           P1: evolve across ALL discovered (asset,TF) pairs
  --map-elites            P2: build MAP-Elites archive (behavioral grid)
  --surrogate             P2: surrogate pre-filter (cheap fitness predictor)
  --drift-immigrants      P3: detect regime drift, inject random immigrants
  --grammatical-evolution P3: parallel GE branch + cross-compare to NSGA-II
  --mode all              All of the above

Default (no mode flags): BTC + ETH on 4h, vanilla evolve().
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [GENETIC] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _load_spread_pct() -> float:
    try:
        import yaml
        path = PROJECT_ROOT / "config.yaml"
        if path.exists():
            cfg = yaml.safe_load(path.read_text()) or {}
            for ex in (cfg.get("exchanges") or []):
                if ex.get("name") == "robinhood":
                    return float(ex.get("round_trip_spread_pct", 0.0) or 0.0)
    except Exception:
        pass
    return 0.0


def _persist_cycle_report(report: dict, name: str = "genetic_cycle"):
    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_latest.json"
    try:
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Failed to persist cycle report: %s", exc)
    _merge_into_adaptation_context(report)


def _merge_into_adaptation_context(report: dict):
    """Surface audit-module outputs into data/adaptation_context.json so the
    LLM seed prompt + LLM tools can read WF/DSR/MAP-Elites/CMA-ES/drift state.

    Picks the FIRST asset's per-asset block that has each subkey (BTC by
    default, ETH fallback) — the LLM cares about "is the new validation
    layer firing", not which asset the report came from.
    """
    ctx_path = PROJECT_ROOT / "data" / "adaptation_context.json"
    try:
        ctx = {}
        if ctx_path.exists():
            try:
                ctx = json.loads(ctx_path.read_text())
            except Exception:
                ctx = {}

        per_asset = report.get("per_asset", {}) or {}
        # Stable order: prefer BTC, then ETH, then anything else.
        ordered = sorted(per_asset.items(),
                         key=lambda kv: (kv[0] != "BTC", kv[0] != "ETH", kv[0]))

        def _first_with(key):
            for _, block in ordered:
                v = block.get(key)
                if v:
                    return v
            return None

        audit = {
            "generated_at": report.get("completed_at"),
            "flags": report.get("flags", {}),
            "walk_forward": _first_with("walk_forward"),
            "best_oos": _first_with("best_oos"),
            "dsr_gate": _first_with("dsr_gate"),
            "cma_es": _first_with("cma_es"),
            "map_elites": _first_with("map_elites"),
            "drift_signal": _first_with("drift_signal"),
            "drift_immigrants": _first_with("drift_immigrants"),
            "llm_mutate": _first_with("llm_mutate"),
            "grammatical_evolution": _first_with("grammatical_evolution"),
            "multi_asset_summary": report.get("multi_asset"),
        }
        # Drop empty keys so the LLM doesn't see zombie nulls.
        audit = {k: v for k, v in audit.items() if v}
        if audit:
            ctx["genetic_audit"] = audit
        ctx_path.write_text(json.dumps(ctx, indent=2, default=str))
    except Exception as exc:
        logger.debug("Failed to merge audit into adaptation_context: %s", exc)


def run_evolution_cycle(
    population_size: int,
    generations: int,
    spread_pct: float,
    walk_forward: bool = False,
    dsr_gate: bool = False,
    cma_es: bool = False,
    llm_mutate: int = 0,
    multi_asset: bool = False,
    map_elites: bool = False,
    surrogate: bool = False,
    drift_immigrants: bool = False,
    grammatical_evolution: bool = False,
):
    """Run one full genetic evolution cycle.

    Routes to the multi-asset orchestrator if `multi_asset=True`.
    Otherwise runs BTC + ETH on 4h with the requested feature flags.
    """
    cycle_report = {
        "started_at": time.time(),
        "flags": {
            "walk_forward": walk_forward, "dsr_gate": dsr_gate,
            "cma_es": cma_es, "llm_mutate": llm_mutate,
            "multi_asset": multi_asset, "map_elites": map_elites,
            "surrogate": surrogate, "drift_immigrants": drift_immigrants,
            "grammatical_evolution": grammatical_evolution,
        },
        "population_size": population_size,
        "generations": generations,
        "per_asset": {},
    }

    # ── Multi-asset path ────────────────────────────────────────────────
    if multi_asset:
        try:
            from src.trading.genetic_multi_asset import run_multi_asset_cycle
            result = run_multi_asset_cycle(
                generations=generations,
                population_size=population_size,
                spread_pct=spread_pct or 1.69,
                walk_forward=walk_forward,
            )
            cycle_report["multi_asset"] = result.to_dict()
            logger.info("[MULTI] %d/%d pairs ok in %.1fs",
                       result.pairs_succeeded, result.pairs_attempted,
                       result.elapsed_s)
        except Exception as exc:
            logger.exception("Multi-asset cycle failed: %s", exc)
        cycle_report["completed_at"] = time.time()
        _persist_cycle_report(cycle_report)
        return

    # ── BTC + ETH path ──────────────────────────────────────────────────
    try:
        from src.trading.genetic_strategy_engine import GeneticStrategyEngine
    except ImportError as exc:
        logger.error("Cannot import genetic engine: %s", exc)
        return

    for asset in ["BTC", "ETH"]:
        per_asset: dict = {"asset": asset}
        try:
            if spread_pct > 0:
                engine = GeneticStrategyEngine(spread_pct=spread_pct)
            else:
                engine = GeneticStrategyEngine()
            ok = engine.load_market_data(asset=asset, timeframe="4h")
            if not ok:
                per_asset["status"] = "data_load_failed"
                cycle_report["per_asset"][asset] = per_asset
                continue

            # Drift detection BEFORE evolution: if drift detected in
            # market history, add fresh immigrants pre-emptively.
            if drift_immigrants and engine.closes is not None:
                try:
                    from src.trading.genetic_drift import detect_drift
                    signal = detect_drift(list(engine.closes))
                    per_asset["drift_signal"] = signal.to_dict()
                except Exception as exc:
                    logger.debug("drift detect skipped: %s", exc)

            # Surrogate pre-filter setup (cold-start: no model yet).
            surrogate_obj = None
            if surrogate:
                try:
                    from src.trading.genetic_surrogate import SurrogateFilter
                    surrogate_obj = SurrogateFilter()
                    per_asset["surrogate_enabled"] = True
                except Exception as exc:
                    logger.warning("Surrogate setup failed: %s", exc)

            # ── Run evolution ──────────────────────────────────────────
            if walk_forward:
                from src.trading.genetic_walk_forward import evolve_walk_forward
                wf = evolve_walk_forward(
                    engine,
                    generations=generations,
                    population_size=population_size,
                )
                per_asset["walk_forward"] = wf
                per_asset["best_oos"] = wf.get("best_oos")
            else:
                hof = engine.evolve(
                    generations=generations,
                    population_size=population_size,
                )
                if hof:
                    best = hof[0]
                    per_asset["best"] = {
                        "name": best.name,
                        "fitness": best.fitness,
                        "total_pnl": best.total_pnl,
                        "win_rate": best.win_rate,
                        "entry_rule": best.entry_rule,
                        "exit_rule": best.exit_rule,
                    }

            # ── DSR gate (post-evolve) ─────────────────────────────────
            if dsr_gate:
                try:
                    from src.trading.genetic_dsr_gate import apply_dsr_gate
                    dsr_stats = apply_dsr_gate(
                        engine.population,
                        n_trials=generations * population_size,
                    )
                    per_asset["dsr_gate"] = dsr_stats
                except Exception as exc:
                    logger.warning("DSR gate failed: %s", exc)

            # ── CMA-ES refine top-K ────────────────────────────────────
            if cma_es:
                try:
                    from src.trading.genetic_cma_es import (
                        hybrid_cma_es_refine_top_k,
                    )
                    cma_results = hybrid_cma_es_refine_top_k(
                        engine, k=5, max_generations=8,
                    )
                    per_asset["cma_es"] = [r.to_dict() for r in cma_results]
                except Exception as exc:
                    logger.warning("CMA-ES refine failed: %s", exc)

            # ── LLM mutation ───────────────────────────────────────────
            if llm_mutate > 0:
                try:
                    from src.trading.genetic_llm_mutation import (
                        inject_llm_mutations_into_population,
                    )
                    llm_results = inject_llm_mutations_into_population(
                        engine, n_calls=llm_mutate,
                    )
                    per_asset["llm_mutate"] = [r.to_dict() for r in llm_results]
                except Exception as exc:
                    logger.warning("LLM mutation failed: %s", exc)

            # ── MAP-Elites archive ────────────────────────────────────
            if map_elites:
                try:
                    from src.trading.genetic_map_elites import MAPElitesArchive
                    archive = MAPElitesArchive()
                    update_stats = archive.update(engine.population)
                    archive.update(engine.hall_of_fame)
                    per_asset["map_elites"] = {
                        "summary": archive.summary(),
                        "update_stats": update_stats,
                        "diverse_top_5": archive.diverse_top_k(5),
                    }
                except Exception as exc:
                    logger.warning("MAP-Elites failed: %s", exc)

            # ── Drift immigrant injection (post-evolve) ───────────────
            if drift_immigrants and engine.closes is not None:
                try:
                    from src.trading.genetic_drift import (
                        maybe_inject_immigrants_on_drift,
                    )
                    drift_out = maybe_inject_immigrants_on_drift(
                        engine, list(engine.closes),
                    )
                    per_asset["drift_immigrants"] = drift_out
                except Exception as exc:
                    logger.warning("Drift immigrants failed: %s", exc)

            # ── Surrogate observation accumulation (for next cycle) ──
            if surrogate_obj is not None:
                try:
                    surrogate_obj.add_population(engine.population)
                    trained = surrogate_obj.train()
                    per_asset["surrogate_trained"] = bool(trained)
                except Exception as exc:
                    logger.warning("Surrogate train failed: %s", exc)

            # ── Grammatical Evolution branch ──────────────────────────
            if grammatical_evolution:
                try:
                    from src.trading.genetic_grammar import evolve_grammatical
                    ge_top = evolve_grammatical(
                        list(engine.closes), list(engine.highs),
                        list(engine.lows),
                        spread_pct=engine.spread_pct,
                        population_size=min(40, population_size),
                        generations=min(8, generations),
                    )
                    per_asset["grammatical_evolution"] = [
                        ind.to_dict() for ind in ge_top[:5]
                    ]
                except Exception as exc:
                    logger.warning("Grammatical evolution failed: %s", exc)

            # Logging
            best_summary = per_asset.get("best") or per_asset.get("best_oos")
            if best_summary:
                logger.info(
                    "  %s best: name=%s fit=%.3f pnl=%+.1f%%",
                    asset,
                    best_summary.get("name") or best_summary.get("dna_name"),
                    best_summary.get("fitness", 0.0)
                        or best_summary.get("test_fitness", 0.0),
                    best_summary.get("total_pnl", 0.0)
                        or best_summary.get("test_pnl_pct", 0.0),
                )
        except Exception as exc:
            logger.exception("  %s evolution failed: %s", asset, exc)
            per_asset["status"] = f"error:{type(exc).__name__}"

        cycle_report["per_asset"][asset] = per_asset

    cycle_report["completed_at"] = time.time()
    _persist_cycle_report(cycle_report)


def main():
    parser = argparse.ArgumentParser(description='Genetic Strategy Evolution Loop')
    parser.add_argument('--population_size', type=int, default=50,
                        help='Strategies in population')
    parser.add_argument('--generations', type=int, default=10,
                        help='Generations per cycle')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Hours between cycles')
    parser.add_argument('--once', action='store_true',
                        help='Run one cycle and exit')

    # Audit mode flags
    parser.add_argument('--walk-forward', action='store_true',
                        help='P0: 80/15/5 train/val/test split with OOS fitness')
    parser.add_argument('--dsr-gate', action='store_true',
                        help='P0: Deflated Sharpe gate (selection-bias correction)')
    parser.add_argument('--cma-es', action='store_true',
                        help='P1: CMA-ES local search refine top-K HOF')
    parser.add_argument('--llm-mutate', type=int, default=0,
                        help='P1: N LLM-mutation calls per cycle (default 0)')
    parser.add_argument('--multi-asset', action='store_true',
                        help='P1: evolve across all (asset,TF) parquet pairs')
    parser.add_argument('--map-elites', action='store_true',
                        help='P2: build MAP-Elites behavioral grid archive')
    parser.add_argument('--surrogate', action='store_true',
                        help='P2: surrogate pre-filter on prior fitness data')
    parser.add_argument('--drift-immigrants', action='store_true',
                        help='P3: detect drift, inject random immigrants')
    parser.add_argument('--grammatical-evolution', action='store_true',
                        help='P3: parallel GE branch (BNF grammar)')
    parser.add_argument('--mode', choices=['default', 'p0', 'p0p1', 'p0p1p2', 'all'],
                        default='default',
                        help='Convenience preset for audit modes')

    args = parser.parse_args()

    # Apply mode preset (combines with explicit flags)
    if args.mode in ('p0', 'p0p1', 'p0p1p2', 'all'):
        args.walk_forward = True
        args.dsr_gate = True
    if args.mode in ('p0p1', 'p0p1p2', 'all'):
        args.cma_es = True
        if args.llm_mutate == 0:
            args.llm_mutate = 1
        args.multi_asset = True
    if args.mode in ('p0p1p2', 'all'):
        args.map_elites = True
        args.surrogate = True
    if args.mode == 'all':
        args.drift_immigrants = True
        args.grammatical_evolution = True

    spread_pct = _load_spread_pct()

    interval_sec = int(args.interval * 3600)
    logger.info("Genetic Evolution Loop starting (mode=%s)", args.mode)
    logger.info("  Population:    %d", args.population_size)
    logger.info("  Generations:   %d", args.generations)
    logger.info("  Interval:      %sh (%ss)", args.interval, interval_sec)
    logger.info("  Walk-forward:  %s", args.walk_forward)
    logger.info("  DSR gate:      %s", args.dsr_gate)
    logger.info("  CMA-ES refine: %s", args.cma_es)
    logger.info("  LLM mutate:    %d/cycle", args.llm_mutate)
    logger.info("  Multi-asset:   %s", args.multi_asset)
    logger.info("  MAP-Elites:    %s", args.map_elites)
    logger.info("  Surrogate:     %s", args.surrogate)
    logger.info("  Drift immig:   %s", args.drift_immigrants)
    logger.info("  Grammatical:   %s", args.grammatical_evolution)

    while True:
        logger.info("=" * 60)
        logger.info("Starting genetic evolution cycle...")
        run_evolution_cycle(
            population_size=args.population_size,
            generations=args.generations,
            spread_pct=spread_pct,
            walk_forward=args.walk_forward,
            dsr_gate=args.dsr_gate,
            cma_es=args.cma_es,
            llm_mutate=args.llm_mutate,
            multi_asset=args.multi_asset,
            map_elites=args.map_elites,
            surrogate=args.surrogate,
            drift_immigrants=args.drift_immigrants,
            grammatical_evolution=args.grammatical_evolution,
        )

        if args.once:
            break

        # C9 brain-to-body cadence (preserved from original).
        sleep_s = interval_sec
        try:
            from src.learning.brain_to_body import (
                get_controller, current_genetic_cadence_s,
            )
            get_controller().refresh()
            dynamic = current_genetic_cadence_s(default=float(interval_sec))
            lo = max(900.0, interval_sec / 4.0)
            hi = float(interval_sec * 2)
            sleep_s = int(max(lo, min(hi, dynamic)))
        except Exception:
            sleep_s = interval_sec

        next_h = sleep_s / 3600.0
        if abs(sleep_s - interval_sec) > 60:
            logger.info(
                "Cycle complete. Next in %.2fh (brain-to-body adjusted from %sh).",
                next_h, args.interval,
            )
        else:
            logger.info("Cycle complete. Next in %sh.", args.interval)
        time.sleep(sleep_s)


if __name__ == '__main__':
    main()
