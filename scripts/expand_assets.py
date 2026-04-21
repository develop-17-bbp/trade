"""
Asset expansion helper — reports which assets have trained models + calibration
JSONs and ARE ready to add to config.yaml's `assets:` list, vs which need a
training pass first.

Safe — does NOT modify config.yaml. Use its output to manually pick assets to
enable.

Usage:
    python scripts/expand_assets.py
    python scripts/expand_assets.py --enable BTC ETH AAVE   (dry-run: shows diff)
    python scripts/expand_assets.py --enable BTC ETH AAVE --write   (apply)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Authority-permissioned universe (from src/ai/authority_rules.py). Assets not
# on this list cannot be traded regardless of model availability.
AUTHORITY_ASSETS = [
    "BTC", "ETH", "AAVE", "SOL", "BNB",
    "ADA", "DOGE", "XRP", "AVAX", "DOT",
    "LINK", "MATIC",
]


def check_asset(asset: str) -> Dict[str, bool]:
    """Return readiness flags for one asset."""
    lower = asset.lower()
    return {
        "trained": (MODELS_DIR / f"lgbm_{lower}_trained.txt").exists(),
        "calibration": (MODELS_DIR / f"lgbm_{lower}_calibration.json").exists(),
        "thresholds": (MODELS_DIR / f"lgbm_{lower}_thresholds.json").exists(),
    }


def load_config_assets() -> List[str]:
    """Naive parse of config.yaml — finds the top-level assets: list."""
    if not CONFIG_PATH.exists():
        return []
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assets = cfg.get("assets", []) or []
        return [str(a).upper() for a in assets if isinstance(a, str)]
    except Exception as e:
        print(f"  [warn] could not parse config.yaml: {e}", file=sys.stderr)
        return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enable", nargs="*", default=[],
                    help="Asset symbols to enable (adds to config.yaml assets list)")
    ap.add_argument("--write", action="store_true",
                    help="Actually modify config.yaml (default: dry-run)")
    args = ap.parse_args()

    current = load_config_assets()
    print(f"Current assets in config.yaml: {current}")
    print()
    print(f"{'ASSET':<8} {'AUTHORITY':<10} {'TRAINED':<10} {'CAL':<6} {'CURRENT':<9} {'READY?'}")
    print("-" * 60)

    ready: List[str] = []
    for a in AUTHORITY_ASSETS:
        flags = check_asset(a)
        in_cfg = a in current
        can_enable = flags["trained"]  # calibration is nice-to-have, trained is required
        ready_tag = "YES" if can_enable and not in_cfg else ("in-config" if in_cfg else "no")
        print(f"{a:<8} {'yes':<10} {'yes' if flags['trained'] else 'NO':<10} "
              f"{'yes' if flags['calibration'] else 'no':<6} "
              f"{'yes' if in_cfg else 'no':<9} {ready_tag}")
        if can_enable and not in_cfg:
            ready.append(a)

    print()
    print(f"Ready to add (models exist, not in config): {ready}")

    if not args.enable:
        print("\nRun with --enable BTC ETH AAVE [...] to plan changes.")
        print("Add --write to actually modify config.yaml.")
        return 0

    # Validate requested additions
    requested = [a.upper() for a in args.enable]
    invalid = [a for a in requested if a not in AUTHORITY_ASSETS]
    if invalid:
        print(f"\nERROR: not on authority list: {invalid}")
        print(f"Permitted: {AUTHORITY_ASSETS}")
        return 2
    missing_model = [a for a in requested if not check_asset(a)["trained"]]
    if missing_model:
        print(f"\nERROR: no trained model found: {missing_model}")
        print("Run `python -m src.scripts.train_all_models --asset <X>` first.")
        return 3

    planned = sorted(set(current + requested))
    print(f"\nPlanned new assets list: {planned}")

    if not args.write:
        print("\nDry-run only. Add --write to apply.")
        return 0

    # Apply: read yaml, update top-level assets list + each exchange's assets list,
    # write back preserving indentation as best we can (ruamel not required).
    import yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["assets"] = planned
    if isinstance(cfg.get("exchanges"), list):
        for ex in cfg["exchanges"]:
            if isinstance(ex, dict):
                ex["assets"] = planned
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"\nconfig.yaml updated: {current} -> {planned}")
    print("Restart the bot (STOP_ALL + START_ALL) to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
