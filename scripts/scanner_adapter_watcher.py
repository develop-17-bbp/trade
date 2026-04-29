"""Scanner adapter watcher — runs as Process 10 on the RTX 5090 box.

The 4060 (India) trains the scanner LoRA every ~3h and uploads the
adapter delta over SSH. This watcher polls models/unsloth_adapters/ for
new `<tag>.ready` markers and, for each one:

  1. Reads the sidecar metadata.json
  2. Merges LoRA into the base model + saves Q4_K_M GGUF (Unsloth)
  3. Builds Modelfile + `ollama create <tag> -f Modelfile`
  4. Builds 50-sample validation set from warm_store
  5. champion_gate.evaluate_gate('scanner', incumbent, challenger, val, infer_fn)
  6. If promote=True → _hot_swap('scanner', tag); bot picks up next tick
  7. If rejected → rename adapter dir <tag>-rejected (preserved for audit)
  8. Delete the .ready marker either way
  9. Log JSON to logs/fine_tune/<YYYY-MM-DD>_scanner_<tag>.json

CLI:
    python -m scripts.scanner_adapter_watcher              # forever loop
    python -m scripts.scanner_adapter_watcher --once       # process queue once
    python -m scripts.scanner_adapter_watcher --dry-run    # simulate, no merge

Honors ACT_DISABLE_FINETUNE=1 global kill switch.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger("watcher_5090")

ADAPTER_DIR = REPO_ROOT / "models" / "unsloth_adapters"
LOG_DIR = REPO_ROOT / "logs" / "fine_tune"
SCANNER_BASE = os.getenv("ACT_SCANNER_BASE_MODEL", "qwen2.5-coder:7b")
POLL_INTERVAL_S = float(os.getenv("ACT_WATCHER_POLL_S", "60"))
MAX_VAL_SAMPLES = int(os.getenv("ACT_WATCHER_VAL_SAMPLES", "50"))


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "watcher_5090.log", encoding="utf-8"),
        ],
    )


def _kill_switch_set() -> bool:
    return os.getenv("ACT_DISABLE_FINETUNE", "0") == "1"


def _list_pending_markers() -> List[Path]:
    if not ADAPTER_DIR.exists():
        return []
    markers = sorted(ADAPTER_DIR.glob("scanner*.ready"))
    return markers


def _read_metadata(adapter_dir: Path) -> Optional[Dict[str, Any]]:
    meta_path = adapter_dir / "metadata.json"
    if not meta_path.exists():
        logger.warning("metadata.json missing in %s", adapter_dir)
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("metadata read failed: %s", e)
        return None


def _build_validation(n: int, asset_class: Optional[str] = None) -> List[Dict[str, Any]]:
    """Pull the last N filtered scanner-relevant samples from warm_store.

    `asset_class` partitions the validation set so a stocks adapter is
    evaluated only on STOCK rows and a crypto adapter only on CRYPTO
    rows — without this filter, champion gate would mix asset-class
    accuracy and silently approve stocks adapters that happen to trade
    well on crypto data."""
    try:
        from src.ai.training_data_filter import load_experience_samples
        samples, _ = load_experience_samples(
            asset=None, asset_class=asset_class,
            max_age_days=7.0, min_pnl_abs_pct=0.3,
        )
        return [s.to_dict() for s in samples[-n:]]
    except Exception as e:
        logger.warning("validation set build failed: %s", e)
        return []


def _merge_and_register(adapter_dir: Path, out_tag: str,
                         dry_run: bool) -> bool:
    """Run Unsloth's merge + GGUF + Ollama register pipeline on an
    adapter dir produced by the 4060 (LoRA-only export).

    In dry_run, simulates success without loading models.
    """
    if dry_run:
        logger.info("DRY RUN — skipping merge of %s", adapter_dir)
        return True

    try:
        from src.ai.unsloth_backend import _require_unsloth, _ollama_create, _resolve_hf_id
        _require_unsloth()
        from unsloth import FastLanguageModel  # type: ignore
    except Exception as e:
        logger.error("unsloth import failed: %s", e)
        return False

    meta = _read_metadata(adapter_dir) or {}
    base_model = meta.get("base_model") or SCANNER_BASE
    hf_repo = meta.get("hf_repo") or _resolve_hf_id(base_model)

    try:
        # Load base in 4-bit and apply the adapter from disk.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=hf_repo, max_seq_length=2048, load_in_4bit=True, dtype=None,
        )
        model.load_adapter(str(adapter_dir), adapter_name="scanner_act")
        model.set_adapter("scanner_act")
    except Exception as e:
        logger.error("base-model load + adapter merge failed: %s", e)
        return False

    try:
        model.save_pretrained_gguf(
            str(adapter_dir), tokenizer, quantization_method="q4_k_m",
        )
    except Exception as e:
        logger.error("GGUF export failed: %s", e)
        return False

    found = sorted(adapter_dir.glob("*Q4_K_M*.gguf")) or sorted(adapter_dir.glob("*.gguf"))
    if not found:
        logger.error("no GGUF produced in %s", adapter_dir)
        return False
    gguf_path = adapter_dir / "model.gguf"
    try:
        gguf_path.unlink(missing_ok=True)
        gguf_path.write_bytes(found[0].read_bytes())
    except Exception as e:
        logger.error("GGUF stage failed: %s", e)
        return False

    modelfile_path = adapter_dir / "Modelfile"
    try:
        try:
            num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
        except (ValueError, TypeError):
            num_ctx = 16384
        modelfile_path.write_text(
            f"FROM {gguf_path.as_posix()}\n"
            f"PARAMETER num_ctx {num_ctx}\n"
            f"PARAMETER temperature 0.3\n",
            encoding="utf-8",
        )
    except Exception as e:
        logger.error("Modelfile write failed: %s", e)
        return False

    return _ollama_create(out_tag, modelfile_path)


def _evaluate_and_swap(out_tag: str, dry_run: bool) -> Dict[str, Any]:
    """Run champion_gate, hot-swap if promote=True. Returns audit dict."""
    audit: Dict[str, Any] = {"out_tag": out_tag, "dry_run": dry_run}

    # Phase D dual-asset: route by adapter prefix.
    #   scanner-stocks-act-* → stocks adapter, validate on STOCK rows,
    #     promote sets ACT_STOCKS_SCANNER_MODEL on the 4060 over Tailscale.
    #   scanner-crypto-act-* (or legacy scanner-act-*) → crypto adapter,
    #     validate on CRYPTO rows, promote sets ACT_SCANNER_MODEL locally.
    is_stocks = out_tag.startswith("scanner-stocks-")
    asset_class_filter = "STOCK" if is_stocks else "CRYPTO"
    audit["asset_class"] = asset_class_filter

    val = _build_validation(MAX_VAL_SAMPLES, asset_class=asset_class_filter)
    audit["validation_n"] = len(val)
    if len(val) < 5:
        audit["error"] = f"insufficient_validation_samples_{asset_class_filter.lower()}"
        return audit

    incumbent = (
        os.getenv("ACT_STOCKS_SCANNER_MODEL") if is_stocks else os.getenv("ACT_SCANNER_MODEL")
    ) or SCANNER_BASE

    if dry_run:
        audit["promote"] = False
        audit["error"] = None
        audit["dry_run_skip"] = "champion_gate"
        return audit

    try:
        from src.ai.champion_gate import evaluate_gate
        from src.ai.unsloth_backend import UnslothQLoRABackend
        backend = UnslothQLoRABackend(export="gguf", lora_r=8)
        gate = evaluate_gate(
            brain="scanner",
            incumbent_id=incumbent,
            challenger_id=out_tag,
            validation_samples=val,
            inference_fn=backend.infer,
        )
        audit["gate"] = gate.to_dict()
        audit["promote"] = bool(gate.promote)
    except Exception as e:
        logger.exception("champion_gate failed: %s", e)
        audit["error"] = f"gate_exception: {e}"
        return audit

    if audit["promote"]:
        try:
            if is_stocks:
                # Cross-box hot-swap: 4060 owns the live stocks model env.
                err = _cross_box_hot_swap_stocks(out_tag)
                audit["swap_error"] = err
                if not err:
                    logger.info(
                        "PROMOTED stocks scanner adapter %s on the 4060 (cross-box ssh)",
                        out_tag,
                    )
            else:
                from src.ai.dual_brain_trainer import _hot_swap
                err = _hot_swap("scanner", out_tag)
                audit["swap_error"] = err
                if not err:
                    logger.info(
                        "PROMOTED crypto scanner adapter %s — bot reads next tick", out_tag,
                    )
        except Exception as e:
            logger.exception("hot-swap failed: %s", e)
            audit["swap_error"] = str(e)

    return audit


def _cross_box_hot_swap_stocks(out_tag: str) -> str:
    """SSH to the 4060 (over Tailscale) and set ACT_STOCKS_SCANNER_MODEL.

    The 4060 is the live stocks-trading box; the env var swap is what
    flips the next agentic-loop tick to use the new adapter. Returns
    "" on success, error string otherwise.
    """
    peer = os.getenv("ACT_4060_SSH_HOST") or os.getenv("ACT_STOCKS_SSH_HOST")
    if not peer:
        return "ACT_4060_SSH_HOST not set"
    import subprocess
    cmd = [
        "ssh", peer,
        f"powershell.exe -Command \"setx ACT_STOCKS_SCANNER_MODEL '{out_tag}'\"",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return f"ssh setx failed rc={result.returncode}: {result.stderr.strip()[:200]}"
        return ""
    except subprocess.TimeoutExpired:
        return "ssh timeout"
    except Exception as e:
        return f"ssh exception: {e}"


def _process_marker(marker: Path, dry_run: bool) -> Dict[str, Any]:
    started = time.time()
    out_tag = marker.stem  # 'scanner-act-1712345.ready' → 'scanner-act-1712345'
    safe = out_tag.replace(":", "_").replace("/", "_")
    adapter_dir = ADAPTER_DIR / safe
    summary: Dict[str, Any] = {
        "out_tag": out_tag, "marker": marker.name,
        "started_at": started, "dry_run": dry_run, "ok": False,
        "phase": "init",
    }

    if not adapter_dir.exists():
        summary["error"] = f"adapter dir missing: {adapter_dir.name}"
        try:
            marker.unlink()  # marker is stale, drop it
        except Exception:
            pass
        return summary

    summary["phase"] = "merge"
    if not _merge_and_register(adapter_dir, out_tag, dry_run):
        summary["error"] = "merge_or_register_failed"
        # Quarantine the dir so a re-poll doesn't loop on it.
        try:
            adapter_dir.rename(ADAPTER_DIR / f"{safe}-failed")
            marker.unlink()
        except Exception:
            pass
        return summary

    summary["phase"] = "champion_gate"
    audit = _evaluate_and_swap(out_tag, dry_run)
    summary["audit"] = audit
    summary["promote"] = audit.get("promote", False)

    summary["phase"] = "cleanup"
    try:
        if not summary["promote"] and not dry_run:
            adapter_dir.rename(ADAPTER_DIR / f"{safe}-rejected")
        marker.unlink(missing_ok=True)
    except Exception as e:
        summary["cleanup_error"] = str(e)

    summary["ok"] = True
    summary["duration_s"] = round(time.time() - started, 1)
    return summary


def _persist_summary(summary: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    tag = summary.get("out_tag", "unknown")
    safe_tag = tag.replace(":", "_").replace("/", "_")
    path = LOG_DIR / f"{date}_scanner_{safe_tag}.json"
    try:
        path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("summary write failed: %s", e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="process pending markers, exit")
    ap.add_argument("--dry-run", action="store_true", help="skip merge + gate")
    args = ap.parse_args()

    _setup_logging()
    logger.info("watcher_5090 boot — adapter_dir=%s poll=%.0fs once=%s dry_run=%s",
                ADAPTER_DIR, POLL_INTERVAL_S, args.once, args.dry_run)

    while True:
        if _kill_switch_set():
            logger.info("ACT_DISABLE_FINETUNE=1 — sleeping")
            if args.once:
                return 0
            time.sleep(POLL_INTERVAL_S)
            continue

        markers = _list_pending_markers()
        if markers:
            logger.info("found %d pending marker(s)", len(markers))
        for m in markers:
            try:
                summary = _process_marker(m, args.dry_run)
            except Exception as e:
                logger.exception("marker %s failed: %s", m.name, e)
                summary = {"out_tag": m.stem, "ok": False, "error": str(e)}
            _persist_summary(summary)
            logger.info("processed %s — ok=%s promote=%s",
                        summary.get("out_tag"), summary.get("ok"),
                        summary.get("promote"))

        if args.once:
            return 0
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
