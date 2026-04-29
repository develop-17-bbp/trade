"""Scanner-only QLoRA fine-tune loop for the RTX 4060 box (India).

Runs unattended every N hours. Pulls warm_store.sqlite from the 5090
(Kirkland) over SSH, runs LoRA training on qwen2.5-coder:7b in 4-bit,
uploads the LoRA delta back, and drops a `.ready` marker that the 5090's
scanner_adapter_watcher.py picks up to run champion-gate.

The 4060 only exports the LoRA adapter (~150 MB), NOT a merged GGUF —
the 5090 watcher does the merge + Q4_K_M quantize + Ollama register
locally, so cross-continent traffic stays small and the 4060 doesn't
need llama.cpp tooling.

Env vars:
    ACT_5090_SSH_HOST            ssh-config alias for the 5090 (e.g. act5090)
    ACT_5090_TRADE_DIR           remote repo root (e.g. C:/Users/admin/trade)
    ACT_SCANNER_FINETUNE_INTERVAL_H   loop interval (default 3.0)
    ACT_DISABLE_FINETUNE         set to 1 to halt the loop (kill switch)
    ACT_SCANNER_BASE_MODEL       Ollama-style tag (default qwen2.5-coder:7b)

CLI:
    python -m scripts.finetune_scanner_4060               # forever loop
    python -m scripts.finetune_scanner_4060 --once        # single iter
    python -m scripts.finetune_scanner_4060 --dry-run     # stub backend, no GPU

Lock file at scripts/.scanner_train.lock prevents overlapping runs.
Last-train marker at scripts/.last_scanner_train_ids.json holds
decision_ids of samples used so the next cycle can detect "no new data,
skip".
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ai.dual_brain_trainer import (  # noqa: E402
    StubBackend, run_cycle, persist_report,
)


logger = logging.getLogger("scanner_4060")

LOCK_FILE = REPO_ROOT / "scripts" / ".scanner_train.lock"
LAST_IDS_FILE = REPO_ROOT / "scripts" / ".last_scanner_train_ids.json"
ADAPTER_DIR = REPO_ROOT / "models" / "unsloth_adapters"
DATA_DIR = REPO_ROOT / "data"
LOG_DIR = REPO_ROOT / "logs" / "fine_tune"


def _asset_class_paths(asset_class: str) -> Dict[str, Path]:
    """Per-class lock + last-ids marker so crypto and stocks cycles
    don't clobber each other when the router schedules them in the
    same outside-RTH window."""
    cls = (asset_class or "CRYPTO").upper()
    suffix = "stocks" if cls == "STOCK" else "crypto"
    return {
        "lock":     REPO_ROOT / "scripts" / f".{suffix}_train.lock",
        "last_ids": REPO_ROOT / "scripts" / f".last_{suffix}_train_ids.json",
    }


def _tag_prefix(asset_class: str) -> str:
    """Output adapter prefix — `scanner-crypto-act-<ts>` vs
    `scanner-stocks-act-<ts>`. The 5090 watcher picks adapter type
    from the prefix to route to the right ACT_*_SCANNER_MODEL env."""
    cls = (asset_class or "CRYPTO").upper()
    return "scanner-stocks" if cls == "STOCK" else "scanner-crypto"

DEFAULT_INTERVAL_H = float(os.getenv("ACT_SCANNER_FINETUNE_INTERVAL_H", "3.0"))
SCANNER_BASE = os.getenv("ACT_SCANNER_BASE_MODEL", "qwen2.5-coder:7b")
MIN_NEW_SAMPLES = int(os.getenv("ACT_SCANNER_MIN_NEW_SAMPLES", "30"))
MIN_FREE_VRAM_MB = int(os.getenv("ACT_SCANNER_MIN_VRAM_MB", "7000"))


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_DIR / "scanner_4060.log", encoding="utf-8"),
        ],
    )


def _kill_switch_set() -> bool:
    return os.getenv("ACT_DISABLE_FINETUNE", "0") == "1"


def _acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            age = time.time() - LOCK_FILE.stat().st_mtime
        except Exception:
            age = 0
        if age < 6 * 3600:
            logger.warning("lock file present (age %.0fs) — another run active; skipping", age)
            return False
        logger.warning("stale lock (%.0fs) — overriding", age)
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def _release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _vram_check() -> bool:
    """Return True if we have enough free VRAM to train. Skip if no GPU."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=False, capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            logger.warning("nvidia-smi failed: %s", r.stderr.strip())
            return False
        free_mb = int(r.stdout.strip().splitlines()[0])
        if free_mb < MIN_FREE_VRAM_MB:
            logger.warning("free VRAM %d MB < threshold %d MB — skipping cycle", free_mb, MIN_FREE_VRAM_MB)
            return False
        logger.info("VRAM check ok — free %d MB", free_mb)
        return True
    except FileNotFoundError:
        logger.error("nvidia-smi not found — install CUDA driver")
        return False
    except Exception as e:
        logger.warning("vram check exception: %s", e)
        return False


def _ssh_settings() -> Dict[str, str]:
    host = os.getenv("ACT_5090_SSH_HOST", "").strip()
    remote = os.getenv("ACT_5090_TRADE_DIR", "").strip()
    if not host or not remote:
        raise RuntimeError(
            "Set ACT_5090_SSH_HOST (ssh-config alias) and ACT_5090_TRADE_DIR "
            "(remote repo path) — see docs/finetune_two_box.md"
        )
    return {"host": host, "remote_dir": remote}


def _ssh_check(host: str) -> bool:
    """One-shot connectivity probe. BatchMode disables key prompts."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             host, "echo ok"],
            check=False, capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0 or "ok" not in r.stdout:
            logger.error("ssh %s failed (rc=%s) stderr=%s", host, r.returncode, r.stderr.strip())
            return False
        return True
    except Exception as e:
        logger.error("ssh probe exception: %s", e)
        return False


def _scp_pull_warm_store(host: str, remote_dir: str) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = DATA_DIR / "warm_store.sqlite"
    remote_path = f"{host}:{remote_dir.rstrip('/')}/data/warm_store.sqlite"
    logger.info("pulling warm_store from %s …", remote_path)
    try:
        r = subprocess.run(
            ["scp", "-o", "BatchMode=yes", remote_path, str(local_path)],
            check=False, capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            logger.error("scp pull failed: %s", r.stderr.strip())
            return False
        size = local_path.stat().st_size if local_path.exists() else 0
        logger.info("pulled warm_store ok — %d bytes", size)
        return True
    except Exception as e:
        logger.error("scp pull exception: %s", e)
        return False


def _read_last_ids() -> Set[str]:
    if not LAST_IDS_FILE.exists():
        return set()
    try:
        return set(json.loads(LAST_IDS_FILE.read_text(encoding="utf-8")) or [])
    except Exception:
        return set()


def _write_last_ids(ids: Set[str]) -> None:
    try:
        LAST_IDS_FILE.write_text(
            json.dumps(sorted(ids))[:5_000_000],  # bounded
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("last_ids write failed: %s", e)


def _count_new_samples(asset_filter: Optional[str] = None) -> tuple[int, Set[str]]:
    """Return (n_new, all_ids) where n_new is the count not seen in the last run.

    Crypto-only legacy entrypoint — preserves the original signature for
    any test or runner that imported it directly. New callers should use
    _count_new_samples_for_class() which is asset_class-aware.
    """
    try:
        from src.ai.training_data_filter import load_experience_samples
        samples, _stats = load_experience_samples(
            asset=asset_filter, max_age_days=14.0, min_pnl_abs_pct=0.3,
        )
        all_ids = {s.decision_id for s in samples}
        prev = _read_last_ids()
        n_new = len(all_ids - prev)
        return n_new, all_ids
    except Exception as e:
        logger.warning("sample count failed: %s", e)
        return 0, set()


def _count_new_samples_for_class(asset_class: str) -> tuple[int, Set[str]]:
    """Asset-class-aware sample counter. Reads the per-class last-IDs marker
    so a stocks cycle's progress doesn't decrement crypto's count and vice
    versa.
    """
    paths = _asset_class_paths(asset_class)
    last_ids_file = paths["last_ids"]
    try:
        from src.ai.training_data_filter import load_experience_samples
        samples, _stats = load_experience_samples(
            asset_class=asset_class, max_age_days=14.0, min_pnl_abs_pct=0.3,
        )
        all_ids = {s.decision_id for s in samples}
        prev: Set[str] = set()
        if last_ids_file.exists():
            try:
                prev = set(json.loads(last_ids_file.read_text(encoding="utf-8")) or [])
            except Exception:
                prev = set()
        n_new = len(all_ids - prev)
        return n_new, all_ids
    except Exception as e:
        logger.warning("sample count failed for %s: %s", asset_class, e)
        return 0, set()


def _write_last_ids_for_class(asset_class: str, ids: Set[str]) -> None:
    paths = _asset_class_paths(asset_class)
    try:
        paths["last_ids"].write_text(json.dumps(sorted(ids))[:5_000_000], encoding="utf-8")
    except Exception as e:
        logger.debug("last_ids write failed: %s", e)


def _tar_adapter_dir(out_tag: str) -> Optional[Path]:
    """Bundle the LoRA artifact dir into a tar.gz for upload."""
    src_dir = ADAPTER_DIR / out_tag.replace(":", "_").replace("/", "_")
    if not src_dir.exists():
        logger.error("adapter dir missing after train: %s", src_dir)
        return None
    archive = ADAPTER_DIR / f"{out_tag}.tar.gz"
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(src_dir, arcname=src_dir.name)
        logger.info("packed adapter → %s (%d bytes)", archive, archive.stat().st_size)
        return archive
    except Exception as e:
        logger.error("tar failed: %s", e)
        return None


def _upload_and_signal(host: str, remote_dir: str, archive: Path, out_tag: str) -> bool:
    remote_adapters = f"{remote_dir.rstrip('/')}/models/unsloth_adapters"
    remote_archive = f"{host}:{remote_adapters}/{archive.name}"
    logger.info("scp adapter → %s …", remote_archive)
    try:
        r = subprocess.run(
            ["scp", "-o", "BatchMode=yes", str(archive), remote_archive],
            check=False, capture_output=True, text=True, timeout=1800,
        )
        if r.returncode != 0:
            logger.error("scp upload failed: %s", r.stderr.strip())
            return False
    except Exception as e:
        logger.error("scp upload exception: %s", e)
        return False

    # Use PowerShell on the remote (Windows 5090) to extract + drop marker.
    safe_tag = out_tag.replace(":", "_").replace("/", "_")
    archive_name = archive.name
    # Single PowerShell command: extract, delete the tarball, drop the .ready
    # marker. tar is built into Win 10/11.
    remote_cmd = (
        f"cd '{remote_adapters}'; "
        f"tar -xzf '{archive_name}'; "
        f"if (Test-Path '{archive_name}') {{ Remove-Item '{archive_name}' -Force }}; "
        f"New-Item -ItemType File -Path '{safe_tag}.ready' -Force | Out-Null; "
        f"Write-Host 'extracted+marked'"
    )
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host,
             "powershell.exe", "-NoProfile", "-Command", remote_cmd],
            check=False, capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            logger.error("remote extract failed: %s", r.stderr.strip())
            return False
        logger.info("remote ack: %s", r.stdout.strip())
        return True
    except Exception as e:
        logger.error("ssh extract exception: %s", e)
        return False


def _run_one_cycle(dry_run: bool, asset_class: str = "CRYPTO") -> Dict[str, Any]:
    """One full cycle. Returns a result dict for the audit log.

    `asset_class` (CRYPTO/STOCK) selects the warm_store filter, the
    output tag prefix, and the per-class lock/marker files so the
    router can interleave crypto + stocks cycles outside RTH without
    conflicts. Defaults to CRYPTO so existing crypto-only invocations
    continue to work unchanged.
    """
    cycle_started = time.time()
    asset_class = (asset_class or "CRYPTO").upper()
    summary: Dict[str, Any] = {
        "started_at": cycle_started,
        "asset_class": asset_class,
        "dry_run": dry_run,
        "ok": False,
        "skipped_reason": None,
    }

    if _kill_switch_set():
        summary["skipped_reason"] = "ACT_DISABLE_FINETUNE=1"
        return summary

    try:
        ssh = _ssh_settings()
    except RuntimeError as e:
        summary["skipped_reason"] = str(e)
        return summary

    if not _ssh_check(ssh["host"]):
        summary["skipped_reason"] = "ssh_unreachable"
        return summary

    if not dry_run and not _vram_check():
        summary["skipped_reason"] = "insufficient_vram"
        return summary

    if not _scp_pull_warm_store(ssh["host"], ssh["remote_dir"]):
        summary["skipped_reason"] = "warm_store_pull_failed"
        return summary

    n_new, all_ids = _count_new_samples_for_class(asset_class)
    summary["n_new_samples"] = n_new
    if n_new < MIN_NEW_SAMPLES:
        summary["skipped_reason"] = f"only {n_new} new {asset_class} samples (< {MIN_NEW_SAMPLES})"
        return summary

    # ── Run the cycle ───────────────────────────────────────────────────
    if dry_run:
        backend = StubBackend()
        logger.info("DRY RUN — using StubBackend, no GPU calls")
    else:
        try:
            from src.ai.unsloth_backend import UnslothQLoRABackend
            backend = UnslothQLoRABackend(export="lora_only", lora_r=8)
        except Exception as e:
            summary["skipped_reason"] = f"unsloth_import_failed: {e}"
            return summary

    ts = int(time.time())
    prefix = _tag_prefix(asset_class)   # 'scanner-crypto' | 'scanner-stocks'
    expected_tag = f"{prefix}-act-{ts}"
    logger.info("starting %s scanner cycle → %s", asset_class, expected_tag)

    report = run_cycle(
        backend, brains=["scanner"], pause_agentic=False,
        scanner_incumbent=SCANNER_BASE,
        scanner_tag_prefix=prefix,
    )
    summary["report"] = report.to_dict()
    persist_report(report)

    if not report.scanner or not report.scanner.training_ok:
        summary["skipped_reason"] = "training_failed"
        return summary

    # ── Tag uses dual_brain_trainer's challenger_tag (timestamp may differ) ──
    out_tag = report.scanner.challenger_tag
    if dry_run:
        summary["ok"] = True
        summary["out_tag"] = out_tag
        summary["note"] = "dry-run skipped upload"
        _write_last_ids_for_class(asset_class, all_ids)
        return summary

    # ── Pack + upload + signal the 5090 watcher ────────────────────────
    archive = _tar_adapter_dir(out_tag)
    if not archive:
        summary["skipped_reason"] = "tar_failed"
        return summary

    if not _upload_and_signal(ssh["host"], ssh["remote_dir"], archive, out_tag):
        summary["skipped_reason"] = "upload_failed"
        return summary

    # Clean up local archive (the dir stays as audit trail).
    try:
        archive.unlink()
    except Exception:
        pass

    _write_last_ids_for_class(asset_class, all_ids)
    summary["ok"] = True
    summary["out_tag"] = out_tag
    summary["duration_s"] = round(time.time() - cycle_started, 1)
    return summary


def _persist_summary(summary: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path = LOG_DIR / f"{date}_scanner_4060.json"
    try:
        prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        prev = []
    prev.append(summary)
    try:
        path.write_text(json.dumps(prev, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logger.debug("summary write failed: %s", e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single iteration then exit")
    ap.add_argument("--dry-run", action="store_true", help="stub backend, no GPU")
    ap.add_argument(
        "--asset-class", default="CRYPTO", choices=["CRYPTO", "STOCK"],
        help="warm_store asset_class filter; selects tag prefix + per-class lock + last-ids marker (default: CRYPTO for back-compat)",
    )
    args = ap.parse_args()

    _setup_logging()
    logger.info("scanner_4060 boot — base=%s interval=%.1fh dry_run=%s once=%s",
                SCANNER_BASE, DEFAULT_INTERVAL_H, args.dry_run, args.once)

    if not _acquire_lock():
        return 2
    try:
        while True:
            try:
                summary = _run_one_cycle(args.dry_run, asset_class=args.asset_class)
            except Exception as e:
                logger.exception("cycle exception: %s", e)
                summary = {"started_at": time.time(), "ok": False,
                           "asset_class": args.asset_class,
                           "skipped_reason": f"exception: {e}"}
            _persist_summary(summary)
            logger.info("cycle complete: ok=%s reason=%s", summary.get("ok"), summary.get("skipped_reason"))
            if args.once:
                return 0 if summary.get("ok") or summary.get("skipped_reason") else 1
            sleep_s = max(60.0, DEFAULT_INTERVAL_H * 3600.0)
            logger.info("sleeping %.0fs until next cycle", sleep_s)
            time.sleep(sleep_s)
    finally:
        _release_lock()


if __name__ == "__main__":
    sys.exit(main())
