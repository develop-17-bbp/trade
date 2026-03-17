"""SHA-256 model integrity verification.

Generates and verifies checksums for model files to detect
unauthorized modifications or corruption.

Usage:
    python -m src.security.model_integrity --generate
    python -m src.security.model_integrity --verify
"""

import argparse
import hashlib
import json
import logging
import os
import stat
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

CHECKSUMS_FILE = "checksums.json"
# Only verify these known model files by default.
MODEL_EXTENSIONS = {".txt", ".pt", ".json", ".csv"}


def _sha256(file_path: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _compute_manifest_hash(checksums: dict) -> str:
    """
    Compute a deterministic hash of all model checksums.
    This acts as a 'root hash' that can be stored externally
    to detect tampering of checksums.json itself.
    """
    content = json.dumps(checksums, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def generate_checksums(models_dir: str = "models") -> dict[str, str]:
    """Scan model files in *models_dir*, compute SHA-256, and save to checksums.json.

    Returns:
        Dict mapping relative file paths to their hex digests.
    """
    models_path = Path(models_dir)
    if not models_path.is_dir():
        logger.error("Models directory not found: %s", models_path)
        return {}

    checksums: dict[str, str] = {}
    for file in sorted(models_path.iterdir()):
        # Skip checksums.json itself to avoid self-referencing
        if file.name == CHECKSUMS_FILE:
            continue
        if file.is_file() and file.suffix in MODEL_EXTENSIONS:
            digest = _sha256(file)
            key = file.as_posix()
            checksums[key] = digest
            logger.info("Checksum generated: %s -> %s", key, digest)

    # Save checksums
    checksums_path = models_path / CHECKSUMS_FILE
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    logger.info("Checksums saved to %s (%d files)", checksums_path, len(checksums))

    # Compute and display manifest hash for external storage
    manifest_hash = _compute_manifest_hash(checksums)
    logger.info("Manifest root hash: %s", manifest_hash)
    logger.info("Store this hash externally (env var MODEL_MANIFEST_HASH) for tamper detection.")

    return checksums


def _load_checksums(models_dir: str = "models") -> dict[str, str]:
    """Load the stored checksums from checksums.json."""
    checksums_path = Path(models_dir) / CHECKSUMS_FILE
    if not checksums_path.is_file():
        logger.error("Checksums file not found: %s. Run --generate first.", checksums_path)
        return {}
    with open(checksums_path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_model(model_path: str, models_dir: str = "models") -> bool:
    """Verify a single model file against its stored checksum."""
    checksums = _load_checksums(models_dir)
    if not checksums:
        return False

    file_path = Path(model_path)
    if not file_path.is_file():
        logger.error("Model file not found: %s", file_path)
        return False

    key = file_path.as_posix()
    expected = checksums.get(key)
    if expected is None:
        logger.error("No stored checksum for %s", key)
        return False

    actual = _sha256(file_path)
    if actual == expected:
        logger.info("PASS: %s", key)
        return True
    else:
        logger.warning("FAIL: %s (expected %s, got %s)", key, expected, actual)
        return False


def verify_all_models(models_dir: str = "models"):
    """Verify every model listed in checksums.json.

    Returns:
        Tuple of (passed_list, failed_list) for file paths.
    """
    checksums = _load_checksums(models_dir)
    if not checksums:
        return [], []

    # Verify manifest hash if environment variable is set
    manifest_env = os.environ.get("MODEL_MANIFEST_HASH", "")
    if manifest_env:
        actual_manifest = _compute_manifest_hash(checksums)
        if actual_manifest != manifest_env:
            logger.error(
                "CRITICAL: checksums.json manifest hash mismatch! "
                "Expected %s, got %s. Possible tampering detected.",
                manifest_env, actual_manifest
            )
            return [], list(checksums.keys())

    passed = []
    failed = []
    for key, expected in checksums.items():
        file_path = Path(key)
        if not file_path.is_file():
            logger.warning("MISSING: %s", key)
            failed.append(key)
            continue

        actual = _sha256(file_path)
        if actual == expected:
            logger.info("PASS: %s", key)
            passed.append(key)
        else:
            logger.warning("FAIL: %s (expected %s, got %s)", key, expected, actual)
            failed.append(key)

    logger.info("Verification complete: %d/%d passed", len(passed), len(passed) + len(failed))
    return passed, failed


def protect_model_files(models_dir: str = "models"):
    """Set model files to read-only to prevent accidental modification."""
    models_path = Path(models_dir)
    if not models_path.is_dir():
        return

    protected = 0
    for file in models_path.iterdir():
        if file.is_file() and file.suffix in MODEL_EXTENSIONS:
            try:
                # Set read-only (remove write permission)
                current = file.stat().st_mode
                file.chmod(current & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
                protected += 1
            except (OSError, PermissionError):
                logger.warning("Could not protect: %s", file)

    if protected:
        logger.info("Write-protected %d model files in %s", protected, models_dir)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SHA-256 model integrity verification"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate", action="store_true",
        help="Generate checksums for all model files",
    )
    group.add_argument(
        "--verify", action="store_true",
        help="Verify all model files against stored checksums",
    )
    group.add_argument(
        "--protect", action="store_true",
        help="Set model files to read-only",
    )
    parser.add_argument(
        "--models-dir", default="models",
        help="Path to models directory (default: models)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    if args.generate:
        checksums = generate_checksums(args.models_dir)
        if not checksums:
            sys.exit(1)
    elif args.verify:
        passed, failed = verify_all_models(args.models_dir)
        if failed:
            sys.exit(1)
    elif args.protect:
        protect_model_files(args.models_dir)


if __name__ == "__main__":
    main()
