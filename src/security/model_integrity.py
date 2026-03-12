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
        if file.is_file() and file.suffix in MODEL_EXTENSIONS:
            digest = _sha256(file)
            # Store paths with forward slashes for cross-platform consistency.
            key = file.as_posix()
            checksums[key] = digest
            logger.info("Checksum generated: %s -> %s", key, digest)

    checksums_path = models_path / CHECKSUMS_FILE
    with open(checksums_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)
    logger.info("Checksums saved to %s (%d files)", checksums_path, len(checksums))

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
    """Verify a single model file against its stored checksum.

    Args:
        model_path: Path to the model file (e.g. ``models/lgbm_btc.txt``).
        models_dir: Directory containing ``checksums.json``.

    Returns:
        True if the file matches its stored checksum, False otherwise.
    """
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


def verify_all_models(models_dir: str = "models") -> dict[str, bool]:
    """Verify every model listed in checksums.json.

    Returns:
        Dict mapping file paths to verification results.
    """
    checksums = _load_checksums(models_dir)
    if not checksums:
        return {}

    results: dict[str, bool] = {}
    for key, expected in checksums.items():
        file_path = Path(key)
        if not file_path.is_file():
            logger.warning("MISSING: %s", key)
            results[key] = False
            continue

        actual = _sha256(file_path)
        passed = actual == expected
        results[key] = passed
        if passed:
            logger.info("PASS: %s", key)
        else:
            logger.warning("FAIL: %s (expected %s, got %s)", key, expected, actual)

    passed_count = sum(results.values())
    total = len(results)
    logger.info("Verification complete: %d/%d passed", passed_count, total)
    return results


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
        results = verify_all_models(args.models_dir)
        if not results or not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
