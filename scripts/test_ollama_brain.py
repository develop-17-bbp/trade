"""Direct Ollama health probe for the active dual_brain profile.

Tests scanner + analyst models with a TINY prompt and reports raw
response, latency, and VRAM state. Use this BEFORE start_all.ps1 (to
verify base health) AND AFTER (to verify pre-load worked).

Catches:
  * model not pulled
  * model returns empty (VRAM OOM, eviction, or wrong tag)
  * Ollama service down
  * num_ctx oversize

Usage:
    cd C:\\Users\\admin\\trade
    python scripts/test_ollama_brain.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
TINY_PROMPT = "Reply with the single word OK."


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _http_post_json(url: str, payload: dict, timeout: float = 60.0):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
            return data, time.time() - t0, None
    except error.HTTPError as e:
        return None, time.time() - t0, f"HTTP {e.code}: {e.reason}"
    except (error.URLError, OSError) as e:
        return None, time.time() - t0, f"network: {e}"
    except json.JSONDecodeError as e:
        return None, time.time() - t0, f"bad json: {e}"


def _http_get_json(url: str, timeout: float = 5.0):
    try:
        with request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def show_vram_state() -> None:
    print(f"\n[VRAM] /api/ps state at {OLLAMA_URL}:")
    data = _http_get_json(f"{OLLAMA_URL}/api/ps")
    if data is None:
        print(f"  (Ollama unreachable at {OLLAMA_URL})")
        return
    models = data.get("models") or []
    if not models:
        print("  (no models resident)")
        return
    for m in models:
        size_vram = m.get("size_vram", 0)
        size_vram_gb = size_vram / (1024 ** 3) if isinstance(size_vram, (int, float)) else 0
        print(f"  - {m.get('name', '?')}  "
              f"vram={size_vram_gb:.1f} GB  "
              f"ctx={m.get('context_length', '?')}  "
              f"expires={m.get('expires_at', '?')}")


def test_model(model_id: str, num_ctx: int) -> bool:
    print(f"\n[TEST] {model_id}  ctx={num_ctx}")
    payload = {
        "model": model_id,
        "prompt": TINY_PROMPT,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": 8, "temperature": 0.0},
    }
    data, dt, err = _http_post_json(
        f"{OLLAMA_URL}/api/generate", payload, timeout=180.0,
    )
    if err is not None:
        _fail(f"{err}  ({dt:.1f}s)")
        return False
    if data is None:
        _fail(f"no response  ({dt:.1f}s)")
        return False
    response_text = (data.get("response") or "").strip()
    if not response_text:
        _fail(f"EMPTY response  ({dt:.1f}s)")
        print(f"    full payload keys: {list(data.keys())}")
        if "error" in data:
            print(f"    error field: {data['error']}")
        return False
    _ok(f"{response_text[:60]!r}  ({dt:.1f}s)")
    return True


def main() -> int:
    print("=" * 60)
    print(" Ollama dual-brain health probe")
    print("=" * 60)
    print(f" ollama: {OLLAMA_URL}")

    show_vram_state()

    try:
        from src.ai.dual_brain import _resolve_profile  # type: ignore
    except Exception as e:
        _fail(f"could not import dual_brain: {e}")
        return 1

    profile = _resolve_profile()
    scanner = profile.get("scanner_model", "")
    analyst = profile.get("analyst_model", "")
    print(f"\n[PROFILE] scanner={scanner}  analyst={analyst}")

    try:
        num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
    except ValueError:
        num_ctx = 16384

    failures = []
    if scanner:
        if not test_model(scanner, num_ctx):
            failures.append(scanner)
    if analyst and analyst != scanner:
        if not test_model(analyst, num_ctx):
            failures.append(analyst)

    show_vram_state()

    print()
    print("=" * 60)
    if not failures:
        print(" RESULT: every brain model responds. The bot's LLM path is healthy.")
        print(" If the bot still shows parse_failure, it's a request-shape issue,")
        print(" not a model-availability issue.")
        print("=" * 60)
        return 0

    print(f" RESULT: {len(failures)} model(s) failed: {failures}")
    print()
    print(" Most likely causes (most common first):")
    print("  1. VRAM oversubscribed -- combined model ctx exceeds GPU memory.")
    print("     FIX: setx OLLAMA_NUM_CTX 8192  + STOP_ALL/START_ALL.")
    print("  2. Model not pulled. Check: ollama list")
    print("     FIX: ollama pull qwen3-coder:30b")
    print("  3. Ollama service crashed. FIX: net stop ollama && ollama serve")
    print("  4. First-load timed out. FIX: warm with `ollama run <model> hi`.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())
