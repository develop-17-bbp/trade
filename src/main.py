"""
Main Entry Point
=================
Loads config and starts EMA(8) + LLM trading executors.
Supports multiple exchanges running independently in parallel.
"""

import os
import sys
import logging
import threading
import yaml
from dotenv import load_dotenv

# Force UTF-8 output on Windows (prevents UnicodeEncodeError with cp1252)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trading.executor import TradingExecutor


def run_exchange(config: dict, exchange_name: str, assets: list):
    """Run a single exchange executor in its own thread."""
    # Override config for this exchange
    ex_config = dict(config)
    ex_config['exchange'] = {'name': exchange_name}
    ex_config['assets'] = assets

    try:
        executor = TradingExecutor(ex_config)
        executor.run()
    except Exception as e:
        print(f"  [{exchange_name.upper()}] FATAL: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Load environment variables
    env_path = os.path.join(PROJECT_ROOT, '.env')
    load_dotenv(env_path, override=True)
    print(f"  [ENV] Loaded {env_path}")

    # Configure logging with rotation (50 MB × 3 backups) so long-running
    # paper trading doesn't fill the disk.
    from logging.handlers import RotatingFileHandler
    _sys_log = os.path.join(PROJECT_ROOT, 'logs', 'system_output.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                _sys_log, maxBytes=50 * 1024 * 1024, backupCount=3,
                encoding='utf-8',
            ),
        ],
    )

    # ── LLM health gate ───────────────────────────────────────────────
    # Every previous tick that produced `parse_failure` and zero trades
    # had the same root cause: Ollama silently returning empty text
    # for both qwen models because of VRAM eviction or cold-load. By
    # the time the operator saw "no trades", the bot had been running
    # for 5+ minutes wasting cycles on a broken LLM. This block probes
    # the brain models BEFORE the trade loop starts and aborts with a
    # loud, specific error if either model is unreachable.
    #
    # Override with ACT_SKIP_LLM_HEALTH_GATE=1 if you want the bot to
    # boot anyway (e.g. testing data-only paths with the LLM down).
    if os.environ.get("ACT_SKIP_LLM_HEALTH_GATE", "").strip() != "1":
        try:
            import json as _json
            import urllib.request as _ureq
            import urllib.error as _uerr
            from src.ai.dual_brain import _resolve_profile  # type: ignore

            _profile = _resolve_profile()
            _scanner = _profile.get("scanner_model", "")
            _analyst = _profile.get("analyst_model", "")
            _ollama = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            _ctx = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

            print(f"  [LLM-GATE] probing scanner={_scanner} + analyst={_analyst} at {_ollama}")
            for _m in [_scanner, _analyst]:
                if not _m or _m in (_scanner if _m == _analyst else "_"):
                    if _m == _analyst and _analyst == _scanner:
                        continue  # don't double-probe when both roles share a model
                _payload = _json.dumps({
                    "model": _m, "prompt": "Reply OK.", "stream": False,
                    "keep_alive": -1,
                    "options": {"num_ctx": _ctx, "num_predict": 8, "temperature": 0.0},
                }).encode("utf-8")
                _req = _ureq.Request(
                    f"{_ollama}/api/generate", data=_payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                try:
                    with _ureq.urlopen(_req, timeout=180.0) as _r:
                        _data = _json.loads(_r.read().decode("utf-8"))
                    _txt = (_data.get("response") or "").strip()
                    if not _txt:
                        print(
                            f"  [LLM-GATE] FAIL: {_m} returned empty -- VRAM eviction "
                            "likely. Drop OLLAMA_NUM_CTX (currently "
                            f"{_ctx}) to 4096 and restart, OR set "
                            "ACT_SKIP_LLM_HEALTH_GATE=1 to boot anyway."
                        )
                        sys.exit(2)
                    print(f"  [LLM-GATE] OK: {_m} responded {_txt[:40]!r}")
                except (_uerr.URLError, OSError) as _e:
                    print(
                        f"  [LLM-GATE] FAIL: {_m} unreachable: {_e}. "
                        f"Verify Ollama is running at {_ollama} and the model is "
                        "pulled (`ollama pull {_m}`). Set "
                        "ACT_SKIP_LLM_HEALTH_GATE=1 to boot without LLM."
                    )
                    sys.exit(2)
        except SystemExit:
            raise
        except Exception as _e:
            print(f"  [LLM-GATE] (skipped — probe raised {type(_e).__name__}: {_e})")

    # Load config
    config_path = os.path.join(PROJECT_ROOT, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Check for multi-exchange config
    exchanges_all = config.get('exchanges', [])

    # Filter by `enabled: true` AND match ACT_BOX_ROLE if both are set.
    # 5090 (crypto box): only robinhood + polymarket should fire.
    # 4060 (stocks box): only alpaca should fire.
    # Without ACT_BOX_ROLE: enabled-only filter (safe default).
    box_role = (os.environ.get('ACT_BOX_ROLE') or '').strip().lower()
    asset_class_for_role = {
        'crypto': 'CRYPTO', 'stocks': 'STOCK', 'stock': 'STOCK',
        'polymarket': 'POLYMARKET',
    }.get(box_role)

    def _exchange_passes(ex_cfg: dict) -> bool:
        cfg_class = (ex_cfg.get('asset_class') or '').upper()
        # When ACT_BOX_ROLE is set, the role-matching exchange is force-enabled
        # AND non-matching exchanges are force-disabled. This lets the same
        # config.yaml ship with `enabled: false` defaults so the 5090 doesn't
        # accidentally fire stocks (which would route SPY through Kraken),
        # while the 4060 with ACT_BOX_ROLE=stocks gets alpaca regardless of
        # the disk default.
        if asset_class_for_role:
            return cfg_class == asset_class_for_role
        # No role filter — fall back to honouring the explicit enabled flag.
        return bool(ex_cfg.get('enabled', True))

    exchanges = [ex for ex in exchanges_all if _exchange_passes(ex)]
    skipped = [ex['name'] for ex in exchanges_all if ex not in exchanges]
    if skipped:
        print(f"  [MAIN] Skipping disabled / role-filtered exchanges: {skipped}")

    if len(exchanges) >= 2:
        # Multi-exchange mode — run each in its own thread
        print("=" * 60)
        print("  MULTI-EXCHANGE MODE")
        print(f"  Exchanges: {[e['name'] for e in exchanges]}")
        if box_role:
            print(f"  ACT_BOX_ROLE={box_role}")
        print("=" * 60)

        threads = []
        for ex_cfg in exchanges:
            name = ex_cfg['name']
            assets = ex_cfg.get('assets', config.get('assets', ['BTC', 'ETH']))

            t = threading.Thread(
                target=run_exchange,
                args=(config, name, assets),
                name=f"executor-{name}",
                daemon=True,
            )
            threads.append(t)
            t.start()
            print(f"  Started {name.upper()} executor: {assets}")

        # Wait for all threads (or Ctrl+C)
        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\n  [SHUTDOWN] Stopping all exchanges...")

    elif len(exchanges) == 1:
        # Single exchange from new config format
        ex_cfg = exchanges[0]
        config['exchange'] = {'name': ex_cfg['name']}
        config['assets'] = ex_cfg.get('assets', config.get('assets', ['BTC', 'ETH']))
        executor = TradingExecutor(config)
        executor.run()

    elif not exchanges and exchanges_all:
        # All exchanges filtered out by enabled/role flags. Don't fall through
        # to legacy single-exchange — that would re-enable the disabled robinhood
        # config silently. Exit cleanly so the operator sees the filter result.
        print("=" * 60)
        print("  [MAIN] No exchanges enabled for this box.")
        if box_role:
            print(f"  ACT_BOX_ROLE={box_role} - no exchange in config matches this role.")
        print("  Edit config.yaml to set `enabled: true` on the exchange you want, ")
        print("  or unset ACT_BOX_ROLE to skip role filtering.")
        print("=" * 60)
        sys.exit(0)

    else:
        # Legacy single exchange config
        executor = TradingExecutor(config)
        executor.run()


if __name__ == '__main__':
    main()
