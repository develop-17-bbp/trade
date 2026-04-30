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

            # Remote-analyst awareness: when the operator points
            # OLLAMA_REMOTE_URL at a peer Ollama (e.g. 4060 → 5090 over
            # Tailscale), the analyst doesn't live locally — probing it
            # at 127.0.0.1 would fail. Probe each model at the URL that
            # actually serves it.
            _remote = (os.environ.get("OLLAMA_REMOTE_URL") or "").strip().rstrip("/")
            _remote_model = (os.environ.get("OLLAMA_REMOTE_MODEL") or "").strip()

            def _probe_url_for(_model: str) -> str:
                # If remote is configured AND this model matches the
                # remote-pinned analyst, probe the remote URL.
                if _remote and _model and _model == _remote_model:
                    return _remote
                return _ollama

            print(f"  [LLM-GATE] probing scanner={_scanner} + analyst={_analyst} at {_ollama}"
                  + (f" (analyst remote: {_remote})" if _remote else ""))
            # Fail-soft: collect probe results instead of sys.exit(2). When
            # ALL probes fail the bot still boots into shadow / legacy-voter
            # mode and silence_watchdog (Process 11) will fire a CRITICAL
            # alert within 30 min if no warm_store decisions appear. Hard
            # exit was too brittle: a 5090 reboot during a 4060 deploy
            # cycle would crash both bots in lockstep with no audible
            # warning to the operator. Operator can still force the old
            # behavior via ACT_LLM_GATE_HARD_FAIL=1.
            _llm_gate_failures = []
            _llm_gate_passes = []
            for _m in [_scanner, _analyst]:
                if not _m or _m in (_scanner if _m == _analyst else "_"):
                    if _m == _analyst and _analyst == _scanner:
                        continue  # don't double-probe when both roles share a model
                _probe_at = _probe_url_for(_m)
                _payload = _json.dumps({
                    "model": _m, "prompt": "Reply OK.", "stream": False,
                    "keep_alive": -1,
                    "options": {"num_ctx": _ctx, "num_predict": 8, "temperature": 0.0},
                }).encode("utf-8")
                _req = _ureq.Request(
                    f"{_probe_at}/api/generate", data=_payload,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                try:
                    with _ureq.urlopen(_req, timeout=180.0) as _r:
                        _data = _json.loads(_r.read().decode("utf-8"))
                    _txt = (_data.get("response") or "").strip()
                    if not _txt:
                        print(
                            f"  [LLM-GATE] WARN: {_m} @ {_probe_at} returned empty -- "
                            "VRAM eviction likely on that host. Consider dropping "
                            f"OLLAMA_NUM_CTX (currently {_ctx}) to 4096."
                        )
                        _llm_gate_failures.append((_m, _probe_at, "empty"))
                        continue
                    print(f"  [LLM-GATE] OK: {_m} @ {_probe_at} responded {_txt[:40]!r}")
                    _llm_gate_passes.append((_m, _probe_at))
                except (_uerr.URLError, OSError) as _e:
                    print(
                        f"  [LLM-GATE] WARN: {_m} unreachable at {_probe_at}: {_e}. "
                        "Verify Ollama is running at that URL and the model is "
                        "pulled there. For a remote analyst, check that "
                        "OLLAMA_REMOTE_URL points at a reachable peer + the model "
                        "is listed by `ollama list` on that peer."
                    )
                    _llm_gate_failures.append((_m, _probe_at, "unreachable"))
            # Tiered fallback when the primary (remote) analyst is down.
            # Operator request 2026-04-30: keep dual-brain dual. Don't
            # collapse to scanner-doing-both — try a dedicated SMALL
            # backup analyst first, so the bot still has two LLMs.
            #
            # Tier order:
            #   1. Primary remote analyst (e.g. qwen3-coder:30b @ 5090)
            #   2. Local backup analyst (default llama3.2:3b — 2 GB,
            #      fits alongside 7B scanner in 8 GB VRAM, different
            #      model family so it's an actual second brain). Set
            #      ACT_ANALYST_BACKUP_MODEL to pick a different one.
            #   3. Scanner-as-both (last resort, single brain). Opt-out
            #      via ACT_DISABLE_LOCAL_ANALYST_FALLBACK=1.
            _analyst_failed_remote = (
                bool(_remote)
                and any(f[0] == _analyst and f[1] == _remote for f in _llm_gate_failures)
            )
            _scanner_passed = any(p[0] == _scanner for p in _llm_gate_passes)
            _opt_out_fallback = os.environ.get(
                "ACT_DISABLE_LOCAL_ANALYST_FALLBACK", ""
            ).strip() == "1"

            if _analyst_failed_remote and not _opt_out_fallback:
                _backup = (
                    os.environ.get("ACT_ANALYST_BACKUP_MODEL") or "llama3.2:3b"
                ).strip()

                # Tier 2: probe the backup analyst at local Ollama.
                _backup_ok = False
                if _backup and _backup != _analyst and _backup != _scanner:
                    _bp_payload = _json.dumps({
                        "model": _backup, "prompt": "Reply OK.", "stream": False,
                        "keep_alive": -1,
                        "options": {"num_ctx": _ctx, "num_predict": 8, "temperature": 0.0},
                    }).encode("utf-8")
                    _bp_req = _ureq.Request(
                        f"{_ollama}/api/generate", data=_bp_payload,
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    try:
                        with _ureq.urlopen(_bp_req, timeout=180.0) as _bp_r:
                            _bp_data = _json.loads(_bp_r.read().decode("utf-8"))
                        _bp_txt = (_bp_data.get("response") or "").strip()
                        if _bp_txt:
                            _backup_ok = True
                            print(
                                f"  [LLM-GATE] FALLBACK (tier 2): backup analyst "
                                f"{_backup} @ {_ollama} responded {_bp_txt[:40]!r} -- "
                                f"using as analyst (primary {_analyst} unreachable at {_remote}). "
                                "Dual-brain stays dual."
                            )
                            os.environ["ACT_ANALYST_MODEL"] = _backup
                            _llm_gate_failures = [
                                f for f in _llm_gate_failures
                                if not (f[0] == _analyst and f[1] == _remote)
                            ]
                            _llm_gate_passes.append((_backup, _ollama))
                        else:
                            print(
                                f"  [LLM-GATE] FALLBACK (tier 2): backup {_backup} "
                                f"returned empty -- not pulled or VRAM-evicted. "
                                "Falling through to scanner-as-both (tier 3)."
                            )
                    except (_uerr.URLError, OSError) as _bp_e:
                        print(
                            f"  [LLM-GATE] FALLBACK (tier 2): backup {_backup} "
                            f"unreachable at {_ollama}: {_bp_e}. "
                            f"Run 'ollama pull {_backup}' to enable the dual-brain "
                            "fallback. Falling through to scanner-as-both (tier 3)."
                        )

                # Tier 3: scanner serves both roles (last resort, single
                # brain). Only fires if tier 2 didn't.
                if not _backup_ok and _scanner_passed and _scanner != _analyst:
                    print(
                        f"  [LLM-GATE] FALLBACK (tier 3): scanner {_scanner} will "
                        "serve BOTH roles for this boot. Quality drops (single brain). "
                        "Pull a small analyst (e.g. 'ollama pull llama3.2:3b') so the "
                        "next failure-mode boot keeps dual-brain via tier 2."
                    )
                    os.environ["ACT_ANALYST_MODEL"] = _scanner
                    _llm_gate_failures = [
                        f for f in _llm_gate_failures
                        if not (f[0] == _analyst and f[1] == _remote)
                    ]
                    _llm_gate_passes.append((_scanner, _ollama))

            # Summary + escalate when both probes fail. The bot continues
            # in either case, but a both-fail boot deserves a louder line
            # so the operator notices before silence_watchdog fires.
            if _llm_gate_failures and not _llm_gate_passes:
                print(
                    f"  [LLM-GATE] DEGRADED: ALL {len(_llm_gate_failures)} probe(s) failed. "
                    "Bot is booting in shadow / legacy-voter mode. silence_watchdog "
                    "will alert if no decisions land in warm_store within 30 min. "
                    "Set ACT_LLM_GATE_HARD_FAIL=1 to revert to the old fail-fast behavior."
                )
                if os.environ.get("ACT_LLM_GATE_HARD_FAIL", "").strip() == "1":
                    sys.exit(2)
            elif _llm_gate_failures:
                print(
                    f"  [LLM-GATE] PARTIAL: {len(_llm_gate_passes)} model(s) reachable, "
                    f"{len(_llm_gate_failures)} model(s) unreachable. Bot booting; "
                    "the working model will handle both roles via dual_brain fallback."
                )
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
    #
    # ACT_BOX_ROLE can be a single class ("stocks") or comma-separated
    # ("stocks,crypto") so a single box can run multiple asset classes.
    # Use case: 4060 with ACT_BOX_ROLE="stocks,crypto" runs alpaca stocks
    # AND alpaca crypto in parallel threads.
    box_roles = [
        r.strip().lower()
        for r in (os.environ.get('ACT_BOX_ROLE') or '').split(',')
        if r.strip()
    ]
    role_to_class = {
        'crypto': 'CRYPTO', 'stocks': 'STOCK', 'stock': 'STOCK',
        'options': 'OPTIONS',
        'polymarket': 'POLYMARKET',
    }
    allowed_classes = {role_to_class[r] for r in box_roles if r in role_to_class}

    def _exchange_passes(ex_cfg: dict) -> bool:
        cfg_class = (ex_cfg.get('asset_class') or '').upper()
        # When ACT_BOX_ROLE is set, role-matching exchanges are force-enabled
        # AND non-matching exchanges are force-disabled. This lets the same
        # config.yaml ship with `enabled: false` defaults so the 5090 doesn't
        # accidentally fire stocks (which would route SPY through Kraken),
        # while the 4060 with ACT_BOX_ROLE=stocks gets alpaca regardless of
        # the disk default.
        if allowed_classes:
            return cfg_class in allowed_classes
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
        if box_roles:
            print(f"  ACT_BOX_ROLE={','.join(box_roles)}")
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
        if box_roles:
            print(f"  ACT_BOX_ROLE={','.join(box_roles)} - no exchange in config matches this role.")
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
