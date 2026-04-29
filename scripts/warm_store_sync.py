"""warm_store delta sync — bidirectional, append-only, parquet-over-Tailscale.

Process 11 in START_ALL.ps1 on both 5090 and 4060. Each tick:
    1. Read local warm_store rows since last successful sync.
    2. Serialize delta as parquet (compact + schema-stable).
    3. scp to peer's data/sync_inbox/.
    4. Drop a `.sync_ready` marker on peer.
    5. (peer-side) ingest delta via INSERT OR IGNORE, idempotent.

Why bidirectional: the 4060 needs crypto rows from 5090 (analyst nightly
trains on combined corpus); the 5090 needs stocks rows from 4060
(per-class readiness gate, dual-asset finetune router).

Conflict model: zero overlap by design. 5090 only writes
`asset_class='CRYPTO'`; 4060 only writes `'STOCK'`. INSERT OR IGNORE
on the (decision_id) primary key dedupes any same-row-twice arrivals.

Bandwidth: ~3000 rows/day total → ~50 KB/day compressed parquet.
Trivial over Tailscale (sub-ms latency).

Usage:
    python -m scripts.warm_store_sync                # forever loop
    python -m scripts.warm_store_sync --once         # one cycle then exit
    python -m scripts.warm_store_sync --recv         # receive-only mode (incoming watcher)
    python -m scripts.warm_store_sync --send         # send-only mode (outgoing batcher)
    python -m scripts.warm_store_sync --dry-run      # build delta but don't ship
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("warm_store_sync")

DEFAULT_DB        = os.getenv("ACT_WARM_DB_PATH", str(REPO_ROOT / "data" / "warm_store.sqlite"))
DEFAULT_INBOX     = REPO_ROOT / "data" / "sync_inbox"
DEFAULT_OUTBOX    = REPO_ROOT / "data" / "sync_outbox"
DEFAULT_MARKER    = REPO_ROOT / "scripts" / ".warm_store_sync_marker.json"
DEFAULT_POLL_S    = float(os.getenv("ACT_SYNC_POLL_S", "60"))

PEER_HOST    = os.getenv("ACT_SYNC_PEER_HOST")           # tailscale alias e.g. 'act5090'
PEER_REMOTE_DIR = os.getenv("ACT_SYNC_PEER_DIR", "C:/Users/admin/trade")  # peer's repo

LOCAL_HOST = os.getenv("ACT_SYNC_LOCAL_HOST") or socket.gethostname()


# ── Marker (per-peer last-sync sequence) ────────────────────────────


def _read_marker(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_marker(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ── Outbox: read local rows since last sync, ship to peer ──────────


def _new_decision_rows(db_path: str, since_ts_ns: int) -> List[dict]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.execute(
            """SELECT decision_id, trace_id, symbol, ts_ns, direction, confidence,
                      consensus, veto, raw_signal, final_action, authority_violations,
                      payload_json, component_signals, plan_json, self_critique,
                      asset_class, venue
                 FROM decisions
                WHERE ts_ns > ?
                ORDER BY ts_ns ASC""",
            (since_ts_ns,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _new_outcome_rows(db_path: str, since_exit_ts: float) -> List[dict]:
    if not Path(db_path).exists():
        return []
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        cur = conn.execute(
            """SELECT decision_id, symbol, direction, entry_price, exit_price,
                      pnl_pct, pnl_usd, duration_s, exit_reason, regime,
                      entry_ts, exit_ts, payload_json, asset_class, venue
                 FROM outcomes
                WHERE exit_ts > ?
                ORDER BY exit_ts ASC""",
            (since_exit_ts,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _write_delta_parquet(out_dir: Path, decisions: List[dict], outcomes: List[dict]) -> Optional[Path]:
    """Write decisions + outcomes as one parquet bundle. Returns path or None on failure."""
    if not decisions and not outcomes:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = out_dir / f"warm_delta_{LOCAL_HOST}_{ts}"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        bundle = {
            "decisions": pa.Table.from_pylist(decisions),
            "outcomes":  pa.Table.from_pylist(outcomes),
        }
        path = base.with_suffix(".parquet")
        # Use one parquet per table written into a directory.
        bundle_dir = base
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if decisions:
            pq.write_table(bundle["decisions"], str(bundle_dir / "decisions.parquet"))
        if outcomes:
            pq.write_table(bundle["outcomes"],  str(bundle_dir / "outcomes.parquet"))
        return bundle_dir
    except ImportError:
        # Fallback to JSON if pyarrow isn't installed (operator hasn't run pip yet).
        logger.warning("warm_store_sync: pyarrow missing; falling back to JSON")
        bundle_dir = base
        bundle_dir.mkdir(parents=True, exist_ok=True)
        if decisions:
            (bundle_dir / "decisions.json").write_text(
                json.dumps(decisions, default=str), encoding="utf-8")
        if outcomes:
            (bundle_dir / "outcomes.json").write_text(
                json.dumps(outcomes, default=str), encoding="utf-8")
        return bundle_dir


def _ship_to_peer(bundle_dir: Path, peer_host: str, peer_dir: str, dry_run: bool = False) -> bool:
    """scp the bundle dir to peer's data/sync_inbox/ then drop a .sync_ready marker."""
    if not bundle_dir.exists():
        return False
    target_remote = f"{peer_host}:{peer_dir}/data/sync_inbox/"
    if dry_run:
        logger.info("warm_store_sync: DRY-RUN would scp %s -> %s", bundle_dir, target_remote)
        return True
    try:
        # scp -r preserves directory structure
        cmd = ["scp", "-r", str(bundle_dir), target_remote]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            logger.warning("warm_store_sync: scp failed: %s", result.stderr.strip()[:200])
            return False
        # Drop marker remotely so the peer's receiver knows to ingest
        marker_name = f"{bundle_dir.name}.sync_ready"
        ssh_cmd = [
            "ssh", peer_host,
            f"cd '{peer_dir}/data/sync_inbox' && powershell.exe -Command \"New-Item -ItemType File -Path '{marker_name}' -Force\"",
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning("warm_store_sync: marker drop failed: %s", result.stderr.strip()[:200])
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning("warm_store_sync: scp/ssh timeout")
        return False
    except Exception as e:
        logger.warning("warm_store_sync: ship failed: %s", e)
        return False


# ── Inbox: receive deltas, ingest into local warm_store ─────────────


def _ingest_delta(bundle_dir: Path, db_path: str) -> Tuple[int, int]:
    """Read parquet/json bundle, insert rows idempotently. Returns (n_decisions, n_outcomes)."""
    decisions: List[dict] = []
    outcomes:  List[dict] = []

    # Try parquet first, fall back to JSON.
    decisions_pq = bundle_dir / "decisions.parquet"
    outcomes_pq  = bundle_dir / "outcomes.parquet"
    decisions_jn = bundle_dir / "decisions.json"
    outcomes_jn  = bundle_dir / "outcomes.json"

    try:
        import pyarrow.parquet as pq
        if decisions_pq.exists():
            decisions = pq.read_table(str(decisions_pq)).to_pylist()
        if outcomes_pq.exists():
            outcomes  = pq.read_table(str(outcomes_pq)).to_pylist()
    except ImportError:
        pass
    if not decisions and decisions_jn.exists():
        decisions = json.loads(decisions_jn.read_text(encoding="utf-8"))
    if not outcomes and outcomes_jn.exists():
        outcomes  = json.loads(outcomes_jn.read_text(encoding="utf-8"))

    if not decisions and not outcomes:
        return 0, 0

    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        n_dec = 0
        n_out = 0
        for d in decisions:
            try:
                cur = conn.execute(
                    """INSERT OR IGNORE INTO decisions
                          (decision_id, trace_id, symbol, ts_ns, direction, confidence, consensus,
                           veto, raw_signal, final_action, authority_violations, payload_json,
                           component_signals, plan_json, self_critique, asset_class, venue)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        d.get("decision_id"), d.get("trace_id"), d.get("symbol"),
                        int(d.get("ts_ns") or 0),
                        int(d.get("direction") or 0),
                        float(d.get("confidence") or 0.0),
                        d.get("consensus") or "UNKNOWN",
                        int(d.get("veto") or 0),
                        int(d.get("raw_signal") or 0),
                        d.get("final_action") or "FLAT",
                        d.get("authority_violations") or "[]",
                        d.get("payload_json") or "{}",
                        d.get("component_signals") or "{}",
                        d.get("plan_json") or "{}",
                        d.get("self_critique") or "{}",
                        d.get("asset_class") or "CRYPTO",
                        d.get("venue") or "robinhood",
                    ),
                )
                # rowcount is per-statement: 1 if inserted, 0 if INSERT OR IGNORE deduped.
                n_dec += int(cur.rowcount or 0)
            except Exception as e:
                logger.debug("ingest decision %s: %s", d.get("decision_id"), e)
        # Outcomes have an autoincrement primary key, so dedup by (decision_id, exit_ts).
        for o in outcomes:
            try:
                # Skip if (decision_id, exit_ts) already present — best we can do
                # without a UNIQUE constraint, and adding one now would migrate
                # existing rows. INSERT OR IGNORE on the autoincrement PK won't
                # dedupe; explicit pre-check handles it.
                exit_ts = float(o.get("exit_ts") or 0.0)
                exists = conn.execute(
                    "SELECT 1 FROM outcomes WHERE decision_id = ? AND exit_ts = ? LIMIT 1",
                    (o.get("decision_id"), exit_ts),
                ).fetchone()
                if exists:
                    continue
                conn.execute(
                    """INSERT INTO outcomes
                          (decision_id, symbol, direction, entry_price, exit_price, pnl_pct, pnl_usd,
                           duration_s, exit_reason, regime, entry_ts, exit_ts, payload_json,
                           asset_class, venue)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        o.get("decision_id"), o.get("symbol"), o.get("direction") or "",
                        float(o.get("entry_price") or 0.0),
                        float(o.get("exit_price") or 0.0),
                        float(o.get("pnl_pct") or 0.0),
                        float(o.get("pnl_usd") or 0.0),
                        float(o.get("duration_s") or 0.0),
                        o.get("exit_reason") or "MANUAL",
                        o.get("regime") or "unknown",
                        float(o.get("entry_ts") or 0.0),
                        exit_ts,
                        o.get("payload_json") or "{}",
                        o.get("asset_class") or "CRYPTO",
                        o.get("venue") or "robinhood",
                    ),
                )
                n_out += 1
            except Exception as e:
                logger.debug("ingest outcome %s: %s", o.get("decision_id"), e)
        conn.commit()
        return n_dec, n_out
    finally:
        conn.close()


# ── Send + receive cycle ────────────────────────────────────────────


def send_cycle(db_path: str, peer_host: str, peer_dir: str,
               outbox: Path, marker_path: Path, dry_run: bool = False) -> dict:
    if not peer_host:
        return {"sent": 0, "skipped": "no_peer"}
    marker = _read_marker(marker_path)
    since_ts_ns = int(marker.get("last_decision_ts_ns") or 0)
    since_exit  = float(marker.get("last_outcome_exit_ts") or 0.0)

    decisions = _new_decision_rows(db_path, since_ts_ns)
    outcomes  = _new_outcome_rows(db_path, since_exit)
    if not decisions and not outcomes:
        return {"sent": 0, "skipped": "nothing_new"}

    bundle_dir = _write_delta_parquet(outbox, decisions, outcomes)
    if bundle_dir is None:
        return {"sent": 0, "skipped": "write_failed"}

    ok = _ship_to_peer(bundle_dir, peer_host, peer_dir, dry_run=dry_run)
    if not ok:
        return {"sent": 0, "skipped": "ship_failed", "bundle": str(bundle_dir)}

    if decisions:
        marker["last_decision_ts_ns"] = max(int(d["ts_ns"]) for d in decisions if d.get("ts_ns"))
    if outcomes:
        marker["last_outcome_exit_ts"] = max(float(o["exit_ts"]) for o in outcomes if o.get("exit_ts"))
    marker["last_send_ts"] = time.time()
    if not dry_run:
        _write_marker(marker_path, marker)

    return {
        "sent":      len(decisions) + len(outcomes),
        "decisions": len(decisions),
        "outcomes":  len(outcomes),
        "bundle":    str(bundle_dir),
    }


def receive_cycle(db_path: str, inbox: Path) -> dict:
    """Look for *.sync_ready markers in inbox, ingest the matching bundle dir."""
    if not inbox.exists():
        return {"received": 0, "skipped": "no_inbox"}
    total_dec = 0
    total_out = 0
    bundles = 0
    for marker in sorted(inbox.glob("*.sync_ready")):
        bundle_dir = inbox / marker.stem
        if not bundle_dir.exists():
            marker.unlink(missing_ok=True)
            continue
        n_d, n_o = _ingest_delta(bundle_dir, db_path)
        total_dec += n_d
        total_out += n_o
        bundles += 1
        # Leave the bundle dir on disk for one cycle as audit trail; cleanup
        # happens via a separate retention pass operator can run.
        marker.unlink(missing_ok=True)
    return {
        "received":  total_dec + total_out,
        "decisions": total_dec,
        "outcomes":  total_out,
        "bundles":   bundles,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--once",     action="store_true", help="single cycle then exit")
    p.add_argument("--send",     action="store_true", help="send-only mode")
    p.add_argument("--recv",     action="store_true", help="receive-only mode")
    p.add_argument("--dry-run",  action="store_true", help="build deltas but don't scp")
    p.add_argument("--db-path",  default=DEFAULT_DB)
    p.add_argument("--inbox",    default=str(DEFAULT_INBOX))
    p.add_argument("--outbox",   default=str(DEFAULT_OUTBOX))
    p.add_argument("--marker",   default=str(DEFAULT_MARKER))
    p.add_argument("--peer-host", default=PEER_HOST)
    p.add_argument("--peer-dir",  default=PEER_REMOTE_DIR)
    p.add_argument("--poll-s",    type=float, default=DEFAULT_POLL_S)
    p.add_argument("--verbose",   action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    inbox  = Path(args.inbox)
    outbox = Path(args.outbox)
    marker_path = Path(args.marker)

    do_send = args.send or (not args.recv)
    do_recv = args.recv or (not args.send)

    def one_cycle() -> dict:
        out = {}
        if do_recv:
            out["recv"] = receive_cycle(args.db_path, inbox)
        if do_send:
            out["send"] = send_cycle(
                args.db_path, args.peer_host or "", args.peer_dir,
                outbox, marker_path, dry_run=args.dry_run,
            )
        return out

    if args.once:
        out = one_cycle()
        logger.info("warm_store_sync once: %s", out)
        return 0

    logger.info(
        "warm_store_sync: starting (peer=%s db=%s poll=%.0fs)",
        args.peer_host, args.db_path, args.poll_s,
    )
    while True:
        try:
            out = one_cycle()
            if any(v.get("sent", 0) or v.get("received", 0) for v in out.values() if isinstance(v, dict)):
                logger.info("warm_store_sync: %s", out)
        except Exception as e:
            logger.exception("warm_store_sync cycle: %s", e)
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
