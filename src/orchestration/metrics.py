"""Prometheus metrics — Phase 1.

Exposes an HTTP endpoint on :9091 that Prometheus (running in WSL2 Docker)
scrapes via host.docker.internal. Metric names match the `act_*` schema
referenced by infra/grafana/dashboards/act_overview.json and the §4.3 table
in the Orchestration Hardening Plan.

Design notes:
- prometheus_client is a SOFT dep. If the package is missing the module
  still imports — every helper becomes a no-op. This keeps the bot boot
  path identical for anyone who hasn't `pip install`-ed Phase 1 deps yet.
- The exporter runs on a daemon thread started by `start_exporter()`.
  Call it exactly once from the executor init. Guarded by env var
  ACT_METRICS_ENABLED (default "1") so a prod rollback is `=0` + restart.
- Never allocate per-cycle. All metrics are pre-registered module-level.
- Histogram buckets sized for the p99 budget from Plan §1.2 (3.5s).
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("ACT_METRICS_ENABLED", "1") == "1"
_PORT = int(os.getenv("ACT_METRICS_PORT", "9091"))

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server
    _HAS_CLIENT = True
except ImportError:
    _HAS_CLIENT = False
    logger.info("prometheus_client not installed — metrics disabled (no-op).")


_started = False
_start_lock = threading.Lock()


if _HAS_CLIENT:
    REGISTRY = CollectorRegistry(auto_describe=True)

    decision_latency_seconds = Histogram(
        "act_decision_latency_seconds",
        "End-to-end latency of one L1→L9 decision cycle.",
        labelnames=("symbol", "action"),
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0, 10.0),
        registry=REGISTRY,
    )

    decisions_total = Counter(
        "act_decisions_total",
        "Count of completed decision cycles.",
        labelnames=("symbol", "action", "consensus"),
        registry=REGISTRY,
    )

    agent_votes_total = Counter(
        "act_agent_votes_total",
        "Agent votes emitted by the L6 consensus layer.",
        labelnames=("agent", "direction"),
        registry=REGISTRY,
    )

    authority_violations_total = Counter(
        "act_authority_violations_total",
        "Authority-rule violations flagged by the guardian.",
        labelnames=("rule",),
        registry=REGISTRY,
    )

    llm_tokens_total = Counter(
        "act_llm_tokens_total",
        "Total LLM tokens consumed, by model and pass.",
        labelnames=("model", "pass_name"),
        registry=REGISTRY,
    )

    equity_usd = Gauge(
        "act_equity_usd",
        "Current account equity in USD.",
        labelnames=("asset",),
        registry=REGISTRY,
    )

    # ── Phase 2: resilience metrics ─────────────────────────────────────
    circuit_breaker_state = Gauge(
        "act_circuit_breaker_state",
        "Named circuit breaker state (0=closed, 1=half_open, 2=open).",
        labelnames=("name",),
        registry=REGISTRY,
    )

    circuit_breaker_trips_total = Counter(
        "act_circuit_breaker_trips_total",
        "Number of times a named circuit breaker has opened.",
        labelnames=("name",),
        registry=REGISTRY,
    )

    stream_publish_total = Counter(
        "act_stream_publish_total",
        "Redis stream publishes, split by success/failure.",
        labelnames=("stream", "result"),
        registry=REGISTRY,
    )

    # ── Phase 4: scheduler + GPU lease metrics ──────────────────────────
    job_runs_total = Counter(
        "act_job_runs_total",
        "PeriodicJob executions, split by success/failure.",
        labelnames=("name", "result"),
        registry=REGISTRY,
    )

    job_duration_seconds = Histogram(
        "act_job_duration_seconds",
        "Wall time of a PeriodicJob run.",
        labelnames=("name",),
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 30.0, 120.0, 600.0),
        registry=REGISTRY,
    )

    gpu_lease_holders = Gauge(
        "act_gpu_lease_holders",
        "Which priority lane currently holds the GPU lease (0 if free).",
        labelnames=("device",),
        registry=REGISTRY,
    )

    # ── Phase 4.5a: learning mesh metrics ───────────────────────────────
    credit_allocation = Gauge(
        "act_credit_allocation",
        "Rolling credit weight assigned to each learning component.",
        labelnames=("component",),
        registry=REGISTRY,
    )

    credit_regression_r2 = Gauge(
        "act_credit_regression_r2",
        "Rolling R² of the credit-assigner regression (target > 0.4).",
        registry=REGISTRY,
    )


def start_exporter(port: Optional[int] = None) -> bool:
    """Start the Prometheus HTTP exporter. Idempotent + safe to call early.

    Returns True if the exporter is now listening, False otherwise (module
    disabled, client missing, or port in use).
    """
    global _started
    if not _ENABLED or not _HAS_CLIENT:
        return False
    with _start_lock:
        if _started:
            return True
        try:
            start_http_server(port if port is not None else _PORT, registry=REGISTRY)
            _started = True
            logger.info("Prometheus exporter listening on :%d", port or _PORT)
            return True
        except OSError as e:
            # Port collision is the common failure (another process already
            # bound). Not fatal — metrics stop, trading continues.
            logger.warning("Prometheus exporter failed to bind :%d: %s", port or _PORT, e)
            return False


def record_decision(symbol: str, action: str, consensus: str, latency_s: float) -> None:
    """Record one completed decision cycle."""
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        decisions_total.labels(symbol=symbol, action=action, consensus=consensus).inc()
        decision_latency_seconds.labels(symbol=symbol, action=action).observe(latency_s)
    except Exception as e:
        logger.debug("record_decision failed: %s", e)


def record_agent_vote(agent: str, direction: int) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        dir_str = "LONG" if direction > 0 else "SHORT" if direction < 0 else "FLAT"
        agent_votes_total.labels(agent=agent, direction=dir_str).inc()
    except Exception:
        pass


def record_authority_violation(rule: str) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        authority_violations_total.labels(rule=rule).inc()
    except Exception:
        pass


def record_llm_tokens(model: str, pass_name: str, tokens: int) -> None:
    if not _ENABLED or not _HAS_CLIENT or tokens <= 0:
        return
    try:
        llm_tokens_total.labels(model=model, pass_name=pass_name).inc(tokens)
    except Exception:
        pass


def set_equity(asset: str, equity: float) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        equity_usd.labels(asset=asset).set(float(equity))
    except Exception:
        pass


# ── Phase 2 emitters ───────────────────────────────────────────────────

_CB_STATE_CODE = {"closed": 0, "half_open": 1, "half-open": 1, "open": 2}


def record_circuit_breaker_state(name: str, state: str) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        code = _CB_STATE_CODE.get(str(state).lower(), 0)
        circuit_breaker_state.labels(name=name).set(code)
    except Exception:
        pass


def record_circuit_breaker_trip(name: str) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        circuit_breaker_trips_total.labels(name=name).inc()
    except Exception:
        pass


def record_stream_publish(stream: str, ok: bool) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        stream_publish_total.labels(stream=stream, result="ok" if ok else "fail").inc()
    except Exception:
        pass


def record_job_run(name: str, ok: bool, duration_s: float) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        job_runs_total.labels(name=name, result="ok" if ok else "fail").inc()
        job_duration_seconds.labels(name=name).observe(max(0.0, duration_s))
    except Exception:
        pass


def record_gpu_lease(device: str, holder_priority: int) -> None:
    """holder_priority: 0 free, 1-4 P1..P4 holders."""
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        gpu_lease_holders.labels(device=device).set(int(holder_priority))
    except Exception:
        pass


def record_credit_allocation(component: str, weight: float) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        credit_allocation.labels(component=component).set(float(weight))
    except Exception:
        pass


def record_credit_r2(r2: float) -> None:
    if not _ENABLED or not _HAS_CLIENT:
        return
    try:
        credit_regression_r2.set(float(r2))
    except Exception:
        pass


@contextmanager
def time_decision(symbol: str) -> Iterator[dict]:
    """Measure one cycle's wall time. Caller fills `action`/`consensus` before exit.

    Usage:
        with time_decision(symbol="BTC") as ctx:
            ...
            ctx["action"] = "LONG"
            ctx["consensus"] = "STRONG"
    """
    import time
    start = time.perf_counter()
    ctx: dict = {"action": "FLAT", "consensus": "UNKNOWN"}
    try:
        yield ctx
    finally:
        dt = time.perf_counter() - start
        record_decision(
            symbol=symbol,
            action=ctx.get("action", "FLAT"),
            consensus=ctx.get("consensus", "UNKNOWN"),
            latency_s=dt,
        )
