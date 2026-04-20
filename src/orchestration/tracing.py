"""OpenTelemetry tracing — Phase 1.

One span per decision cycle, exported over OTLP/gRPC to the collector that
runs in WSL2 Docker (see infra/docker-compose.yml). The collector forwards
to Tempo; Grafana queries Tempo via the provisioned datasource.

Design notes:
- OTel packages are SOFT deps. If they're missing, the module still imports
  and `decision_span` becomes a no-op context manager — the bot keeps running.
- BatchSpanProcessor exports asynchronously on a background thread. Export
  errors never block the caller. A dead collector produces warnings but
  zero latency impact on the decision path.
- trace_id returned by `decision_span` mirrors the envelope's decision_id as
  a hex-encoded parent — this way the audit JSONL and OTel spans cross-link
  on a single grep without waiting for Phase 4 full log→trace correlation.
- Sensitive attributes (api keys, secrets, tokens, bearers) are scrubbed at
  the SpanProcessor level before export. This is a defense-in-depth duplicate
  of the collector-side drop list in infra/otel/otel-config.yaml.
"""

from __future__ import annotations

import logging
import os
import platform
import threading
from contextlib import contextmanager
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("ACT_TRACING_ENABLED", "1") == "1"
_SERVICE_NAME = os.getenv("ACT_SERVICE_NAME", "act-bot")

_SECRET_ATTR_SUBSTRINGS = ("api_key", "secret", "token", "bearer", "password")


def _default_endpoint() -> str:
    """Pick the right OTLP endpoint for the current host.

    - Windows host running native Python (the bot): OTel collector lives in
      WSL2 Docker, reachable via the same localhost port mapped in
      docker-compose.yml (127.0.0.1:4317 → container :4317).
    - Inside WSL2 itself: localhost:4317 still works.
    - Inside a container on the same compose network: `otel-collector:4317`
      is the service name — callers can override via ACT_OTEL_ENDPOINT.
    """
    override = os.getenv("ACT_OTEL_ENDPOINT")
    if override:
        return override
    # Both Windows native and WSL2 can reach the mapped port on localhost.
    _ = platform.system()  # kept for future per-OS branching
    return "http://localhost:4317"


try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    _HAS_OTEL = True
except Exception as _e:
    # Catch Exception (not just ImportError): the protobuf ↔ Python-version
    # compatibility chain can raise TypeError during module init on Python
    # 3.14 with protobuf <5. We want the bot to boot in that case, not crash.
    _HAS_OTEL = False
    logger.info("opentelemetry unavailable — tracing disabled (no-op). reason=%r", _e)


_provider_started = False
_start_lock = threading.Lock()
_tracer = None


if _HAS_OTEL:

    class _ScrubProcessor(SpanProcessor):
        """Strip obvious secrets from span attributes before export.

        Runs BEFORE BatchSpanProcessor so the exporter only sees clean spans.
        Not a substitute for not-logging-secrets-in-the-first-place — defense
        in depth against a future caller who attaches a raw auth header.
        """

        def on_start(self, span, parent_context=None):  # pragma: no cover - trivial
            pass

        def on_end(self, span):
            attrs = getattr(span, "_attributes", None)
            if not attrs:
                return
            # attrs is a BoundedAttributes (dict-like). Build a redacted copy.
            keys = list(attrs.keys()) if hasattr(attrs, "keys") else []
            for k in keys:
                lk = k.lower() if isinstance(k, str) else ""
                if any(s in lk for s in _SECRET_ATTR_SUBSTRINGS):
                    try:
                        attrs[k] = "[REDACTED]"
                    except Exception:
                        pass

        def shutdown(self):  # pragma: no cover - trivial
            pass

        def force_flush(self, timeout_millis: int = 30000):  # pragma: no cover - trivial
            return True


def init_tracer() -> Optional[object]:
    """Initialize the global TracerProvider exactly once. Idempotent.

    Returns the tracer, or None if tracing is disabled / OTel missing.
    """
    global _provider_started, _tracer
    if not _ENABLED or not _HAS_OTEL:
        return None
    with _start_lock:
        if _provider_started:
            return _tracer
        try:
            resource = Resource.create({
                "service.name": _SERVICE_NAME,
                "service.version": os.getenv("ACT_VERSION", "v8.0"),
                "host.name": platform.node(),
            })
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(_ScrubProcessor())
            provider.add_span_processor(BatchSpanProcessor(
                OTLPSpanExporter(endpoint=_default_endpoint(), insecure=True),
                max_queue_size=2048,
                max_export_batch_size=256,
                schedule_delay_millis=2000,
            ))
            trace.set_tracer_provider(provider)
            _tracer = trace.get_tracer(_SERVICE_NAME)
            _provider_started = True
            logger.info("OTel tracer initialized (endpoint=%s)", _default_endpoint())
            return _tracer
        except Exception as e:
            logger.warning("OTel tracer init failed: %s", e)
            return None


@contextmanager
def decision_span(decision_id: str, symbol: str) -> Iterator[dict]:
    """Open one span for a full decision cycle.

    Yields a dict that callers can populate as the cycle progresses:
        with decision_span(decision_id, symbol) as ctx:
            ctx["final_action"] = "LONG"
            ctx["consensus"] = "STRONG"
            ctx["trace_id"] = ...  # filled in by us on entry
            ...

    When tracing is off or OTel is missing, the span is a no-op but `ctx`
    still carries `trace_id == decision_id` so the executor can always
    attach a stable trace_id to the audit row.
    """
    ctx: dict = {"trace_id": decision_id}
    if not _ENABLED or not _HAS_OTEL or _tracer is None:
        yield ctx
        return

    with _tracer.start_as_current_span(
        name="decision_cycle",
        attributes={
            "act.decision_id": decision_id,
            "act.symbol": symbol,
        },
    ) as span:
        # OTel's trace_id is a 128-bit int. Expose hex so the JSONL audit row
        # and Tempo search use the same string representation.
        sc = span.get_span_context()
        if sc and sc.trace_id:
            ctx["trace_id"] = format(sc.trace_id, "032x")
        try:
            yield ctx
        finally:
            # Let the caller annotate the span with final values.
            final_action = ctx.get("final_action")
            if final_action:
                span.set_attribute("act.final_action", str(final_action))
            consensus = ctx.get("consensus")
            if consensus:
                span.set_attribute("act.consensus", str(consensus))
            err = ctx.get("error")
            if err:
                span.set_attribute("act.error", str(err)[:500])
