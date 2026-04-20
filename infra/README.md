# ACT observability + state stack

WSL2 Docker Compose that hosts the non-GPU services: Redis, Prometheus,
Grafana, Tempo, OpenTelemetry Collector. The bot and Ollama stay native
on Windows for RTX 5090 driver compatibility.

This directory is a **Phase 1 scaffold** — nothing is wired into the
running bot yet. Phase 1 will drop the first Prometheus metrics, OTel
spans, and Grafana dashboard into the existing folders.

## Prerequisites

- WSL2 on Windows 11 (verified installed, kernel 6.6.87.2-1)
- Docker Desktop with WSL2 integration enabled
- At least 4 GB of RAM available to WSL2

## Start the stack

```bash
# From WSL2 (or Windows PowerShell with Docker Desktop running):
cd /mnt/c/Users/convo/trade/infra   # or wherever the repo lives in WSL
docker compose up -d
```

## What you get

| Service | Port (host) | Exposed publicly? | Purpose |
|---|---|---|---|
| Grafana | 3000 | **yes** (via Cloudflare tunnel, Phase 1) | dashboards |
| Prometheus | 9090 | no (localhost only) | metrics TSDB |
| Tempo | 3200 | no (localhost only) | traces backend |
| OTel Collector | 4317, 4318 | no (localhost only) | trace ingress |
| Redis | 6379 | no (localhost only) | streams + hot state |

Cloudflare tunnel config should point only at `http://localhost:3000`.
**Never expose `:9090` or `:4317` publicly** — Prometheus has no auth and
OTel has no rate limiting.

## Environment variables

Set these in a `.env` file next to `docker-compose.yml` (gitignored):

```env
GRAFANA_ADMIN_PASSWORD=<a-real-password>
```

## What Phase 1 will add to this directory

- `grafana/dashboards/act_overview.json` — the first dashboard (equity,
  decision latency, agent votes, authority violations, LLM throughput).
- Metric registration in `src/orchestration/metrics.py` that exposes
  `/metrics` on `:9091` of the host, matching the `act-bot` scrape target
  in `prometheus.yml`.
- OTel SDK setup in `src/orchestration/tracing.py` pointing at
  `host.docker.internal:4317` from the Windows side, or `localhost:4317`
  from within WSL.

## Stop the stack

```bash
docker compose down
```

Use `down -v` to also wipe the named volumes (Grafana dashboards history,
Prometheus TSDB, Redis AOF). Don't do this casually once Phase 1 is live —
you lose metric history.
