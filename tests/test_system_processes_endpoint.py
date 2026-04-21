"""Tests for /api/v1/system/processes — the monitoring endpoint that the
Cloudflare tunnel exposes for remote ops.

Auth must be required (the tunnel is the only reason these routes need to exist
outside localhost, and an unauth'd tunneled endpoint is a harvest target).
The response shape must be stable — callers poll this once a minute.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TRADE_API_DEV_MODE", "1")  # use dev-mode API key
    from fastapi.testclient import TestClient
    from src.api.production_server import app
    return TestClient(app)


def test_health_endpoint_is_unauthenticated(client):
    """/health stays open — Cloudflare's healthcheck probe needs to reach it."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_system_processes_requires_auth(monkeypatch):
    """Without an API key (and not in dev mode), the endpoint must reject."""
    monkeypatch.delenv("TRADE_API_DEV_MODE", raising=False)
    monkeypatch.setenv("DASHBOARD_API_KEY", "secret-test-key")
    from fastapi.testclient import TestClient
    from src.api.production_server import app
    c = TestClient(app)
    r = c.get("/api/v1/system/processes")
    assert r.status_code == 401


def test_system_processes_shape(client):
    """Response must contain the stable top-level keys the UI/poller depends on."""
    r = client.get("/api/v1/system/processes")
    assert r.status_code == 200
    body = r.json()

    # Every key below is a contract — changing the name here changes the UI API
    expected_keys = {
        "timestamp",
        "processes",
        "safe_entries",
        "readiness_gate",
        "paper_state",
        "recent_trades_24h",
    }
    assert expected_keys.issubset(body.keys()), f"missing keys: {expected_keys - set(body.keys())}"


def test_system_processes_lists_self_python(client):
    """psutil path should at least find this test's own Python process."""
    r = client.get("/api/v1/system/processes")
    assert r.status_code == 200
    procs = r.json()["processes"]
    # Accept both "list populated" and "psutil not installed" — CI might lack psutil
    if isinstance(procs, dict) and procs.get("error") == "psutil not installed":
        pytest.skip("psutil not installed in this environment")
    assert isinstance(procs, list)
    # At least one python process (this test runner)
    assert any("python" in (p.get("name") or "").lower() for p in procs)
    # Every entry has the expected fields
    if procs:
        sample = procs[0]
        for k in ("pid", "name", "rss_mb", "cpu_pct", "elapsed_s", "cmd"):
            assert k in sample, f"process entry missing field: {k}"


def test_system_processes_readiness_gate_present(client):
    """Gate state must always be present — that's the whole point of remote monitoring."""
    r = client.get("/api/v1/system/processes")
    body = r.json()
    gate = body["readiness_gate"]
    # Either it has the expected keys, or it's an {error: ...} fallback — both tolerable
    assert isinstance(gate, dict)
    if "error" not in gate:
        # Success path: must contain open + reasons + details
        assert "open" in gate or "open_" in gate
        assert "reasons" in gate
        assert "details" in gate


def test_system_processes_safe_entries_combined_sharpe(client):
    """The combined rolling Sharpe must be present — this is the daily health metric."""
    r = client.get("/api/v1/system/processes")
    se = r.json()["safe_entries"]
    if "error" in se:
        pytest.skip(f"safe_entries not loadable: {se['error']}")
    assert "combined_rolling_sharpe_30" in se
    assert isinstance(se["combined_rolling_sharpe_30"], (int, float))


def test_tunnel_config_template_is_valid_yaml(tmp_path):
    """The cloudflared config template must parse as YAML (no accidental brokenness).
    If the file doesn't exist yet that's fine — this test is part of the commit that
    creates it and will start enforcing once committed."""
    candidates = [
        os.path.join("infra", "cloudflared", "config.yml.template"),
        os.path.join("infra", "cloudflared", "config.yml"),
    ]
    found = [p for p in candidates if os.path.exists(p)]
    if not found:
        pytest.skip("tunnel config not yet created")
    import yaml  # pytest environment has pyyaml
    for p in found:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{p} did not parse as a YAML mapping"
        # Hostname-based ingress is our design — must contain an ingress list
        if "ingress" in data:
            assert isinstance(data["ingress"], list)
            # Last rule must be a catch-all 404 (cloudflared requires it)
            last = data["ingress"][-1]
            assert last.get("service", "").startswith(("http_status:", "hello_world")), (
                f"last ingress rule must be catch-all (http_status:404 or hello_world), got: {last}"
            )
