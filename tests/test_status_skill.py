"""Tests for /status skill — one-shot subsystem verification."""
from __future__ import annotations

import os
from unittest import mock

import pytest


def test_status_skill_registered():
    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    assert reg.get("status") is not None


def test_status_returns_lights_for_all_subsystems(monkeypatch):
    # Force ollama check to fail fast (no real port hit).
    monkeypatch.setenv("OLLAMA_REMOTE_URL", "http://127.0.0.1:1")

    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    r = reg.dispatch("status", {}, invoker="operator")
    assert isinstance(r.data, dict)
    lights = r.data.get("lights") or {}
    expected = {"env", "ollama", "warm_store", "brain_memory",
                "graph", "personas", "readiness"}
    assert expected <= set(lights.keys())
    for v in lights.values():
        assert v in ("green", "yellow", "red")


def test_status_message_includes_env_summary(monkeypatch):
    monkeypatch.setenv("ACT_AGENTIC_LOOP", "1")
    monkeypatch.setenv("ACT_BRAIN_PROFILE", "dense_r1")
    monkeypatch.setenv("OLLAMA_REMOTE_URL", "http://127.0.0.1:1")

    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    r = reg.dispatch("status", {}, invoker="operator")
    assert "ACT_AGENTIC_LOOP" in r.message
    assert "dense_r1" in r.message or "Brain profile" in r.message
    # Always emits the traffic-light list.
    assert "traffic lights" in r.message


def test_status_red_light_when_kill_switch_engaged(monkeypatch):
    monkeypatch.setenv("ACT_DISABLE_AGENTIC_LOOP", "1")
    monkeypatch.setenv("OLLAMA_REMOTE_URL", "http://127.0.0.1:1")

    from src.skills.registry import get_registry
    reg = get_registry(refresh=True)
    r = reg.dispatch("status", {}, invoker="operator")
    assert r.data["lights"]["env"] == "red"
