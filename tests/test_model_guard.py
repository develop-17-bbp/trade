"""Regression tests for ACT_FORBID_MODELS circuit-breaker."""

from __future__ import annotations

import pytest

from src.ai.model_guard import is_forbidden, resolve_safe_model


def test_empty_forbid_list_allows_anything(monkeypatch):
    monkeypatch.delenv("ACT_FORBID_MODELS", raising=False)
    assert not is_forbidden("deepseek-r1:7b")
    assert not is_forbidden("qwen3-coder:30b")
    assert resolve_safe_model(["deepseek-r1:7b"]) == "deepseek-r1:7b"


def test_family_head_blocks_all_sizes(monkeypatch):
    monkeypatch.setenv("ACT_FORBID_MODELS", "deepseek-r1")
    assert is_forbidden("deepseek-r1:7b")
    assert is_forbidden("deepseek-r1:32b")
    assert is_forbidden("deepseek-r1")
    assert not is_forbidden("qwen3-coder:30b")


def test_exact_tag_blocks_only_that_size(monkeypatch):
    monkeypatch.setenv("ACT_FORBID_MODELS", "deepseek-r1:7b")
    assert is_forbidden("deepseek-r1:7b")
    # 32b is NOT blocked when only 7b is listed
    assert not is_forbidden("deepseek-r1:32b")


def test_resolve_skips_forbidden(monkeypatch):
    monkeypatch.setenv("ACT_FORBID_MODELS", "deepseek-r1:7b")
    got = resolve_safe_model([
        "deepseek-r1:7b",
        "",
        "qwen2.5-coder:7b",
    ])
    assert got == "qwen2.5-coder:7b"


def test_resolve_returns_none_when_all_forbidden(monkeypatch):
    monkeypatch.setenv("ACT_FORBID_MODELS", "deepseek-r1, qwen")
    assert resolve_safe_model(["deepseek-r1:7b", "qwen3-coder:30b"]) is None


def test_resolve_handles_empty_and_none(monkeypatch):
    monkeypatch.delenv("ACT_FORBID_MODELS", raising=False)
    assert resolve_safe_model([]) is None
    assert resolve_safe_model(["", None, "  "]) is None


def test_resolve_first_non_forbidden_wins(monkeypatch):
    monkeypatch.setenv("ACT_FORBID_MODELS", "deepseek-r1")
    got = resolve_safe_model([
        "deepseek-r1:7b",
        "deepseek-r1:32b",
        "qwen2.5-coder:7b",
        "qwen3-coder:30b",
    ])
    assert got == "qwen2.5-coder:7b"


@pytest.mark.parametrize(
    "model_id",
    ["DeepSeek-R1:7B", "deepseek-r1:7B", "DEEPSEEK-R1:7b"],
)
def test_case_insensitive_match(monkeypatch, model_id):
    monkeypatch.setenv("ACT_FORBID_MODELS", "deepseek-r1:7b")
    assert is_forbidden(model_id)


def test_unset_env_in_resolve_explicit_arg(monkeypatch):
    monkeypatch.delenv("ACT_FORBID_MODELS", raising=False)
    got = resolve_safe_model(
        ["deepseek-r1:7b", "qwen3-coder:30b"],
        forbid_list=["deepseek-r1"],
    )
    assert got == "qwen3-coder:30b"
