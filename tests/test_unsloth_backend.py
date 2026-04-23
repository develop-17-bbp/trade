"""Tests for src/ai/unsloth_backend.py — UnslothQLoRABackend stub-testable path.

Real training requires unsloth + GPU + several model downloads — those
can't run in CI. This test file exercises:
  * Module imports cleanly without unsloth installed.
  * UnslothUnavailable raised on train() when unsloth is missing.
  * HF-id resolution for every default mapping + env override.
  * infer() HTTP JSON handling (Ollama /api/generate) via mocked urllib.
  * _format_rows_for_sft shapes prompt/completion pairs correctly for
    both chat-template and plain-USER/ASSISTANT fallback tokenizers.
  * _sample_to_prompt contract.
  * _ollama_has_model / _ollama_create behavior under happy + error paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


def test_module_imports_without_unsloth():
    """The backend module must import cleanly on machines without unsloth."""
    import src.ai.unsloth_backend as ub
    assert hasattr(ub, "UnslothQLoRABackend")
    assert hasattr(ub, "UnslothUnavailable")


def test_hf_id_resolves_default_map():
    from src.ai.unsloth_backend import _resolve_hf_id
    assert _resolve_hf_id("deepseek-r1:32b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    assert _resolve_hf_id("deepseek-r1:7b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    assert _resolve_hf_id("qwen3-coder:30b") == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert _resolve_hf_id("qwen2.5-coder:7b") == "Qwen/Qwen2.5-Coder-7B-Instruct"


def test_hf_id_strips_act_suffix():
    """Hot-swapped tags like `deepseek-r1:32b-act-1712345` should resolve
    back to the base family's HF repo."""
    from src.ai.unsloth_backend import _resolve_hf_id
    assert _resolve_hf_id("deepseek-r1:32b:act-1712345") == \
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"


def test_hf_id_env_override(monkeypatch):
    monkeypatch.setenv(
        "ACT_UNSLOTH_HF_MAP",
        json.dumps({"deepseek-r1:7b": "my-org/my-custom-fork"}),
    )
    from src.ai.unsloth_backend import _resolve_hf_id
    assert _resolve_hf_id("deepseek-r1:7b") == "my-org/my-custom-fork"


def test_hf_id_unknown_passes_through():
    from src.ai.unsloth_backend import _resolve_hf_id
    assert _resolve_hf_id("some-obscure:model") == "some-obscure:model"


def test_train_raises_when_unsloth_missing(tmp_path, monkeypatch):
    """With unsloth not installed, train() must return False (logging
    UnslothUnavailable internally) — never raising out of the call."""
    from src.ai.unsloth_backend import UnslothQLoRABackend
    # Simulate unsloth import failure.
    import builtins
    real_import = builtins.__import__

    def _no_unsloth(name, *a, **kw):
        if name == "unsloth" or name.startswith("unsloth."):
            raise ImportError("simulated missing unsloth")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_unsloth)
    backend = UnslothQLoRABackend(output_root=str(tmp_path))
    ok = backend.train("deepseek-r1:7b",
                      [{"prompt": "p", "completion": "c"} for _ in range(20)],
                      "deepseek-r1:7b-act-test")
    assert ok is False


def test_infer_hits_ollama_generate_endpoint(monkeypatch, tmp_path):
    """infer() should POST to /api/generate and return the 'response' field."""
    from src.ai.unsloth_backend import UnslothQLoRABackend
    backend = UnslothQLoRABackend(output_root=str(tmp_path),
                                   ollama_host="http://fake-ollama:11434")

    captured = {}

    class _FakeResp:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({"response": '{"direction": "LONG"}'}).encode()

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = req.data.decode("utf-8") if req.data else ""
        captured["method"] = req.get_method()
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    out = backend.infer("deepseek-r1:7b-act-1",
                        {"asset": "BTC", "scanner_tag": {"top_signals": ["trend"]}})
    assert out == '{"direction": "LONG"}'
    assert captured["url"] == "http://fake-ollama:11434/api/generate"
    assert captured["method"] == "POST"
    body = json.loads(captured["body"])
    assert body["model"] == "deepseek-r1:7b-act-1"
    assert body["stream"] is False


def test_infer_returns_empty_on_http_error(monkeypatch, tmp_path):
    from src.ai.unsloth_backend import UnslothQLoRABackend
    backend = UnslothQLoRABackend(output_root=str(tmp_path))

    def _boom(req, timeout=None):
        raise ConnectionError("ollama down")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    out = backend.infer("m", {"asset": "BTC"})
    assert out == ""


def test_infer_returns_empty_on_bad_json(monkeypatch, tmp_path):
    from src.ai.unsloth_backend import UnslothQLoRABackend
    backend = UnslothQLoRABackend(output_root=str(tmp_path))

    class _BadResp:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return b"not json"

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _BadResp())
    out = backend.infer("m", {"asset": "BTC"})
    assert out == ""


def test_format_rows_chat_template():
    """Tokenizer with a chat_template — row produces a 'text' key with
    both user and assistant turns rendered by the template."""
    from src.ai.unsloth_backend import _format_rows_for_sft

    class _TokWithTemplate:
        chat_template = "<template>"
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
            return "\n".join(f"[{m['role'].upper()}]{m['content']}" for m in msgs)

    rows = _format_rows_for_sft(
        [{"prompt": "what is BTC", "completion": "a digital currency"}],
        _TokWithTemplate(),
    )
    assert len(rows) == 1
    assert "[USER]what is BTC" in rows[0]["text"]
    assert "[ASSISTANT]a digital currency" in rows[0]["text"]


def test_format_rows_fallback_when_no_template():
    """Tokenizer with no chat_template — falls back to plain
    'USER:.../ASSISTANT:...' encoding."""
    from src.ai.unsloth_backend import _format_rows_for_sft

    class _TokBare:
        chat_template = None

    rows = _format_rows_for_sft(
        [{"prompt": "p1", "completion": "c1"}, {"prompt": "p2", "completion": "c2"}],
        _TokBare(),
    )
    assert len(rows) == 2
    for r in rows:
        assert r["text"].startswith("USER:")
        assert "ASSISTANT:" in r["text"]


def test_format_rows_skips_empty():
    from src.ai.unsloth_backend import _format_rows_for_sft

    class _Tok:
        chat_template = None

    rows = _format_rows_for_sft(
        [
            {"prompt": "ok", "completion": "ok"},
            {"prompt": "", "completion": "missing prompt"},
            {"prompt": "missing completion", "completion": ""},
            {"prompt": "ok2", "completion": "ok2"},
        ],
        _Tok(),
    )
    assert len(rows) == 2


def test_sample_to_prompt_includes_asset_and_scanner_tag():
    from src.ai.unsloth_backend import _sample_to_prompt
    out = _sample_to_prompt({
        "asset": "ETH", "scanner_tag": {"top_signals": ["breakout"]},
    })
    assert "ETH" in out
    assert "breakout" in out
    assert "JSON" in out
    assert "direction" in out


def test_sample_to_prompt_non_dict():
    from src.ai.unsloth_backend import _sample_to_prompt
    assert _sample_to_prompt("just a string") == "just a string"


def test_ollama_has_model_true(monkeypatch):
    from src.ai.unsloth_backend import _ollama_has_model

    class _Resp:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False
        def read(self):
            return json.dumps({"models": [{"name": "deepseek-r1:7b"},
                                           {"name": "something-else"}]}).encode()

    monkeypatch.setattr("urllib.request.urlopen",
                        lambda req, timeout=None: _Resp())
    assert _ollama_has_model("deepseek-r1:7b") is True
    assert _ollama_has_model("deepseek-r1:7b-act-1") is True  # prefix match
    assert _ollama_has_model("never-pulled:999b") is False


def test_ollama_has_model_unreachable(monkeypatch):
    from src.ai.unsloth_backend import _ollama_has_model

    def _boom(req, timeout=None):
        raise ConnectionError("down")
    monkeypatch.setattr("urllib.request.urlopen", _boom)
    assert _ollama_has_model("anything") is False


def test_ollama_create_success(monkeypatch, tmp_path):
    from src.ai.unsloth_backend import _ollama_create
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM x.gguf\n", encoding="utf-8")

    captured = {}

    class _FakeResult:
        returncode = 0
        stderr = ""

    def _fake_run(argv, **kw):
        captured["argv"] = argv
        return _FakeResult()

    monkeypatch.setattr("subprocess.run", _fake_run)
    assert _ollama_create("my-model:act-1", modelfile) is True
    assert captured["argv"][0] == "ollama"
    assert captured["argv"][1] == "create"
    assert captured["argv"][2] == "my-model:act-1"


def test_ollama_create_nonzero_returns_false(monkeypatch, tmp_path):
    from src.ai.unsloth_backend import _ollama_create
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("x", encoding="utf-8")

    class _Fail:
        returncode = 1
        stderr = "model exists"

    monkeypatch.setattr("subprocess.run", lambda *a, **kw: _Fail())
    assert _ollama_create("m", modelfile) is False


def test_backend_integrates_with_dual_brain_trainer_protocol(tmp_path):
    """UnslothQLoRABackend should be drop-in-compatible with the
    TrainerBackend Protocol in dual_brain_trainer."""
    from src.ai.unsloth_backend import UnslothQLoRABackend
    backend = UnslothQLoRABackend(output_root=str(tmp_path))
    assert hasattr(backend, "train")
    assert hasattr(backend, "infer")
    # Method signatures match what dual_brain_trainer calls.
    import inspect
    sig = inspect.signature(backend.train)
    assert "base_model" in sig.parameters
    assert "sft_rows" in sig.parameters
    assert "out_tag" in sig.parameters
    sig_infer = inspect.signature(backend.infer)
    assert "model_id" in sig_infer.parameters
    assert "sample" in sig_infer.parameters
