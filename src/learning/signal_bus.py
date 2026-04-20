"""Cross-learner signal bus — Phase 4.5b (§4.3).

Thin namespacing layer over orchestration/streams.py. Every learner
publishes to `model.signal.{producer}.{signal_name}` and subscribes to
the signals it needs — the stream names are fixed here so producers and
consumers agree on the schema without passing strings around.

Signal envelope:
    {
      "producer": "l1_rl" | "l7_lora" | "l9_genetic" | "calibrator",
      "signal":   "value_estimate" | ... (see SIGNAL_* below),
      "model_version": str,    # stale-drop key; consumers skip older versions
      "ts_ns":    int,
      "payload":  dict,        # signal-specific
    }

Soft-fail throughout: a dead Redis means publish() returns None and
subscribe() yields nothing, which each learner treats as "no update".
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, Optional, Tuple

logger = logging.getLogger(__name__)


# Producers
PRODUCER_L1_RL = "l1_rl"
PRODUCER_L7_LORA = "l7_lora"
PRODUCER_L9_GENETIC = "l9_genetic"
PRODUCER_CALIBRATOR = "calibrator"

# Signals (Plan §4.1)
SIGNAL_VALUE_ESTIMATE = "value_estimate"          # RL V(s)
SIGNAL_POLICY_ENTROPY = "policy_entropy"          # RL H(π)
SIGNAL_ADVANTAGE = "advantage"                     # RL A(s,a)
SIGNAL_TOKEN_LOGPROB = "token_logprob"            # LoRA log-prob of chosen dir
SIGNAL_PROMPT_LOSS = "prompt_loss"                 # LoRA per-prompt loss
SIGNAL_CALIBRATION_ERROR = "calibration_error"    # LoRA calibration gap
SIGNAL_TOP_K_DNA = "top_k_dna"                    # Genetic top-K fingerprints
SIGNAL_FITNESS_DIST = "fitness_dist"              # Genetic regime-conditioned fitness
SIGNAL_MUTATION_VAR = "mutation_var"              # Genetic mutation variance
SIGNAL_BRIER = "brier"                            # Calibrator Brier rolling 50
SIGNAL_ECE = "ece"                                # Calibrator expected calibration error


def _stream_name(producer: str, signal: str) -> str:
    return f"model.signal.{producer}.{signal}"


def publish(producer: str, signal: str, payload: Dict[str, Any], model_version: str = "") -> Optional[str]:
    """Publish one signal tick. Never raises — dead Redis returns None."""
    from src.orchestration.streams import publish as _stream_publish
    if not model_version:
        model_version = os.getenv("ACT_MODEL_VERSION", "dev")
    envelope = {
        "producer": producer,
        "signal": signal,
        "model_version": model_version,
        "ts_ns": time.time_ns(),
        "payload": payload,
    }
    return _stream_publish(_stream_name(producer, signal), envelope)


def subscribe(
    producer: str,
    signal: str,
    group: str,
    consumer: str,
    count: int = 16,
    block_ms: int = 250,
    accept_versions: Optional[set] = None,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Stream messages as (msg_id, envelope). Drops stale-version messages.

    `accept_versions`: if set, only yield envelopes whose model_version is
    in the set — protects consumers from training on signals produced by a
    prior checkpoint that was already rolled back (Plan §4.3).
    """
    from src.orchestration.streams import ensure_group, read_group
    stream = _stream_name(producer, signal)
    ensure_group(stream, group, start_id="$")
    for mid, env in read_group(stream, group, consumer, count=count, block_ms=block_ms):
        if accept_versions is not None and env.get("model_version") not in accept_versions:
            # Stale / unknown version — ack handled by caller, we just skip.
            yield mid, {"_stale": True, **env}
            continue
        yield mid, env


def ack(producer: str, signal: str, group: str, msg_id: str) -> bool:
    from src.orchestration.streams import ack as _ack
    return _ack(_stream_name(producer, signal), group, msg_id)


# Fast helpers for common signals — no envelope ceremony at call site.

def publish_rl_value(state_key: str, v: float, entropy: float, model_version: str = "") -> None:
    publish(PRODUCER_L1_RL, SIGNAL_VALUE_ESTIMATE,
            {"state_key": state_key, "v": float(v), "entropy": float(entropy)},
            model_version=model_version)


def publish_lora_logprob(prompt_hash: str, direction: str, logprob: float, loss: float, model_version: str = "") -> None:
    publish(PRODUCER_L7_LORA, SIGNAL_TOKEN_LOGPROB,
            {"prompt_hash": prompt_hash, "direction": direction,
             "logprob": float(logprob), "loss": float(loss)},
            model_version=model_version)


def publish_genetic_top_k(dna_list: list, regime: str, model_version: str = "") -> None:
    publish(PRODUCER_L9_GENETIC, SIGNAL_TOP_K_DNA,
            {"regime": regime, "dna": dna_list},
            model_version=model_version)


def publish_calibrator(brier: float, ece: float, sample_size: int, model_version: str = "") -> None:
    publish(PRODUCER_CALIBRATOR, SIGNAL_BRIER,
            {"brier": float(brier), "ece": float(ece), "n": int(sample_size)},
            model_version=model_version)
