"""
Thompson-sampling bandit over the strategy repository (M4 of the plan).

Claude-Code-pattern parallel: Claude Code's autonomous loop balances
"stick with what works" vs "try something new" through tool choice. ACT's
equivalent is picking which strategy (DNA lineage in the repository) to
allocate trades to. This module gives that choice a principled posterior:

  * Each strategy has a Beta(α, β) posterior on its win rate.
  * α = prior_alpha + live_wins, β = prior_beta + live_losses.
  * To "try" a strategy, draw p ~ Beta(α, β) and pick the one with the
    highest sample. Winning strategies get more α → more draws → more
    allocation. Losing strategies drift down but aren't zeroed —
    occasional re-sampling keeps exploration alive.

Reuses:
  * src/trading/strategy_repository.py — source of strategies + live
    aggregates (live_wins, live_losses, live_trades).
  * src/orchestration/readiness_gate.py — emergency mode biases toward
    exploitation (more weight on proven champions, less random sampling).

Not a trading gate. The bandit's output is a *recommendation* to the
autonomous loop; the conviction gate still runs on every trade the
recommended strategy produces.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# Uniform Beta(1, 1) prior by default — no opinion before seeing data.
# Raise to Beta(5, 5) via env if you want stronger regularization when
# strategies are fresh (pulls posteriors toward 0.5 WR for longer).
DEFAULT_PRIOR_ALPHA = float(os.getenv("ACT_BANDIT_PRIOR_ALPHA", "1.0"))
DEFAULT_PRIOR_BETA = float(os.getenv("ACT_BANDIT_PRIOR_BETA", "1.0"))

# Emergency-mode tilt toward exploitation. When `is_emergency_mode()` is
# True, the sampler biases strongly toward the strategy with the highest
# posterior mean — less random exploration when we're already underwater.
EMERGENCY_EXPLOIT_BIAS = float(os.getenv("ACT_BANDIT_EMERGENCY_BIAS", "3.0"))


@dataclass
class BanditDraw:
    """One sample from the posterior ensemble."""
    strategy_id: str
    sampled_p: float       # the draw that won the sample
    posterior_mean: float
    alpha: float
    beta: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "sampled_p": round(self.sampled_p, 4),
            "posterior_mean": round(self.posterior_mean, 4),
            "alpha": round(self.alpha, 2),
            "beta": round(self.beta, 2),
        }


def _posterior_params(
    live_wins: int,
    live_losses: int,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
) -> Tuple[float, float]:
    return prior_alpha + float(live_wins), prior_beta + float(live_losses)


def _posterior_mean(alpha: float, beta: float) -> float:
    total = alpha + beta
    if total <= 0:
        return 0.5
    return alpha / total


def sample_from_records(
    records: Sequence,                                    # list of StrategyRecord
    *,
    rng: Optional[random.Random] = None,
    emergency_mode: bool = False,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
) -> Optional[BanditDraw]:
    """Draw one strategy via Thompson sampling.

    `records` is a sequence of StrategyRecord (or any object with
    `.strategy_id`, `.live_wins`, `.live_losses`). Under emergency mode
    we raise every sample to the power of EMERGENCY_EXPLOIT_BIAS — this
    sharpens the ranking (higher-EV strategies pull away from the pack)
    without becoming deterministic.

    Returns None if `records` is empty.
    """
    if not records:
        return None
    rng = rng or random.Random()

    draws: List[BanditDraw] = []
    for rec in records:
        sid = getattr(rec, "strategy_id", None)
        if sid is None:
            continue
        wins = int(getattr(rec, "live_wins", 0) or 0)
        losses = int(getattr(rec, "live_losses", 0) or 0)
        alpha, beta = _posterior_params(wins, losses, prior_alpha, prior_beta)
        p = rng.betavariate(alpha, beta)
        if emergency_mode and EMERGENCY_EXPLOIT_BIAS > 1.0:
            p = p ** EMERGENCY_EXPLOIT_BIAS
        draws.append(BanditDraw(
            strategy_id=sid,
            sampled_p=p,
            posterior_mean=_posterior_mean(alpha, beta),
            alpha=alpha, beta=beta,
        ))
    if not draws:
        return None
    return max(draws, key=lambda d: d.sampled_p)


def top_k_by_posterior_mean(
    records: Sequence,
    k: int = 3,
    *,
    prior_alpha: float = DEFAULT_PRIOR_ALPHA,
    prior_beta: float = DEFAULT_PRIOR_BETA,
) -> List[BanditDraw]:
    """Deterministic ranking — no random draw. Useful for dashboards."""
    out: List[BanditDraw] = []
    for rec in records:
        sid = getattr(rec, "strategy_id", None)
        if sid is None:
            continue
        wins = int(getattr(rec, "live_wins", 0) or 0)
        losses = int(getattr(rec, "live_losses", 0) or 0)
        alpha, beta = _posterior_params(wins, losses, prior_alpha, prior_beta)
        out.append(BanditDraw(
            strategy_id=sid,
            sampled_p=_posterior_mean(alpha, beta),
            posterior_mean=_posterior_mean(alpha, beta),
            alpha=alpha, beta=beta,
        ))
    out.sort(key=lambda d: d.posterior_mean, reverse=True)
    return out[: max(1, int(k))]


# ── Convenience wrapper that reads the repo directly ────────────────────


def sample_from_repo(
    regime: Optional[str] = None,
    status: Optional[str] = None,
    *,
    rng: Optional[random.Random] = None,
    emergency_mode: Optional[bool] = None,
) -> Optional[BanditDraw]:
    """Pull candidates from the strategy repository and sample one.

    `regime` / `status` filter the candidate pool. If `emergency_mode` is
    None we auto-detect via readiness_gate.is_emergency_mode() — callers
    that want a hard override pass True/False.
    """
    try:
        from src.trading.strategy_repository import get_repo
        repo = get_repo()
        # Include candidates + challengers + champion so a brand-new
        # candidate still gets draws. Quarantined/retired excluded.
        if status:
            recs = repo.search(status=status, regime=regime, limit=50)
        else:
            cand = repo.search(status="candidate", regime=regime, limit=50)
            chal = repo.search(status="challenger", regime=regime, limit=50)
            champ = repo.search(status="champion", regime=regime, limit=50)
            recs = cand + chal + champ
    except Exception as e:
        logger.debug("thompson_bandit: repo read failed: %s", e)
        return None

    if emergency_mode is None:
        try:
            from src.orchestration.readiness_gate import is_emergency_mode
            emergency_mode = is_emergency_mode()
        except Exception:
            emergency_mode = False

    return sample_from_records(recs, rng=rng, emergency_mode=bool(emergency_mode))
