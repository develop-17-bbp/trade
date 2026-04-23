"""Skill: /regime-check — current regime + champion + top bandit picks."""
from __future__ import annotations

from typing import Any, Dict

from src.skills.registry import SkillResult


def run(args: Dict[str, Any]) -> SkillResult:
    data: Dict[str, Any] = {}

    # Current regime via existing regime classifier (best-effort).
    regime = "UNKNOWN"
    try:
        from src.trading.regime_classifier import classify_current_regime
        regime = classify_current_regime() or "UNKNOWN"
    except Exception:
        # regime_classifier may require live price data; don't fail the skill.
        pass
    data["regime"] = regime

    # Strategy repository snapshot — champion + top-3 posterior-mean picks.
    try:
        from src.trading.strategy_repository import get_repo
        from src.learning.thompson_bandit import top_k_by_posterior_mean

        repo = get_repo()
        champ = repo.current_champion(regime=regime)
        data["champion"] = champ.to_dict() if champ else None
        counts = repo.count_by_status()
        data["counts_by_status"] = counts

        candidates = repo.search(regime=regime, limit=50)
        top = top_k_by_posterior_mean(candidates, k=3)
        data["top3"] = [d.to_dict() for d in top]
    except Exception as e:
        return SkillResult(ok=False, error=f"repo_read_failed: {type(e).__name__}: {e}",
                           data=data)

    champ_line = (f"{champ.name} (sharpe={champ.live_sharpe:.2f}, n={champ.live_trades})"
                  if champ else "no champion")
    top_line = ", ".join(f"{d.strategy_id[:8]}({d.posterior_mean:.2f})" for d in top) or "none"
    message = (
        f"Regime: {regime}\n"
        f"Champion: {champ_line}\n"
        f"Counts by status: {counts}\n"
        f"Top-3 by posterior mean: {top_line}"
    )
    return SkillResult(ok=True, message=message, data=data)
