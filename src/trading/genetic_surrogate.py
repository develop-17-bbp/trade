"""
Surrogate-Assisted Pre-Filter (P2 of genetic-loop audit)
=========================================================
Trains a fast regressor on prior (genome → backtest fitness) pairs so
future genome candidates can be pre-screened cheaply, skipping the
expensive backtest_dna evaluation for predicted-low-fitness candidates.

Background: ACT's `backtest_dna` runs a per-bar loop over hundreds of
bars per DNA; on a 50-strategy population × 10 generations that's 500
backtests per cycle. A surrogate trained on past observations predicts
fitness in ~microseconds, letting us evaluate only the top fraction.

Surrogate model (in priority order):
  1. LightGBM regressor (if `lightgbm` is installed) — best accuracy
  2. scikit-learn GradientBoostingRegressor (if `sklearn` is installed)
  3. Hand-rolled k-NN regressor (no extra dependency) — graceful fallback

Anti-overfit guards:
  * ε-greedy: ε fraction of candidates are evaluated unconditionally
    (default 0.20). Prevents the surrogate from getting stuck in a
    local fitness pocket.
  * Re-train cadence: surrogate retrained every N observations
    (default 200), not every generation — avoids overfitting to a
    single short evolution run.
  * Bias logging: every prediction stores residual after the true
    backtest is run, so the operator can detect surrogate drift.
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURROGATE_LOG_PATH = os.path.join(PROJECT_ROOT, "logs", "genetic_surrogate.jsonl")


# ── Feature extraction (gene-dict → numeric vector) ─────────────────────────


def _genome_to_features(
    genes: Dict[str, float], entry_rule: str, exit_rule: str,
    feature_keys: List[str],
) -> np.ndarray:
    """Map a genome to a fixed numeric feature vector.

    Numeric genes go directly; entry_rule/exit_rule are hashed into
    a one-hot-equivalent 2-int suffix (cheap; tree models handle it).
    """
    v = np.zeros(len(feature_keys) + 2)
    for i, k in enumerate(feature_keys):
        try:
            v[i] = float(genes.get(k, 0.0))
        except (TypeError, ValueError):
            v[i] = 0.0
    # categorical-as-int (CFG hash is stable per process; for larger
    # systems the operator should hash & embed externally).
    v[-2] = float(hash(str(entry_rule)) % 997)
    v[-1] = float(hash(str(exit_rule)) % 997)
    return v


# ── Backend selectors ───────────────────────────────────────────────────────


def _try_lightgbm():
    try:
        import lightgbm as lgb  # type: ignore
        return lgb
    except ImportError:
        return None


def _try_sklearn_gbr():
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
        return GradientBoostingRegressor
    except ImportError:
        return None


# ── Hand-rolled k-NN fallback ──────────────────────────────────────────────


class _KNNRegressor:
    """Tiny weighted k-NN regressor (Euclidean) for fallback case."""

    def __init__(self, k: int = 5):
        self.k = k
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None
        self._mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if X.size == 0:
            return
        self._X = X.astype(float)
        self._y = y.astype(float)
        self._mean = float(np.mean(y)) if len(y) else 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._X is None or len(self._X) == 0:
            return np.full(len(X), self._mean)
        out = np.zeros(len(X))
        for i, x in enumerate(X):
            dists = np.linalg.norm(self._X - x, axis=1)
            k_eff = min(self.k, len(dists))
            idx = np.argsort(dists)[:k_eff]
            weights = 1 / (dists[idx] + 1e-9)
            out[i] = float(np.sum(self._y[idx] * weights) / np.sum(weights))
        return out


# ── Main surrogate ──────────────────────────────────────────────────────────


@dataclass
class SurrogatePrediction:
    dna_name: str
    predicted_fitness: float
    is_kept: bool
    rng_kept: bool = False  # was kept by epsilon-greedy not surrogate
    actual_fitness: Optional[float] = None
    residual: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "dna_name": self.dna_name,
            "predicted_fitness": round(self.predicted_fitness, 4),
            "is_kept": self.is_kept,
            "rng_kept": self.rng_kept,
        }
        if self.actual_fitness is not None:
            out["actual_fitness"] = round(self.actual_fitness, 4)
        if self.residual is not None:
            out["residual"] = round(self.residual, 4)
        return out


class SurrogateFilter:
    """Surrogate model for genome → fitness prediction."""

    def __init__(
        self,
        feature_keys: Optional[List[str]] = None,
        epsilon: float = 0.20,
        retrain_every_n: int = 200,
        backend: str = "auto",
        knn_k: int = 5,
    ):
        # Default to all known indicator-gene names
        if feature_keys is None:
            try:
                from src.trading.genetic_strategy_engine import INDICATOR_GENES
                feature_keys = sorted(INDICATOR_GENES.keys())
            except ImportError:
                feature_keys = []
        self.feature_keys = feature_keys
        self.epsilon = float(epsilon)
        self.retrain_every_n = int(retrain_every_n)
        self.backend = backend

        self._observations_X: List[np.ndarray] = []
        self._observations_y: List[float] = []
        self._n_since_train: int = 0
        self._model: Any = None
        self._knn_k = knn_k

    # -- Backend selection --------------------------------------------------

    def _build_model(self):
        if self.backend in ("auto", "lightgbm"):
            lgb = _try_lightgbm()
            if lgb is not None:
                return lgb.LGBMRegressor(
                    n_estimators=150, learning_rate=0.05,
                    num_leaves=31, min_child_samples=5,
                    verbose=-1,
                )
        if self.backend in ("auto", "sklearn"):
            gbr = _try_sklearn_gbr()
            if gbr is not None:
                return gbr(n_estimators=100, max_depth=4, learning_rate=0.05)
        return _KNNRegressor(k=self._knn_k)

    # -- Observation accumulation ------------------------------------------

    def add_observation(
        self, genes: Dict[str, float], entry_rule: str, exit_rule: str,
        fitness: float,
    ) -> None:
        x = _genome_to_features(genes, entry_rule, exit_rule, self.feature_keys)
        self._observations_X.append(x)
        self._observations_y.append(float(fitness))
        self._n_since_train += 1

    def add_population(self, population: Iterable[Any]) -> None:
        for dna in population:
            try:
                self.add_observation(
                    genes=dict(dna.genes),
                    entry_rule=dna.entry_rule,
                    exit_rule=dna.exit_rule,
                    fitness=float(dna.fitness or 0.0),
                )
            except Exception:
                continue

    # -- Training ----------------------------------------------------------

    def train(self, force: bool = False) -> bool:
        n = len(self._observations_y)
        if n < 30:
            return False
        if not force and self._n_since_train < self.retrain_every_n:
            return False

        X = np.array(self._observations_X, dtype=float)
        y = np.array(self._observations_y, dtype=float)
        try:
            model = self._build_model()
            model.fit(X, y)
            self._model = model
            self._n_since_train = 0
            return True
        except Exception as exc:
            logger.warning("Surrogate train failed: %s", exc)
            return False

    # -- Prediction --------------------------------------------------------

    def _predict_array(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or X.size == 0:
            return np.zeros(len(X))
        try:
            return np.asarray(self._model.predict(X)).ravel()
        except Exception as exc:
            logger.warning("Surrogate predict failed: %s", exc)
            return np.zeros(len(X))

    def predict_fitness(
        self, genes: Dict[str, float], entry_rule: str, exit_rule: str,
    ) -> float:
        x = _genome_to_features(genes, entry_rule, exit_rule, self.feature_keys)[None, :]
        return float(self._predict_array(x)[0])

    # -- Population filter -------------------------------------------------

    def filter_population(
        self, population: List[Any], keep_fraction: float = 0.5,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[Any], List[SurrogatePrediction]]:
        """Return the subset of population to evaluate this generation.

        keep_fraction: top fraction of predicted-fitness retained.
        epsilon: also unconditionally keep ε fraction picked at random.

        Returns (kept_population, predictions) for logging.
        """
        rng = rng or random.Random()
        if not self._model or not population:
            # No model yet → keep all (cold-start phase).
            return list(population), [
                SurrogatePrediction(
                    dna_name=getattr(d, "name", "?"),
                    predicted_fitness=0.0, is_kept=True, rng_kept=True,
                ) for d in population
            ]

        keep_n = max(1, int(len(population) * keep_fraction))
        eps_n = int(len(population) * self.epsilon) if self.epsilon > 0 else 0

        X = np.array([
            _genome_to_features(
                d.genes, d.entry_rule, d.exit_rule, self.feature_keys,
            ) for d in population
        ])
        preds = self._predict_array(X)

        order = np.argsort(-preds)  # descending
        kept_idx = set(order[:keep_n].tolist())

        # epsilon-greedy: also keep random low-predicted ones
        remaining = [i for i in range(len(population)) if i not in kept_idx]
        if remaining and eps_n > 0:
            random_keep = rng.sample(remaining, min(eps_n, len(remaining)))
            for i in random_keep:
                kept_idx.add(i)

        predictions: List[SurrogatePrediction] = []
        kept_pop: List[Any] = []
        for i, dna in enumerate(population):
            is_kept = i in kept_idx
            rng_kept = is_kept and i not in order[:keep_n].tolist()
            predictions.append(SurrogatePrediction(
                dna_name=getattr(dna, "name", "?"),
                predicted_fitness=float(preds[i]),
                is_kept=is_kept,
                rng_kept=rng_kept,
            ))
            if is_kept:
                kept_pop.append(dna)

        return kept_pop, predictions

    # -- Residual logging --------------------------------------------------

    def log_residuals(
        self, predictions: List[SurrogatePrediction],
        evaluated_population: List[Any],
    ) -> Dict[str, Any]:
        """After backtest is done, compute residuals and persist."""
        name_to_actual = {
            getattr(d, "name", "?"): float(getattr(d, "fitness", 0.0) or 0.0)
            for d in evaluated_population
        }
        residuals: List[float] = []
        for p in predictions:
            if p.dna_name in name_to_actual:
                p.actual_fitness = name_to_actual[p.dna_name]
                p.residual = p.actual_fitness - p.predicted_fitness
                residuals.append(p.residual)

        try:
            os.makedirs(os.path.dirname(SURROGATE_LOG_PATH), exist_ok=True)
            with open(SURROGATE_LOG_PATH, "a") as f:
                for p in predictions:
                    if p.actual_fitness is None:
                        continue
                    f.write(json.dumps(p.to_dict()) + "\n")
        except Exception:
            pass

        if not residuals:
            return {"n_residuals": 0, "mae": 0.0, "bias": 0.0}
        return {
            "n_residuals": len(residuals),
            "mae": float(np.mean(np.abs(residuals))),
            "bias": float(np.mean(residuals)),
            "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        }


__all__ = [
    "SurrogateFilter",
    "SurrogatePrediction",
]
