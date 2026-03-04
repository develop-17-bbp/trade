from typing import List, Optional


class BaseForecaster:
    """Abstract base class for price forecasters."""

    def fit(self, series: List[float]):
        raise NotImplementedError

    def predict(self, series: List[float], horizon: int = 1) -> List[float]:
        """Return a prediction series of same length as input.
        Default: echo last price."""
        return [series[-1]] * len(series)

    def generate_signal(self, series: List[float]) -> List[float]:
        """Produce a directional signal (+1/-1/0) based on prediction.

        For each point i we compare predicted value to actual price and
        return +1 if forecast>price, -1 if < price, else 0.
        """
        preds = self.predict(series)
        out: List[float] = []
        for p, x in zip(preds, series):
            if p > x:
                out.append(1.0)
            elif p < x:
                out.append(-1.0)
            else:
                out.append(0.0)
        # if preds shorter than series, pad with zeros
        out.extend([0.0] * (len(series) - len(out)))
        return out


class FinGPTForecaster(BaseForecaster):
    def __init__(self, model_name: str = "finGPT-default"):
        try:
            import fingpt
            # placeholder; actual API may differ
            self.model = fingpt.FinGPT(model=model_name)
        except Exception:
            self.model = None

    def predict(self, series: List[float], horizon: int = 1) -> List[float]:
        if self.model is None:
            return super().predict(series, horizon)
        try:
            # hypothetical interface
            preds = self.model.forecast(series, horizon=horizon)
            return preds
        except Exception:
            return super().predict(series, horizon)


class LightGBMForecaster(BaseForecaster):
    def __init__(self):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor()
            self._trained = False
        except Exception:
            self.model = None
            self._trained = False

    def fit(self, series: List[float]):
        if self.model is None:
            return
        # simple one-step autoregressive training using previous values
        X = []
        y = []
        for i in range(1, len(series)):
            X.append([series[i-1]])
            y.append(series[i])
        if X:
            self.model.fit(X, y)
            self._trained = True

    def predict(self, series: List[float], horizon: int = 1) -> List[float]:
        if self.model is None or not self._trained:
            return super().predict(series, horizon)
        preds: List[float] = []
        last = series[-1]
        for _ in range(len(series)):
            try:
                val = self.model.predict([[last]])[0]
            except Exception:
                val = last
            preds.append(val)
            last = val
        return preds


class NBeatsForecaster(BaseForecaster):
    def __init__(self):
        try:
            from pytorch_forecasting.models import NBeats
            import torch
            self.model = None
            self._torch = torch
        except Exception:
            self.model = None
            self._torch = None

    def fit(self, series: List[float]):
        # placeholder: user should implement training via pytorch_forecasting dataset
        pass

    def predict(self, series: List[float], horizon: int = 1) -> List[float]:
        # fallback to naive if not trained
        return super().predict(series, horizon)


class TFTForecaster(BaseForecaster):
    def __init__(self):
        try:
            from pytorch_forecasting.models import TemporalFusionTransformer
            import torch
            self.model = None
            self._torch = torch
        except Exception:
            self.model = None
            self._torch = None

    def fit(self, series: List[float]):
        pass

    def predict(self, series: List[float], horizon: int = 1) -> List[float]:
        return super().predict(series, horizon)


class ForecastingEngine:
    """Wrapper to manage multiple forecasters according to config."""

    def __init__(self, cfg: Optional[dict] = None):
        cfg = cfg or {}
        self.fg = None
        self.lgbm = None
        self.nbeats = None
        self.tft = None
        if cfg.get("use_fingpt"):
            self.fg = FinGPTForecaster(cfg.get("fingpt_model", "finGPT-default"))
        if cfg.get("use_lgbm"):
            self.lgbm = LightGBMForecaster()
        if cfg.get("use_nbeats"):
            self.nbeats = NBeatsForecaster()
        if cfg.get("use_tft"):
            self.tft = TFTForecaster()

    def generate_signal(self, series: List[float]) -> List[float]:
        # try each model in priority order and return first non-zero signal
        if self.fg:
            sig = self.fg.generate_signal(series)
            if any(s != 0 for s in sig):
                return sig
        if self.lgbm:
            # fit/update model on historical series
            self.lgbm.fit(series)
            sig = self.lgbm.generate_signal(series)
            if any(s != 0 for s in sig):
                return sig
        if self.nbeats:
            sig = self.nbeats.generate_signal(series)
            if any(s != 0 for s in sig):
                return sig
        if self.tft:
            sig = self.tft.generate_signal(series)
            if any(s != 0 for s in sig):
                return sig
        # fallback zero signal
        return [0.0] * len(series)
