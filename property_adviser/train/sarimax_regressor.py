"""statsmodels SARIMAX adapter for scikit-learn style pipelines."""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def _as_tuple(values: Sequence[int], expected_len: int, *, param_name: str) -> Tuple[int, ...]:
    if isinstance(values, tuple):
        result = values
    else:
        result = tuple(int(x) for x in values)
    if len(result) != expected_len:
        raise ValueError(f"{param_name} must have length {expected_len}, got {len(result)}")
    return result


def _prepare_exog(array: Optional[Any]) -> Optional[np.ndarray]:
    if array is None:
        return None
    exog = np.asarray(array)
    if exog.ndim == 1:
        exog = exog.reshape(-1, 1)
    if exog.ndim != 2:
        raise ValueError("Exogenous regressors must be 2-dimensional.")
    if exog.shape[1] == 0:
        return None
    return exog


class SarimaxRegressor:
    """Thin SARIMAX wrapper that mimics a scikit-learn regressor interface."""

    def __init__(
        self,
        *,
        order: Sequence[int] = (1, 0, 0),
        seasonal_order: Sequence[int] = (0, 0, 0, 0),
        trend: Optional[str] = None,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        maxiter: int = 100,
        disp: bool = False,
        fit_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.order = _as_tuple(order, 3, param_name="order")
        self.seasonal_order = _as_tuple(seasonal_order, 4, param_name="seasonal_order")
        self.trend = trend
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.maxiter = maxiter
        self.disp = disp
        self.fit_kwargs = fit_kwargs
        self._results = None
        self._n_exog = None

    # scikit-learn compatibility -------------------------------------------------
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "trend": self.trend,
            "enforce_stationarity": self.enforce_stationarity,
            "enforce_invertibility": self.enforce_invertibility,
            "maxiter": self.maxiter,
            "disp": self.disp,
        }
        if deep:
            if isinstance(self.fit_kwargs, dict):
                params["fit_kwargs"] = self.fit_kwargs.copy()
            elif self.fit_kwargs is None:
                params["fit_kwargs"] = None
            else:
                params["fit_kwargs"] = dict(self.fit_kwargs)
        else:
            params["fit_kwargs"] = self.fit_kwargs
        return params

    def set_params(self, **params: Any) -> "SarimaxRegressor":
        for key, value in params.items():
            if key == "order":
                self.order = _as_tuple(value, 3, param_name="order")
            elif key == "seasonal_order":
                self.seasonal_order = _as_tuple(value, 4, param_name="seasonal_order")
            elif key == "trend":
                self.trend = value
            elif key == "enforce_stationarity":
                self.enforce_stationarity = bool(value)
            elif key == "enforce_invertibility":
                self.enforce_invertibility = bool(value)
            elif key == "maxiter":
                self.maxiter = int(value)
            elif key == "disp":
                self.disp = bool(value)
            elif key == "fit_kwargs":
                self.fit_kwargs = value
            else:
                raise ValueError(f"Unknown parameter '{key}' for SarimaxRegressor")
        return self

    # modelling ------------------------------------------------------------------
    def fit(self, X: Optional[Any], y: Sequence[float]) -> "SarimaxRegressor":
        if y is None:
            raise ValueError("y must be provided for SARIMAX fitting")
        endog = np.asarray(y, dtype=float)
        if endog.ndim != 1:
            endog = endog.reshape(-1)
        exog = _prepare_exog(X)
        self._n_exog = 0 if exog is None else exog.shape[1]
        model = SARIMAX(
            endog,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
        )
        if self.fit_kwargs:
            fit_options = self.fit_kwargs if isinstance(self.fit_kwargs, dict) else dict(self.fit_kwargs)
        else:
            fit_options = {}
        self._results = model.fit(maxiter=self.maxiter, disp=self.disp, **fit_options)
        return self

    def predict(self, X: Optional[Any]) -> np.ndarray:
        if self._results is None:
            raise RuntimeError("SarimaxRegressor is not fitted yet")
        n_steps = 0
        if X is not None:
            n_steps = int(np.asarray(X).shape[0])
        exog = _prepare_exog(X)
        if exog is not None and self._n_exog is not None and exog.shape[1] != self._n_exog:
            raise ValueError("Exogenous feature count differs from training data")
        if n_steps == 0:
            return np.empty(0, dtype=float)
        forecast = self._results.get_forecast(steps=n_steps, exog=exog)
        return np.asarray(forecast.predicted_mean, dtype=float)


__all__ = ["SarimaxRegressor"]
