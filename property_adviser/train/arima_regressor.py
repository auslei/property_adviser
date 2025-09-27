# --- AutoARIMA adapter (works inside sklearn Pipeline) ---
from pmdarima.arima import AutoARIMA
import numpy as np

class AutoARIMARegressor:
    """A thin wrapper so AutoARIMA works as the final 'model' step in a sklearn Pipeline.
    fit(X_transformed, y): trains AutoARIMA(y, exogenous=X)
    predict(X_transformed): forecasts len(X) steps using exogenous=X
    """
    def __init__(self, seasonal=False, m=1, suppress_warnings=True, maxiter=50, **kwargs):
        self.seasonal = seasonal
        self.m = m
        self.suppress_warnings = suppress_warnings
        self.maxiter = maxiter
        self.kwargs = kwargs
        self._model = None

    # make it sklearn-compatible
    def get_params(self, deep=True):
        return {
            "seasonal": self.seasonal,
            "m": self.m,
            "suppress_warnings": self.suppress_warnings,
            "maxiter": self.maxiter,
            **self.kwargs,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if k in ("seasonal", "m", "suppress_warnings", "maxiter"):
                setattr(self, k, v)
            else:
                self.kwargs[k] = v
        return self

    def fit(self, X, y):
        # X is the preprocessed design matrix (numpy array) from ColumnTransformer
        self._model = AutoARIMA(
            seasonal=self.seasonal,
            m=self.m,
            suppress_warnings=self.suppress_warnings,
            maxiter=self.maxiter,
            **self.kwargs
        )
        self._model.fit(y, X)  # y is the series, X are exogenous features
        return self

    def predict(self, X):
        # Forecast exactly as many steps as rows in X, conditioning on exogenous X
        n = X.shape[0]
        # pmdarima's predict takes n_periods and exogenous
        yhat = self._model.predict(n_periods=n, X=X)
        return np.asarray(yhat).reshape(-1)
