import numpy as np
import pytest
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from property_adviser.train.sarimax_regressor import SarimaxRegressor


@pytest.mark.filterwarnings("ignore:Maximum Likelihood optimization failed to converge:statsmodels.tools.sm_exceptions.ConvergenceWarning")
def test_sarimax_regressor_forecast_shapes():
    rng = np.random.default_rng(42)
    n_obs = 60
    trend = np.linspace(100.0, 140.0, n_obs)
    seasonal = 5.0 * np.sin(np.linspace(0, 4 * np.pi, n_obs))
    noise = rng.normal(scale=2.0, size=n_obs)
    y = trend + seasonal + noise

    X = np.column_stack([
        np.arange(n_obs),
        np.sin(np.linspace(0, 2 * np.pi, n_obs)),
    ])

    split = 45
    model = SarimaxRegressor(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), trend="c", maxiter=50)
    fitted = model.fit(X[:split], y[:split])

    preds = fitted.predict(X[split:])

    assert preds.shape == (n_obs - split,)
    assert np.isfinite(preds).all()
