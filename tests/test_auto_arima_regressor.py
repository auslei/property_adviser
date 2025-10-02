import numpy as np

from property_adviser.train.arima_regressor import AutoARIMARegressor


def test_auto_arima_regressor_fit_predict_shapes():
    rng = np.random.default_rng(42)
    n_obs = 40
    trend = np.linspace(100, 140, n_obs)
    seasonal = 5 * np.sin(np.linspace(0, 4 * np.pi, n_obs))
    noise = rng.normal(scale=2.0, size=n_obs)
    y = trend + seasonal + noise

    # simple exogenous signal (e.g., lagged features)
    X = np.column_stack([
        np.arange(n_obs),
        np.sin(np.linspace(0, 2 * np.pi, n_obs)),
    ])

    model = AutoARIMARegressor(seasonal=False, maxiter=10)
    fitted = model.fit(X, y)

    preds = fitted.predict(X)

    assert preds.shape == (n_obs,)
    assert np.isfinite(preds).all()
