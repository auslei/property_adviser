import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODELS_DIR, RANDOM_STATE, TRAINING_DIR
from .suburb_median import GLOBAL_SUBURB_KEY, load_suburb_median_history


def _prepare_baseline_maps() -> Tuple[Dict[Tuple[object, int, int], float], Dict[Tuple[int, int], float]]:
    history = load_suburb_median_history()
    suburb_map = (
        history[history["suburb"] != GLOBAL_SUBURB_KEY][
            ["suburb", "saleYear", "saleMonth", "medianPrice"]
        ]
        .drop_duplicates(["suburb", "saleYear", "saleMonth"], keep="last")
        .set_index(["suburb", "saleYear", "saleMonth"])["medianPrice"]
        .to_dict()
    )
    global_map = (
        history[history["suburb"] == GLOBAL_SUBURB_KEY][
            ["saleYear", "saleMonth", "medianPrice"]
        ]
        .drop_duplicates(["saleYear", "saleMonth"], keep="last")
        .set_index(["saleYear", "saleMonth"])["medianPrice"]
        .to_dict()
    )
    return suburb_map, global_map


def _baseline_from_maps(
    frame: pd.DataFrame,
    suburb_map: Dict[Tuple[object, int, int], float],
    global_map: Dict[Tuple[int, int], float],
) -> pd.Series:
    required = ["suburb", "saleYear", "saleMonth"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing baseline lookup columns in feature set: {missing}. Ensure feature selection retained these fields."
        )

    values = []
    for suburb, year, month in zip(
        frame["suburb"], frame["saleYear"], frame["saleMonth"]
    ):
        year_int = int(year) if pd.notna(year) else None
        month_int = int(month) if pd.notna(month) else None
        if suburb is None or year_int is None or month_int is None:
            values.append(np.nan)
            continue
        primary_key = (suburb, year_int, month_int)
        baseline = suburb_map.get(primary_key)
        if baseline is None or (isinstance(baseline, float) and np.isnan(baseline)):
            baseline = global_map.get((year_int, month_int))
        values.append(baseline)
    return pd.Series(values, index=frame.index, dtype=float)


def _load_training_data() -> tuple[pd.DataFrame, pd.Series, Dict[str, object]]:
    X_path = TRAINING_DIR / "X.parquet"
    y_path = TRAINING_DIR / "y.parquet"
    metadata_path = TRAINING_DIR / "feature_metadata.json"

    if not X_path.exists() or not y_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Training data or metadata not found. Run feature selection first."
        )

    X = pd.read_parquet(X_path)
    y_df = pd.read_parquet(y_path)
    metadata = json.loads(metadata_path.read_text())

    target_col = metadata["target"]
    y = y_df[target_col]

    # Ensure column order matches metadata
    selected_features = metadata["selected_features"]
    missing = [col for col in selected_features if col not in X.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    X = X[selected_features]

    return X, y, metadata


def build_preprocessor(metadata: Dict[str, object]) -> ColumnTransformer:
    numeric_features: List[str] = metadata["numeric_features"]
    categorical_features: List[str] = metadata["categorical_features"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No features available to build preprocessor.")

    return ColumnTransformer(transformers)


def train_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, metadata = _load_training_data()

    suburb_map, global_map = _prepare_baseline_maps()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    baseline_train = _baseline_from_maps(X_train, suburb_map, global_map)
    baseline_val = _baseline_from_maps(X_val, suburb_map, global_map)

    # Persist split datasets for traceability
    X_train_path = TRAINING_DIR / "X_train.parquet"
    X_val_path = TRAINING_DIR / "X_val.parquet"
    y_train_path = TRAINING_DIR / "y_train.parquet"
    y_val_path = TRAINING_DIR / "y_val.parquet"

    X_train.assign(baselineMedian=baseline_train).to_parquet(X_train_path, index=False)
    X_val.assign(baselineMedian=baseline_val).to_parquet(X_val_path, index=False)
    y_train.to_frame(name=metadata["target"]).to_parquet(y_train_path, index=False)
    y_val.to_frame(name=metadata["target"]).to_parquet(y_val_path, index=False)

    candidate_models = {
        "LinearRegression": (
            LinearRegression(),
            {},
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }

    results = []
    best_model_name = None
    best_pipeline = None
    best_score = -np.inf
    best_model_params: Dict[str, object] = {}

    for name, (model, param_grid) in candidate_models.items():
        preprocessor = build_preprocessor(metadata)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        if param_grid:
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring="r2",
                cv=3,
            )
            search.fit(X_train, y_train)
            tuned_pipeline = search.best_estimator_
            tuned_params = search.best_params_
            cv_best_score = float(search.best_score_)
        else:
            tuned_pipeline = pipeline.fit(X_train, y_train)
            tuned_params = {}
            cv_best_score = None

        predictions = tuned_pipeline.predict(X_val)
        prediction_series = pd.Series(predictions, index=X_val.index, name="predictedFactor")

        valid_mask = baseline_val.notna()
        if not valid_mask.any():
            raise ValueError("Baseline lookup returned no valid medians for validation set.")

        factor_mae = mean_absolute_error(y_val[valid_mask], prediction_series[valid_mask])
        factor_rmse = mean_squared_error(y_val[valid_mask], prediction_series[valid_mask]) ** 0.5
        factor_r2 = r2_score(y_val[valid_mask], prediction_series[valid_mask])

        baseline_values = baseline_val[valid_mask]
        actual_prices = y_val[valid_mask] * baseline_values
        predicted_prices = prediction_series[valid_mask] * baseline_values

        mae = mean_absolute_error(actual_prices, predicted_prices)
        rmse = mean_squared_error(actual_prices, predicted_prices) ** 0.5
        r2 = r2_score(actual_prices, predicted_prices)

        results.append(
            {
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "best_params": tuned_params,
                "cv_best_score": cv_best_score,
                "factor_mae": factor_mae,
                "factor_rmse": factor_rmse,
                "factor_r2": factor_r2,
            }
        )

        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_pipeline = tuned_pipeline
            best_model_params = tuned_params

    if best_pipeline is None:
        raise RuntimeError("Model training failed to produce a valid pipeline.")

    model_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_pipeline, model_path)

    metrics_path = MODELS_DIR / "model_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    selection_path = MODELS_DIR / "best_model.json"
    selection_path.write_text(
        json.dumps(
            {
                "best_model": best_model_name,
                "r2": best_score,
                "model_path": str(model_path),
                "best_params": best_model_params,
                "target_type": metadata.get("target_type", "price_factor"),
            },
            indent=2,
        )
    )

    return {
        "best_model": best_model_name,
        "metrics": results,
        "model_path": model_path,
        "best_params": best_model_params,
    }


if __name__ == "__main__":
    outcome = train_models()
    print(f"Best model: {outcome['best_model']} (saved to {outcome['model_path']})")
