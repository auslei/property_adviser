import json
from pathlib import Path
from typing import Dict, List

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

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Persist split datasets for traceability
    X_train_path = TRAINING_DIR / "X_train.parquet"
    X_val_path = TRAINING_DIR / "X_val.parquet"
    y_train_path = TRAINING_DIR / "y_train.parquet"
    y_val_path = TRAINING_DIR / "y_val.parquet"

    X_train.to_parquet(X_train_path, index=False)
    X_val.to_parquet(X_val_path, index=False)
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

        mae = mean_absolute_error(y_val, predictions)
        rmse = mean_squared_error(y_val, predictions) ** 0.5
        r2 = r2_score(y_val, predictions)

        results.append(
            {
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "best_params": tuned_params,
                "cv_best_score": cv_best_score,
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
