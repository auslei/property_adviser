import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from .config import MODEL_CONFIG_PATH, MODELS_DIR, RANDOM_STATE, TRAINING_DIR



MODEL_FACTORY = {
    "LinearRegression": lambda random_state: LinearRegression(),
    "RandomForestRegressor": lambda random_state: RandomForestRegressor(
        random_state=random_state
    ),
    "GradientBoostingRegressor": lambda random_state: GradientBoostingRegressor(
        random_state=random_state
    ),
    "Lasso": lambda random_state: Lasso(),
    "ElasticNet": lambda random_state: ElasticNet(),
}


def _load_model_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is not None:
        return config
    return load_yaml(MODEL_CONFIG_PATH)


def _prepare_model_candidates(
    config: Dict[str, Any],
    random_state: int,
) -> Dict[str, Tuple[object, Dict[str, List[Any]]]]:
    if not config:
        config = {
            "LinearRegression": {"enabled": True, "grid": {}},
            "RandomForestRegressor": {
                "enabled": True,
                "grid": {
                    "n_estimators": [200, 400],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "GradientBoostingRegressor": {
                "enabled": True,
                "grid": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [2, 3, 4],
                    "subsample": [0.8, 1.0],
                },
            },
        }

    candidates: Dict[str, Tuple[object, Dict[str, List[Any]]]] = {}
    for name, spec in config.items():
        if not isinstance(spec, dict):
            continue
        if not spec.get("enabled", True):
            continue
        factory = MODEL_FACTORY.get(name)
        if factory is None:
            raise ValueError(f"Model '{name}' is not supported. Update MODEL_FACTORY to add it.")
        estimator = factory(random_state)
        grid_spec = spec.get("grid") or {}
        grid: Dict[str, List[Any]] = {}
        for param, values in grid_spec.items():
            if isinstance(values, (list, tuple)):
                grid_values = list(values)
            else:
                grid_values = [values]
            grid[f"model__{param}"] = grid_values
        candidates[name] = (estimator, grid)

    if not candidates:
        raise ValueError("No models enabled in configuration. Enable at least one model to train.")

    return candidates


def _apply_feature_adjustments(
    X: pd.DataFrame,
    metadata: Dict[str, Any],
    adjustments: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, List[str]]]:
    adjustments = adjustments or {}
    include = adjustments.get("include") or []
    exclude = adjustments.get("exclude") or []

    selected_features = [
        feature for feature in metadata.get("selected_features", []) if feature in X.columns
    ]

    for feature in include:
        if feature in X.columns and feature not in selected_features:
            selected_features.append(feature)

    removed = [feature for feature in exclude if feature in selected_features]
    if removed:
        selected_features = [feature for feature in selected_features if feature not in removed]
        X = X.drop(columns=removed)

    if not selected_features:
        raise ValueError("No features remaining after manual adjustments. Update configuration.")

    numeric_features = [
        feature
        for feature in metadata.get("numeric_features", [])
        if feature in selected_features
    ]
    categorical_features = [
        feature
        for feature in metadata.get("categorical_features", [])
        if feature in selected_features
    ]

    updated_metadata = dict(metadata)
    updated_metadata["selected_features"] = selected_features
    updated_metadata["numeric_features"] = numeric_features
    updated_metadata["categorical_features"] = categorical_features

    info = {
        "include": [feat for feat in include if feat in X.columns],
        "missing_include": [feat for feat in include if feat not in X.columns],
        "exclude": removed,
    }

    return X[selected_features], updated_metadata, info


# def _prepare_baseline_maps() -> Tuple[Dict[Tuple[object, int, int], float], Dict[Tuple[int, int], float]]:
#     history = load_suburb_median_history()
#     suburb_map = (
#         history[history["suburb"] != GLOBAL_SUBURB_KEY][
#             ["suburb", "saleYear", "saleMonth", "medianPrice"]
#         ]
#         .drop_duplicates(["suburb", "saleYear", "saleMonth"], keep="last")
#         .set_index(["suburb", "saleYear", "saleMonth"])["medianPrice"]
#         .to_dict()
#     )
#     global_map = (
#         history[history["suburb"] == GLOBAL_SUBURB_KEY][
#             ["saleYear", "saleMonth", "medianPrice"]
#         ]
#         .drop_duplicates(["saleYear", "saleMonth"], keep="last")
#         .set_index(["saleYear", "saleMonth"])["medianPrice"]
#         .to_dict()
#     )
#     return suburb_map, global_map





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


def train_timeseries_model(config: Optional[Dict[str, Any]] = None):
    """
    Trains a model to predict property prices based on time series data.

    This function performs the following steps:
    1.  Loads the training data and metadata.
    2.  Filters the features to those relevant for time series prediction.
    3.  Applies manual feature adjustments from the configuration.
    4.  Splits the data into training and validation sets.
    5.  Trains multiple candidate models with hyperparameter tuning.
    6.  Evaluates the models and selects the best one based on R2 score.
    7.  Saves the best model and its metrics.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    resolved_config = _load_model_config(config)
    split_cfg = resolved_config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    random_state = int(split_cfg.get("random_state", RANDOM_STATE))

    X, y, metadata = _load_training_data()
    
    # Focus on the required features for timeseries prediction
    required_features = ["yearmonth", "bed", "bath", "car", "propertyType", "street"]
    
    # If yearmonth doesn't exist, create it from saleYear and saleMonth
    if "saleYear" in X.columns and "saleMonth" in X.columns:
        X = X.copy()
        X["yearmonth"] = X["saleYear"] * 100 + X["saleMonth"]
        if "yearmonth" not in metadata.get("selected_features", []):
            selected_features = metadata.get("selected_features", [])
            selected_features.append("yearmonth")
            metadata["selected_features"] = selected_features
        if "yearmonth" not in metadata.get("numeric_features", []):
            numeric_features = metadata.get("numeric_features", [])
            numeric_features.append("yearmonth")
            metadata["numeric_features"] = numeric_features
    
    # Filter to only include relevant features
    # Get all features that are relevant for timeseries prediction
    relevant_feature_names = [col for col in X.columns if col in ["yearmonth", "bed", "bath", "car", "propertyType", "street"] or col in metadata.get("selected_features", [])]
    X_filtered = X[relevant_feature_names]
    
    # Apply feature adjustments as specified in the configuration
    X_filtered, training_metadata, adjustment_info = _apply_feature_adjustments(
        X_filtered,
        metadata,
        resolved_config.get("manual_feature_adjustments"),
    )

    models_config = resolved_config.get("models", {})
    candidate_models = _prepare_model_candidates(models_config, random_state)

    # Target is now salePrice directly, not a factor
    target_type = "regression"  # We're predicting actual prices, not factors

    X_train, X_val, y_train, y_val = train_test_split(
        X_filtered,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Persist split datasets for traceability
    X_train_path = TRAINING_DIR / "X_train.parquet"
    X_val_path = TRAINING_DIR / "X_val.parquet"
    y_train_path = TRAINING_DIR / "y_train.parquet"
    y_val_path = TRAINING_DIR / "y_val.parquet"

    train_dump = X_train.copy()
    val_dump = X_val.copy()
    train_dump.to_parquet(X_train_path, index=False)
    val_dump.to_parquet(X_val_path, index=False)
    y_train.to_frame(name=metadata["target"]).to_parquet(y_train_path, index=False)
    y_val.to_frame(name=metadata["target"]).to_parquet(y_val_path, index=False)

    # Train and evaluate each candidate model
    results = []
    best_model_name = None
    best_pipeline = None
    best_score = -np.inf
    best_model_params: Dict[str, object] = {}

    for name, (model, param_grid) in candidate_models.items():
        preprocessor = build_preprocessor(training_metadata)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # If a parameter grid is defined, perform a grid search to find the best hyperparameters
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
            # Otherwise, train the model with its default hyperparameters
            tuned_pipeline = pipeline.fit(X_train, y_train)
            tuned_params = {}
            cv_best_score = None

        # Evaluate the model on the validation set
        predictions = tuned_pipeline.predict(X_val)
        prediction_series = pd.Series(predictions, index=X_val.index, name="prediction")

        # Calculate metrics directly on price prediction
        actual_prices = y_val
        predicted_prices = prediction_series
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
                "factor_mae": None,  # Not applicable for direct price prediction
                "factor_rmse": None,
                "factor_r2": None,
            }
        )

        # Keep track of the best model based on the R2 score
        if r2 > best_score:
            best_score = r2
            best_model_name = name
            best_pipeline = tuned_pipeline
            best_model_params = tuned_params

    # Save the best model, its metrics, and metadata
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
                "target_type": target_type,
                "manual_adjustments": adjustment_info,
                "split": {"test_size": test_size, "random_state": random_state},
                "model_config": models_config,
            },
            indent=2,
        )
    )

    return {
        "best_model": best_model_name,
        "metrics": results,
        "model_path": model_path,
        "best_params": best_model_params,
        "target_type": target_type,
    }