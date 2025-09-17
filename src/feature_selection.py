import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from .config import PREPROCESS_DIR, RANDOM_STATE, TRAINING_DIR


def _load_clean_data() -> pd.DataFrame:
    data_path = PREPROCESS_DIR / "cleaned.parquet"
    if not data_path.exists():
        raise FileNotFoundError(
            "Preprocessed data not found. Run src/preprocess.py first."
        )
    return pd.read_parquet(data_path)


def _drop_low_variance(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    columns_to_drop = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].nunique(dropna=False) <= 1:
            columns_to_drop.append(col)
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    return df


def _remove_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    drop_candidates = [
        "openInRpdata",
        "parcelDetails",
        "streetAddress",
    ]
    existing = [col for col in drop_candidates if col in df.columns]
    if existing:
        df = df.drop(columns=existing)
    return df


def _correlated_features(df: pd.DataFrame, numeric_columns: List[str], threshold: float = 0.9) -> List[str]:
    if not numeric_columns:
        return []
    corr_matrix = df[numeric_columns].corr().abs()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    upper = corr_matrix.where(mask)
    correlated = [
        column
        for column in upper.columns
        if any(upper[column].fillna(0) > threshold)
    ]
    return correlated


def run_feature_selection() -> Dict[str, object]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_clean_data()
    if "salePrice" not in df.columns:
        raise ValueError("Expected 'salePrice' column to be present in preprocessed data.")

    df = _remove_identifiers(df)
    target_col = "salePrice"
    df = df[df[target_col].notna()]  # guard, imputed already but keep safe

    df = _drop_low_variance(df, exclude=[target_col])

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != target_col]

    # treat saleDate as numeric if possible
    if "saleDate" in df.columns:
        if not np.issubdtype(df["saleDate"].dtype, np.number):
            df["saleDate"] = pd.to_numeric(df["saleDate"], errors="coerce")
        if "saleDate" not in numeric_features:
            numeric_features.append("saleDate")

    categorical_features = [
        col
        for col in df.columns
        if col not in numeric_features + [target_col]
    ]

    correlated_numeric = _correlated_features(df, numeric_features)
    numeric_features = [col for col in numeric_features if col not in correlated_numeric]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: List[Tuple[str, Pipeline, List[str]]] = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No features available for modelling after preprocessing.")

    preprocessor = ColumnTransformer(transformers)

    model = DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=6)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(df[numeric_features + categorical_features], df[target_col])

    preprocessor_step: ColumnTransformer = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor_step.get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    aggregated: Dict[str, float] = {}
    for name, importance in zip(feature_names, importances):
        if name.startswith("num__"):
            original = name.split("__", 1)[1]
        elif name.startswith("cat__"):
            remainder = name.split("__", 1)[1]
            original = remainder.split("_", 1)[0]
        else:
            original = name
        aggregated[original] = aggregated.get(original, 0.0) + float(importance)

    sorted_features = sorted(aggregated.items(), key=lambda item: item[1], reverse=True)
    if sorted_features:
        threshold = max(0.01, sorted_features[min(len(sorted_features) - 1, 14)][1])
        selected_features = [feat for feat, score in sorted_features if score >= threshold]
        if len(selected_features) < min(5, len(sorted_features)):
            selected_features = [feat for feat, _ in sorted_features[: min(5, len(sorted_features))]]
    else:
        selected_features = numeric_features + categorical_features

    selected_features = sorted(set(selected_features))

    numeric_selected = [feat for feat in selected_features if feat in numeric_features]
    categorical_selected = [feat for feat in selected_features if feat in categorical_features]

    X = df[selected_features].copy()
    y = df[target_col].copy()

    # Store
    X_path = TRAINING_DIR / "X.parquet"
    y_path = TRAINING_DIR / "y.parquet"
    X.to_parquet(X_path, index=False)
    y.to_frame(name=target_col).to_parquet(y_path, index=False)

    feature_importance_path = TRAINING_DIR / "feature_importances.json"
    feature_importance_path.write_text(
        json.dumps(
            [
                {"feature": feat, "importance": score}
                for feat, score in sorted_features
            ],
            indent=2,
        )
    )

    categorical_levels = {
        col: sorted(value for value in X[col].unique())
        for col in categorical_selected
    }
    numeric_summary = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "median": float(X[col].median()),
        }
        for col in numeric_selected
    }

    metadata = {
        "target": target_col,
        "selected_features": selected_features,
        "numeric_features": numeric_selected,
        "categorical_features": categorical_selected,
        "dropped_correlated_features": correlated_numeric,
        "rows": int(X.shape[0]),
        "categorical_levels": categorical_levels,
        "numeric_summary": numeric_summary,
    }
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


if __name__ == "__main__":
    info = run_feature_selection()
    print("Feature selection completed. Selected features:")
    for feature in info["selected_features"]:
        print(f" - {feature}")
