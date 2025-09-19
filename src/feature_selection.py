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

from .config import EXCLUDE_COLUMNS, PREPROCESS_DIR, RANDOM_STATE, TRAINING_DIR
from .suburb_median import (
    GLOBAL_SUBURB_KEY,
    HISTORY_FILENAME,
    load_suburb_median_history,
)

TIME_FEATURES = {"saleDate"}


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


def _attach_baseline_median(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    history = load_suburb_median_history()
    key_cols = ["suburb", "saleYear", "saleMonth"]
    history = history.copy()

    for col in key_cols:
        if col == "suburb":
            if col in history.columns:
                history[col] = history[col].fillna("Unknown").astype(str)
            if col in df.columns:
                df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        else:
            if col in history.columns:
                history[col] = pd.to_numeric(history[col], errors="coerce").astype("Int64")
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].round().astype("Int64")

    df = df.dropna(subset=[col for col in key_cols if col != "suburb"])

    suburb_history = (
        history[history["suburb"] != GLOBAL_SUBURB_KEY][
            key_cols + ["medianPrice", "transactionCount"]
        ]
        .drop_duplicates(key_cols, keep="last")
    )
    suburb_history = suburb_history.rename(
        columns={
            "medianPrice": "baselineMedian",
            "transactionCount": "baselineTransactions",
        }
    )

    merged = df.merge(suburb_history, on=key_cols, how="left")

    missing_mask = merged["baselineMedian"].isna()
    if missing_mask.any():
        global_history = (
            history[history["suburb"] == GLOBAL_SUBURB_KEY][
                ["saleYear", "saleMonth", "medianPrice", "transactionCount"]
            ]
            .drop_duplicates(["saleYear", "saleMonth"], keep="last")
            .rename(
                columns={
                    "medianPrice": "baselineMedian_global",
                    "transactionCount": "baselineTransactions_global",
                }
            )
        )
        merged = merged.merge(
            global_history,
            on=["saleYear", "saleMonth"],
            how="left",
        )
        merged.loc[missing_mask, "baselineMedian"] = merged.loc[
            missing_mask, "baselineMedian_global"
        ]
        merged.loc[missing_mask, "baselineTransactions"] = merged.loc[
            missing_mask, "baselineTransactions_global"
        ]
        merged = merged.drop(columns=["baselineMedian_global", "baselineTransactions_global"])

    merged["baselineTransactions"] = merged["baselineTransactions"].fillna(0)

    return merged, "baselineMedian", "baselineTransactions"


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

    target_col = "salePrice"

    df = _remove_identifiers(df)
    drop_time_cols = [col for col in TIME_FEATURES if col in df.columns]
    if drop_time_cols:
        df = df.drop(columns=drop_time_cols)

    configured_exclusions = [
        col for col in EXCLUDE_COLUMNS if col in df.columns and col != target_col
    ]
    if configured_exclusions:
        df = df.drop(columns=configured_exclusions)

    df = df[df[target_col].notna()]

    df, baseline_col, baseline_tx_col = _attach_baseline_median(df)
    df = df[df[baseline_col].notna()]
    df = df[df[baseline_col] > 0]

    df["priceFactor"] = df[target_col] / df[baseline_col]
    df = df.replace({"priceFactor": {np.inf: np.nan, -np.inf: np.nan}})
    df = df[df["priceFactor"].notna()]
    df = df[df["priceFactor"] > 0]

    df = df.drop(columns=[target_col, baseline_col])
    target_col = "priceFactor"

    for numeric_key in ["saleYear", "saleMonth", baseline_tx_col]:
        if numeric_key and numeric_key in df.columns:
            df[numeric_key] = pd.to_numeric(df[numeric_key], errors="coerce")

    preserve_columns = [target_col, "saleYear", "saleMonth", "suburb"]
    if baseline_tx_col in df.columns:
        preserve_columns.append(baseline_tx_col)

    df = _drop_low_variance(df, exclude=preserve_columns)

    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_features = [col for col in numeric_features if col != target_col]

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
        threshold_index = min(len(sorted_features) - 1, 14)
        threshold = max(0.01, sorted_features[threshold_index][1])
        selected_features = [feat for feat, score in sorted_features if score >= threshold]
        minimum = min(5, len(sorted_features))
        if len(selected_features) < minimum:
            selected_features = [feat for feat, _ in sorted_features[:minimum]]
    else:
        selected_features = numeric_features + categorical_features

    priority_candidates = [
        "street",
        "suburb",
        "propertyType",
        "bed",
        "bath",
        "car",
        "comparableCount",
        "saleYear",
        "saleMonth",
    ]
    priority_features = [col for col in priority_candidates if col in df.columns]
    for feature in priority_features:
        if feature not in selected_features:
            selected_features.append(feature)

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
        "raw_target": "salePrice",
        "target_type": "price_factor",
        "baseline_lookup_keys": ["suburb", "saleYear", "saleMonth"],
        "baseline_transactions_column": baseline_tx_col
        if baseline_tx_col in df.columns
        else None,
        "selected_features": selected_features,
        "numeric_features": numeric_selected,
        "categorical_features": categorical_selected,
        "dropped_correlated_features": correlated_numeric,
        "config_excluded_columns": configured_exclusions,
        "rows": int(X.shape[0]),
        "categorical_levels": categorical_levels,
        "numeric_summary": numeric_summary,
        "baseline_history_file": HISTORY_FILENAME,
    }
    metadata_path = TRAINING_DIR / "feature_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return metadata


if __name__ == "__main__":
    info = run_feature_selection()
    print("Feature selection completed. Selected features:")
    for feature in info["selected_features"]:
        print(f" - {feature}")
