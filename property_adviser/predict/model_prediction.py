from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd

from functools import lru_cache

from property_adviser.core.artifacts import load_model_artifacts
from property_adviser.core.paths import MODELS_DIR
from property_adviser.config import PREPROCESS_CONFIG_PATH
from property_adviser.core.config import load_config
from property_adviser.predict.feature_store import (
    fetch_reference_features,
    list_streets,
    list_suburbs,
)
from property_adviser.preprocess.preprocess_derive import extract_street as _extract_street


def load_trained_model() -> tuple:
    """
    Load the trained model, its metadata and feature information
    """
    artifacts = load_model_artifacts()
    return artifacts.model, artifacts.metadata




def _target_is_delta(metadata: Dict[str, Any]) -> bool:
    target = metadata.get("target")
    return isinstance(target, str) and target.endswith("_delta")


def _base_candidates(metadata: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    target_info = metadata.get("target_output_info")
    if isinstance(target_info, dict):
        base_col = target_info.get("base_column")
        if isinstance(base_col, str):
            candidates.append(base_col)
    candidates.extend([
        "current_price_median",
        "suburb_price_median_current",
        "suburb_type_price_median_current",
        "price_current_median",
    ])
    return candidates


def _convert_prediction(raw_value: float, input_row: pd.Series, metadata: Dict[str, Any]) -> tuple[float, float, Optional[float]]:
    actual = float(raw_value)
    base_value: Optional[float] = None
    if _target_is_delta(metadata):
        for column in _base_candidates(metadata):
            if column and column in input_row.index:
                candidate = input_row[column]
                if candidate is None or (isinstance(candidate, float) and np.isnan(candidate)):
                    continue
                base_value = float(candidate)
                actual = base_value * (1.0 + float(raw_value))
                break
    return actual, float(raw_value), base_value


def _predict_single(
    *,
    yearmonth: int,
    bed: int,
    bath: int,
    car: int,
    property_type: str,
    street: str,
    suburb: str,
    land_size: Optional[float],
    floor_size: Optional[float],
    year_built: Optional[int],
    model: Optional[Any],
    metadata: Optional[Dict[str, Any]],
) -> tuple[float, float, Optional[float], Any, Dict[str, Any]]:
    if model is None or metadata is None:
        model, metadata = load_trained_model()

    properties = [
        {
            "yearmonth": yearmonth,
            "bed": bed,
            "bath": bath,
            "car": car,
            "propertyType": property_type,
            "street": street,
            "suburb": suburb,
            "landSize": land_size,
            "floorSize": floor_size,
            "yearBuild": year_built,
        }
    ]
    input_df = _prepare_prediction_data(properties, metadata)
    raw_value = float(model.predict(input_df)[0])
    actual, raw_value, base_value = _convert_prediction(raw_value, input_df.iloc[0], metadata)
    return actual, raw_value, base_value, model, metadata

@lru_cache(maxsize=1)
def _load_derive_buckets_and_mappings() -> dict:
    """Load bucket edges/labels and simple mappings from derive config, if available."""
    try:
        pre_cfg = load_config(PREPROCESS_CONFIG_PATH)
        derive_path = pre_cfg.get("derivation", {}).get("config_path")
        if not derive_path:
            return {}
        derive_cfg = load_config(derive_path)
        steps = derive_cfg.get("steps", []) or []
        info: dict = {"buckets": {}, "mappings": {}}
        for step in steps:
            if not isinstance(step, dict):
                continue
            stype = str(step.get("type") or "")
            if stype == "bin":
                output = step.get("output")
                config = step.get("config") or {}
                edges = config.get("edges") or config.get("bins")
                labels = config.get("labels")
                if output and edges:
                    info["buckets"][str(output)] = {"edges": list(edges), "labels": list(labels) if labels else None}
            elif stype == "simple" and str(step.get("method") or "").lower() == "map_values":
                output = step.get("output")
                config = step.get("config") or {}
                mapping = config.get("mapping") or {}
                default = config.get("default")
                if output and mapping:
                    info["mappings"][str(output)] = {"mapping": {str(k).upper(): v for k, v in mapping.items()}, "default": default}
        return info
    except Exception:
        return {}


def _prepare_prediction_data(
    properties: List[Dict[str, Any]], metadata: Dict[str, Any]
) -> pd.DataFrame:
    """Prepare model-ready features from minimal property descriptors."""
    if not properties:
        raise ValueError("At least one property must be provided for prediction")

    feature_metadata = metadata.get("feature_metadata") or {}
    model_columns: List[str] = feature_metadata.get("model_input_columns") or metadata.get("selected_features", [])
    if not model_columns:
        raise ValueError("Feature metadata is missing required model column information")

    numeric_features: List[str] = feature_metadata.get("numeric_features", metadata.get("numeric_features", []))
    categorical_features: List[str] = feature_metadata.get("categorical_features", metadata.get("categorical_features", []))

    impute_numeric: Dict[str, Any] = (feature_metadata.get("impute", {}).get("numeric")
                                      or metadata.get("impute", {}).get("numeric")
                                      or {})
    impute_categorical: Dict[str, Any] = (feature_metadata.get("impute", {}).get("categorical")
                                          or metadata.get("impute", {}).get("categorical")
                                          or {})

    df = pd.DataFrame(properties)

    # Normalise expected column names
    rename_map = {
        "yearmonth": "saleYearMonth",
        "landSize": "landSizeM2",
        "floorSize": "floorSizeM2",
        "yearBuild": "yearBuilt",
        "yearbuild": "yearBuilt",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "suburb" not in df.columns:
        raise ValueError("'suburb' is required for prediction inputs")

    # Ensure saleYearMonth is present and numeric
    if "saleYearMonth" not in df.columns:
        raise ValueError("'saleYearMonth' is required for prediction inputs")
    df["saleYearMonth"] = pd.to_numeric(df["saleYearMonth"], errors="coerce")

    # Fill known numeric identifiers before derivations
    if "yearBuilt" not in df.columns:
        df["yearBuilt"] = np.nan
    if "yearBuilt" in impute_numeric:
        df["yearBuilt"] = df["yearBuilt"].fillna(impute_numeric["yearBuilt"])

    # Derive temporal components
    sale_year = (df["saleYearMonth"] // 100).astype("Int64")
    sale_month = (df["saleYearMonth"] % 100).astype("Int64")
    if "saleYear" in model_columns:
        df["saleYear"] = sale_year
    if "saleMonth" in model_columns:
        df["saleMonth"] = sale_month

    # Property age features
    if {"propertyAge", "propertyAgeBand"}.intersection(model_columns):
        property_age = sale_year.astype("Float64") - pd.to_numeric(df.get("yearBuilt"), errors="coerce")
        property_age = property_age.where(property_age >= 0)
        df["propertyAge"] = property_age

        if "propertyAgeBand" in model_columns:
            age_meta = feature_metadata.get("property_age", {})
            bands = age_meta.get("bands", [5, 20])
            labels = age_meta.get("labels", ["0-5", "6-20", "21+"])
            bins = [-np.inf] + list(bands) + [np.inf]
            df["propertyAgeBand"] = pd.cut(df["propertyAge"], bins=bins, labels=labels)

    # Align optional numeric columns
    if "landSizeM2" in df.columns:
        df["landSizeM2"] = pd.to_numeric(df["landSizeM2"], errors="coerce")
    if "floorSizeM2" in df.columns:
        df["floorSizeM2"] = pd.to_numeric(df["floorSizeM2"], errors="coerce")

    # Normalize propertyType if present in model columns
    def _canon_property_type(x: Any) -> Optional[str]:
        s = str(x or "").strip().lower()
        if not s:
            return None
        if any(tok in s for tok in ["house", "townhouse", "dwelling"]):
            return "House"
        if any(tok in s for tok in ["unit", "villa", "duplex"]):
            return "Unit"
        if any(tok in s for tok in ["apartment", "flat", "studio"]):
            return "Apartment"
        return None

    if "propertyType" in model_columns or "propertyType" in df.columns:
        df["propertyType"] = df.get("propertyType", "").apply(_canon_property_type)

    # Buckets computed from inputs if present in model columns
    def _apply_bucket(source_col: str, output_col: str) -> None:
        if output_col not in model_columns:
            return
        if source_col not in df.columns:
            return
        cfg = derive_info.get("buckets", {}).get(output_col)
        values = pd.to_numeric(df[source_col], errors="coerce")
        if cfg and cfg.get("edges"):
            edges = [float(x) for x in cfg["edges"]]
            labels = cfg.get("labels")
            bins = [-np.inf] + edges + [np.inf]
            cats = pd.cut(values, bins=bins, labels=labels if labels and len(labels) == len(bins) - 1 else None)
            df[output_col] = cats.astype(str).replace({"nan": np.nan})
        else:
            # Fallback simple buckets if derive config unavailable
            if output_col == "bed_bucket":
                edges = [2, 3, 4]
            elif output_col == "bath_bucket":
                edges = [1, 2, 3]
            elif output_col == "land_bucket":
                edges = [400, 700]
            elif output_col == "floor_bucket":
                edges = [150, 220]
            else:
                edges = []
            bins = [-np.inf] + edges + [np.inf] if edges else [-np.inf, np.inf]
            df[output_col] = pd.cut(values, bins=bins).astype(str).replace({"nan": np.nan})

    _apply_bucket("bed", "bed_bucket")
    _apply_bucket("bath", "bath_bucket")
    _apply_bucket("landSizeM2", "land_bucket")
    _apply_bucket("floorSizeM2", "floor_bucket")

    # Build the model-ready frame
    prepared = pd.DataFrame(index=df.index, columns=model_columns)
    for idx, row in df.iterrows():
        for column in model_columns:
            if column in row.index:
                prepared.at[idx, column] = row[column]

        suburb = str(row.get("suburb", "")).strip()
        sale_year_month = pd.to_numeric(row.get("saleYearMonth"), errors="coerce")
        if suburb and not pd.isna(sale_year_month):
            try:
                # Provide extra filters for group-aware feature hydration when available
                extra_filters: Dict[str, Any] = {}
                for key in ("propertyType", "bed_bucket", "bath_bucket", "floor_bucket"):
                    if key in prepared.columns and not pd.isna(prepared.at[idx, key]):
                        extra_filters[key] = prepared.at[idx, key]
                # Include street when available to hydrate street-level aggregates accurately
                if "street" in row.index and isinstance(row["street"], str) and row["street"].strip():
                    extra_filters["street"] = row["street"].strip()
                reference = fetch_reference_features(
                    suburb=suburb,
                    sale_year_month=int(sale_year_month),
                    columns=[col for col in model_columns if col not in {"saleYearMonth"}],
                    extra_filters=extra_filters if extra_filters else None,
                )
            except FileNotFoundError:
                reference = None

            if reference is not None:
                for column, value in reference.items():
                    if column in prepared.columns and (
                        pd.isna(prepared.at[idx, column]) or column not in row.index
                    ):
                        prepared.at[idx, column] = value

    for column in model_columns:
        if column not in prepared.columns:
            continue
        if prepared[column].isna().all():
            if column in impute_numeric:
                prepared[column] = impute_numeric[column]
            elif column in impute_categorical:
                prepared[column] = impute_categorical[column]

    # Enforce dtypes
    for column in numeric_features:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in categorical_features:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("object")

    # Final imputation fallback
    for column, value in impute_numeric.items():
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(value)

    for column, value in impute_categorical.items():
        if column in prepared.columns:
            prepared[column] = prepared[column].fillna(value)

    # Ensure categorical bands become plain strings
    if "propertyAgeBand" in prepared.columns:
        prepared["propertyAgeBand"] = prepared["propertyAgeBand"].astype(str).replace({"nan": np.nan})
        prepared["propertyAgeBand"] = prepared["propertyAgeBand"].fillna(impute_categorical.get("propertyAgeBand", "Unknown"))

    return prepared


def predict_property_price(
    yearmonth: int,
    bed: int,
    bath: int,
    car: int,
    property_type: str,
    street: str,
    suburb: str,
    *,
    land_size: Optional[float] = None,
    floor_size: Optional[float] = None,
    year_built: Optional[int] = None,
    model: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> float:
    """Predict sale price for a single property using the trained model."""
    actual, _, _, _, _ = _predict_single(
        yearmonth=yearmonth,
        bed=bed,
        bath=bath,
        car=car,
        property_type=property_type,
        street=street,
        suburb=suburb,
        land_size=land_size,
        floor_size=floor_size,
        year_built=year_built,
        model=model,
        metadata=metadata,
    )
    return actual


def predict_property_prices_batch(
    properties: List[Dict[str, Any]]
) -> List[float]:
    """Predict property prices for multiple properties at once."""
    model, metadata = load_trained_model()
    input_data = _prepare_prediction_data(properties, metadata)
    raw_predictions = model.predict(input_data)

    results: List[float] = []
    for idx, raw_value in enumerate(raw_predictions):
        actual, _, _ = _convert_prediction(float(raw_value), input_data.iloc[idx], metadata)
        results.append(actual)
    return results




def predict_with_confidence_interval(
    yearmonth: int,
    bed: int,
    bath: int,
    car: int,
    property_type: str,
    street: str,
    suburb: str,
    *,
    land_size: Optional[float] = None,
    floor_size: Optional[float] = None,
    year_built: Optional[int] = None,
    confidence_level: float = 0.95,
    model: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Predict property price and a simple confidence interval."""
    price, raw_value, base_value, model, metadata = _predict_single(
        yearmonth=yearmonth,
        bed=bed,
        bath=bath,
        car=car,
        property_type=property_type,
        street=street,
        suburb=suburb,
        land_size=land_size,
        floor_size=floor_size,
        year_built=year_built,
        model=model,
        metadata=metadata,
    )

    rmse = 0.0
    summary_candidates = [
        MODELS_DIR / "model_final" / "best_model.json",
        MODELS_DIR / "best_model.json",
    ]
    for candidate in summary_candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception:
            continue
        metrics = payload.get("metrics") or {}
        if isinstance(metrics, dict):
            rmse_value = metrics.get("val_rmse")
            if isinstance(rmse_value, (int, float)):
                rmse = float(rmse_value)
                if rmse:
                    break

    if rmse == 0.0:
        score_candidates = []
        # New daily layout
        score_candidates.extend(MODELS_DIR.glob("*/*/model_scores.csv"))
        # Legacy wildcard under roots
        for directory in (MODELS_DIR / "model_final", MODELS_DIR):
            if directory.exists():
                score_candidates.extend(directory.glob("model_scores_*.csv"))
        score_files = sorted(
            score_candidates,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if score_files:
            try:
                scores_df = pd.read_csv(score_files[0])
            except Exception:
                scores_df = pd.DataFrame()
            if not scores_df.empty and {"val_r2", "val_rmse"}.issubset(scores_df.columns):
                top_row = scores_df.sort_values("val_r2", ascending=False).iloc[0]
                rmse = float(top_row.get("val_rmse", 0.0) or 0.0)

    z_score = 1.96 if confidence_level == 0.95 else 2.58
    if base_value is not None and _target_is_delta(metadata):
        margin_of_error = z_score * rmse * base_value
    else:
        margin_of_error = z_score * rmse

    return {
        "predicted_price": price,
        "lower_bound": price - margin_of_error,
        "upper_bound": price + margin_of_error,
        "confidence_level": confidence_level,
        "raw_prediction": raw_value,
        "base_value": base_value,
    }




if __name__ == "__main__":
    # Example usage that mirrors the production contract
    try:
        sample_street = "Mitcham Road"
        sample_suburb = "Mitcham"

        try:
            streets = list_streets()
            suburbs = list_suburbs()
            if streets:
                sample_street = streets[0]
            if suburbs:
                sample_suburb = suburbs[0]
        except FileNotFoundError:
            # Fall back to sensible defaults when no derived dataset is available
            pass

        future_year = datetime.now().year + 1
        future_yearmonth = future_year * 100 + 1  # January of next year

        price = predict_property_price(
            yearmonth=future_yearmonth,
            bed=3,
            bath=2,
            car=2,
            property_type="House",
            street=sample_street,
            suburb=sample_suburb,
        )
        print(f"Predicted price: ${price:,.2f}")

        result = predict_with_confidence_interval(
            yearmonth=future_yearmonth,
            bed=3,
            bath=2,
            car=2,
            property_type="House",
            street=sample_street,
            suburb=sample_suburb,
        )
        print(f"Predicted price: ${result['predicted_price']:,.2f}")
        print(f"Confidence interval: ${result['lower_bound']:,.2f} - ${result['upper_bound']:,.2f}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model has been trained before making predictions.")
    # Normalise street if present to match derive extraction (title-cased, unit/number stripped)
    if "street" in df.columns:
        df["street"] = df["street"].apply(lambda s: _extract_street(s, {"unknown_value": "Unknown"}))
