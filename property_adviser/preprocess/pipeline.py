"""Preprocessing pipeline orchestration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from property_adviser.core.app_logging import log
from property_adviser.core.io import ensure_dir, save_parquet_or_csv
from property_adviser.preprocess.config import PreprocessConfig
from property_adviser.preprocess.preprocess_clean import clean_data
from property_adviser.preprocess.preprocess_derive import derive_features, build_segments
from property_adviser.core.config import load_config


@dataclass
class PreprocessResult:
    cleaned: pd.DataFrame
    derived: pd.DataFrame
    cleaned_path: Path
    derived_path: Path
    metadata_path: Path
    metadata: dict[str, Any]
    segments: Optional[pd.DataFrame] = None
    segment_path: Optional[Path] = None
    detailed_path: Optional[Path] = None


def _build_metadata(
    derived: pd.DataFrame,
    config: PreprocessConfig,
    cleaning_cfg: dict[str, Any],
    derivation_cfg: dict[str, Any],
    *,
    segments: Optional[pd.DataFrame] = None,
    segment_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    numeric_cols = derived.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = derived.select_dtypes(include=["object", "category"]).columns.tolist()
    sources = (
        sorted(derived["__source_file"].dropna().unique().tolist())
        if "__source_file" in derived.columns
        else []
    )

    metadata = {
        "rows": int(derived.shape[0]),
        "columns": derived.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "sources": sources,
        "config": {
            "preprocess": config.to_dict(),
            "cleaning": cleaning_cfg,
            "derivation": derivation_cfg,
        },
    }

    if segments is not None:
        metadata["segments"] = {
            "rows": int(segments.shape[0]),
            "columns": segments.columns.tolist(),
        }
        if segment_meta:
            metadata["segments"].update(segment_meta)

    return metadata


def run_preprocessing(config: PreprocessConfig, *, write_outputs: bool = True) -> PreprocessResult:
    """Execute the cleaning + derivation pipeline and optionally persist artefacts."""
    log(
        "preprocess.start",
        clean_config=str(config.clean.config_path),
        derive_config=str(config.derive.config_path),
    )

    cleaning_cfg = load_config(config.clean.config_path)
    if config.clean.dropped_rows_path:
        cleaning_cfg["dropped_rows_path"] = str(config.clean.dropped_rows_path)

    cleaned = clean_data(cleaning_cfg, dropped_rows_path=config.clean.dropped_rows_path)
    log("preprocess.clean.complete", rows=int(cleaned.shape[0]), cols=int(cleaned.shape[1]))

    derivation_cfg = load_config(config.derive.config_path)
    derived = derive_features(cleaned.copy(), derivation_cfg)
    log("preprocess.derive.complete", rows=int(derived.shape[0]), cols=int(derived.shape[1]))

    segments, segment_meta = build_segments(derived, derivation_cfg)
    if segments is not None:
        log("preprocess.segments", rows=int(segments.shape[0]), cols=int(segments.shape[1]))

    cleaned_path = config.clean.output_path
    derived_path = config.derive.output_path
    metadata_path = config.derive.metadata_path
    segment_output_path = config.derive.segment_output_path or derived_path
    detailed_output_path = config.derive.detailed_output_path

    metadata = _build_metadata(
        segments if segments is not None else derived,
        config,
        cleaning_cfg,
        derivation_cfg,
        segments=segments,
        segment_meta=segment_meta,
    )

    if write_outputs:
        ensure_dir(cleaned_path.parent)
        ensure_dir(derived_path.parent)
        ensure_dir(metadata_path.parent)

        save_parquet_or_csv(cleaned, cleaned_path)

        if segments is not None:
            ensure_dir(Path(segment_output_path).parent)
            save_parquet_or_csv(segments, segment_output_path)
            log("preprocess.segment_saved", path=str(segment_output_path))

            if detailed_output_path:
                ensure_dir(Path(detailed_output_path).parent)
                save_parquet_or_csv(derived, detailed_output_path)
                log("preprocess.detailed_saved", path=str(detailed_output_path))

            if segment_output_path != derived_path:
                save_parquet_or_csv(segments, derived_path)
        else:
            save_parquet_or_csv(derived, derived_path)

        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
        log(
            "preprocess.outputs_saved",
            cleaned=str(cleaned_path),
            derived=str(segment_output_path if segments is not None else derived_path),
            metadata=str(metadata_path),
        )

    return PreprocessResult(
        cleaned=cleaned,
        derived=segments if segments is not None else derived,
        cleaned_path=cleaned_path,
        derived_path=segment_output_path if segments is not None else derived_path,
        metadata_path=metadata_path,
        metadata=metadata,
        segments=segments,
        segment_path=segment_output_path if segments is not None else None,
        detailed_path=Path(detailed_output_path) if detailed_output_path else None,
    )


__all__ = ["PreprocessResult", "run_preprocessing"]
