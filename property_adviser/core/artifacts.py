"""Helpers for loading persisted pipeline artefacts with consistent validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import joblib

from property_adviser.core.io import read_json
from property_adviser.core.paths import MODELS_DIR, TRAINING_DIR

_DEFAULT_MODEL_CANDIDATES: tuple[Path, ...] = (
    MODELS_DIR / "best_model.joblib",
    MODELS_DIR / "best_model.pkl",
)


@dataclass(frozen=True)
class ModelArtifacts:
    """Container for a loaded model bundle and its companion metadata."""

    model: Any
    metadata: dict[str, Any]
    summary: dict[str, Any]
    model_path: Path
    metadata_path: Path
    summary_path: Optional[Path]


def locate_model_path(candidates: Optional[Iterable[Path]] = None) -> Path:
    """Return the first existing model path from the provided candidates."""
    search = tuple(candidates) if candidates is not None else _DEFAULT_MODEL_CANDIDATES
    for path in search:
        if path.exists():
            return path
    first = search[0] if search else MODELS_DIR / "best_model.joblib"
    raise FileNotFoundError(
        f"Trained model not found. Expected one of: {', '.join(str(p) for p in search)}."
    )


def load_feature_metadata(path: Optional[Path] = None) -> dict[str, Any]:
    """Load feature metadata emitted by the training stage."""
    metadata_path = path or (TRAINING_DIR / "feature_metadata.json")
    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"Feature metadata at {metadata_path} must be a JSON object.")
    return metadata


def _extract_model(bundle: Any) -> Any:
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        if hasattr(model, "predict"):
            return model
        raise ValueError("Model bundle does not expose a usable 'predict' method.")
    if hasattr(bundle, "predict"):
        return bundle
    raise ValueError("Loaded model artefact is not usable; expected an object with 'predict'.")


def load_model_artifacts(
    *,
    model_path: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
) -> ModelArtifacts:
    """Load the persisted model bundle along with metadata and summary JSON."""
    resolved_model_path = model_path or locate_model_path()
    bundle = joblib.load(resolved_model_path)
    model = _extract_model(bundle)

    resolved_metadata_path = metadata_path or (TRAINING_DIR / "feature_metadata.json")
    metadata = load_feature_metadata(resolved_metadata_path)

    resolved_summary_path = summary_path or (MODELS_DIR / "best_model.json")
    summary = {}
    if resolved_summary_path.exists():
        raw_summary = read_json(resolved_summary_path)
        if isinstance(raw_summary, dict):
            summary = raw_summary

    return ModelArtifacts(
        model=model,
        metadata=metadata,
        summary=summary,
        model_path=resolved_model_path,
        metadata_path=resolved_metadata_path,
        summary_path=resolved_summary_path if resolved_summary_path.exists() else None,
    )


__all__ = [
    "ModelArtifacts",
    "locate_model_path",
    "load_feature_metadata",
    "load_model_artifacts",
]
