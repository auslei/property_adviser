"""Utility helpers to promote trained models into a final deployment location."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional, Sequence

from property_adviser.core.app_logging import log
from property_adviser.core.io import ensure_dir, read_json
from property_adviser.core.paths import MODELS_DIR, PROJECT_ROOT, TRAINING_DIR


@dataclass
class PromotionRecord:
    """Stores the outcome of a single target promotion."""

    target_name: str
    target: str
    forecast_window: Optional[str]
    model: str
    timestamp: str
    source_model: Path
    promoted_model: Path
    promoted_summary: Optional[Path]
    promoted_scores: Optional[Path]
    metadata_path: Optional[Path]


class PromotionError(RuntimeError):
    pass


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _load_report(report_path: Optional[str | Path]) -> tuple[Path, dict]:
    if report_path:
        resolved = _resolve_path(report_path)
    else:
        reports = sorted(MODELS_DIR.glob("training_report_*.json"))
        if not reports:
            raise PromotionError("No training report found under models/.")
        resolved = reports[-1]
    payload = read_json(resolved)
    if not isinstance(payload, dict):
        raise PromotionError(f"Training report at {resolved} is not a JSON object.")
    return resolved, payload


def _best_val_r2(entry: dict) -> float:
    best_model = entry.get("best_model")
    scores = entry.get("scores") or []
    for score in scores:
        if isinstance(score, dict) and score.get("model") == best_model:
            value = score.get("val_r2")
            if isinstance(value, (int, float)):
                return float(value)
    return float("-inf")


def _best_per_forecast_window(entries: List[dict]) -> List[dict]:
    ordered_keys: List[str] = []
    best_by_window: dict[str, tuple[dict, float]] = {}
    for entry in entries:
        key_raw = entry.get("forecast_window") or entry.get("target") or entry.get("name")
        key = str(key_raw) if key_raw is not None else ""
        score = _best_val_r2(entry)
        if key not in best_by_window:
            ordered_keys.append(key)
            best_by_window[key] = (entry, score)
            continue
        _, existing_score = best_by_window[key]
        if score > existing_score:
            best_by_window[key] = (entry, score)
    return [best_by_window[key][0] for key in ordered_keys]


def _select_targets(report: dict, targets: Optional[Sequence[str]], include_all: bool) -> List[dict]:
    entries = report.get("targets") or []
    if not isinstance(entries, list) or not entries:
        raise PromotionError("Training report does not contain any target entries.")

    if include_all:
        return [entry for entry in entries if isinstance(entry, dict)]

    if targets:
        requested = {t.lower() for t in targets}
        selected = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and (entry.get("name", "").lower() in requested or entry.get("target", "").lower() in requested)
        ]
        if not selected:
            raise PromotionError(f"Targets {targets} were not found in training report.")
        return selected

    best = report.get("best_overall")
    if isinstance(best, dict):
        name = str(best.get("target_name") or best.get("name") or "").lower()
        target = str(best.get("target") or "").lower()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("name", "").lower() == name or entry.get("target", "").lower() == target:
                return [entry]

    # Fallback: choose the most recent entry
    entries_sorted = sorted(
        (entry for entry in entries if isinstance(entry, dict) and entry.get("timestamp")),
        key=lambda item: str(item.get("timestamp")),
        reverse=True,
    )
    if entries_sorted:
        return [entries_sorted[0]]
    raise PromotionError("Unable to determine a target to promote from the training report.")


def _copy_if_exists(source: Optional[Path], destination: Path) -> Optional[Path]:
    if source is None or not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def promote_models(
    *,
    report_path: Optional[str | Path] = None,
    destination: Optional[str | Path] = None,
    targets: Optional[Sequence[str]] = None,
    include_all_targets: bool = False,
    copy_scores: bool = False,
    activate_target: Optional[str] = None,
    best_per_window: bool = True,
) -> dict:
    """Promote one or more trained models into a final deployment directory.

    Parameters
    ----------
    report_path:
        Specific training_report_*.json to read. Defaults to the latest report in models/.
    destination:
        Directory that will receive the promoted artefacts. Defaults to models/model_final.
    targets:
        Optional collection of target names (or target column identifiers) to promote.
    include_all_targets:
        Promote every target present in the training report when True.
    copy_scores:
        Copy the per-target model_scores CSV into the destination if available.
    activate_target:
        Target name (or column identifier) to mark as the active bundle. If omitted and exactly
        one model is promoted, that target becomes active automatically. The active model is
        copied to models/model_final/best_model.joblib (and companion JSON), and its metadata
        is written to data/training/feature_metadata.json for prediction usage.
    best_per_window:
        When True (default), select the highest validation R^2 entry per forecast window among
        the chosen targets to avoid promoting multiple bundles for the same horizon.
    """

    report_file, report = _load_report(report_path)
    dest_base = _resolve_path(destination) if destination else (MODELS_DIR / "model_final")
    dest_base = ensure_dir(dest_base)

    selected = _select_targets(report, targets, include_all_targets)

    if best_per_window and not targets:
        grouped = _best_per_forecast_window(selected)
        if len(grouped) < len(selected):
            log(
                "train.promote.window_dedup",
                before=len(selected),
                after=len(grouped),
            )
        selected = grouped

    log(
        "train.promote.start",
        report=str(report_file),
        destination=str(dest_base),
        targets=[entry.get("name") for entry in selected],
    )

    promotions: List[PromotionRecord] = []
    for entry in selected:
        name = entry.get("name") or entry.get("target")
        if not name:
            raise PromotionError("Training report entry is missing a 'name' field.")
        target_name = str(name)
        forecast_window = entry.get("forecast_window")
        if forecast_window is not None:
            forecast_window = str(forecast_window)
        model_label = str(entry.get("best_model") or "unknown")
        timestamp = str(entry.get("timestamp") or "")

        canonical_model = _resolve_path(entry.get("canonical_model_path"))
        if not canonical_model.exists():
            raise PromotionError(f"Model artefact not found: {canonical_model}")

        summary_path = entry.get("summary_path")
        summary = _resolve_path(summary_path) if summary_path else None
        scores_entry = entry.get("scores_path")
        scores = _resolve_path(scores_entry) if scores_entry else None

        metadata_entry = entry.get("metadata_path")
        metadata = Path(metadata_entry) if metadata_entry else None
        if metadata and not metadata.is_absolute():
            metadata = PROJECT_ROOT / metadata

        dest_dir = ensure_dir(dest_base / target_name)
        promoted_model = _copy_if_exists(canonical_model, dest_dir / "best_model.joblib")
        promoted_summary = _copy_if_exists(summary, dest_dir / "best_model.json")
        promoted_scores = None
        if copy_scores and scores:
            promoted_scores = _copy_if_exists(scores, dest_dir / scores.name)

        promotions.append(
            PromotionRecord(
                target_name=target_name,
                target=str(entry.get("target") or target_name),
                forecast_window=forecast_window,
                model=model_label,
                timestamp=timestamp,
                source_model=canonical_model,
                promoted_model=promoted_model if promoted_model else dest_dir / "best_model.joblib",
                promoted_summary=promoted_summary,
                promoted_scores=promoted_scores,
                metadata_path=metadata,
            )
        )

        manifest_path = dest_dir / "promotion_manifest.json"
        manifest_payload = {
            "promoted_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source_report": str(report_file),
            "model": model_label,
            "timestamp": timestamp,
            "forecast_window": forecast_window,
            "source_model": str(canonical_model),
            "source_summary": str(summary) if summary else None,
            "source_scores": str(scores) if scores else None,
            "metadata_path": str(metadata) if metadata else None,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

        log(
            "train.promote.target",
            target=target_name,
            model=model_label,
            timestamp=timestamp,
            forecast_window=forecast_window,
            destination=str(dest_dir),
        )

    activated = None
    if activate_target:
        requested = activate_target.lower()
        record = next(
            (rec for rec in promotions if rec.target_name.lower() == requested or rec.target.lower() == requested),
            None,
        )
    elif len(promotions) == 1:
        record = promotions[0]
    else:
        record = None

    if record:
        active_dir = ensure_dir(dest_base)
        _copy_if_exists(record.promoted_model, active_dir / "best_model.joblib")
        if record.promoted_summary:
            _copy_if_exists(record.promoted_summary, active_dir / "best_model.json")
        if record.promoted_scores:
            _copy_if_exists(record.promoted_scores, active_dir / record.promoted_scores.name)
        if record.metadata_path and record.metadata_path.exists():
            _copy_if_exists(record.metadata_path, TRAINING_DIR / "feature_metadata.json")
        activated = record.target_name
        log(
            "train.promote.activated",
            target=record.target_name,
            model=record.model,
            destination=str(active_dir / "best_model.joblib"),
        )

    summary = {
        "report": str(report_file),
        "destination": str(dest_base),
        "promotions": [
            {
                "target_name": record.target_name,
                "target": record.target,
                "forecast_window": record.forecast_window,
                "model": record.model,
                "timestamp": record.timestamp,
                "promoted_model": str(record.promoted_model),
                "promoted_summary": str(record.promoted_summary) if record.promoted_summary else None,
                "promoted_scores": str(record.promoted_scores) if record.promoted_scores else None,
                "metadata_path": str(record.metadata_path) if record.metadata_path else None,
            }
            for record in promotions
        ],
        "activated_target": activated,
    }

    log(
        "train.promote.complete",
        destination=str(dest_base),
        activated=activated,
        promoted=len(promotions),
    )
    return summary


__all__ = ["PromotionRecord", "PromotionError", "promote_models"]
