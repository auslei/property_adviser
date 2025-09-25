import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATA_DIR, PREPROCESS_DIR, RAW_DATA_PATTERN


def _collect_raw_sources() -> List[Dict[str, object]]:
    """Collects metadata about raw data files."""
    sources: List[Dict[str, object]] = []
    for path in sorted(DATA_DIR.glob(RAW_DATA_PATTERN)):
        if not path.exists():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            relative_path = str(path.relative_to(DATA_DIR))
        except ValueError:
            relative_path = str(path)
        sources.append(
            {
                "path": relative_path,
                "size": stat.st_size,
                "modified": round(stat.st_mtime, 3),
            }
        )
    return sources


def load_preprocess_metadata() -> Optional[Dict[str, object]]:
    """Loads the metadata of the preprocessed data."""
    metadata_path = PREPROCESS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def raw_data_signature() -> List[Dict[str, object]]:
    """Returns the signature of the raw data."""
    return _collect_raw_sources()


def raw_data_has_changed(existing_metadata: Optional[Dict[str, object]] = None) -> bool:
    """Checks if the raw data has changed since the last preprocessing."""
    if existing_metadata is None:
        existing_metadata = load_preprocess_metadata()
    if existing_metadata is None:
        return True
    previous_sources = existing_metadata.get("raw_sources", [])
    current_sources = raw_data_signature()
    return current_sources != previous_sources


def update_metadata_with_sources(metadata: Dict[str, object]) -> Dict[str, object]:
    """Updates the metadata with the current raw data sources."""
    metadata = dict(metadata)
    metadata["raw_sources"] = raw_data_signature()
    return metadata
