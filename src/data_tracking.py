import json
from pathlib import Path
from typing import Dict, List, Optional

from .config import DATA_DIR, PREPROCESS_DIR, RAW_DATA_PATTERN


def _collect_raw_sources() -> List[Dict[str, object]]:
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
    metadata_path = PREPROCESS_DIR / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def raw_data_signature() -> List[Dict[str, object]]:
    return _collect_raw_sources()


def raw_data_has_changed(existing_metadata: Optional[Dict[str, object]] = None) -> bool:
    if existing_metadata is None:
        existing_metadata = load_preprocess_metadata()
    if existing_metadata is None:
        return True
    previous_sources = existing_metadata.get("raw_sources", [])
    current_sources = raw_data_signature()
    return current_sources != previous_sources


def update_metadata_with_sources(metadata: Dict[str, object]) -> Dict[str, object]:
    metadata = dict(metadata)
    metadata["raw_sources"] = raw_data_signature()
    return metadata
