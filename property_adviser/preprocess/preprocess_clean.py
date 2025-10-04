"""Backward-compatible re-export for the cleaning stage."""
from __future__ import annotations

from property_adviser.preprocess.clean.engine import clean_data, run_cleaning_stage

__all__ = ["clean_data", "run_cleaning_stage"]
