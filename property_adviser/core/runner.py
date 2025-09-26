# src/common/runner.py
from typing import Callable
import pandas as pd
from .app_logging import log, error, time_block

def run_step(name: str, fn: Callable[..., pd.DataFrame], df: pd.DataFrame, **params) -> pd.DataFrame:
    before_rows = int(df.shape[0])
    before_cols = list(df.columns)
    log("step.begin", step=name, params=params, rows=before_rows, cols=len(before_cols))

    try:
        with time_block(f"step.{name}"):
            out = fn(df, **params)
    except Exception as e:
        error("step.fail", step=name, error=str(e), exc_type=type(e).__name__, params=params)
        raise

    if not isinstance(out, pd.DataFrame):
        error("step.fail", step=name, error="step did not return a DataFrame", returned_type=type(out).__name__)
        raise TypeError(f"Step '{name}' did not return a pandas DataFrame")

    after_rows = int(out.shape[0])
    out_cols = list(out.columns)
    new_cols = [c for c in out_cols if c not in before_cols]
    removed_cols = [c for c in before_cols if c not in out_cols]

    log(
        "step.end",
        step=name,
        rows_in=before_rows,
        rows_out=after_rows,
        cols=len(out_cols),
        new_cols=new_cols[:10],
        removed_cols=removed_cols[:10],
    )
    return out