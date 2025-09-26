# common modules

## app_logging.py
Provides simple JSON-style logging helpers for consistent, structured logs.
- setup_logging(verbose): configure logging once
- log, warn, error: log messages with action + key/value metadata
- log_exc: logs exceptions with stack traces
- time_block: context manager to time blocks of code

Example:
```
    from property_adviser.common.app_logging import setup_logging, log, warn, error, log_exc, time_block

    setup_logging(verbose=True)
    log("startup", version="1.0.0")

    try:
        with time_block("data.load"):
            pass  # expensive operation
    except Exception as e:
        log_exc("data.load", e)
```
## config.py
Strict YAML config loader + accessor.
- load_config(path): loads YAML; raises if missing, empty, or malformed
- require(cfg, *keys): fetch nested config keys without defaults; raises if missing

Example:
```
    from pathlib import Path
    from property_adviser.common.config import load_config, require

    cfg = load_config(Path("config/preprocessing.yml"))
    data_path = require(cfg, "data_source", "path")
    encoding = require(cfg, "data_source", "encoding")
```

## io.py
Basic save/load utilities for DataFrames with Parquet → CSV fallback.
- save_parquet_or_csv(df, path): saves to Parquet; falls back to CSV if needed
- load_parquet_or_csv(path): loads Parquet if available, otherwise CSV; raises if none exist

Example:
```
    import pandas as pd
    from pathlib import Path
    from property_adviser.common.io import save_parquet_or_csv, load_parquet_or_csv

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    save_path = save_parquet_or_csv(df, Path("data/cleaned.parquet"))
    df2 = load_parquet_or_csv(Path("data/cleaned"))  # no extension needed
```

## runner.py
Generic step runner for DataFrame transformations.
- run_step(name, fn, df, **params): logs start/end, times execution, validates output type, reports row/column changes

Example:
```
    import pandas as pd
    from property_adviser.common.runner import run_step
    from property_adviser.common.app_logging import setup_logging

    setup_logging()

    def add_ratio(df, num, den, out):
        df[out] = df[num] / df[den]
        return df

    df = pd.DataFrame({"x": [10, 20], "y": [2, 4]})
    df = run_step("add_ratio", add_ratio, df, num="x", den="y", out="ratio")
```