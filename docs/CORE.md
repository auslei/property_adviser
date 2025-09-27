# Core Module (`property_adviser.core`)

Shared utilities that underpin every pipeline stage (logging, configs, IO, orchestration).

## app_logging.py
Structured JSON-style logging helpers used across CLIs and Streamlit hooks.
- `setup_logging(verbose=False)`: configure root logger once per process.
- `log`, `warn`, `error`: emit events with name + key/value metadata.
- `log_exc`: capture exceptions with traceback details.
- `time_block(name)`: context manager for timing critical sections.

Example:
```python
from property_adviser.core.app_logging import setup_logging, log, log_exc, time_block

setup_logging(verbose=True)
log("startup", version="1.0.0")

try:
    with time_block("data.load"):
        ...  # expensive operation
except Exception as exc:
    log_exc("data.load", exc)
```

## config.py
Strict YAML loader + convenience accessors.
- `load_config(path: Path | str)`: read YAML (raises on missing/invalid files).
- `require(cfg, *keys)`: fetch nested keys without default fallbacks.

Example:
```python
from pathlib import Path
from property_adviser.core.config import load_config, require

cfg = load_config(Path("config/preprocessing.yml"))
data_path = require(cfg, "clean", "config_path")
threshold = require(cfg, "options", "verbose")
```

## io.py
Thin persistence layer for DataFrame artefacts.
- `save_parquet_or_csv(df, path)`: write using suffix-driven format.
- `load_parquet_or_csv(path)`: read either format, raising if file missing.
- `write_list(items, path)`: simple text writer (used for selected features).
- `ensure_dir(path)`: mkdir helper used by feature selection outputs.

Example:
```python
import pandas as pd
from pathlib import Path
from property_adviser.core.io import save_parquet_or_csv, load_parquet_or_csv

df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
save_parquet_or_csv(df, Path("data/tmp.csv"))
roundtrip = load_parquet_or_csv(Path("data/tmp.csv"))
```

## runner.py
Execution harness for derivation steps (handles logging + DataFrame validation).
- `run_step(name, fn, df, **params)`: logs start/end, timing, cell deltas, and enforces that the step returns a DataFrame.

Example:
```python
import pandas as pd
from property_adviser.core.runner import run_step
from property_adviser.core.app_logging import setup_logging

setup_logging()

def add_ratio(df, num, den, out):
    df[out] = df[num] / df[den]
    return df

df = pd.DataFrame({"x": [10, 20], "y": [2, 4]})
df = run_step("add_ratio", add_ratio, df, num="x", den="y", out="ratio")
```
