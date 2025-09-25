# src/common/app_logging.py
import json
import logging
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Mapping, Optional

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger once. Later calls won't add duplicate handlers."""
    if logging.getLogger().handlers:
        # Already configured elsewhere (tests, notebooks, streamlit, etc.)
        logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _to_json(payload: Mapping[str, Any]) -> str:
    # Be forgiving with values (e.g., numpy types, Paths)
    return json.dumps(payload, ensure_ascii=False, default=str)

def log(action: str, status: str = "ok", **kw) -> None:
    logging.info(_to_json({"action": action, "status": status, **kw}))

def warn(action: str, **kw) -> None:
    logging.warning(_to_json({"action": action, "status": "warn", **kw}))

def error(action: str, **kw) -> None:
    logging.error(_to_json({"action": action, "status": "error", **kw}))

def log_exc(action: str, exc: BaseException, **kw) -> None:
    """Convenience: include exception type/message and stack trace."""
    logging.error(
        _to_json({"action": action, "status": "error", "error": str(exc), "exc_type": type(exc).__name__, **kw}),
        exc_info=True,
    )

@contextmanager
def time_block(action: str, **meta):
    start = perf_counter()
    try:
        yield
    finally:
        dur = perf_counter() - start
        log(f"{action}.timing", duration_sec=round(dur, 3), **meta)