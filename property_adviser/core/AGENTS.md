# Core Module Agent Guide

## Purpose & Scope
- Provide reusable utilities (logging, configuration, IO, filesystem paths, orchestration helpers) that keep the broader system consistent.
- Act as the only shared dependency between modules, minimising cross-package coupling while promoting reuse.

## Design Commitments
- **Clear interface**: Expose stable functions and constants; avoid leaking implementation details of consumer modules.
- **High cohesion**: Only generic, multi-module helpers live here. Module-specific logic should remain in its owning package.
- **Low coupling**: Downstream modules import from `property_adviser.core` rather than each other, keeping dependencies acyclic.
- **Reusability**: Add new helpers here when multiple modules benefit; document intended usage patterns.

## Key Components

### `paths.py`
- Defines canonical directories (`PROJECT_ROOT`, `DATA_DIR`, `PREPROCESS_DIR`, `TRAINING_DIR`, `MODELS_DIR`) and config paths.
- Import via `property_adviser.config` to keep references consistent.

### `app_logging.py`
- Structured logging helpers (`setup_logging`, `log`, `warn`, `error`, `log_exc`, `time_block`).
- Ensures every module emits uniform events and metadata fields.

### `config.py`
- Strict YAML loader (`load_config`) and `require` helper for nested key access.
- Encourages config-driven behaviour rather than hardcoded paths.

### `io.py`
- Persistence helpers (`save_parquet_or_csv`, `load_parquet_or_csv`, `write_list`, `ensure_dir`).
- Guarantees artefacts are written/read consistently across modules.

### `runner.py`
- `run_step` context for executing transformation functions with timing, logging, and DataFrame validation.
- Promote new orchestration helpers here when pipelines need shared behaviour.

## Maintenance Checklist
1. Keep APIs stable; if a breaking change is unavoidable, update all consumers and document migration steps.
2. Avoid pulling module-specific dependencies (e.g., scikit-learn) into `core` to prevent circular imports.
3. Ensure new helpers include lightweight tests and docstrings explaining expected usage.
