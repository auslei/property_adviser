## Development Standards & Guidelines

This document sets baseline practices for development within the project, with emphasis on **agentic programming** and reproducible data science.

### Core Principles
1. **Separation of concerns** — clean vs derive vs feature select vs model vs app are distinct modules.
2. **Reproducibility** — all runs are determined by data + YAML. Outputs are versionable artefacts.
3. **Transparency** — structured logs with key params and counts.
4. **Simplicity** — prefer declarative YAML for mappings and thresholds.

### Configuration Standards
- One concept per YAML (`pp_clean.yml`, `pp_derive.yml`, `features.yml`, `model.yml`).
- Validate schema before running any step.
- No magic defaults hidden in code; CLI accepts `--config`.

### Code Standards
- Small, composable functions; shared helpers under `property_adviser/core`.
- **CLI/GUI split**:
  - Every CLI script exposes a **reusable function** (e.g. `run_feature_selection(cfg, ...)`) that accepts a **dict** and returns typed results (e.g. X, y, scores table).
  - `main()` is a thin wrapper: parse args → load YAML → call reusable function → print.
- **IO**:
  - Always use `core.io.save_parquet_or_csv` / `load_parquet_or_csv` / `ensure_dir` / `write_list`.
  - File format is decided by extension (csv/parquet).
- **Logging**:
  - Use `core.app_logging.log(...)` for structured events (e.g. `feature_selection.complete`, thresholds, counts).
- **Testing**:
  - Unit tests for derivations, cleaning rules, and selection logic (incl. MI normalisation cases).
  - Golden tests on small fixtures for end-to-end reproducibility.

### Feature Selection — Specific Guidelines
- **Metrics**: support `pearson_abs`, `mutual_info` (min–max **normalised** per run), `eta` (categorical association).
- **Selection logic**:
  - Compute `best_score = max(pearson_abs, mutual_info, eta)` and `best_metric`.
  - Threshold mode: select where `best_score >= correlation_threshold`.
  - Top-k mode: if enabled (`use_top_k` True or `top_k` present in config), select `k` highest by `best_score`.
- **Manual overrides**:
  - `include` features are always selected and labelled with reason “manual include”.
  - `exclude` features are never selected and labelled with “manual exclude (not selected)”.
- **Single scores file**:
  - Produce exactly one `feature_scores.(parquet|csv)` containing:
    - `feature, pearson_abs, mutual_info, eta, best_metric, best_score, selected, reason`.
- **Interfaces**:
  - Reusable function returns `scores_table`, `selected_columns`, `X`, `y` and optionally `output_dir`, `scores_path` for CLI.
  - GUI calls the same function with overrides (`include/exclude/use_top_k/top_k`) and `write_outputs=False`.
