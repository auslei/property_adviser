# Development Standards & Guidelines

This document sets baseline practices for development within the project,  
with special emphasis on **agentic programming** (LLM-driven workflows).

---

## 🔑 Core Principles

1. **Separation of Concerns**
   - Clean vs derive vs model vs app kept in different modules.
   - Config files drive behaviour (no hard-coded logic).

2. **Reproducibility**
   - Every run can be reproduced from config + data.
   - Outputs saved with clear naming (`cleaned.parquet`, `derived.parquet`).

3. **Transparency**
   - Always log what happens (rows in/out, columns added, warnings).
   - Use JSON-style structured logs for easy parsing.

4. **Simplicity**
   - Prefer YAML-driven rules to Python if logic is simple (mappings, thresholds).
   - Functions should be small, composable, and testable.

---

## ⚙️ Config Standards

- **One concept = one YAML file**  
  Example: `pp_clean.yml`, `pp_derive.yml`.

- **YAML keys are declarative**  
  - `enabled: true/false` → simple toggles  
  - Explicit input/output column names

- **Schema checks**  
  Always validate that required keys exist before running a derivation.

---

## 🧑‍💻 Code Standards

- **Functions**
  - One purpose each (`derive_ratio`, `derive_age`).
  - Use helpers for repeat logic (`_safe_ratio`, `_norm_token`).

- **Logging**
  - Use `log(...)` for success, `warn(...)` for skipped/missing cases.
  - Log key parameters (`rows_in`, `rows_out`, `output`, etc.).

- **Error handling**
  - Fail *soft* in cleaning (warn + skip).
  - Fail *hard* only in CLI entrypoint if config is invalid.

- **Testing**
  - Unit test each derive/clean step with small sample data.
  - Regression tests on known datasets.

---

## 🧠 Agentic Programming Guidelines

1. **Agent role clarity**
   - Each agent has a narrow role (e.g., *clean data*, *derive features*, *train model*).
   - Avoid mixing responsibilities.

2. **Tool interface**
   - Wrap functions as clear “tools” with input schema + output schema.
   - Agents should call tools, not internal utilities directly.

3. **Config-driven agents**
   - Agents should respect YAML configs rather than invent parameters.
   - Example: an agent shouldn’t hardcode “Unit = flat”; it should read `pp_clean.yml`.

4. **Logging & Traceability**
   - Agents must output reasoning logs (actions taken, decisions made).
   - All steps should be reconstructible after the fact.

5. **Human-in-the-loop**
   - For ambiguous mappings or thresholds, design agents to ask for confirmation.
   - Never silently overwrite domain-specific rules.

6. **Extensibility**
   - When adding new derivations or mappings, prefer to extend YAML rather than code.
   - Agents should be able to propose config updates (in YAML) for review.

---

## ✅ Summary

- Keep **cleaning**, **derivation**, **model**, and **app** modules separate.
- Drive logic through **config files**.
- Maintain **structured logs** and **reproducibility**.
- For agentic workflows: treat each function as a tool, log everything, and keep humans in the loop for domain-specific rules.
