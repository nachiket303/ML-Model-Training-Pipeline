# Project: Telco Churn ML Training Pipeline (Sky UK MLE interview, Option 1)

## Goal
A config-driven pipeline that ingests a raw dataset, trains 2+ model types, tunes and
evaluates them, and saves versioned model artifacts with full metadata. Local build,
production-framed for Vertex AI / Kubeflow / Vertex Model Registry / BigQuery.

## Commit hygiene
- Never add a Co-Authored-By line or any AI attribution to commit messages.

## Non-negotiable principles
- Strict separation of concerns: config, data, preprocessing, models, tuning, evaluation,
  artifact registry, orchestration are separate modules.
- Everything driven by a YAML config — no hardcoded paths, columns, or hyperparameters in logic.
- Full reproducibility: seed control everywhere; pinned deps; fixed Python version; metadata
  captures config hash, data hash, library versions, timestamp, git commit.
- Every design choice must be simple enough to explain out loud. Prefer clear, conventional
  code over clever code. Docstrings explain WHY, not just what.
- Do not over-engineer. No unnecessary abstraction. Restraint is a production standard —
  introduce abstraction only when a second use case justifies it.

## Engineering standard: production-grade, NOT experiment-grade
- Type hints on every function signature and return. No untyped functions.
- Module-level structured logging via the `logging` library (logging.getLogger(__name__)),
  configured once centrally. NO print statements except the final CLI run-summary.
- Specific custom exceptions where they aid clarity (ConfigError, DataValidationError).
  Never bare `except:`, never swallow errors silently.
- Validate at boundaries: config, data ingestion, public function inputs. Fail fast with
  actionable messages.
- Pure functions where possible; no hidden global state; no mutable default arguments.
- Constants and config/column keys defined once, never scattered magic strings/numbers.
- Docstrings (Google style) on every public function/class explaining intent and rationale.
- Seeds threaded through config, never hardcoded inline.
- Small, single-responsibility functions. Split anything doing data + training + saving.
- Clean dependency direction: orchestration depends on components, not vice versa. No circular
  imports.
- Code must pass black + ruff cleanly. Format accordingly.
- NO experiment-grade smells: no commented-out/dead code, no TODO-as-implementation, no
  notebook-style single-file scripts, no hardcoded paths/hyperparameters in logic.
- If a feature can't be done to this standard in the time available, prefer a smaller, cleaner,
  fully-defensible implementation over a larger sloppy one.

## Dataset facts (already verified — do not re-derive)
- 7,043 rows, 21 columns. Target = Churn (Yes/No), ~26.5% positive → mild imbalance, so report
  precision/recall/f1/roc_auc, not just accuracy.
- customerID is a unique ID → must be dropped before training.
- TotalCharges is stored as a STRING with blank values for tenure=0 customers → coerce to
  numeric and handle those rows deliberately in the data-validation layer, with a docstring
  justifying the choice.
- One-plus numeric features (tenure, MonthlyCharges, TotalCharges) and many categoricals →
  justifies a proper preprocessing layer with encoding + scaling.

## Local environment notes (this build)
- Data file on disk is `data/Telco_Customer_Churn_Dataset.csv` (standard Telco schema; the
  brief's `WA_Fn-UseC_-Telco-Customer-Churn.csv` is the same dataset under a different name).
- Target interpreter is Python 3.11 (`requires-python = ">=3.11"`, CI pinned to 3.11). The
  only interpreter installed on the build machine was 3.13, so local verification ran on 3.13;
  pins were chosen to support both. Every run records its actual Python version in metadata.
