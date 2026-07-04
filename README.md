# Telco Churn ML Training Pipeline

A config-driven training pipeline (Option 1 of the Sky UK MLE challenge) that ingests a raw
dataset, trains and evaluates multiple model types, optionally tunes them with leakage-free nested
cross-validation, and writes **versioned model artifacts with full reproducibility metadata**.

It is built to run locally but framed for production on Google Cloud (Vertex AI, Kubeflow,
BigQuery) — the local components are deliberate stand-ins for their managed equivalents.

---

## 1. Overview

Given the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers, 21 columns, ~26.5% churn), the pipeline:

1. Loads and **validates** the raw CSV/Parquet at the boundary.
2. Cleans a known data-quality issue (`TotalCharges`) deliberately.
3. Splits into a stratified, seeded train/test set.
4. For each configured model (`logistic_regression`, `random_forest`, `gradient_boosting`):
   trains via the configured strategy (single fit / cross-validation / hyperparameter search),
   then evaluates **once** on the held-out test set.
5. Writes a **versioned artifact** (serialized fitted pipeline + metadata) per model.
6. Produces a **model comparison** ranking and records the winner.
7. Optionally logs everything to **MLflow**.

Everything is driven by a single YAML config — no paths, columns, or hyperparameters are hardcoded
in logic — so the same tool runs on any dataset via `--data`/`--config`.

### Example result (tuned run on the full dataset)

| Rank | Model | ROC-AUC | F1 | Precision | Recall | Best params |
|-----:|-------|--------:|---:|----------:|-------:|-------------|
| 1 | gradient_boosting | 0.8455 | 0.570 | 0.673 | 0.495 | `n_estimators=50, learning_rate=0.1` |
| 2 | logistic_regression | 0.8421 | 0.604 | 0.657 | 0.559 | `C=1.0` |
| 3 | random_forest | 0.8386 | 0.577 | 0.658 | 0.513 | `max_depth=10, n_estimators=200` |

Churn is mildly imbalanced, so the primary metric is **ROC-AUC** and precision/recall/F1 are
reported alongside accuracy rather than accuracy alone.

---

## 2. Design decisions & architecture

### Separation of concerns

Each stage of the ML lifecycle is its own single-responsibility module under
`src/churn_pipeline/`. The orchestrator depends on the components; components never depend on the
orchestrator, so the dependency graph is acyclic and every piece is independently testable.

| Module | Responsibility | Why it exists |
|--------|----------------|---------------|
| `config.py` | Load + validate YAML into typed pydantic models | Config is a boundary; fail loudly on bad input, give the rest of the code typed objects |
| `logging_config.py` | Configure structured logging once; log env/versions | One place owns logging; libraries just `getLogger(__name__)` |
| `data.py` | Load, validate, clean, encode target | Isolates all file-format and dataset-quirk knowledge |
| `preprocessing.py` | Build the (unfitted) `ColumnTransformer` | Encoding/scaling/imputation as a reusable, fit-on-train-only unit |
| `models.py` | Factory: config name + params → estimator | Adding a model is a one-line change; seed injected centrally |
| `tuning.py` | The three training strategies (search / CV / plain) | Where leakage would hide — kept explicit and isolated |
| `evaluate.py` | Compute metrics from a fitted model | Pure, reusable metrics; ROC-AUC from probabilities |
| `registry.py` | Versioned artifacts + reproducibility metadata | Local stand-in for a model registry |
| `tracking.py` | Optional MLflow logging (lazy import) | Optional integration isolated from the critical path |
| `pipeline.py` | Orchestrate the end-to-end run | The only module that knows the order of operations |
| `cli.py` | `churn-train` Typer entry point | The single command for the demo |

### Config-driven, no hardcoding

`config/default.yaml` fully specifies the run: seed, data location/target/id columns, split,
cross-validation, tuning (method + per-model grids), model list + base params, evaluation metric,
tracking, and artifact root. The typed `Config` (pydantic, `extra="forbid"`) rejects unknown keys
and out-of-range values, and cross-validates that tuning grids exist when tuning is enabled — a
mis-configured run fails at load time, not three minutes into training.

### Deliberate data-quality handling

- **`TotalCharges`** ships as *strings* with blank values for `tenure = 0` accounts (11 rows in
  the full dataset). Those are brand-new customers that have genuinely accrued **zero** charges, so
  a blank means *zero*, not *unknown*. The data layer coerces the column to numeric and fills those
  structural blanks with `0` — a domain decision, documented in the code, rather than letting the
  preprocessing median-imputer substitute a misleading central value. The median-imputer remains as
  a safety net for genuinely unexpected missing values at inference time. Which columns get this
  treatment is **config-driven** (`data.numeric_coerce_columns`), so nothing is hardcoded and the
  path is a no-op on datasets without the column.
- **`customerID`** is a unique identifier with no predictive value and is dropped before training
  (via `data.id_columns`).

### No-leakage tuning nested with cross-validation

This is the most important correctness property, implemented in `tuning.py`:

- The estimator handed to `GridSearchCV`/`RandomizedSearchCV` is the **entire**
  preprocessing + model `Pipeline`. scikit-learn therefore re-fits the imputers, scaler, and
  one-hot encoder **inside each CV fold**, on that fold's training portion only — validation-fold
  statistics never leak into fitting.
- The search runs on the **training split only**. `tuning.train_model()` has no test parameter in
  its signature, so the search *cannot* see the test set (there is a unit test asserting this).
- With `refit=True`, the returned `best_estimator_` is retrained on the full training split using
  the winning params.
- The orchestrator then evaluates that estimator on the untouched test set **exactly once**. This
  single evaluation lives in `pipeline.py` (not in the training strategies) so it is easy to verify
  the test set is touched once on **every** code path (search, plain CV, and single fit).

### Reproducibility by construction

Pinned dependencies, a fixed Python target, seeds threaded from config through split/models/CV, and
per-model metadata that captures the resolved config + its hash, the input data hash, library and
Python versions, and the git commit. See [§5](#5-reproducibility).

---

## 3. How to run

Requires Python **3.11+** (see the note in [§4](#4-assumptions) on the interpreter used to build).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux

# 2. Install pinned dependencies + the package (editable)
pip install -r requirements.txt
pip install -e .

# 3. Run the pipeline (the single command an operator/interviewer runs)
churn-train --config config/default.yaml
```

### Copy-paste: everything, start to finish

```bash
python -m venv .venv && source .venv/Scripts/activate \
  && pip install -r requirements.txt && pip install -e . \
  && churn-train --config config/default.yaml \
  && pytest -q
```

### CLI options

```bash
churn-train --help
churn-train --config config/default.yaml                 # default: CV on, tuning off
churn-train --config config/tuning_example.yaml          # hyperparameter search (small grids)
churn-train --config config/default.yaml --data path/to/other.csv   # run on ANY dataset
churn-train --config config/default.yaml --verbose       # DEBUG logging
```

### Where things land

- **Model artifacts:** `artifacts/<model_name>/<timestamp>/model.joblib` + `metadata.json`
- **Run comparison / winner:** `artifacts/runs/<timestamp>_summary.json`
- The final ranked **run summary** is printed to stdout at the end of every run.

### MLflow (optional)

Set `tracking.mlflow: true` in the config, then:

```bash
churn-train --config config/default.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
```

Params (incl. tuned best params), metrics, and the model artifact are logged per model. MLflow is
lazily imported and fully optional — the pipeline runs identically when it is disabled.

### Tests

```bash
pytest                       # fast, synthetic-data tests (no dependency on the full CSV)
black --check . && ruff check .
```

---

## 4. Assumptions

- **Binary classification** on the `Churn` (Yes/No) target; `positive_label` is configurable.
- **The dataset fits in memory** (7k rows). Batch/streaming is out of scope for this option — see
  the production notes for how this maps to BigQuery/Dataflow at scale.
- **`TotalCharges` blanks are structurally zero** (tenure = 0). Justified above and in code.
- **Dataset filename:** the file on disk is `data/Telco_Customer_Churn_Dataset.csv` — the same
  standard Telco dataset the brief refers to as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Python interpreter:** the project targets **3.11** (`requires-python = ">=3.11"`, CI pinned to
  3.11). The build machine only had **3.13** installed, so local verification ran on 3.13; the pins
  were chosen to support both, and every run records the *actual* interpreter version in its
  metadata. CI runs the pinned install + checks on 3.11.
- **`requirements.txt` pins direct dependencies** plus the numerically-critical transitive
  libraries (numpy/scipy/joblib). `pyproject.toml` carries abstract lower bounds; `requirements.txt`
  is the concrete lock.

---

## 5. Reproducibility

Any run can be reproduced from the metadata it writes:

- **Pinned dependencies** (`requirements.txt`) + **fixed Python target** (3.11).
- **Seed** threaded from `config.seed` through the train/test split, every model
  (`random_state`), and the CV splitter — set once in `pipeline.set_global_seeds` and injected in
  the model factory, never hardcoded inline.
- **Per-model `metadata.json`** captures: the fully-resolved config, a **config hash** (canonical
  JSON SHA-256), a **data hash** (SHA-256 of the input file), library versions
  (scikit-learn/pandas/numpy/joblib), the Python version, the git commit, all metrics, and — when
  tuning ran — the chosen best params and best CV score.

To reproduce a past run: check out its `git_commit`, `pip install -r requirements.txt`, and run
with the `config` embedded in its metadata against data whose hash matches `data_hash`. The config
hash lets you confirm two runs used identical settings without diffing YAML by hand.

---

## 6. Production considerations (local by design, production-aware)

The build is intentionally local, but each component maps cleanly onto the team's GCP stack:

| This project (local) | Production on GCP |
|----------------------|-------------------|
| `data.py` reading a CSV/Parquet file | **BigQuery** source — swap the reader for a `bigquery.Client` query (or Parquet export on GCS); the validation/cleaning layer is unchanged |
| `pipeline.py` stages (load → split → preprocess → train → evaluate → register) | **Kubeflow / Vertex Pipeline** components — each function becomes a containerized step with typed inputs/outputs |
| `registry.py` versioned artifacts + metadata | **Vertex AI Model Registry** — same idea (immutable, versioned, metadata-rich); `save_artifact` becomes a registry upload |
| `config/*.yaml` + `.github/workflows/ci.yml` | **CI/CD triggers** — config-driven runs already gate on black/ruff/pytest; extend to build images and submit Vertex Pipeline jobs on merge |
| MLflow (`tracking.py`) | **Vertex AI Experiments / managed MLflow** — the tracking abstraction stays; only the backend URI changes |
| Printed run summary + logs | **Cloud Logging / Monitoring** — the central logging config is the single place to attach a JSON/Cloud Logging handler |

Where the missing production pieces attach:

- **Serving:** load a registered `model.joblib` (or the MLflow model) behind a FastAPI/Vertex
  Endpoint. The fitted `Pipeline` already bundles preprocessing, so serving needs no feature code.
- **Drift & performance monitoring:** log predictions + (delayed) actuals, compare live feature
  distributions against the training data hash referenced in metadata, and alert on drift or metric
  degradation. The metadata's data/config hashes are the anchor for "what was this model trained
  on?".

---

## 7. What I'd improve with more time

Honest and specific — none of these are implemented:

- **Dockerisation / containerisation.** Package the pipeline as a container image so each stage can
  run as a Kubeflow/Vertex Pipeline component; add a `Dockerfile` and pin the base image.
- **Feature store.** Move feature definitions into a store (e.g. Vertex Feature Store) so training
  and serving share exactly one feature computation and avoid train/serve skew.
- **Larger-scale hyperparameter search.** Swap the local grid/random search for **Vertex Vizier**
  (Bayesian optimisation) with wider spaces and parallel trials; the `tuning.py` interface is
  designed to accommodate this.
- **Drift monitoring & automated retraining.** A monitoring job (Evidently or a custom
  KS/PSI check) that triggers retraining via CI/CD when drift or metric degradation crosses a
  threshold.
- **Richer data validation.** A schema contract (e.g. Pandera / Great Expectations) at ingestion
  in addition to the current boundary checks.
- **Full transitive dependency lock.** A hash-pinned lockfile (`pip-compile`/`uv`) for
  bit-for-bit environment reproduction, and a CI matrix across 3.11–3.13.
- **Calibration & threshold selection.** Probability calibration and business-cost-aware threshold
  tuning, since churn interventions have asymmetric costs.

---

## Project layout

```
telco-churn-pipeline/
├── CLAUDE.md                     # standing engineering context for the build
├── README.md
├── pyproject.toml                # installable package (src layout) + black/ruff/pytest config
├── requirements.txt              # pinned reproducibility lock
├── config/
│   ├── default.yaml              # CV on, tuning off
│   └── tuning_example.yaml       # hyperparameter search (small grids)
├── data/                         # dataset lives here
├── src/churn_pipeline/           # config, data, preprocessing, models, tuning, evaluate,
│                                 # registry, tracking, pipeline, logging_config, cli
├── tests/                        # config / data / pipeline tests + synthetic-data fixture
├── .github/workflows/ci.yml      # black + ruff + pytest on Python 3.11
└── artifacts/                    # versioned models + run summaries (git-ignored)
```
