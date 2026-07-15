# Telco Churn ML Training Pipeline

A configurable ML training pipeline: it takes a raw dataset, trains and compares several models, and
saves **versioned model artifacts with full metadata**.

Everything is driven by a single YAML config file. No file paths, columns, or hyperparameters are
hardcoded in the code. It runs locally, but each part is designed to map cleanly onto a Google Cloud
stack (Vertex AI, Kubeflow, BigQuery).

---


## What it does

The example dataset is the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
set (7,043 customers, 21 columns, about 26.5% churn). The pipeline:

1. Loads and **checks** the raw CSV or Parquet file before doing anything with it.
2. Fixes one known data problem (`TotalCharges`) on purpose.
3. Splits the data into a train and test set, keeping the churn rate the same in both.
4. For each model (`logistic_regression`, `random_forest`, `gradient_boosting`): trains it using
   the strategy set in the config (single fit, cross-validation, or hyperparameter search), then
   scores it **once** on the held-out test set.
5. Saves a **versioned artifact** (the trained model plus a metadata file) for each model.
6. Ranks the models and records the winner.
7. Optionally logs everything to **MLflow**.

Because everything comes from the config, the same tool runs on any dataset by pointing `--data` or
`--config` at it.

### Example result (tuned run on the full dataset)

| Rank | Model | ROC-AUC | F1 | Precision | Recall | Best params |
|-----:|-------|--------:|---:|----------:|-------:|-------------|
| 1 | gradient_boosting | 0.8455 | 0.570 | 0.673 | 0.495 | `n_estimators=50, learning_rate=0.1` |
| 2 | logistic_regression | 0.8421 | 0.604 | 0.657 | 0.559 | `C=1.0` |
| 3 | random_forest | 0.8386 | 0.577 | 0.658 | 0.513 | `max_depth=10, n_estimators=200` |

Churn is a bit imbalanced, so I use **ROC-AUC** as the main metric and report precision, recall, and
F1 next to accuracy, because accuracy on its own would be misleading here.

---

## Approach and design decisions

### One job per module

Each step of the ML lifecycle is its own small module under `src/churn_pipeline/`. The orchestrator
uses the components, but the components never depend on the orchestrator. This keeps the structure
simple to follow and lets every part be tested on its own.

| Module | What it does | Why it exists |
|--------|--------------|---------------|
| `config.py` | Loads and checks the YAML into typed objects | Config is an entry point; reject bad input early and give the rest of the code clean, typed values |
| `logging_config.py` | Sets up logging once; logs versions | One place owns logging; other modules just get a logger |
| `data.py` | Loads, checks, cleans, and encodes the target | Keeps all file-format and dataset-quirk knowledge in one place |
| `preprocessing.py` | Builds the (unfitted) `ColumnTransformer` | Encoding, scaling, and imputation as a reusable unit that is fit on training data only |
| `models.py` | Turns a config name + params into a model | Adding a model is a one-line change; the seed is set in one place |
| `tuning.py` | The three training strategies (search / CV / plain) | This is where data leakage could hide, so it is kept explicit and separate |
| `evaluate.py` | Works out the metrics from a trained model | Simple, reusable metric code; ROC-AUC from probabilities |
| `registry.py` | Versioned artifacts + reproducibility metadata | A local stand-in for a model registry |
| `tracking.py` | Optional MLflow logging (imported only if used) | Keeps an optional tool off the main path |
| `pipeline.py` | Runs the whole thing end to end | The only module that knows the order of steps |
| `cli.py` | The `churn-train` command | A single command to run the whole pipeline |

### Driven by config, nothing hardcoded

`config/default.yaml` sets the whole run: seed, data location, target and id columns, the split,
cross-validation, tuning (method plus a grid per model), the list of models and their base params,
the metric to rank by, tracking, and where artifacts go. The typed config (built with pydantic,
`extra="forbid"`) rejects unknown keys and out-of-range values, and it checks that a tuning grid
exists for every model when tuning is on. So a bad config fails when it loads, not three minutes
into training.

### Deliberate data cleaning

- **`TotalCharges`** comes as *text* with blank values for accounts where `tenure = 0` (11 rows in
  the full dataset). These are brand-new customers who have genuinely paid **nothing** yet, so a
  blank means *zero*, not *missing*. The data layer turns the column into numbers and fills those
  blanks with `0`. This is a deliberate choice, written down in the code, instead of letting the imputer
  guess a middle value. The imputer is still there as a safety net for truly unexpected missing
  values later. Which columns get this treatment comes from the config
  (`data.numeric_coerce_columns`), so nothing is hardcoded and it simply does nothing on datasets
  that don't have the column.
- **`customerID`** is a unique id with no predictive value, so it is dropped before training (via
  `data.id_columns`).

### No data leakage in tuning

This is the most important correctness point, and it lives in `tuning.py`:

- The thing handed to `GridSearchCV` / `RandomizedSearchCV` is the **whole**
  preprocessing + model `Pipeline`. So scikit-learn re-fits the imputers, scaler, and one-hot
  encoder **inside each fold**, on that fold's training part only, so test-fold information never leaks
  into training.
- The search only ever sees the **training split**. The training function has no test argument at
  all, so it *cannot* touch the test set (there is a test that checks this).
- With `refit=True`, the best model is retrained on the full training split using the winning
  params.
- The orchestrator then scores that model on the untouched test set **exactly once**. This single
  scoring step lives in `pipeline.py`, not inside the training strategies, so it is easy to see the
  test set is used once on **every** path (search, plain CV, and single fit).

### Reproducible by design

Pinned dependencies, a fixed Python target, and one seed threaded from the config through the split,
the models, and the CV splitter. Each model also writes a metadata file with the full config and its
hash, a hash of the input data, the library and Python versions, and the git commit. See
[Reproducibility](#reproducibility) below.

---

## How to run

Needs Python **3.11+** (see the note in [Assumptions](#assumptions) about the interpreter used to
build this).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux

# 2. Install the pinned dependencies and the package
pip install -r requirements.txt
pip install -e .

# 3. Run the pipeline
churn-train --config config/default.yaml
```

### One block, start to finish

```bash
python -m venv .venv && source .venv/Scripts/activate \
  && pip install -r requirements.txt && pip install -e . \
  && churn-train --config config/default.yaml \
  && pytest -q
```

### Command options

```bash
churn-train --help
churn-train --config config/default.yaml                          # default run (tuning on)
churn-train --config config/default.yaml --data path/to/other.csv # run on ANY dataset
churn-train --config config/default.yaml --verbose                # DEBUG logging
```

### Where things are saved

- **Model artifacts:** `artifacts/<model_name>/<timestamp>/model.joblib` + `metadata.json`
- **Run comparison:** `artifacts/runs/<timestamp>_summary.json`
- The final ranked summary is also printed to the screen at the end of every run.

### MLflow (optional)

Set `tracking.mlflow: true` in the config, then:

```bash
churn-train --config config/default.yaml
mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
```

Params (including tuned best params), metrics, and the model are logged per model. MLflow is only
imported when it is used, so the pipeline runs the same way with it turned off.

### Tests

```bash
pytest                       # fast tests on synthetic data (no need for the full CSV)
black --check . && ruff check .
```

---

## Assumptions

- **Binary classification** on the `Churn` (Yes/No) target; the positive label is configurable.
- **The data has only two column types:** text (object) and numeric. Other types such as datetime
  are not expected and are not handled specially.
- **Input files are CSV or Parquet only.** No other formats are supported.
- **The data is outlier-free.** No outlier detection or removal is done before training.
- **No new feature engineering is needed.** The pipeline uses the columns as they are and does not
  create extra features during training.
- **`TotalCharges` blanks mean zero** (tenure = 0). Explained above and in the code.
- **The top model is chosen by ROC-AUC.** This is a reasonable default for a mildly imbalanced
  problem, but picking the truly best metric would need a deeper look at all metrics and more
  experiments (for example, the business cost of a false positive vs a false negative).
- **The target column has no missing values.** Rows with a missing label are not expected, so the
  target is assumed to be complete for every row.

---

## Reproducibility

Any run can be reproduced from the metadata it writes:

- **Pinned dependencies** (`requirements.txt`) and a **fixed Python target** (3.11).
- **One seed** from `config.seed`, used in the split, every model (`random_state`), and the CV
  splitter, set once and passed in, never hardcoded.
- **A `metadata.json` per model** with: the full resolved config, a **config hash** (SHA-256 of
  canonical JSON), a **data hash** (SHA-256 of the input file), library versions, the Python
  version, the git commit, all metrics, and (when tuning ran) the best params and best CV score.

To reproduce a past run: check out its `git_commit`, run `pip install -r requirements.txt`, and run
with the `config` from its metadata against data whose hash matches `data_hash`. The config hash
lets you confirm two runs used the same settings without comparing YAML by hand.

---

## Production notes (local by design, production-aware)

The build is local on purpose, but each part maps onto the team's GCP stack:

| This project (local) | On GCP |
|----------------------|--------|
| `data.py` reading a CSV/Parquet file | **BigQuery**: swap the reader for a BigQuery query; the checking and cleaning stay the same |
| `pipeline.py` stages (load → split → preprocess → train → evaluate → register) | **Kubeflow / Vertex Pipeline** steps: each function becomes a containerized component |
| `registry.py` versioned artifacts + metadata | **Vertex AI Model Registry**: same idea; saving becomes a registry upload |
| `config/*.yaml` + `.github/workflows/ci.yml` | **CI/CD triggers**: runs already gate on black/ruff/pytest; extend to build images and submit Vertex jobs on merge |
| MLflow (`tracking.py`) | **Vertex AI Experiments / managed MLflow**: only the backend URI changes |
| Printed summary + logs | **Cloud Logging / Monitoring**: the central logging config is the one place to attach a handler |

Where the missing pieces attach:

- **Serving:** load a saved `model.joblib` (or the MLflow model) behind a FastAPI or Vertex
  Endpoint. The trained `Pipeline` already includes preprocessing, so serving needs no extra feature
  code.
- **Drift and performance monitoring:** log predictions and (later) real outcomes, compare live
  feature distributions against the training data referenced in metadata, and alert on drift or
  metric drops. The data/config hashes in metadata answer "what was this model trained on?".

---

## What I'd improve with more time

Broader design:

- **Support more problem types.** The pipeline only does binary classification right now. It could
  be extended to handle multiclass classification and regression as well.
- **Make it more general across datasets.** The overall structure was built with this Telco churn
  project in mind. With more time it could be made more general and robust so it works well across
  many different datasets, not just this one.
- **Handle large data.** The pipeline is not designed for datasets that do not fit in memory.
  Loading and scoring the data in chunks (batch or streaming) would be needed for that.
- **Support more models.** Only three model types are wired in. More could be added through the same
  factory, giving more options to compare.
- **Stronger tests.** The unit tests could be much stronger and cover every part of the pipeline,
  not just the main paths.
- **Move from prototype to production grade.** This is a working prototype, not a production-ready
  pipeline. With more time it would be hardened by finding the critical bottlenecks and failure
  points and fixing them.

More specific technical steps:

- **Dockerisation.** Package the pipeline as a container image so each stage can run as a
  Kubeflow/Vertex component; add a `Dockerfile` and pin the base image.
- **Feature store.** Move feature definitions into a store (e.g. Vertex Feature Store) so training
  and serving use one feature computation and avoid train/serve skew.
- **Bigger hyperparameter search.** Swap the local grid/random search for **Vertex Vizier**
  (Bayesian search) with wider spaces and parallel trials; `tuning.py` is built to allow this.
- **Drift monitoring and auto-retraining.** A monitoring job (Evidently or a custom KS/PSI check)
  that triggers retraining through CI/CD when drift or a metric drop crosses a threshold.
- **Stronger data validation.** A schema contract (e.g. Pandera or Great Expectations) at ingestion,
  on top of the current checks.
- **Calibration and threshold choice.** Probability calibration and cost-aware threshold tuning,
  since churn actions have uneven costs.

---

## Project layout

```
telco-churn-pipeline/
├── README.md
├── pyproject.toml                # installable package (src layout) + black/ruff/pytest config
├── requirements.txt              # pinned reproducibility lock
├── config/
│   └── default.yaml              # the run configuration (tuning on)
├── data/                         # dataset lives here
├── src/churn_pipeline/           # config, data, preprocessing, models, tuning, evaluate,
│                                 # registry, tracking, pipeline, logging_config, cli
├── tests/                        # config / data / pipeline tests + synthetic-data fixture
├── .github/workflows/ci.yml      # black + ruff + pytest on Python 3.11
└── artifacts/                    # versioned models + run summaries (git-ignored)
```
