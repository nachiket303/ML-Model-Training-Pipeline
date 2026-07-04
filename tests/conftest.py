"""Shared pytest fixtures.

Tests must never depend on the full 7k-row dataset (it may be absent in CI and is slow), so a
tiny synthetic CSV is generated with the same schema, both target classes, and the real-world
``TotalCharges`` quirk (blank strings for tenure=0 accounts). A matching YAML config points at it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_SAMPLE_ROWS = 200
_SEED = 7


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a small Telco-schema CSV to a temp path and return it.

    The frame mirrors the real dataset closely enough to exercise the whole pipeline: mixed
    numeric/categorical columns, a string ``TotalCharges`` with blanks for tenure=0 rows, an id
    column to drop, and a Yes/No target with mild signal so models beat chance.
    """
    rng = np.random.default_rng(_SEED)
    n = _SAMPLE_ROWS
    internet_levels = ["Yes", "No", "No internet service"]
    frame = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-ABCDE" for i in range(n)],
            "gender": rng.choice(["Male", "Female"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(internet_levels, n),
            "OnlineBackup": rng.choice(internet_levels, n),
            "DeviceProtection": rng.choice(internet_levels, n),
            "TechSupport": rng.choice(internet_levels, n),
            "StreamingTV": rng.choice(internet_levels, n),
            "StreamingMovies": rng.choice(internet_levels, n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)"], n
            ),
            "MonthlyCharges": rng.uniform(18.0, 120.0, n).round(2),
        }
    )

    # Force a few tenure=0 accounts so the TotalCharges blank-coercion path is always exercised.
    frame.loc[:4, "tenure"] = 0
    total_charges = (frame["tenure"] * frame["MonthlyCharges"]).round(2).astype(str)
    total_charges[frame["tenure"] == 0] = " "  # the real dataset's blank-string quirk
    frame["TotalCharges"] = total_charges

    logit = (
        -1.5
        + 0.03 * (frame["MonthlyCharges"] - 60.0)
        - 0.04 * frame["tenure"]
        + rng.normal(0.0, 0.5, n)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    frame["Churn"] = np.where(rng.uniform(0.0, 1.0, n) < churn_prob, "Yes", "No")
    # Guarantee both classes are present regardless of sampling.
    frame.loc[0, "Churn"] = "Yes"
    frame.loc[1, "Churn"] = "No"

    csv_path = tmp_path / "sample.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config_dict(sample_csv: Path, tmp_path: Path) -> dict:
    """A valid config dict pointing at the sample CSV, with small/fast settings."""
    return {
        "seed": 42,
        "data": {
            "path": str(sample_csv),
            "target": "Churn",
            "id_columns": ["customerID"],
            "positive_label": "Yes",
            "numeric_coerce_columns": ["TotalCharges"],
            "numeric_fill": 0.0,
        },
        "split": {"test_size": 0.25, "stratify": True},
        "cross_validation": {"enabled": False, "folds": 3},
        "tuning": {
            "enabled": False,
            "method": "grid",
            "scoring": "roc_auc",
            "n_iter": 5,
            "param_grids": {
                "logistic_regression": {"model__C": [0.1, 1.0]},
                "random_forest": {"model__n_estimators": [25, 50]},
            },
        },
        "models": [
            {"name": "logistic_regression", "params": {"max_iter": 1000}},
            {"name": "random_forest", "params": {"n_estimators": 25}},
        ],
        "evaluation": {"primary_metric": "roc_auc"},
        "tracking": {"mlflow": False, "experiment_name": "test"},
        "artifacts": {"root": str(tmp_path / "artifacts")},
    }


@pytest.fixture
def config_path(config_dict: dict, tmp_path: Path) -> Path:
    """Write ``config_dict`` to a YAML file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    return path
