"""End-to-end pipeline smoke tests on the synthetic sample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.model_selection import GridSearchCV

from churn_pipeline.config import load_config
from churn_pipeline.pipeline import run_pipeline


def test_pipeline_with_tuning_produces_artifacts(config_path: Path) -> None:
    """Running with tuning writes a versioned artifact, complete metadata, and best params.

    Asserts the metadata contains everything needed to reproduce the run and that the comparison
    summary names a winner.
    """
    config = load_config(config_path)
    config.tuning.enabled = True
    result = run_pipeline(config)

    assert len(result.results) == len(config.models)
    assert result.winner is not None
    assert result.summary_path is not None and result.summary_path.is_file()

    for model_result in result.results:
        # Artifact files exist.
        model_file = model_result.artifact_dir / "model.joblib"
        metadata_file = model_result.artifact_dir / "metadata.json"
        assert model_file.is_file()
        assert metadata_file.is_file()

        # Metadata is complete enough to reproduce the run.
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        for key in (
            "config",
            "config_hash",
            "data_hash",
            "seed",
            "library_versions",
            "python_version",
            "metrics",
        ):
            assert key in metadata, f"metadata missing '{key}'"

        # Tuning ran, so best params and a CV score were recorded.
        assert model_result.best_params is not None
        assert metadata["best_params"] is not None
        assert metadata["best_cv_score"] is not None

    # The comparison summary on disk agrees on the winner.
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["winner"] == result.winner
    assert summary["models"][0]["rank"] == 1


def test_search_sees_only_training_data(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The hyperparameter search is fitted on the training split only — never the test set.

    We spy on ``GridSearchCV.fit`` and assert every call received exactly ``train_rows`` rows, so
    the held-out test set cannot have leaked into the search. The single test evaluation happens
    afterwards in the orchestrator.
    """
    config = load_config(config_path)
    config.tuning.enabled = True

    seen_row_counts: list[int] = []
    original_fit = GridSearchCV.fit

    def spy_fit(
        self: GridSearchCV, features: object, target: object = None, **kwargs: object
    ) -> GridSearchCV:
        seen_row_counts.append(len(features))  # type: ignore[arg-type]
        return original_fit(self, features, target, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", spy_fit)
    result = run_pipeline(config)

    assert result.train_rows > 0
    assert result.test_rows > 0
    # One search fit per model, each on exactly the training rows (never the full dataset).
    assert seen_row_counts == [result.train_rows] * len(config.models)
    assert all(count < result.train_rows + result.test_rows for count in seen_row_counts)
