"""Pipeline orchestration.

This is the only module that knows the end-to-end order of operations. It depends on the
components (data, preprocessing, models, evaluate, registry) and never the other way round, so the
dependency graph stays acyclic and each component remains independently testable.

Flow: set seeds -> load & validate data -> stratified train/test split -> for each model, fit a
preprocessing+model Pipeline on the training split and evaluate once on the held-out test split ->
register a versioned artifact with reproducibility metadata. The single test-set evaluation lives
here (not inside the training strategies) so it is easy to see that the test set is touched exactly
once per model, on every code path.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import Config, ModelSpec
from .data import DataSummary, load_data
from .evaluate import evaluate_classifier
from .models import build_model
from .preprocessing import build_preprocessor
from .registry import build_metadata, hash_file, save_artifact

logger = logging.getLogger(__name__)

# The estimator step is named "model" so tuning grids can address it as ``model__<param>``.
_MODEL_STEP = "model"
_PREPROCESS_STEP = "preprocess"


@dataclass
class ModelResult:
    """Outcome of training and evaluating one model.

    Attributes:
        model_name: The model's config name.
        metrics: Held-out test metrics.
        artifact_dir: Directory the versioned artifact was written to.
        best_params: Tuned best parameters, or ``None`` when tuning did not run.
        best_cv_score: Best cross-validation score, or ``None`` when tuning did not run.
    """

    model_name: str
    metrics: dict[str, float]
    artifact_dir: Path
    best_params: dict[str, Any] | None = None
    best_cv_score: float | None = None


@dataclass
class PipelineResult:
    """Aggregate result of a full pipeline run.

    Attributes:
        results: One :class:`ModelResult` per configured model.
        data_summary: Summary produced by the data layer.
        primary_metric: Metric used to rank models.
        train_rows: Number of training rows used.
        test_rows: Number of held-out test rows used.
    """

    results: list[ModelResult] = field(default_factory=list)
    data_summary: DataSummary | None = None
    primary_metric: str = "roc_auc"
    train_rows: int = 0
    test_rows: int = 0


def set_global_seeds(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility.

    Per-model determinism is handled by injecting ``random_state`` in the model factory; seeding
    the global RNGs here covers any remaining stochastic operations (e.g. sampling in a search).

    Args:
        seed: The global seed from config.
    """
    random.seed(seed)
    np.random.seed(seed)


def build_pipeline(features, model_spec: ModelSpec, seed: int) -> Pipeline:
    """Compose an unfitted preprocessing+model Pipeline for one model spec.

    Building preprocessing and estimator into a single Pipeline is what makes leakage-free CV
    possible later: the whole thing is fitted as a unit, so preprocessing is re-fit on each
    training fold rather than on data that includes the validation fold.

    Args:
        features: Training feature frame (used to auto-detect column types).
        model_spec: The model name and base parameters.
        seed: Global seed injected into the estimator.

    Returns:
        An unfitted ``Pipeline`` with steps ``preprocess`` and ``model``.
    """
    preprocessor = build_preprocessor(features)
    estimator = build_model(model_spec.name, model_spec.params, seed)
    return Pipeline(steps=[(_PREPROCESS_STEP, preprocessor), (_MODEL_STEP, estimator)])


def _train_on_full_train(
    model_spec: ModelSpec, x_train, y_train, config: Config
) -> tuple[Pipeline, dict[str, Any] | None, float | None]:
    """Fit a single model on the full training split (no tuning, no CV).

    This is the Stage-4 core training path. Stage 5 replaces this call with the tuning module,
    which returns the same ``(fitted_pipeline, best_params, best_cv_score)`` contract so the
    orchestrator does not need to change. It only ever sees training data — the test set is never
    passed in here.
    """
    pipeline = build_pipeline(x_train, model_spec, config.seed)
    pipeline.fit(x_train, y_train)
    return pipeline, None, None


def run_pipeline(config: Config) -> PipelineResult:
    """Run the full training pipeline for every configured model.

    Args:
        config: The validated pipeline configuration.

    Returns:
        A :class:`PipelineResult` aggregating each model's metrics and artifact location.
    """
    set_global_seeds(config.seed)

    features, target, data_summary = load_data(config)
    data_hash = hash_file(Path(config.data.path))

    stratify = target if config.split.stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=config.split.test_size,
        random_state=config.seed,
        stratify=stratify,
    )
    logger.info(
        "Split: %d train rows, %d test rows (stratify=%s).",
        len(x_train),
        len(x_test),
        config.split.stratify,
    )

    result = PipelineResult(
        data_summary=data_summary,
        primary_metric=config.evaluation.primary_metric,
        train_rows=int(len(x_train)),
        test_rows=int(len(x_test)),
    )

    for model_spec in config.models:
        logger.info("=== Training model: %s ===", model_spec.name)
        fitted_pipeline, best_params, best_cv_score = _train_on_full_train(
            model_spec, x_train, y_train, config
        )

        # The one and only time the held-out test set is used for this model.
        metrics = evaluate_classifier(fitted_pipeline, x_test, y_test)

        metadata = build_metadata(
            model_name=model_spec.name,
            config=config,
            metrics=metrics,
            data_path=Path(config.data.path),
            data_hash=data_hash,
            train_rows=result.train_rows,
            test_rows=result.test_rows,
            best_params=best_params,
            best_cv_score=best_cv_score,
        )
        artifact_dir = save_artifact(config.artifacts.root, fitted_pipeline, metadata)

        result.results.append(
            ModelResult(
                model_name=model_spec.name,
                metrics=metrics,
                artifact_dir=artifact_dir,
                best_params=best_params,
                best_cv_score=best_cv_score,
            )
        )

    return result


def format_run_summary(result: PipelineResult) -> str:
    """Render a concise, human-readable run summary.

    Returned as a string (not printed) so the library never writes to stdout; the CLI is
    responsible for printing it. This keeps logging/printing concerns at the application edge.

    Args:
        result: The completed pipeline result.

    Returns:
        A multi-line summary string ranking models by the primary metric.
    """
    metric = result.primary_metric
    ranked = sorted(
        result.results, key=lambda r: r.metrics.get(metric, float("-inf")), reverse=True
    )

    lines = ["", "Run summary", "-----------"]
    if result.data_summary is not None:
        lines.append(
            f"Data: {result.data_summary.n_rows} rows, "
            f"{result.data_summary.n_features} features, "
            f"positive rate {result.data_summary.positive_rate:.3f}"
        )
    lines.append(f"Split: {result.train_rows} train / {result.test_rows} test")
    lines.append(f"Models ranked by {metric}:")
    for rank, model_result in enumerate(ranked, start=1):
        scores = " ".join(f"{name}={value:.4f}" for name, value in model_result.metrics.items())
        lines.append(f"  {rank}. {model_result.model_name:20s} {scores}")
    if ranked:
        lines.append(f"Winner: {ranked[0].model_name} ({metric}={ranked[0].metrics[metric]:.4f})")
    return "\n".join(lines)
