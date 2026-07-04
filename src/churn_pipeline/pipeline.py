"""Pipeline orchestration.

This is the only module that knows the end-to-end order of operations. It depends on the
components (data, preprocessing, models, tuning, evaluate, registry, tracking) and never the other
way round, so the dependency graph stays acyclic and each component remains independently testable.

Flow: set seeds -> load & validate data -> stratified train/test split -> for each model, train on
the training split via the configured strategy (tuning / CV / plain) and evaluate once on the
held-out test split -> register a versioned artifact -> optionally log to MLflow. After all models,
write a comparison summary naming the winner. The single test-set evaluation lives here (not inside
the training strategies) so it is easy to see the test set is touched exactly once per model, on
every code path.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from . import tracking
from .config import Config
from .data import DataSummary, load_data
from .evaluate import evaluate_classifier
from .registry import (
    build_metadata,
    git_commit_hash,
    hash_config,
    hash_file,
    save_artifact,
    save_run_summary,
)
from .tuning import train_model

logger = logging.getLogger(__name__)


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
        winner: Name of the best model by the primary metric.
        summary_path: Path to the written run comparison summary.
    """

    results: list[ModelResult] = field(default_factory=list)
    data_summary: DataSummary | None = None
    primary_metric: str = "roc_auc"
    train_rows: int = 0
    test_rows: int = 0
    winner: str | None = None
    summary_path: Path | None = None


def set_global_seeds(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility.

    Per-model determinism is handled by injecting ``random_state`` in the model factory; seeding
    the global RNGs here covers any remaining stochastic operations (e.g. sampling in a search).

    Args:
        seed: The global seed from config.
    """
    random.seed(seed)
    np.random.seed(seed)


def rank_results(results: list[ModelResult], metric: str) -> list[ModelResult]:
    """Rank model results by a metric, highest first.

    Args:
        results: Model results to rank.
        metric: Metric name to rank by (missing metrics sort last).

    Returns:
        A new list ordered best-to-worst by ``metric``.
    """
    return sorted(results, key=lambda r: r.metrics.get(metric, float("-inf")), reverse=True)


def run_pipeline(config: Config) -> PipelineResult:
    """Run the full training pipeline for every configured model.

    Args:
        config: The validated pipeline configuration.

    Returns:
        A :class:`PipelineResult` aggregating each model's metrics, artifact location, and the
        overall winner.
    """
    run_started = datetime.now(UTC)
    run_version = run_started.strftime("%Y%m%dT%H%M%SZ")
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
        outcome = train_model(model_spec, x_train, y_train, config)

        # The one and only time the held-out test set is used for this model.
        metrics = evaluate_classifier(outcome.fitted_pipeline, x_test, y_test)

        metadata = build_metadata(
            model_name=model_spec.name,
            config=config,
            metrics=metrics,
            data_path=Path(config.data.path),
            data_hash=data_hash,
            train_rows=result.train_rows,
            test_rows=result.test_rows,
            best_params=outcome.best_params,
            best_cv_score=outcome.best_cv_score,
        )
        artifact_dir = save_artifact(config.artifacts.root, outcome.fitted_pipeline, metadata)

        if config.tracking.mlflow:
            tracking.log_model_run(
                tracking_uri=config.tracking.tracking_uri,
                experiment_name=config.tracking.experiment_name,
                model_name=model_spec.name,
                params=outcome.best_params or model_spec.params,
                metrics=metrics,
                fitted_pipeline=outcome.fitted_pipeline,
                best_cv_score=outcome.best_cv_score,
            )

        result.results.append(
            ModelResult(
                model_name=model_spec.name,
                metrics=metrics,
                artifact_dir=artifact_dir,
                best_params=outcome.best_params,
                best_cv_score=outcome.best_cv_score,
            )
        )

    summary = _build_comparison(result, config, run_version, run_started, data_hash)
    result.winner = summary["winner"]
    result.summary_path = save_run_summary(config.artifacts.root, summary)
    return result


def _build_comparison(
    result: PipelineResult,
    config: Config,
    run_version: str,
    run_started: datetime,
    data_hash: str,
) -> dict[str, Any]:
    """Build the cross-model comparison summary dict for a run.

    Ranks models by the primary metric and records the winner alongside provenance hashes so the
    summary alone answers "which model won this run, and against what data/config?".
    """
    ranked = rank_results(result.results, result.primary_metric)
    return {
        "version": run_version,
        "created_at": run_started.isoformat(),
        "primary_metric": result.primary_metric,
        "winner": ranked[0].model_name if ranked else None,
        "config_hash": hash_config(config),
        "data_hash": data_hash,
        "git_commit": git_commit_hash(),
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
        "models": [
            {
                "rank": rank,
                "model_name": model_result.model_name,
                "metrics": model_result.metrics,
                "best_params": model_result.best_params,
                "best_cv_score": model_result.best_cv_score,
                "artifact_dir": str(model_result.artifact_dir),
            }
            for rank, model_result in enumerate(ranked, start=1)
        ],
    }


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
    ranked = rank_results(result.results, metric)

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
        if model_result.best_params is not None:
            lines.append(
                f"       tuned: {model_result.best_params} "
                f"(cv={model_result.best_cv_score:.4f})"
            )
    if ranked:
        lines.append(f"Winner: {ranked[0].model_name} ({metric}={ranked[0].metrics[metric]:.4f})")
    if result.summary_path is not None:
        lines.append(f"Comparison summary: {result.summary_path}")
    return "\n".join(lines)
