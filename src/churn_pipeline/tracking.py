"""Optional MLflow experiment tracking.

Kept as a separate module (and beyond the initial component list) on purpose: it is the only place
that imports MLflow, and it does so *lazily* inside the function. That means the package imports and
the pipeline runs with zero MLflow overhead when tracking is disabled, and a broken/absent MLflow
install can never break a core training run. This mirrors how an optional integration would be
isolated in production so the critical path has no incidental dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def log_model_run(
    *,
    tracking_uri: str,
    experiment_name: str,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    fitted_pipeline: Pipeline,
    best_cv_score: float | None = None,
) -> None:
    """Log one model's params, metrics, and serialized pipeline to MLflow.

    Called once per model only when ``tracking.mlflow`` is true. Any MLflow failure is caught and
    logged rather than raised, because experiment tracking is an observability concern that must
    never fail the training run itself.

    Args:
        tracking_uri: MLflow backend store URI (e.g. ``sqlite:///mlflow.db``).
        experiment_name: MLflow experiment to log under.
        model_name: Name used as the run name and logged model artifact path.
        params: Hyperparameters to log (tuned best params, or the model's base params).
        metrics: Held-out test metrics to log.
        fitted_pipeline: The fitted pipeline to log as a model artifact.
        best_cv_score: Best cross-validation score to log, when tuning ran.
    """
    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            if best_cv_score is not None:
                mlflow.log_metric("best_cv_score", best_cv_score)
            # cloudpickle format keeps the sklearn model flavor (pyfunc-servable) and avoids the
            # skops safe-loading validation that rejects numpy.dtype on MLflow 3.x.
            mlflow.sklearn.log_model(
                fitted_pipeline, name=model_name, serialization_format="cloudpickle"
            )
        logger.info(
            "Logged MLflow run for '%s' under experiment '%s'.", model_name, experiment_name
        )
    except Exception:  # noqa: BLE001 - tracking must never break training
        logger.exception("MLflow logging failed for '%s'; continuing without tracking.", model_name)
