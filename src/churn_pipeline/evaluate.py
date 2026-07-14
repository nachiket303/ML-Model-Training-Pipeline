"""Evaluation metrics.

Churn is a mildly imbalanced problem (~26.5% positive), so accuracy alone is misleading a model
predicting "never churns" would score ~73%. We therefore report precision, recall, f1, and roc_auc
alongside accuracy. roc_auc is computed from predicted probabilities (not hard labels), because it
measures ranking quality across all thresholds; the estimator is asserted to support
``predict_proba`` so this can never silently fall back to labels.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# Canonical metric names, defined once so config, comparison, and logging never drift apart.
METRIC_NAMES: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")


class EvaluationError(Exception):
    """Raised when an estimator cannot be evaluated (e.g. it lacks predict_proba)."""


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """Structural type for the estimator interface the evaluator needs.

    A fitted ``Pipeline`` ending in a classifier satisfies this without being a nominal subclass,
    which is exactly why a Protocol is used instead of a concrete base class.
    """

    def predict(self, features: ArrayLike) -> np.ndarray: ...

    def predict_proba(self, features: ArrayLike) -> np.ndarray: ...


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike, y_proba: ArrayLike) -> dict[str, float]:
    """Compute the standard classification metrics from labels and positive-class probabilities.

    Pure function (no model, no I/O) so it is trivial to unit test. ``zero_division=0`` keeps
    precision/recall well-defined on degenerate splits (e.g. a tiny test fold with no predicted
    positives) instead of emitting warnings.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probability of the positive class, used only for roc_auc.

    Returns:
        Mapping of metric name to value, keyed by :data:`METRIC_NAMES`.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def evaluate_classifier(
    estimator: ProbabilisticClassifier, features: ArrayLike, y_true: ArrayLike
) -> dict[str, float]:
    """Evaluate a fitted probabilistic classifier on a held-out set.

    Args:
        estimator: A fitted estimator/pipeline exposing ``predict`` and ``predict_proba``.
        features: Held-out feature matrix.
        y_true: Held-out ground-truth labels.

    Returns:
        Mapping of metric name to value (see :func:`compute_metrics`).

    Raises:
        EvaluationError: If the estimator does not expose ``predict_proba`` (roc_auc would be
            undefined).
    """
    if not hasattr(estimator, "predict_proba"):
        raise EvaluationError(
            f"Estimator {type(estimator).__name__} has no predict_proba; roc_auc requires it."
        )
    y_pred = estimator.predict(features)
    y_proba = estimator.predict_proba(features)[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_proba)
    logger.info("Evaluation: %s", {name: round(metrics[name], 4) for name in METRIC_NAMES})
    return metrics
