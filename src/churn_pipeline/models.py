"""Model factory.

A single place that maps a configuration model name + parameter dict to an unfitted scikit-learn
estimator. Centralising construction here means the orchestrator never imports concrete estimator
classes, adding a new model type is a one-line change, and every model is constructed the same way
— crucially, each receives the global ``seed`` so runs are reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ModelError(Exception):
    """Raised for an unknown model name or invalid model parameters."""


# Supported estimators. All three accept ``random_state`` and expose ``predict_proba``, which the
# evaluation layer requires for roc_auc. Extend this mapping to add a model type.
_MODEL_BUILDERS: dict[str, type[ClassifierMixin]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}


def available_models() -> list[str]:
    """Return the sorted list of supported model names."""
    return sorted(_MODEL_BUILDERS)


def build_model(name: str, params: dict[str, Any], seed: int) -> ClassifierMixin:
    """Build an unfitted estimator from a config name and parameters.

    The global ``seed`` is injected as ``random_state`` so results are reproducible without the
    seed being hardcoded inside any estimator. Invalid parameter names surface as a clear
    :class:`ModelError` rather than a raw ``TypeError`` deep in scikit-learn.

    Args:
        name: Model key, one of :func:`available_models`.
        params: Hyperparameters from config, passed straight to the estimator constructor.
        seed: Global random seed applied as ``random_state``.

    Returns:
        An unfitted scikit-learn estimator.

    Raises:
        ModelError: If ``name`` is unknown or ``params`` contains an invalid argument.
    """
    if name not in _MODEL_BUILDERS:
        raise ModelError(f"Unknown model '{name}'. Available: {available_models()}")

    estimator_cls = _MODEL_BUILDERS[name]
    try:
        estimator = estimator_cls(random_state=seed, **params)
    except TypeError as exc:
        raise ModelError(f"Invalid parameters for model '{name}': {exc}") from exc

    logger.info("Built model '%s' with params %s (seed=%d).", name, params, seed)
    return estimator
