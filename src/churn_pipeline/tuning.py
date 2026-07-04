"""Training strategies: tuning and cross-validation.

This module contains the three mutually exclusive ways a single model is trained. The branch is
chosen by config and is deliberately explicit and flat, because "how did we train and where could
leakage sneak in?" is the first question an interviewer (or reviewer) asks.

The three paths, all of which touch only the training split:

1. ``tuning.enabled``                      -> Grid/RandomizedSearchCV over the full Pipeline.
2. ``cross_validation.enabled`` (no tuning) -> plain stratified k-fold CV, then fit on full train.
3. neither                                  -> a single fit on the full training split.

No-leakage guarantee: in paths (1) and (2) the estimator passed to the search / cross-validator is
the *entire* preprocessing+model Pipeline. scikit-learn therefore re-fits the imputers, scaler, and
one-hot encoder on each fold's training portion only — validation-fold statistics never leak into
fitting. The held-out test set is never passed to this module; the orchestrator evaluates the
returned estimator on it exactly once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod
from typing import Any

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from .config import Config, ModelSpec
from .evaluate import METRIC_NAMES
from .models import build_model
from .preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

# The estimator step is named "model" so tuning grids can address params as ``model__<param>``.
PREPROCESS_STEP = "preprocess"
MODEL_STEP = "model"


@dataclass
class TrainOutcome:
    """Result of training one model on the training split (before test evaluation).

    Attributes:
        fitted_pipeline: The pipeline fitted on the full training split, ready for test evaluation.
        best_params: Chosen hyperparameters when tuning ran; ``None`` otherwise.
        best_cv_score: Best cross-validation score when tuning ran; ``None`` otherwise.
    """

    fitted_pipeline: Pipeline
    best_params: dict[str, Any] | None = None
    best_cv_score: float | None = None


def build_pipeline(features: pd.DataFrame, model_spec: ModelSpec, seed: int) -> Pipeline:
    """Compose an unfitted preprocessing+model Pipeline for one model spec.

    Composing preprocessing and estimator into a single Pipeline is precisely what makes
    leakage-free CV possible: the unit is fitted together, so preprocessing is re-fit per fold.

    Args:
        features: Training feature frame (used to auto-detect column types).
        model_spec: The model name and base parameters.
        seed: Global seed injected into the estimator.

    Returns:
        An unfitted ``Pipeline`` with steps ``preprocess`` and ``model``.
    """
    preprocessor = build_preprocessor(features)
    estimator = build_model(model_spec.name, model_spec.params, seed)
    return Pipeline(steps=[(PREPROCESS_STEP, preprocessor), (MODEL_STEP, estimator)])


def _make_cv(folds: int, seed: int) -> StratifiedKFold:
    """Create a seeded, shuffled stratified k-fold splitter shared by CV and search paths."""
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def train_model(
    model_spec: ModelSpec, x_train: pd.DataFrame, y_train: pd.Series, config: Config
) -> TrainOutcome:
    """Train one model using the configured strategy, on training data only.

    Args:
        model_spec: The model to train.
        x_train: Training features.
        y_train: Training labels.
        config: The validated configuration selecting the training strategy.

    Returns:
        A :class:`TrainOutcome` whose ``fitted_pipeline`` is fitted on the full training split.
    """
    if config.tuning.enabled:
        return _train_with_search(model_spec, x_train, y_train, config)
    if config.cross_validation.enabled:
        return _train_with_cv(model_spec, x_train, y_train, config)
    return _train_plain(model_spec, x_train, y_train, config)


def _train_plain(
    model_spec: ModelSpec, x_train: pd.DataFrame, y_train: pd.Series, config: Config
) -> TrainOutcome:
    """Path 3: a single fit on the full training split (no tuning, no CV)."""
    pipeline = build_pipeline(x_train, model_spec, config.seed)
    pipeline.fit(x_train, y_train)
    logger.info("Trained '%s' with a single fit (no tuning, no CV).", model_spec.name)
    return TrainOutcome(fitted_pipeline=pipeline)


def _train_with_cv(
    model_spec: ModelSpec, x_train: pd.DataFrame, y_train: pd.Series, config: Config
) -> TrainOutcome:
    """Path 2: plain stratified k-fold CV for reporting, then fit once on the full training split.

    CV here is diagnostic — it estimates generalisation with mean±std of each metric — and does not
    change the final model, which is refit on all training data before the single test evaluation.
    """
    pipeline = build_pipeline(x_train, model_spec, config.seed)
    cv = _make_cv(config.cross_validation.folds, config.seed)
    cv_results = cross_validate(
        pipeline,
        x_train,
        y_train,
        cv=cv,
        scoring=list(METRIC_NAMES),
        n_jobs=None,
    )
    for metric in METRIC_NAMES:
        scores = cv_results[f"test_{metric}"]
        logger.info(
            "CV[%s] %s: %.4f +/- %.4f", model_spec.name, metric, scores.mean(), scores.std()
        )

    pipeline.fit(x_train, y_train)
    return TrainOutcome(fitted_pipeline=pipeline)


def _grid_combinations(grid: dict[str, list[Any]]) -> int:
    """Return the number of distinct configurations in a discrete parameter grid."""
    return prod(len(values) for values in grid.values()) if grid else 1


def _train_with_search(
    model_spec: ModelSpec, x_train: pd.DataFrame, y_train: pd.Series, config: Config
) -> TrainOutcome:
    """Path 1: hyperparameter search with CV nested over the full Pipeline (no leakage).

    The search fits and scores the whole preprocessing+model Pipeline on each fold, so
    preprocessing is re-fit per fold. With ``refit=True`` the returned ``best_estimator_`` is
    retrained on the full training split using the winning params; the orchestrator then evaluates
    it once on the test set.
    """
    base_pipeline = build_pipeline(x_train, model_spec, config.seed)
    grid = config.tuning.param_grids[model_spec.name]
    cv = _make_cv(config.cross_validation.folds, config.seed)

    if config.tuning.method == "grid":
        search = GridSearchCV(
            base_pipeline,
            param_grid=grid,
            scoring=config.tuning.scoring,
            cv=cv,
            refit=True,
            n_jobs=None,
        )
    else:
        # Cap n_iter at the grid size so a small grid cannot exceed the discrete search space.
        n_iter = min(config.tuning.n_iter, _grid_combinations(grid))
        search = RandomizedSearchCV(
            base_pipeline,
            param_distributions=grid,
            n_iter=n_iter,
            scoring=config.tuning.scoring,
            cv=cv,
            refit=True,
            random_state=config.seed,
            n_jobs=None,
        )

    search.fit(x_train, y_train)  # training data only
    logger.info(
        "Search[%s] best %s=%.4f with params %s",
        model_spec.name,
        config.tuning.scoring,
        search.best_score_,
        search.best_params_,
    )
    return TrainOutcome(
        fitted_pipeline=search.best_estimator_,
        best_params=dict(search.best_params_),
        best_cv_score=float(search.best_score_),
    )
