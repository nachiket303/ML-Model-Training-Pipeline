"""Feature preprocessing.

Builds a scikit-learn ``ColumnTransformer`` that turns the mixed-type feature frame into a
purely numeric matrix. Two branches:

* numeric  -> median imputation (robust to outliers/skew) + standard scaling (so the scale of
  ``TotalCharges`` does not dominate distance/gradient-based models such as logistic regression).
* categorical -> most-frequent imputation + one-hot encoding with ``handle_unknown="ignore"`` so a
  category unseen at training time becomes an all-zero vector instead of crashing at inference.

IMPORTANT — leakage boundary: this function returns an *unfitted* transformer. It is composed
with an estimator into a single ``Pipeline`` and fitted on the training split only; the fitted
statistics (medians, category vocabulary, scaler mean/std) are then reused unchanged on the test
split and at inference. Fitting preprocessing on all data before splitting would leak test
information into training, so we never do that.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def detect_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical by pandas dtype.

    Numeric columns are anything pandas considers a number; everything else (object, category,
    bool) is treated as categorical. This dtype-based rule is deliberately simple and predictable;
    callers that need finer control can pass explicit column lists to :func:`build_preprocessor`.

    Args:
        features: The feature dataframe.

    Returns:
        A ``(numeric_columns, categorical_columns)`` tuple whose union is all columns.
    """
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def build_preprocessor(
    features: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for numeric and categorical features.

    Args:
        features: Training feature frame, used to auto-detect column types when explicit lists
            are not supplied.
        numeric_features: Optional explicit numeric column names; auto-detected when ``None``.
        categorical_features: Optional explicit categorical column names; auto-detected when
            ``None``.

    Returns:
        An unfitted ``ColumnTransformer``. It must be fitted on the training split only (typically
        by wrapping it in a ``Pipeline`` with an estimator) and then reused on test/inference data.
    """
    if numeric_features is None or categorical_features is None:
        numeric_features, categorical_features = detect_feature_types(features)

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # sparse_output=False keeps the output dense so downstream estimators and metadata
            # inspection stay simple; the one-hot width here is small.
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    logger.info(
        "Preprocessor: %d numeric, %d categorical feature(s).",
        len(numeric_features),
        len(categorical_features),
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
