"""Data ingestion and validation layer.

This module owns the boundary between raw files and the modelling code. Everything downstream
assumes a clean feature matrix ``X`` and an integer-encoded target ``y``; this layer is the only
place allowed to know about file formats, identifier columns, and dataset-specific quirks. It
validates hard and fails fast, because a silent data problem is far more expensive than a loud
one.

Deliberate data-quality decision (Telco ``TotalCharges``): the column ships as strings with
blank/whitespace values for accounts whose ``tenure`` is 0. Those rows are genuinely new
customers that have accrued no charges yet, so a blank means *zero*, not *unknown*. We therefore
coerce the column to numeric and fill the resulting NaNs with the configured ``numeric_fill``
(0.0 for this dataset) rather than leaving them for the preprocessing median-imputer, which would
substitute a misleading central value. The median-imputer in the preprocessing layer remains as a
safety net for genuinely unexpected missing values seen at inference time. Which columns get this
treatment is driven by config (``data.numeric_coerce_columns``), so nothing dataset-specific is
hardcoded here and the same code path works on any dataset supplied via ``--data``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import Config

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when a dataset fails a boundary check (empty, missing columns, or bad target)."""


@dataclass(frozen=True)
class DataSummary:
    """Immutable summary of what the data layer produced, for logging and tests.

    Attributes:
        n_rows: Number of rows after loading.
        n_features: Number of feature columns after dropping id and target columns.
        positive_rate: Fraction of rows in the positive class (sanity check for imbalance).
        dropped_columns: Identifier columns removed before modelling.
        coerced_columns: Mapping of column name to how many blank/invalid values were filled.
    """

    n_rows: int
    n_features: int
    positive_rate: float
    dropped_columns: list[str]
    coerced_columns: dict[str, int]


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a dataframe from CSV or Parquet, chosen by file extension.

    Args:
        path: Path to a ``.csv`` or ``.parquet``/``.pq`` file.

    Returns:
        The loaded dataframe.

    Raises:
        DataValidationError: If the file extension is not a supported format.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise DataValidationError(
        f"Unsupported data format '{suffix}' for '{path}'. Supported: .csv, .parquet."
    )


def _coerce_numeric_columns(
    frame: pd.DataFrame, columns: list[str], fill_value: float
) -> dict[str, int]:
    """Coerce stringly-typed numeric columns to numbers and fill structural blanks.

    Columns configured but absent from the frame are skipped, so a config tuned for the Telco
    schema does not break when the tool is pointed at a different dataset via ``--data``.

    Args:
        frame: Dataframe to modify in place.
        columns: Column names to coerce (from ``data.numeric_coerce_columns``).
        fill_value: Value used to fill NaNs produced by coercion.

    Returns:
        Mapping of column name to the number of blank/invalid values that were filled.
    """
    coerced: dict[str, int] = {}
    for column in columns:
        if column not in frame.columns:
            logger.warning(
                "Configured numeric_coerce column '%s' not present in data; skipping.", column
            )
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        n_filled = int(numeric.isna().sum())
        frame[column] = numeric.fillna(fill_value)
        coerced[column] = n_filled
        if n_filled:
            logger.info(
                "Coerced '%s' to numeric; filled %d blank value(s) with %s.",
                column,
                n_filled,
                fill_value,
            )
    return coerced


def load_data(config: Config) -> tuple[pd.DataFrame, pd.Series, DataSummary]:
    """Load, validate, and clean the dataset into a feature matrix and encoded target.

    Steps, in order: read the file by extension; validate it is non-empty and contains the
    configured target and id columns; coerce configured numeric columns; drop id columns; encode
    the target to 1/0 using ``positive_label``. Each failure raises a specific, actionable error.

    Args:
        config: Validated pipeline configuration.

    Returns:
        A tuple ``(X, y, summary)`` where ``X`` is the feature dataframe, ``y`` is the integer
        target Series (1 = positive class), and ``summary`` describes what was produced.

    Raises:
        FileNotFoundError: If the data file does not exist.
        DataValidationError: If the dataset is empty, is an unsupported format, is missing the
            target or id columns, or the ``positive_label`` never appears in the target.
    """
    data_cfg = config.data
    path = Path(data_cfg.path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    frame = _read_frame(path)
    if frame.empty:
        raise DataValidationError(f"Dataset contains no rows: {path}")

    if data_cfg.target not in frame.columns:
        raise DataValidationError(
            f"Target column '{data_cfg.target}' not found. Available columns: {list(frame.columns)}"
        )
    missing_ids = [column for column in data_cfg.id_columns if column not in frame.columns]
    if missing_ids:
        raise DataValidationError(f"Configured id_columns not found in data: {missing_ids}")

    coerced = _coerce_numeric_columns(frame, data_cfg.numeric_coerce_columns, data_cfg.numeric_fill)

    dropped_columns = list(data_cfg.id_columns)
    frame = frame.drop(columns=dropped_columns)

    target_raw = frame.pop(data_cfg.target)
    positive = data_cfg.positive_label
    if not (target_raw.astype(str) == str(positive)).any():
        raise DataValidationError(
            f"positive_label '{positive}' never appears in target '{data_cfg.target}'. "
            f"Observed values: {sorted(target_raw.astype(str).unique())[:10]}"
        )
    target = (target_raw.astype(str) == str(positive)).astype(int)

    summary = DataSummary(
        n_rows=int(len(frame)),
        n_features=int(frame.shape[1]),
        positive_rate=float(target.mean()),
        dropped_columns=dropped_columns,
        coerced_columns=coerced,
    )
    logger.info(
        "Loaded %d rows, %d features (positive rate %.3f); dropped id columns %s.",
        summary.n_rows,
        summary.n_features,
        summary.positive_rate,
        dropped_columns,
    )
    return frame, target, summary
