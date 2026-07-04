"""Tests for the data ingestion and validation layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from churn_pipeline.config import load_config
from churn_pipeline.data import DataValidationError, load_data


def test_total_charges_coerced_to_numeric(config_path: Path) -> None:
    """TotalCharges is coerced to a numeric dtype with no residual NaNs, and blanks are counted."""
    config = load_config(config_path)
    features, _, summary = load_data(config)
    assert features["TotalCharges"].dtype.kind == "f"
    assert features["TotalCharges"].isna().sum() == 0
    # The fixture forces 5 tenure=0 accounts, so at least that many blanks were filled.
    assert summary.coerced_columns["TotalCharges"] >= 5


def test_id_column_dropped(config_path: Path) -> None:
    """The configured id column is removed from the feature matrix."""
    config = load_config(config_path)
    features, _, summary = load_data(config)
    assert "customerID" not in features.columns
    assert summary.dropped_columns == ["customerID"]


def test_target_encoded_to_binary(config_path: Path) -> None:
    """The Yes/No target maps to integer 1/0 with both classes present."""
    config = load_config(config_path)
    _, target, summary = load_data(config)
    assert set(target.unique()) == {0, 1}
    assert target.dtype.kind == "i"
    assert 0.0 < summary.positive_rate < 1.0


def test_missing_target_raises(config_path: Path) -> None:
    """A configured target that is absent from the data raises DataValidationError."""
    config = load_config(config_path)
    config.data.target = "NotAColumn"
    with pytest.raises(DataValidationError, match="Target column"):
        load_data(config)


def test_missing_file_raises(config_path: Path) -> None:
    """A missing data file raises FileNotFoundError from the data layer."""
    config = load_config(config_path)
    config.data.path = Path("no_such_file.csv")
    with pytest.raises(FileNotFoundError):
        load_data(config)
