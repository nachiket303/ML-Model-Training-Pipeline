"""Tests for the typed configuration layer."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from churn_pipeline.config import ConfigError, load_config


def test_valid_config_loads(config_path: Path) -> None:
    """A well-formed config parses into a typed Config with expected values."""
    config = load_config(config_path)
    assert config.seed == 42
    assert config.data.target == "Churn"
    assert config.data.id_columns == ["customerID"]
    assert [model.name for model in config.models] == [
        "logistic_regression",
        "random_forest",
    ]


def test_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent config path raises ConfigError, not a bare OSError."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_invalid_field_raises(config_dict: dict, tmp_path: Path) -> None:
    """An out-of-range field (test_size >= 1) fails validation as ConfigError."""
    config_dict["split"]["test_size"] = 1.5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_unknown_key_raises(config_dict: dict, tmp_path: Path) -> None:
    """An unexpected key is rejected (extra='forbid') so typos never pass silently."""
    config_dict["data"]["typo_field"] = "oops"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_tuning_without_grid_raises(config_dict: dict, tmp_path: Path) -> None:
    """Enabling tuning without a grid for every model fails fast (no silent no-op tuning)."""
    config_dict["tuning"]["enabled"] = True
    config_dict["tuning"]["param_grids"].pop("random_forest")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config_dict), encoding="utf-8")
    with pytest.raises(ConfigError, match="param_grids"):
        load_config(path)
