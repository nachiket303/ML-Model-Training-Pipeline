"""Typed configuration loading and validation.

The whole pipeline is config-driven, so config is a boundary that must be validated hard and
early. We parse ``config/default.yaml`` into pydantic models rather than passing raw dicts
around: this gives the rest of the codebase typed attributes, editor autocomplete, and a single
place that fails loudly (with the offending file path) on a missing or malformed field. Every
model uses ``extra="forbid"`` so a typo in the YAML raises instead of being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class ConfigError(Exception):
    """Raised when a configuration file is missing, unparseable, or fails validation."""


class DataConfig(BaseModel):
    """Where the data lives and how to interpret its target/identifier columns."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    target: str
    id_columns: list[str] = Field(default_factory=list)
    positive_label: str
    # Columns stored as strings but semantically numeric (blanks/whitespace -> NaN). They are
    # coerced with errors="coerce" and the resulting NaNs filled with `numeric_fill`. Kept in
    # config (not hardcoded) so the tool stays generic across datasets; empty by default.
    numeric_coerce_columns: list[str] = Field(default_factory=list)
    numeric_fill: float = 0.0


class SplitConfig(BaseModel):
    """Train/test split settings."""

    model_config = ConfigDict(extra="forbid")

    test_size: float = Field(gt=0.0, lt=1.0)
    stratify: bool = True


class CrossValidationConfig(BaseModel):
    """Cross-validation settings used when tuning is off but CV is requested."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    folds: int = Field(default=5, ge=2)


class TuningConfig(BaseModel):
    """Hyperparameter search settings and per-model parameter grids."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: Literal["grid", "random"] = "grid"
    scoring: str = "roc_auc"
    n_iter: int = Field(default=10, ge=1)
    # {model_name: {pipeline_param: [candidate values]}}. Values may include null (-> None).
    param_grids: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)


class ModelSpec(BaseModel):
    """A single model to train: its factory name and base hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation settings, notably the metric used to rank models."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = "roc_auc"


class TrackingConfig(BaseModel):
    """Experiment tracking settings (MLflow is optional and cleanly separable)."""

    model_config = ConfigDict(extra="forbid")

    mlflow: bool = False
    experiment_name: str = "telco-churn"
    # Explicit MLflow backend so runs are reproducible and viewable via
    # `mlflow ui --backend-store-uri <this>`. A DB store is used because MLflow's file store is
    # deprecated. Kept in config so nothing points at a hardcoded path in logic.
    tracking_uri: str = "sqlite:///mlflow.db"


class ArtifactsConfig(BaseModel):
    """Where versioned model artifacts and run summaries are written."""

    model_config = ConfigDict(extra="forbid")

    root: Path = Path("artifacts")


class Config(BaseModel):
    """Fully validated pipeline configuration (the typed view of ``default.yaml``)."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    data: DataConfig
    split: SplitConfig
    cross_validation: CrossValidationConfig
    tuning: TuningConfig
    models: list[ModelSpec] = Field(min_length=1)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig
    artifacts: ArtifactsConfig

    @model_validator(mode="after")
    def _require_grids_when_tuning(self) -> Config:
        """Fail fast if tuning is enabled but a model has no search grid.

        Without this check a mis-configured run would "tune" a model over an empty grid and
        silently behave like a single fit — exactly the kind of quiet no-op we want to reject.
        """
        if self.tuning.enabled:
            missing = [m.name for m in self.models if m.name not in self.tuning.param_grids]
            if missing:
                raise ValueError(
                    f"tuning.enabled is true but tuning.param_grids has no entry for: {missing}"
                )
        return self


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file into a typed :class:`Config`.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        The validated configuration object.

    Raises:
        ConfigError: If the file is missing, is not valid YAML, does not contain a mapping at
            its root, or fails schema validation. The original error is chained for context.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config '{path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__} in '{path}'.")

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration in '{path}':\n{exc}") from exc
