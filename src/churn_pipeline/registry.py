"""Versioned artifact registry.

Every trained model is persisted to ``artifacts/<model_name>/<version>/`` as two files:

* ``model.joblib``   — the *fitted* preprocessing+model Pipeline, so inference needs no retraining.
* ``metadata.json``  — everything required to understand and reproduce the run.

The metadata is the heart of the reproducibility story. It records the resolved config plus a hash
of it, a hash of the exact input data file, the library and Python versions, the git commit, the
metrics, and (when tuning ran) the chosen best params and CV score. Given the same config hash,
data hash, seed, and pinned dependencies, a run reproduces. This local directory layout is a
deliberate stand-in for a Vertex AI Model Registry entry — same idea (immutable, versioned,
metadata-rich artifacts), different backend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from .config import Config

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "model.joblib"
_METADATA_FILENAME = "metadata.json"
_RUNS_DIRNAME = "runs"
_HASH_CHUNK_BYTES = 1 << 20  # read files in 1 MiB chunks when hashing


class ArtifactError(Exception):
    """Raised when an artifact cannot be found or loaded."""


def library_versions() -> dict[str, str]:
    """Return versions of the libraries that affect model reproducibility.

    Exposed publicly so the CLI can log the same versions at startup that are written into
    metadata, keeping logs and artifacts consistent.
    """
    return {
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks to bound memory use.

    Args:
        path: File to hash (the input dataset).

    Returns:
        Hex-encoded SHA-256 digest, prefixed ``sha256:``.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def hash_config(config: Config) -> str:
    """Return a stable SHA-256 digest of the resolved configuration.

    Uses canonical JSON (sorted keys) so logically identical configs hash identically regardless
    of key ordering.

    Args:
        config: The validated configuration.

    Returns:
        Hex-encoded SHA-256 digest, prefixed ``sha256:``.
    """
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def git_commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable.

    Best-effort: the pipeline must still run outside a git checkout, so any failure is swallowed
    and reported as ``None`` rather than raising.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def _json_default(value: Any) -> Any:
    """Fallback JSON encoder for numpy scalars and Path objects in metadata."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_metadata(
    *,
    model_name: str,
    config: Config,
    metrics: dict[str, float],
    data_path: Path,
    data_hash: str,
    train_rows: int,
    test_rows: int,
    best_params: dict[str, Any] | None = None,
    best_cv_score: float | None = None,
) -> dict[str, Any]:
    """Assemble the reproducibility metadata for a single trained model.

    Kept separate from :func:`save_artifact` so the exact contents can be unit-tested without
    touching the filesystem. ``data_hash`` is passed in (computed once per run) to avoid
    re-hashing the input file for every model.

    Args:
        model_name: Name of the model, used as the artifact namespace.
        config: The resolved configuration (embedded and hashed).
        metrics: Held-out test metrics for this model.
        data_path: Path to the input dataset (recorded for provenance).
        data_hash: Precomputed hash of the input dataset.
        train_rows: Number of training rows.
        test_rows: Number of held-out test rows.
        best_params: Tuned best parameters, when tuning ran; otherwise ``None``.
        best_cv_score: Best cross-validation score, when tuning ran; otherwise ``None``.

    Returns:
        A JSON-serialisable metadata dictionary including a filesystem-safe ``version`` string.
    """
    now = datetime.now(UTC)
    return {
        "version": now.strftime("%Y%m%dT%H%M%SZ"),
        "created_at": now.isoformat(),
        "model_name": model_name,
        "seed": config.seed,
        "metrics": metrics,
        "best_params": best_params,
        "best_cv_score": best_cv_score,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "data_path": str(data_path),
        "data_hash": data_hash,
        "config_hash": hash_config(config),
        "config": config.model_dump(mode="json"),
        "git_commit": git_commit_hash(),
        "python_version": platform.python_version(),
        "library_versions": library_versions(),
    }


def save_artifact(
    artifacts_root: Path, fitted_pipeline: Pipeline, metadata: dict[str, Any]
) -> Path:
    """Persist a fitted pipeline and its metadata to a versioned directory.

    Args:
        artifacts_root: Root directory for all artifacts (from ``config.artifacts.root``).
        fitted_pipeline: The fitted preprocessing+model pipeline to serialise.
        metadata: Metadata dict from :func:`build_metadata` (must contain ``model_name`` and
            ``version``).

    Returns:
        The directory the artifact was written to.
    """
    version_dir = Path(artifacts_root) / metadata["model_name"] / metadata["version"]
    version_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(fitted_pipeline, version_dir / _MODEL_FILENAME)
    (version_dir / _METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8"
    )
    logger.info("Saved artifact for '%s' -> %s", metadata["model_name"], version_dir)
    return version_dir


def save_run_summary(artifacts_root: Path, summary: dict[str, Any]) -> Path:
    """Persist a cross-model comparison summary for one pipeline run.

    Written to ``artifacts/runs/<version>_summary.json`` so a run's model ranking and winner are
    recorded alongside (but separate from) the per-model artifacts.

    Args:
        artifacts_root: Root artifacts directory.
        summary: Comparison summary dict (must contain a ``version`` key).

    Returns:
        The path the summary was written to.
    """
    runs_dir = Path(artifacts_root) / _RUNS_DIRNAME
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = runs_dir / f"{summary['version']}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    logger.info("Saved run comparison summary -> %s", summary_path)
    return summary_path


def list_versions(artifacts_root: Path, model_name: str) -> list[str]:
    """List available version identifiers for a model, oldest first.

    Args:
        artifacts_root: Root artifacts directory.
        model_name: Model namespace to list.

    Returns:
        Sorted list of version directory names (empty if the model has no artifacts).
    """
    model_dir = Path(artifacts_root) / model_name
    if not model_dir.is_dir():
        return []
    return sorted(child.name for child in model_dir.iterdir() if child.is_dir())


def load_artifact(
    artifacts_root: Path, model_name: str, version: str | None = None
) -> tuple[Pipeline, dict[str, Any]]:
    """Load a fitted pipeline and its metadata by model name and version.

    Args:
        artifacts_root: Root artifacts directory.
        model_name: Model namespace to load.
        version: Specific version; the latest version is loaded when ``None``.

    Returns:
        A tuple ``(fitted_pipeline, metadata)``.

    Raises:
        ArtifactError: If the model has no artifacts or the requested version is missing.
    """
    if version is None:
        versions = list_versions(artifacts_root, model_name)
        if not versions:
            raise ArtifactError(f"No artifacts found for model '{model_name}'.")
        version = versions[-1]

    version_dir = Path(artifacts_root) / model_name / version
    model_path = version_dir / _MODEL_FILENAME
    metadata_path = version_dir / _METADATA_FILENAME
    if not model_path.is_file() or not metadata_path.is_file():
        raise ArtifactError(f"Incomplete or missing artifact at {version_dir}.")

    fitted_pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return fitted_pipeline, metadata
