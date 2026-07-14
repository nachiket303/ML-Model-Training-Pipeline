"""Command-line entry point.

The single command runs in the demo: ``churn-train --config config/default.yaml``.
It configures logging once, logs the environment for reproducibility, optionally overrides the
dataset path (so the exact same tool works on any dataset, not just Telco), runs the pipeline, and
prints the run summary. Expected boundary errors (bad config, bad data) are reported as a clean
one-line message with a non-zero exit code rather than a raw traceback.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from .config import ConfigError, load_config
from .data import DataValidationError
from .logging_config import configure_logging, log_environment
from .models import ModelError
from .pipeline import format_run_summary, run_pipeline
from .registry import library_versions

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    help="Train, tune, evaluate, and version churn models from a YAML config.",
)


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to the YAML config file."),
    ] = Path("config/default.yaml"),
    data: Annotated[
        Path | None,
        typer.Option("--data", "-d", help="Override the dataset path from the config."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG-level logging."),
    ] = False,
) -> None:
    """Train, tune, evaluate, and version every model defined in the config.

    Per-argument help lives on each option below (shown in ``--help``); this stays short so the
    generated help text is clean.
    """
    configure_logging(logging.DEBUG if verbose else logging.INFO)
    log_environment(library_versions())

    try:
        resolved = load_config(config)
        if data is not None:
            resolved.data.path = data
            logger.info("Overriding dataset path with --data %s", data)
        result = run_pipeline(resolved)
    except (ConfigError, DataValidationError, ModelError, FileNotFoundError) as exc:
        # Expected boundary failures: report cleanly, no traceback, non-zero exit for CI/scripts.
        logger.error("Pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc

    # The single sanctioned stdout print: the final run summary for the operator.
    typer.echo(format_run_summary(result))


if __name__ == "__main__":  # pragma: no cover - manual invocation convenience
    app()
