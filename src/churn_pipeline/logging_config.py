"""Central logging configuration.

This is the single place where logging is configured for the whole application. Library
modules only ever call ``logging.getLogger(__name__)`` and never touch handlers, so import
order never changes logging behaviour and tests can capture logs predictably. The CLI calls
:func:`configure_logging` exactly once at startup.
"""

from __future__ import annotations

import logging
import platform
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging handlers once for the application.

    Uses ``force=True`` so repeated calls (e.g. in tests) reset handlers instead of stacking
    duplicate log lines. Kept deliberately simple — structured console logging is enough for a
    local pipeline; in production this is where a JSON formatter / Cloud Logging handler attaches.

    Args:
        level: Root logging level; defaults to ``logging.INFO``.
    """
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        force=True,
    )


def log_environment(component_versions: dict[str, str]) -> None:
    """Log the interpreter and key library versions for reproducibility visibility.

    Emitting versions at startup makes the exact environment that produced a run visible in the
    logs during the demo, mirroring the same facts captured in each artifact's metadata.

    Args:
        component_versions: Mapping of library name to version string (e.g. sklearn, pandas).
    """
    logger.info("Python: %s (%s)", platform.python_version(), sys.executable)
    for name, version in component_versions.items():
        logger.info("%s: %s", name, version)
