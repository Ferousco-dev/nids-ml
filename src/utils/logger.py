"""Structured logging setup built on loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import LoggingConfig, get_config

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
)

_configured = False


def configure_logging(config: LoggingConfig | None = None, force: bool = False) -> None:
    """Attach console and file sinks to the global logger.

    Repeated calls are ignored unless ``force`` is set, so importing modules
    never duplicates sinks.
    """
    global _configured
    if _configured and not force:
        return

    settings = config or get_config().logging
    log_dir = Path(settings.log_dir)
    if not log_dir.is_absolute():
        from src.utils.config import PROJECT_ROOT

        log_dir = PROJECT_ROOT / log_dir

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create log directory {}: {}", log_dir, exc)

    logger.remove()
    logger.add(sys.stderr, level=settings.level, format=_CONSOLE_FORMAT, colorize=True)
    logger.add(
        log_dir / "app.log",
        level=settings.level,
        rotation=settings.rotation,
        retention=settings.retention,
        serialize=settings.json_format,
        enqueue=True,
    )
    logger.add(
        log_dir / "errors.log",
        level="ERROR",
        rotation=settings.rotation,
        retention=settings.retention,
        serialize=settings.json_format,
        backtrace=True,
        diagnose=False,
        enqueue=True,
    )
    _configured = True


def get_logger(name: str):
    """Return a logger bound to ``name``."""
    configure_logging()
    return logger.bind(module=name)
