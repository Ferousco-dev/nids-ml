"""Shared runtime state for the API process."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.alerting.alert_manager import AlertManager
from src.alerting.notifier import build_notifier
from src.alerting.throttler import AlertThrottler
from src.detection.engine import DetectionEngine
from src.detection.stream_processor import StreamProcessor
from src.utils.config import Config, get_config
from src.utils.logger import get_logger
from src.utils.validators import ValidationError

log = get_logger(__name__)


@dataclass
class ServiceState:
    """Objects shared by every request handler."""

    config: Config
    started_at: float
    engine: DetectionEngine | None = None
    processor: StreamProcessor | None = None
    alert_manager: AlertManager | None = None
    load_error: str | None = None

    @property
    def model_loaded(self) -> bool:
        return self.engine is not None

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def require_engine(self) -> DetectionEngine:
        """Return the engine or raise if the model failed to load."""
        if self.engine is None:
            raise ValidationError(self.load_error or "No detection model is loaded")
        return self.engine


state = ServiceState(config=get_config(), started_at=time.monotonic())


def initialise(config: Config | None = None) -> ServiceState:
    """Load the model and wire up detection, alerting and streaming."""
    settings = config or get_config()
    state.config = settings
    state.started_at = time.monotonic()

    alerting = settings.alerting
    state.alert_manager = AlertManager(
        notifier=build_notifier(
            alerting.channels,
            alert_log_path=settings.resolve(alerting.alert_log_path),
            webhook_url=alerting.webhook_url,
        ),
        throttler=AlertThrottler(
            window_seconds=alerting.rate_limit.window_seconds,
            max_alerts=alerting.rate_limit.max_alerts,
        ),
        severity_thresholds=alerting.severity_thresholds,
    )

    try:
        state.engine = DetectionEngine.from_path(
            model_path=settings.resolve(settings.detection.model_path),
            confidence_threshold=settings.detection.confidence_threshold,
        )
        state.processor = StreamProcessor(
            state.engine,
            batch_size=settings.detection.batch_size,
            on_detection=state.alert_manager.handle,
        )
        state.load_error = None
        log.info("Detection service ready")
    except ValidationError as exc:
        state.engine = None
        state.processor = None
        state.load_error = str(exc)
        log.error("Detection service started without a model: {}", exc)

    return state


def shutdown() -> None:
    """Release runtime references held by the service."""
    state.engine = None
    state.processor = None
    state.alert_manager = None
    log.info("Detection service shut down")
