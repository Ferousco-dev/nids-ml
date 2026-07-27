"""Alert construction, severity grading and dispatch."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

from src.alerting.notifier import Notifier
from src.alerting.throttler import AlertThrottler
from src.detection.engine import DetectionResult
from src.utils.logger import get_logger
from src.utils.validators import ValidationError

log = get_logger(__name__)


class Severity(str, Enum):
    """Alert severity ordered from least to most urgent."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEVERITY_ORDER.index(self)


_SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

DEFAULT_THRESHOLDS = {"low": 0.5, "medium": 0.7, "high": 0.85, "critical": 0.95}

ATTACK_WEIGHT = {"u2r": 1, "r2l": 1, "dos": 0, "probe": 0, "attack": 0}


@dataclass
class Alert:
    """An enriched, dispatchable intrusion alert."""

    alert_id: str
    timestamp: str
    severity: Severity
    predicted_class: str
    confidence: float
    message: str
    flow_summary: dict[str, Any] = field(default_factory=dict)
    class_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "message": self.message,
            "flow_summary": self.flow_summary,
            "class_probabilities": {k: round(v, 6) for k, v in self.class_probabilities.items()},
        }


class AlertManager:
    """Turns detections into alerts, throttles them and dispatches them."""

    def __init__(
        self,
        notifier: Notifier | None = None,
        throttler: AlertThrottler | None = None,
        severity_thresholds: dict[str, float] | None = None,
        min_severity: Severity = Severity.LOW,
        history_size: int = 500,
    ) -> None:
        self.notifier = notifier or Notifier()
        self.throttler = throttler or AlertThrottler()
        self.thresholds = self._validate_thresholds(severity_thresholds or DEFAULT_THRESHOLDS)
        self.min_severity = min_severity
        self.history_size = max(1, history_size)
        self.history: list[Alert] = []

    @staticmethod
    def _validate_thresholds(thresholds: dict[str, float]) -> dict[str, float]:
        missing = [name for name in DEFAULT_THRESHOLDS if name not in thresholds]
        if missing:
            raise ValidationError(f"Missing severity threshold(s): {', '.join(missing)}")
        ordered = [thresholds[name] for name in ("low", "medium", "high", "critical")]
        if any(later < earlier for earlier, later in zip(ordered, ordered[1:])):
            raise ValidationError("Severity thresholds must increase from low to critical")
        return {name: float(thresholds[name]) for name in DEFAULT_THRESHOLDS}

    def grade(self, confidence: float, predicted_class: str) -> Severity:
        """Map a confidence score and attack family onto a severity level."""
        if confidence >= self.thresholds["critical"]:
            base = Severity.CRITICAL
        elif confidence >= self.thresholds["high"]:
            base = Severity.HIGH
        elif confidence >= self.thresholds["medium"]:
            base = Severity.MEDIUM
        else:
            base = Severity.LOW

        bump = ATTACK_WEIGHT.get(predicted_class.lower(), 0)
        return _SEVERITY_ORDER[min(base.rank + bump, len(_SEVERITY_ORDER) - 1)]

    def build_alert(self, detection: DetectionResult) -> Alert:
        """Create an enriched alert from a detection result."""
        severity = self.grade(detection.confidence, detection.predicted_class)
        summary = ", ".join(f"{key}={value}" for key, value in detection.flow_summary.items())
        return Alert(
            alert_id=uuid.uuid4().hex[:12],
            timestamp=detection.timestamp or datetime.now(timezone.utc).isoformat(),
            severity=severity,
            predicted_class=detection.predicted_class,
            confidence=detection.confidence,
            message=(
                f"{detection.predicted_class.upper()} traffic detected with "
                f"{detection.confidence:.1%} confidence"
                + (f" [{summary}]" if summary else "")
            ),
            flow_summary=dict(detection.flow_summary),
            class_probabilities=dict(detection.class_probabilities),
        )

    def _remember(self, alert: Alert) -> None:
        self.history.append(alert)
        if len(self.history) > self.history_size:
            del self.history[: len(self.history) - self.history_size]

    def handle(self, detection: DetectionResult) -> Alert | None:
        """Process one detection, returning the alert if one was dispatched."""
        if not detection.is_attack:
            return None

        alert = self.build_alert(detection)
        if alert.severity.rank < self.min_severity.rank:
            return None

        decision = self.throttler.check(f"{alert.predicted_class}:{alert.severity.value}")
        if not decision.allowed:
            log.debug("Alert {} suppressed by throttler", alert.alert_id)
            return None

        self.notifier.dispatch(alert)
        self._remember(alert)
        return alert

    def handle_many(self, detections: Sequence[DetectionResult]) -> list[Alert]:
        """Process a batch of detections and return the alerts dispatched."""
        return [alert for alert in (self.handle(item) for item in detections) if alert is not None]

    def recent(self, limit: int = 50) -> list[Alert]:
        """Return the most recent dispatched alerts, newest last."""
        if limit < 1:
            raise ValidationError(f"limit must be positive, got {limit}")
        return self.history[-limit:]

    def severity_counts(self) -> dict[str, int]:
        """Count alerts in history by severity."""
        counts = {severity.value: 0 for severity in _SEVERITY_ORDER}
        for alert in self.history:
            counts[alert.severity.value] += 1
        return counts
