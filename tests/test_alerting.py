"""Tests for alert generation, throttling and delivery."""

from __future__ import annotations

import json

import pytest

from src.alerting.alert_manager import AlertManager, Severity
from src.alerting.notifier import Channel, ConsoleChannel, FileChannel, Notifier, build_notifier
from src.alerting.throttler import AlertThrottler
from src.detection.engine import DetectionResult
from src.utils.validators import ValidationError


class RecordingChannel(Channel):
    name = "recording"

    def __init__(self) -> None:
        self.received: list = []

    def send(self, alert) -> None:
        self.received.append(alert)


class FailingChannel(Channel):
    name = "failing"

    def send(self, alert) -> None:
        raise RuntimeError("channel offline")


def make_detection(predicted_class: str = "dos", confidence: float = 0.9) -> DetectionResult:
    return DetectionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        is_attack=predicted_class != "normal",
        timestamp="2026-01-01T00:00:00+00:00",
        class_probabilities={predicted_class: confidence},
        flow_summary={"protocol_type": "tcp", "service": "http"},
    )


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_throttler_limits_within_window() -> None:
    clock = FakeClock()
    throttler = AlertThrottler(window_seconds=60, max_alerts=3, clock=clock)
    decisions = [throttler.check("dos:HIGH") for _ in range(5)]
    assert [decision.allowed for decision in decisions] == [True, True, True, False, False]
    assert throttler.suppressed_count("dos:HIGH") == 2


def test_throttler_window_expires() -> None:
    clock = FakeClock()
    throttler = AlertThrottler(window_seconds=60, max_alerts=1, clock=clock)
    assert throttler.check("dos:HIGH").allowed
    assert not throttler.check("dos:HIGH").allowed
    clock.advance(61)
    assert throttler.check("dos:HIGH").allowed


def test_throttler_keys_are_independent() -> None:
    throttler = AlertThrottler(window_seconds=60, max_alerts=1)
    assert throttler.check("dos:HIGH").allowed
    assert throttler.check("probe:LOW").allowed


def test_throttler_reset() -> None:
    throttler = AlertThrottler(window_seconds=60, max_alerts=1)
    throttler.check("dos:HIGH")
    throttler.reset("dos:HIGH")
    assert throttler.check("dos:HIGH").allowed


def test_throttler_rejects_invalid_configuration() -> None:
    with pytest.raises(ValidationError):
        AlertThrottler(window_seconds=0)
    with pytest.raises(ValidationError):
        AlertThrottler(max_alerts=0)
    with pytest.raises(ValidationError):
        AlertThrottler().check("")


@pytest.mark.parametrize(
    ("confidence", "expected"),
    [(0.55, Severity.LOW), (0.75, Severity.MEDIUM), (0.9, Severity.HIGH), (0.99, Severity.CRITICAL)],
)
def test_severity_grading(confidence: float, expected: Severity) -> None:
    assert AlertManager(notifier=Notifier([RecordingChannel()])).grade(confidence, "dos") == expected


def test_privileged_attacks_escalate_severity() -> None:
    manager = AlertManager(notifier=Notifier([RecordingChannel()]))
    assert manager.grade(0.9, "u2r") == Severity.CRITICAL
    assert manager.grade(0.9, "dos") == Severity.HIGH


def test_manager_dispatches_attacks_only() -> None:
    channel = RecordingChannel()
    manager = AlertManager(notifier=Notifier([channel]))
    assert manager.handle(make_detection("normal")) is None
    alert = manager.handle(make_detection("dos", 0.95))
    assert alert is not None
    assert len(channel.received) == 1
    assert alert.alert_id and alert.message


def test_manager_applies_throttling() -> None:
    channel = RecordingChannel()
    manager = AlertManager(
        notifier=Notifier([channel]), throttler=AlertThrottler(window_seconds=60, max_alerts=2)
    )
    detections = [make_detection("dos", 0.9) for _ in range(5)]
    alerts = manager.handle_many(detections)
    assert len(alerts) == 2
    assert len(channel.received) == 2


def test_manager_respects_minimum_severity() -> None:
    channel = RecordingChannel()
    manager = AlertManager(notifier=Notifier([channel]), min_severity=Severity.CRITICAL)
    assert manager.handle(make_detection("probe", 0.72)) is None


def test_manager_history_and_counts() -> None:
    manager = AlertManager(notifier=Notifier([RecordingChannel()]), history_size=3)
    for _ in range(5):
        manager.throttler.reset()
        manager.handle(make_detection("dos", 0.96))
    assert len(manager.history) == 3
    assert manager.severity_counts()["CRITICAL"] == 3
    assert len(manager.recent(2)) == 2
    with pytest.raises(ValidationError):
        manager.recent(0)


def test_manager_rejects_unordered_thresholds() -> None:
    with pytest.raises(ValidationError):
        AlertManager(severity_thresholds={"low": 0.9, "medium": 0.7, "high": 0.8, "critical": 0.95})
    with pytest.raises(ValidationError):
        AlertManager(severity_thresholds={"low": 0.5})


def test_file_channel_appends_json_lines(tmp_path) -> None:
    path = tmp_path / "alerts.log"
    manager = AlertManager(notifier=Notifier([FileChannel(path)]))
    manager.handle(make_detection("r2l", 0.88))
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["predicted_class"] == "r2l"


def test_notifier_isolates_channel_failures() -> None:
    good = RecordingChannel()
    notifier = Notifier([FailingChannel(), good])
    manager = AlertManager(notifier=notifier)
    manager.handle(make_detection("dos", 0.9))
    assert len(good.received) == 1
    assert notifier.failures["failing"] == 1


def test_build_notifier_from_config(tmp_path) -> None:
    notifier = build_notifier(["console", "file", "webhook", "unknown"], tmp_path / "alerts.log")
    names = [channel.name for channel in notifier.channels]
    assert names == ["console", "file"]

    fallback = build_notifier([])
    assert isinstance(fallback.channels[0], ConsoleChannel)
