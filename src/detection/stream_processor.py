"""Sliding-window processing of a live flow stream."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Mapping

from src.detection.engine import DetectionEngine, DetectionResult
from src.utils.logger import get_logger
from src.utils.validators import ValidationError

log = get_logger(__name__)


@dataclass
class StreamStats:
    """Rolling statistics over the most recent detections."""

    processed: int
    attacks: int
    attack_rate: float
    class_counts: dict[str, int]
    mean_confidence: float
    trending_class: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processed": self.processed,
            "attacks": self.attacks,
            "attack_rate": round(self.attack_rate, 4),
            "class_counts": self.class_counts,
            "mean_confidence": round(self.mean_confidence, 4),
            "trending_class": self.trending_class,
        }


class StreamProcessor:
    """Consumes flows in batches and keeps a rolling detection buffer."""

    def __init__(
        self,
        engine: DetectionEngine,
        window_size: int = 500,
        batch_size: int = 100,
        on_detection: Callable[[DetectionResult], None] | None = None,
    ) -> None:
        if window_size < 1:
            raise ValidationError(f"window_size must be positive, got {window_size}")
        if batch_size < 1:
            raise ValidationError(f"batch_size must be positive, got {batch_size}")
        self.engine = engine
        self.batch_size = batch_size
        self.on_detection = on_detection
        self.buffer: deque[DetectionResult] = deque(maxlen=window_size)
        self.total_processed = 0
        self.total_attacks = 0

    def _record(self, result: DetectionResult) -> None:
        self.buffer.append(result)
        self.total_processed += 1
        if result.is_attack:
            self.total_attacks += 1
        if self.on_detection is not None:
            try:
                self.on_detection(result)
            except Exception as exc:  # noqa: BLE001 - a bad sink must not stop the stream
                log.error("Detection callback failed: {}", exc)

    def process_batch(self, flows: list[Mapping[str, Any]]) -> list[DetectionResult]:
        """Detect on one batch of flows and add them to the rolling window."""
        if not flows:
            return []
        results = self.engine.predict_many(flows)
        for result in results:
            self._record(result)
        return results

    def process_stream(self, flows: Iterable[Mapping[str, Any]]) -> Iterator[DetectionResult]:
        """Consume an iterable of flows, yielding detections batch by batch."""
        batch: list[Mapping[str, Any]] = []
        for flow in flows:
            batch.append(flow)
            if len(batch) >= self.batch_size:
                yield from self.process_batch(batch)
                batch = []
        if batch:
            yield from self.process_batch(batch)

    def stats(self) -> StreamStats:
        """Return statistics over the current window."""
        if not self.buffer:
            return StreamStats(0, 0, 0.0, {}, 0.0, None)

        counts = Counter(result.predicted_class for result in self.buffer)
        attacks = sum(1 for result in self.buffer if result.is_attack)
        confidence = sum(result.confidence for result in self.buffer) / len(self.buffer)
        attack_classes = Counter(
            result.predicted_class for result in self.buffer if result.is_attack
        )
        trending = attack_classes.most_common(1)[0][0] if attack_classes else None

        return StreamStats(
            processed=len(self.buffer),
            attacks=attacks,
            attack_rate=attacks / len(self.buffer),
            class_counts=dict(counts),
            mean_confidence=confidence,
            trending_class=trending,
        )

    def lifetime_stats(self) -> dict[str, Any]:
        """Return counters covering every flow seen since start-up."""
        rate = self.total_attacks / self.total_processed if self.total_processed else 0.0
        return {
            "total_processed": self.total_processed,
            "total_attacks": self.total_attacks,
            "lifetime_attack_rate": round(rate, 4),
            "window": self.stats().to_dict(),
        }

    def reset(self) -> None:
        """Clear the rolling window and lifetime counters."""
        self.buffer.clear()
        self.total_processed = 0
        self.total_attacks = 0
