"""Rate limiting for repeated alerts."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from src.utils.logger import get_logger
from src.utils.validators import ValidationError

log = get_logger(__name__)


@dataclass
class ThrottleDecision:
    """Outcome of a throttling check for one alert key."""

    allowed: bool
    key: str
    count_in_window: int
    suppressed_total: int


class AlertThrottler:
    """Sliding-window rate limiter keyed by attack type and severity.

    At most ``max_alerts`` alerts sharing a key are emitted per
    ``window_seconds``; the rest are counted as suppressed.
    """

    def __init__(
        self,
        window_seconds: int = 60,
        max_alerts: int = 5,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValidationError(f"window_seconds must be positive, got {window_seconds}")
        if max_alerts <= 0:
            raise ValidationError(f"max_alerts must be positive, got {max_alerts}")
        self.window_seconds = window_seconds
        self.max_alerts = max_alerts
        self._clock = clock or time.monotonic
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._suppressed: dict[str, int] = defaultdict(int)

    def _prune(self, key: str, now: float) -> deque[float]:
        events = self._events[key]
        cutoff = now - self.window_seconds
        while events and events[0] < cutoff:
            events.popleft()
        return events

    def check(self, key: str) -> ThrottleDecision:
        """Decide whether an alert with this key may be emitted now."""
        if not key:
            raise ValidationError("Throttle key must be a non-empty string")

        now = self._clock()
        events = self._prune(key, now)
        if len(events) >= self.max_alerts:
            self._suppressed[key] += 1
            if self._suppressed[key] == 1:
                log.warning("Throttling alerts for '{}' ({}/{}s)", key, self.max_alerts, self.window_seconds)
            return ThrottleDecision(False, key, len(events), self._suppressed[key])

        events.append(now)
        return ThrottleDecision(True, key, len(events), self._suppressed[key])

    def suppressed_count(self, key: str) -> int:
        """Number of alerts suppressed so far for a key."""
        return self._suppressed.get(key, 0)

    def reset(self, key: str | None = None) -> None:
        """Clear throttling state for one key or all keys."""
        if key is None:
            self._events.clear()
            self._suppressed.clear()
            return
        self._events.pop(key, None)
        self._suppressed.pop(key, None)
