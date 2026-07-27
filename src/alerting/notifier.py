"""Alert delivery across console, file and webhook channels."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from email.message import EmailMessage
from pathlib import Path
from smtplib import SMTP, SMTPException
from typing import TYPE_CHECKING, Sequence

import requests

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, ensure_directory

if TYPE_CHECKING:
    from src.alerting.alert_manager import Alert

log = get_logger(__name__)

ANSI_RESET = "\033[0m"
SEVERITY_COLOURS = {
    "LOW": "\033[36m",
    "MEDIUM": "\033[33m",
    "HIGH": "\033[35m",
    "CRITICAL": "\033[31;1m",
}


class Channel(ABC):
    """A single alert delivery destination."""

    name: str = "channel"

    @abstractmethod
    def send(self, alert: "Alert") -> None:
        """Deliver one alert; failures must raise."""


class ConsoleChannel(Channel):
    """Prints colourised alerts to the application log."""

    name = "console"

    def __init__(self, colour: bool = True) -> None:
        self.colour = colour

    def send(self, alert: "Alert") -> None:
        severity = alert.severity.value
        prefix = f"{SEVERITY_COLOURS.get(severity, '')}[{severity}]{ANSI_RESET}" if self.colour else f"[{severity}]"
        log.warning("{} {} | id={} | {}", prefix, alert.message, alert.alert_id, alert.timestamp)


class FileChannel(Channel):
    """Appends alerts to a JSON-lines log file."""

    name = "file"

    def __init__(self, path: Path | str = "logs/alerts.log") -> None:
        self.path = Path(path)
        ensure_directory(self.path.parent)

    def send(self, alert: "Alert") -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(alert.to_dict()) + "\n")
        except OSError as exc:
            raise ValidationError(f"Could not append alert to {self.path}: {exc}") from exc


class WebhookChannel(Channel):
    """POSTs alerts as JSON to a configured HTTP endpoint."""

    name = "webhook"

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        if not url:
            raise ValidationError("Webhook channel requires a non-empty URL")
        self.url = url
        self.timeout = timeout

    def send(self, alert: "Alert") -> None:
        try:
            response = requests.post(self.url, json=alert.to_dict(), timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValidationError(f"Webhook delivery to {self.url} failed: {exc}") from exc


class EmailChannel(Channel):
    """Sends alerts over SMTP."""

    name = "email"

    def __init__(
        self,
        host: str,
        port: int,
        sender: str,
        recipients: Sequence[str],
        username: str | None = None,
        password: str | None = None,
        use_tls: bool = True,
        timeout: float = 10.0,
    ) -> None:
        if not recipients:
            raise ValidationError("Email channel requires at least one recipient")
        self.host = host
        self.port = port
        self.sender = sender
        self.recipients = list(recipients)
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.timeout = timeout

    def send(self, alert: "Alert") -> None:
        message = EmailMessage()
        message["Subject"] = f"[NIDS {alert.severity.value}] {alert.predicted_class}"
        message["From"] = self.sender
        message["To"] = ", ".join(self.recipients)
        message.set_content(json.dumps(alert.to_dict(), indent=2))

        try:
            with SMTP(self.host, self.port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(message)
        except (SMTPException, OSError) as exc:
            raise ValidationError(f"SMTP delivery to {self.host} failed: {exc}") from exc


class Notifier:
    """Fans an alert out to every configured channel."""

    def __init__(self, channels: Sequence[Channel] | None = None) -> None:
        self.channels: list[Channel] = list(channels) if channels else [ConsoleChannel()]
        self.failures: dict[str, int] = {}

    def add_channel(self, channel: Channel) -> None:
        """Register an additional delivery channel."""
        self.channels.append(channel)

    def dispatch(self, alert: "Alert") -> dict[str, bool]:
        """Send an alert to all channels, reporting per-channel success."""
        outcome: dict[str, bool] = {}
        for channel in self.channels:
            try:
                channel.send(alert)
                outcome[channel.name] = True
            except Exception as exc:  # noqa: BLE001 - one bad channel must not block the rest
                outcome[channel.name] = False
                self.failures[channel.name] = self.failures.get(channel.name, 0) + 1
                log.error("Channel '{}' failed to deliver alert {}: {}", channel.name, alert.alert_id, exc)
        return outcome


def build_notifier(
    channel_names: Sequence[str],
    alert_log_path: Path | str = "logs/alerts.log",
    webhook_url: str = "",
) -> Notifier:
    """Construct a notifier from configuration channel names."""
    channels: list[Channel] = []
    for name in channel_names:
        if name == "console":
            channels.append(ConsoleChannel())
        elif name == "file":
            channels.append(FileChannel(alert_log_path))
        elif name == "webhook":
            if webhook_url:
                channels.append(WebhookChannel(webhook_url))
            else:
                log.warning("Webhook channel requested without a URL; skipping it")
        else:
            log.warning("Unknown alert channel '{}'; skipping it", name)

    if not channels:
        log.warning("No usable alert channels configured; defaulting to console")
        channels.append(ConsoleChannel())
    return Notifier(channels)
