"""Configuration loading for the NIDS-ML system.

Settings are read from a YAML file and may be overridden by environment
variables of the form ``NIDS_<SECTION>__<KEY>`` (e.g. ``NIDS_API__PORT=9000``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PREFIX = "NIDS_"


@dataclass
class AppConfig:
    name: str = "NIDS-ML"
    version: str = "0.1.0"
    debug: bool = False


@dataclass
class DataConfig:
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42


@dataclass
class FeatureConfig:
    target_column: str = "label"
    drop_columns: list[str] = field(default_factory=list)
    selection_method: str = "mutual_info"
    top_k: int = 30


@dataclass
class DetectionConfig:
    confidence_threshold: float = 0.7
    batch_size: int = 100
    model_path: str = "models/best_model.pkl"


@dataclass
class RateLimitConfig:
    window_seconds: int = 60
    max_alerts: int = 5


@dataclass
class AlertingConfig:
    severity_thresholds: dict[str, float] = field(
        default_factory=lambda: {"low": 0.5, "medium": 0.7, "high": 0.85, "critical": 0.95}
    )
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    channels: list[str] = field(default_factory=lambda: ["console", "file"])
    webhook_url: str = ""
    alert_log_path: str = "logs/alerts.log"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    json_format: bool = False
    rotation: str = "10 MB"
    retention: str = "14 days"


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a config path against the project root."""
        path = Path(relative_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        logger.warning("Config file {} not found; falling back to defaults", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Unable to read configuration from {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise RuntimeError(f"Configuration at {path} must be a mapping, got {type(content).__name__}")
    return content


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX) or "__" not in env_key:
            continue
        section, _, key = env_key[len(ENV_PREFIX):].partition("__")
        section, key = section.lower(), key.lower()
        raw.setdefault(section, {})
        if isinstance(raw[section], dict):
            raw[section][key] = _coerce(env_value)
            logger.debug("Config override from environment: {}.{}", section, key)
    return raw


def _build_section(cls: type, values: Any) -> Any:
    if not isinstance(values, dict):
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in values.items() if k in known})


def load_config(path: Path | str | None = None) -> Config:
    """Load configuration from YAML, applying environment overrides."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw = _apply_env_overrides(_read_yaml(config_path))

    alerting_raw = raw.get("alerting", {}) or {}
    alerting = _build_section(AlertingConfig, alerting_raw)
    alerting.rate_limit = _build_section(RateLimitConfig, alerting_raw.get("rate_limit", {}))

    return Config(
        app=_build_section(AppConfig, raw.get("app", {})),
        data=_build_section(DataConfig, raw.get("data", {})),
        features=_build_section(FeatureConfig, raw.get("features", {})),
        models=raw.get("models", {}) or {},
        detection=_build_section(DetectionConfig, raw.get("detection", {})),
        alerting=alerting,
        logging=_build_section(LoggingConfig, raw.get("logging", {})),
        api=_build_section(ApiConfig, raw.get("api", {})),
    )


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the process-wide configuration singleton."""
    return load_config()
