"""Input validation helpers shared across the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class ValidationError(ValueError):
    """Raised when input data fails a validation rule."""


def validate_dataframe(
    frame: Any,
    required_columns: Sequence[str] | None = None,
    min_rows: int = 1,
) -> pd.DataFrame:
    """Validate that ``frame`` is a non-empty DataFrame with the required columns."""
    if not isinstance(frame, pd.DataFrame):
        raise ValidationError(f"Expected a pandas DataFrame, got {type(frame).__name__}")
    if len(frame) < min_rows:
        raise ValidationError(f"Expected at least {min_rows} row(s), got {len(frame)}")
    if required_columns:
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValidationError(f"Missing required column(s): {', '.join(missing)}")
    return frame


def validate_target(frame: pd.DataFrame, target_column: str, min_classes: int = 2) -> pd.Series:
    """Validate the target column and return it."""
    validate_dataframe(frame, required_columns=[target_column])
    target = frame[target_column]
    if target.isna().any():
        raise ValidationError(f"Target column '{target_column}' contains missing values")
    if target.nunique() < min_classes:
        raise ValidationError(
            f"Target column '{target_column}' needs at least {min_classes} classes, "
            f"found {target.nunique()}"
        )
    return target


def validate_path(path: Path | str, must_exist: bool = True, suffixes: Iterable[str] | None = None) -> Path:
    """Validate a filesystem path and return it as a :class:`Path`."""
    resolved = Path(path)
    if must_exist and not resolved.exists():
        raise ValidationError(f"Path does not exist: {resolved}")
    if suffixes is not None:
        allowed = {suffix.lower() for suffix in suffixes}
        if resolved.suffix.lower() not in allowed:
            raise ValidationError(
                f"Unsupported file type '{resolved.suffix}'; expected one of {sorted(allowed)}"
            )
    return resolved


def validate_ratio(value: float, name: str, lower: float = 0.0, upper: float = 1.0) -> float:
    """Validate that a numeric value lies within ``[lower, upper]``."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"'{name}' must be numeric, got {type(value).__name__}")
    if not lower <= float(value) <= upper:
        raise ValidationError(f"'{name}' must be between {lower} and {upper}, got {value}")
    return float(value)


def validate_flow(flow: Mapping[str, Any], expected_features: Sequence[str]) -> dict[str, Any]:
    """Validate a single network flow record against the expected feature list."""
    if not isinstance(flow, Mapping):
        raise ValidationError(f"A flow must be a mapping, got {type(flow).__name__}")
    missing = [feature for feature in expected_features if feature not in flow]
    if missing:
        raise ValidationError(f"Flow is missing feature(s): {', '.join(missing[:10])}")

    cleaned: dict[str, Any] = {}
    for feature in expected_features:
        value = flow[feature]
        if value is None:
            raise ValidationError(f"Feature '{feature}' is null")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite(float(value)):
                raise ValidationError(f"Feature '{feature}' must be finite, got {value}")
        cleaned[feature] = value
    return cleaned


def ensure_directory(path: Path | str) -> Path:
    """Create a directory (and parents) if needed and return it."""
    directory = Path(path)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValidationError(f"Could not create directory {directory}: {exc}") from exc
    return directory
