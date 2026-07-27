"""Feature selection by mutual information, RFE or correlation pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, ensure_directory, validate_dataframe

log = get_logger(__name__)

SelectionMethod = Literal["mutual_info", "rfe", "correlation", "importance"]


@dataclass
class SelectionResult:
    """Selected feature names with their ranking scores."""

    method: str
    selected: list[str]
    scores: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"method": self.method, "selected": self.selected, "scores": self.scores}


def drop_correlated(features: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Return feature names left after removing highly correlated duplicates."""
    validate_dataframe(features)
    numeric = features.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return numeric.columns.tolist()

    correlation = numeric.corr().abs().fillna(0.0)
    upper = correlation.where(np.triu(np.ones(correlation.shape, dtype=bool), k=1))
    redundant = [column for column in upper.columns if (upper[column] > threshold).any()]
    if redundant:
        log.info("Dropping {} correlated feature(s) above r={}", len(redundant), threshold)
    return [column for column in numeric.columns if column not in redundant]


def _mutual_info_scores(features: pd.DataFrame, target: np.ndarray, random_state: int) -> pd.Series:
    scores = mutual_info_classif(features.to_numpy(), target, random_state=random_state)
    return pd.Series(scores, index=features.columns).sort_values(ascending=False)


def _importance_scores(features: pd.DataFrame, target: np.ndarray, random_state: int) -> pd.Series:
    model = RandomForestClassifier(
        n_estimators=120, max_depth=16, n_jobs=-1, random_state=random_state
    )
    model.fit(features.to_numpy(), target)
    return pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)


def _rfe_scores(
    features: pd.DataFrame, target: np.ndarray, top_k: int, random_state: int
) -> pd.Series:
    estimator = RandomForestClassifier(
        n_estimators=80, max_depth=12, n_jobs=-1, random_state=random_state
    )
    selector = RFE(estimator, n_features_to_select=top_k, step=0.2)
    selector.fit(features.to_numpy(), target)
    ranking = pd.Series(selector.ranking_, index=features.columns)
    return (1.0 / ranking).sort_values(ascending=False)


def select_features(
    features: pd.DataFrame,
    target: Sequence,
    method: SelectionMethod = "mutual_info",
    top_k: int = 30,
    correlation_threshold: float = 0.95,
    random_state: int = 42,
) -> SelectionResult:
    """Rank features and return the top ``top_k`` after correlation pruning."""
    validate_dataframe(features)
    y = np.asarray(target)
    if len(y) != len(features):
        raise ValidationError(f"Feature/target length mismatch: {len(features)} vs {len(y)}")
    if top_k < 1:
        raise ValidationError(f"top_k must be positive, got {top_k}")

    candidates = drop_correlated(features, threshold=correlation_threshold)
    if not candidates:
        raise ValidationError("No numeric features remain after correlation pruning")
    reduced = features[candidates]
    effective_k = min(top_k, reduced.shape[1])

    if method == "mutual_info":
        scores = _mutual_info_scores(reduced, y, random_state)
    elif method == "importance":
        scores = _importance_scores(reduced, y, random_state)
    elif method == "rfe":
        scores = _rfe_scores(reduced, y, effective_k, random_state)
    elif method == "correlation":
        scores = pd.Series(
            {column: float(abs(np.corrcoef(reduced[column], y)[0, 1])) for column in reduced.columns}
        ).fillna(0.0).sort_values(ascending=False)
    else:
        raise ValidationError(f"Unknown selection method '{method}'")

    selected = scores.head(effective_k).index.tolist()
    log.info("Selected {} feature(s) using {}", len(selected), method)
    return SelectionResult(
        method=method,
        selected=selected,
        scores={name: float(value) for name, value in scores.items()},
    )


def save_selection(result: SelectionResult, path: Path | str) -> Path:
    """Write the selection result to JSON."""
    file_path = Path(path)
    ensure_directory(file_path.parent)
    try:
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(result.to_dict(), handle, indent=2)
    except OSError as exc:
        raise ValidationError(f"Could not save feature selection to {file_path}: {exc}") from exc
    log.info("Feature selection saved to {}", file_path)
    return file_path


def load_selection(path: Path | str) -> SelectionResult:
    """Read a selection result previously written by :func:`save_selection`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValidationError(f"Feature selection file not found: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Could not read feature selection from {file_path}: {exc}") from exc
    return SelectionResult(
        method=payload.get("method", "unknown"),
        selected=list(payload.get("selected", [])),
        scores=dict(payload.get("scores", {})),
    )
