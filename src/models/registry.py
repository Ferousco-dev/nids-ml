"""Persistence and versioning of trained detection models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, ensure_directory

log = get_logger(__name__)

BEST_MODEL_NAME = "best_model.pkl"
METADATA_SUFFIX = ".meta.json"


@dataclass
class ModelMetadata:
    """Descriptive metadata stored alongside a serialised model."""

    model_name: str
    version: str
    created_at: str
    feature_names: list[str]
    class_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBundle:
    """A model paired with everything needed to serve it."""

    estimator: BaseEstimator
    metadata: ModelMetadata
    preprocessor: Any | None = None


def _serialisable(params: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class ModelRegistry:
    """File-backed registry that stores versioned model bundles."""

    def __init__(self, root: Path | str = "models") -> None:
        self.root = ensure_directory(root)

    def _version(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    def save(
        self,
        estimator: BaseEstimator,
        model_name: str,
        feature_names: list[str],
        class_names: list[str],
        metrics: dict[str, float] | None = None,
        preprocessor: Any | None = None,
        as_best: bool = False,
    ) -> ModelBundle:
        """Serialise a model with its metadata and optionally mark it as best."""
        if not feature_names:
            raise ValidationError("feature_names must not be empty")
        if not class_names:
            raise ValidationError("class_names must not be empty")

        metadata = ModelMetadata(
            model_name=model_name,
            version=self._version(),
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_names=list(feature_names),
            class_names=list(class_names),
            metrics=dict(metrics or {}),
            hyperparameters=_serialisable(estimator.get_params()),
        )
        bundle = ModelBundle(estimator=estimator, metadata=metadata, preprocessor=preprocessor)

        versioned = self.root / f"{model_name}_{metadata.version}.pkl"
        self._write(bundle, versioned)
        if as_best:
            self._write(bundle, self.root / BEST_MODEL_NAME)
        return bundle

    def _write(self, bundle: ModelBundle, path: Path) -> Path:
        try:
            joblib.dump(bundle, path)
            with path.with_suffix(METADATA_SUFFIX).open("w", encoding="utf-8") as handle:
                json.dump(bundle.metadata.to_dict(), handle, indent=2)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Could not save model to {path}: {exc}") from exc
        log.info("Model '{}' saved to {}", bundle.metadata.model_name, path)
        return path

    def load(self, path: Path | str | None = None) -> ModelBundle:
        """Load a model bundle, defaulting to the registered best model."""
        file_path = Path(path) if path is not None else self.root / BEST_MODEL_NAME
        if not file_path.is_file():
            raise ValidationError(
                f"Model file not found: {file_path}. Run scripts/run_pipeline.py to train one."
            )
        try:
            bundle = joblib.load(file_path)
        except (OSError, ValueError, ModuleNotFoundError) as exc:
            raise ValidationError(f"Could not load model from {file_path}: {exc}") from exc
        if not isinstance(bundle, ModelBundle):
            raise ValidationError(f"{file_path} does not contain a ModelBundle")
        log.info("Loaded model '{}' version {}", bundle.metadata.model_name, bundle.metadata.version)
        return bundle

    def list_models(self) -> list[ModelMetadata]:
        """List metadata for every versioned model in the registry."""
        entries: list[ModelMetadata] = []
        for meta_file in sorted(self.root.glob(f"*{METADATA_SUFFIX}")):
            try:
                with meta_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                entries.append(ModelMetadata(**payload))
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                log.warning("Skipping unreadable metadata {}: {}", meta_file, exc)
        return entries

    def best_model_path(self) -> Path:
        """Path of the currently registered best model."""
        return self.root / BEST_MODEL_NAME
