"""Inference engine that turns raw flows into detection results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.features.engineer import engineer_features
from src.models.evaluator import predict_proba
from src.models.registry import ModelBundle, ModelRegistry
from src.utils.logger import get_logger
from src.utils.validators import ValidationError, validate_dataframe

log = get_logger(__name__)

NORMAL_LABELS = {"normal", "benign"}


@dataclass
class DetectionResult:
    """Prediction for a single network flow."""

    predicted_class: str
    confidence: float
    is_attack: bool
    timestamp: str
    class_probabilities: dict[str, float] = field(default_factory=dict)
    flow_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "is_attack": self.is_attack,
            "timestamp": self.timestamp,
            "class_probabilities": {k: round(v, 6) for k, v in self.class_probabilities.items()},
            "flow_summary": self.flow_summary,
        }


SUMMARY_FIELDS = ("duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "count")


def _summarise(flow: Mapping[str, Any]) -> dict[str, Any]:
    return {key: flow[key] for key in SUMMARY_FIELDS if key in flow}


class DetectionEngine:
    """Applies a trained model bundle to live or batched network flows."""

    def __init__(
        self,
        bundle: ModelBundle,
        confidence_threshold: float = 0.7,
        apply_feature_engineering: bool = True,
    ) -> None:
        self.bundle = bundle
        self.confidence_threshold = confidence_threshold
        self.apply_feature_engineering = apply_feature_engineering
        self.preprocessor = bundle.preprocessor
        self.feature_names = bundle.metadata.feature_names
        self.class_names = bundle.metadata.class_names

    @classmethod
    def from_path(
        cls,
        model_path: Path | str | None = None,
        confidence_threshold: float = 0.7,
        registry_root: Path | str = "models",
    ) -> DetectionEngine:
        """Build an engine from a serialised model bundle."""
        registry = ModelRegistry(registry_root)
        return cls(registry.load(model_path), confidence_threshold=confidence_threshold)

    def _prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        validate_dataframe(frame)
        prepared = engineer_features(frame) if self.apply_feature_engineering else frame
        if self.preprocessor is not None:
            prepared = self.preprocessor.transform_features(prepared)
        missing = [name for name in self.feature_names if name not in prepared.columns]
        if missing:
            raise ValidationError(f"Missing model feature(s): {', '.join(missing[:10])}")
        return prepared[self.feature_names].astype(np.float64)

    def _decode(self, index: int) -> str:
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return str(index)

    def predict_frame(self, frame: pd.DataFrame) -> list[DetectionResult]:
        """Run detection over every row of a flow frame."""
        prepared = self._prepare(frame)
        matrix = prepared.to_numpy(dtype=np.float64)
        predictions = np.asarray(self.bundle.estimator.predict(matrix))
        probabilities = predict_proba(self.bundle.estimator, prepared)
        timestamp = datetime.now(timezone.utc).isoformat()

        results: list[DetectionResult] = []
        raw_records = frame.to_dict(orient="records")
        for position, prediction in enumerate(predictions):
            label = self._decode(int(prediction))
            if probabilities is not None:
                row = probabilities[position]
                confidence = float(row.max())
                distribution = {
                    self._decode(index): float(value) for index, value in enumerate(row)
                }
            else:
                confidence = 1.0
                distribution = {label: 1.0}
            results.append(
                DetectionResult(
                    predicted_class=label,
                    confidence=confidence,
                    is_attack=label.lower() not in NORMAL_LABELS
                    and confidence >= self.confidence_threshold,
                    timestamp=timestamp,
                    class_probabilities=distribution,
                    flow_summary=_summarise(raw_records[position]),
                )
            )
        return results

    def predict_one(self, flow: Mapping[str, Any]) -> DetectionResult:
        """Run detection on a single flow record."""
        if not isinstance(flow, Mapping):
            raise ValidationError(f"A flow must be a mapping, got {type(flow).__name__}")
        return self.predict_frame(pd.DataFrame([dict(flow)]))[0]

    def predict_many(self, flows: Sequence[Mapping[str, Any]]) -> list[DetectionResult]:
        """Run detection on a batch of flow records."""
        if not flows:
            raise ValidationError("Batch detection requires at least one flow")
        return self.predict_frame(pd.DataFrame([dict(flow) for flow in flows]))
