"""Tests for the detection engine and stream processor."""

from __future__ import annotations

import pandas as pd
import pytest

from src.detection.stream_processor import StreamProcessor
from src.utils.validators import ValidationError


def test_predict_one_returns_known_class(engine, sample_flow) -> None:
    result = engine.predict_one(sample_flow)
    assert result.predicted_class in engine.class_names
    assert 0.0 <= result.confidence <= 1.0
    assert result.timestamp


def test_probabilities_sum_to_one(engine, sample_flow) -> None:
    result = engine.predict_one(sample_flow)
    assert sum(result.class_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_normal_flow_is_not_flagged(engine, raw_dataset: pd.DataFrame) -> None:
    normal = raw_dataset[raw_dataset["label"] == "normal"].drop(columns=["label"]).iloc[0].to_dict()
    result = engine.predict_one(normal)
    assert result.is_attack == (result.predicted_class != "normal")


def test_predict_many_matches_input_length(engine, raw_dataset: pd.DataFrame) -> None:
    flows = raw_dataset.drop(columns=["label"]).head(25).to_dict(orient="records")
    results = engine.predict_many(flows)
    assert len(results) == 25


def test_predict_many_rejects_empty_batch(engine) -> None:
    with pytest.raises(ValidationError):
        engine.predict_many([])


def test_predict_one_rejects_non_mapping(engine) -> None:
    with pytest.raises(ValidationError):
        engine.predict_one([1, 2, 3])


def test_predict_rejects_missing_features(engine) -> None:
    with pytest.raises(ValidationError):
        engine.predict_one({"duration": 1.0})


def test_confidence_threshold_controls_flagging(model_bundle, raw_dataset: pd.DataFrame) -> None:
    from src.detection.engine import DetectionEngine

    flows = raw_dataset.drop(columns=["label"]).head(50).to_dict(orient="records")
    strict = DetectionEngine(model_bundle, confidence_threshold=1.01).predict_many(flows)
    assert not any(result.is_attack for result in strict)


def test_stream_processor_batches_and_tracks(engine, raw_dataset: pd.DataFrame) -> None:
    flows = raw_dataset.drop(columns=["label"]).head(60).to_dict(orient="records")
    processor = StreamProcessor(engine, window_size=40, batch_size=10)
    results = list(processor.process_stream(flows))

    assert len(results) == 60
    assert processor.total_processed == 60
    stats = processor.stats()
    assert stats.processed == 40
    assert 0.0 <= stats.attack_rate <= 1.0
    assert sum(stats.class_counts.values()) == 40


def test_stream_processor_invokes_callback(engine, raw_dataset: pd.DataFrame) -> None:
    seen = []
    processor = StreamProcessor(engine, batch_size=5, on_detection=seen.append)
    processor.process_batch(raw_dataset.drop(columns=["label"]).head(12).to_dict(orient="records"))
    assert len(seen) == 12


def test_stream_processor_survives_failing_callback(engine, raw_dataset: pd.DataFrame) -> None:
    def failing(_result) -> None:
        raise RuntimeError("sink is down")

    processor = StreamProcessor(engine, batch_size=5, on_detection=failing)
    results = processor.process_batch(
        raw_dataset.drop(columns=["label"]).head(5).to_dict(orient="records")
    )
    assert len(results) == 5


def test_stream_processor_empty_state(engine) -> None:
    processor = StreamProcessor(engine)
    assert processor.process_batch([]) == []
    stats = processor.stats()
    assert stats.processed == 0
    assert stats.trending_class is None


def test_stream_processor_reset(engine, raw_dataset: pd.DataFrame) -> None:
    processor = StreamProcessor(engine, batch_size=5)
    processor.process_batch(raw_dataset.drop(columns=["label"]).head(10).to_dict(orient="records"))
    processor.reset()
    assert processor.lifetime_stats()["total_processed"] == 0


def test_stream_processor_rejects_invalid_sizes(engine) -> None:
    with pytest.raises(ValidationError):
        StreamProcessor(engine, window_size=0)
    with pytest.raises(ValidationError):
        StreamProcessor(engine, batch_size=0)
