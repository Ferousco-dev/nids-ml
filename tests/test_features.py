"""Tests for feature engineering and selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import add_traffic_rates, engineer_features
from src.features.selector import (
    drop_correlated,
    load_selection,
    save_selection,
    select_features,
)
from src.utils.validators import ValidationError


def test_engineering_adds_expected_columns(raw_dataset: pd.DataFrame) -> None:
    engineered = engineer_features(raw_dataset.head(200))
    for column in ("src_byte_rate", "byte_ratio", "duration_band", "packet_rate", "privilege_activity"):
        assert column in engineered.columns


def test_engineering_is_finite(raw_dataset: pd.DataFrame) -> None:
    engineered = engineer_features(raw_dataset.head(200))
    numeric = engineered.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


def test_engineering_tolerates_missing_source_columns() -> None:
    frame = pd.DataFrame({"duration": [1.0, 2.0], "count": [3, 4]})
    engineered = engineer_features(frame)
    assert "packet_rate" in engineered.columns
    assert "byte_ratio" not in engineered.columns


def test_traffic_rates_avoid_division_by_zero() -> None:
    frame = pd.DataFrame({"duration": [0.0], "src_bytes": [100], "dst_bytes": [0]})
    result = add_traffic_rates(frame)
    assert np.isfinite(result["src_byte_rate"]).all()
    assert np.isfinite(result["byte_ratio"]).all()


def test_engineering_rejects_non_dataframe() -> None:
    with pytest.raises(ValidationError):
        engineer_features([1, 2, 3])


def test_drop_correlated_removes_duplicate_column() -> None:
    base = pd.Series(np.linspace(0, 10, 100))
    frame = pd.DataFrame({"a": base, "b": base * 2.0, "c": np.sin(base)})
    remaining = drop_correlated(frame, threshold=0.95)
    assert len(remaining) < 3
    assert "c" in remaining


@pytest.mark.parametrize("method", ["mutual_info", "correlation", "importance"])
def test_select_features_returns_requested_count(encoded_data, method: str) -> None:
    features, target = encoded_data
    result = select_features(features, target, method=method, top_k=10, random_state=0)
    assert len(result.selected) == 10
    assert set(result.selected).issubset(set(features.columns))
    assert result.method == method


def test_select_features_caps_at_available_columns(encoded_data) -> None:
    features, target = encoded_data
    result = select_features(features.iloc[:, :5], target, top_k=50, random_state=0)
    assert 0 < len(result.selected) <= 5


def test_select_features_rejects_bad_arguments(encoded_data) -> None:
    features, target = encoded_data
    with pytest.raises(ValidationError):
        select_features(features, target, top_k=0)
    with pytest.raises(ValidationError):
        select_features(features, target[:-1])
    with pytest.raises(ValidationError):
        select_features(features, target, method="nonexistent")


def test_selection_roundtrip(tmp_path, encoded_data) -> None:
    features, target = encoded_data
    result = select_features(features, target, top_k=8, random_state=0)
    path = save_selection(result, tmp_path / "selection.json")
    restored = load_selection(path)
    assert restored.selected == result.selected
    assert restored.method == result.method


def test_load_selection_missing_file(tmp_path) -> None:
    with pytest.raises(ValidationError):
        load_selection(tmp_path / "absent.json")
