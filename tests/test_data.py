"""Tests for dataset loading, preprocessing and splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    ATTACK_CLASSES,
    generate_synthetic_dataset,
    load_dataset,
    load_or_generate,
    save_dataset,
)
from src.data.preprocessor import FlowPreprocessor, clean_dataset
from src.data.splitter import class_distribution, split_dataset
from src.utils.validators import ValidationError


def test_synthetic_dataset_shape_and_classes(raw_dataset: pd.DataFrame) -> None:
    assert len(raw_dataset) == 1500
    assert "label" in raw_dataset.columns
    assert set(raw_dataset["label"]).issubset(set(ATTACK_CLASSES))
    assert raw_dataset.isna().sum().sum() == 0


def test_synthetic_dataset_is_reproducible() -> None:
    first = generate_synthetic_dataset(n_rows=200, random_state=11)
    second = generate_synthetic_dataset(n_rows=200, random_state=11)
    pd.testing.assert_frame_equal(first, second)


def test_binary_labels_option() -> None:
    dataset = generate_synthetic_dataset(n_rows=300, random_state=3, binary_labels=True)
    assert set(dataset["label"]) == {"normal", "attack"}


def test_generate_rejects_tiny_row_counts() -> None:
    with pytest.raises(ValidationError):
        generate_synthetic_dataset(n_rows=2)


@pytest.mark.parametrize("suffix", [".csv", ".parquet"])
def test_save_and_load_roundtrip(tmp_path, raw_dataset: pd.DataFrame, suffix: str) -> None:
    path = save_dataset(raw_dataset.head(100), tmp_path / f"flows{suffix}")
    loaded = load_dataset(path)
    assert len(loaded) == 100
    assert list(loaded.columns) == list(raw_dataset.columns)


def test_load_dataset_rejects_unknown_suffix(tmp_path) -> None:
    path = tmp_path / "flows.txt"
    path.write_text("not a dataset", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_dataset(path)


def test_load_or_generate_creates_missing_file(tmp_path) -> None:
    target = tmp_path / "generated.csv"
    dataset = load_or_generate(target, n_rows=300, random_state=5)
    assert target.is_file()
    assert len(dataset) == 300


def test_clean_dataset_drops_duplicates_and_constants(raw_dataset: pd.DataFrame) -> None:
    noisy = pd.concat([raw_dataset.head(50), raw_dataset.head(50)], ignore_index=True)
    noisy["constant_column"] = 1
    cleaned = clean_dataset(noisy)
    assert "constant_column" not in cleaned.columns
    assert len(cleaned) <= 50


def test_preprocessor_encodes_and_scales(engineered_dataset: pd.DataFrame) -> None:
    preprocessor = FlowPreprocessor()
    features, target = preprocessor.fit_transform(engineered_dataset)
    assert features.shape[0] == len(engineered_dataset)
    assert features.select_dtypes(include=[np.number]).shape[1] == features.shape[1]
    assert set(np.unique(target)).issubset(set(range(len(preprocessor.class_names))))
    assert abs(features.mean().mean()) < 1e-6


def test_preprocessor_handles_unseen_categories(
    fitted_preprocessor: FlowPreprocessor, engineered_dataset: pd.DataFrame
) -> None:
    frame = engineered_dataset.head(10).copy()
    frame["service"] = "totally_new_service"
    transformed = fitted_preprocessor.transform_features(frame)
    assert list(transformed.columns) == fitted_preprocessor.feature_names


def test_preprocessor_requires_fitting(engineered_dataset: pd.DataFrame) -> None:
    with pytest.raises(ValidationError):
        FlowPreprocessor().transform_features(engineered_dataset)


def test_preprocessor_save_load_roundtrip(
    tmp_path, fitted_preprocessor: FlowPreprocessor, engineered_dataset: pd.DataFrame
) -> None:
    path = fitted_preprocessor.save(tmp_path / "preprocessor.pkl")
    restored = FlowPreprocessor.load(path)
    pd.testing.assert_frame_equal(
        restored.transform_features(engineered_dataset.head(20)),
        fitted_preprocessor.transform_features(engineered_dataset.head(20)),
    )


def test_preprocessor_rejects_unseen_labels(fitted_preprocessor: FlowPreprocessor) -> None:
    with pytest.raises(ValidationError):
        fitted_preprocessor.transform_target(pd.Series(["brand_new_class"]))


def test_split_is_stratified(encoded_data) -> None:
    features, target = encoded_data
    split = split_dataset(features, target, test_size=0.2, val_size=0.2, random_state=1)
    assert sum(split.sizes.values()) == len(features)

    overall = class_distribution(target)
    train = class_distribution(split.y_train)
    assert np.allclose(overall.to_numpy(), train.reindex(overall.index).to_numpy(), atol=0.05)


def test_split_rejects_invalid_ratios(encoded_data) -> None:
    features, target = encoded_data
    with pytest.raises(ValidationError):
        split_dataset(features, target, test_size=0.6, val_size=0.5)


def test_split_rejects_length_mismatch(encoded_data) -> None:
    features, target = encoded_data
    with pytest.raises(ValidationError):
        split_dataset(features, target[:-5])
