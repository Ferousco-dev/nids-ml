"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import generate_synthetic_dataset
from src.data.preprocessor import FlowPreprocessor, clean_dataset
from src.detection.engine import DetectionEngine
from src.features.engineer import engineer_features
from src.models.registry import ModelBundle, ModelMetadata


@pytest.fixture(scope="session")
def raw_dataset() -> pd.DataFrame:
    """A small synthetic dataset covering every attack class."""
    return generate_synthetic_dataset(n_rows=1500, random_state=7)


@pytest.fixture(scope="session")
def engineered_dataset(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(clean_dataset(raw_dataset))


@pytest.fixture(scope="session")
def fitted_preprocessor(engineered_dataset: pd.DataFrame) -> FlowPreprocessor:
    preprocessor = FlowPreprocessor()
    preprocessor.fit(engineered_dataset)
    return preprocessor


@pytest.fixture(scope="session")
def encoded_data(
    fitted_preprocessor: FlowPreprocessor, engineered_dataset: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    features = fitted_preprocessor.transform_features(engineered_dataset)
    target = fitted_preprocessor.transform_target(engineered_dataset["label"])
    return features, target


@pytest.fixture(scope="session")
def model_bundle(
    encoded_data: tuple[pd.DataFrame, pd.Series], fitted_preprocessor: FlowPreprocessor
) -> ModelBundle:
    """A lightweight trained bundle usable by the detection engine."""
    features, target = encoded_data
    estimator = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0, n_jobs=-1)
    estimator.fit(features.to_numpy(), target)
    metadata = ModelMetadata(
        model_name="random_forest",
        version="test",
        created_at="2026-01-01T00:00:00+00:00",
        feature_names=features.columns.tolist(),
        class_names=fitted_preprocessor.class_names,
    )
    return ModelBundle(estimator=estimator, metadata=metadata, preprocessor=fitted_preprocessor)


@pytest.fixture()
def engine(model_bundle: ModelBundle) -> DetectionEngine:
    return DetectionEngine(model_bundle, confidence_threshold=0.5)


@pytest.fixture()
def sample_flow(raw_dataset: pd.DataFrame) -> dict:
    """One raw attack flow as a plain dictionary."""
    attacks = raw_dataset[raw_dataset["label"] != "normal"]
    return attacks.drop(columns=["label"]).iloc[0].to_dict()
