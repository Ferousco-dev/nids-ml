"""Tests for training, evaluation and the model registry."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.evaluator import compute_metrics, evaluate_model, predict_proba
from src.models.registry import ModelRegistry
from src.models.trainer import ModelTrainer, build_estimator
from src.utils.validators import ValidationError


@pytest.fixture(scope="module")
def small_training_set(encoded_data):
    features, target = encoded_data
    return features.iloc[:600, :12], np.asarray(target)[:600]


def test_build_estimator_known_and_unknown() -> None:
    assert build_estimator("random_forest") is not None
    with pytest.raises(ValidationError):
        build_estimator("not_a_model")


def test_trainer_trains_and_scores(small_training_set) -> None:
    features, target = small_training_set
    trainer = ModelTrainer(model_configs={"random_forest": {"n_estimators": 20}}, cv_folds=3)
    trained = trainer.train_one("random_forest", features, target)
    assert 0.0 <= trained.cv_mean <= 1.0
    assert trained.feature_names == features.columns.tolist()
    assert trained.summary()["n_features"] == features.shape[1]


def test_trainer_compares_multiple_models(small_training_set) -> None:
    features, target = small_training_set
    trainer = ModelTrainer(
        model_configs={
            "random_forest": {"n_estimators": 20},
            "logistic_regression": {"max_iter": 200},
        },
        cv_folds=3,
    )
    trainer.train_all(features, target)
    table = trainer.comparison_table()
    assert len(table) == len(trainer.results)
    assert trainer.best_model().name in trainer.results


def test_trainer_rejects_mismatched_lengths(small_training_set) -> None:
    features, target = small_training_set
    trainer = ModelTrainer(cv_folds=2)
    with pytest.raises(ValidationError):
        trainer.train_one("random_forest", features, target[:-3])


def test_best_model_requires_training() -> None:
    with pytest.raises(ValidationError):
        ModelTrainer().best_model()


def test_compute_metrics_perfect_prediction() -> None:
    y = np.array([0, 1, 2, 1, 0])
    metrics = compute_metrics(y, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_compute_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValidationError):
        compute_metrics(np.array([0, 1]), np.array([0, 1, 1]))


def test_predict_proba_returns_distribution(model_bundle, encoded_data) -> None:
    features, _ = encoded_data
    probabilities = predict_proba(model_bundle.estimator, features.head(10))
    assert probabilities is not None
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_evaluate_model_writes_plots(tmp_path, model_bundle, encoded_data) -> None:
    features, target = encoded_data
    report = evaluate_model(
        model_bundle.estimator,
        features.head(300),
        np.asarray(target)[:300],
        model_bundle.metadata.class_names,
        model_name="unit_test",
        output_dir=tmp_path,
    )
    assert 0.0 <= report.metrics["accuracy"] <= 1.0
    assert report.confusion.shape == (len(report.class_names), len(report.class_names))
    assert "confusion_matrix" in report.plots
    for path in report.plots.values():
        assert (tmp_path / path.rsplit("/", 1)[-1]).is_file()
    assert "accuracy" in report.summary_text()


def test_registry_save_and_load(tmp_path, model_bundle) -> None:
    registry = ModelRegistry(tmp_path)
    registry.save(
        model_bundle.estimator,
        model_name="random_forest",
        feature_names=model_bundle.metadata.feature_names,
        class_names=model_bundle.metadata.class_names,
        metrics={"f1_macro": 0.97},
        preprocessor=model_bundle.preprocessor,
        as_best=True,
    )
    loaded = registry.load()
    assert loaded.metadata.model_name == "random_forest"
    assert loaded.metadata.metrics["f1_macro"] == pytest.approx(0.97)
    assert len(registry.list_models()) >= 1


def test_registry_rejects_empty_metadata(tmp_path, model_bundle) -> None:
    registry = ModelRegistry(tmp_path)
    with pytest.raises(ValidationError):
        registry.save(model_bundle.estimator, "rf", feature_names=[], class_names=["a"])


def test_registry_load_missing_file(tmp_path) -> None:
    with pytest.raises(ValidationError):
        ModelRegistry(tmp_path).load()
