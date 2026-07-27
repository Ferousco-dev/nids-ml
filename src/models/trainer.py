"""Training of candidate intrusion detection classifiers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, validate_dataframe

log = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - xgboost also fails when its native library is missing
    XGBOOST_AVAILABLE = False
    log.warning("xgboost is unavailable ({}); the XGBoost candidate will be skipped", exc)

DEFAULT_PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "random_forest": {"n_estimators": [150, 250], "max_depth": [16, 24, None]},
    "xgboost": {"n_estimators": [150, 250], "max_depth": [4, 6], "learning_rate": [0.1, 0.2]},
    "logistic_regression": {"C": [0.5, 1.0, 2.0]},
    "svm": {"C": [0.5, 1.0]},
}


@dataclass
class TrainedModel:
    """A fitted estimator with its training metadata."""

    name: str
    estimator: BaseEstimator
    params: dict[str, Any]
    cv_mean: float
    cv_std: float
    train_seconds: float
    feature_names: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "cv_mean_f1_macro": round(self.cv_mean, 4),
            "cv_std_f1_macro": round(self.cv_std, 4),
            "train_seconds": round(self.train_seconds, 2),
            "n_features": len(self.feature_names),
        }


def build_estimator(name: str, params: dict[str, Any] | None = None) -> BaseEstimator:
    """Instantiate a candidate estimator by name."""
    settings = dict(params or {})
    settings.setdefault("random_state", 42)

    if name == "random_forest":
        settings.setdefault("n_jobs", -1)
        settings.setdefault("class_weight", "balanced_subsample")
        return RandomForestClassifier(**settings)
    if name == "xgboost":
        if not XGBOOST_AVAILABLE:
            raise ValidationError("xgboost is unavailable in this environment")
        settings.setdefault("n_jobs", -1)
        settings.setdefault("tree_method", "hist")
        settings.setdefault("eval_metric", "mlogloss")
        return XGBClassifier(**settings)
    if name == "logistic_regression":
        settings.pop("n_jobs", None)
        settings.setdefault("max_iter", 1000)
        settings.setdefault("class_weight", "balanced")
        return LogisticRegression(**settings)
    if name == "svm":
        settings.pop("n_jobs", None)
        settings.setdefault("max_iter", 5000)
        settings.setdefault("class_weight", "balanced")
        settings.setdefault("dual", "auto")
        return LinearSVC(**settings)
    raise ValidationError(f"Unknown model '{name}'")


class ModelTrainer:
    """Train and compare several classifiers on a shared feature matrix."""

    def __init__(
        self,
        model_configs: dict[str, dict[str, Any]] | None = None,
        cv_folds: int = 5,
        scoring: str = "f1_macro",
        random_state: int = 42,
    ) -> None:
        self.model_configs = model_configs or {
            "random_forest": {"n_estimators": 200, "max_depth": 20},
            "xgboost": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            "logistic_regression": {"max_iter": 1000},
        }
        self.cv_folds = max(2, cv_folds)
        self.scoring = scoring
        self.random_state = random_state
        self.results: dict[str, TrainedModel] = {}

    def _cv(self, y: np.ndarray) -> StratifiedKFold:
        min_class = int(pd.Series(y).value_counts().min())
        folds = min(self.cv_folds, max(2, min_class))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)

    def train_one(
        self,
        name: str,
        features: pd.DataFrame,
        target: Sequence,
        tune: bool = False,
        param_grid: dict[str, list[Any]] | None = None,
    ) -> TrainedModel:
        """Train a single named model, optionally with a grid search."""
        validate_dataframe(features)
        y = np.asarray(target)
        if len(y) != len(features):
            raise ValidationError(f"Feature/target length mismatch: {len(features)} vs {len(y)}")

        x = features.to_numpy(dtype=np.float64)
        estimator = build_estimator(name, self.model_configs.get(name))
        started = time.perf_counter()

        if tune:
            grid = param_grid or DEFAULT_PARAM_GRIDS.get(name, {})
            if grid:
                search = GridSearchCV(
                    estimator, grid, scoring=self.scoring, cv=self._cv(y), n_jobs=-1, refit=True
                )
                search.fit(x, y)
                estimator = search.best_estimator_
                cv_mean, cv_std = float(search.best_score_), float(
                    search.cv_results_["std_test_score"][search.best_index_]
                )
                log.info("{}: grid search selected {}", name, search.best_params_)
            else:
                estimator.fit(x, y)
                cv_mean = cv_std = float("nan")
        else:
            scores = cross_val_score(
                estimator, x, y, scoring=self.scoring, cv=self._cv(y), n_jobs=-1
            )
            cv_mean, cv_std = float(scores.mean()), float(scores.std())
            estimator.fit(x, y)

        trained = TrainedModel(
            name=name,
            estimator=estimator,
            params=estimator.get_params(),
            cv_mean=cv_mean,
            cv_std=cv_std,
            train_seconds=time.perf_counter() - started,
            feature_names=features.columns.tolist(),
        )
        self.results[name] = trained
        log.info(
            "Trained {} in {:.2f}s (cv {}={:.4f})",
            name,
            trained.train_seconds,
            self.scoring,
            cv_mean,
        )
        return trained

    def train_all(
        self,
        features: pd.DataFrame,
        target: Sequence,
        tune: bool = False,
    ) -> dict[str, TrainedModel]:
        """Train every configured model, skipping ones that fail."""
        for name in self.model_configs:
            if name == "xgboost" and not XGBOOST_AVAILABLE:
                continue
            try:
                self.train_one(name, features, target, tune=tune)
            except (ValidationError, ValueError, MemoryError) as exc:
                log.error("Training {} failed: {}", name, exc)
        if not self.results:
            raise ValidationError("No model could be trained successfully")
        return self.results

    def best_model(self) -> TrainedModel:
        """Return the model with the highest cross-validation score."""
        if not self.results:
            raise ValidationError("No models have been trained yet")
        return max(
            self.results.values(),
            key=lambda model: model.cv_mean if np.isfinite(model.cv_mean) else -np.inf,
        )

    def comparison_table(self) -> pd.DataFrame:
        """Return a comparison of all trained models sorted by CV score."""
        if not self.results:
            raise ValidationError("No models have been trained yet")
        rows = [model.summary() for model in self.results.values()]
        return pd.DataFrame(rows).sort_values("cv_mean_f1_macro", ascending=False).reset_index(drop=True)
