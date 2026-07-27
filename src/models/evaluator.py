"""Model evaluation metrics and diagnostic plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, ensure_directory

log = get_logger(__name__)

PLOT_STYLE = {"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 10}


@dataclass
class EvaluationReport:
    """Metrics and artefacts produced by a single evaluation run."""

    model_name: str
    metrics: dict[str, float]
    confusion: np.ndarray
    report: dict[str, Any]
    class_names: list[str]
    plots: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": self.metrics,
            "confusion_matrix": self.confusion.tolist(),
            "classification_report": self.report,
            "class_names": self.class_names,
            "plots": self.plots,
        }

    def summary_text(self) -> str:
        """Render a human-readable metric summary."""
        lines = [f"Model: {self.model_name}", "-" * 46]
        lines += [f"{name:<28}{value:.4f}" for name, value in self.metrics.items()]
        return "\n".join(lines)


def predict_proba(estimator: BaseEstimator, features: pd.DataFrame) -> np.ndarray | None:
    """Return class probabilities when the estimator supports them."""
    x = features.to_numpy(dtype=np.float64)
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(x))
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        exponent = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exponent / exponent.sum(axis=1, keepdims=True)
    return None


def _auc(y_true: np.ndarray, probabilities: np.ndarray | None, n_classes: int) -> float:
    if probabilities is None:
        return float("nan")
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, probabilities[:, 1]))
        return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro"))
    except ValueError as exc:
        log.warning("AUC-ROC could not be computed: {}", exc)
        return float("nan")


def compute_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the headline classification metrics."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValidationError(f"Shape mismatch: y_true {true.shape} vs y_pred {pred.shape}")

    return {
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(true, pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "auc_roc": _auc(true, probabilities, len(np.unique(true))),
    }


def plot_confusion_matrix(
    confusion: np.ndarray, class_names: Sequence[str], path: Path | str, title: str
) -> Path:
    """Save a normalised confusion matrix heatmap."""
    file_path = Path(path)
    ensure_directory(file_path.parent)
    totals = confusion.sum(axis=1, keepdims=True)
    normalised = np.divide(confusion, totals, out=np.zeros_like(confusion, dtype=float), where=totals > 0)

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(1.4 * len(class_names) + 3, 1.2 * len(class_names) + 2.5))
        image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        ax.set_yticks(range(len(class_names)), class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for row in range(confusion.shape[0]):
            for column in range(confusion.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{confusion[row, column]:,}\n{normalised[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if normalised[row, column] > 0.5 else "black",
                )
        fig.colorbar(image, ax=ax, label="Row-normalised rate")
        fig.savefig(file_path)
        plt.close(fig)
    return file_path


def plot_roc_curves(
    y_true: np.ndarray, probabilities: np.ndarray, class_names: Sequence[str], path: Path | str
) -> Path:
    """Save one-vs-rest ROC curves for every class."""
    file_path = Path(path)
    ensure_directory(file_path.parent)
    classes = np.arange(len(class_names))
    binarised = label_binarize(y_true, classes=classes)
    if binarised.shape[1] == 1:
        binarised = np.column_stack([1 - binarised, binarised])

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        for index, name in enumerate(class_names):
            if binarised[:, index].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binarised[:, index], probabilities[:, index])
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})", linewidth=1.6)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("One-vs-rest ROC curves")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
        fig.savefig(file_path)
        plt.close(fig)
    return file_path


def plot_feature_importance(
    estimator: BaseEstimator, feature_names: Sequence[str], path: Path | str, top_n: int = 20
) -> Path | None:
    """Save a bar chart of the most influential features, when available."""
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            log.info("Estimator exposes no feature importances; skipping plot")
            return None
        importances = np.abs(np.asarray(coefficients)).mean(axis=0)

    file_path = Path(path)
    ensure_directory(file_path.parent)
    ranked = (
        pd.Series(np.asarray(importances), index=list(feature_names))
        .sort_values(ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(ranked) + 1.8))
        ax.barh(ranked.index, ranked.to_numpy(), color="#2c6fbb")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {len(ranked)} features")
        ax.grid(axis="x", alpha=0.3)
        fig.savefig(file_path)
        plt.close(fig)
    return file_path


def evaluate_model(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    y_true: Sequence,
    class_names: Sequence[str],
    model_name: str = "model",
    output_dir: Path | str | None = None,
) -> EvaluationReport:
    """Evaluate an estimator and optionally write diagnostic plots."""
    true = np.asarray(y_true)
    if len(true) != len(features):
        raise ValidationError(f"Feature/target length mismatch: {len(features)} vs {len(true)}")

    predictions = np.asarray(estimator.predict(features.to_numpy(dtype=np.float64)))
    probabilities = predict_proba(estimator, features)
    metrics = compute_metrics(true, predictions, probabilities)
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(true, predictions, labels=labels)
    report = classification_report(
        true,
        predictions,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )

    plots: dict[str, str] = {}
    if output_dir is not None:
        directory = ensure_directory(output_dir)
        plots["confusion_matrix"] = str(
            plot_confusion_matrix(
                matrix, class_names, directory / f"{model_name}_confusion_matrix.png",
                f"Confusion matrix - {model_name}",
            )
        )
        if probabilities is not None:
            plots["roc_curves"] = str(
                plot_roc_curves(true, probabilities, class_names, directory / f"{model_name}_roc.png")
            )
        importance_path = plot_feature_importance(
            estimator, features.columns, directory / f"{model_name}_feature_importance.png"
        )
        if importance_path is not None:
            plots["feature_importance"] = str(importance_path)

    log.info(
        "{} evaluated: accuracy={:.4f} f1_macro={:.4f}",
        model_name,
        metrics["accuracy"],
        metrics["f1_macro"],
    )
    return EvaluationReport(
        model_name=model_name,
        metrics=metrics,
        confusion=matrix,
        report=report,
        class_names=list(class_names),
        plots=plots,
    )
