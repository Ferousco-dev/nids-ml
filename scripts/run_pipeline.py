"""End-to-end training pipeline: load, preprocess, train, evaluate, register.

Usage::

    python scripts/run_pipeline.py --rows 50000 --tune
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_or_generate  # noqa: E402
from src.data.preprocessor import FlowPreprocessor, clean_dataset  # noqa: E402
from src.data.splitter import split_dataset  # noqa: E402
from src.features.engineer import engineer_features  # noqa: E402
from src.features.selector import save_selection, select_features  # noqa: E402
from src.models.evaluator import evaluate_model  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.models.trainer import ModelTrainer  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.validators import ValidationError, ensure_directory  # noqa: E402

log = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and register the NIDS model")
    parser.add_argument("--dataset", default="", help="Path to a CSV/Parquet capture")
    parser.add_argument("--rows", type=int, default=50_000, help="Synthetic rows when no dataset exists")
    parser.add_argument("--top-k", type=int, default=0, help="Override the number of selected features")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter grid search")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--skip-selection", action="store_true", help="Train on all engineered features")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the full pipeline and return a summary payload."""
    config = get_config()
    target = config.features.target_column
    raw_dir = config.resolve(config.data.raw_path)
    processed_dir = ensure_directory(config.resolve(config.data.processed_path))
    reports_dir = ensure_directory(config.resolve("reports"))

    dataset_path = Path(args.dataset) if args.dataset else raw_dir / "flows.csv"
    dataset = load_or_generate(
        dataset_path, n_rows=args.rows, random_state=config.data.random_state, target_column=target
    )

    cleaned = clean_dataset(dataset, target_column=target)
    engineered = engineer_features(cleaned)

    preprocessor = FlowPreprocessor(
        target_column=target, scaler="standard", drop_columns=config.features.drop_columns
    )
    features, encoded_target = preprocessor.fit_transform(engineered)

    if args.skip_selection:
        selected = features.columns.tolist()
    else:
        selection = select_features(
            features,
            encoded_target,
            method=config.features.selection_method,
            top_k=args.top_k or config.features.top_k,
            random_state=config.data.random_state,
        )
        save_selection(selection, processed_dir / "selected_features.json")
        selected = selection.selected
    features = features[selected]

    split = split_dataset(
        features,
        encoded_target,
        test_size=config.data.test_size,
        val_size=config.data.val_size,
        random_state=config.data.random_state,
    )

    trainer = ModelTrainer(
        model_configs=config.models, cv_folds=args.cv_folds, random_state=config.data.random_state
    )
    trainer.train_all(split.x_train, split.y_train, tune=args.tune)
    comparison = trainer.comparison_table()

    class_names = preprocessor.class_names
    validation = {
        name: evaluate_model(
            model.estimator, split.x_val, split.y_val, class_names, model_name=f"{name}_val"
        )
        for name, model in trainer.results.items()
    }
    best_name = max(validation, key=lambda name: validation[name].metrics["f1_macro"])
    best = trainer.results[best_name]
    log.info("Best model on validation: {}", best_name)

    test_report = evaluate_model(
        best.estimator,
        split.x_test,
        split.y_test,
        class_names,
        model_name=best_name,
        output_dir=reports_dir,
    )

    registry = ModelRegistry(config.resolve("models"))
    registry.save(
        best.estimator,
        model_name=best_name,
        feature_names=selected,
        class_names=class_names,
        metrics=test_report.metrics,
        preprocessor=preprocessor,
        as_best=True,
    )
    preprocessor.save(processed_dir / "preprocessor.pkl")

    summary = {
        "dataset": str(dataset_path),
        "rows": int(len(cleaned)),
        "features_selected": len(selected),
        "classes": class_names,
        "split_sizes": split.sizes,
        "model_comparison": comparison.to_dict(orient="records"),
        "best_model": best_name,
        "validation_f1_macro": {
            name: round(report.metrics["f1_macro"], 4) for name, report in validation.items()
        },
        "test_metrics": test_report.metrics,
        "plots": test_report.plots,
    }
    with (reports_dir / "pipeline_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    """Print a readable end-of-run report."""
    print("\n" + "=" * 62)
    print("NIDS-ML TRAINING PIPELINE SUMMARY")
    print("=" * 62)
    print(f"Dataset            : {summary['dataset']}")
    print(f"Rows after cleaning: {summary['rows']:,}")
    print(f"Selected features  : {summary['features_selected']}")
    print(f"Classes            : {', '.join(summary['classes'])}")
    print(f"Split sizes        : {summary['split_sizes']}")
    print("-" * 62)
    print("Model comparison (cross-validated F1 macro):")
    for row in summary["model_comparison"]:
        print(f"  {row['name']:<22}{row['cv_mean_f1_macro']:.4f} (+/- {row['cv_std_f1_macro']:.4f})")
    print("-" * 62)
    print(f"Best model         : {summary['best_model']}")
    print("Held-out test metrics:")
    for name, value in summary["test_metrics"].items():
        print(f"  {name:<22}{value:.4f}")
    print("=" * 62 + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except ValidationError as exc:
        log.error("Pipeline failed: {}", exc)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
