"""Evaluate a registered model against a labelled dataset.

Usage::

    python scripts/evaluate_model.py --dataset data/raw/flows.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset  # noqa: E402
from src.data.preprocessor import clean_dataset  # noqa: E402
from src.features.engineer import engineer_features  # noqa: E402
from src.models.evaluator import evaluate_model  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.validators import ValidationError, ensure_directory  # noqa: E402

log = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved NIDS model")
    parser.add_argument("--dataset", required=True, help="Labelled CSV/Parquet dataset")
    parser.add_argument("--model", default="", help="Model bundle path (defaults to models/best_model.pkl)")
    parser.add_argument("--output-dir", default="reports", help="Where to write plots and the report")
    parser.add_argument("--sample", type=int, default=0, help="Evaluate on a random sample of N rows")
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    """Load a model bundle and score it on the supplied dataset."""
    config = get_config()
    target = config.features.target_column

    registry = ModelRegistry(config.resolve("models"))
    bundle = registry.load(Path(args.model) if args.model else None)
    preprocessor = bundle.preprocessor
    if preprocessor is None:
        raise ValidationError("The model bundle has no preprocessor; retrain with run_pipeline.py")

    dataset = load_dataset(args.dataset, target_column=target)
    if args.sample and args.sample < len(dataset):
        dataset = dataset.sample(args.sample, random_state=config.data.random_state)

    cleaned = clean_dataset(dataset, target_column=target)
    engineered = engineer_features(cleaned)
    features = preprocessor.transform_features(engineered)
    missing = [name for name in bundle.metadata.feature_names if name not in features.columns]
    if missing:
        raise ValidationError(f"Dataset is missing model feature(s): {', '.join(missing[:10])}")

    y_true = preprocessor.transform_target(cleaned[target])
    report = evaluate_model(
        bundle.estimator,
        features[bundle.metadata.feature_names],
        y_true,
        bundle.metadata.class_names,
        model_name=f"{bundle.metadata.model_name}_eval",
        output_dir=ensure_directory(args.output_dir),
    )

    payload = report.to_dict()
    payload["dataset"] = str(args.dataset)
    payload["model_version"] = bundle.metadata.version
    with (Path(args.output_dir) / "evaluation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(report.summary_text())
    print("\nPer-class results")
    print(pd.DataFrame(report.report).transpose().round(4).to_string())
    print(f"\nArtefacts written to {Path(args.output_dir).resolve()}")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evaluate(args)
    except ValidationError as exc:
        log.error("Evaluation failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
