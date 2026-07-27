"""Send real labelled flows to a running detection API and show the verdicts.

Start the API first, then::

    python scripts/demo_detect.py
    python scripts/demo_detect.py --url http://localhost:8000 --per-class 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_or_generate  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.validators import ValidationError  # noqa: E402

log = get_logger(__name__)

REQUEST_TIMEOUT = 15


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise the detection API with labelled flows")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the running API")
    parser.add_argument("--per-class", type=int, default=2, help="Flows to send per traffic class")
    parser.add_argument("--dataset", default="", help="Dataset to sample from")
    return parser.parse_args(argv)


def sample_flows(dataset_path: Path, per_class: int, target: str, seed: int) -> list[tuple[str, dict]]:
    """Take a stratified sample of flows with their true labels."""
    dataset = load_or_generate(dataset_path, target_column=target)
    samples: list[tuple[str, dict]] = []
    for label, group in dataset.groupby(target):
        picked = group.sample(min(per_class, len(group)), random_state=seed)
        samples += [(str(label), row) for row in picked.drop(columns=[target]).to_dict("records")]
    return samples


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = get_config()
    dataset_path = (
        Path(args.dataset) if args.dataset else config.resolve(config.data.raw_path) / "flows.csv"
    )

    try:
        health = requests.get(f"{args.url}/health", timeout=REQUEST_TIMEOUT).json()
    except requests.RequestException as exc:
        log.error("Cannot reach the API at {}: {}", args.url, exc)
        print(f"Start the API first:  uvicorn src.api.main:app --port 8000")
        return 1

    if not health.get("model_loaded"):
        print("The API is running but no model is loaded. Run scripts/run_pipeline.py first.")
        return 1

    print(f"Model: {health['model_name']} v{health['model_version']}  ({args.url})\n")
    header = f"{'TRUE':<8}{'PREDICTED':<12}{'CONFIDENCE':<13}{'SEVERITY':<11}RESULT"
    print(header)
    print("-" * len(header))

    try:
        samples = sample_flows(dataset_path, args.per_class, config.features.target_column, 42)
    except ValidationError as exc:
        log.error("Could not load sample flows: {}", exc)
        return 1

    correct = 0
    for true_label, flow in samples:
        try:
            response = requests.post(f"{args.url}/detect", json=flow, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
        except requests.RequestException as exc:
            log.error("Detection request failed: {}", exc)
            return 1

        body = response.json()
        hit = body["predicted_class"] == true_label
        correct += hit
        print(
            f"{true_label:<8}{body['predicted_class']:<12}{body['confidence']:<13.4f}"
            f"{body.get('severity') or '-':<11}{'correct' if hit else 'MISS'}"
        )

    print("-" * len(header))
    print(f"{correct}/{len(samples)} correct")
    print(f"\nAlerts raised so far: {requests.get(f'{args.url}/detect/stats').json()['alert_severity_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
