"""Fetch a public intrusion detection dataset, or generate a synthetic one.

Usage::

    python scripts/download_data.py --rows 50000
    python scripts/download_data.py --url https://example.org/flows.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import generate_synthetic_dataset, save_dataset  # noqa: E402
from src.utils.config import get_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.validators import ValidationError, ensure_directory  # noqa: E402

log = get_logger(__name__)

DOWNLOAD_TIMEOUT = 60
CHUNK_SIZE = 1 << 16


def download(url: str, destination: Path) -> Path:
    """Stream a remote CSV to disk."""
    ensure_directory(destination.parent)
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    handle.write(chunk)
    except (requests.RequestException, OSError) as exc:
        raise ValidationError(f"Download from {url} failed: {exc}") from exc
    log.info("Downloaded {} to {}", url, destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the raw NIDS dataset")
    parser.add_argument("--url", default="", help="Optional URL of a CSV capture to download")
    parser.add_argument("--rows", type=int, default=50_000, help="Rows to synthesise when no URL is given")
    parser.add_argument("--output", default="", help="Destination file (defaults to data/raw/flows.csv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic generation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = get_config()
    destination = (
        Path(args.output)
        if args.output
        else config.resolve(config.data.raw_path) / "flows.csv"
    )

    try:
        if args.url:
            download(args.url, destination)
        else:
            log.info("No URL supplied; generating {} synthetic flows", args.rows)
            dataset = generate_synthetic_dataset(n_rows=args.rows, random_state=args.seed)
            save_dataset(dataset, destination)
    except ValidationError as exc:
        log.error("Dataset preparation failed: {}", exc)
        return 1

    print(f"Dataset ready at {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
