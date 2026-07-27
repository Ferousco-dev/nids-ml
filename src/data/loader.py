"""Dataset loading and synthetic traffic generation.

Real captures (CSV or Parquet exports of KDD/NSL-KDD/CICIDS style flows) are
loaded directly. When no capture is available, :func:`generate_synthetic_dataset`
produces a labelled dataset with the same schema and class-conditional
distributions that mirror the behaviour of each attack family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, ensure_directory, validate_path

log = get_logger(__name__)

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

NUMERIC_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

FEATURE_COLUMNS = ["duration"] + CATEGORICAL_FEATURES + NUMERIC_FEATURES[1:]

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = [
    "http", "https", "smtp", "ftp", "ftp_data", "ssh", "telnet", "domain_u",
    "ecr_i", "eco_i", "private", "pop_3", "finger", "auth", "other",
]
FLAGS = ["SF", "S0", "REJ", "RSTO", "RSTR", "S1", "S2", "S3", "SH", "OTH"]

ATTACK_CLASSES = ["normal", "dos", "probe", "r2l", "u2r"]
CLASS_WEIGHTS = {"normal": 0.53, "dos": 0.30, "probe": 0.11, "r2l": 0.05, "u2r": 0.01}


def _rates(rng: np.random.Generator, size: int, low: float, high: float) -> np.ndarray:
    return np.clip(rng.uniform(low, high, size) + rng.normal(0, 0.05, size), 0.0, 1.0)


def _base_frame(rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    zeros = np.zeros(size, dtype=np.int64)
    return {
        "land": zeros.copy(),
        "wrong_fragment": zeros.copy(),
        "urgent": zeros.copy(),
        "hot": zeros.copy(),
        "num_failed_logins": zeros.copy(),
        "logged_in": zeros.copy(),
        "num_compromised": zeros.copy(),
        "root_shell": zeros.copy(),
        "su_attempted": zeros.copy(),
        "num_root": zeros.copy(),
        "num_file_creations": zeros.copy(),
        "num_shells": zeros.copy(),
        "num_access_files": zeros.copy(),
        "num_outbound_cmds": zeros.copy(),
        "is_host_login": zeros.copy(),
        "is_guest_login": zeros.copy(),
    }


def _normal(rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    data = _base_frame(rng, size)
    data.update(
        duration=rng.gamma(2.0, 12.0, size).round(2),
        protocol_type=rng.choice(PROTOCOLS, size, p=[0.78, 0.19, 0.03]),
        service=rng.choice(SERVICES, size, p=_service_profile("normal")),
        flag=rng.choice(FLAGS, size, p=_flag_profile("normal")),
        src_bytes=rng.lognormal(5.6, 1.3, size).astype(np.int64),
        dst_bytes=rng.lognormal(6.4, 1.6, size).astype(np.int64),
        logged_in=rng.binomial(1, 0.72, size),
        hot=rng.poisson(0.2, size),
        count=rng.poisson(9, size) + 1,
        srv_count=rng.poisson(8, size) + 1,
        serror_rate=_rates(rng, size, 0.0, 0.04),
        srv_serror_rate=_rates(rng, size, 0.0, 0.04),
        rerror_rate=_rates(rng, size, 0.0, 0.05),
        srv_rerror_rate=_rates(rng, size, 0.0, 0.05),
        same_srv_rate=_rates(rng, size, 0.85, 1.0),
        diff_srv_rate=_rates(rng, size, 0.0, 0.1),
        srv_diff_host_rate=_rates(rng, size, 0.0, 0.15),
        dst_host_count=rng.integers(20, 255, size),
        dst_host_srv_count=rng.integers(20, 255, size),
        dst_host_same_srv_rate=_rates(rng, size, 0.8, 1.0),
        dst_host_diff_srv_rate=_rates(rng, size, 0.0, 0.12),
        dst_host_same_src_port_rate=_rates(rng, size, 0.0, 0.25),
        dst_host_srv_diff_host_rate=_rates(rng, size, 0.0, 0.12),
        dst_host_serror_rate=_rates(rng, size, 0.0, 0.05),
        dst_host_srv_serror_rate=_rates(rng, size, 0.0, 0.05),
        dst_host_rerror_rate=_rates(rng, size, 0.0, 0.06),
        dst_host_srv_rerror_rate=_rates(rng, size, 0.0, 0.06),
    )
    return data


def _dos(rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    data = _base_frame(rng, size)
    data.update(
        duration=rng.exponential(0.4, size).round(2),
        protocol_type=rng.choice(PROTOCOLS, size, p=[0.6, 0.08, 0.32]),
        service=rng.choice(SERVICES, size, p=_service_profile("dos")),
        flag=rng.choice(FLAGS, size, p=_flag_profile("dos")),
        src_bytes=rng.lognormal(3.2, 1.0, size).astype(np.int64),
        dst_bytes=np.zeros(size, dtype=np.int64),
        wrong_fragment=rng.binomial(1, 0.12, size) * rng.integers(1, 4, size),
        land=rng.binomial(1, 0.01, size),
        count=rng.integers(150, 511, size),
        srv_count=rng.integers(120, 511, size),
        serror_rate=_rates(rng, size, 0.75, 1.0),
        srv_serror_rate=_rates(rng, size, 0.75, 1.0),
        rerror_rate=_rates(rng, size, 0.0, 0.1),
        srv_rerror_rate=_rates(rng, size, 0.0, 0.1),
        same_srv_rate=_rates(rng, size, 0.9, 1.0),
        diff_srv_rate=_rates(rng, size, 0.0, 0.06),
        srv_diff_host_rate=_rates(rng, size, 0.0, 0.08),
        dst_host_count=np.full(size, 255),
        dst_host_srv_count=rng.integers(200, 255, size),
        dst_host_same_srv_rate=_rates(rng, size, 0.9, 1.0),
        dst_host_diff_srv_rate=_rates(rng, size, 0.0, 0.05),
        dst_host_same_src_port_rate=_rates(rng, size, 0.6, 1.0),
        dst_host_srv_diff_host_rate=_rates(rng, size, 0.0, 0.05),
        dst_host_serror_rate=_rates(rng, size, 0.8, 1.0),
        dst_host_srv_serror_rate=_rates(rng, size, 0.8, 1.0),
        dst_host_rerror_rate=_rates(rng, size, 0.0, 0.08),
        dst_host_srv_rerror_rate=_rates(rng, size, 0.0, 0.08),
    )
    return data


def _probe(rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    data = _base_frame(rng, size)
    data.update(
        duration=rng.exponential(1.5, size).round(2),
        protocol_type=rng.choice(PROTOCOLS, size, p=[0.55, 0.15, 0.30]),
        service=rng.choice(SERVICES, size, p=_service_profile("probe")),
        flag=rng.choice(FLAGS, size, p=_flag_profile("probe")),
        src_bytes=rng.lognormal(2.6, 1.1, size).astype(np.int64),
        dst_bytes=rng.lognormal(2.0, 1.4, size).astype(np.int64),
        count=rng.integers(40, 300, size),
        srv_count=rng.integers(1, 25, size),
        serror_rate=_rates(rng, size, 0.1, 0.5),
        srv_serror_rate=_rates(rng, size, 0.1, 0.5),
        rerror_rate=_rates(rng, size, 0.45, 0.95),
        srv_rerror_rate=_rates(rng, size, 0.45, 0.95),
        same_srv_rate=_rates(rng, size, 0.0, 0.2),
        diff_srv_rate=_rates(rng, size, 0.6, 1.0),
        srv_diff_host_rate=_rates(rng, size, 0.5, 1.0),
        dst_host_count=rng.integers(120, 255, size),
        dst_host_srv_count=rng.integers(1, 40, size),
        dst_host_same_srv_rate=_rates(rng, size, 0.0, 0.2),
        dst_host_diff_srv_rate=_rates(rng, size, 0.55, 1.0),
        dst_host_same_src_port_rate=_rates(rng, size, 0.0, 0.2),
        dst_host_srv_diff_host_rate=_rates(rng, size, 0.3, 0.9),
        dst_host_serror_rate=_rates(rng, size, 0.1, 0.5),
        dst_host_srv_serror_rate=_rates(rng, size, 0.1, 0.5),
        dst_host_rerror_rate=_rates(rng, size, 0.5, 1.0),
        dst_host_srv_rerror_rate=_rates(rng, size, 0.5, 1.0),
    )
    return data


def _r2l(rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    data = _base_frame(rng, size)
    data.update(
        duration=rng.gamma(3.0, 25.0, size).round(2),
        protocol_type=rng.choice(PROTOCOLS, size, p=[0.94, 0.05, 0.01]),
        service=rng.choice(SERVICES, size, p=_service_profile("r2l")),
        flag=rng.choice(FLAGS, size, p=_flag_profile("r2l")),
        src_bytes=rng.lognormal(5.0, 1.4, size).astype(np.int64),
        dst_bytes=rng.lognormal(5.4, 1.5, size).astype(np.int64),
        hot=rng.poisson(2.6, size),
        num_failed_logins=rng.poisson(1.8, size),
        logged_in=rng.binomial(1, 0.55, size),
        is_guest_login=rng.binomial(1, 0.45, size),
        num_access_files=rng.poisson(0.9, size),
        num_file_creations=rng.poisson(0.6, size),
        count=rng.poisson(6, size) + 1,
        srv_count=rng.poisson(5, size) + 1,
        serror_rate=_rates(rng, size, 0.0, 0.1),
        srv_serror_rate=_rates(rng, size, 0.0, 0.1),
        rerror_rate=_rates(rng, size, 0.0, 0.2),
        srv_rerror_rate=_rates(rng, size, 0.0, 0.2),
        same_srv_rate=_rates(rng, size, 0.7, 1.0),
        diff_srv_rate=_rates(rng, size, 0.0, 0.2),
        srv_diff_host_rate=_rates(rng, size, 0.0, 0.3),
        dst_host_count=rng.integers(1, 120, size),
        dst_host_srv_count=rng.integers(1, 60, size),
        dst_host_same_srv_rate=_rates(rng, size, 0.5, 1.0),
        dst_host_diff_srv_rate=_rates(rng, size, 0.0, 0.3),
        dst_host_same_src_port_rate=_rates(rng, size, 0.3, 0.9),
        dst_host_srv_diff_host_rate=_rates(rng, size, 0.0, 0.3),
        dst_host_serror_rate=_rates(rng, size, 0.0, 0.12),
        dst_host_srv_serror_rate=_rates(rng, size, 0.0, 0.12),
        dst_host_rerror_rate=_rates(rng, size, 0.0, 0.25),
        dst_host_srv_rerror_rate=_rates(rng, size, 0.0, 0.25),
    )
    return data


def _u2r(rng: np.random.Generator, size: int) -> dict[str, np.ndarray]:
    data = _base_frame(rng, size)
    data.update(
        duration=rng.gamma(4.0, 60.0, size).round(2),
        protocol_type=rng.choice(PROTOCOLS, size, p=[0.97, 0.02, 0.01]),
        service=rng.choice(SERVICES, size, p=_service_profile("u2r")),
        flag=rng.choice(FLAGS, size, p=_flag_profile("u2r")),
        src_bytes=rng.lognormal(6.2, 1.1, size).astype(np.int64),
        dst_bytes=rng.lognormal(7.1, 1.2, size).astype(np.int64),
        hot=rng.poisson(9.0, size),
        logged_in=np.ones(size, dtype=np.int64),
        root_shell=rng.binomial(1, 0.75, size),
        su_attempted=rng.binomial(1, 0.45, size),
        num_root=rng.poisson(6.0, size),
        num_compromised=rng.poisson(3.0, size),
        num_file_creations=rng.poisson(4.0, size),
        num_shells=rng.poisson(1.4, size),
        num_access_files=rng.poisson(2.0, size),
        count=rng.poisson(3, size) + 1,
        srv_count=rng.poisson(3, size) + 1,
        serror_rate=_rates(rng, size, 0.0, 0.06),
        srv_serror_rate=_rates(rng, size, 0.0, 0.06),
        rerror_rate=_rates(rng, size, 0.0, 0.08),
        srv_rerror_rate=_rates(rng, size, 0.0, 0.08),
        same_srv_rate=_rates(rng, size, 0.85, 1.0),
        diff_srv_rate=_rates(rng, size, 0.0, 0.1),
        srv_diff_host_rate=_rates(rng, size, 0.0, 0.1),
        dst_host_count=rng.integers(1, 80, size),
        dst_host_srv_count=rng.integers(1, 40, size),
        dst_host_same_srv_rate=_rates(rng, size, 0.7, 1.0),
        dst_host_diff_srv_rate=_rates(rng, size, 0.0, 0.2),
        dst_host_same_src_port_rate=_rates(rng, size, 0.4, 1.0),
        dst_host_srv_diff_host_rate=_rates(rng, size, 0.0, 0.2),
        dst_host_serror_rate=_rates(rng, size, 0.0, 0.08),
        dst_host_srv_serror_rate=_rates(rng, size, 0.0, 0.08),
        dst_host_rerror_rate=_rates(rng, size, 0.0, 0.1),
        dst_host_srv_rerror_rate=_rates(rng, size, 0.0, 0.1),
    )
    return data


_SERVICE_PROFILES: dict[str, dict[str, float]] = {
    "normal": {"http": 0.34, "https": 0.22, "smtp": 0.08, "domain_u": 0.08, "ftp_data": 0.06,
               "pop_3": 0.04, "ssh": 0.04, "auth": 0.03, "other": 0.11},
    "dos": {"ecr_i": 0.34, "http": 0.28, "private": 0.24, "eco_i": 0.06, "other": 0.08},
    "probe": {"private": 0.40, "eco_i": 0.16, "http": 0.10, "finger": 0.08, "auth": 0.06,
              "telnet": 0.05, "ftp": 0.05, "other": 0.10},
    "r2l": {"ftp": 0.25, "telnet": 0.20, "smtp": 0.16, "pop_3": 0.12, "ftp_data": 0.10,
            "http": 0.09, "other": 0.08},
    "u2r": {"telnet": 0.45, "ssh": 0.25, "ftp": 0.14, "http": 0.08, "other": 0.08},
}

_FLAG_PROFILES: dict[str, dict[str, float]] = {
    "normal": {"SF": 0.93, "S1": 0.03, "RSTO": 0.02, "OTH": 0.02},
    "dos": {"S0": 0.72, "SF": 0.16, "REJ": 0.07, "RSTR": 0.05},
    "probe": {"REJ": 0.48, "S0": 0.22, "SF": 0.18, "RSTR": 0.07, "SH": 0.05},
    "r2l": {"SF": 0.86, "RSTO": 0.07, "S1": 0.04, "REJ": 0.03},
    "u2r": {"SF": 0.94, "S1": 0.04, "RSTO": 0.02},
}


def _profile_to_probabilities(profile: dict[str, float], vocabulary: list[str]) -> list[float]:
    weights = np.array([profile.get(value, 0.0) for value in vocabulary], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValidationError("Categorical profile must contain at least one positive weight")
    return (weights / total).tolist()


def _service_profile(label: str) -> list[float]:
    return _profile_to_probabilities(_SERVICE_PROFILES[label], SERVICES)


def _flag_profile(label: str) -> list[float]:
    return _profile_to_probabilities(_FLAG_PROFILES[label], FLAGS)


_GENERATORS: dict[str, Callable[[np.random.Generator, int], dict[str, np.ndarray]]] = {
    "normal": _normal,
    "dos": _dos,
    "probe": _probe,
    "r2l": _r2l,
    "u2r": _u2r,
}


def generate_synthetic_dataset(
    n_rows: int = 50_000,
    random_state: int = 42,
    class_weights: dict[str, float] | None = None,
    binary_labels: bool = False,
) -> pd.DataFrame:
    """Generate a labelled synthetic flow dataset with class-conditional profiles."""
    if n_rows < len(ATTACK_CLASSES):
        raise ValidationError(f"n_rows must be at least {len(ATTACK_CLASSES)}, got {n_rows}")

    weights = class_weights or CLASS_WEIGHTS
    unknown = set(weights) - set(_GENERATORS)
    if unknown:
        raise ValidationError(f"Unknown class label(s): {', '.join(sorted(unknown))}")

    rng = np.random.default_rng(random_state)
    total_weight = sum(weights.values())
    counts = {label: max(1, int(n_rows * weight / total_weight)) for label, weight in weights.items()}
    counts["normal"] += n_rows - sum(counts.values())

    frames = []
    for label, size in counts.items():
        if size <= 0:
            continue
        block = pd.DataFrame(_GENERATORS[label](rng, size))
        block["label"] = label
        frames.append(block)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    dataset["src_bytes"] = dataset["src_bytes"].clip(upper=10_000_000)
    dataset["dst_bytes"] = dataset["dst_bytes"].clip(upper=10_000_000)

    if binary_labels:
        dataset["label"] = np.where(dataset["label"] == "normal", "normal", "attack")

    ordered = FEATURE_COLUMNS + ["label"]
    dataset = dataset[[column for column in ordered if column in dataset.columns]]
    log.info("Generated synthetic dataset with {} rows and {} columns", *dataset.shape)
    return dataset


def load_dataset(path: Path | str, target_column: str = "label") -> pd.DataFrame:
    """Load a dataset from CSV or Parquet and verify the target column exists."""
    file_path = validate_path(path, must_exist=True, suffixes=[".csv", ".parquet"])
    try:
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path)
        else:
            frame = pd.read_parquet(file_path)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Failed to read dataset {file_path}: {exc}") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    if target_column not in frame.columns:
        raise ValidationError(f"Dataset {file_path} has no target column '{target_column}'")
    if frame.empty:
        raise ValidationError(f"Dataset {file_path} is empty")

    log.info("Loaded {} rows and {} columns from {}", frame.shape[0], frame.shape[1], file_path)
    return frame


def load_or_generate(
    path: Path | str,
    n_rows: int = 50_000,
    random_state: int = 42,
    target_column: str = "label",
) -> pd.DataFrame:
    """Load a dataset if present, otherwise generate and cache a synthetic one."""
    file_path = Path(path)
    if file_path.is_file():
        return load_dataset(file_path, target_column=target_column)

    log.warning("Dataset {} not found; generating synthetic traffic instead", file_path)
    dataset = generate_synthetic_dataset(n_rows=n_rows, random_state=random_state)
    save_dataset(dataset, file_path)
    return dataset


def save_dataset(frame: pd.DataFrame, path: Path | str) -> Path:
    """Persist a dataset as CSV or Parquet based on the file suffix."""
    file_path = Path(path)
    ensure_directory(file_path.parent)
    try:
        if file_path.suffix.lower() == ".parquet":
            frame.to_parquet(file_path, index=False)
        else:
            frame.to_csv(file_path, index=False)
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Failed to write dataset to {file_path}: {exc}") from exc
    log.info("Saved dataset with {} rows to {}", len(frame), file_path)
    return file_path
