"""Stratified train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, validate_dataframe, validate_ratio

log = get_logger(__name__)


@dataclass
class DataSplit:
    """Feature/target frames for the three dataset partitions."""

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.x_train), "val": len(self.x_val), "test": len(self.x_test)}


def _stratify_or_none(target: Sequence, min_per_class: int = 2) -> np.ndarray | None:
    values = pd.Series(np.asarray(target))
    if values.value_counts().min() < min_per_class:
        log.warning("Some classes are too rare to stratify; falling back to a random split")
        return None
    return values.to_numpy()


def split_dataset(
    features: pd.DataFrame,
    target: Sequence,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> DataSplit:
    """Split features and target into stratified train/validation/test sets."""
    validate_dataframe(features, min_rows=3)
    validate_ratio(test_size, "test_size", 0.0, 0.9)
    validate_ratio(val_size, "val_size", 0.0, 0.9)
    if test_size + val_size >= 1.0:
        raise ValidationError("test_size + val_size must be below 1.0")

    y = np.asarray(target)
    if len(y) != len(features):
        raise ValidationError(f"Feature/target length mismatch: {len(features)} vs {len(y)}")

    x_rest, x_test, y_rest, y_test = train_test_split(
        features,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_or_none(y),
    )

    relative_val = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest,
        y_rest,
        test_size=relative_val,
        random_state=random_state,
        stratify=_stratify_or_none(y_rest),
    )

    split = DataSplit(
        x_train=x_train.reset_index(drop=True),
        x_val=x_val.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    log.info("Dataset split into {}", split.sizes)
    return split


def class_distribution(target: Sequence, class_names: Sequence[str] | None = None) -> pd.Series:
    """Return the normalised class distribution of a target vector."""
    values = pd.Series(np.asarray(target))
    if class_names is not None:
        values = values.map(lambda index: class_names[int(index)] if index < len(class_names) else index)
    return values.value_counts(normalize=True).sort_index()
