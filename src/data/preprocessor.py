"""Cleaning, encoding and scaling of network flow data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from src.utils.logger import get_logger
from src.utils.validators import ValidationError, ensure_directory, validate_dataframe

log = get_logger(__name__)

ScalerName = Literal["standard", "minmax", "none"]


@dataclass
class PreprocessorArtifacts:
    """Fitted transformers and the resulting feature schema."""

    feature_names: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    class_names: list[str]


class FlowPreprocessor:
    """Fit/transform pipeline for tabular network flow records.

    Categorical columns are one-hot encoded, numeric columns are imputed with
    the training median and scaled, and the target is label encoded.
    """

    def __init__(
        self,
        target_column: str = "label",
        scaler: ScalerName = "standard",
        drop_columns: list[str] | None = None,
    ) -> None:
        self.target_column = target_column
        self.scaler_name = scaler
        self.drop_columns = list(drop_columns or [])
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.label_encoder = LabelEncoder()
        self.categorical_columns: list[str] = []
        self.numeric_columns: list[str] = []
        self.feature_names: list[str] = []
        self.medians: pd.Series | None = None
        self.category_levels: dict[str, list[str]] = {}
        self.defaults: dict[str, Any] = {}
        self.fitted = False

    def _split_columns(self, frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        features = frame.drop(columns=[self.target_column], errors="ignore")
        features = features.drop(columns=self.drop_columns, errors="ignore")
        categorical = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numeric = features.select_dtypes(include=[np.number]).columns.tolist()
        return categorical, numeric

    def _make_scaler(self) -> StandardScaler | MinMaxScaler | None:
        if self.scaler_name == "standard":
            return StandardScaler()
        if self.scaler_name == "minmax":
            return MinMaxScaler()
        if self.scaler_name == "none":
            return None
        raise ValidationError(f"Unsupported scaler '{self.scaler_name}'")

    def _encode(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(self.medians)

        if not self.categorical_columns:
            return numeric

        categorical = frame[self.categorical_columns].astype("object")
        for column in self.categorical_columns:
            levels = self.category_levels[column]
            values = categorical[column].where(categorical[column].isin(levels), other="__unknown__")
            categorical[column] = pd.Categorical(values, categories=levels + ["__unknown__"])
        dummies = pd.get_dummies(categorical, columns=self.categorical_columns, dtype=np.float64)
        return pd.concat([numeric.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    def fit(self, frame: pd.DataFrame) -> FlowPreprocessor:
        """Fit transformers on the training frame."""
        validate_dataframe(frame, required_columns=[self.target_column])
        self.categorical_columns, self.numeric_columns = self._split_columns(frame)
        if not self.numeric_columns and not self.categorical_columns:
            raise ValidationError("No usable feature columns found after dropping configured columns")

        numeric = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce")
        self.medians = numeric.replace([np.inf, -np.inf], np.nan).median()
        self.medians = self.medians.fillna(0.0)
        self.category_levels = {
            column: sorted(frame[column].astype("object").dropna().unique().tolist())
            for column in self.categorical_columns
        }

        self.defaults = {column: float(value) for column, value in self.medians.items()}
        for column, levels in self.category_levels.items():
            mode = frame[column].astype("object").mode()
            self.defaults[column] = mode.iloc[0] if not mode.empty else levels[0]

        encoded = self._encode(frame)
        self.feature_names = encoded.columns.tolist()

        self.scaler = self._make_scaler()
        if self.scaler is not None:
            self.scaler.fit(encoded.to_numpy(dtype=np.float64))

        self.label_encoder.fit(frame[self.target_column].astype(str))
        self.fitted = True
        log.info(
            "Preprocessor fitted: {} features ({} categorical), {} classes",
            len(self.feature_names),
            len(self.categorical_columns),
            len(self.label_encoder.classes_),
        )
        return self

    def fill_defaults(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add any absent input columns using the values learned during fitting."""
        missing = [column for column in self.defaults if column not in frame.columns]
        if not missing:
            return frame
        filled = frame.copy()
        for column in missing:
            filled[column] = self.defaults[column]
        log.debug("Filled {} absent input column(s) with training defaults", len(missing))
        return filled

    def transform_features(self, frame: pd.DataFrame, fill_missing: bool = False) -> pd.DataFrame:
        """Transform a frame into the fitted feature space.

        With ``fill_missing`` set, columns absent from the input are populated
        with the training median (numeric) or most frequent value (categorical)
        instead of raising, which lets callers submit partial flow records.
        """
        if not self.fitted:
            raise ValidationError("Preprocessor must be fitted before calling transform_features")
        validate_dataframe(frame)
        if fill_missing:
            frame = self.fill_defaults(frame)

        missing = [
            column
            for column in self.numeric_columns + self.categorical_columns
            if column not in frame.columns
        ]
        if missing:
            raise ValidationError(f"Input is missing column(s): {', '.join(missing[:10])}")

        encoded = self._encode(frame).reindex(columns=self.feature_names, fill_value=0.0)
        values = encoded.to_numpy(dtype=np.float64)
        if self.scaler is not None:
            values = self.scaler.transform(values)
        return pd.DataFrame(values, columns=self.feature_names, index=frame.index)

    def transform_target(self, target: pd.Series) -> np.ndarray:
        """Encode string labels into integer class indices."""
        if not self.fitted:
            raise ValidationError("Preprocessor must be fitted before calling transform_target")
        values = target.astype(str)
        unseen = set(values.unique()) - set(self.label_encoder.classes_)
        if unseen:
            raise ValidationError(f"Unseen label(s) in target: {', '.join(sorted(unseen))}")
        return self.label_encoder.transform(values)

    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Fit on a frame and return its encoded features and target."""
        self.fit(frame)
        return self.transform_features(frame), self.transform_target(frame[self.target_column])

    def inverse_transform_target(self, encoded: np.ndarray) -> np.ndarray:
        """Map integer class indices back to their original labels."""
        if not self.fitted:
            raise ValidationError("Preprocessor must be fitted before inverse transforming")
        return self.label_encoder.inverse_transform(np.asarray(encoded, dtype=int))

    @property
    def class_names(self) -> list[str]:
        """Ordered class labels known to the preprocessor."""
        if not self.fitted:
            raise ValidationError("Preprocessor must be fitted before reading class names")
        return [str(label) for label in self.label_encoder.classes_]

    @property
    def artifacts(self) -> PreprocessorArtifacts:
        """Summary of the fitted schema."""
        return PreprocessorArtifacts(
            feature_names=list(self.feature_names),
            categorical_columns=list(self.categorical_columns),
            numeric_columns=list(self.numeric_columns),
            class_names=self.class_names,
        )

    def save(self, path: Path | str) -> Path:
        """Persist the fitted preprocessor with joblib."""
        if not self.fitted:
            raise ValidationError("Refusing to save an unfitted preprocessor")
        file_path = Path(path)
        ensure_directory(file_path.parent)
        try:
            joblib.dump(self, file_path)
        except OSError as exc:
            raise ValidationError(f"Could not save preprocessor to {file_path}: {exc}") from exc
        log.info("Preprocessor saved to {}", file_path)
        return file_path

    @staticmethod
    def load(path: Path | str) -> FlowPreprocessor:
        """Load a preprocessor previously saved with :meth:`save`."""
        file_path = Path(path)
        if not file_path.is_file():
            raise ValidationError(f"Preprocessor file not found: {file_path}")
        try:
            preprocessor = joblib.load(file_path)
        except (OSError, ValueError) as exc:
            raise ValidationError(f"Could not load preprocessor from {file_path}: {exc}") from exc
        if not isinstance(preprocessor, FlowPreprocessor):
            raise ValidationError(f"{file_path} does not contain a FlowPreprocessor")
        return preprocessor


def clean_dataset(frame: pd.DataFrame, target_column: str = "label") -> pd.DataFrame:
    """Drop duplicates, constant columns and rows with a missing target."""
    validate_dataframe(frame, required_columns=[target_column])
    cleaned = frame.replace([np.inf, -np.inf], np.nan)
    before = len(cleaned)

    cleaned = cleaned.dropna(subset=[target_column]).drop_duplicates()
    constant = [
        column
        for column in cleaned.columns
        if column != target_column and cleaned[column].nunique(dropna=False) <= 1
    ]
    if constant:
        cleaned = cleaned.drop(columns=constant)
        log.info("Dropped {} constant column(s): {}", len(constant), ", ".join(constant))

    log.info("Cleaning removed {} row(s); {} remain", before - len(cleaned), len(cleaned))
    return cleaned.reset_index(drop=True)
