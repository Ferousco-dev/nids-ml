"""Model training, evaluation and persistence."""

from src.models.evaluator import EvaluationReport, evaluate_model
from src.models.registry import ModelBundle, ModelMetadata, ModelRegistry
from src.models.trainer import ModelTrainer, TrainedModel

__all__ = [
    "EvaluationReport",
    "ModelBundle",
    "ModelMetadata",
    "ModelRegistry",
    "ModelTrainer",
    "TrainedModel",
    "evaluate_model",
]
