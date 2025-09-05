from dataclasses import dataclass
from os import PathLike

import torch


@dataclass
class Config:
    """Configuration class for hyperparameters and settings."""

    LEARNING_RATE: float
    EPOCHS: int
    BATCH_SIZE: int
    PATIENCE: int
    MIN_DELTA: float
    DATA_DIR: str | PathLike
    DEVICE: torch.device

    # MLflow settings
    EXPERIMENT_NAME: str
    MLFLOW_TRACKING_URI: str
