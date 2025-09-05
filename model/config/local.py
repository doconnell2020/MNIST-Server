import torch

from .base import Config

config = Config(
    LEARNING_RATE=0.1,
    EPOCHS=10,
    BATCH_SIZE=64,
    PATIENCE=3,
    MIN_DELTA=0.01,
    DATA_DIR="./data",
    DEVICE=torch.device("cpu"),
    EXPERIMENT_NAME="mnist-cnn-experiments",
    MLFLOW_TRACKING_URI="file:./mlruns",
)
