from os import PathLike
from pathlib import Path

import pytest
import torch

from config.base import Config


@pytest.fixture
def config() -> Config:
    return Config(
        LEARNING_RATE=0.0,
        EPOCHS=0,
        BATCH_SIZE=0,
        PATIENCE=0,
        MIN_DELTA=0.0,
        DATA_DIR="0.0",
        DEVICE=torch.device("cpu"),
        EXPERIMENT_NAME="test",
        MLFLOW_TRACKING_URI="www.test.domain",
    )


def test_learning_rate(config: Config) -> None:
    assert isinstance(config.LEARNING_RATE, float)


def test_epochs(config: Config) -> None:
    assert isinstance(config.EPOCHS, int)


def test_batch_size(config: Config) -> None:
    assert isinstance(config.BATCH_SIZE, int)


def test_patience(config: Config) -> None:
    assert isinstance(config.PATIENCE, int)


def test_min_delta(config: Config) -> None:
    assert isinstance(config.MIN_DELTA, float)


def test_data_dir(config: Config) -> None:
    assert isinstance(config.DATA_DIR, str)


def test_data_dir_paths(config: Config) -> None:
    config.DATA_DIR = Path(config.DATA_DIR)
    assert isinstance(config.DATA_DIR, PathLike)


def test_device(config: Config) -> None:
    assert isinstance(config.DEVICE, torch.device)


def test_experiment_name(config: Config) -> None:
    assert isinstance(config.EXPERIMENT_NAME, str)


def test_mlflow_tracking_uri(config: Config) -> None:
    assert isinstance(config.MLFLOW_TRACKING_URI, str)
