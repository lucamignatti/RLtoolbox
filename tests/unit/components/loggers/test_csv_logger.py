import pytest
import csv
from pathlib import Path
from typing import Dict, Any

from rltoolbox.components.loggers.csv_logger import CSVLogger


@pytest.fixture
def csv_logger_config(tmp_path):
    return {
        "log_dir": str(tmp_path / "logs"),
        "filename_prefix": "test_training",
    }


def test_csv_logger_initialization(csv_logger_config):
    logger = CSVLogger(csv_logger_config)
    assert logger.log_dir.exists()
    assert logger.episode_file.name.startswith("test_training_episodes_")
    assert logger.step_file.name.startswith("test_training_steps_")


def test_csv_logger_episode_end(csv_logger_config):
    logger = CSVLogger(csv_logger_config)
    context = {
        "training": {"episode": 0, "episode_reward": 10.0, "episode_length": 100, "total_steps": 100},
        "metrics": {"episode_rewards": [10.0], "episode_lengths": [100]},
    }
    logger.episode_end(context)
    assert logger.episode_file.exists()
    with open(logger.episode_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["reward"]) == 10.0


def test_csv_logger_step_end(csv_logger_config):
    csv_logger_config["save_step_data"] = True
    logger = CSVLogger(csv_logger_config)
    context = {
        "training": {"episode": 0, "step": 0},
        "reward": 1.0,
        "action": 0,
        "done": False,
    }
    logger.step_end(context)
    assert logger.step_file.exists()
    with open(logger.step_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert float(rows[0]["reward"]) == 1.0
