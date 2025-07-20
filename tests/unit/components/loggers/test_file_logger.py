import pytest
import json
from pathlib import Path
from typing import Dict, Any

from rltoolbox.components.loggers.file_logger import FileLogger


@pytest.fixture
def file_logger_config(tmp_path):
    return {
        "log_dir": str(tmp_path / "logs"),
        "save_frequency": 1,
        "save_config": True,
    }


def test_file_logger_initialization(file_logger_config):
    logger = FileLogger(file_logger_config)
    assert logger.log_dir.exists()
    assert logger.save_frequency == 1


def test_file_logger_training_start(file_logger_config):
    logger = FileLogger(file_logger_config)
    context = {"config": {"test": "config"}}
    logger.training_start(context)
    config_file = logger.run_dir / "config.json"
    assert config_file.exists()
    with open(config_file, "r") as f:
        loaded_config = json.load(f)
    assert loaded_config == context["config"]


def test_file_logger_episode_end(file_logger_config):
    logger = FileLogger(file_logger_config)
    context = {
        "training": {"episode": 0, "episode_reward": 10.0, "episode_length": 100, "total_steps": 100},
        "metrics": {"episode_rewards": [10.0], "episode_lengths": [100]},
    }
    logger.episode_end(context)
    episodes_file = logger.run_dir / "episodes.json"
    assert episodes_file.exists()
    with open(episodes_file, "r") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["reward"] == 10.0


def test_file_logger_training_end(file_logger_config):
    logger = FileLogger(file_logger_config)
    context = {
        "training": {"episode": 5, "total_steps": 500},
        "metrics": {"episode_rewards": [10, 20, 30, 40, 50], "episode_lengths": [100]*5},
    }
    logger.training_end(context)
    summary_file = logger.run_dir / "summary.json"
    assert summary_file.exists()
    with open(summary_file, "r") as f:
        summary = json.load(f)
    assert summary["total_episodes"] == 5
    assert summary["total_steps"] == 500


def test_file_logger_get_set_state(file_logger_config):
    logger = FileLogger(file_logger_config)
    context = {
        "training": {"episode": 0, "episode_reward": 10.0, "episode_length": 100, "total_steps": 100},
        "metrics": {"episode_rewards": [10.0], "episode_lengths": [100]},
    }
    logger.episode_end(context)
    state = logger.get_state()
    new_logger = FileLogger(file_logger_config)
    new_logger.set_state(state)
    assert len(new_logger.episode_data) == len(logger.episode_data)
    assert new_logger.run_id == logger.run_id
