import pytest
from unittest.mock import patch
from typing import Dict, Any

from rltoolbox.components.loggers.console_logger import ConsoleLogger


@pytest.fixture
def console_logger_config():
    return {
        "log_frequency": 1,
        "verbose": True,
        "log_steps": False,
    }


def test_console_logger_initialization(console_logger_config):
    logger = ConsoleLogger(console_logger_config)
    assert logger.log_frequency == 1
    assert logger.verbose is True


@patch("builtins.print")
def test_console_logger_training_start(mock_print, console_logger_config):
    logger = ConsoleLogger(console_logger_config)
    context = {"config": {"test": "config"}}
    logger.training_start(context)
    mock_print.assert_any_call("=== Training Started ===")
    mock_print.assert_any_call(f"Configuration: {logger.config}")


@patch("builtins.print")
def test_console_logger_episode_end(mock_print, console_logger_config):
    logger = ConsoleLogger(console_logger_config)
    context = {
        "training": {"episode": 0, "episode_reward": 10.0, "episode_length": 100, "total_steps": 100},
        "metrics": {"episode_rewards": [10.0], "episode_lengths": [100]},
    }
    logger.episode_end(context)
    mock_print.assert_called_once() # Should print episode info


@patch("builtins.print")
def test_console_logger_training_end(mock_print, console_logger_config):
    logger = ConsoleLogger(console_logger_config)
    logger.training_start({"config": {}})
    context = {
        "training": {"episode": 5, "total_steps": 500},
        "metrics": {"episode_rewards": [10, 20, 30, 40, 50], "episode_lengths": [100]*5},
    }
    logger.training_end(context)
    mock_print.assert_any_call("=== Training Completed ===")
    mock_print.assert_any_call("Total Episodes: 5")
    mock_print.assert_any_call("Total Steps: 500")
    mock_print.assert_any_call("Final Mean Reward:    30.00")
