import pytest
from unittest.mock import patch, MagicMock
from rltoolbox.components.loggers.wandb_logger import WandbLogger


@pytest.fixture
def wandb_logger_config():
    return {
        "project": "test_project",
        "entity": "test_entity",
        "run_name": "test_run",
    }


@patch("wandb.init")
def test_wandb_logger_initialization(mock_wandb_init, wandb_logger_config):
    logger = WandbLogger(wandb_logger_config)
    mock_wandb_init.assert_called_once_with(
        project="test_project",
        entity="test_entity",
        name="test_run",
        config=wandb_logger_config,
    )


@patch("wandb.log")
@patch("wandb.init")
def test_wandb_logger_episode_end(mock_wandb_init, mock_wandb_log, wandb_logger_config):
    logger = WandbLogger(wandb_logger_config)
    context = {
        "training": {
            "episode": 0,
            "episode_reward": 10.0,
            "episode_length": 100,
            "total_steps": 100,
        },
        "loss": 0.1,
    }
    logger.episode_end(context)
    mock_wandb_log.assert_called_once_with({
        "episode": 0,
        "episode_reward": 10.0,
        "episode_length": 100,
        "total_steps": 100,
        "loss": 0.1,
    })


@patch("wandb.finish")
@patch("wandb.init")
def test_wandb_logger_training_end(mock_wandb_init, mock_wandb_finish, wandb_logger_config):
    logger = WandbLogger(wandb_logger_config)
    logger.training_end({})
    mock_wandb_finish.assert_called_once()
