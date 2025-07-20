import pytest
import numpy as np
from typing import Dict, Any

from rltoolbox.components.loggers.metrics_logger import MetricsLogger


@pytest.fixture
def metrics_logger_config():
    return {
        "window_size": 3,
        "compute_frequency": 1,
    }


def test_metrics_logger_initialization(metrics_logger_config):
    logger = MetricsLogger(metrics_logger_config)
    assert logger.window_size == 3
    assert logger.compute_frequency == 1
    assert len(logger.metrics_history) == 0


def test_metrics_logger_episode_end(metrics_logger_config):
    logger = MetricsLogger(metrics_logger_config)
    context = {
        "training": {"episode": 0},
        "metrics": {"episode_rewards": [10.0, 20.0, 30.0], "episode_lengths": [100, 110, 120]},
    }
    logger.episode_end(context)
    assert len(logger.metrics_history) == 1
    metrics = logger.metrics_history[0]
    assert metrics["episode"] == 0
    assert metrics["mean_reward"] == 20.0
    assert metrics["mean_length"] == 110.0


def test_metrics_logger_windowed_metrics(metrics_logger_config):
    logger = MetricsLogger(metrics_logger_config)
    context = {
        "training": {"episode": 0},
        "metrics": {"episode_rewards": [1, 2, 3, 4, 5], "episode_lengths": [10, 20, 30, 40, 50]},
    }
    # Simulate multiple episodes to fill window
    for i in range(5):
        context["training"]["episode"] = i
        logger.episode_end(context)

    metrics = logger.metrics_history[-1]
    assert metrics["mean_reward_3"] == np.mean([3, 4, 5])
    assert metrics["mean_length_3"] == np.mean([30, 40, 50])


def test_metrics_logger_evaluation_hooks(metrics_logger_config):
    logger = MetricsLogger(metrics_logger_config)
    context = {}
    logger.evaluation_start(context)
    assert "evaluation_metrics" in context

    context["training"] = {"episode_reward": 50.0, "episode_length": 200}
    logger.evaluation_episode_end(context)
    assert context["evaluation_metrics"]["episode_rewards"] == [50.0]

    logger.evaluation_end(context)
    assert "computed_evaluation_metrics" in context
    assert context["computed_evaluation_metrics"]["mean_reward"] == 50.0
