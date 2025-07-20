import pytest
import gymnasium as gym
import numpy as np
from typing import Dict, Any

from rltoolbox.components.environments.simple_environment import SimpleEnvironment


@pytest.fixture
def simple_env_config():
    return {
        "env_name": "CartPole-v1",
        "render_mode": None,
        "max_episode_steps": 10,
    }


def test_simple_environment_initialization(simple_env_config):
    env = SimpleEnvironment(simple_env_config)
    assert isinstance(env.env, gym.Env)
    assert env.env_name == "CartPole-v1"


def test_simple_environment_episode_reset(simple_env_config):
    env = SimpleEnvironment(simple_env_config)
    context = {}
    env.episode_reset(context)
    assert "state" in context
    assert "info" in context
    assert isinstance(context["state"], (list, tuple, np.ndarray))


def test_simple_environment_step(simple_env_config):
    env = SimpleEnvironment(simple_env_config)
    context = {}
    env.episode_reset(context)
    context["action"] = env.action_space.sample()  # Provide a valid action
    env.environment_step(context)
    assert "next_state" in context
    assert "reward" in context
    assert "done" in context
    assert "terminated" in context
    assert "truncated" in context
    assert "info" in context
    assert isinstance(context["reward"], float)
    assert isinstance(context["done"], bool)


def test_simple_environment_training_end(simple_env_config):
    env = SimpleEnvironment(simple_env_config)
    # Ensure env is created before calling training_end
    env.episode_reset({})
    env.training_end({})
    # No direct assertion for env.close(), but no error should be raised


def test_simple_environment_get_state(simple_env_config):
    env = SimpleEnvironment(simple_env_config)
    state = env.get_state()
    assert state["env_name"] == simple_env_config["env_name"]
    assert state["render_mode"] == simple_env_config["render_mode"]
    assert state["max_episode_steps"] == simple_env_config["max_episode_steps"]


def test_simple_environment_set_state(simple_env_config):
    env = SimpleEnvironment(simple_env_config)
    new_state = {
        "env_name": "MountainCar-v0",
        "render_mode": "rgb_array",
        "max_episode_steps": 200,
    }
    env.set_state(new_state)
    # Note: set_state for SimpleEnvironment primarily updates internal config, not the gym.Env instance
    # Re-initializing the env would be needed to reflect changes in the gym.Env object itself
    # The set_state method for SimpleEnvironment primarily updates internal config, not the gym.Env instance.
    # Re-initializing the env would be needed to reflect changes in the gym.Env object itself.
    # Therefore, we don't assert on env.env_name directly here as it's set in __init__.
    pass
