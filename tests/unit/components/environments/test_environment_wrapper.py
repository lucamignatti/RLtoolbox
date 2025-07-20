import pytest
import numpy as np
from typing import Dict, Any

from rltoolbox.components.environments.environment_wrapper import EnvironmentWrapper


@pytest.fixture
def env_wrapper_config():
    return {
        "normalize_observations": True,
        "clip_rewards": True,
        "reward_scale": 0.5,
    }


def test_environment_wrapper_initialization(env_wrapper_config):
    wrapper = EnvironmentWrapper(env_wrapper_config)
    assert wrapper.normalize_observations is True
    assert wrapper.clip_rewards is True
    assert wrapper.reward_scale == 0.5


def test_environment_wrapper_observation_received_normalize(env_wrapper_config):
    wrapper = EnvironmentWrapper(env_wrapper_config)
    context = {"state": np.array([1.0, 2.0, 3.0])}
    wrapper.observation_received(context)
    assert "state" in context
    # Check if normalization happened (values should be different from original)
    assert not np.allclose(context["state"], np.array([1.0, 2.0, 3.0]))


def test_environment_wrapper_transition_received_clip_scale(env_wrapper_config):
    wrapper = EnvironmentWrapper(env_wrapper_config)
    context = {"reward": 10.0}
    wrapper.transition_received(context)
    assert "reward" in context
    # Reward should be scaled (10.0 * 0.5 = 5.0) and then clipped to 1.0
    assert context["reward"] == 1.0

    context = {"reward": -10.0}
    wrapper.transition_received(context)
    assert context["reward"] == -1.0

    context = {"reward": 0.2}
    wrapper.transition_received(context)
    assert context["reward"] == 0.1 # 0.2 * 0.5 = 0.1, not clipped


def test_environment_wrapper_get_set_state(env_wrapper_config):
    wrapper = EnvironmentWrapper(env_wrapper_config)
    context = {"state": np.array([1.0, 2.0, 3.0])}
    wrapper.observation_received(context) # Update internal stats

    state = wrapper.get_state()
    new_wrapper = EnvironmentWrapper(env_wrapper_config)
    new_wrapper.set_state(state)

    assert np.array_equal(new_wrapper.obs_mean, wrapper.obs_mean)
    assert np.array_equal(new_wrapper.obs_std, wrapper.obs_std)
    assert new_wrapper.obs_count == wrapper.obs_count
