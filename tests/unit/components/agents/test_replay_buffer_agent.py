import pytest
import numpy as np
import random
from typing import Dict, Any

from rltoolbox.components.agents.replay_buffer_agent import ReplayBufferAgent


@pytest.fixture
def replay_buffer_config():
    return {
        "buffer_size": 10,
        "batch_size": 2,
        "min_buffer_size": 5,
    }


@pytest.fixture
def sample_transition():
    return {
        "state": np.array([1, 2, 3]),
        "action": 0,
        "reward": 1.0,
        "next_state": np.array([4, 5, 6]),
        "done": False,
    }


def test_replay_buffer_initialization(replay_buffer_config):
    buffer = ReplayBufferAgent(replay_buffer_config)
    assert buffer.buffer_size == 10
    assert buffer.batch_size == 2
    assert buffer.min_buffer_size == 5
    assert len(buffer.buffer) == 0
    assert buffer.buffer_index == 0
    assert buffer.buffer_full is False


def test_replay_buffer_experience_storage(replay_buffer_config, sample_transition):
    buffer = ReplayBufferAgent(replay_buffer_config)
    context = sample_transition.copy()
    buffer.experience_storage(context)
    assert len(buffer.buffer) == 1
    # Fill buffer
    for i in range(buffer.buffer_size):
        buffer.experience_storage(sample_transition.copy())
    assert len(buffer.buffer) == buffer.buffer_size
    assert buffer.buffer_full is True

    # Overwrite oldest entry
    new_transition = {"state": np.array([7, 8, 9]), "action": 1, "reward": 2.0, "next_state": np.array([1, 2, 3]), "done": True}
    old_index_to_be_overwritten = buffer.buffer_index # This is the index that will be overwritten
    buffer.experience_storage(new_transition)
    assert np.array_equal(buffer.buffer[old_index_to_be_overwritten]["state"], new_transition["state"])
    assert buffer.buffer[old_index_to_be_overwritten]["action"] == new_transition["action"]
    assert buffer.buffer[old_index_to_be_overwritten]["reward"] == new_transition["reward"]
    assert np.array_equal(buffer.buffer[old_index_to_be_overwritten]["next_state"], new_transition["next_state"])
    assert buffer.buffer[old_index_to_be_overwritten]["done"] == new_transition["done"]


def test_replay_buffer_learning_update(replay_buffer_config, sample_transition):
    buffer = ReplayBufferAgent(replay_buffer_config)
    # Fill buffer to min_buffer_size
    for _ in range(buffer.min_buffer_size):
        buffer.experience_storage(sample_transition.copy())

    context = {}
    buffer.learning_update(context)
    assert "batch" in context
    assert context["batch"]["size"] == buffer.batch_size
    assert len(context["batch"]["states"]) == buffer.batch_size


def test_replay_buffer_learning_update_not_enough_samples(replay_buffer_config, sample_transition):
    buffer = ReplayBufferAgent(replay_buffer_config)
    # Fill buffer but not to min_buffer_size
    for _ in range(buffer.min_buffer_size - 1):
        buffer.experience_storage(sample_transition.copy())

    context = {}
    buffer.learning_update(context)
    assert "batch" not in context


def test_replay_buffer_get_set_state(replay_buffer_config, sample_transition):
    buffer = ReplayBufferAgent(replay_buffer_config)
    for _ in range(3):
        buffer.experience_storage(sample_transition.copy())

    state = buffer.get_state()
    new_buffer = ReplayBufferAgent(replay_buffer_config)
    new_buffer.set_state(state)

    assert len(new_buffer.buffer) == len(buffer.buffer)
    assert new_buffer.buffer_index == buffer.buffer_index
    assert new_buffer.buffer_full == buffer.buffer_full
    # Deep comparison for buffer content
    for i in range(len(buffer.buffer)):
        for key in buffer.buffer[i]:
            if isinstance(buffer.buffer[i][key], np.ndarray):
                assert np.array_equal(new_buffer.buffer[i][key], buffer.buffer[i][key])
            else:
                assert new_buffer.buffer[i][key] == buffer.buffer[i][key]
