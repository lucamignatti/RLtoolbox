import pytest
import random
from typing import Dict, Any

from rltoolbox.components.agents.epsilon_greedy_agent import EpsilonGreedyAgent


@pytest.fixture
def epsilon_greedy_config():
    return {
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.9,
        "num_actions": 5,
        "decay_type": "exponential",
    }


def test_epsilon_greedy_initialization(epsilon_greedy_config):
    agent = EpsilonGreedyAgent(epsilon_greedy_config)
    assert agent.current_epsilon == 1.0
    assert agent.num_actions == 5


def test_epsilon_greedy_action_selection_explore(epsilon_greedy_config):
    agent = EpsilonGreedyAgent(epsilon_greedy_config)
    context = {"q_values": [0.1, 0.2, 0.9, 0.4, 0.5], "training": {"total_steps": 0}}
    random.seed(0)  # Ensure random.random() < epsilon
    agent.action_selection(context)
    assert "action" in context
    assert context["action"] in range(agent.num_actions)
    assert context["epsilon"] < 1.0  # Epsilon should decay


def test_epsilon_greedy_action_selection_exploit(epsilon_greedy_config):
    agent = EpsilonGreedyAgent(epsilon_greedy_config)
    agent.current_epsilon = 0.0  # Force exploitation
    context = {"q_values": [0.1, 0.2, 0.9, 0.4, 0.5], "training": {"total_steps": 0}}
    agent.action_selection(context)
    assert "action" in context
    assert context["action"] == 2  # Index of max q_value


def test_epsilon_greedy_exponential_decay(epsilon_greedy_config):
    agent = EpsilonGreedyAgent(epsilon_greedy_config)
    initial_epsilon = agent.current_epsilon
    context = {"training": {"total_steps": 0}}
    agent._update_epsilon(context)
    assert agent.current_epsilon == initial_epsilon * agent.epsilon_decay


def test_epsilon_greedy_linear_decay(epsilon_greedy_config):
    epsilon_greedy_config["decay_type"] = "linear"
    epsilon_greedy_config["decay_steps"] = 10
    agent = EpsilonGreedyAgent(epsilon_greedy_config)
    agent.current_epsilon = 1.0
    context = {"training": {"total_steps": 5}}
    agent._update_epsilon(context)
    expected_epsilon = 1.0 - (1.0 - 0.1) * (5 / 10)
    assert agent.current_epsilon == expected_epsilon


def test_epsilon_greedy_get_set_state(epsilon_greedy_config):
    agent = EpsilonGreedyAgent(epsilon_greedy_config)
    agent.current_epsilon = 0.5
    state = agent.get_state()
    new_agent = EpsilonGreedyAgent(epsilon_greedy_config)
    new_agent.set_state(state)
    assert new_agent.current_epsilon == 0.5
