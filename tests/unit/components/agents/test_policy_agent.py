import pytest
import numpy as np
import random
from typing import Dict, Any

from rltoolbox.components.agents.policy_agent import PolicyAgent


@pytest.fixture
def policy_agent_config():
    return {
        "num_actions": 3,
        "deterministic": False,
    }


def test_policy_agent_initialization(policy_agent_config):
    agent = PolicyAgent(policy_agent_config)
    assert agent.num_actions == 3
    assert agent.deterministic is False


def test_policy_agent_action_selection_stochastic(policy_agent_config):
    agent = PolicyAgent(policy_agent_config)
    context = {"action_probs": np.array([0.2, 0.5, 0.3])}
    random.seed(0)  # For reproducibility
    # Since it's stochastic, we can't assert a specific action, but check if it's valid
    agent.action_selection(context)
    assert "action" in context
    assert 0 <= context["action"] < agent.num_actions


def test_policy_agent_action_selection_deterministic(policy_agent_config):
    policy_agent_config["deterministic"] = True
    agent = PolicyAgent(policy_agent_config)
    context = {"action_probs": np.array([0.2, 0.5, 0.3])}
    agent.action_selection(context)
    assert "action" in context
    assert context["action"] == 1  # Index of max probability


def test_policy_agent_no_action_probs_fallback(policy_agent_config):
    agent = PolicyAgent(policy_agent_config)
    context = {}
    random.seed(0)  # For reproducibility
    agent.action_selection(context)
    assert "action" in context
    assert 0 <= context["action"] < agent.num_actions
