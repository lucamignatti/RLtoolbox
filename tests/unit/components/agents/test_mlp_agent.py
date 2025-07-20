import pytest
import torch
from rltoolbox.components.agents.mlp_agent import MLPAgent


@pytest.fixture
def mlp_agent_config():
    return {
        "state_dim": 4,
        "action_dim": 2,
        "hidden_layers": [16, 16],
        "learning_rate": 0.01,
    }


def test_mlp_agent_initialization(mlp_agent_config):
    agent = MLPAgent(mlp_agent_config)
    assert isinstance(agent.network, torch.nn.Sequential)
    assert len(agent.network) == 5  # 2 linear, 2 relu, 1 linear


def test_mlp_agent_action_selection(mlp_agent_config):
    agent = MLPAgent(mlp_agent_config)
    context = {"state": torch.randn(4).numpy()}
    agent.action_selection(context)
    assert "q_values" in context
    assert context["q_values"].shape == (2,)


def test_mlp_agent_learning_update(mlp_agent_config):
    agent = MLPAgent(mlp_agent_config)
    batch = {
        "states": torch.randn(2, 4).numpy(),
        "actions": torch.randint(0, 2, (2,)).numpy(),
        "rewards": torch.randn(2).numpy(),
        "next_states": torch.randn(2, 4).numpy(),
        "dones": torch.zeros(2).numpy(),
    }
    context = {"batch": batch}
    agent.learning_update(context)
    assert "loss" in context
    assert isinstance(context["loss"], float)
