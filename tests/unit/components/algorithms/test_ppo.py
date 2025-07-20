import pytest
import torch
from rltoolbox.components.algorithms.ppo import PPO
from rltoolbox.components.agents.mlp_agent import MLPAgent


@pytest.fixture
def ppo_config():
    return {
        "clip_epsilon": 0.1,
        "ppo_epochs": 2,
    }


@pytest.fixture
def actor_critic_config():
    return {
        "state_dim": 4,
        "action_dim": 2,
        "hidden_layers": [16, 16],
        "learning_rate": 0.01,
    }


def test_ppo_learning_update(ppo_config, actor_critic_config):
    ppo = PPO(ppo_config)
    actor = MLPAgent(actor_critic_config)
    critic = MLPAgent(actor_critic_config)

    batch = {
        "states": torch.randn(2, 4).numpy(),
        "actions": torch.randint(0, 2, (2,)).numpy(),
        "log_probs": torch.randn(2).numpy(),
        "advantages": torch.randn(2).numpy(),
        "returns": torch.randn(2).numpy(),
    }
    context = {
        "batch": batch,
        "actor": actor.network,
        "critic": critic.network,
        "actor_optimizer": torch.optim.Adam(actor.network.parameters()),
        "critic_optimizer": torch.optim.Adam(critic.network.parameters()),
    }

    ppo.learning_update(context)

    assert "actor_loss" in context
    assert "critic_loss" in context
    assert isinstance(context["actor_loss"], float)
    assert isinstance(context["critic_loss"], float)
