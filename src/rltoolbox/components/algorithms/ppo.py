import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Dict, Any

from ....core.base import RLComponent


class PPO(RLComponent):
    """
    Proximal Policy Optimization (PPO) algorithm.

    This component implements the PPO-Clip objective.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # PPO configuration
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.ppo_epochs = config.get("ppo_epochs", 10)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.value_loss_coef = config.get("value_loss_coef", 0.5)

    def learning_update(self, context: Dict[str, Any]) -> None:
        """Update policy and value networks using PPO."""
        if "batch" not in context:
            return

        batch = context["batch"]
        states = torch.from_numpy(batch["states"]).float()
        actions = torch.from_numpy(batch["actions"]).long()
        old_log_probs = torch.from_numpy(batch["log_probs"]).float()
        advantages = torch.from_numpy(batch["advantages"]).float()
        returns = torch.from_numpy(batch["returns"]).float()

        # Get actor and critic from context
        actor = context["actor"]
        critic = context["critic"]

        # PPO update loop
        for _ in range(self.ppo_epochs):
            # Get new log probs, values, and entropy
            logits = actor(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            values = critic(states).squeeze()

            # Calculate ratio
            ratio = (new_log_probs - old_log_probs).exp()

            # Calculate surrogate objectives
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(returns, values)
            loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

            # Optimize
            context["actor_optimizer"].zero_grad()
            context["critic_optimizer"].zero_grad()
            loss.backward()
            context["actor_optimizer"].step()
            context["critic_optimizer"].step()

        context["actor_loss"] = actor_loss.item()
        context["critic_loss"] = critic_loss.item()

    def validate_config(self) -> bool:
        """Validate PPO configuration."""
        if self.clip_epsilon <= 0:
            return False
        if self.ppo_epochs <= 0:
            return False
        return True
