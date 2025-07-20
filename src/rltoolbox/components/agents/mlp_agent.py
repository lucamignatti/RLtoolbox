import numpy as np
import random
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim

from ....core.base import RLComponent


class MLPAgent(RLComponent):
    """
    MLP-based agent that learns a policy and/or value function.

    Can be configured for different RL algorithms (e.g., DQN, A2C).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Network configuration
        self.state_dim = config.get("state_dim", 4)
        self.action_dim = config.get("action_dim", 2)
        self.hidden_layers = config.get("hidden_layers", [64, 64])
        self.activation = config.get("activation", "relu")

        # Learning configuration
        self.learning_rate = config.get("learning_rate", 0.001)
        self.gamma = config.get("gamma", 0.99)
        self.optimizer_type = config.get("optimizer", "adam")

        # Create network and optimizer
        self._setup_network()
        self._setup_optimizer()

    def _setup_network(self):
        """Create the MLP network."""
        layers = []
        input_dim = self.state_dim
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self._get_activation())
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, self.action_dim))

        self.network = nn.Sequential(*layers)

    def _get_activation(self):
        """Get activation function from config."""
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _setup_optimizer(self):
        """Create the optimizer."""
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select action based on network output."""
        if "state" not in context:
            return

        state = torch.from_numpy(context["state"]).float().unsqueeze(0)
        with torch.no_grad():
            q_values = self.network(state)

        context["q_values"] = q_values.numpy().flatten()

    def learning_update(self, context: Dict[str, Any]) -> None:
        """Update network from a batch of experiences."""
        if "batch" not in context:
            return

        batch = context["batch"]
        states = torch.from_numpy(batch["states"]).float()
        actions = torch.from_numpy(batch["actions"]).long()
        rewards = torch.from_numpy(batch["rewards"]).float()
        next_states = torch.from_numpy(batch["next_states"]).float()
        dones = torch.from_numpy(batch["dones"]).float()

        # Compute Q-values for current states
        q_values = self.network(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        loss = nn.functional.mse_loss(q_values_for_actions, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        context["loss"] = loss.item()

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set agent state from checkpoint."""
        self.network.load_state_dict(state["network_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

    def validate_config(self) -> bool:
        """Validate MLP agent configuration."""
        if self.state_dim <= 0 or self.action_dim <= 0:
            return False
        if self.learning_rate <= 0:
            return False
        return True
