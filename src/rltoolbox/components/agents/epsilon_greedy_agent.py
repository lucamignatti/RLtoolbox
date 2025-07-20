import numpy as np
import random
from typing import Dict, Any

from ....core.base import RLComponent


class EpsilonGreedyAgent(RLComponent):
    """
    Epsilon-greedy agent that selects actions based on Q-values.

    Requires Q-values to be provided in context by another component.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Epsilon configuration
        self.epsilon_start = config.get("epsilon_start", 1.0)
        self.epsilon_end = config.get("epsilon_end", 0.1)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)
        self.current_epsilon = self.epsilon_start

        # Action space
        self.num_actions = config.get("num_actions", 2)

        # Decay configuration
        self.decay_type = config.get("decay_type", "exponential")  # exponential, linear
        self.decay_steps = config.get("decay_steps", 10000)

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select action using epsilon-greedy policy."""
        # Update epsilon
        self._update_epsilon(context)

        # Epsilon-greedy action selection
        if random.random() < self.current_epsilon:
            # Random action
            action = random.randint(0, self.num_actions - 1)
        else:
            # Greedy action
            if "q_values" not in context:
                # Fallback to random if no Q-values available
                action = random.randint(0, self.num_actions - 1)
            else:
                q_values = context["q_values"]
                action = np.argmax(q_values)

        context["action"] = action
        context["epsilon"] = self.current_epsilon

    def _update_epsilon(self, context: Dict[str, Any]) -> None:
        """Update epsilon based on decay schedule."""
        if self.decay_type == "exponential":
            self.current_epsilon = max(
                self.epsilon_end,
                self.current_epsilon * self.epsilon_decay
            )
        elif self.decay_type == "linear":
            total_steps = context["training"]["total_steps"]
            if total_steps < self.decay_steps:
                self.current_epsilon = self.epsilon_start - (
                    self.epsilon_start - self.epsilon_end
                ) * (total_steps / self.decay_steps)
            else:
                self.current_epsilon = self.epsilon_end

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            "current_epsilon": self.current_epsilon
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set agent state from checkpoint."""
        self.current_epsilon = state["current_epsilon"]

    def validate_config(self) -> bool:
        """Validate epsilon-greedy agent configuration."""
        if self.epsilon_start < 0 or self.epsilon_end < 0:
            return False
        if self.epsilon_start < self.epsilon_end:
            return False
        if self.num_actions <= 0:
            return False
        if self.decay_type not in ["exponential", "linear"]:
            return False
        return True
