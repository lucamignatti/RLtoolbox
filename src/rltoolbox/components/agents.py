"""
Basic agent components for RLtoolbox.

Provides simple agent implementations for action selection and learning.
"""

import numpy as np
import random
from typing import Dict, Any, Optional, List
from ..core.base import RLComponent


class RandomAgent(RLComponent):
    """
    Random agent that selects actions uniformly at random.

    Useful for baseline comparisons and testing.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Action space configuration
        self.action_space_type = config.get("action_space_type", "discrete")
        self.num_actions = config.get("num_actions", 2)
        self.action_low = config.get("action_low", -1.0)
        self.action_high = config.get("action_high", 1.0)
        self.action_dim = config.get("action_dim", 1)

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select random action."""
        if self.action_space_type == "discrete":
            action = random.randint(0, self.num_actions - 1)
        else:
            # Continuous action space
            if self.action_dim == 1:
                action = np.random.uniform(self.action_low, self.action_high)
            else:
                action = np.random.uniform(
                    self.action_low, self.action_high, size=self.action_dim
                )

        context["action"] = action

    def validate_config(self) -> bool:
        """Validate random agent configuration."""
        if self.action_space_type not in ["discrete", "continuous"]:
            return False

        if self.action_space_type == "discrete" and self.num_actions <= 0:
            return False

        return True


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


class GreedyAgent(RLComponent):
    """
    Greedy agent that always selects the action with highest Q-value.

    Useful for evaluation and testing.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_actions = config.get("num_actions", 2)

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select greedy action."""
        if "q_values" not in context:
            # Fallback to random if no Q-values available
            action = random.randint(0, self.num_actions - 1)
        else:
            q_values = context["q_values"]
            action = np.argmax(q_values)

        context["action"] = action

    def validate_config(self) -> bool:
        """Validate greedy agent configuration."""
        return self.num_actions > 0


class SoftmaxAgent(RLComponent):
    """
    Softmax agent that selects actions based on Boltzmann distribution.

    Uses temperature parameter to control exploration.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Temperature configuration
        self.temperature_start = config.get("temperature_start", 1.0)
        self.temperature_end = config.get("temperature_end", 0.1)
        self.temperature_decay = config.get("temperature_decay", 0.995)
        self.current_temperature = self.temperature_start

        # Action space
        self.num_actions = config.get("num_actions", 2)

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select action using softmax policy."""
        # Update temperature
        self._update_temperature()

        if "q_values" not in context:
            # Fallback to random if no Q-values available
            action = random.randint(0, self.num_actions - 1)
        else:
            q_values = np.array(context["q_values"])

            # Apply temperature scaling
            scaled_q = q_values / self.current_temperature

            # Softmax probabilities
            exp_q = np.exp(scaled_q - np.max(scaled_q))  # Numerical stability
            probabilities = exp_q / np.sum(exp_q)

            # Sample action
            action = np.random.choice(self.num_actions, p=probabilities)

        context["action"] = action
        context["temperature"] = self.current_temperature

    def _update_temperature(self) -> None:
        """Update temperature based on decay schedule."""
        self.current_temperature = max(
            self.temperature_end,
            self.current_temperature * self.temperature_decay
        )

    def get_state(self) -> Dict[str, Any]:
        """Get agent state for checkpointing."""
        return {
            "current_temperature": self.current_temperature
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set agent state from checkpoint."""
        self.current_temperature = state["current_temperature"]

    def validate_config(self) -> bool:
        """Validate softmax agent configuration."""
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            return False
        if self.num_actions <= 0:
            return False
        return True


class PolicyAgent(RLComponent):
    """
    Policy-based agent that selects actions from action probabilities.

    Expects action probabilities to be provided in context by another component.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_actions = config.get("num_actions", 2)
        self.deterministic = config.get("deterministic", False)

    def action_selection(self, context: Dict[str, Any]) -> None:
        """Select action based on policy probabilities."""
        if "action_probs" not in context:
            # Fallback to random if no probabilities available
            action = random.randint(0, self.num_actions - 1)
        else:
            action_probs = np.array(context["action_probs"])

            if self.deterministic:
                # Select action with highest probability
                action = np.argmax(action_probs)
            else:
                # Sample from probability distribution
                action = np.random.choice(self.num_actions, p=action_probs)

        context["action"] = action

    def validate_config(self) -> bool:
        """Validate policy agent configuration."""
        return self.num_actions > 0


class ReplayBufferAgent(RLComponent):
    """
    Agent component that manages experience replay buffer.

    Stores transitions and provides batches for learning.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Buffer configuration
        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 32)
        self.min_buffer_size = config.get("min_buffer_size", 1000)

        # Initialize buffer
        self.buffer = []
        self.buffer_index = 0
        self.buffer_full = False

    def experience_storage(self, context: Dict[str, Any]) -> None:
        """Store experience in replay buffer."""
        # Check if we have all required components
        required_keys = ["state", "action", "reward", "next_state", "done"]
        if not all(key in context for key in required_keys):
            return

        # Create transition
        transition = {
            "state": context["state"],
            "action": context["action"],
            "reward": context["reward"],
            "next_state": context["next_state"],
            "done": context["done"]
        }

        # Store transition
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.buffer_index] = transition
            self.buffer_index = (self.buffer_index + 1) % self.buffer_size
            self.buffer_full = True

    def learning_update(self, context: Dict[str, Any]) -> None:
        """Provide batch of experiences for learning."""
        if len(self.buffer) < self.min_buffer_size:
            return

        # Sample batch
        batch_indices = random.sample(range(len(self.buffer)), self.batch_size)
        batch = [self.buffer[i] for i in batch_indices]

        # Convert to arrays
        states = np.array([t["state"] for t in batch])
        actions = np.array([t["action"] for t in batch])
        rewards = np.array([t["reward"] for t in batch])
        next_states = np.array([t["next_state"] for t in batch])
        dones = np.array([t["done"] for t in batch])

        # Add batch to context
        context["batch"] = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "size": self.batch_size
        }

    def get_state(self) -> Dict[str, Any]:
        """Get buffer state for checkpointing."""
        return {
            "buffer": self.buffer,
            "buffer_index": self.buffer_index,
            "buffer_full": self.buffer_full
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set buffer state from checkpoint."""
        self.buffer = state["buffer"]
        self.buffer_index = state["buffer_index"]
        self.buffer_full = state["buffer_full"]

    def validate_config(self) -> bool:
        """Validate replay buffer configuration."""
        if self.buffer_size <= 0 or self.batch_size <= 0:
            return False
        if self.min_buffer_size > self.buffer_size:
            return False
        return True
