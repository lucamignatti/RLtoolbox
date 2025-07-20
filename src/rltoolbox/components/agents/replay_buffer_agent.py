import numpy as np
import random
from typing import Dict, Any

from ...core.base import RLComponent


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
