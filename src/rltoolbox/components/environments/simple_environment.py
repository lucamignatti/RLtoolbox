import gymnasium as gym
import numpy as np
from typing import Dict, Any

from ...core.base import RLComponent


class SimpleEnvironment(RLComponent):
    """
    Simple environment wrapper for Gymnasium environments.

    Handles environment creation, reset, and step operations.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Get environment configuration
        self.env_name = config.get("env_name", "CartPole-v1")
        self.render_mode = config.get("render_mode", None)
        self.max_episode_steps = config.get("max_episode_steps", None)

        # Create environment
        self.env = gym.make(
            self.env_name,
            render_mode=self.render_mode,
            max_episode_steps=self.max_episode_steps
        )

        # Store environment info
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def episode_reset(self, context: Dict[str, Any]) -> None:
        """Reset environment and set initial state."""
        observation, info = self.env.reset()
        context["state"] = observation
        context["info"] = info

    def environment_step(self, context: Dict[str, Any]) -> None:
        """Step environment with current action."""
        if "action" not in context:
            raise ValueError("No action available in context for environment step")

        action = context["action"]

        # Step environment
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update context
        context["next_state"] = observation
        context["reward"] = reward
        context["done"] = terminated or truncated
        context["terminated"] = terminated
        context["truncated"] = truncated
        context["info"] = info

        # Update state for next step
        context["state"] = observation

    def training_end(self, context: Dict[str, Any]) -> None:
        """Close environment at end of training."""
        if hasattr(self, 'env'):
            self.env.close()

    def get_state(self) -> Dict[str, Any]:
        """Get environment state for checkpointing."""
        return {
            "env_name": self.env_name,
            "render_mode": self.render_mode,
            "max_episode_steps": self.max_episode_steps
        }

    def validate_config(self) -> bool:
        """Validate environment configuration."""
        required_keys = []  # No required keys for basic environment

        for key in required_keys:
            if key not in self.config:
                return False

        return True
