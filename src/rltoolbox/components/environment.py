"""
Basic environment components for RLtoolbox.

Provides simple environment wrappers and implementations.
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional

from ..core.base import RLComponent


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


class RandomEnvironment(RLComponent):
    """
    Random environment for testing.

    Generates random observations and rewards.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Environment parameters
        self.state_dim = config.get("state_dim", 4)
        self.action_dim = config.get("action_dim", 2)
        self.episode_length = config.get("episode_length", 200)
        self.reward_range = config.get("reward_range", [-1.0, 1.0])

        # Internal state
        self.current_step = 0

    def episode_reset(self, context: Dict[str, Any]) -> None:
        """Reset to random initial state."""
        self.current_step = 0
        context["state"] = np.random.randn(self.state_dim)

    def environment_step(self, context: Dict[str, Any]) -> None:
        """Generate random next state and reward."""
        if "action" not in context:
            raise ValueError("No action available in context for environment step")

        self.current_step += 1

        # Generate random next state and reward
        context["next_state"] = np.random.randn(self.state_dim)
        context["reward"] = np.random.uniform(*self.reward_range)
        context["done"] = self.current_step >= self.episode_length

        # Update state
        context["state"] = context["next_state"]

    def validate_config(self) -> bool:
        """Validate random environment configuration."""
        return True


class EnvironmentWrapper(RLComponent):
    """
    Generic environment wrapper that can modify observations and rewards.

    Useful for preprocessing, normalization, etc.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Wrapper configuration
        self.normalize_observations = config.get("normalize_observations", False)
        self.clip_rewards = config.get("clip_rewards", False)
        self.reward_scale = config.get("reward_scale", 1.0)

        # Statistics for normalization
        self.obs_mean = None
        self.obs_std = None
        self.obs_count = 0

    def observation_received(self, context: Dict[str, Any]) -> None:
        """Process observations."""
        if "state" in context and self.normalize_observations:
            context["state"] = self._normalize_observation(context["state"])

        if "next_state" in context and self.normalize_observations:
            context["next_state"] = self._normalize_observation(context["next_state"])

    def transition_received(self, context: Dict[str, Any]) -> None:
        """Process rewards."""
        if "reward" in context:
            reward = context["reward"]

            # Scale reward
            reward = reward * self.reward_scale

            # Clip reward
            if self.clip_rewards:
                reward = np.clip(reward, -1.0, 1.0)

            context["reward"] = reward

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation using running statistics."""
        obs = np.array(obs)

        # Update statistics
        if self.obs_mean is None:
            self.obs_mean = obs.copy()
            self.obs_std = np.ones_like(obs)
        else:
            # Running average
            self.obs_count += 1
            alpha = 1.0 / self.obs_count
            self.obs_mean = (1 - alpha) * self.obs_mean + alpha * obs
            self.obs_std = np.sqrt((1 - alpha) * self.obs_std**2 + alpha * (obs - self.obs_mean)**2)

        # Normalize
        normalized = (obs - self.obs_mean) / (self.obs_std + 1e-8)
        return normalized

    def get_state(self) -> Dict[str, Any]:
        """Get wrapper state for checkpointing."""
        return {
            "obs_mean": self.obs_mean.tolist() if self.obs_mean is not None else None,
            "obs_std": self.obs_std.tolist() if self.obs_std is not None else None,
            "obs_count": self.obs_count
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set wrapper state from checkpoint."""
        if state["obs_mean"] is not None:
            self.obs_mean = np.array(state["obs_mean"])
            self.obs_std = np.array(state["obs_std"])
            self.obs_count = state["obs_count"]

    def validate_config(self) -> bool:
        """Validate wrapper configuration."""
        return True
