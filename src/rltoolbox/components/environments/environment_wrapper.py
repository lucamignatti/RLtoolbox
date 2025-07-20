import numpy as np
from typing import Dict, Any

from ...core.base import RLComponent


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
