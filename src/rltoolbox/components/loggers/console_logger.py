import time
from typing import Dict, Any
import numpy as np

from ...core.base import RLComponent


class ConsoleLogger(RLComponent):
    """
    Simple console logger that prints training progress.

    Logs episode rewards, lengths, and other metrics to stdout.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Logging configuration
        self.log_frequency = config.get("log_frequency", 10)  # Episodes
        self.verbose = config.get("verbose", True)
        self.log_steps = config.get("log_steps", False)

        # Internal state
        self.start_time = None

    def training_start(self, context: Dict[str, Any]) -> None:
        """Log training start."""
        self.start_time = time.time()
        if self.verbose:
            print("=== Training Started ===")
            print(f"Configuration: {self.config}")

    def episode_end(self, context: Dict[str, Any]) -> None:
        """Log episode completion."""
        episode = context["training"]["episode"]

        if episode % self.log_frequency == 0:
            episode_reward = context["training"]["episode_reward"]
            episode_length = context["training"]["episode_length"]
            total_steps = context["training"]["total_steps"]

            # Calculate statistics
            rewards = context["metrics"]["episode_rewards"]
            if len(rewards) >= 100:
                mean_reward_100 = np.mean(rewards[-100:])
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Mean100: {mean_reward_100:8.2f} | "
                      f"Steps: {total_steps}")
            else:
                print(f"Episode {episode:4d} | "
                      f"Reward: {episode_reward:8.2f} | "
                      f"Length: {episode_length:3d} | "
                      f"Steps: {total_steps}")

    def step_end(self, context: Dict[str, Any]) -> None:
        """Log step completion if enabled."""
        if self.log_steps:
            step = context["training"]["step"]
            reward = context.get("reward", 0.0)
            action = context.get("action", None)

            print(f"  Step {step:3d} | Action: {action} | Reward: {reward:6.3f}")

    def training_end(self, context: Dict[str, Any]) -> None:
        """Log training completion."""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            total_episodes = context["training"]["episode"]
            total_steps = context["training"]["total_steps"]

            print("=== Training Completed ===")
            print(f"Total Episodes: {total_episodes}")
            print(f"Total Steps: {total_steps}")
            print(f"Training Time: {elapsed:.2f} seconds")

            # Final statistics
            rewards = context["metrics"]["episode_rewards"]
            if rewards:
                print(f"Final Mean Reward: {np.mean(rewards):8.2f}")
                print(f"Final Std Reward: {np.std(rewards):8.2f}")
                print(f"Best Episode Reward: {np.max(rewards):8.2f}")

    def evaluation_start(self, context: Dict[str, Any]) -> None:
        """Log evaluation start."""
        if self.verbose:
            print("=== Evaluation Started ===")

    def evaluation_episode_end(self, context: Dict[str, Any]) -> None:
        """Log evaluation episode completion."""
        episode = context["training"]["episode"]
        episode_reward = context["training"]["episode_reward"]
        episode_length = context["training"]["episode_length"]

        if self.verbose:
            print(f"Eval Episode {episode:2d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Length: {episode_length:3d}")

    def evaluation_end(self, context: Dict[str, Any]) -> None:
        """Log evaluation completion."""
        if self.verbose:
            print("=== Evaluation Completed ===")

    def validate_config(self) -> bool:
        """Validate logger configuration."""
        return self.log_frequency > 0
