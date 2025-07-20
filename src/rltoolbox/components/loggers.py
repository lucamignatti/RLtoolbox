"""
Basic logger components for RLtoolbox.

Provides logging functionality for metrics, console output, and file logging.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from ..core.base import RLComponent


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

    def validate_config(self) -> bool:
        """Validate logger configuration."""
        return self.log_frequency > 0


class FileLogger(RLComponent):
    """
    File logger that saves metrics and logs to files.

    Saves training metrics, episode data, and configuration to JSON files.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # File configuration
        self.log_dir = Path(config.get("log_dir", "./logs"))
        self.save_frequency = config.get("save_frequency", 100)  # Episodes
        self.save_config = config.get("save_config", True)

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize log data
        self.episode_data = []
        self.step_data = []

        # Generate unique run ID
        timestamp = int(time.time())
        self.run_id = f"run_{timestamp}"

        # Create run directory
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def training_start(self, context: Dict[str, Any]) -> None:
        """Save configuration at training start."""
        if self.save_config:
            config_path = self.run_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(context["config"], f, indent=2, default=str)

    def episode_end(self, context: Dict[str, Any]) -> None:
        """Save episode data."""
        episode_info = {
            "episode": context["training"]["episode"],
            "reward": context["training"]["episode_reward"],
            "length": context["training"]["episode_length"],
            "total_steps": context["training"]["total_steps"],
            "timestamp": time.time()
        }

        # Add any additional metrics from context
        if "epsilon" in context:
            episode_info["epsilon"] = context["epsilon"]
        if "temperature" in context:
            episode_info["temperature"] = context["temperature"]

        self.episode_data.append(episode_info)

        # Save periodically
        episode = context["training"]["episode"]
        if episode % self.save_frequency == 0:
            self._save_data()

    def step_end(self, context: Dict[str, Any]) -> None:
        """Save step data if configured."""
        # Only save step data if explicitly requested (can be memory intensive)
        if self.config.get("save_step_data", False):
            step_info = {
                "episode": context["training"]["episode"],
                "step": context["training"]["step"],
                "reward": context.get("reward", 0.0),
                "action": context.get("action", None),
                "done": context.get("done", False)
            }
            self.step_data.append(step_info)

    def training_end(self, context: Dict[str, Any]) -> None:
        """Save final data at training end."""
        self._save_data()

        # Save final summary
        summary = {
            "total_episodes": context["training"]["episode"],
            "total_steps": context["training"]["total_steps"],
            "final_metrics": context["metrics"],
            "run_id": self.run_id,
            "completion_time": time.time()
        }

        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def _save_data(self) -> None:
        """Save episode and step data to files."""
        # Save episode data
        if self.episode_data:
            episodes_path = self.run_dir / "episodes.json"
            with open(episodes_path, 'w') as f:
                json.dump(self.episode_data, f, indent=2, default=str)

        # Save step data
        if self.step_data:
            steps_path = self.run_dir / "steps.json"
            with open(steps_path, 'w') as f:
                json.dump(self.step_data, f, indent=2, default=str)

    def get_state(self) -> Dict[str, Any]:
        """Get logger state for checkpointing."""
        return {
            "episode_data": self.episode_data,
            "step_data": self.step_data,
            "run_id": self.run_id
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set logger state from checkpoint."""
        self.episode_data = state["episode_data"]
        self.step_data = state["step_data"]
        self.run_id = state["run_id"]

        # Recreate run directory
        self.run_dir = self.log_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def validate_config(self) -> bool:
        """Validate file logger configuration."""
        return self.save_frequency > 0


class CSVLogger(RLComponent):
    """
    CSV logger that saves metrics in CSV format for easy analysis.

    Creates separate CSV files for episode and step data.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # CSV configuration
        self.log_dir = Path(config.get("log_dir", "./logs"))
        self.filename_prefix = config.get("filename_prefix", "training")

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate unique filenames
        timestamp = int(time.time())
        self.episode_file = self.log_dir / f"{self.filename_prefix}_episodes_{timestamp}.csv"
        self.step_file = self.log_dir / f"{self.filename_prefix}_steps_{timestamp}.csv"

        # Initialize CSV files
        self.episode_file_initialized = False
        self.step_file_initialized = False

    def episode_end(self, context: Dict[str, Any]) -> None:
        """Write episode data to CSV."""
        import csv

        episode_data = {
            "episode": context["training"]["episode"],
            "reward": context["training"]["episode_reward"],
            "length": context["training"]["episode_length"],
            "total_steps": context["training"]["total_steps"],
            "timestamp": time.time()
        }

        # Add any additional metrics
        if "epsilon" in context:
            episode_data["epsilon"] = context["epsilon"]
        if "temperature" in context:
            episode_data["temperature"] = context["temperature"]

        # Write to CSV
        with open(self.episode_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=episode_data.keys())

            # Write header if first time
            if not self.episode_file_initialized:
                writer.writeheader()
                self.episode_file_initialized = True

            writer.writerow(episode_data)

    def step_end(self, context: Dict[str, Any]) -> None:
        """Write step data to CSV if configured."""
        if self.config.get("save_step_data", False):
            import csv

            step_data = {
                "episode": context["training"]["episode"],
                "step": context["training"]["step"],
                "reward": context.get("reward", 0.0),
                "action": context.get("action", None),
                "done": context.get("done", False),
                "timestamp": time.time()
            }

            # Write to CSV
            with open(self.step_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=step_data.keys())

                # Write header if first time
                if not self.step_file_initialized:
                    writer.writeheader()
                    self.step_file_initialized = True

                writer.writerow(step_data)

    def validate_config(self) -> bool:
        """Validate CSV logger configuration."""
        return True


class MetricsLogger(RLComponent):
    """
    Metrics logger that computes and tracks various training metrics.

    Computes running averages, best scores, and other useful statistics.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Metrics configuration
        self.window_size = config.get("window_size", 100)
        self.compute_frequency = config.get("compute_frequency", 1)  # Episodes

        # Metrics storage
        self.metrics_history = []

    def episode_end(self, context: Dict[str, Any]) -> None:
        """Compute and store metrics."""
        episode = context["training"]["episode"]

        if episode % self.compute_frequency == 0:
            rewards = context["metrics"]["episode_rewards"]
            lengths = context["metrics"]["episode_lengths"]

            if rewards:
                # Compute metrics
                metrics = {
                    "episode": episode,
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "min_reward": np.min(rewards),
                    "max_reward": np.max(rewards),
                    "mean_length": np.mean(lengths),
                    "std_length": np.std(lengths),
                    "total_episodes": len(rewards)
                }

                # Windowed metrics
                if len(rewards) >= self.window_size:
                    recent_rewards = rewards[-self.window_size:]
                    recent_lengths = lengths[-self.window_size:]

                    metrics.update({
                        f"mean_reward_{self.window_size}": np.mean(recent_rewards),
                        f"std_reward_{self.window_size}": np.std(recent_rewards),
                        f"mean_length_{self.window_size}": np.mean(recent_lengths),
                    })

                # Store metrics
                self.metrics_history.append(metrics)

                # Add to context for other components
                context["computed_metrics"] = metrics

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all computed metrics."""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]
        return {
            "latest_metrics": latest,
            "metrics_history": self.metrics_history,
            "num_metric_points": len(self.metrics_history)
        }

    def get_state(self) -> Dict[str, Any]:
        """Get metrics state for checkpointing."""
        return {
            "metrics_history": self.metrics_history
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set metrics state from checkpoint."""
        self.metrics_history = state["metrics_history"]

    def validate_config(self) -> bool:
        """Validate metrics logger configuration."""
        return self.window_size > 0 and self.compute_frequency > 0
