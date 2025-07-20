from typing import Dict, Any
import numpy as np

from ....core.base import RLComponent


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

    def evaluation_start(self, context: Dict[str, Any]) -> None:
        """Initialize evaluation metrics tracking."""
        # Store evaluation metrics separately
        context["evaluation_metrics"] = {
            "episode_rewards": [],
            "episode_lengths": []
        }

    def evaluation_episode_end(self, context: Dict[str, Any]) -> None:
        """Track evaluation episode metrics."""
        if "evaluation_metrics" in context:
            context["evaluation_metrics"]["episode_rewards"].append(
                context["training"]["episode_reward"]
            )
            context["evaluation_metrics"]["episode_lengths"].append(
                context["training"]["episode_length"]
            )

    def evaluation_end(self, context: Dict[str, Any]) -> None:
        """Compute final evaluation metrics."""
        if "evaluation_metrics" in context:
            eval_rewards = context["evaluation_metrics"]["episode_rewards"]
            eval_lengths = context["evaluation_metrics"]["episode_lengths"]

            if eval_rewards:
                eval_summary = {
                    "mean_reward": np.mean(eval_rewards),
                    "std_reward": np.std(eval_rewards),
                    "min_reward": np.min(eval_rewards),
                    "max_reward": np.max(eval_rewards),
                    "mean_length": np.mean(eval_lengths),
                    "std_length": np.std(eval_lengths),
                    "num_episodes": len(eval_rewards)
                }

                # Add to context for other components
                context["computed_evaluation_metrics"] = eval_summary

    def validate_config(self) -> bool:
        """Validate metrics logger configuration."""
        return self.window_size > 0 and self.compute_frequency > 0
