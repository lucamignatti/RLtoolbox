import wandb
from typing import Dict, Any

from ....core.base import RLComponent


class WandbLogger(RLComponent):
    """
    Logger that sends metrics to Weights & Biases.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Wandb configuration
        self.project = config.get("project", "rltoolbox")
        self.entity = config.get("entity", None)
        self.run_name = config.get("run_name", None)
        self.log_frequency = config.get("log_frequency", 1)  # Episodes

        # Initialize wandb
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config=config
        )

    def episode_end(self, context: Dict[str, Any]) -> None:
        """Log metrics to wandb."""
        episode = context["training"]["episode"]

        if episode % self.log_frequency == 0:
            metrics = {
                "episode": episode,
                "episode_reward": context["training"]["episode_reward"],
                "episode_length": context["training"]["episode_length"],
                "total_steps": context["training"]["total_steps"],
            }

            # Add any other metrics from context
            if "loss" in context:
                metrics["loss"] = context["loss"]
            if "epsilon" in context:
                metrics["epsilon"] = context["epsilon"]
            if "temperature" in context:
                metrics["temperature"] = context["temperature"]
            if "computed_metrics" in context:
                metrics.update(context["computed_metrics"])

            wandb.log(metrics)

    def training_end(self, context: Dict[str, Any]) -> None:
        """Finish wandb run."""
        wandb.finish()

    def validate_config(self) -> bool:
        """Validate wandb logger configuration."""
        return self.project is not None
