import csv
from pathlib import Path
from typing import Dict, Any
import time

from ...core.base import RLComponent


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
