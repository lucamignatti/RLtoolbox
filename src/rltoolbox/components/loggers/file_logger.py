import json
import time
from pathlib import Path
from typing import Dict, Any

from ....core.base import RLComponent


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

    def evaluation_start(self, context: Dict[str, Any]) -> None:
        """Save evaluation start information."""
        eval_info = {
            "evaluation_started": True,
            "start_time": time.time(),
            "training_episodes_completed": context["training"]["episode"]
        }

        eval_path = self.run_dir / "evaluation_info.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_info, f, indent=2, default=str)

    def evaluation_episode_end(self, context: Dict[str, Any]) -> None:
        """Save evaluation episode data."""
        eval_episode_info = {
            "eval_episode": context["training"]["episode"],
            "reward": context["training"]["episode_reward"],
            "length": context["training"]["episode_length"],
            "timestamp": time.time()
        }

        # Save to separate evaluation episodes file
        eval_episodes_path = self.run_dir / "evaluation_episodes.json"
        if eval_episodes_path.exists():
            with open(eval_episodes_path, 'r') as f:
                eval_data = json.load(f)
        else:
            eval_data = []

        eval_data.append(eval_episode_info)

        with open(eval_episodes_path, 'w') as f:
            json.dump(eval_data, f, indent=2, default=str)

    def evaluation_end(self, context: Dict[str, Any]) -> None:
        """Save evaluation completion information."""
        # Load existing evaluation info
        eval_path = self.run_dir / "evaluation_info.json"
        if eval_path.exists():
            with open(eval_path, 'r') as f:
                eval_info = json.load(f)
        else:
            eval_info = {}

        # Add completion information
        eval_info.update({
            "evaluation_completed": True,
            "end_time": time.time(),
            "final_evaluation_results": context.get("final_evaluation", {})
        })

        with open(eval_path, 'w') as f:
            json.dump(eval_info, f, indent=2, default=str)

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
