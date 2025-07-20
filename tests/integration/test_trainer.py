import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from rltoolbox import RLTrainer
from rltoolbox.core.base import RLComponent


@pytest.fixture
def simple_config():
    """Basic configuration for testing."""
    return {
        "seed": 42,
        "development_mode": True,
        "packages": {},
        "components": {
            "env": {
                "package": "rltoolbox",
                "type": "SimpleEnvironment",
                "env_name": "CartPole-v1",
                "max_episode_steps": 5
            },
            "agent": {
                "package": "rltoolbox",
                "type": "MLPAgent",
                "state_dim": 4,
                "action_dim": 2,
                "hidden_layers": [16, 16],
                "learning_rate": 0.01,
            },
            "exploration_agent": {
                "package": "rltoolbox",
                "type": "EpsilonGreedyAgent",
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 0.9,
                "num_actions": 2,
                "decay_type": "exponential"
            },
            "logger": {
                "package": "rltoolbox",
                "type": "ConsoleLogger",
                "log_frequency": 1,
                "verbose": False,
                "log_steps": False
            }
        },
        "hooks": {
            "training_start": ["logger"],
            "episode_reset": ["env"],
            "action_selection": ["agent", "exploration_agent"],
            "environment_step": ["env"],
            "episode_end": ["logger"],
            "training_end": ["logger"]
        },
        "training": {
            "max_episodes": 3,
            "max_steps_per_episode": 5
        }
    }


class TrackingComponent(RLComponent):
    """Component that tracks its execution for testing."""

    execution_log = []  # Class variable to track all executions

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.instance_calls = []
        TrackingComponent.execution_log = []  # Reset on new instance

    def episode_start(self, context: Dict[str, Any]) -> None:
        self._log_call("episode_start", context)

    def episode_reset(self, context: Dict[str, Any]) -> None:
        self._log_call("episode_reset", context)
        context["state"] = [0, 0, 0, 0]

    def action_selection(self, context: Dict[str, Any]) -> None:
        self._log_call("action_selection", context)
        context["action"] = 0

    def environment_step(self, context: Dict[str, Any]) -> None:
        self._log_call("environment_step", context)
        context["next_state"] = [1, 1, 1, 1]
        context["reward"] = 1.0
        context["done"] = context["training"]["step"] >= 3

    def learning_update(self, context: Dict[str, Any]) -> None:
        self._log_call("learning_update", context)

    def episode_end(self, context: Dict[str, Any]) -> None:
        self._log_call("episode_end", context)

    def training_start(self, context: Dict[str, Any]) -> None:
        self._log_call("training_start", context)

    def training_end(self, context: Dict[str, Any]) -> None:
        self._log_call("training_end", context)

    def _log_call(self, hook_name: str, context: Dict[str, Any]) -> None:
        call_info = {
            "component": self.name,
            "hook": hook_name,
            "run_count": context.get("_run_count", 1),
            "episode": context.get("training", {}).get("episode", 0),
            "step": context.get("training", {}).get("step", 0)
        }
        self.instance_calls.append(call_info)
        TrackingComponent.execution_log.append(call_info)


class TestRLTrainerIntegration:
    """Integration tests for RLTrainer."""

    def test_trainer_initialization_from_dict(self, simple_config):
        """Test trainer initialization from configuration dictionary."""
        trainer = RLTrainer(config_dict=simple_config)

        assert trainer.config == simple_config
        assert trainer.config_path is None
        assert len(trainer.components) == 4  # env, agent, exploration_agent, logger
        assert trainer.context is not None

    def test_trainer_initialization_from_file(self, temp_config_file):
        """Test trainer initialization from configuration file."""
        trainer = RLTrainer(config_path=temp_config_file)

        assert trainer.config_path == Path(temp_config_file)
        assert trainer.config is not None
        assert len(trainer.components) > 0

    def test_trainer_initialization_errors(self):
        """Test trainer initialization error cases."""
        # No config provided
        with pytest.raises(ValueError, match="Must provide either config_path or config_dict"):
            RLTrainer()

        # Both configs provided
        with pytest.raises(ValueError, match="Provide either config_path or config_dict, not both"):
            RLTrainer(config_path="test.json", config_dict={})

    def test_complete_training_workflow(self, simple_config):
        """Test complete training workflow from start to finish."""
        # Modify config for faster testing
        simple_config["training"]["max_episodes"] = 2
        simple_config["training"]["max_steps_per_episode"] = 3

        trainer = RLTrainer(config_dict=simple_config)

        # Run training
        trainer.train()

        # Verify training completed
        assert trainer.context["training"]["episode"] >= 1
        assert len(trainer.context["metrics"]["episode_rewards"]) >= 1
        assert len(trainer.context["metrics"]["episode_lengths"]) >= 1

    def test_evaluation_workflow(self, simple_config):
        """Test evaluation workflow."""
        trainer = RLTrainer(config_dict=simple_config)

        # Run evaluation
        results = trainer.evaluate(num_episodes=3)

        # Verify evaluation results
        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_length" in results
        assert "std_length" in results
        assert "episode_rewards" in results
        assert "episode_lengths" in results
        assert len(results["episode_rewards"]) == 3
        assert len(results["episode_lengths"]) == 3

    def test_checkpoint_save_and_load(self, simple_config, temp_checkpoint_file):
        """Test checkpoint saving and loading."""
        # Create trainer and run partial training
        trainer1 = RLTrainer(config_dict=simple_config)
        trainer1.context["test_value"] = "checkpoint_test"

        # Save checkpoint
        trainer1.save_checkpoint(temp_checkpoint_file)

        # Verify checkpoint file exists
        assert Path(temp_checkpoint_file).exists()

        # Load checkpoint into new trainer
        trainer2 = RLTrainer(config_dict=simple_config)
        trainer2.load_checkpoint(temp_checkpoint_file)

        # Verify checkpoint was loaded
        assert trainer2.context["test_value"] == "checkpoint_test"
