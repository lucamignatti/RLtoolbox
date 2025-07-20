"""
Integration tests for the RLTrainer class.

Tests the complete training workflow including configuration loading,
component integration, and training execution.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from rltoolbox import RLTrainer
from rltoolbox.core.base import RLComponent


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
        assert len(trainer.components) == 3  # env, agent, logger
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

    def test_hook_execution_order(self):
        """Test that hooks execute in the correct order."""
        # Register tracking component
        from rltoolbox.core.registry import get_registry
        registry = get_registry()
        registry._components["test.TrackingComponent"] = TrackingComponent
        registry._packages["test"] = {"version": "dev", "components": ["TrackingComponent"]}

        config = {
            "seed": 42,
            "development_mode": True,
            "packages": {},
            "components": {
                "tracker1": {
                    "package": "test",
                    "type": "TrackingComponent",
                    "name": "tracker1"
                },
                "tracker2": {
                    "package": "test",
                    "type": "TrackingComponent",
                    "name": "tracker2"
                }
            },
            "hooks": {
                "training_start": ["tracker1"],
                "episode_reset": ["tracker1"],
                "action_selection": ["tracker1", "tracker2"],
                "environment_step": ["tracker2"],
                "learning_update": ["tracker2", "tracker1"],
                "episode_end": ["tracker1"],
                "training_end": ["tracker1"]
            },
            "training": {
                "max_episodes": 1,
                "max_steps_per_episode": 2
            }
        }

        trainer = RLTrainer(config_dict=config)
        trainer.train()

        # Verify execution order
        log = TrackingComponent.execution_log
        assert len(log) > 0

        # Check that hooks executed in correct order
        hook_sequence = [entry["hook"] for entry in log]
        assert hook_sequence[0] == "training_start"
        assert "episode_reset" in hook_sequence
        assert "action_selection" in hook_sequence
        assert "environment_step" in hook_sequence
        assert hook_sequence[-1] == "training_end"

    def test_multiple_component_calls_same_hook(self):
        """Test multiple calls to same component in one hook."""
        from rltoolbox.core.registry import get_registry
        registry = get_registry()
        registry._components["test.TrackingComponent"] = TrackingComponent
        registry._packages["test"] = {"version": "dev", "components": ["TrackingComponent"]}

        config = {
            "seed": 42,
            "development_mode": True,
            "packages": {},
            "components": {
                "tracker": {
                    "package": "test",
                    "type": "TrackingComponent",
                    "name": "tracker"
                }
            },
            "hooks": {
                "episode_reset": ["tracker"],
                "action_selection": ["tracker", "tracker", "tracker"],
                "environment_step": ["tracker"],
                "training_start": ["tracker"],
                "training_end": ["tracker"]
            },
            "training": {
                "max_episodes": 1,
                "max_steps_per_episode": 1
            }
        }

        trainer = RLTrainer(config_dict=config)
        trainer.train()

        # Find action_selection calls
        action_calls = [entry for entry in TrackingComponent.execution_log
                       if entry["hook"] == "action_selection"]

        # Should have 3 calls with increasing run_count
        assert len(action_calls) == 3
        assert action_calls[0]["run_count"] == 1
        assert action_calls[1]["run_count"] == 2
        assert action_calls[2]["run_count"] == 3

    def test_config_validation_errors(self):
        """Test configuration validation catches errors."""
        # Missing components section
        invalid_config = {
            "seed": 42,
            "hooks": {}
        }

        with pytest.raises(ValueError, match="Missing required configuration section: components"):
            RLTrainer(config_dict=invalid_config)

        # Missing hooks section
        invalid_config = {
            "seed": 42,
            "components": {}
        }

        with pytest.raises(ValueError, match="Missing required configuration section: hooks"):
            RLTrainer(config_dict=invalid_config)

        # Invalid component config
        invalid_config = {
            "seed": 42,
            "components": {
                "bad_comp": "not_a_dict"
            },
            "hooks": {}
        }

        with pytest.raises(ValueError, match="Component 'bad_comp' configuration must be a dictionary"):
            RLTrainer(config_dict=invalid_config)

    def test_training_termination_conditions(self):
        """Test different training termination conditions."""
        # Test max_episodes termination
        config = {
            "seed": 42,
            "development_mode": True,
            "packages": {},
            "components": {
                "env": {
                    "package": "rltoolbox",
                    "type": "RandomEnvironment",
                    "episode_length": 100  # Long episodes
                },
                "agent": {
                    "package": "rltoolbox",
                    "type": "RandomAgent",
                    "action_space_type": "discrete",
                    "num_actions": 2
                }
            },
            "hooks": {
                "episode_reset": ["env"],
                "action_selection": ["agent"],
                "environment_step": ["env"]
            },
            "training": {
                "max_episodes": 3,
                "max_steps_per_episode": 1000
            }
        }

        trainer = RLTrainer(config_dict=config)
        trainer.train()

        # Should stop at max_episodes
        assert trainer.context["training"]["episode"] == 2  # 0-indexed, so 3 episodes = episodes 0,1,2

        # Test max_steps_per_episode termination
        config["training"]["max_episodes"] = 100
        config["training"]["max_steps_per_episode"] = 2

        trainer = RLTrainer(config_dict=config)
        trainer.train()

        # Episodes should be short due to step limit
        episode_lengths = trainer.context["metrics"]["episode_lengths"]
        assert all(length <= 2 for length in episode_lengths)

    def test_context_data_flow(self, simple_config):
        """Test that context data flows correctly between components."""
        # Add a component that modifies context
        from rltoolbox.core.registry import get_registry
        registry = get_registry()

        class ContextModifier(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                context["action"] = 42
                context["custom_data"] = "test_value"

            def learning_update(self, context: Dict[str, Any]) -> None:
                if "custom_data" in context:
                    context["learning_used_custom"] = context["custom_data"]

        registry._components["test.ContextModifier"] = ContextModifier
        registry._packages["test"] = {"version": "dev", "components": ["ContextModifier"]}

        simple_config["components"]["modifier"] = {
            "package": "test",
            "type": "ContextModifier"
        }
        simple_config["hooks"]["action_selection"] = ["modifier"]
        simple_config["hooks"]["learning_update"] = ["modifier"]

        trainer = RLTrainer(config_dict=simple_config)
        trainer.train()

        # Check that context modifications persisted
        context = trainer.get_context()
        assert "learning_used_custom" in context
        assert context["learning_used_custom"] == "test_value"

    def test_error_handling_during_training(self):
        """Test error handling during training execution."""
        from rltoolbox.core.registry import get_registry
        registry = get_registry()

        class ErrorComponent(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                raise RuntimeError("Intentional test error")

        registry._components["test.ErrorComponent"] = ErrorComponent
        registry._packages["test"] = {"version": "dev", "components": ["ErrorComponent"]}

        config = {
            "seed": 42,
            "development_mode": True,
            "packages": {},
            "components": {
                "env": {
                    "package": "rltoolbox",
                    "type": "RandomEnvironment",
                    "episode_length": 5
                },
                "error_comp": {
                    "package": "test",
                    "type": "ErrorComponent"
                }
            },
            "hooks": {
                "episode_reset": ["env"],
                "action_selection": ["error_comp"],
                "environment_step": ["env"]
            },
            "training": {
                "max_episodes": 1
            }
        }

        trainer = RLTrainer(config_dict=config)

        # Training should raise the error
        with pytest.raises(RuntimeError):
            trainer.train()

    def test_metrics_collection(self, simple_config):
        """Test that training metrics are properly collected."""
        simple_config["training"]["max_episodes"] = 5

        trainer = RLTrainer(config_dict=simple_config)
        trainer.train()

        metrics = trainer.get_metrics()

        # Check metrics structure
        assert "episode_rewards" in metrics
        assert "episode_lengths" in metrics
        assert "losses" in metrics

        # Check metrics content
        assert len(metrics["episode_rewards"]) == 5
        assert len(metrics["episode_lengths"]) == 5
        assert all(isinstance(reward, (int, float)) for reward in metrics["episode_rewards"])
        assert all(isinstance(length, int) for length in metrics["episode_lengths"])

    def test_seed_reproducibility(self):
        """Test that same seed produces reproducible results."""
        config = {
            "seed": 12345,
            "development_mode": True,
            "packages": {},
            "components": {
                "env": {
                    "package": "rltoolbox",
                    "type": "RandomEnvironment",
                    "episode_length": 10
                },
                "agent": {
                    "package": "rltoolbox",
                    "type": "RandomAgent",
                    "action_space_type": "discrete",
                    "num_actions": 2
                }
            },
            "hooks": {
                "episode_reset": ["env"],
                "action_selection": ["agent"],
                "environment_step": ["env"]
            },
            "training": {
                "max_episodes": 3
            }
        }

        # Run training twice with same seed
        trainer1 = RLTrainer(config_dict=config.copy())
        trainer1.train()
        rewards1 = trainer1.get_metrics()["episode_rewards"]

        trainer2 = RLTrainer(config_dict=config.copy())
        trainer2.train()
        rewards2 = trainer2.get_metrics()["episode_rewards"]

        # Results should be identical (within floating point precision)
        assert len(rewards1) == len(rewards2)
        for r1, r2 in zip(rewards1, rewards2):
            assert abs(r1 - r2) < 1e-10

    def test_cartpole_training_workflow(self, cartpole_trainer):
        """Test training workflow with CartPole environment."""
        # Run a short training session
        cartpole_trainer.train()

        # Verify training completed successfully
        metrics = cartpole_trainer.get_metrics()
        assert len(metrics["episode_rewards"]) > 0
        assert len(metrics["episode_lengths"]) > 0

        # Verify CartPole-specific behavior
        context = cartpole_trainer.get_context()
        assert "epsilon" in context  # EpsilonGreedyAgent should set epsilon

    def test_trainer_context_readonly_copy(self, simple_config):
        """Test that get_context returns read-only copy."""
        trainer = RLTrainer(config_dict=simple_config)

        # Get context and modify it
        context = trainer.get_context()
        context["modified"] = True

        # Get context again - should not have modification
        context2 = trainer.get_context()
        assert "modified" not in context2
