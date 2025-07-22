"""
Integration tests for RLtoolbox.

These tests verify that all components work together correctly
and test the full workflow from configuration to training.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

from rltoolbox.core.trainer import RLTrainer
from rltoolbox.core.base import RLComponent
from rltoolbox.core.registry import get_registry


class IntegrationEnvironment(RLComponent):
    """Simple environment for integration testing."""

    def __init__(self, config):
        super().__init__(config)
        self.state = [0.0, 0.0]
        self.step_count = 0
        self.max_steps = config.get("max_steps", 10)

    def episode_reset(self, context):
        """Reset environment."""
        self.state = [0.1, 0.2]
        self.step_count = 0
        context["state"] = self.state.copy()

    def environment_step(self, context):
        """Take environment step."""
        action = context.get("action", 0)

        # Simple dynamics: state changes based on action
        self.state[0] += 0.1 * action
        self.state[1] += 0.05 * (1 - action)
        self.step_count += 1

        # Simple reward: positive if both state values > 0.5
        reward = 1.0 if all(s > 0.5 for s in self.state) else 0.0
        done = self.step_count >= self.max_steps

        context["next_state"] = self.state.copy()
        context["reward"] = reward
        context["done"] = done


class IntegrationAgent(RLComponent):
    """Simple agent for integration testing."""

    def __init__(self, config):
        super().__init__(config)
        self.epsilon = config.get("epsilon", 0.1)
        self.action_count = [0, 0]

    def action_selection(self, context):
        """Select action using simple policy."""
        state = context.get("state", [0, 0])

        # Simple policy: choose action 1 if first state component < 0.5
        if state[0] < 0.5:
            action = 1
        else:
            action = 0

        self.action_count[action] += 1
        context["action"] = action

    def transition_received(self, context):
        """Process transition (could do learning here)."""
        # Simple adaptation: reduce epsilon over time
        if context["training"]["step"] % 10 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.99)


class IntegrationLogger(RLComponent):
    """Simple logger for integration testing."""

    def __init__(self, config):
        super().__init__(config)
        self.episode_logs = []
        self.training_logs = []

    def training_start(self, context):
        """Log training start."""
        self.training_logs.append("Training started")

    def episode_end(self, context):
        """Log episode end."""
        episode_data = {
            "episode": context["training"]["episode"],
            "reward": context["training"]["episode_reward"],
            "length": context["training"]["episode_length"]
        }
        self.episode_logs.append(episode_data)

    def training_end(self, context):
        """Log training end."""
        self.training_logs.append("Training completed")


@pytest.fixture
def integration_config():
    """Configuration for integration tests."""
    return {
        "seed": 42,
        "packages": {
            "integration_package": {
                "path": "/integration/path",
                "components": ["IntegrationEnvironment", "IntegrationAgent", "IntegrationLogger"]
            }
        },
        "components": {
            "environment": {
                "package": "integration_package",
                "type": "IntegrationEnvironment",
                "max_steps": 5
            },
            "agent": {
                "package": "integration_package",
                "type": "IntegrationAgent",
                "epsilon": 0.2
            },
            "logger": {
                "package": "integration_package",
                "type": "IntegrationLogger"
            }
        },
        "hooks": {
            "training_start": ["logger"],
            "episode_reset": ["environment"],
            "action_selection": ["agent"],
            "environment_step": ["environment"],
            "transition_received": ["agent"],
            "episode_end": ["logger"],
            "training_end": ["logger"]
        },
        "training": {
            "max_episodes": 3,
            "max_steps_per_episode": 5
        }
    }


@pytest.fixture
def mock_integration_import():
    """Mock import for integration components."""
    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.IntegrationEnvironment = IntegrationEnvironment
            mock_module.IntegrationAgent = IntegrationAgent
            mock_module.IntegrationLogger = IntegrationLogger
            mock_import.return_value = mock_module
            yield mock_import


def test_full_training_workflow(integration_config, mock_integration_import):
    """Test complete training workflow."""
    trainer = RLTrainer(config_dict=integration_config)

    # Verify components are created and available in context
    assert len(trainer.components) == 3
    assert "environment" in trainer.components
    assert "agent" in trainer.components
    assert "logger" in trainer.components
    assert "components" in trainer.context

    # Run training
    trainer.train()

    # Verify training completed
    logger = trainer.components["logger"]
    assert len(logger.episode_logs) == 3  # 3 episodes
    assert "Training started" in logger.training_logs
    assert "Training completed" in logger.training_logs

    # Verify metrics were collected
    metrics = trainer.get_metrics()
    assert len(metrics["episode_rewards"]) == 3
    assert len(metrics["episode_lengths"]) == 3

    # Verify all rewards are non-negative
    assert all(r >= 0 for r in metrics["episode_rewards"])


def test_component_interaction(integration_config, mock_integration_import):
    """Test that components can access each other through context."""
    trainer = RLTrainer(config_dict=integration_config)

    # Create a component that accesses others
    class InteractingComponent(RLComponent):
        def training_start(self, context):
            # Access other components through context
            env = self.get_component(context, "environment")
            agent = self.get_component(context, "agent")
            logger = self.get_component(context, "logger")

            context["interaction_test"] = {
                "env_name": env.name,
                "agent_epsilon": agent.epsilon,
                "logger_episodes": len(logger.episode_logs)
            }

    # Add interacting component
    trainer.components["interactor"] = InteractingComponent({"name": "interactor"})
    trainer.context["components"]["interactor"] = trainer.components["interactor"]

    # Add the component to the training_start hook
    if "training_start" not in trainer.config["hooks"]:
        trainer.config["hooks"]["training_start"] = []
    trainer.config["hooks"]["training_start"].append("interactor")

    # Execute training_start hook with the new component
    trainer._execute_hook("training_start")

    # Verify interaction worked
    assert "interaction_test" in trainer.context
    interaction_data = trainer.context["interaction_test"]
    assert interaction_data["env_name"] == "environment"
    assert interaction_data["agent_epsilon"] == 0.2
    assert interaction_data["logger_episodes"] == 0


def test_evaluation_workflow(integration_config, mock_integration_import):
    """Test evaluation workflow."""
    trainer = RLTrainer(config_dict=integration_config)

    # Run evaluation
    results = trainer.evaluate(num_episodes=2)

    # Verify evaluation results structure
    required_keys = ["mean_reward", "std_reward", "mean_length", "std_length",
                    "episode_rewards", "episode_lengths"]
    for key in required_keys:
        assert key in results

    # Verify correct number of episodes
    assert len(results["episode_rewards"]) == 2
    assert len(results["episode_lengths"]) == 2

    # Verify metrics are reasonable
    assert results["mean_reward"] >= 0
    assert results["mean_length"] > 0


def test_checkpoint_save_load_workflow(integration_config, mock_integration_import):
    """Test checkpoint save/load workflow."""
    trainer = RLTrainer(config_dict=integration_config)

    # Modify trainer state
    trainer.context["test_checkpoint_value"] = "checkpoint_test"
    trainer.components["agent"].epsilon = 0.05

    # Test checkpoint saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.json")

        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)

        # Verify checkpoint file exists
        assert os.path.exists(checkpoint_path)

        # Create new trainer and load checkpoint
        trainer2 = RLTrainer(config_dict=integration_config)
        trainer2.load_checkpoint(checkpoint_path)

        # Verify state was restored
        assert trainer2.context["test_checkpoint_value"] == "checkpoint_test"
        assert "components" in trainer2.context
        assert "agent" in trainer2.context["components"]


def test_configuration_validation(mock_integration_import):
    """Test configuration validation."""
    # Test missing required sections
    with pytest.raises(ValueError, match="Missing required configuration section"):
        RLTrainer(config_dict={"components": {}})

    with pytest.raises(ValueError, match="Missing required configuration section"):
        RLTrainer(config_dict={"hooks": {}})

    # Test invalid component configuration
    invalid_config = {
        "components": {"comp1": "invalid"},
        "hooks": {}
    }
    with pytest.raises(ValueError, match="must be a dictionary"):
        RLTrainer(config_dict=invalid_config)


def test_hook_execution_order(integration_config, mock_integration_import):
    """Test that hooks are executed in correct order."""
    # Use existing integration components and modify their behavior
    trainer = RLTrainer(config_dict=integration_config)

    # Track execution order by modifying the logger component
    execution_order = []

    class OrderTrackingLogger(IntegrationLogger):
        def __init__(self, config):
            super().__init__(config)
            self.order_id = config.get("order_id", "default")

        def training_start(self, context):
            execution_order.append(f"training_start_{self.order_id}")
            super().training_start(context)

    # Replace components with order tracking versions
    trainer.components["logger1"] = OrderTrackingLogger({"name": "logger1", "order_id": "A"})
    trainer.components["logger2"] = OrderTrackingLogger({"name": "logger2", "order_id": "C"})
    trainer.components["logger3"] = OrderTrackingLogger({"name": "logger3", "order_id": "B"})

    # Update context
    trainer.context["components"]["logger1"] = trainer.components["logger1"]
    trainer.context["components"]["logger2"] = trainer.components["logger2"]
    trainer.context["components"]["logger3"] = trainer.components["logger3"]

    # Set hook execution order
    trainer.config["hooks"]["training_start"] = ["logger1", "logger2", "logger3"]

    # Execute training_start hook
    trainer._execute_hook("training_start")

    # Verify execution order
    assert execution_order == ["training_start_A", "training_start_C", "training_start_B"]


def test_warning_for_missing_seed(integration_config, mock_integration_import):
    """Test that missing seed produces warning."""
    config = integration_config.copy()
    del config["seed"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trainer = RLTrainer(config_dict=config)

        # Should warn about missing seed
        assert len(w) >= 1
        assert any("No seed specified" in str(warning.message) for warning in w)


def test_automatic_evaluation(integration_config, mock_integration_import):
    """Test automatic evaluation after training."""
    config = integration_config.copy()
    config["evaluation"] = {
        "enabled": True,
        "num_episodes": 2
    }

    trainer = RLTrainer(config_dict=config)
    trainer.train()

    # Verify automatic evaluation was performed
    assert "final_evaluation" in trainer.context
    eval_results = trainer.context["final_evaluation"]
    assert len(eval_results["episode_rewards"]) == 2


def test_skip_automatic_evaluation(integration_config, mock_integration_import):
    """Test skipping automatic evaluation."""
    config = integration_config.copy()
    config["evaluation"] = {
        "enabled": True,
        "num_episodes": 2
    }

    trainer = RLTrainer(config_dict=config, skip_automatic_evaluation=True)
    trainer.train()

    # Verify automatic evaluation was skipped
    assert "final_evaluation" not in trainer.context


def test_max_total_steps_termination(integration_config, mock_integration_import):
    """Test training termination by max_total_steps."""
    config = integration_config.copy()
    config["training"] = {
        "max_episodes": 100,  # High number
        "max_total_steps": 8   # Should terminate first
    }

    trainer = RLTrainer(config_dict=config)
    trainer.train()

    # Should have terminated due to max_total_steps
    assert trainer.context["training"]["total_steps"] <= 8
    metrics = trainer.get_metrics()
    # Should have fewer episodes than max_episodes
    assert len(metrics["episode_rewards"]) < 100


def test_error_handling_in_components(integration_config, mock_integration_import):
    """Test error handling when components fail."""
    class FailingComponent(RLComponent):
        def training_start(self, context):
            raise RuntimeError("Component failed!")

    trainer = RLTrainer(config_dict=integration_config)
    trainer.components["failing"] = FailingComponent({"name": "failing"})
    trainer.context["components"]["failing"] = trainer.components["failing"]
    trainer.config["hooks"]["training_start"].append("failing")

    # Should propagate the error
    with pytest.raises(RuntimeError, match="Error executing training_start on component failing"):
        trainer.train()


def test_empty_hooks_config(integration_config, mock_integration_import):
    """Test training with empty hooks configuration."""
    config = integration_config.copy()
    config["hooks"] = {}

    trainer = RLTrainer(config_dict=config)
    # Should run without errors, just won't do much
    trainer.train()

    # Should still have basic context structure
    assert "training" in trainer.context
    assert "metrics" in trainer.context


def test_development_mode(integration_config, mock_integration_import):
    """Test development mode functionality."""
    config = integration_config.copy()
    config["development_mode"] = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trainer = RLTrainer(config_dict=config)

        # Should warn about development mode
        assert any("Development mode enabled" in str(warning.message) for warning in w)


def test_context_persistence_across_episodes(integration_config, mock_integration_import):
    """Test that context persists correctly across episodes."""
    class ContextTrackingComponent(RLComponent):
        def episode_start(self, context):
            if "episode_starts" not in context:
                context["episode_starts"] = 0
            context["episode_starts"] += 1

    trainer = RLTrainer(config_dict=integration_config)
    tracker = ContextTrackingComponent({"name": "tracker"})
    trainer.components["tracker"] = tracker
    trainer.context["components"]["tracker"] = tracker
    trainer.config["hooks"]["episode_start"] = ["tracker"]

    trainer.train()

    # Should have tracked all episode starts
    assert trainer.context["episode_starts"] == 3


def test_get_context_method(integration_config, mock_integration_import):
    """Test trainer's get_context method returns read-only copy."""
    trainer = RLTrainer(config_dict=integration_config)

    context_copy = trainer.get_context()

    # Should be a copy, not the same object
    assert context_copy is not trainer.context

    # Should have same content
    assert context_copy["config"] == trainer.context["config"]
    assert "components" in context_copy

    # Modifying copy shouldn't affect original
    context_copy["test_modification"] = "test"
    assert "test_modification" not in trainer.context
