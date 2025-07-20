"""
Unit tests for the base RLComponent class.

Tests the core functionality of the base component class including
initialization, hook methods, and state management.
"""

import pytest
from typing import Dict, Any

from rltoolbox.core.base import RLComponent


class TestRLComponent:
    """Test the base RLComponent class."""

    def test_initialization(self):
        """Test component initialization."""
        config = {"param1": "value1", "param2": 42}
        component = RLComponent(config)

        assert component.config == config
        assert component.name == "RLComponent"

    def test_initialization_with_name(self):
        """Test component initialization with custom name."""
        config = {"name": "custom_component", "param1": "value1"}
        component = RLComponent(config)

        assert component.config == config
        assert component.name == "custom_component"

    def test_all_hooks_are_no_ops(self):
        """Test that all hook methods are implemented as no-ops."""
        config = {}
        component = RLComponent(config)
        context = {"state": [1, 2, 3, 4]}

        # Test that all hook methods can be called without error
        hook_methods = [
            "training_start", "training_end",
            "episode_start", "episode_reset", "episode_end",
            "step_start", "observation_received", "action_selection",
            "action_chosen", "environment_step", "transition_received",
            "experience_storage", "learning_update", "step_end",
            "evaluation_start", "evaluation_episode_start",
            "evaluation_episode_end", "evaluation_end",
            "metrics_computation", "checkpoint_save", "logging_update",
            "visualization_update", "epoch_start", "epoch_end",
            "batch_start", "batch_end"
        ]

        for hook_name in hook_methods:
            hook_method = getattr(component, hook_name)
            # Should not raise any exception
            result = hook_method(context)
            # Should return None (no-op)
            assert result is None

    def test_context_modification(self):
        """Test that components can modify context."""
        class TestComponent(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                context["action"] = 42
                context["test_value"] = "modified"

        config = {}
        component = TestComponent(config)
        context = {"state": [1, 2, 3, 4]}

        component.action_selection(context)

        assert context["action"] == 42
        assert context["test_value"] == "modified"
        assert context["state"] == [1, 2, 3, 4]  # Original value preserved

    def test_run_count_access(self):
        """Test that components can access run count from context."""
        class TestComponent(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                run_count = context.get("_run_count", 1)
                context["run_count_accessed"] = run_count

        config = {}
        component = TestComponent(config)

        # Test with no run count
        context = {}
        component.action_selection(context)
        assert context["run_count_accessed"] == 1

        # Test with run count
        context = {"_run_count": 3}
        component.action_selection(context)
        assert context["run_count_accessed"] == 3

    def test_validate_config_default(self):
        """Test default config validation."""
        config = {}
        component = RLComponent(config)

        # Default implementation should return True
        assert component.validate_config() is True

    def test_get_state_default(self):
        """Test default state getter."""
        config = {}
        component = RLComponent(config)

        # Default implementation should return empty dict
        state = component.get_state()
        assert state == {}
        assert isinstance(state, dict)

    def test_set_state_default(self):
        """Test default state setter."""
        config = {}
        component = RLComponent(config)

        # Default implementation should not raise error
        component.set_state({"some_state": "value"})
        # No way to verify since default implementation does nothing

    def test_inheritance_and_override(self):
        """Test that subclasses can override methods correctly."""
        class CustomComponent(RLComponent):
            def __init__(self, config: Dict[str, Any]):
                super().__init__(config)
                self.custom_state = "initial"

            def action_selection(self, context: Dict[str, Any]) -> None:
                context["action"] = self.config.get("default_action", 0)
                self.custom_state = "action_selected"

            def learning_update(self, context: Dict[str, Any]) -> None:
                if "loss" in context:
                    self.custom_state = f"learned_with_loss_{context['loss']}"

            def validate_config(self) -> bool:
                return "required_param" in self.config

            def get_state(self) -> Dict[str, Any]:
                return {"custom_state": self.custom_state}

            def set_state(self, state: Dict[str, Any]) -> None:
                self.custom_state = state.get("custom_state", "initial")

        # Test with valid config
        config = {"required_param": "value", "default_action": 5}
        component = CustomComponent(config)

        assert component.validate_config() is True
        assert component.custom_state == "initial"

        # Test action selection
        context = {}
        component.action_selection(context)
        assert context["action"] == 5
        assert component.custom_state == "action_selected"

        # Test learning update
        context = {"loss": 0.5}
        component.learning_update(context)
        assert component.custom_state == "learned_with_loss_0.5"

        # Test state management
        state = component.get_state()
        assert state == {"custom_state": "learned_with_loss_0.5"}

        component.set_state({"custom_state": "restored"})
        assert component.custom_state == "restored"

        # Test with invalid config
        invalid_config = {"default_action": 5}  # Missing required_param
        invalid_component = CustomComponent(invalid_config)
        assert invalid_component.validate_config() is False

    def test_multiple_context_modifications(self):
        """Test that multiple hooks can modify context sequentially."""
        class MultiHookComponent(RLComponent):
            def episode_start(self, context: Dict[str, Any]) -> None:
                context["episode_data"] = []

            def action_selection(self, context: Dict[str, Any]) -> None:
                context["action"] = 1
                if "episode_data" in context:
                    context["episode_data"].append("action_selected")

            def environment_step(self, context: Dict[str, Any]) -> None:
                context["next_state"] = [0, 1, 2, 3]
                context["reward"] = 1.0
                context["done"] = False
                if "episode_data" in context:
                    context["episode_data"].append("env_stepped")

        config = {}
        component = MultiHookComponent(config)
        context = {}

        # Call hooks in sequence
        component.episode_start(context)
        assert context["episode_data"] == []

        component.action_selection(context)
        assert context["action"] == 1
        assert context["episode_data"] == ["action_selected"]

        component.environment_step(context)
        assert context["next_state"] == [0, 1, 2, 3]
        assert context["reward"] == 1.0
        assert context["done"] is False
        assert context["episode_data"] == ["action_selected", "env_stepped"]

    def test_config_immutability(self):
        """Test that modifying config doesn't affect other instances."""
        config1 = {"param": "value1"}
        config2 = {"param": "value2"}

        component1 = RLComponent(config1)
        component2 = RLComponent(config2)

        # Modify config1
        config1["param"] = "modified"

        # component1 should see the change (same reference)
        assert component1.config["param"] == "modified"
        # component2 should be unaffected
        assert component2.config["param"] == "value2"

    def test_context_sharing_between_calls(self):
        """Test that context is properly shared between multiple hook calls."""
        class ContextTrackingComponent(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                if "call_count" not in context:
                    context["call_count"] = 0
                context["call_count"] += 1
                context["action"] = context["call_count"]

        config = {}
        component = ContextTrackingComponent(config)
        context = {}

        # First call
        component.action_selection(context)
        assert context["call_count"] == 1
        assert context["action"] == 1

        # Second call with same context
        component.action_selection(context)
        assert context["call_count"] == 2
        assert context["action"] == 2

    def test_error_handling_in_hooks(self):
        """Test that errors in hook methods are properly raised."""
        class ErrorComponent(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                if "trigger_error" in context:
                    raise ValueError("Test error")
                context["action"] = 0

        config = {}
        component = ErrorComponent(config)

        # Normal operation should work
        context = {}
        component.action_selection(context)
        assert context["action"] == 0

        # Error should be raised
        context = {"trigger_error": True}
        with pytest.raises(ValueError, match="Test error"):
            component.action_selection(context)
