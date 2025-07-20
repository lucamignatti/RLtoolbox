"""
Unit tests for the hook execution system.

Tests hook execution, validation, and run count tracking.
"""

import pytest
import warnings
from typing import Dict, Any
from unittest.mock import MagicMock

from rltoolbox.core.hooks import HookExecutor
from rltoolbox.core.base import RLComponent


class MockComponent(RLComponent):
    """Mock component for testing hook execution."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.calls = []

    def action_selection(self, context: Dict[str, Any]) -> None:
        run_count = context.get("_run_count", 1)
        self.calls.append(("action_selection", run_count))
        context["action"] = self.config.get("default_action", 0)

    def environment_step(self, context: Dict[str, Any]) -> None:
        run_count = context.get("_run_count", 1)
        self.calls.append(("environment_step", run_count))
        context["next_state"] = [0, 1, 2, 3]
        context["reward"] = 1.0
        context["done"] = False

    def learning_update(self, context: Dict[str, Any]) -> None:
        run_count = context.get("_run_count", 1)
        self.calls.append(("learning_update", run_count))

    def episode_reset(self, context: Dict[str, Any]) -> None:
        run_count = context.get("_run_count", 1)
        self.calls.append(("episode_reset", run_count))
        context["state"] = [0, 0, 0, 0]


class PartialComponent:
    """Component that doesn't inherit from RLComponent and doesn't have all hooks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calls = []

    def action_selection(self, context: Dict[str, Any]) -> None:
        self.calls.append("action_selection")
        context["action"] = 42

    # Note: doesn't implement environment_step at all


class TestHookExecutor:
    """Test the hook execution system."""

    def test_initialization(self):
        """Test hook executor initialization."""
        executor = HookExecutor()

        assert executor.validation_enabled is True
        assert executor._hook_run_counts == {}
        assert len(executor.HOOK_EXPECTATIONS) > 0

    def test_validation_toggle(self):
        """Test enabling/disabling validation."""
        executor = HookExecutor()

        # Initially enabled
        assert executor.validation_enabled is True

        # Disable
        executor.set_validation_enabled(False)
        assert executor.validation_enabled is False

        # Re-enable
        executor.set_validation_enabled(True)
        assert executor.validation_enabled is True

    def test_execute_hook_single_component(self):
        """Test executing hook with single component."""
        executor = HookExecutor()
        component = MockComponent({"name": "test_comp"})
        components = {"test_comp": component}
        context = {}

        executor.execute_hook("action_selection", ["test_comp"], components, context)

        assert len(component.calls) == 1
        assert component.calls[0] == ("action_selection", 1)
        assert context["action"] == 0
        assert context["_run_count"] == 1

    def test_execute_hook_multiple_components(self):
        """Test executing hook with multiple components."""
        executor = HookExecutor()
        comp1 = MockComponent({"name": "comp1", "default_action": 1})
        comp2 = MockComponent({"name": "comp2", "default_action": 2})
        components = {"comp1": comp1, "comp2": comp2}
        context = {}

        executor.execute_hook("action_selection", ["comp1", "comp2"], components, context)

        assert len(comp1.calls) == 1
        assert len(comp2.calls) == 1
        assert comp1.calls[0] == ("action_selection", 1)
        assert comp2.calls[0] == ("action_selection", 1)
        # Last component should set the action
        assert context["action"] == 2

    def test_execute_hook_multiple_calls_same_component(self):
        """Test executing hook with same component multiple times."""
        executor = HookExecutor()
        component = MockComponent({"name": "test_comp"})
        components = {"test_comp": component}
        context = {}

        executor.execute_hook(
            "action_selection",
            ["test_comp", "test_comp", "test_comp"],
            components,
            context
        )

        assert len(component.calls) == 3
        assert component.calls[0] == ("action_selection", 1)
        assert component.calls[1] == ("action_selection", 2)
        assert component.calls[2] == ("action_selection", 3)

    def test_run_counts_in_context(self):
        """Test that run counts are properly added to context."""
        executor = HookExecutor()
        comp1 = MockComponent({"name": "comp1"})
        comp2 = MockComponent({"name": "comp2"})
        components = {"comp1": comp1, "comp2": comp2}
        context = {}

        executor.execute_hook(
            "action_selection",
            ["comp1", "comp2", "comp1"],
            components,
            context
        )

        # Check final run counts
        assert context["_run_count"] == 2  # Last component's run count
        assert context["_run_counts"]["comp1"] == 2
        assert context["_run_counts"]["comp2"] == 1

    def test_hook_expectations_validation(self):
        """Test validation of expected context modifications."""
        executor = HookExecutor()
        component = MockComponent({"name": "test_comp"})
        components = {"test_comp": component}
        context = {}

        # Test hook that should add required keys
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("episode_reset", ["test_comp"], components, context)

        # Should not warn because episode_reset adds "state"
        validation_warnings = [warning for warning in w if "did not add expected context keys" in str(warning.message)]
        assert len(validation_warnings) == 0
        assert "state" in context

    def test_missing_expected_keys_warning(self):
        """Test warning when expected keys are missing."""
        executor = HookExecutor()

        # Component that doesn't add required keys
        class BadComponent(RLComponent):
            def episode_reset(self, context: Dict[str, Any]) -> None:
                pass  # Doesn't add "state"

        component = BadComponent({})
        components = {"bad_comp": component}
        context = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("episode_reset", ["bad_comp"], components, context)

        # Should warn about missing "state"
        validation_warnings = [warning for warning in w if "did not add expected context keys" in str(warning.message)]
        assert len(validation_warnings) == 1
        assert "state" in str(validation_warnings[0].message)

    def test_none_values_warning(self):
        """Test warning when expected keys have None values."""
        executor = HookExecutor()

        # Component that adds None values
        class NoneComponent(RLComponent):
            def episode_reset(self, context: Dict[str, Any]) -> None:
                context["state"] = None

        component = NoneComponent({})
        components = {"none_comp": component}
        context = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("episode_reset", ["none_comp"], components, context)

        # Should warn about None values
        validation_warnings = [warning for warning in w if "but with None values" in str(warning.message)]
        assert len(validation_warnings) == 1

    def test_validation_disabled(self):
        """Test that validation can be disabled."""
        executor = HookExecutor()
        executor.set_validation_enabled(False)

        # Component that doesn't add required keys
        class BadComponent(RLComponent):
            def episode_reset(self, context: Dict[str, Any]) -> None:
                pass  # Doesn't add "state"

        component = BadComponent({})
        components = {"bad_comp": component}
        context = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("episode_reset", ["bad_comp"], components, context)

        # Should not warn when validation is disabled
        validation_warnings = [warning for warning in w if "did not add expected context keys" in str(warning.message)]
        assert len(validation_warnings) == 0

    def test_component_not_found_error(self):
        """Test error when component is not found."""
        executor = HookExecutor()
        components = {}
        context = {}

        with pytest.raises(ValueError, match="Component 'nonexistent' not found"):
            executor.execute_hook("action_selection", ["nonexistent"], components, context)

    def test_component_missing_hook_method(self):
        """Test warning when component doesn't implement hook."""
        executor = HookExecutor()
        component = PartialComponent({})
        components = {"partial": component}
        context = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("environment_step", ["partial"], components, context)

        # Should warn about missing hook implementation
        hook_warnings = [warning for warning in w if "does not implement hook" in str(warning.message)]
        assert len(hook_warnings) == 1

    def test_hook_method_error_propagation(self):
        """Test that errors in hook methods are properly propagated."""
        executor = HookExecutor()

        class ErrorComponent(RLComponent):
            def action_selection(self, context: Dict[str, Any]) -> None:
                raise ValueError("Test error in hook")

        component = ErrorComponent({})
        components = {"error_comp": component}
        context = {}

        with pytest.raises(RuntimeError, match="Error executing action_selection on component error_comp"):
            executor.execute_hook("action_selection", ["error_comp"], components, context)

    def test_unknown_hook_warning(self):
        """Test warning for unknown hook names."""
        executor = HookExecutor()
        component = MockComponent({})
        components = {"test_comp": component}
        context = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("unknown_hook", ["test_comp"], components, context)

        # Should warn about unknown hook
        unknown_warnings = [warning for warning in w if "Unknown hook" in str(warning.message)]
        assert len(unknown_warnings) == 1

    def test_get_hook_names(self):
        """Test getting list of known hook names."""
        executor = HookExecutor()
        hook_names = executor.get_hook_names()

        assert isinstance(hook_names, list)
        assert len(hook_names) > 0
        assert "action_selection" in hook_names
        assert "environment_step" in hook_names
        assert "episode_reset" in hook_names

    def test_add_custom_hook(self):
        """Test adding custom hook with expectations."""
        executor = HookExecutor()

        # Add custom hook
        executor.add_custom_hook("custom_hook", ["custom_key1", "custom_key2"])

        # Verify it's added
        assert "custom_hook" in executor.HOOK_EXPECTATIONS
        assert executor.HOOK_EXPECTATIONS["custom_hook"] == ["custom_key1", "custom_key2"]

        # Test validation works for custom hook
        class CustomComponent(RLComponent):
            def custom_hook(self, context: Dict[str, Any]) -> None:
                context["custom_key1"] = "value1"
                # Missing custom_key2

        component = CustomComponent({})
        components = {"custom_comp": component}
        context = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.execute_hook("custom_hook", ["custom_comp"], components, context)

        # Should warn about missing custom_key2
        validation_warnings = [warning for warning in w if "did not add expected context keys" in str(warning.message)]
        assert len(validation_warnings) == 1
        assert "custom_key2" in str(validation_warnings[0].message)

    def test_validate_hook_configuration(self):
        """Test validation of hook configuration."""
        executor = HookExecutor()

        # Valid configuration
        valid_config = {
            "action_selection": ["comp1"],
            "environment_step": ["comp2"]
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.validate_hook_configuration(valid_config)

        unknown_warnings = [warning for warning in w if "Unknown hooks" in str(warning.message)]
        assert len(unknown_warnings) == 0

        # Invalid configuration with unknown hooks
        invalid_config = {
            "action_selection": ["comp1"],
            "unknown_hook1": ["comp2"],
            "unknown_hook2": ["comp3"]
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            executor.validate_hook_configuration(invalid_config)

        unknown_warnings = [warning for warning in w if "Unknown hooks" in str(warning.message)]
        assert len(unknown_warnings) == 1
        assert "unknown_hook1" in str(unknown_warnings[0].message)
        assert "unknown_hook2" in str(unknown_warnings[0].message)

    def test_hook_expectations_constants(self):
        """Test that hook expectations are properly defined."""
        executor = HookExecutor()

        # Check some key expectations
        assert "episode_reset" in executor.HOOK_EXPECTATIONS
        assert "state" in executor.HOOK_EXPECTATIONS["episode_reset"]

        assert "action_selection" in executor.HOOK_EXPECTATIONS
        assert "action" in executor.HOOK_EXPECTATIONS["action_selection"]

        assert "environment_step" in executor.HOOK_EXPECTATIONS
        expected_env_keys = executor.HOOK_EXPECTATIONS["environment_step"]
        assert "next_state" in expected_env_keys
        assert "reward" in expected_env_keys
        assert "done" in expected_env_keys

    def test_run_count_reset_between_hooks(self):
        """Test that run counts are reset between different hooks."""
        executor = HookExecutor()
        component = MockComponent({"name": "test_comp"})
        components = {"test_comp": component}
        context = {}

        # Execute first hook multiple times
        executor.execute_hook("action_selection", ["test_comp", "test_comp"], components, context)

        # Execute different hook - run count should reset
        executor.execute_hook("learning_update", ["test_comp"], components, context)

        # Check calls
        assert len(component.calls) == 3
        assert component.calls[0] == ("action_selection", 1)
        assert component.calls[1] == ("action_selection", 2)
        assert component.calls[2] == ("learning_update", 1)  # Reset to 1

    def test_context_preservation_across_components(self):
        """Test that context is preserved and modified across components."""
        executor = HookExecutor()

        class ContextModifyingComponent(RLComponent):
            def __init__(self, config):
                super().__init__(config)
                self.modifier = config["modifier"]

            def action_selection(self, context: Dict[str, Any]) -> None:
                if "values" not in context:
                    context["values"] = []
                context["values"].append(self.modifier)
                context["action"] = self.modifier

        comp1 = ContextModifyingComponent({"modifier": "first"})
        comp2 = ContextModifyingComponent({"modifier": "second"})
        comp3 = ContextModifyingComponent({"modifier": "third"})

        components = {"comp1": comp1, "comp2": comp2, "comp3": comp3}
        context = {}

        executor.execute_hook("action_selection", ["comp1", "comp2", "comp3"], components, context)

        assert context["values"] == ["first", "second", "third"]
        assert context["action"] == "third"  # Last component wins
