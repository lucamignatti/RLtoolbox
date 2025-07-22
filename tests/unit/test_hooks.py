import pytest
import warnings
from unittest.mock import MagicMock

from rltoolbox.core.hooks import HookExecutor
from rltoolbox.core.base import RLComponent


class MockComponent(RLComponent):
    def __init__(self, config):
        super().__init__(config)
        self.call_log = []
        self.update_count = 0

    def training_start(self, context):
        self.call_log.append("training_start")
        context["training_started"] = True

    def episode_reset(self, context):
        self.call_log.append("episode_reset")
        context["state"] = [0, 0, 0, 0]

    def action_selection(self, context):
        self.call_log.append("action_selection")
        context["action"] = 1

    def environment_step(self, context):
        self.call_log.append("environment_step")
        context["next_state"] = [1, 1, 1, 1]
        context["reward"] = 1.0
        context["done"] = False

    def transition_received(self, context):
        self.call_log.append("transition_received")
        self.update_count += 1

    def episode_end(self, context):
        self.call_log.append("episode_end")

    def training_end(self, context):
        self.call_log.append("training_end")

    def step_end(self, context):
        self.call_log.append("step_end")


class NoOpComponent(RLComponent):
    """Component that implements no hooks (all no-ops)."""
    pass


class ComponentWithoutTrainingStart(RLComponent):
    """Component that doesn't implement training_start hook."""
    def episode_reset(self, context):
        context["state"] = [0, 0, 0, 0]


def test_hook_execution():
    """Test basic hook execution."""
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    executor.execute_hook("training_start", ["mock"], components, context)
    assert component.call_log == ["training_start"]
    assert context["training_started"] is True


def test_hook_validation():
    """Test hook validation with expected context changes."""
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    executor.execute_hook("episode_reset", ["mock"], components, context)
    assert context["state"] == [0, 0, 0, 0]
    assert "episode_reset" in component.call_log


def test_hook_validation_warning():
    """Test that hooks produce warnings when expected context keys are missing."""
    executor = HookExecutor()
    component = NoOpComponent({})  # Uses no-op methods
    components = {"mock": component}
    context = {}

    # The base RLComponent's action_selection is a no-op.
    # The HookExecutor should warn that the expected context key 'action' was not added.
    with pytest.warns(UserWarning, match="did not add expected context keys:.*action"):
        executor.execute_hook("action_selection", ["mock"], components, context)


def test_multiple_components_execution():
    """Test executing multiple components in order."""
    executor = HookExecutor()
    component1 = MockComponent({"name": "comp1"})
    component2 = MockComponent({"name": "comp2"})
    components = {"comp1": component1, "comp2": component2}
    context = {}

    executor.execute_hook("training_start", ["comp1", "comp2"], components, context)

    assert component1.call_log == ["training_start"]
    assert component2.call_log == ["training_start"]
    assert context["training_started"] is True


def test_hook_execution_order():
    """Test that hooks are executed in the specified order."""
    executor = HookExecutor()
    execution_order = []

    class OrderedComponent(RLComponent):
        def __init__(self, config):
            super().__init__(config)
            self.order_id = config["order_id"]

        def training_start(self, context):
            execution_order.append(self.order_id)

    comp1 = OrderedComponent({"order_id": 1})
    comp2 = OrderedComponent({"order_id": 2})
    comp3 = OrderedComponent({"order_id": 3})
    components = {"comp1": comp1, "comp2": comp2, "comp3": comp3}
    context = {}

    executor.execute_hook("training_start", ["comp2", "comp1", "comp3"], components, context)
    assert execution_order == [2, 1, 3]


def test_hook_run_counts():
    """Test that run counts are properly tracked."""
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    # Execute same hook multiple times
    executor.execute_hook("training_start", ["mock"], components, context)
    executor.execute_hook("training_start", ["mock"], components, context)
    executor.execute_hook("training_start", ["mock"], components, context)

    assert len(component.call_log) == 3


def test_component_not_found_error():
    """Test error when component is not found."""
    executor = HookExecutor()
    components = {}
    context = {}

    with pytest.raises(ValueError, match="Component 'nonexistent' not found"):
        executor.execute_hook("training_start", ["nonexistent"], components, context)


def test_component_missing_hook_method():
    """Test warning when component doesn't implement hook method."""
    executor = HookExecutor()

    # Create a component that explicitly doesn't have training_start
    class ComponentWithoutHook:
        def __init__(self, config):
            self.config = config
            self.name = config.get("name", "test")

    component = ComponentWithoutHook({})
    components = {"mock": component}
    context = {}

    with pytest.warns(UserWarning, match="does not implement hook"):
        executor.execute_hook("training_start", ["mock"], components, context)


def test_hook_execution_error_handling():
    """Test error handling during hook execution."""
    executor = HookExecutor()

    class FailingComponent(RLComponent):
        def training_start(self, context):
            raise RuntimeError("Component failed!")

    component = FailingComponent({})
    components = {"failing": component}
    context = {}

    with pytest.raises(RuntimeError, match="Error executing training_start on component failing"):
        executor.execute_hook("training_start", ["failing"], components, context)


def test_context_validation_disabled():
    """Test that context validation can be disabled."""
    executor = HookExecutor()
    executor.set_validation_enabled(False)

    component = NoOpComponent({})
    components = {"mock": component}
    context = {}

    # Should not produce warnings when validation is disabled
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        executor.execute_hook("action_selection", ["mock"], components, context)

        # Should only warn about missing hook implementation, not missing context keys
        context_warnings = [warning for warning in w if "did not add expected context keys" in str(warning.message)]
        assert len(context_warnings) == 0


def test_unknown_hook_warning():
    """Test that unknown hooks produce warnings."""
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    # Test that unknown hooks (like the removed learning_update) produce warnings
    with pytest.warns(UserWarning, match="Unknown hook: learning_update"):
        executor.execute_hook("learning_update", ["mock"], components, context)


def test_removed_learning_update_hook():
    """Test that learning_update hook is no longer in the expected hooks."""
    executor = HookExecutor()

    # Verify learning_update is not in expected hooks
    assert "learning_update" not in executor.HOOK_EXPECTATIONS

    # Verify other hooks are still present
    assert "experience_storage" in executor.HOOK_EXPECTATIONS
    assert "step_end" in executor.HOOK_EXPECTATIONS
    assert "episode_end" in executor.HOOK_EXPECTATIONS


def test_all_expected_hooks_exist():
    """Test that all expected hooks are properly defined."""
    executor = HookExecutor()
    expected_hooks = [
        "episode_reset", "action_selection", "environment_step",
        "observation_received", "transition_received", "experience_storage",
        "metrics_computation", "checkpoint_save", "logging_update",
        "visualization_update", "training_start", "training_end",
        "episode_start", "episode_end", "step_start", "step_end",
        "action_chosen", "evaluation_start", "evaluation_episode_start",
        "evaluation_episode_end", "evaluation_end", "epoch_start",
        "epoch_end", "batch_start", "batch_end"
    ]

    for hook in expected_hooks:
        assert hook in executor.HOOK_EXPECTATIONS, f"Hook {hook} missing from HOOK_EXPECTATIONS"


def test_custom_hook_addition():
    """Test adding custom hooks."""
    executor = HookExecutor()

    # Add a custom hook
    executor.add_custom_hook("custom_hook", ["custom_key"])

    assert "custom_hook" in executor.HOOK_EXPECTATIONS
    assert executor.HOOK_EXPECTATIONS["custom_hook"] == ["custom_key"]


def test_hook_configuration_validation():
    """Test validation of hook configuration."""
    executor = HookExecutor()

    # Valid configuration
    valid_config = {
        "training_start": ["comp1"],
        "episode_reset": ["comp2", "comp3"]
    }

    # Should not raise any errors
    executor.validate_hook_configuration(valid_config)

    # Invalid configuration with unknown hook
    invalid_config = {
        "training_start": ["comp1"],
        "unknown_hook": ["comp2"]
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        executor.validate_hook_configuration(invalid_config)

        # Should warn about unknown hook
        assert len(w) >= 1
        assert any("Unknown hooks in configuration" in str(warning.message) for warning in w)


def test_get_hook_names():
    """Test getting list of all hook names."""
    executor = HookExecutor()
    hook_names = executor.get_hook_names()

    assert isinstance(hook_names, list)
    assert "training_start" in hook_names
    assert "episode_reset" in hook_names
    assert "action_selection" in hook_names
    assert "learning_update" not in hook_names  # Should be removed
