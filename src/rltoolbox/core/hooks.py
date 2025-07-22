"""
Hook execution system with validation.

Manages execution of hooks and validates expected context modifications.
"""

import warnings
from typing import Dict, Any, List, Set, Optional
from .base import RLComponent


class HookExecutor:
    """
    Executes hooks and validates expected context modifications.

    Tracks which components have run how many times and validates
    that required context keys are added by the appropriate hooks.
    """

    # Define expected context modifications for each hook
    HOOK_EXPECTATIONS = {
        "episode_reset": ["state"],
        "action_selection": ["action"],
        "environment_step": ["next_state", "reward", "done"],
        "observation_received": [],
        "transition_received": [],
        "experience_storage": [],
        "metrics_computation": [],
        "checkpoint_save": [],
        "logging_update": [],
        "visualization_update": [],
        "training_start": [],
        "training_end": [],
        "episode_start": [],
        "episode_end": [],
        "step_start": [],
        "step_end": [],
        "action_chosen": [],
        "evaluation_start": [],
        "evaluation_episode_start": [],
        "evaluation_episode_end": [],
        "evaluation_end": [],
        "epoch_start": [],
        "epoch_end": [],
        "batch_start": [],
        "batch_end": [],
    }

    def __init__(self):
        self.validation_enabled = True
        self._hook_run_counts: Dict[str, Dict[str, int]] = {}

    def set_validation_enabled(self, enabled: bool) -> None:
        """Enable or disable context validation."""
        self.validation_enabled = enabled

    def execute_hook(
        self,
        hook_name: str,
        component_names: List[str],
        components: Dict[str, RLComponent],
        context: Dict[str, Any]
    ) -> None:
        """
        Execute a hook with the specified components.

        Args:
            hook_name: Name of the hook to execute
            component_names: List of component names to execute in order
            components: Dictionary mapping component names to instances
            context: Context dictionary to pass to components
        """
        if hook_name not in self.HOOK_EXPECTATIONS:
            warnings.warn(f"Unknown hook: {hook_name}")

        # Initialize run counts for this hook cycle
        if hook_name not in self._hook_run_counts:
            self._hook_run_counts[hook_name] = {}

        # Track context keys before execution
        context_before = set(context.keys()) if self.validation_enabled else set()

        # Execute each component
        for component_name in component_names:
            if component_name not in components:
                raise ValueError(f"Component '{component_name}' not found")

            component = components[component_name]

            # Update run count for this component
            self._hook_run_counts[hook_name][component_name] = (
                self._hook_run_counts[hook_name].get(component_name, 0) + 1
            )

            # Add run count to context
            context["_run_count"] = self._hook_run_counts[hook_name][component_name]
            context["_run_counts"] = self._hook_run_counts[hook_name].copy()

            # Execute the hook method
            hook_method = getattr(component, hook_name, None)
            if hook_method is None:
                warnings.warn(f"Component '{component_name}' does not implement hook '{hook_name}'")
                continue

            try:
                hook_method(context)
            except Exception as e:
                raise RuntimeError(
                    f"Error executing {hook_name} on component {component_name}: {e}"
                ) from e

        # Validate expected context modifications
        if self.validation_enabled:
            self._validate_context_modifications(hook_name, context_before, context)

        # Reset run counts after hook cycle (for next hook)
        if hook_name in self._hook_run_counts:
            self._hook_run_counts[hook_name].clear()

    def _validate_context_modifications(
        self,
        hook_name: str,
        context_before: Set[str],
        context_after: Dict[str, Any]
    ) -> None:
        """
        Validate that expected context modifications were made.

        Args:
            hook_name: Name of the hook that was executed
            context_before: Set of context keys before execution
            context_after: Context dictionary after execution
        """
        expected_keys = self.HOOK_EXPECTATIONS.get(hook_name, [])
        context_after_keys = set(context_after.keys())
        added_keys = context_after_keys - context_before

        # Check if all expected keys were added
        missing_keys = []
        for expected_key in expected_keys:
            if expected_key not in added_keys and expected_key not in context_before:
                missing_keys.append(expected_key)

        if missing_keys:
            warnings.warn(
                f"Hook '{hook_name}' did not add expected context keys: {missing_keys}. "
                f"Added keys: {list(added_keys)}"
            )

        # Check for None values in expected keys
        none_keys = []
        for expected_key in expected_keys:
            if expected_key in context_after and context_after[expected_key] is None:
                none_keys.append(expected_key)

        if none_keys:
            warnings.warn(
                f"Hook '{hook_name}' added expected keys but with None values: {none_keys}"
            )

    def get_hook_names(self) -> List[str]:
        """Get list of all known hook names."""
        return list(self.HOOK_EXPECTATIONS.keys())

    def add_custom_hook(self, hook_name: str, expected_keys: Optional[List[str]] = None) -> None:
        """
        Add a custom hook with expected context modifications.

        Args:
            hook_name: Name of the custom hook
            expected_keys: List of context keys this hook should add
        """
        if expected_keys is None:
            expected_keys = []

        self.HOOK_EXPECTATIONS[hook_name] = expected_keys

    def validate_hook_configuration(self, hooks_config: Dict[str, List[str]]) -> None:
        """
        Validate that all hooks in configuration are known.

        Args:
            hooks_config: Dictionary mapping hook names to component lists
        """
        unknown_hooks = []
        for hook_name in hooks_config.keys():
            if hook_name not in self.HOOK_EXPECTATIONS:
                unknown_hooks.append(hook_name)

        if unknown_hooks:
            warnings.warn(f"Unknown hooks in configuration: {unknown_hooks}")
