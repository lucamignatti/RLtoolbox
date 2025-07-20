"""
Base component class for RLtoolbox framework.

All components must inherit from RLComponent and override the hooks they need.
All communication happens via context dict modification - no return values.
"""

from abc import ABC
from typing import Dict, Any


class RLComponent(ABC):
    """
    Base class for all RL components in the framework.

    Components should override the hook methods they need to participate in.
    All hooks receive a context dict that should be modified in-place.
    The run count for this component in the current hook cycle is available
    at context["_run_count"].
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize component with configuration.

        Args:
            config: Configuration dictionary for this component
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    # Training lifecycle hooks
    def training_start(self, context: Dict[str, Any]) -> None:
        """Called once at the beginning of training."""
        pass

    def training_end(self, context: Dict[str, Any]) -> None:
        """Called once at the end of training."""
        pass

    # Episode lifecycle hooks
    def episode_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of each episode."""
        pass

    def episode_reset(self, context: Dict[str, Any]) -> None:
        """
        Called to reset the environment for a new episode.
        Expected to add: context["state"]
        """
        pass

    def episode_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of each episode."""
        pass

    # Step lifecycle hooks
    def step_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of each step."""
        pass

    def observation_received(self, context: Dict[str, Any]) -> None:
        """Called after receiving observation from environment."""
        pass

    def action_selection(self, context: Dict[str, Any]) -> None:
        """
        Called to select an action.
        Expected to add: context["action"]
        """
        pass

    def action_chosen(self, context: Dict[str, Any]) -> None:
        """Called after action is chosen but before environment step."""
        pass

    def environment_step(self, context: Dict[str, Any]) -> None:
        """
        Called to step the environment.
        Expected to add: context["next_state"], context["reward"], context["done"]
        """
        pass

    def transition_received(self, context: Dict[str, Any]) -> None:
        """Called after receiving transition from environment."""
        pass

    def experience_storage(self, context: Dict[str, Any]) -> None:
        """Called to store experience (e.g., in replay buffer)."""
        pass

    def learning_update(self, context: Dict[str, Any]) -> None:
        """Called to perform learning updates."""
        pass

    def step_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of each step."""
        pass

    # Evaluation hooks
    def evaluation_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of evaluation."""
        pass

    def evaluation_episode_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of each evaluation episode."""
        pass

    def evaluation_episode_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of each evaluation episode."""
        pass

    def evaluation_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of evaluation."""
        pass

    # Monitoring and logging hooks
    def metrics_computation(self, context: Dict[str, Any]) -> None:
        """Called to compute metrics."""
        pass

    def checkpoint_save(self, context: Dict[str, Any]) -> None:
        """Called to save checkpoints."""
        pass

    def logging_update(self, context: Dict[str, Any]) -> None:
        """Called to update logs."""
        pass

    def visualization_update(self, context: Dict[str, Any]) -> None:
        """Called to update visualizations."""
        pass

    # Batch/epoch hooks (for offline RL)
    def epoch_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of each epoch (for batch methods)."""
        pass

    def epoch_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of each epoch (for batch methods)."""
        pass

    def batch_start(self, context: Dict[str, Any]) -> None:
        """Called at the start of each batch."""
        pass

    def batch_end(self, context: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass

    def validate_config(self) -> bool:
        """
        Validate component configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get component state for checkpointing.

        Returns:
            Dictionary containing component state
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set component state from checkpoint.

        Args:
            state: Dictionary containing component state
        """
        pass
