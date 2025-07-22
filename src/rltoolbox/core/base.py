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

    Components are the building blocks of RLtoolbox. They are self-contained modules
    that implement a specific piece of functionality, such as an environment, an
    agent, or a logger. Components interact with each other through a shared
    `context` dictionary, which is passed to each hook method.

    To create a new component, you must inherit from this class and implement the
    hook methods that you need.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize component with configuration.

        Args:
            config: Configuration dictionary for this component. This is the
                component's configuration from the main JSON configuration file.
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    # -------------------
    # Training Lifecycle
    # -------------------

    def training_start(self, context: Dict[str, Any]) -> None:
        """
        Called once at the very beginning of the training process.

        This hook is ideal for setting up resources that are used throughout
        training, such as initializing logging services, creating directories,
        or setting up network connections.

        Context Changes:
            - No specific changes are expected, but you can use this hook to
              add any initial state to the context.
        """
        pass

    def training_end(self, context: Dict[str, Any]) -> None:
        """
        Called once at the very end of the training process.

        This hook is ideal for cleaning up resources, such as closing files,
        saving final models, or shutting down services.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    # -----------------
    # Episode Lifecycle
    # -----------------

    def episode_start(self, context: Dict[str, Any]) -> None:
        """
        Called at the beginning of each training episode.

        This hook is useful for any setup that needs to be done on a
        per-episode basis, such as resetting episode-specific metrics.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def episode_reset(self, context: Dict[str, Any]) -> None:
        """
        Called to reset the environment for a new episode.

        This hook is typically implemented by an environment component. It is
        responsible for resetting the environment and providing the initial
        state of the new episode.

        Context Changes:
            - Expected to add `state`: The initial state of the environment.
        """
        pass

    def episode_end(self, context: Dict[str, Any]) -> None:
        """
        Called at the end of each training episode.

        This hook is useful for any cleanup or logging that needs to be done
        on a per-episode basis, such as logging the total episode reward.
        It's also a common place to implement learning updates for algorithms
        that learn at episode boundaries (e.g., policy gradient methods).

        Context Changes:
            - No specific changes are expected.
        """
        pass

    # --------------
    # Step Lifecycle
    # --------------

    def step_start(self, context: Dict[str, Any]) -> None:
        """
        Called at the beginning of each step within an episode.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def observation_received(self, context: Dict[str, Any]) -> None:
        """
        Called after an observation is received from the environment.

        This hook is useful for any preprocessing that needs to be done on the
        observation before it is used for action selection.

        Context Changes:
            - Can be used to modify the `state` in the context.
        """
        pass

    def action_selection(self, context: Dict[str, Any]) -> None:
        """
        Called to select an action to take in the environment.

        This hook is typically implemented by an agent component. It is
        responsible for using the current state to select an action.

        Context Changes:
            - Expected to add `action`: The action to take in the environment.
        """
        pass

    def action_chosen(self, context: Dict[str, Any]) -> None:
        """
        Called after an action has been selected but before it is executed.

        This hook is useful for any logic that needs to be executed after the
        action is known but before the environment state changes.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def environment_step(self, context: Dict[str, Any]) -> None:
        """
        Called to execute a step in the environment.

        This hook is typically implemented by an environment component. It is
        responsible for executing the chosen action and returning the new
        state, reward, and done flag.

        Context Changes:
            - Expected to add `next_state`: The state of the environment after
              the action is executed.
            - Expected to add `reward`: The reward received from the environment.
            - Expected to add `done`: A boolean indicating whether the episode
              is over.
        """
        pass

    def transition_received(self, context: Dict[str, Any]) -> None:
        """
        Called after a transition (state, action, reward, next_state, done)
        has been received from the environment.

        This hook is useful for any logic that needs to be executed after the
        full transition is known, such as storing the transition in a replay
        buffer.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def experience_storage(self, context: Dict[str, Any]) -> None:
        """
        Called to store experience, typically in a replay buffer.

        This hook is useful for implementing experience replay buffers or other
        memory mechanisms. Components can also implement learning updates here
        if they need to learn immediately after storing experience.

        Context Changes:
            - No specific changes are expected.
        """
        pass


    def step_end(self, context: Dict[str, Any]) -> None:
        """
        Called at the end of each step within an episode.

        This hook can be used to implement learning updates that need to happen
        on a custom schedule (e.g., every N steps, when certain conditions are met).

        Context Changes:
            - No specific changes are expected.
        """
        pass

    # -----------------
    # Evaluation Hooks
    # -----------------

    def evaluation_start(self, context: Dict[str, Any]) -> None:
        """
        Called at the start of the evaluation phase.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def evaluation_episode_start(self, context: Dict[str, Any]) -> None:
        """
        Called at the start of each evaluation episode.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def evaluation_episode_end(self, context: Dict[str, Any]) -> None:
        """
        Called at the end of each evaluation episode.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def evaluation_end(self, context: Dict[str, Any]) -> None:
        """
        Called at the end of the evaluation phase.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    # --------------------------
    # Monitoring and Logging Hooks
    # --------------------------

    def metrics_computation(self, context: Dict[str, Any]) -> None:
        """
        Called to compute and update metrics.

        This hook is useful for calculating and logging any metrics that are
        not directly related to the training process, such as performance
        metrics or system resource usage.

        Context Changes:
            - Can be used to add any computed metrics to the `metrics`
              dictionary in the context.
        """
        pass

    def checkpoint_save(self, context: Dict[str, Any]) -> None:
        """
        Called to save a checkpoint of the training state.

        This hook is useful for implementing custom checkpointing logic.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def logging_update(self, context: Dict[str, Any]) -> None:
        """
        Called to perform logging.

        This hook is typically implemented by a logger component. It is
        responsible for logging the current state of the training process to
        the console, a file, or a logging service.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def visualization_update(self, context: Dict[str, Any]) -> None:
        """
        Called to update visualizations.

        This hook is useful for creating real-time visualizations of the
        training process, such as plotting the episode rewards.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    # ----------------------------------
    # Batch/Epoch Hooks (for offline RL)
    # ----------------------------------

    def epoch_start(self, context: Dict[str, Any]) -> None:
        """
        Called at the start of each epoch in batch-based training.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def epoch_end(self, context: Dict[str, Any]) -> None:
        """
        Called at the end of each epoch in batch-based training.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def batch_start(self, context: Dict[str, Any]) -> None:
        """
        Called at the start of each batch in batch-based training.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    def batch_end(self, context: Dict[str, Any]) -> None:
        """
        Called at the end of each batch in batch-based training.

        Context Changes:
            - No specific changes are expected.
        """
        pass

    # -------------------
    # Component Utilities
    # -------------------

    def validate_config(self) -> bool:
        """
        Validate the component's configuration.

        This method is called after the component is initialized. It should
        return `True` if the configuration is valid, and `False` otherwise.

        Returns:
            True if the configuration is valid, False otherwise.
        """
        return True

    def get_state(self) -> Dict[str, Any]:
        """
        Get the component's state for checkpointing.

        This method is called when a checkpoint is saved. It should return a
        dictionary containing the component's state.

        Returns:
            A dictionary containing the component's state.
        """
        return {}

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Set the component's state from a checkpoint.

        This method is called when a checkpoint is loaded. It should restore
        the component's state from the given dictionary.

        Args:
            state: A dictionary containing the component's state.
        """
        pass
