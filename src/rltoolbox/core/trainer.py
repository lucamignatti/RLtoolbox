"""
Main trainer class that orchestrates the RL training loop.

The trainer loads configuration, manages components, and executes the training loop
using the hook system.
"""

import json
import random
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from .base import RLComponent
from .registry import get_registry
from .hooks import HookExecutor


class RLTrainer:
    """
    Main trainer class that orchestrates RL training.

    Loads configuration, manages components, and executes the training loop
    using the hook-based system.
    """

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None, skip_automatic_evaluation: bool = False):
        """
        Initialize trainer with configuration.

        Args:
            config_path: Path to JSON configuration file
            config_dict: Configuration dictionary (alternative to config_path)
            skip_automatic_evaluation: If True, skip automatic evaluation even if configured
        """
        if config_path is not None and config_dict is not None:
            raise ValueError("Provide either config_path or config_dict, not both")

        if config_path is None and config_dict is None:
            raise ValueError("Must provide either config_path or config_dict")

        # Store settings
        self.skip_automatic_evaluation = skip_automatic_evaluation

        # Load configuration
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.config_path = Path(config_path)
        else:
            self.config = config_dict
            self.config_path = None

        # Validate required configuration sections
        self._validate_config()

        # Set up random seed for reproducibility
        if "seed" in self.config:
            self._set_seed(self.config["seed"])

        # Initialize systems
        self.registry = get_registry()
        self.hook_executor = HookExecutor()

        # Check for development mode
        if self.config.get("development_mode", False):
            self.registry.set_development_mode(True)

        # Load packages and components
        self._setup_packages()
        self._setup_components()

        # Initialize context
        self.context = self._initialize_context()

    def _validate_config(self) -> None:
        """Validate the configuration structure."""
        required_sections = ["components", "hooks"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate seed is present
        if "seed" not in self.config:
            warnings.warn("No seed specified in configuration - results may not be reproducible")

        # Validate components section
        if not isinstance(self.config["components"], dict):
            raise ValueError("'components' section must be a dictionary")

        for name, component_config in self.config["components"].items():
            if not isinstance(component_config, dict):
                raise ValueError(f"Component '{name}' configuration must be a dictionary")

            if "type" not in component_config:
                raise ValueError(f"Component '{name}' missing 'type' field")

            if "package" not in component_config:
                # Default to rltoolbox package
                component_config["package"] = "rltoolbox"

        # Validate hooks section
        if not isinstance(self.config["hooks"], dict):
            raise ValueError("'hooks' section must be a dictionary")

        for hook_name, component_list in self.config["hooks"].items():
            if not isinstance(component_list, list):
                raise ValueError(f"Hook '{hook_name}' must specify a list of components")

        # Validate evaluation section if present
        if "evaluation" in self.config:
            evaluation_config = self.config["evaluation"]
            if not isinstance(evaluation_config, dict):
                raise ValueError("'evaluation' section must be a dictionary")

            if "enabled" in evaluation_config and not isinstance(evaluation_config["enabled"], bool):
                raise ValueError("'evaluation.enabled' must be a boolean")

            if "num_episodes" in evaluation_config:
                if not isinstance(evaluation_config["num_episodes"], int) or evaluation_config["num_episodes"] <= 0:
                    raise ValueError("'evaluation.num_episodes' must be a positive integer")

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        # Note: Additional seeds (torch, tensorflow) should be set by components if needed

    def _setup_packages(self) -> None:
        """Set up package registry."""
        # Register built-in components
        self.registry.register_builtin_components()

        # Register packages from configuration
        if "packages" in self.config:
            self.registry.register_packages(self.config["packages"])

    def _setup_components(self) -> None:
        """Create component instances from configuration."""
        self.components: Dict[str, RLComponent] = {}

        for name, component_config in self.config["components"].items():
            package_name = component_config["package"]
            component_type = component_config["type"]

            # Add component name to config
            config_with_name = component_config.copy()
            config_with_name["name"] = name

            # Create component instance
            try:
                component = self.registry.create_component(
                    package_name, component_type, config_with_name
                )
                self.components[name] = component
            except Exception as e:
                raise RuntimeError(f"Failed to create component '{name}': {e}") from e

    def _initialize_context(self) -> Dict[str, Any]:
        """Initialize the context dictionary."""
        context = {
            "config": self.config.copy(),
            "training": {
                "step": 0,
                "episode": 0,
                "total_steps": 0,
                "episode_reward": 0.0,
                "episode_length": 0,
            },
            "metrics": {
                "episode_rewards": [],
                "episode_lengths": [],
                "losses": [],
            }
        }

        # Add any additional context from config
        if "context" in self.config:
            context.update(self.config["context"])

        return context

    def train(self) -> None:
        """Execute the main training loop."""
        try:
            # Training start
            self._execute_hook("training_start")

            # Get training parameters
            training_config = self.config.get("training", {})
            max_episodes = training_config.get("max_episodes", 1000)
            max_steps_per_episode = training_config.get("max_steps_per_episode", None)
            max_total_steps = training_config.get("max_total_steps", None)

            # Main training loop
            for episode in range(max_episodes):
                self.context["training"]["episode"] = episode
                self.context["training"]["episode_reward"] = 0.0
                self.context["training"]["episode_length"] = 0

                # Episode start
                self._execute_hook("episode_start")

                # Reset environment
                self._execute_hook("episode_reset")

                # Episode loop
                step = 0
                while True:
                    # Check termination conditions
                    if max_steps_per_episode is not None and step >= max_steps_per_episode:
                        break

                    if max_total_steps is not None and self.context["training"]["total_steps"] >= max_total_steps:
                        break

                    # Step execution
                    self._execute_step()

                    step += 1
                    self.context["training"]["step"] = step
                    self.context["training"]["total_steps"] += 1
                    self.context["training"]["episode_length"] = step

                    # Check if episode is done
                    if self.context.get("done", False):
                        break

                # Episode end
                self._execute_hook("episode_end")

                # Store episode metrics
                self.context["metrics"]["episode_rewards"].append(
                    self.context["training"]["episode_reward"]
                )
                self.context["metrics"]["episode_lengths"].append(
                    self.context["training"]["episode_length"]
                )

                # Check total steps termination
                if max_total_steps is not None and self.context["training"]["total_steps"] >= max_total_steps:
                    break

            # Training end
            self._execute_hook("training_end")

            # Run automatic evaluation if configured
            self._run_automatic_evaluation()

        except KeyboardInterrupt:
            print("Training interrupted by user")
            self._execute_hook("training_end")
        except Exception as e:
            print(f"Training failed: {e}")
            self._execute_hook("training_end")
            raise

    def _execute_step(self) -> None:
        """Execute a single step of the training loop."""
        # Step start
        self._execute_hook("step_start")

        # Get observation (if needed)
        self._execute_hook("observation_received")

        # Select action
        self._execute_hook("action_selection")

        # Action chosen (post-processing)
        self._execute_hook("action_chosen")

        # Environment step
        self._execute_hook("environment_step")

        # Transition received (post-processing)
        self._execute_hook("transition_received")

        # Update episode reward
        if "reward" in self.context:
            self.context["training"]["episode_reward"] += self.context["reward"]

        # Store experience
        self._execute_hook("experience_storage")

        # Learning update
        self._execute_hook("learning_update")

        # Step end
        self._execute_hook("step_end")

    def _execute_hook(self, hook_name: str) -> None:
        """
        Execute a hook if it's configured.

        Args:
            hook_name: Name of the hook to execute
        """
        if hook_name not in self.config["hooks"]:
            return

        component_names = self.config["hooks"][hook_name]

        self.hook_executor.execute_hook(
            hook_name, component_names, self.components, self.context
        )

    def _run_automatic_evaluation(self) -> None:
        """Run automatic evaluation if configured in JSON."""
        # Skip if explicitly disabled
        if self.skip_automatic_evaluation:
            return

        evaluation_config = self.config.get("evaluation", {})

        if evaluation_config.get("enabled", False):
            num_episodes = evaluation_config.get("num_episodes", 10)
            print(f"\nRunning automatic evaluation for {num_episodes} episodes...")

            # Temporarily override hooks for evaluation if specified
            original_hooks = self.config["hooks"].copy()
            eval_hooks = evaluation_config.get("hooks", {})

            if eval_hooks:
                self.config["hooks"].update(eval_hooks)

            eval_results = self.evaluate(num_episodes=num_episodes)

            # Restore original hooks
            self.config["hooks"] = original_hooks

            print("Automatic Evaluation Results:")
            print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
            print(f"  Mean Length: {eval_results['mean_length']:.2f} ± {eval_results['std_length']:.2f}")

            # Store evaluation results in context for hooks
            self.context["final_evaluation"] = eval_results

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Run evaluation episodes.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary containing evaluation metrics
        """
        # Save current context
        training_context = self.context["training"].copy()

        # Evaluation start
        self._execute_hook("evaluation_start")

        eval_rewards = []
        eval_lengths = []

        for episode in range(num_episodes):
            self.context["training"]["episode"] = episode
            self.context["training"]["episode_reward"] = 0.0
            self.context["training"]["episode_length"] = 0

            # Evaluation episode start
            self._execute_hook("evaluation_episode_start")

            # Reset environment
            self._execute_hook("episode_reset")

            # Episode loop
            step = 0
            while True:
                # Step execution (same as training)
                self._execute_step()

                step += 1
                self.context["training"]["episode_length"] = step

                # Check if episode is done
                if self.context.get("done", False):
                    break

            # Evaluation episode end
            self._execute_hook("evaluation_episode_end")

            # Store metrics
            eval_rewards.append(self.context["training"]["episode_reward"])
            eval_lengths.append(self.context["training"]["episode_length"])

        # Evaluation end
        self._execute_hook("evaluation_end")

        # Restore training context
        self.context["training"] = training_context

        # Return evaluation metrics
        return {
            "mean_reward": np.mean(eval_rewards),
            "std_reward": np.std(eval_rewards),
            "mean_length": np.mean(eval_lengths),
            "std_length": np.std(eval_lengths),
            "episode_rewards": eval_rewards,
            "episode_lengths": eval_lengths,
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save trainer checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "config": self.config,
            "context": self.context,
            "components": {}
        }

        # Save component states
        for name, component in self.components.items():
            try:
                checkpoint["components"][name] = component.get_state()
            except Exception as e:
                warnings.warn(f"Could not save state for component {name}: {e}")

        # Save checkpoint
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def load_checkpoint(self, path: str) -> None:
        """
        Load trainer checkpoint.

        Args:
            path: Path to checkpoint file
        """
        with open(path, 'r') as f:
            checkpoint = json.load(f)

        # Restore context
        self.context.update(checkpoint["context"])

        # Restore component states
        for name, component in self.components.items():
            if name in checkpoint["components"]:
                try:
                    component.set_state(checkpoint["components"][name])
                except Exception as e:
                    warnings.warn(f"Could not restore state for component {name}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return self.context["metrics"].copy()

    def get_context(self) -> Dict[str, Any]:
        """Get current context (read-only copy)."""
        return self.context.copy()
