# RLtoolbox

RLtoolbox is a lightweight, modular, and extensible toolkit for reinforcement learning research. It provides a minimal core framework and a flexible hook-based system for building and running RL experiments.

## Philosophy

RLtoolbox is designed to be a toolkit, not a framework. It provides the basic building blocks for running RL experiments, but it doesn't force you to use any specific components or algorithms. You can use any environment, model, or RL algorithm you want. Its hook-based architecture allows for complete customization and reproducibility of training loops through JSON configuration files.

## Installation

To install the core RLtoolbox, run:

```bash
pip install .
```

To install the example wrappers (which include common environments, agents, and algorithms), run:

```bash
pip install -e tools/rltoolbox_wrappers
```

## Usage

The main entry point for RLtoolbox is the `rltoolbox` command-line interface (CLI). The CLI has the following commands:

### `rltoolbox-train`

Trains a model based on a specified configuration file.

```bash
python -m rltoolbox.cli train <path_to_config.json> [OPTIONS]
```

**Arguments:**
*   `<path_to_config.json>`: Path to your JSON configuration file.

**Options:**
*   `--evaluate <num_episodes>`: Number of evaluation episodes to run after training.
*   `--checkpoint <path_to_save_checkpoint>`: Path to save the final training checkpoint.
*   `--load-checkpoint <path_to_load_checkpoint>`: Path to load a checkpoint from before starting training.

**Example:**
```bash
python -m rltoolbox.cli train config.ppo.json --evaluate 10 --checkpoint my_ppo_run.json
```

### `rltoolbox-evaluate`

Evaluates a trained model from a checkpoint.

```bash
python -m rltoolbox.cli evaluate <path_to_config.json> <path_to_checkpoint.json> [OPTIONS]
```

**Arguments:**
*   `<path_to_config.json>`: Path to the original JSON configuration file used for training.
*   `<path_to_checkpoint.json>`: Path to the checkpoint file of the trained model.

**Options:**
*   `--num-episodes <num>`: Number of episodes to evaluate (default: 10).

**Example:**
```bash
python -m rltoolbox.cli evaluate config.ppo.json my_ppo_run.json --num-episodes 20
```

### `rltoolbox-list`

Lists available components in the registry. (Note: The registry needs to be populated by loading a config file first for this command to show custom components).

```bash
python -m rltoolbox.cli list
```

**Example:**
```bash
python -m rltoolbox.cli list
```

### `rltoolbox-new`

Creates a new component package template. This scaffolds a new Python package with the basic structure for RLtoolbox components.

```bash
python -m rltoolbox.cli new <package_name>
```

**Arguments:**
*   `<package_name>`: The name of the new component package to create.

**Example:**
```bash
python -m rltoolbox.cli new MyCustomComponents
```
This will create a directory `MyCustomComponents` with a basic `setup.py`, `README.md`, and a `src/MyCustomComponents/mycomponent.py` template.

## Configuration File Structure

RLtoolbox experiments are defined using JSON configuration files. These files specify the components to be used, how they interact via hooks, and training parameters.

A typical configuration file includes the following top-level keys:

*   `packages`: (Optional) Defines external Python packages containing custom `RLComponent` implementations.
    *   Each entry specifies a package name and either a `path` (for local development packages) or a `version` (for installed packages).
    *   Example:
        ```json
        "packages": {
            "my_rl_components": {
                "path": "/path/to/my/rl/components"
            },
            "stable_baselines3_components": {
                "version": "1.0.0"
            }
        }
        ```
*   `components`: Defines all `RLComponent` instances used in the experiment.
    *   Each entry is a key-value pair where the key is the unique name of the component instance (e.g., `"env"`, `"agent"`) and the value is a dictionary defining the component.
    *   Component definition includes:
        *   `package`: The name of the package where the component's class is located (e.g., `"rltoolbox"`, `"my_rl_components"`). Defaults to `"rltoolbox"` if not specified.
        *   `type`: The class name of the `RLComponent` to instantiate (e.g., `"GymnasiumEnvWrapper"`, `"MLPAgent"`).
        *   Additional key-value pairs are passed as `config` to the component's `__init__` method.
*   `hooks`: Defines the execution order of components for various events in the training loop.
    *   Each entry is a key-value pair where the key is a hook name (e.g., `"episode_reset"`, `"action_selection"`) and the value is a list of component names (from the `components` section) to execute for that hook, in order.
*   `training`: Defines global training parameters.
    *   `max_episodes`: Maximum number of episodes to train for.
    *   `max_steps_per_episode`: Maximum steps per episode.
    *   `max_total_steps`: Maximum total steps across all episodes.
*   `evaluation`: (Optional) Defines parameters for automatic evaluation after training.
    *   `enabled`: Boolean to enable/disable automatic evaluation.
    *   `num_episodes`: Number of episodes for evaluation.
    *   `hooks`: (Optional) Override specific hooks for evaluation (e.g., to use a greedy policy instead of an exploratory one).
*   `seed`: (Optional) Integer seed for reproducibility.
*   `development_mode`: (Optional) Boolean. If `true`, skips package version validation.

### Example Configuration (`config.ppo.json`)

This example demonstrates a PPO agent training on CartPole-v1 using components from a hypothetical `my_rl_components` package.

```json
{
    "packages": {
        "my_rl_components": {
            "path": "/path/to/my/rl/components"
        }
    },
    "components": {
        "env": {
            "package": "my_rl_components",
            "type": "GymnasiumEnvWrapper",
            "env_name": "CartPole-v1"
        },
        "actor_critic": {
            "package": "my_rl_components",
            "type": "MLPActorCritic",
            "observation_space": 4,
            "action_space": 2,
            "hidden_dims": [64, 64]
        },
        "ppo_algorithm": {
            "package": "my_rl_components",
            "type": "PPOAlgorithm",
            "actor_critic_component": "actor_critic",
            "clip_epsilon": 0.2,
            "ppo_epochs": 10,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5
        },
        "experience_buffer": {
            "package": "my_rl_components",
            "type": "ExperienceBuffer",
            "buffer_size": 2048
        },
        "console_logger": {
            "package": "my_rl_components",
            "type": "ConsoleLogger"
        }
    },
    "hooks": {
        "training_start": ["console_logger"],
        "episode_reset": ["env"],
        "environment_step": ["env"],
        "action_selection": ["actor_critic"],
        "transition_received": ["experience_buffer"],
        "learning_update": ["ppo_algorithm"],
        "episode_end": ["console_logger"]
    },
    "training": {
        "max_episodes": 1000,
        "max_steps_per_episode": 500
    }
}
```

**Explanation of the Example:**
*   **`packages`**: Specifies that `my_rl_components` is a local package located at `/path/to/my/rl/components`.
*   **`components`**:
    *   `env`: An instance of `GymnasiumEnvWrapper` for the `CartPole-v1` environment.
    *   `actor_critic`: An `MLPActorCritic` neural network for policy and value estimation.
    *   `ppo_algorithm`: The PPO algorithm, configured to use the `actor_critic` component.
    *   `experience_buffer`: A buffer to store transitions for learning.
    *   `console_logger`: A simple logger to print training progress.
*   **`hooks`**: Defines the flow of execution:
    *   `training_start`: `console_logger` is called at the beginning of training.
    *   `episode_reset`: `env` resets the environment.
    *   `environment_step`: `env` takes a step in the environment.
    *   `action_selection`: `actor_critic` selects an action.
    *   `transition_received`: `experience_buffer` stores the collected transition.
    *   `learning_update`: `ppo_algorithm` performs a learning update.
    *   `episode_end`: `console_logger` logs episode summary.
*   **`training`**: Sets the maximum number of episodes and steps per episode.

## The Hook System

The core of RLtoolbox's flexibility lies in its hook system. Components communicate and interact by implementing specific hook methods, which are called at predefined points in the training or evaluation loop. All communication between components happens via a shared `context` dictionary, which is passed to each hook method. Components modify this `context` to pass information to subsequent components.

### `RLComponent` Base Class

All components must inherit from `rltoolbox.core.base.RLComponent`. This base class defines a comprehensive set of hook methods that can be overridden by custom components. If a component does not implement a specific hook, that hook call is simply ignored for that component.

### Available Hooks and Context Modifications

Below is a detailed list of the hooks defined in `RLComponent`, their purpose, and the expected context modifications. The `context` dictionary is a mutable object, and components are expected to add or modify keys within it.

**Global Context Keys (Commonly Used):**
*   `config`: The full configuration dictionary loaded by the trainer.
*   `training`: A dictionary containing training state (e.g., `step`, `episode`, `total_steps`, `episode_reward`, `episode_length`).
*   `metrics`: A dictionary to accumulate various metrics (e.g., `episode_rewards`, `episode_lengths`, `losses`).
*   `components`: A dictionary mapping component names to their instances (allows components to access other components).
*   `_run_count`: (Internal) The number of times a specific component has been called within the current hook execution cycle.
*   `_run_counts`: (Internal) A dictionary of run counts for all components within the current hook execution cycle.

---

#### Training Lifecycle Hooks

*   **`training_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called once at the very beginning of the training process. Ideal for initializing logging services, creating directories, or setting up global resources.
    *   **Context Changes:** No specific changes are expected, but can be used to add initial state.

*   **`training_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called once at the very end of the training process (or upon interruption). Ideal for cleaning up resources, saving final models, or summarizing results.
    *   **Context Changes:** No specific changes are expected.

---

#### Episode Lifecycle Hooks

*   **`episode_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the beginning of each training episode. Useful for per-episode setup, like resetting episode-specific metrics.
    *   **Context Changes:** No specific changes are expected.

*   **`episode_reset(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to reset the environment for a new episode. Typically implemented by an environment component.
    *   **Context Changes (Expected):**
        *   `state`: The initial observation/state of the environment.

*   **`episode_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the end of each training episode. Useful for cleanup or logging per-episode results (e.g., total episode reward).
    *   **Context Changes:** No specific changes are expected.

---

#### Step Lifecycle Hooks

*   **`step_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the beginning of each step within an episode.
    *   **Context Changes:** No specific changes are expected.

*   **`observation_received(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called after an observation is received from the environment. Useful for preprocessing the observation before action selection.
    *   **Context Changes:** Can modify the `state` in the context.

*   **`action_selection(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to select an action to take in the environment. Typically implemented by an agent component.
    *   **Context Changes (Expected):**
        *   `action`: The action to take in the environment.
        *   (Optional, for policy gradient methods) `log_prob`: Log probability of the selected action.
        *   (Optional, for actor-critic methods) `value`: The estimated value of the current state.

*   **`action_chosen(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called after an action has been selected but before it is executed in the environment. Useful for pre-execution logic.
    *   **Context Changes:** No specific changes are expected.

*   **`environment_step(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to execute a step in the environment. Typically implemented by an environment component.
    *   **Context Changes (Expected):**
        *   `next_state`: The state of the environment after the action is executed.
        *   `reward`: The reward received from the environment.
        *   `done`: A boolean indicating whether the episode is over (terminated or truncated).
        *   `info`: (Optional) Dictionary with additional step information from the environment.

*   **`transition_received(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called after a full transition (`state`, `action`, `reward`, `next_state`, `done`) has been received. Useful for storing the transition in a replay buffer.
    *   **Context Changes:** No specific changes are expected.

*   **`experience_storage(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to store experience, typically in a replay buffer.
    *   **Context Changes:** No specific changes are expected.

*   **`learning_update(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to perform a learning update. Typically implemented by an algorithm component.
    *   **Context Changes:** Can add learning-related metrics (e.g., `loss`, `policy_loss`, `value_loss`, `entropy_loss`).

*   **`step_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the end of each step within an episode.
    *   **Context Changes:** No specific changes are expected.

---

#### Evaluation Hooks

*   **`evaluation_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the start of the evaluation phase.
    *   **Context Changes:** No specific changes are expected.

*   **`evaluation_episode_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the start of each evaluation episode.
    *   **Context Changes:** No specific changes are expected.

*   **`evaluation_episode_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the end of each evaluation episode.
    *   **Context Changes:** No specific changes are expected.

*   **`evaluation_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the end of the evaluation phase.
    *   **Context Changes:** No specific changes are expected.
        *   (Trainer adds) `final_evaluation`: Dictionary containing evaluation metrics (e.g., `mean_reward`, `std_reward`, `mean_length`, `std_length`).

---

#### Monitoring and Logging Hooks

*   **`metrics_computation(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to compute and update metrics. Useful for calculating performance metrics or system resource usage.
    *   **Context Changes:** Can add computed metrics to the `metrics` dictionary in the context.

*   **`checkpoint_save(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to save a checkpoint of the training state. Useful for implementing custom checkpointing logic.
    *   **Context Changes:** No specific changes are expected.

*   **`logging_update(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to perform logging. Typically implemented by a logger component.
    *   **Context Changes:** No specific changes are expected.

*   **`visualization_update(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called to update visualizations. Useful for real-time visualizations of the training process.
    *   **Context Changes:** No specific changes are expected.

---

#### Batch/Epoch Hooks (for offline RL)

*   **`epoch_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the start of each epoch in batch-based training.
    *   **Context Changes:** No specific changes are expected.

*   **`epoch_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the end of each epoch in batch-based training.
    *   **Context Changes:** No specific changes are expected.

*   **`batch_start(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the start of each batch in batch-based training.
    *   **Context Changes:** No specific changes are expected.

*   **`batch_end(self, context: Dict[str, Any]) -> None`**
    *   **Purpose:** Called at the end of each batch in batch-based training.
    *   **Context Changes:** No specific changes are expected.

---

### Component Utilities

*   **`validate_config(self) -> bool`**
    *   **Purpose:** Called after the component is initialized to validate its configuration. Should return `True` if valid, `False` otherwise.
*   **`get_state(self) -> Dict[str, Any]`**
    *   **Purpose:** Called when a checkpoint is saved. Should return a dictionary containing the component's internal state for persistence.
*   **`set_state(self, state: Dict[str, Any]) -> None`**
    *   **Purpose:** Called when a checkpoint is loaded. Should restore the component's internal state from the given dictionary.

## Components Overview

RLtoolbox components are Python classes that inherit from `rltoolbox.core.base.RLComponent`. They encapsulate specific functionalities (e.g., environments, agents, algorithms, loggers) and interact with the training loop via the hook system.

The `rltoolbox_wrappers` package provides a set of example components that you can use as a starting point or directly in your configurations:

*   **Environments:**
    *   `GymnasiumEnvWrapper`: Wraps standard Gymnasium environments.
    *   `SimpleEnvironment`: A simplified Gymnasium wrapper for testing.
    *   `RandomEnvironment`: A dummy environment for testing purposes.
*   **Agents:**
    *   `RandomAgent`: Selects actions randomly.
    *   `MLPAgent`: A simple Multi-Layer Perceptron agent for discrete action spaces.
    *   `EpsilonGreedyAgent`: Implements an epsilon-greedy exploration strategy, wrapping another agent.
    *   `MLPActorCritic`: Provides an actor (policy) and critic (value) network, typically used by algorithm components.
*   **Algorithms:**
    *   `PPOAlgorithm`: Implements the Proximal Policy Optimization (PPO) algorithm.
*   **Buffers:**
    *   `ExperienceBuffer`: A simple replay buffer for storing and sampling experiences.
*   **Loggers:**
    *   `ConsoleLogger`: Prints training progress to the console.