# RLtoolbox

RLtoolbox is a lightweight, modular, and extensible toolkit for reinforcement learning research. It provides a minimal core framework and a flexible hook-based system for building and running RL experiments.

## Philosophy

RLtoolbox is designed to be a toolkit, not a framework. It provides the basic building blocks for running RL experiments, but it doesn't force you to use any specific components or algorithms. You can use any environment, model, or RL algorithm you want.

## Installation

To install the core RLtoolbox, run:

```
pip install .
```

To install the example wrappers, run:

```
pip install -e tools/rltoolbox_wrappers
```

## Usage

The main entry point for RLtoolbox is the `rltoolbox` command-line interface (CLI). The CLI has the following commands:

*   `train`: Train a model
*   `evaluate`: Evaluate a trained model
*   `list`: List available components
*   `new`: Create a new component template

To train a model, you need to create a JSON configuration file that specifies the components and hooks to use. Here's an example configuration that uses a hypothetical `my_rl_components` package to train a PPO agent on CartPole:

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

To run the training, use the `train` command:

```
python -m rltoolbox.cli train path/to/config.json
```

## Creating Components

To create a new component, use the `new` command:

```
python -m rltoolbox.cli new MyComponent
```

This will create a new file called `mycomponent.py` with a template for a new component. You can then implement the required hooks in this file.
