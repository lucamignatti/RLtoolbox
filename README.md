# RLtoolbox

A highly configurable reinforcement learning framework that enables reproducible and flexible RL research through JSON configuration files.

## Overview

RLtoolbox is a hook-based RL framework that breaks down the training loop into discrete, configurable steps. Each step can be customized by specifying which components should run and in what order, all through JSON configuration files. This approach enables:

- **Complete Reproducibility**: Every aspect of training is specified in configuration
- **Maximum Flexibility**: Support for any RL paradigm (online/offline/batch/streaming)
- **Easy Experimentation**: Test different algorithms and configurations without code changes
- **Research-Friendly**: Custom components can be added without modifying the framework

## Key Features

### Standalone Training Command
Train models directly from JSON configuration with zero wrapping code:
```bash
rltoolbox-train config.json --verbose
```
- Automatic evaluation at the end if configured
- Complete isolation from wrapping code influence
- Built-in checkpointing and logging
- See `TRAINING_GUIDE.md` for full documentation

### Hook-Based Architecture
The training loop is broken into 25+ discrete hook points where components can inject behavior, giving you complete control over every aspect of the training process:

**Core Training Hooks:**
- `training_start`, `training_end` - Training lifecycle
- `episode_start`, `episode_reset`, `episode_end` - Episode lifecycle  
- `step_start`, `step_end` - Individual step lifecycle
- `action_selection`, `action_chosen` - Action selection process
- `environment_step` - Environment interaction
- `learning_update` - Model updates

**Data Flow Hooks:**
- `observation_received`, `transition_received` - Data reception
- `experience_storage` - Experience replay management
- `metrics_computation`, `logging_update` - Metrics and logging
- `checkpoint_save`, `visualization_update` - Persistence and visualization

**Evaluation Hooks:**
- `evaluation_start`, `evaluation_end` - Evaluation lifecycle
- `evaluation_episode_start`, `evaluation_episode_end` - Evaluation episodes

**Batch Processing Hooks:**
- `epoch_start`, `epoch_end` - Training epochs
- `batch_start`, `batch_end` - Batch processing

### JSON Configuration
Complete experiments are defined in JSON files:
```json
{
  "seed": 42,
  "packages": {
    "my_research": {"version": "1.0.0", "components": ["CustomDQN"]}
  },
  "components": {
    "model": {"package": "my_research", "type": "CustomDQN", "lr": 0.001},
    "env": {"package": "rltoolbox", "type": "SimpleEnvironment", "env_name": "CartPole-v1"}
  },
  "hooks": {
    "action_selection": ["model"],
    "environment_step": ["env"],
    "learning_update": ["model", "replay_buffer", "model"]
  }
}
```

### Component System
- **Flexible Components**: Any component can participate in any hook
- **Multiple Execution**: Components can run multiple times per hook (e.g., `["model", "algo", "model"]`)
- **Package Management**: Support for custom packages with version control
- **Development Mode**: Skip version validation during research iteration

## Quick Start

### 1. Install RLtoolbox
```bash
pip install -e .
```

### 2. Run a Training Example
```bash
rltoolbox-train configs/cartpole_epsilon_greedy.json --verbose
```

This will:
- Train an epsilon-greedy agent on CartPole
- Automatically run evaluation at the end
- Save comprehensive logs and metrics
- Display real-time training progress

### 3. Explore Configurations
Check out the example configurations in `configs/`:
- `cartpole_random.json` - Basic random agent
- `cartpole_epsilon_greedy.json` - RL agent with replay buffer
- `cartpole_comprehensive_eval.json` - Full evaluation setup

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/RLtoolbox.git
cd RLtoolbox

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install rltoolbox
```

## Quick Start

### 1. Basic Training

Train a random agent on CartPole:

```python
from rltoolbox import RLTrainer

# Load configuration and train
trainer = RLTrainer(config_path="configs/cartpole_random.json")
trainer.train()

# Evaluate
results = trainer.evaluate(num_episodes=10)
print(f"Mean reward: {results['mean_reward']:.2f}")
```

### 2. Command Line Interface

```bash
# Train using a configuration file
python examples/train_cartpole.py configs/cartpole_random.json

# Enable development mode
python examples/train_cartpole.py configs/cartpole_random.json --development

# Run evaluation after training
python examples/train_cartpole.py configs/cartpole_random.json --evaluate 100

# Save checkpoint
python examples/train_cartpole.py configs/cartpole_random.json --checkpoint final_model.json
```

## Configuration Reference

### Basic Structure

```json
{
  "seed": 42,                    // Required for reproducibility
  "development_mode": false,     // Skip version validation

  "packages": {                  // Package specifications
    "package_name": {
      "version": "1.0.0",        // Exact version required
      "components": ["Component1", "Component2"]
    },
    "local_package": {
      "path": "./my_components",  // Local development package
      "git_commit": "abc123",     // Optional commit hash
      "components": ["MyComponent"]
    }
  },

  "components": {                // Component instances
    "my_agent": {
      "package": "package_name",
      "type": "Component1",
      "param1": "value1"
    }
  },

  "hooks": {                     // Hook execution order
    "action_selection": ["my_agent"],
    "learning_update": ["replay", "my_agent", "replay"]
  },

  "training": {                  // Training parameters
    "max_episodes": 1000,
    "max_steps_per_episode": 500
  }
}
```

### Available Hooks

All 25 available hooks with their purposes and expected context modifications:

| Hook | Purpose | Expected Context Additions |
|------|---------|---------------------------|
| **Core Training Lifecycle** |
| `training_start` | Initialize training | - |
| `training_end` | Finalize training | - |
| `episode_start` | Start new episode | - |
| `episode_reset` | Reset environment | `state` |
| `episode_end` | End episode processing | - |
| `step_start` | Start individual step | - |
| `step_end` | End individual step | - |
| **Action & Environment** |
| `observation_received` | Process observation | - |
| `action_selection` | Choose action | `action` |
| `action_chosen` | Post-action processing | - |
| `environment_step` | Step environment | `next_state`, `reward`, `done` |
| `transition_received` | Process transition | - |
| **Learning & Data** |
| `experience_storage` | Store experience | - |
| `learning_update` | Update model | - |
| `metrics_computation` | Compute metrics | - |
| **Persistence & Logging** |
| `checkpoint_save` | Save checkpoints | - |
| `logging_update` | Update logs | - |
| `visualization_update` | Update visualizations | - |
| **Evaluation** |
| `evaluation_start` | Start evaluation | - |
| `evaluation_end` | End evaluation | - |
| `evaluation_episode_start` | Start eval episode | - |
| `evaluation_episode_end` | End eval episode | - |
| **Batch Processing** |
| `epoch_start` | Start training epoch | - |
| `epoch_end` | End training epoch | - |
| `batch_start` | Start batch processing | - |
| `batch_end` | End batch processing | - |

## Creating Custom Components

### 1. Basic Component

```python
from rltoolbox import RLComponent
import numpy as np

class MyAgent(RLComponent):
    def __init__(self, config):
        super().__init__(config)
        self.learning_rate = config.get("learning_rate", 0.001)

    def action_selection(self, context):
        # Select action based on current state
        state = context["state"]
        action = self.policy(state)
        context["action"] = action

    def learning_update(self, context):
        # Update model with experience
        if "batch" in context:
            loss = self.update_model(context["batch"])
            context["loss"] = loss
```

### 2. Multi-Hook Component

```python
class ComplexAgent(RLComponent):
    def episode_start(self, context):
        # Initialize episode-specific state
        self.episode_actions = []

    def action_selection(self, context):
        # Different behavior on multiple calls within same hook
        run_count = context["_run_count"]
        if run_count == 1:
            # First call: exploration
            action = self.explore(context["state"])
        else:
            # Second call: exploitation
            action = self.exploit(context["state"])

        context["action"] = action
        self.episode_actions.append(action)

    def episode_end(self, context):
        # Process full episode
        self.update_policy(self.episode_actions, context["training"]["episode_reward"])
```

### 3. Package Registration

```python
# In your package's __init__.py
from rltoolbox import get_registry

# Register your components
registry = get_registry()
registry.register_component_type("MyAgent", MyAgent)
registry.register_component_type("ComplexAgent", ComplexAgent)
```

## Built-in Components

### Environments
- `SimpleEnvironment`: Gymnasium environment wrapper
- `RandomEnvironment`: Random environment for testing
- `EnvironmentWrapper`: Observation/reward preprocessing

### Agents
- `RandomAgent`: Random action selection
- `EpsilonGreedyAgent`: Epsilon-greedy policy
- `GreedyAgent`: Greedy action selection
- `SoftmaxAgent`: Boltzmann exploration
- `ReplayBufferAgent`: Experience replay management

### Loggers
- `ConsoleLogger`: Terminal output
- `FileLogger`: JSON file logging
- `CSVLogger`: CSV format logging
- `MetricsLogger`: Statistics computation

## Example Configurations

### DQN-style Training
```json
{
  "hooks": {
    "episode_reset": ["env"],
    "action_selection": ["dqn_model"],
    "environment_step": ["env"],
    "experience_storage": ["replay_buffer"],
    "learning_update": ["replay_buffer", "dqn_model"]
  }
}
```

### Policy Gradient
```json
{
  "hooks": {
    "episode_reset": ["env"],
    "action_selection": ["policy_network"],
    "environment_step": ["env"],
    "episode_end": ["policy_gradient_update"]
  }
}
```

### Multi-Agent
```json
{
  "hooks": {
    "action_selection": ["agent1", "agent2"],
    "environment_step": ["multi_agent_env"],
    "learning_update": ["agent1", "agent2"]
  }
}
```

## Advanced Features

### Development Workflow
- Use `"development_mode": true` to skip version validation
- Local packages with `"path": "./my_dev"` for iteration
- Automatic built-in component registration

### Reproducibility
- Required seed specification
- Exact package version pinning
- Complete experiment specification in JSON
- No runtime discovery or fallbacks

### Performance
- Minimal hook execution overhead
- Lazy component loading
- Optional context validation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for your changes
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use RLtoolbox in your research, please cite:

```bibtex
@software{rltoolbox2024,
  title={RLtoolbox: A Highly Configurable Reinforcement Learning Framework},
  author={RLtoolbox Contributors},
  year={2025},
  url={https://github.com/lucamignatti/RLtoolbox}
}
```
