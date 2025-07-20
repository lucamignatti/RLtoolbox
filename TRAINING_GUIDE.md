# RLtoolbox Standalone Training Guide

This guide explains how to use RLtoolbox's standalone training command that loads JSON configurations and executes training with automatic evaluation, requiring no wrapping code.

## Overview

The `rltoolbox-train` command provides a complete training solution that:
- Loads configuration from JSON files
- Executes training loops without any wrapping code influence
- Automatically runs evaluation at the end if configured
- Saves comprehensive logs and metrics
- Supports checkpointing and resuming

## Installation

First, install RLtoolbox:

```bash
pip install -e .
```

This will make the `rltoolbox-train` command available in your environment.

## Basic Usage

### Simple Training

```bash
rltoolbox-train configs/cartpole_random.json
```

### Training with Verbose Output

```bash
rltoolbox-train configs/cartpole_epsilon_greedy.json --verbose
```

### Training with Checkpointing

```bash
# Save checkpoint after training
rltoolbox-train configs/cartpole_epsilon_greedy.json --checkpoint model.ckpt

# Resume from checkpoint
rltoolbox-train configs/cartpole_epsilon_greedy.json --load-checkpoint model.ckpt
```

### Skip Automatic Evaluation

```bash
rltoolbox-train configs/cartpole_epsilon_greedy.json --no-evaluation
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `config` | Path to JSON configuration file (required) |
| `--checkpoint PATH` | Save final checkpoint to PATH |
| `--load-checkpoint PATH` | Load checkpoint from PATH before training |
| `--verbose` | Enable verbose output for debugging |
| `--no-evaluation` | Skip automatic evaluation even if configured |

## JSON Configuration Format

### Basic Structure

```json
{
  "seed": 42,
  "development_mode": false,
  "packages": {},
  "components": { ... },
  "hooks": { ... },
  "training": { ... },
  "evaluation": { ... }
}
```

### Training Configuration

```json
"training": {
  "max_episodes": 1000,
  "max_steps_per_episode": 500,
  "max_total_steps": null
}
```

### Evaluation Configuration

The evaluation section enables automatic evaluation at the end of training:

```json
"evaluation": {
  "enabled": true,
  "num_episodes": 10,
  "hooks": {
    "evaluation_start": ["console_logger", "file_logger", "metrics"],
    "evaluation_episode_end": ["console_logger", "file_logger", "metrics"],
    "evaluation_end": ["console_logger", "file_logger", "metrics"]
  }
}
```

#### Evaluation Options

- `enabled`: Boolean to enable/disable automatic evaluation
- `num_episodes`: Number of episodes to run during evaluation
- `hooks`: Hook configuration specific to evaluation (optional)

If `hooks` is provided in the evaluation section, these hooks will temporarily override the main hooks during evaluation.

## Example Configurations

### Basic Random Agent

```json
{
  "seed": 42,
  "components": {
    "env": {
      "package": "rltoolbox",
      "type": "SimpleEnvironment",
      "env_name": "CartPole-v1"
    },
    "agent": {
      "package": "rltoolbox",
      "type": "RandomAgent",
      "num_actions": 2
    },
    "console_logger": {
      "package": "rltoolbox",
      "type": "ConsoleLogger",
      "log_frequency": 10
    }
  },
  "hooks": {
    "training_start": ["console_logger"],
    "episode_reset": ["env"],
    "action_selection": ["agent"],
    "environment_step": ["env"],
    "episode_end": ["console_logger"],
    "training_end": ["console_logger"]
  },
  "training": {
    "max_episodes": 100
  },
  "evaluation": {
    "enabled": true,
    "num_episodes": 10
  }
}
```

### Advanced Configuration with Multiple Loggers

```json
{
  "seed": 42,
  "components": {
    "env": { ... },
    "agent": { ... },
    "console_logger": { ... },
    "file_logger": {
      "package": "rltoolbox",
      "type": "FileLogger",
      "log_dir": "./logs/experiment",
      "save_frequency": 50
    },
    "csv_logger": {
      "package": "rltoolbox",
      "type": "CSVLogger",
      "log_dir": "./logs/experiment",
      "filename_prefix": "training"
    },
    "metrics": {
      "package": "rltoolbox",
      "type": "MetricsLogger",
      "window_size": 100
    }
  },
  "hooks": {
    "episode_end": ["console_logger", "file_logger", "csv_logger", "metrics"]
  },
  "evaluation": {
    "enabled": true,
    "num_episodes": 20,
    "hooks": {
      "evaluation_start": ["console_logger", "file_logger", "metrics"],
      "evaluation_episode_end": ["console_logger", "file_logger", "csv_logger", "metrics"],
      "evaluation_end": ["console_logger", "file_logger", "metrics"]
    }
  }
}
```

## Output and Logging

### Console Output

The training command provides real-time feedback:

```
Loading configuration from: configs/cartpole_epsilon_greedy.json
Starting training...
=== Training Started ===
Episode    0 | Reward:     9.00 | Length:   9 | Steps: 9
Episode   10 | Reward:    15.00 | Length:  15 | Steps: 203
...
=== Training Completed ===
Total Episodes: 1000
Training Time: 0.86 seconds

Running automatic evaluation for 10 episodes...
=== Evaluation Started ===
Eval Episode  0 | Reward:    13.00 | Length:  13
...
=== Evaluation Completed ===
Automatic Evaluation Results:
  Mean Reward: 23.20 ± 8.23
  Mean Length: 23.20 ± 8.23
```

### File Outputs

When using `FileLogger`, the following files are created:

- `config.json`: Complete configuration used for training
- `episodes.json`: Detailed episode data
- `evaluation_episodes.json`: Evaluation episode data
- `evaluation_info.json`: Evaluation metadata
- `summary.json`: Final training summary

When using `CSVLogger`, CSV files are created for easy analysis in spreadsheet applications or data science tools.

## Evaluation Hooks

The evaluation system supports the following hooks that can be configured with different components:

- `evaluation_start`: Called at the beginning of evaluation
- `evaluation_episode_start`: Called at the start of each evaluation episode
- `evaluation_episode_end`: Called at the end of each evaluation episode
- `evaluation_end`: Called at the completion of evaluation

These hooks allow you to:
- Log evaluation progress differently from training
- Compute evaluation-specific metrics
- Save evaluation data separately
- Implement custom evaluation behaviors

## Best Practices

1. **Always set a seed** for reproducible results
2. **Use verbose mode** when debugging configurations
3. **Save checkpoints** for long training runs
4. **Configure evaluation** to automatically assess performance
5. **Use multiple loggers** to capture different types of data
6. **Organize logs** with descriptive directory names

## Troubleshooting

### Configuration Errors

- Ensure all required components are specified
- Check that hook references match component names
- Verify JSON syntax is valid

### Missing Dependencies

- Install required packages listed in the configuration
- Ensure RLtoolbox is properly installed

### Performance Issues

- Reduce logging frequency for long training runs
- Disable step-level logging unless needed
- Use appropriate buffer sizes for replay-based agents

## Examples

See the `configs/` directory for working examples:

- `cartpole_random.json`: Basic random agent
- `cartpole_epsilon_greedy.json`: Epsilon-greedy agent with replay buffer
- `cartpole_comprehensive_eval.json`: Full-featured configuration with comprehensive evaluation

Run any of these with:

```bash
rltoolbox-train configs/example_name.json --verbose
```
