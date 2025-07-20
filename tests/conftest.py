"""
Pytest configuration and fixtures for RLtoolbox tests.

This module provides common fixtures and utilities for testing the RLtoolbox framework.
"""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

import sys
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from rltoolbox import RLTrainer, RLComponent
from rltoolbox.core.registry import get_registry


@pytest.fixture
def simple_config():
    """Basic configuration for testing."""
    return {
        "seed": 42,
        "development_mode": True,
        "packages": {},
        "components": {
            "env": {
                "package": "rltoolbox",
                "type": "RandomEnvironment",
                "state_dim": 4,
                "action_dim": 2,
                "episode_length": 5,
                "reward_range": [-1.0, 1.0]
            },
            "agent": {
                "package": "rltoolbox",
                "type": "RandomAgent",
                "action_space_type": "discrete",
                "num_actions": 2
            },
            "logger": {
                "package": "rltoolbox",
                "type": "ConsoleLogger",
                "log_frequency": 1,
                "verbose": False,
                "log_steps": False
            }
        },
        "hooks": {
            "training_start": ["logger"],
            "episode_reset": ["env"],
            "action_selection": ["agent"],
            "environment_step": ["env"],
            "episode_end": ["logger"],
            "training_end": ["logger"]
        },
        "training": {
            "max_episodes": 3,
            "max_steps_per_episode": 5
        }
    }


@pytest.fixture
def cartpole_config():
    """CartPole configuration for testing."""
    return {
        "seed": 42,
        "development_mode": True,
        "packages": {},
        "components": {
            "env": {
                "package": "rltoolbox",
                "type": "SimpleEnvironment",
                "env_name": "CartPole-v1",
                "max_episode_steps": 50
            },
            "agent": {
                "package": "rltoolbox",
                "type": "MLPAgent",
                "state_dim": 4,
                "action_dim": 2,
                "hidden_layers": [16, 16],
                "learning_rate": 0.01,
            },
            "exploration_agent": {
                "package": "rltoolbox",
                "type": "EpsilonGreedyAgent",
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay": 0.9,
                "num_actions": 2,
                "decay_type": "exponential"
            },
            "logger": {
                "package": "rltoolbox",
                "type": "ConsoleLogger",
                "log_frequency": 5,
                "verbose": False
            }
        },
        "hooks": {
            "episode_reset": ["env"],
            "action_selection": ["agent", "exploration_agent"],
            "environment_step": ["env"],
            "episode_end": ["logger"]
        },
        "training": {
            "max_episodes": 5,
            "max_steps_per_episode": 50
        }
    }


@pytest.fixture
def temp_config_file(simple_config):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(simple_config, f, indent=2)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def temp_checkpoint_file():
    """Create a temporary checkpoint file path."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        temp_path = f.name

    # Delete the file so trainer can create it
    Path(temp_path).unlink()

    yield temp_path

    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def trainer(simple_config):
    """Create a trainer instance with simple config."""
    return RLTrainer(config_dict=simple_config)


@pytest.fixture
def cartpole_trainer(cartpole_config):
    """Create a trainer instance with CartPole config."""
    return RLTrainer(config_dict=cartpole_config)


@pytest.fixture
def reset_registry():
    """Reset the global registry between tests."""
    registry = get_registry()

    # Store original state
    original_components = registry._components.copy()
    original_packages = registry._packages.copy()
    original_dev_mode = registry._development_mode

    yield registry

    # Restore original state
    registry._components = original_components
    registry._packages = original_packages
    registry._development_mode = original_dev_mode


class MockComponent(RLComponent):
    """Mock component for testing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.call_log = []

    def episode_start(self, context: Dict[str, Any]) -> None:
        self.call_log.append(("episode_start", context.get("_run_count", 1)))

    def action_selection(self, context: Dict[str, Any]) -> None:
        self.call_log.append(("action_selection", context.get("_run_count", 1)))
        context["action"] = 0

    def environment_step(self, context: Dict[str, Any]) -> None:
        self.call_log.append(("environment_step", context.get("_run_count", 1)))
        context["next_state"] = [0, 0, 0, 0]
        context["reward"] = 1.0
        context["done"] = False

    def learning_update(self, context: Dict[str, Any]) -> None:
        self.call_log.append(("learning_update", context.get("_run_count", 1)))

    def episode_end(self, context: Dict[str, Any]) -> None:
        self.call_log.append(("episode_end", context.get("_run_count", 1)))


@pytest.fixture
def mock_component():
    """Create a mock component for testing."""
    config = {"name": "mock_component"}
    return MockComponent(config)


@pytest.fixture
def mock_config_with_multiple_calls():
    """Configuration that tests multiple component calls in same hook."""
    return {
        "seed": 42,
        "development_mode": True,
        "packages": {},
        "components": {
            "mock1": {
                "package": "test",
                "type": "MockComponent"
            },
            "mock2": {
                "package": "test",
                "type": "MockComponent"
            }
        },
        "hooks": {
            "action_selection": ["mock1", "mock2", "mock1"],
            "learning_update": ["mock2", "mock1", "mock2", "mock1"]
        },
        "training": {
            "max_episodes": 1,
            "max_steps_per_episode": 2
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Reset random seeds for reproducible tests
    import random
    import numpy as np

    random.seed(42)
    np.random.seed(42)

    yield

    # Cleanup after test
    pass
