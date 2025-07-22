import pytest
import warnings
import os
import tempfile
from unittest.mock import patch, MagicMock

from rltoolbox.core.trainer import RLTrainer
from rltoolbox.core.base import RLComponent


class MockComponent(RLComponent):
    def __init__(self, config):
        super().__init__(config)
        self.call_log = []

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
        context["done"] = True

    def episode_end(self, context):
        self.call_log.append("episode_end")

    def training_end(self, context):
        self.call_log.append("training_end")


def test_trainer_initialization():
    """Test basic trainer initialization."""
    config = {
        "components": {},
        "hooks": {},
    }
    trainer = RLTrainer(config_dict=config)
    assert trainer.config == config
    assert "components" in trainer.context
    assert "config" in trainer.context
    assert "training" in trainer.context
    assert "metrics" in trainer.context


def test_trainer_initialization_with_seed():
    """Test trainer initialization with seed."""
    config = {
        "seed": 42,
        "components": {},
        "hooks": {},
    }
    trainer = RLTrainer(config_dict=config)
    assert trainer.config["seed"] == 42


def test_trainer_initialization_missing_required_sections():
    """Test trainer initialization fails with missing required sections."""
    # Missing components section
    config = {"hooks": {}}
    with pytest.raises(ValueError, match="Missing required configuration section: components"):
        RLTrainer(config_dict=config)

    # Missing hooks section
    config = {"components": {}}
    with pytest.raises(ValueError, match="Missing required configuration section: hooks"):
        RLTrainer(config_dict=config)


def test_trainer_initialization_invalid_components():
    """Test trainer initialization fails with invalid components configuration."""
    # Components not a dict
    config = {
        "components": "invalid",
        "hooks": {}
    }
    with pytest.raises(ValueError, match="'components' section must be a dictionary"):
        RLTrainer(config_dict=config)

    # Component config not a dict
    config = {
        "components": {"comp1": "invalid"},
        "hooks": {}
    }
    with pytest.raises(ValueError, match="Component 'comp1' configuration must be a dictionary"):
        RLTrainer(config_dict=config)

    # Missing type field
    config = {
        "components": {"comp1": {"package": "test"}},
        "hooks": {}
    }
    with pytest.raises(ValueError, match="Component 'comp1' missing 'type' field"):
        RLTrainer(config_dict=config)


def test_trainer_runs_hooks():
    """Test that trainer properly executes hooks."""
    config = {
        "seed": 42,
        "components": {
            "mock_component": {
                "package": "mock_package",
                "type": "MockComponent"
            }
        },
        "hooks": {
            "training_start": ["mock_component"]
        },
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponent"]
            }
        },
        "training": {
            "max_episodes": 0
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponent = MockComponent
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)
            trainer.train()

    assert trainer.context.get("training_started") is True
    assert "mock_component" in trainer.components
    component = trainer.components["mock_component"]
    assert "training_start" in component.call_log


def test_trainer_full_episode():
    """Test trainer runs a full episode with all hooks."""
    config = {
        "seed": 42,
        "components": {
            "mock_component": {
                "package": "mock_package",
                "type": "MockComponent"
            }
        },
        "hooks": {
            "training_start": ["mock_component"],
            "episode_reset": ["mock_component"],
            "action_selection": ["mock_component"],
            "environment_step": ["mock_component"],
            "episode_end": ["mock_component"],
            "training_end": ["mock_component"]
        },
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponent"]
            }
        },
        "training": {
            "max_episodes": 1,
            "max_steps_per_episode": 1
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponent = MockComponent
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)
            trainer.train()

    component = trainer.components["mock_component"]
    expected_hooks = ["training_start", "episode_reset", "action_selection",
                     "environment_step", "episode_end", "training_end"]

    for hook in expected_hooks:
        assert hook in component.call_log, f"Hook {hook} was not called"

    # Check metrics were recorded
    assert len(trainer.context["metrics"]["episode_rewards"]) == 1
    assert len(trainer.context["metrics"]["episode_lengths"]) == 1


def test_trainer_components_in_context():
    """Test that components are available in context."""
    config = {
        "seed": 42,
        "components": {
            "mock_component": {
                "package": "mock_package",
                "type": "MockComponent"
            }
        },
        "hooks": {},
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponent"]
            }
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponent = MockComponent
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)

    # Verify components are in context
    assert "components" in trainer.context
    assert "mock_component" in trainer.context["components"]
    assert trainer.context["components"]["mock_component"] is trainer.components["mock_component"]


def test_trainer_evaluation():
    """Test trainer evaluation functionality."""
    config = {
        "seed": 42,
        "components": {
            "mock_component": {
                "package": "mock_package",
                "type": "MockComponent"
            }
        },
        "hooks": {
            "episode_reset": ["mock_component"],
            "action_selection": ["mock_component"],
            "environment_step": ["mock_component"]
        },
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponent"]
            }
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponent = MockComponent
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)
            results = trainer.evaluate(num_episodes=2)

    # Check evaluation results structure
    assert "mean_reward" in results
    assert "std_reward" in results
    assert "mean_length" in results
    assert "std_length" in results
    assert "episode_rewards" in results
    assert "episode_lengths" in results
    assert len(results["episode_rewards"]) == 2
    assert len(results["episode_lengths"]) == 2


def test_trainer_config_validation_warnings():
    """Test that trainer produces appropriate warnings."""
    config = {
        "components": {},
        "hooks": {}
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trainer = RLTrainer(config_dict=config)

        # Should warn about missing seed
        assert len(w) >= 1
        assert any("No seed specified" in str(warning.message) for warning in w)


def test_trainer_checkpoint_functionality():
    """Test trainer save/load checkpoint functionality."""
    config = {
        "seed": 42,
        "components": {
            "mock_component": {
                "package": "mock_package",
                "type": "MockComponent"
            }
        },
        "hooks": {},
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponent"]
            }
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponent = MockComponent
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)

            # Test that save_checkpoint doesn't crash
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
                tmp_path = tmp_file.name

            try:
                trainer.save_checkpoint(tmp_path)
                # Verify checkpoint file was created
                assert os.path.exists(tmp_path)

                # Test loading the checkpoint
                trainer2 = RLTrainer(config_dict=config)
                trainer2.load_checkpoint(tmp_path)

                # Verify components are still in context after loading
                assert "components" in trainer2.context
                assert "mock_component" in trainer2.context["components"]

            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
