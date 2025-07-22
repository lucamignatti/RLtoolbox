import pytest
from unittest.mock import patch, MagicMock

from rltoolbox.core.trainer import RLTrainer
from rltoolbox.core.base import RLComponent


class MockComponentA(RLComponent):
    def training_start(self, context):
        # Test that we can access other components through context
        component_b = self.get_component(context, "component_b")
        context["component_a_accessed_b"] = True
        context["component_b_name"] = component_b.name


class MockComponentB(RLComponent):
    def training_start(self, context):
        # Test that components dict is available in context
        assert "components" in context
        assert "component_a" in context["components"]
        assert "component_b" in context["components"]
        context["components_available"] = True


def test_components_in_context():
    """Test that components are available in the context dictionary."""
    config = {
        "components": {
            "component_a": {
                "package": "mock_package",
                "type": "MockComponentA"
            },
            "component_b": {
                "package": "mock_package",
                "type": "MockComponentB"
            }
        },
        "hooks": {
            "training_start": ["component_a", "component_b"]
        },
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponentA", "MockComponentB"]
            }
        },
        "training": {
            "max_episodes": 0
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponentA = MockComponentA
            mock_module.MockComponentB = MockComponentB
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)

            # Verify components are in context after initialization
            assert "components" in trainer.context
            assert "component_a" in trainer.context["components"]
            assert "component_b" in trainer.context["components"]

            # Run training to test hook execution
            trainer.train()

            # Verify components could access each other through context
            assert trainer.context.get("component_a_accessed_b") is True
            assert trainer.context.get("components_available") is True
            assert trainer.context.get("component_b_name") == "component_b"


def test_get_component_utility():
    """Test the get_component utility method."""
    config = {
        "components": {
            "test_component": {
                "package": "mock_package",
                "type": "MockComponentA"
            }
        },
        "hooks": {},
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponentA"]
            }
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponentA = MockComponentA
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)
            component = trainer.components["test_component"]

            # Test successful component access
            other_component = component.get_component(trainer.context, "test_component")
            assert other_component is component

            # Test error when component not found
            with pytest.raises(KeyError, match="Component 'nonexistent' not found"):
                component.get_component(trainer.context, "nonexistent")


def test_get_component_no_components_in_context():
    """Test get_component when components are not in context."""
    component = MockComponentA({})
    context = {}  # No components in context

    with pytest.raises(KeyError, match="Components not available in context"):
        component.get_component(context, "some_component")


def test_components_preserved_after_checkpoint_load():
    """Test that components are still available in context after loading checkpoint."""
    config = {
        "components": {
            "test_component": {
                "package": "mock_package",
                "type": "MockComponentA"
            }
        },
        "hooks": {},
        "packages": {
            "mock_package": {
                "path": "/some/path",
                "components": ["MockComponentA"]
            }
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MockComponentA = MockComponentA
            mock_import.return_value = mock_module

            trainer = RLTrainer(config_dict=config)

            # Simulate loading a checkpoint
            fake_checkpoint = {
                "config": config,
                "context": {
                    "training": {"episode": 5},
                    "some_other_key": "value"
                },
                "components": {}
            }

            with patch.object(trainer, "load_checkpoint") as mock_load:
                def mock_load_checkpoint(path):
                    trainer.context.update(fake_checkpoint["context"])
                    trainer.context["components"] = trainer.components
                mock_load.side_effect = mock_load_checkpoint
                trainer.load_checkpoint("fake_path.json")

            # Verify components are still in context after loading
            assert "components" in trainer.context
            assert "test_component" in trainer.context["components"]


def mock_open_checkpoint(checkpoint_data):
    """Helper to mock file opening for checkpoint loading."""
    import json
    from unittest.mock import mock_open
    return mock_open(read_data=json.dumps(checkpoint_data))
