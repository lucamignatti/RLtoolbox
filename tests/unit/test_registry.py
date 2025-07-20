"""
Unit tests for the component registry system.

Tests registration, loading, and validation of components from different packages.
"""

import pytest
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from rltoolbox.core.registry import ComponentRegistry, get_registry
from rltoolbox.core.base import RLComponent


class DummyComponent(RLComponent):
    """Dummy component for testing."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.test_param = config.get("test_param", "default")

    def action_selection(self, context: Dict[str, Any]) -> None:
        context["action"] = self.test_param

    def validate_config(self) -> bool:
        return "test_param" in self.config


class InvalidComponent:
    """Component that doesn't inherit from RLComponent."""

    def __init__(self, config):
        self.config = config


class TestComponentRegistry:
    """Test the component registry system."""

    def test_initialization(self):
        """Test registry initialization."""
        registry = ComponentRegistry()

        assert registry._components == {}
        assert registry._packages == {}
        assert registry._development_mode is False

    def test_development_mode(self):
        """Test development mode setting."""
        registry = ComponentRegistry()

        # Test enabling development mode
        with pytest.warns(UserWarning, match="Development mode enabled"):
            registry.set_development_mode(True)

        assert registry._development_mode is True

        # Test disabling development mode
        registry.set_development_mode(False)
        assert registry._development_mode is False

    def test_register_component_directly(self):
        """Test direct component registration."""
        registry = ComponentRegistry()

        # Register component directly
        registry._components["test.DummyComponent"] = DummyComponent

        # Verify it's registered
        assert "test.DummyComponent" in registry._components
        assert registry._components["test.DummyComponent"] == DummyComponent

    def test_get_component_class_success(self):
        """Test successful component class retrieval."""
        registry = ComponentRegistry()
        registry._components["test_package.DummyComponent"] = DummyComponent
        registry._packages["test_package"] = {"version": "1.0.0", "components": ["DummyComponent"]}

        component_class = registry.get_component_class("test_package", "DummyComponent")
        assert component_class == DummyComponent

    def test_get_component_class_not_found(self):
        """Test component class retrieval when component not found."""
        registry = ComponentRegistry()
        registry._packages["test_package"] = {"version": "1.0.0", "components": ["DummyComponent"]}

        with pytest.raises(ValueError, match="Component test_package.NonExistentComponent not found"):
            registry.get_component_class("test_package", "NonExistentComponent")

    def test_get_component_class_package_not_registered(self):
        """Test component class retrieval when package not registered."""
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="Package unknown_package not registered"):
            registry.get_component_class("unknown_package", "SomeComponent")

    def test_create_component_success(self):
        """Test successful component creation."""
        registry = ComponentRegistry()
        registry._components["test_package.DummyComponent"] = DummyComponent
        registry._packages["test_package"] = {"version": "1.0.0", "components": ["DummyComponent"]}

        config = {"test_param": "test_value"}
        component = registry.create_component("test_package", "DummyComponent", config)

        assert isinstance(component, DummyComponent)
        assert component.test_param == "test_value"

    def test_create_component_invalid_config(self):
        """Test component creation with invalid config."""
        registry = ComponentRegistry()
        registry._components["test_package.DummyComponent"] = DummyComponent
        registry._packages["test_package"] = {"version": "1.0.0", "components": ["DummyComponent"]}

        # Config missing required test_param
        config = {"other_param": "value"}

        with pytest.raises(ValueError, match="Invalid configuration"):
            registry.create_component("test_package", "DummyComponent", config)

    def test_list_components(self):
        """Test listing registered components."""
        registry = ComponentRegistry()

        # Initially empty
        assert registry.list_components() == []

        # Add some components
        registry._components["pkg1.Comp1"] = DummyComponent
        registry._components["pkg2.Comp2"] = DummyComponent

        components = registry.list_components()
        assert set(components) == {"pkg1.Comp1", "pkg2.Comp2"}

    @patch('rltoolbox.core.registry.importlib.import_module')
    def test_import_component_success(self, mock_import):
        """Test successful component import."""
        registry = ComponentRegistry()

        # Mock module with component
        mock_module = MagicMock()
        mock_module.DummyComponent = DummyComponent
        mock_import.return_value = mock_module

        component_class = registry._import_component("test_package.DummyComponent")

        assert component_class == DummyComponent
        mock_import.assert_called_once_with("test_package")

    @patch('rltoolbox.core.registry.importlib.import_module')
    def test_import_component_module_not_found(self, mock_import):
        """Test component import when module not found."""
        registry = ComponentRegistry()

        mock_import.side_effect = ImportError("Module not found")

        with pytest.raises(ValueError, match="Could not import component"):
            registry._import_component("nonexistent.Component")

    @patch('rltoolbox.core.registry.importlib.import_module')
    def test_import_component_class_not_found(self, mock_import):
        """Test component import when class not found in module."""
        registry = ComponentRegistry()

        # Mock module without the component
        mock_module = MagicMock()
        del mock_module.NonExistentComponent  # Ensure it doesn't exist
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="Component NonExistentComponent not found"):
            registry._import_component("test_package.NonExistentComponent")

    @patch('rltoolbox.core.registry.importlib.import_module')
    def test_import_component_invalid_inheritance(self, mock_import):
        """Test component import when class doesn't inherit from RLComponent."""
        registry = ComponentRegistry()

        # Mock module with invalid component
        mock_module = MagicMock()
        mock_module.InvalidComponent = InvalidComponent
        mock_import.return_value = mock_module

        with pytest.raises(ValueError, match="must inherit from RLComponent"):
            registry._import_component("test_package.InvalidComponent")

    def test_register_packages(self):
        """Test registering multiple packages."""
        registry = ComponentRegistry()
        registry.set_development_mode(True)  # Skip validation

        packages_config = {
            "package1": {
                "version": "1.0.0",
                "components": ["Component1"]
            },
            "package2": {
                "version": "2.0.0",
                "components": ["Component2"]
            }
        }

        with patch.object(registry, '_load_package_components'):
            registry.register_packages(packages_config)

        assert "package1" in registry._packages
        assert "package2" in registry._packages
        assert registry._packages["package1"]["version"] == "1.0.0"
        assert registry._packages["package2"]["version"] == "2.0.0"

    @patch('rltoolbox.core.registry.pkg_resources.get_distribution')
    def test_validate_package_installed_correct_version(self, mock_get_dist):
        """Test package validation with correct installed version."""
        registry = ComponentRegistry()

        # Mock installed package
        mock_dist = MagicMock()
        mock_dist.version = "1.0.0"
        mock_get_dist.return_value = mock_dist

        package_config = {"version": "1.0.0", "components": []}

        # Should not raise
        registry._validate_package("test_package", package_config)

    @patch('rltoolbox.core.registry.pkg_resources.get_distribution')
    def test_validate_package_installed_wrong_version(self, mock_get_dist):
        """Test package validation with wrong installed version."""
        registry = ComponentRegistry()

        # Mock installed package with wrong version
        mock_dist = MagicMock()
        mock_dist.version = "2.0.0"
        mock_get_dist.return_value = mock_dist

        package_config = {"version": "1.0.0", "components": []}

        with pytest.raises(ValueError, match="Version mismatch"):
            registry._validate_package("test_package", package_config)

    @patch('rltoolbox.core.registry.pkg_resources.get_distribution')
    def test_validate_package_not_installed(self, mock_get_dist):
        """Test package validation when package not installed."""
        registry = ComponentRegistry()

        # Mock package not found
        from pkg_resources import DistributionNotFound
        mock_get_dist.side_effect = DistributionNotFound()

        package_config = {"version": "1.0.0", "components": []}

        with pytest.raises(ValueError, match="Package test_package not found"):
            registry._validate_package("test_package", package_config)

    def test_validate_package_dev_version(self):
        """Test package validation with dev version."""
        registry = ComponentRegistry()

        package_config = {"version": "dev", "components": []}

        # Should not raise (dev version skips validation)
        registry._validate_package("test_package", package_config)

    def test_validate_package_local_path_exists(self):
        """Test package validation with existing local path."""
        registry = ComponentRegistry()

        with patch('pathlib.Path.exists', return_value=True):
            package_config = {"path": "/some/path", "components": []}

            # Should not raise
            registry._validate_package("test_package", package_config)

    def test_validate_package_local_path_not_exists(self):
        """Test package validation with non-existent local path."""
        registry = ComponentRegistry()

        with patch('pathlib.Path.exists', return_value=False):
            package_config = {"path": "/nonexistent/path", "components": []}

            with pytest.raises(ValueError, match="Local package path does not exist"):
                registry._validate_package("test_package", package_config)

    def test_validate_package_no_version_or_path(self):
        """Test package validation with neither version nor path."""
        registry = ComponentRegistry()

        package_config = {"components": []}

        with pytest.raises(ValueError, match="must specify either 'version' or 'path'"):
            registry._validate_package("test_package", package_config)

    @patch('rltoolbox.core.registry.subprocess.run')
    def test_validate_package_git_commit_match(self, mock_run):
        """Test package validation with matching git commit."""
        registry = ComponentRegistry()

        # Mock git command success
        mock_result = MagicMock()
        mock_result.stdout = "abc123def456\n"
        mock_run.return_value = mock_result

        with patch('pathlib.Path.exists', return_value=True):
            package_config = {
                "path": "/some/path",
                "git_commit": "abc123",
                "components": []
            }

            # Should not raise
            registry._validate_package("test_package", package_config)

    @patch('rltoolbox.core.registry.subprocess.run')
    def test_validate_package_git_commit_mismatch(self, mock_run):
        """Test package validation with mismatching git commit."""
        registry = ComponentRegistry()

        # Mock git command success with different commit
        mock_result = MagicMock()
        mock_result.stdout = "different123\n"
        mock_run.return_value = mock_result

        with patch('pathlib.Path.exists', return_value=True):
            package_config = {
                "path": "/some/path",
                "git_commit": "abc123",
                "components": []
            }

            with pytest.warns(UserWarning, match="Git commit mismatch"):
                registry._validate_package("test_package", package_config)

    def test_register_builtin_components(self):
        """Test registration of built-in components."""
        registry = ComponentRegistry()

        registry.register_builtin_components()

        # Check that built-in components are registered
        components = registry.list_components()
        expected_components = [
            "rltoolbox.SimpleEnvironment",
            "rltoolbox.MLPAgent",
            "rltoolbox.EpsilonGreedyAgent",
            "rltoolbox.PolicyAgent",
            "rltoolbox.ReplayBufferAgent",
            "rltoolbox.ConsoleLogger",
            "rltoolbox.FileLogger",
            "rltoolbox.CSVLogger",
            "rltoolbox.MetricsLogger",
            "rltoolbox.WandbLogger",
            "rltoolbox.PPO",
        ]

        for component in expected_components:
            assert component in components

        # Check that rltoolbox package is registered
        assert "rltoolbox" in registry._packages

    def test_development_mode_skips_validation(self):
        """Test that development mode skips package validation."""
        registry = ComponentRegistry()
        registry.set_development_mode(True)

        # This would normally fail validation but should pass in dev mode
        package_config = {"version": "nonexistent", "components": []}

        with patch.object(registry, '_load_package_components'):
            # Should not raise
            registry._register_package("test_package", package_config)

    def test_global_registry_singleton(self):
        """Test that get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()

        assert registry1 is registry2

    @patch('sys.path')
    def test_load_local_component_adds_to_path(self, mock_path):
        """Test that loading local components adds path to sys.path."""
        registry = ComponentRegistry()
        mock_path.__contains__ = lambda self, item: False  # Path not in sys.path
        mock_path.insert = MagicMock()

        result = registry._load_local_component("test_pkg", "/some/path", "Component")

        mock_path.insert.assert_called_once_with(0, "/some/path")
        assert result == "test_pkg.Component"

    def test_invalid_component_path(self):
        """Test import with invalid component path."""
        registry = ComponentRegistry()

        with pytest.raises(ValueError, match="Invalid component path"):
            registry._import_component("invalid_path_no_dots")