import pytest
import warnings
from unittest.mock import patch, MagicMock
import subprocess

from rltoolbox.core.registry import ComponentRegistry, get_registry
from rltoolbox.core.base import RLComponent


class DummyComponent(RLComponent):
    def __init__(self, config):
        super().__init__(config)
        self.test_value = config.get("test_value", "default")

    def validate_config(self):
        return True


class InvalidComponent:
    """Component that doesn't inherit from RLComponent."""
    pass


class InvalidConfigComponent(RLComponent):
    def validate_config(self):
        return False


def test_registry_initialization():
    """Test registry initialization and auto-registration of rltoolbox package."""
    registry = ComponentRegistry()

    # Should have rltoolbox package registered
    assert "rltoolbox" in registry._packages
    assert registry._packages["rltoolbox"]["version"] == "internal"


def test_development_mode():
    """Test development mode functionality."""
    registry = ComponentRegistry()

    # Test enabling development mode
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        registry.set_development_mode(True)

        assert registry._development_mode is True
        assert len(w) >= 1
        assert "Development mode enabled" in str(w[0].message)


def test_register_and_create_component():
    """Test basic package registration and component creation."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "components": ["DummyComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.DummyComponent = DummyComponent
            mock_import.return_value = mock_module

            registry.register_packages(packages_config)
            component = registry.create_component("dummy_package", "DummyComponent", {"test_value": "test"})

    assert isinstance(component, DummyComponent)
    assert component.test_value == "test"


def test_component_not_found():
    """Test error when package is not registered."""
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="Package non_existent_package not registered"):
        registry.create_component("non_existent_package", "DummyComponent", {})


def test_component_type_not_found():
    """Test error when component type is not found in package."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "components": ["DummyComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.DummyComponent = DummyComponent
            mock_import.return_value = mock_module

            registry.register_packages(packages_config)

            with pytest.raises(ValueError, match="Component dummy_package.NonExistentComponent not found"):
                registry.create_component("dummy_package", "NonExistentComponent", {})


def test_invalid_component_class():
    """Test error when component class doesn't inherit from RLComponent."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "components": ["InvalidComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.InvalidComponent = InvalidComponent
            mock_import.return_value = mock_module

            with pytest.raises(ValueError, match="must inherit from RLComponent"):
                registry.register_packages(packages_config)


def test_component_config_validation():
    """Test component configuration validation."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "components": ["InvalidConfigComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.InvalidConfigComponent = InvalidConfigComponent
            mock_import.return_value = mock_module

            registry.register_packages(packages_config)

            with pytest.raises(ValueError, match="Invalid configuration"):
                registry.create_component("dummy_package", "InvalidConfigComponent", {})


def test_package_path_validation():
    """Test package path validation."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/nonexistent/path",
            "components": ["DummyComponent"]
        }
    }

    # Should raise error for non-existent path when not in development mode
    with pytest.raises(ValueError, match="Local package path does not exist"):
        registry.register_packages(packages_config)


def test_package_version_validation():
    """Test package version validation."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "version": "1.0.0"
        }
    }

    # Should raise error for non-existent package
    with pytest.raises(ValueError, match="Package dummy_package not found"):
        registry.register_packages(packages_config)


def test_git_commit_validation():
    """Test git commit validation for local packages."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "git_commit": "abc123",
            "components": ["DummyComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("subprocess.run") as mock_subprocess:
            with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
                mock_module = MagicMock()
                mock_module.DummyComponent = DummyComponent
                mock_import.return_value = mock_module

                mock_subprocess.return_value.stdout = "different_commit\n"
                mock_subprocess.return_value.check_return_value = None

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    registry.register_packages(packages_config)

                    # Should warn about commit mismatch
                    assert len(w) >= 1
                    assert any("Git commit mismatch" in str(warning.message) for warning in w)


def test_development_mode_skips_validation():
    """Test that development mode skips package validation."""
    registry = ComponentRegistry()
    registry.set_development_mode(True)

    packages_config = {
        "dummy_package": {
            "path": "/nonexistent/path",
            "components": ["DummyComponent"]
        }
    }

    with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_module.DummyComponent = DummyComponent
        mock_import.return_value = mock_module

        # Should not raise error in development mode
        registry.register_packages(packages_config)


def test_rltoolbox_component_handling():
    """Test handling of rltoolbox package components."""
    registry = ComponentRegistry()

    # Test that rltoolbox components are handled specially
    with pytest.raises(ValueError, match="No rltoolbox components are currently implemented"):
        registry.get_component_class("rltoolbox", "NonExistentComponent")


def test_list_components():
    """Test listing registered components."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "components": ["DummyComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.DummyComponent = DummyComponent
            mock_import.return_value = mock_module

            registry.register_packages(packages_config)
            components = registry.list_components()

    assert "dummy_package.DummyComponent" in components


def test_get_registry_singleton():
    """Test that get_registry returns the same instance."""
    registry1 = get_registry()
    registry2 = get_registry()

    assert registry1 is registry2


def test_import_error_handling():
    """Test handling of import errors."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "path": "/some/path",
            "components": ["DummyComponent"]
        }
    }

    with patch("pathlib.Path.exists", return_value=True):
        with patch("rltoolbox.core.registry.importlib.import_module", side_effect=ImportError("Module not found")):
            with pytest.raises(ValueError, match="Could not import component"):
                registry.register_packages(packages_config)


def test_installed_package_registration():
    """Test registration of installed packages."""
    registry = ComponentRegistry()
    packages_config = {
        "dummy_package": {
            "version": "dev"  # Special version that skips version check
        }
    }

    # Should not raise error for dev version
    registry.register_packages(packages_config)
