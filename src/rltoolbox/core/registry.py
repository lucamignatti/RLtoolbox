"""
Component registry for managing component types and packages.

Handles registration, loading, and validation of components from different packages.
"""

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Type, Optional, List
import subprocess
import pkg_resources

from .base import RLComponent


class ComponentRegistry:
    """
    Registry for managing RL components from different packages.

    Handles component discovery, package validation, and instantiation.
    """

    def __init__(self):
        self._components: Dict[str, Type[RLComponent]] = {}
        self._packages: Dict[str, Dict[str, Any]] = {}
        self._development_mode: bool = False

        # Auto-register rltoolbox package
        self._register_rltoolbox_package()

    def set_development_mode(self, enabled: bool) -> None:
        """Enable or disable development mode (skips version validation)."""
        self._development_mode = enabled
        if enabled:
            warnings.warn("Development mode enabled - version validation disabled")

    def register_packages(self, packages_config: Dict[str, Dict[str, Any]]) -> None:
        """
        Register packages from configuration.

        Args:
            packages_config: Dictionary mapping package names to their config
        """
        for package_name, package_config in packages_config.items():
            self._register_package(package_name, package_config)

    def _register_rltoolbox_package(self) -> None:
        """Register the rltoolbox package itself."""
        self._packages["rltoolbox"] = {
            "version": "internal",
            "components": []
        }

    def _register_package(self, package_name: str, package_config: Dict[str, Any]) -> None:
        """
        Register a single package.

        Args:
            package_name: Name of the package
            package_config: Package configuration
        """
        self._packages[package_name] = package_config

        # Validate package availability and version
        if not self._development_mode:
            self._validate_package(package_name, package_config)

        # Load components from the package
        self._load_package_components(package_name, package_config)

    def _validate_package(self, package_name: str, package_config: Dict[str, Any]) -> None:
        """
        Validate that package exists and has correct version.

        Args:
            package_name: Name of the package
            package_config: Package configuration
        """
        if "path" in package_config:
            # Local package - validate path exists
            path = Path(package_config["path"])
            if not path.exists():
                raise ValueError(f"Local package path does not exist: {path}")

            # If git_commit specified, validate it
            if "git_commit" in package_config:
                try:
                    result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=path,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    current_commit = result.stdout.strip()
                    expected_commit = package_config["git_commit"]
                    if not current_commit.startswith(expected_commit):
                        warnings.warn(
                            f"Git commit mismatch for {package_name}: "
                            f"expected {expected_commit}, got {current_commit}"
                        )
                except subprocess.CalledProcessError:
                    warnings.warn(f"Could not validate git commit for {package_name}")

        elif "version" in package_config:
            # Installed package - validate version
            required_version = package_config["version"]
            if required_version == "dev":
                return  # Skip version check for dev version

            try:
                installed_version = pkg_resources.get_distribution(package_name).version
                if installed_version != required_version:
                    raise ValueError(
                        f"Version mismatch for {package_name}: "
                        f"required {required_version}, installed {installed_version}"
                    )
            except pkg_resources.DistributionNotFound:
                raise ValueError(f"Package {package_name} not found")

        else:
            raise ValueError(f"Package {package_name} must specify either 'version' or 'path'")

    def _load_package_components(self, package_name: str, package_config: Dict[str, Any]) -> None:
        """
        Load components from a package.

        Args:
            package_name: Name of the package
            package_config: Package configuration
        """
        components = package_config.get("components", [])

        for component_name in components:
            # Determine full component path
            if "path" in package_config:
                # Local package
                component_path = self._load_local_component(
                    package_name, package_config["path"], component_name
                )
            else:
                # Installed package
                component_path = f"{package_name}.{component_name}"

            # Import and register the component
            component_class = self._import_component(component_path)
            full_name = f"{package_name}.{component_name}"
            self._components[full_name] = component_class

    def _load_local_component(self, package_name: str, package_path: str, component_name: str) -> str:
        """
        Load component from local path.

        Args:
            package_name: Name of the package
            package_path: Path to the package
            component_name: Name of the component

        Returns:
            Full path to the component
        """
        package_path = Path(package_path).resolve()

        # Add package path to Python path if not already there
        if str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))

        return f"{package_name}.{component_name}"

    def _import_component(self, component_path: str) -> Type[RLComponent]:
        """
        Import a component class from its path.

        Args:
            component_path: Full path to the component

        Returns:
            Component class
        """
        try:
            # Split module and class name
            if "." in component_path:
                module_path, class_name = component_path.rsplit(".", 1)
            else:
                raise ValueError(f"Invalid component path: {component_path}")

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            if not hasattr(module, class_name):
                raise ValueError(f"Component {class_name} not found in {module_path}")

            component_class = getattr(module, class_name)

            # Validate it's an RLComponent
            if not issubclass(component_class, RLComponent):
                raise ValueError(f"Component {component_path} must inherit from RLComponent")

            return component_class

        except ImportError as e:
            raise ValueError(f"Could not import component {component_path}: {e}")

    def get_component_class(self, package_name: str, component_type: str) -> Type[RLComponent]:
        """
        Get a component class by package and type.

        Args:
            package_name: Name of the package
            component_type: Type of the component

        Returns:
            Component class
        """
        # Check if component exists in this package
        if package_name not in self._packages:
            raise ValueError(f"Package {package_name} not registered")

        # For rltoolbox package, try to import directly from the components module
        if package_name == "rltoolbox":
            return self._get_rltoolbox_component(component_type)

        full_name = f"{package_name}.{component_type}"
        if full_name not in self._components:
            # For rltoolbox components, they should already be registered
            pass

        if full_name not in self._components:
            available = list(self._components.keys())
            raise ValueError(
                f"Component {full_name} not found. Available components: {available}"
            )

        return self._components[full_name]

    def _get_rltoolbox_component(self, component_type: str) -> Type[RLComponent]:
        """
        Get a component from the rltoolbox package.

        Args:
            component_type: Type of the component

        Returns:
            Component class
        """
        # Try to import from rltoolbox.components
        try:
            module = importlib.import_module("rltoolbox.components")
            if hasattr(module, component_type):
                component_class = getattr(module, component_type)
                if issubclass(component_class, RLComponent):
                    return component_class
        except ImportError:
            pass

        # If not found in components, check if it's already registered
        full_name = f"rltoolbox.{component_type}"
        if full_name in self._components:
            return self._components[full_name]

        # If still not found, raise error with helpful message
        available = [name.split(".", 1)[1] for name in self._components.keys() if name.startswith("rltoolbox.")]
        if available:
            raise ValueError(f"Component 'rltoolbox.{component_type}' not found. Available rltoolbox components: {available}")
        else:
            raise ValueError(f"Component 'rltoolbox.{component_type}' not found. No rltoolbox components are currently implemented.")

    def create_component(self, package_name: str, component_type: str, config: Dict[str, Any]) -> RLComponent:
        """
        Create an instance of a component.

        Args:
            package_name: Name of the package
            component_type: Type of the component
            config: Configuration for the component

        Returns:
            Component instance
        """
        component_class = self.get_component_class(package_name, component_type)

        # Create instance
        component = component_class(config)

        # Validate configuration
        if not component.validate_config():
            raise ValueError(f"Invalid configuration for {package_name}.{component_type}")

        return component

    def list_components(self) -> List[str]:
        """
        List all registered components.

        Returns:
            List of component names
        """
        return list(self._components.keys())




# Global registry instance
_registry = ComponentRegistry()

def get_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _registry
