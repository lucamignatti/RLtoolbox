
import pytest
from unittest.mock import patch, MagicMock

from rltoolbox.core.registry import ComponentRegistry
from rltoolbox.core.base import RLComponent


class DummyComponent(RLComponent):
    pass


def test_register_and_create_component():
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
            component = registry.create_component("dummy_package", "DummyComponent", {})

    assert isinstance(component, DummyComponent)


def test_component_not_found():
    registry = ComponentRegistry()
    with pytest.raises(ValueError, match="Package non_existent_package not registered"):
        registry.create_component("non_existent_package", "DummyComponent", {})
