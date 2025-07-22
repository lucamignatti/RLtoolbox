
import pytest
from unittest.mock import MagicMock

from rltoolbox.core.trainer import RLTrainer


def test_trainer_initialization():
    config = {
        "components": {},
        "hooks": {},
    }
    trainer = RLTrainer(config_dict=config)
    assert trainer.config == config


from unittest.mock import patch, MagicMock

from rltoolbox.core.trainer import RLTrainer
from rltoolbox.core.base import RLComponent


class MockComponent(RLComponent):
    def training_start(self, context):
        context["training_started"] = True


def test_trainer_runs_hooks():
    config = {
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
