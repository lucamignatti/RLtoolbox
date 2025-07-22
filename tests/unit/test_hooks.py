import pytest
from unittest.mock import MagicMock

from rltoolbox.core.hooks import HookExecutor
from rltoolbox.core.base import RLComponent


class MockComponent(RLComponent):
    def __init__(self, config):
        super().__init__(config)
        self.call_log = []

    def training_start(self, context):
        self.call_log.append("training_start")

    def episode_reset(self, context):
        self.call_log.append("episode_reset")
        context["state"] = [0, 0, 0, 0]


def test_hook_execution():
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    executor.execute_hook("training_start", ["mock"], components, context)
    assert component.call_log == ["training_start"]


def test_hook_validation():
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    executor.execute_hook("episode_reset", ["mock"], components, context)
    assert context["state"] == [0, 0, 0, 0]


def test_hook_validation_warning():
    executor = HookExecutor()
    component = MockComponent({})
    components = {"mock": component}
    context = {}

    # The base RLComponent's action_selection is a no-op.
    # The HookExecutor should warn that the expected context key 'action' was not added.
    with pytest.warns(UserWarning, match="did not add expected context keys:.*action"):
        executor.execute_hook("action_selection", ["mock"], components, context)