"""
RLtoolbox: A highly configurable reinforcement learning framework.

A hook-based RL framework that allows complete configuration of training loops
through JSON configuration files, enabling reproducible and flexible RL research.
"""

from .core.trainer import RLTrainer
from .core.base import RLComponent
from .core.registry import get_registry, ComponentRegistry
from .core.hooks import HookExecutor

__version__ = "0.1.0"
__author__ = "RLtoolbox Contributors"

__all__ = [
    "RLTrainer",
    "RLComponent",
    "get_registry",
    "ComponentRegistry",
    "HookExecutor"
]
