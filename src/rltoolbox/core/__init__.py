"""
Core components of the RLtoolbox framework.

This module contains the fundamental building blocks of the framework:
- Base component class
- Component registry system
- Hook execution system
- Main trainer class
"""

from .base import RLComponent
from .registry import ComponentRegistry, get_registry
from .hooks import HookExecutor
from .trainer import RLTrainer

__all__ = [
    "RLComponent",
    "ComponentRegistry",
    "get_registry",
    "HookExecutor",
    "RLTrainer"
]
