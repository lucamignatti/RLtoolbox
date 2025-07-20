"""
Test suite for RLtoolbox framework.

This module provides utilities and fixtures for testing the RLtoolbox framework.
"""

import sys
from pathlib import Path

# Add src to path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Import test utilities
from .conftest import *

__version__ = "0.1.0"
