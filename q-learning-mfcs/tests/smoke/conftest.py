"""Conftest for smoke tests."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in this directory as smoke tests."""
    for item in items:
        if "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
