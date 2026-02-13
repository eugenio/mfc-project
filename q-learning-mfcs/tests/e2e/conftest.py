"""Conftest for end-to-end tests."""

import os
import sys

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def pytest_collection_modifyitems(items):
    """Auto-mark all tests in this directory as e2e tests."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
