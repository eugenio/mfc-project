"""Additional tests for compliance/__init__.py - cover lines 42, 66."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import importlib
import types
from unittest.mock import MagicMock

import pytest


@pytest.mark.coverage_extra
class TestInitDataPrivacyImport:
    """Cover lines 42, 66: DataPrivacyManager import success path."""

    def test_init_with_data_privacy_manager_available(self):
        """Mock the data_privacy_manager module so the try/import succeeds,
        covering lines 42 (_has_data_privacy = True) and 66 (__all__.append)."""
        # Create a fake data_privacy_manager module
        fake_module = types.ModuleType("compliance.data_privacy_manager")
        fake_module.DataPrivacyManager = MagicMock()

        # Install it into sys.modules so 'from .data_privacy_manager import ...' works
        sys.modules["compliance.data_privacy_manager"] = fake_module

        try:
            # Force re-import of compliance.__init__
            if "compliance" in sys.modules:
                del sys.modules["compliance"]

            import compliance

            # Verify the success path was taken
            assert hasattr(compliance, "_has_data_privacy") or "DataPrivacyManager" in compliance.__all__
            assert "DataPrivacyManager" in compliance.__all__
        finally:
            # Clean up
            sys.modules.pop("compliance.data_privacy_manager", None)
            # Re-import to restore original state
            if "compliance" in sys.modules:
                del sys.modules["compliance"]
            import compliance  # noqa: F811

    def test_init_without_data_privacy_manager(self):
        """Verify the except ImportError path (lines 43-44) - this is the default."""
        # Remove the fake module if it exists
        sys.modules.pop("compliance.data_privacy_manager", None)

        if "compliance" in sys.modules:
            del sys.modules["compliance"]

        import compliance

        assert "DataPrivacyManager" not in compliance.__all__
