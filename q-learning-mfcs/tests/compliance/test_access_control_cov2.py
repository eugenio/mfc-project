"""Additional tests for access_control.py - cover lines 374-376 (main() error path)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from unittest.mock import patch

import pytest

from compliance.access_control import main


@pytest.mark.coverage_extra
class TestMainErrorPath:
    """Cover the except branch in main() that returns False."""

    def test_main_returns_false_on_exception(self):
        """Force main() to hit the except block by breaking login."""
        with patch(
            "compliance.access_control.AccessControl.login",
            side_effect=RuntimeError("simulated failure"),
        ):
            result = main()
            assert result is False
