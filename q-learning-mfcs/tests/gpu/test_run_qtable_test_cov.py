"""Tests for run_qtable_test.py - coverage target 98%+."""
import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


class TestRunQtableTest:
    def test_import_and_run(self):
        mock_func = MagicMock(return_value=({"test": True}, "/tmp/out"))
        with patch.dict(
            "sys.modules",
            {"mfc_gpu_accelerated": MagicMock(
                run_gpu_accelerated_simulation=mock_func
            )},
        ):
            # Re-import to pick up mock
            if "run_qtable_test" in sys.modules:
                del sys.modules["run_qtable_test"]

            with patch("builtins.print"):
                import run_qtable_test  # noqa: F401

            mock_func.assert_called_once_with(24)
