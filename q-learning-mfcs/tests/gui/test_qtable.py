#!/usr/bin/env python3
"""Test Q-table visualization."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

mock_st = MagicMock()
mock_go = MagicMock()
sys.modules["streamlit"] = mock_st
sys.modules["plotly.graph_objects"] = mock_go


class TestQTableViz(unittest.TestCase):
    """Test Q-table visualization."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.qtable_visualization
        self.assertIsNotNone(gui.qtable_visualization)

    @patch("numpy.load")
    def test_render_qtable(self, mock_load):
        """Test render Q-table visualization."""
        mock_load.return_value = {
            "q_table": np.array([[0, 1], [1, 0]]),
            "states": ["s0", "s1"],
            "actions": ["a0", "a1"],
        }

        from gui.qtable_visualization import render_qtable_analysis_interface

        # Should not raise - function execution is the test
        try:
            render_qtable_analysis_interface()
        except Exception as e:
            # Log errors from missing directories/files are acceptable
            if "not found" not in str(e).lower():
                raise


if __name__ == "__main__":
    unittest.main()
