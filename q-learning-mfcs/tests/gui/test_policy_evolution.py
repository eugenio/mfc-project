#!/usr/bin/env python3
"""Test policy evolution visualization."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

# Mock dependencies before importing the module under test.
# Save originals and restore IMMEDIATELY after import so that other test files
# collected in the same session are not polluted.
mock_st = MagicMock()
mock_st.slider = MagicMock(return_value=0)
mock_st.button = MagicMock(return_value=False)
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.plotly_chart = MagicMock()
mock_st.pyplot = MagicMock()
_orig_st = sys.modules.get("streamlit")
_orig_plotly_go = sys.modules.get("plotly.graph_objects")
_orig_plt = sys.modules.get("matplotlib.pyplot")
sys.modules["streamlit"] = mock_st
sys.modules["plotly.graph_objects"] = MagicMock()
sys.modules["matplotlib.pyplot"] = MagicMock()

import gui.policy_evolution_viz  # noqa: E402

# Restore originals immediately — gui.policy_evolution_viz already cached mock refs
if _orig_st is not None:
    sys.modules["streamlit"] = _orig_st
else:
    sys.modules.pop("streamlit", None)
if _orig_plotly_go is not None:
    sys.modules["plotly.graph_objects"] = _orig_plotly_go
else:
    sys.modules.pop("plotly.graph_objects", None)
if _orig_plt is not None:
    sys.modules["matplotlib.pyplot"] = _orig_plt
else:
    sys.modules.pop("matplotlib.pyplot", None)


class TestPolicyEvolution(unittest.TestCase):
    """Test policy evolution visualization."""

    def test_module_import(self):
        """Test module can be imported."""
        self.assertIsNotNone(gui.policy_evolution_viz)

    @patch("os.path.exists")
    @patch("numpy.load")
    def test_render_policy_evolution(self, mock_load, mock_exists):
        """Test render policy evolution."""
        mock_exists.return_value = True
        mock_load.return_value = {"policies": [[[0, 1], [1, 0]]]}

        from gui.policy_evolution_viz import render_policy_evolution_interface

        # Should not raise - function execution is the test
        try:
            render_policy_evolution_interface()
        except Exception as e:
            # Log errors from missing directories/files are acceptable
            if "not found" not in str(e).lower():
                raise


if __name__ == "__main__":
    unittest.main()
