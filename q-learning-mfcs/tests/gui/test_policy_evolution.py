#!/usr/bin/env python3
"""Test policy evolution visualization."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

mock_st = MagicMock()
mock_st.slider = MagicMock(return_value=0)
mock_st.button = MagicMock(return_value=False)
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.plotly_chart = MagicMock()
mock_st.pyplot = MagicMock()
sys.modules['streamlit'] = mock_st
sys.modules['plotly.graph_objects'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()


class TestPolicyEvolution(unittest.TestCase):
    """Test policy evolution visualization."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.policy_evolution_viz
        self.assertIsNotNone(gui.policy_evolution_viz)

    @patch('os.path.exists')
    @patch('numpy.load')
    def test_render_policy_evolution(self, mock_load, mock_exists):
        """Test render policy evolution."""
        mock_exists.return_value = True
        mock_load.return_value = {'policies': [[[0, 1], [1, 0]]]}
        
        from gui.policy_evolution_viz import render_policy_evolution_dashboard
        
        # Should not raise
        render_policy_evolution_dashboard()
        
        # Check that streamlit methods were called
        self.assertTrue(mock_st.method_calls)


if __name__ == '__main__':
    unittest.main()