#!/usr/bin/env python3
"""Test Q-table visualization."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

mock_st = MagicMock()
mock_go = MagicMock()
sys.modules['streamlit'] = mock_st
sys.modules['plotly.graph_objects'] = mock_go


class TestQTableViz(unittest.TestCase):
    """Test Q-table visualization."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.qtable_visualization
        self.assertIsNotNone(gui.qtable_visualization)

    @patch('numpy.load')
    def test_render_qtable(self, mock_load):
        """Test render Q-table visualization."""
        mock_load.return_value = {
            'q_table': np.array([[0, 1], [1, 0]]),
            'states': ['s0', 's1'],
            'actions': ['a0', 'a1']
        }
        
        from gui.qtable_visualization import render_qtable_visualization
        
        # Should not raise
        render_qtable_visualization()
        
        # Check that streamlit methods were called
        self.assertTrue(mock_st.method_calls)


if __name__ == '__main__':
    unittest.main()