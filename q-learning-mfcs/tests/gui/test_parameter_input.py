#!/usr/bin/env python3
"""Test parameter input module."""

import unittest
from unittest.mock import MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.number_input = MagicMock(return_value=100.0)
mock_st.slider = MagicMock(return_value=0.5)
mock_st.selectbox = MagicMock(return_value="option1")
mock_st.checkbox = MagicMock(return_value=False)
mock_st.columns = MagicMock(side_effect=lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))])
sys.modules['streamlit'] = mock_st


class TestParameterInput(unittest.TestCase):
    """Test parameter input module."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.parameter_input
        self.assertIsNotNone(gui.parameter_input)

    def test_render_functions(self):
        """Test render functions exist."""
        import gui.parameter_input as pi
        
        # Check key function exists
        self.assertTrue(hasattr(pi, 'render_parameter_input_interface'))

    def test_parameter_input_interface(self):
        """Test parameter input interface rendering."""
        from gui.parameter_input import render_parameter_input_interface
        
        # Call function - should not raise
        render_parameter_input_interface()
        
        # Check that streamlit methods were called
        self.assertTrue(mock_st.method_calls)


if __name__ == '__main__':
    unittest.main()