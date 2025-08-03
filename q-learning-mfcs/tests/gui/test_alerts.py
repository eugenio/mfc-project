#!/usr/bin/env python3
"""Test alert configuration UI."""

import unittest
from unittest.mock import MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.form = MagicMock()
mock_st.form_submit_button = MagicMock(return_value=False)
mock_st.selectbox = MagicMock(return_value="Voltage")
mock_st.number_input = MagicMock(return_value=0.5)
mock_st.checkbox = MagicMock(return_value=True)
mock_st.success = MagicMock()
mock_st.dataframe = MagicMock()
sys.modules['streamlit'] = mock_st


class TestAlertConfiguration(unittest.TestCase):
    """Test alert configuration UI."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.alert_configuration_ui
        self.assertIsNotNone(gui.alert_configuration_ui)

    def test_render_alert_configuration(self):
        """Test render alert configuration."""
        from gui.alert_configuration_ui import render_alert_configuration
        
        # Should not raise
        render_alert_configuration()
        
        # Check that streamlit methods were called
        self.assertTrue(mock_st.method_calls)

    def test_alert_functions(self):
        """Test alert functions exist."""
        import gui.alert_configuration_ui as acu
        
        self.assertTrue(hasattr(acu, 'create_alert_rule'))
        self.assertTrue(hasattr(acu, 'check_alerts'))


if __name__ == '__main__':
    unittest.main()