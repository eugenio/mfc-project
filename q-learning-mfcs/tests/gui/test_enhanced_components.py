#!/usr/bin/env python3
"""Test enhanced components module."""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

# Mock streamlit
mock_st = MagicMock()
mock_st.columns = MagicMock(return_value=[MagicMock() for _ in range(3)])
mock_st.metric = MagicMock()
mock_st.progress = MagicMock()
mock_st.expander = MagicMock()
mock_st.plotly_chart = MagicMock()
sys.modules['streamlit'] = mock_st

# Mock plotly
sys.modules['plotly.graph_objects'] = MagicMock()


class TestEnhancedComponents(unittest.TestCase):
    """Test enhanced components module."""

    def test_module_import(self):
        """Test module can be imported."""
        import gui.enhanced_components
        self.assertIsNotNone(gui.enhanced_components)

    def test_component_theme(self):
        """Test ComponentTheme enum."""
        from gui.enhanced_components import ComponentTheme
        
        self.assertIsNotNone(ComponentTheme.SCIENTIFIC)
        self.assertIsNotNone(ComponentTheme.LIGHT)
        self.assertIsNotNone(ComponentTheme.DARK)
        self.assertIsNotNone(ComponentTheme.HIGH_CONTRAST)

    def test_ui_theme_config(self):
        """Test UIThemeConfig class."""
        from gui.enhanced_components import UIThemeConfig
        
        config = UIThemeConfig(primary_color="#1f77b4")
        self.assertEqual(config.primary_color, "#1f77b4")

    def test_initialize_enhanced_ui(self):
        """Test initialize_enhanced_ui function."""
        from gui.enhanced_components import initialize_enhanced_ui, ComponentTheme
        
        config, session = initialize_enhanced_ui(ComponentTheme.SCIENTIFIC)
        self.assertIsNotNone(config)
        self.assertIsInstance(session, dict)


if __name__ == '__main__':
    unittest.main()