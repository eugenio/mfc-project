"""
GUI test suite for MFC Streamlit application.

This module contains comprehensive tests for the Streamlit GUI including:
- Browser-based interaction tests (Selenium)
- HTTP endpoint tests
- Memory-based data loading tests
- Autorefresh functionality tests
- Error handling and race condition tests

Run the test suite with:
    python -m tests.gui.test_gui_simple      # Quick HTTP-based tests
    python -m tests.gui.test_gui_browser     # Full browser-based tests
"""

__version__ = "1.0.0"
__author__ = "MFC Q-Learning Project"

# Import main test classes for convenience
try:
    from .test_gui_browser import StreamlitGUITester
    from .test_gui_simple import SimpleGUITester

    __all__ = ['SimpleGUITester', 'StreamlitGUITester']
except ImportError:
    # Handle missing dependencies gracefully
    __all__ = []
