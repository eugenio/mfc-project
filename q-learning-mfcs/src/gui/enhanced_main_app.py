#!/usr/bin/env python3
"""Enhanced MFC Platform - Main Application with Multi-Page Navigation.

This is the new main application that implements the Enhanced Streamlit architecture
with phase-based navigation, real-time monitoring, and professional scientific interface.

Architecture Features:
- Phase-based navigation (5 phases + monitoring + config)
- Real-time performance monitoring
- Scientific parameter validation
- Literature-backed citations
- GPU acceleration monitoring
- Professional research-grade UI

Created: 2025-08-02
Version: 2.0 (Enhanced Architecture)
"""

import logging
import os
import sys

import streamlit as st

# Configure logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
logging.getLogger("jax").setLevel(logging.WARNING)

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing components for compatibility
from gui.navigation_controller import NavigationController


def main() -> None:
    """Main application entry point for Enhanced MFC Platform."""
    try:
        # Initialize and run the navigation controller
        nav_controller = NavigationController()
        nav_controller.run()

    except Exception as e:
        st.error(f"Application Error: {e!s}")
        st.exception(e)


if __name__ == "__main__":
    main()
