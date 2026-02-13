#!/usr/bin/env python3
"""Navigation Controller for Enhanced MFC Platform"""

import streamlit as st
from gui.core_layout import apply_enhanced_theme, create_navigation_sidebar
from gui.pages.advanced_physics import render_advanced_physics_page
from gui.pages.cell_config import render_cell_configuration_page
from gui.pages.dashboard import render_dashboard_page
from gui.pages.electrode_enhanced import render_enhanced_electrode_page
from gui.pages.gsm_integration import render_gsm_integration_page
from gui.pages.literature_validation import render_literature_validation_page
from gui.pages.ml_optimization import render_ml_optimization_page
from gui.pages.performance_monitor import render_performance_monitor_page
from gui.pages.system_configuration import render_system_configuration_page


class NavigationController:
    def __init__(self):
        self.pages = {
            "dashboard": render_dashboard_page,
            "electrode_system": render_enhanced_electrode_page,
            "cell_configuration": render_cell_configuration_page,
            "advanced_physics": render_advanced_physics_page,
            "ml_optimization": render_ml_optimization_page,
            "gsm_integration": render_gsm_integration_page,
            "literature_validation": render_literature_validation_page,
            "performance_monitor": render_performance_monitor_page,
            "system_configuration": render_system_configuration_page
        }

    def run(self):
        apply_enhanced_theme()
        st.set_page_config(
            page_title="MFC Platform",
            page_icon="🔬",
            layout="wide"
        )

        selected_page = create_navigation_sidebar()
        if selected_page in self.pages:
            self.pages[selected_page]()

    def _render_placeholder(self):
        st.title("Coming Soon")
        st.info("This page is under development!")
