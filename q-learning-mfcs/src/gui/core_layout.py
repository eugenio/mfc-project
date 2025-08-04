#!/usr/bin/env python3
"""
Core Layout Framework for Enhanced MFC Platform

Provides standardized layout components and navigation system.
"""


import streamlit as st

# Phase Status Configuration
PHASE_STATUS = {
    "electrode_system": {"status": "complete", "progress": 100},
    "advanced_physics": {"status": "complete", "progress": 100},
    "ml_optimization": {"status": "ready", "progress": 90},
    "gsm_integration": {"status": "complete", "progress": 100},
    "literature_validation": {"status": "complete", "progress": 100}
}

def apply_enhanced_theme():
    """Apply enhanced scientific theme."""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def create_phase_header(phase_name: str, phase_description: str, phase_key: str):
    """Create standardized phase header with status indicators."""
    phase_info = PHASE_STATUS.get(phase_key, {"status": "pending", "progress": 0})

    col1, col2, col3 = st.columns([3, 1, 1])

    with col1:
        st.title(phase_name)
        st.caption(phase_description)

    with col2:
        status = phase_info["status"]
        progress = phase_info["progress"]

        if status == "complete":
            st.success(f"✅ Complete ({progress}%)")
        elif status == "ready":
            st.info(f"🔄 Ready ({progress}%)")
        else:
            st.warning(f"⏳ Pending ({progress}%)")

    with col3:
        st.metric("System Status", "Operational")

def create_navigation_sidebar() -> str:
    """Create enhanced navigation sidebar."""
    st.sidebar.title("🔬 MFC Platform")

    nav_options = {
        "🏠 Dashboard": "dashboard",
        "🔋 Electrode System": "electrode_system",
        "🏗️ Cell Configuration": "cell_configuration",
        "⚗️ Physics Simulation": "advanced_physics",
        "🧠 ML Optimization": "ml_optimization",
        "🧬 GSM Integration": "gsm_integration",
        "📚 Literature Validation": "literature_validation",
        "📊 Performance Monitor": "performance_monitor",
        "⚙️ Configuration": "system_configuration"
    }

    selected_page = st.sidebar.radio("Navigate to:", list(nav_options.keys()))

    # Show phase status
    st.sidebar.subheader("Phase Status")
    for phase_key, phase_info in PHASE_STATUS.items():
        status = phase_info["status"]
        progress = phase_info["progress"]
        emoji = "✅" if status == "complete" else "🔄" if status == "ready" else "⏳"
        st.sidebar.text(f"{emoji} {phase_key.replace('_', ' ').title()}: {progress}%")

    return nav_options[selected_page]

def create_page_layout(title: str):
    """Create standardized page layout."""
    apply_enhanced_theme()
    st.title(title)
    return {
        "header": st.container(),
        "content": st.container(),
        "sidebar": st.sidebar
    }

def render_header(title: str, subtitle: str = None):
    """Render standardized page header."""
    st.title(title)
    if subtitle:
        st.markdown(f"*{subtitle}*")
