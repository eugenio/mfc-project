"""
Enhanced GUI Package for MFC Research Platform

This package provides advanced user interface components designed specifically
for scientific researchers and practitioners working with MFC systems.

Modules:
- enhanced_components: Core enhanced UI components with scientific validation
- qlearning_viz: Advanced Q-learning visualization and analysis tools  
- enhanced_mfc_gui: Main enhanced application interface

Features:
- Scientific parameter validation with literature references
- Interactive Q-learning analysis and visualization
- Real-time monitoring with publication-ready exports
- Collaborative research tools and data sharing capabilities
- Advanced performance analysis and insights

Created: 2025-07-31
"""

from .enhanced_components import (
    ScientificParameterInput,
    InteractiveVisualization,
    ExportManager,
    initialize_enhanced_ui,
    render_enhanced_sidebar,
    ComponentTheme,
    UIThemeConfig
)

from .qlearning_viz import (
    QLearningVisualizer,
    QLearningVisualizationConfig,
    QLearningVisualizationType,
    create_demo_qlearning_data,
    load_qtable_from_file
)

__version__ = "1.0.0"
__author__ = "MFC Research Team"
__description__ = "Enhanced GUI components for scientific MFC research"

# Package metadata
__all__ = [
    # Enhanced components
    "ScientificParameterInput",
    "InteractiveVisualization",
    "ExportManager",
    "initialize_enhanced_ui",
    "render_enhanced_sidebar",
    "ComponentTheme",
    "UIThemeConfig",

    # Q-learning visualization
    "QLearningVisualizer",
    "QLearningVisualizationConfig",
    "QLearningVisualizationType",
    "create_demo_qlearning_data",
    "load_qtable_from_file"
]
