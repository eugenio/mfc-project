"""Plot modules for MFC Streamlit GUI.

This package provides visualization functions for MFC simulation data.

Modules:
- realtime_plots: Real-time monitoring plots and basic visualizations
- biofilm_plots: Biofilm analysis visualizations
- metabolic_plots: Metabolic pathway visualizations
- sensing_plots: EIS and QCM sensing visualizations
- spatial_plots: Spatial distribution visualizations
- performance_plots: Performance metrics and correlation visualizations
"""

from .biofilm_plots import create_biofilm_analysis_plots
from .metabolic_plots import create_metabolic_analysis_plots
from .performance_plots import (
    create_parameter_correlation_matrix,
    create_performance_analysis_plots,
)
from .realtime_plots import (
    create_performance_dashboard,
    create_real_time_plots,
)
from .sensing_plots import create_sensing_analysis_plots
from .spatial_plots import create_spatial_distribution_plots

__all__ = [
    "create_biofilm_analysis_plots",
    "create_metabolic_analysis_plots",
    "create_parameter_correlation_matrix",
    "create_performance_analysis_plots",
    "create_performance_dashboard",
    "create_real_time_plots",
    "create_sensing_analysis_plots",
    "create_spatial_distribution_plots",
]
