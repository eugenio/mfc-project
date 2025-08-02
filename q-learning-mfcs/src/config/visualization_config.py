"""
Visualization Configuration Classes

This module provides comprehensive configuration classes for MFC visualization systems,
including plot styling, data processing, color schemes, and layout configurations.

Classes:
- PlotStyleConfig: Matplotlib plot styling parameters
- ColorSchemeConfig: Color scheme definitions
- DataProcessingConfig: Data processing and sampling parameters
- LayoutConfig: Plot layout and arrangement parameters
- VisualizationConfig: Complete visualization configuration

Literature References:
1. Tufte, E. R. (2001). "The Visual Display of Quantitative Information"
2. Wilkinson, L. (2005). "The Grammar of Graphics"
3. Hunter, J. D. (2007). "Matplotlib: A 2D graphics environment"
"""

from dataclasses import dataclass, field
from enum import Enum

import matplotlib.pyplot as plt


class ColorScheme(Enum):
    """Available color schemes for visualizations."""
    DEFAULT = "default"
    SCIENTIFIC = "scientific"
    COLORBLIND_FRIENDLY = "colorblind_friendly"
    HIGH_CONTRAST = "high_contrast"
    GRAYSCALE = "grayscale"
    PUBLICATION = "publication"


class PlotType(Enum):
    """Types of plots available."""
    TIME_SERIES = "time_series"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    HEATMAP = "heatmap"
    BAR = "bar"
    BOX_PLOT = "box_plot"
    VIOLIN_PLOT = "violin_plot"
    CONTOUR = "contour"


class LegendPosition(Enum):
    """Legend positioning options."""
    BEST = "best"
    UPPER_RIGHT = "upper right"
    UPPER_LEFT = "upper left"
    LOWER_LEFT = "lower left"
    LOWER_RIGHT = "lower right"
    RIGHT = "right"
    CENTER_LEFT = "center left"
    CENTER_RIGHT = "center right"
    LOWER_CENTER = "lower center"
    UPPER_CENTER = "upper center"
    CENTER = "center"


@dataclass
class PlotStyleConfig:
    """Matplotlib plot styling configuration."""

    # Figure parameters (Hunter, 2007)
    figure_width: float = 12.0  # Figure width (inches)
    figure_height: float = 8.0  # Figure height (inches)
    dpi: int = 300  # Dots per inch for high-quality output

    # Line styling
    line_width: float = 2.0  # Default line width
    marker_size: float = 6.0  # Default marker size
    alpha: float = 0.8  # Default transparency

    # Grid styling
    grid_enabled: bool = True  # Enable grid
    grid_alpha: float = 0.3  # Grid transparency
    grid_line_width: float = 0.5  # Grid line width
    grid_style: str = "-"  # Grid line style

    # Axis styling
    axis_label_size: float = 12.0  # Axis label font size
    tick_label_size: float = 10.0  # Tick label font size
    title_size: float = 14.0  # Title font size
    legend_font_size: float = 10.0  # Legend font size

    # Spacing and margins
    subplot_spacing: float = 0.3  # Spacing between subplots
    figure_padding: float = 0.1  # Padding around figure
    tight_layout: bool = True  # Use tight layout

    # Font configuration
    font_family: str = "DejaVu Sans"  # Font family
    font_weight: str = "normal"  # Font weight

    # Error bar styling
    error_bar_width: float = 1.0  # Error bar line width
    error_bar_cap_size: float = 3.0  # Error bar cap size

    # Annotation styling
    annotation_font_size: float = 9.0  # Annotation font size
    annotation_arrow_width: float = 1.0  # Annotation arrow width


@dataclass
class ColorSchemeConfig:
    """Color scheme configuration for different visualization contexts."""

    # Primary colors for main data series
    primary_colors: list[str] = field(default_factory=lambda: [
        "#2E86AB",  # Blue
        "#A23B72",  # Purple
        "#F18F01",  # Orange
        "#C73E1D",  # Red
        "#27AE60",  # Green
        "#8E44AD",  # Violet
        "#E67E22",  # Dark orange
        "#34495E"   # Dark gray
    ])

    # Scientific publication colors (publication ready)
    scientific_colors: list[str] = field(default_factory=lambda: [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f"   # Gray
    ])

    # Colorblind-friendly palette (Wong, 2011)
    colorblind_colors: list[str] = field(default_factory=lambda: [
        "#0173B2",  # Blue
        "#DE8F05",  # Orange
        "#029E73",  # Green
        "#CC78BC",  # Pink
        "#CA9161",  # Brown
        "#FBAFE4",  # Light pink
        "#949494",  # Gray
        "#ECE133"   # Yellow
    ])

    # High contrast colors for presentations
    high_contrast_colors: list[str] = field(default_factory=lambda: [
        "#000000",  # Black
        "#E69F00",  # Orange
        "#56B4E9",  # Sky blue
        "#009E73",  # Green
        "#F0E442",  # Yellow
        "#0072B2",  # Blue
        "#D55E00",  # Vermillion
        "#CC79A7"   # Reddish purple
    ])

    # Grayscale colors
    grayscale_colors: list[str] = field(default_factory=lambda: [
        "#000000",  # Black
        "#404040",  # Dark gray
        "#808080",  # Medium gray
        "#B0B0B0",  # Light gray
        "#D0D0D0",  # Very light gray
        "#FFFFFF"   # White
    ])

    # Background and accent colors
    background_color: str = "#FFFFFF"  # Plot background color
    grid_color: str = "#E0E0E0"  # Grid color
    text_color: str = "#000000"  # Text color
    accent_color: str = "#FF6B6B"  # Accent color for highlights

    # Status colors for different states
    success_color: str = "#28A745"  # Success/good status
    warning_color: str = "#FFC107"  # Warning status
    error_color: str = "#DC3545"  # Error/bad status
    info_color: str = "#17A2B8"  # Information status


@dataclass
class DataProcessingConfig:
    """Data processing configuration for visualizations."""

    # Sampling parameters
    time_series_sampling_rate: int = 100  # Sample every N points for time series
    scatter_sampling_rate: int = 1000  # Sample every N points for scatter plots

    # Smoothing parameters
    enable_smoothing: bool = True  # Enable data smoothing
    smoothing_window: int = 10  # Smoothing window size
    smoothing_method: str = "moving_average"  # Smoothing method

    # Outlier detection and handling
    enable_outlier_detection: bool = True  # Enable outlier detection
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection
    outlier_handling: str = "remove"  # How to handle outliers: "remove", "clip", "flag"

    # Data aggregation
    enable_data_aggregation: bool = False  # Enable data aggregation
    aggregation_window: float = 60.0  # Aggregation window (seconds)
    aggregation_method: str = "mean"  # Aggregation method: "mean", "median", "max", "min"

    # Missing data handling
    interpolate_missing: bool = True  # Interpolate missing data
    interpolation_method: str = "linear"  # Interpolation method
    max_gap_size: int = 10  # Maximum gap size to interpolate

    # Performance optimization
    max_points_per_plot: int = 10000  # Maximum points per plot for performance
    enable_downsampling: bool = True  # Enable intelligent downsampling

    # Statistical analysis
    compute_statistics: bool = True  # Compute and display statistics
    confidence_level: float = 0.95  # Confidence level for error bars
    enable_trend_lines: bool = False  # Enable trend line fitting


@dataclass
class LayoutConfig:
    """Plot layout and arrangement configuration."""

    # Subplot configuration
    n_rows: int = 2  # Number of subplot rows
    n_cols: int = 2  # Number of subplot columns
    subplot_width_ratios: list[float] | None = None  # Width ratios for subplots
    subplot_height_ratios: list[float] | None = None  # Height ratios for subplots

    # Spacing configuration
    horizontal_spacing: float = 0.3  # Horizontal spacing between subplots
    vertical_spacing: float = 0.3  # Vertical spacing between subplots
    left_margin: float = 0.1  # Left margin
    right_margin: float = 0.9  # Right margin
    top_margin: float = 0.9  # Top margin
    bottom_margin: float = 0.1  # Bottom margin

    # Legend configuration
    legend_position: LegendPosition = LegendPosition.BEST  # Legend position
    legend_columns: int = 1  # Number of legend columns
    legend_frame: bool = True  # Show legend frame
    legend_shadow: bool = False  # Show legend shadow

    # Title and labels
    main_title: str = ""  # Main figure title
    subplot_titles: list[str] = field(default_factory=list)  # Individual subplot titles

    # Axis configuration
    share_x_axis: bool = False  # Share x-axis across subplots
    share_y_axis: bool = False  # Share y-axis across subplots

    # Scale configuration
    x_scale: str = "linear"  # X-axis scale: "linear", "log", "symlog"
    y_scale: str = "linear"  # Y-axis scale: "linear", "log", "symlog"

    # Limits configuration
    auto_limits: bool = True  # Automatically determine axis limits
    x_limits: tuple[float, float] | None = None  # Manual x-axis limits
    y_limits: tuple[float, float] | None = None  # Manual y-axis limits


@dataclass
class AnimationConfig:
    """Animation configuration for dynamic visualizations."""

    # Animation parameters
    enable_animation: bool = False  # Enable animation
    animation_interval: float = 100.0  # Animation interval (ms)
    animation_frames: int = 100  # Number of animation frames

    # Animation style
    fade_effect: bool = True  # Enable fade effect for old data
    trail_length: int = 50  # Length of data trail

    # Performance
    blit_animation: bool = True  # Use blitting for performance
    cache_frames: bool = True  # Cache animation frames

    # Export options
    save_animation: bool = False  # Save animation to file
    animation_format: str = "mp4"  # Animation format: "mp4", "gif"
    animation_fps: int = 10  # Frames per second for saved animation


@dataclass
class VisualizationConfig:
    """Complete visualization system configuration."""

    # Component configurations
    plot_style: PlotStyleConfig = field(default_factory=PlotStyleConfig)
    color_scheme: ColorSchemeConfig = field(default_factory=ColorSchemeConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    animation: AnimationConfig = field(default_factory=AnimationConfig)

    # Global visualization parameters
    color_scheme_type: ColorScheme = ColorScheme.SCIENTIFIC  # Active color scheme
    output_format: str = "png"  # Output format: "png", "pdf", "svg", "eps"
    save_figures: bool = True  # Automatically save figures
    show_figures: bool = True  # Display figures interactively

    # File output configuration
    output_directory: str = "figures"  # Output directory for saved figures
    filename_prefix: str = "mfc"  # Filename prefix for saved figures
    include_timestamp: bool = True  # Include timestamp in filenames

    # Interactive features
    enable_zooming: bool = True  # Enable plot zooming
    enable_panning: bool = True  # Enable plot panning
    enable_tooltips: bool = False  # Enable interactive tooltips

    # Quality settings
    vector_graphics: bool = True  # Use vector graphics when possible
    high_dpi: bool = True  # Use high DPI for displays

    # Performance settings
    lazy_loading: bool = True  # Enable lazy loading of data
    memory_limit_mb: int = 1000  # Memory limit for visualization data (MB)

    # Configuration metadata
    configuration_name: str = "default_visualization"  # Configuration name
    configuration_version: str = "1.0.0"  # Configuration version
    description: str = "Default visualization configuration"  # Description


# Pre-defined visualization configurations

def get_publication_visualization_config() -> VisualizationConfig:
    """Get publication-ready visualization configuration."""
    config = VisualizationConfig()

    # Publication styling
    config.plot_style.dpi = 600  # High DPI for publication
    config.plot_style.font_family = "Times New Roman"
    config.plot_style.line_width = 1.5
    config.color_scheme_type = ColorScheme.SCIENTIFIC
    config.output_format = "pdf"
    config.vector_graphics = True

    # Clean layout
    config.layout.legend_frame = False
    config.plot_style.grid_alpha = 0.2

    config.configuration_name = "publication_visualization"
    config.description = "Publication-ready visualization settings"

    return config


def get_presentation_visualization_config() -> VisualizationConfig:
    """Get presentation visualization configuration."""
    config = VisualizationConfig()

    # Large fonts for presentations
    config.plot_style.axis_label_size = 16.0
    config.plot_style.tick_label_size = 14.0
    config.plot_style.title_size = 18.0
    config.plot_style.legend_font_size = 14.0

    # High contrast colors
    config.color_scheme_type = ColorScheme.HIGH_CONTRAST
    config.plot_style.line_width = 3.0
    config.plot_style.marker_size = 8.0

    # Larger figure
    config.plot_style.figure_width = 16.0
    config.plot_style.figure_height = 10.0

    config.configuration_name = "presentation_visualization"
    config.description = "Presentation visualization settings with high contrast"

    return config


def get_interactive_visualization_config() -> VisualizationConfig:
    """Get interactive visualization configuration."""
    config = VisualizationConfig()

    # Interactive features
    config.enable_zooming = True
    config.enable_panning = True
    config.enable_tooltips = True

    # Animation enabled
    config.animation.enable_animation = True
    config.animation.animation_interval = 50.0

    # Real-time processing
    config.data_processing.enable_downsampling = True
    config.data_processing.max_points_per_plot = 5000

    config.configuration_name = "interactive_visualization"
    config.description = "Interactive visualization with animation support"

    return config


def get_colorblind_friendly_config() -> VisualizationConfig:
    """Get colorblind-friendly visualization configuration."""
    config = VisualizationConfig()

    # Colorblind-friendly palette
    config.color_scheme_type = ColorScheme.COLORBLIND_FRIENDLY

    # Different line styles to distinguish series
    config.plot_style.line_width = 2.5

    # High contrast
    config.plot_style.alpha = 1.0
    config.plot_style.marker_size = 7.0

    config.configuration_name = "colorblind_friendly_visualization"
    config.description = "Colorblind-friendly visualization configuration"

    return config


# Utility functions for color management

def get_colors_for_scheme(scheme: ColorScheme, color_config: ColorSchemeConfig) -> list[str]:
    """
    Get color list for specified color scheme.

    Args:
        scheme: Color scheme type
        color_config: Color scheme configuration

    Returns:
        List of color strings
    """
    if scheme == ColorScheme.DEFAULT:
        return color_config.primary_colors
    elif scheme == ColorScheme.SCIENTIFIC:
        return color_config.scientific_colors
    elif scheme == ColorScheme.COLORBLIND_FRIENDLY:
        return color_config.colorblind_colors
    elif scheme == ColorScheme.HIGH_CONTRAST:
        return color_config.high_contrast_colors
    elif scheme == ColorScheme.GRAYSCALE:
        return color_config.grayscale_colors
    elif scheme == ColorScheme.PUBLICATION:
        return color_config.scientific_colors
    else:
        return color_config.primary_colors


def apply_style_config(style_config: PlotStyleConfig) -> None:
    """
    Apply plot style configuration to matplotlib.

    Args:
        style_config: Plot style configuration
    """
    plt.rcParams.update({
        'figure.figsize': (style_config.figure_width, style_config.figure_height),
        'figure.dpi': style_config.dpi,
        'lines.linewidth': style_config.line_width,
        'lines.markersize': style_config.marker_size,
        'font.size': style_config.tick_label_size,
        'axes.labelsize': style_config.axis_label_size,
        'axes.titlesize': style_config.title_size,
        'legend.fontsize': style_config.legend_font_size,
        'font.family': style_config.font_family,
        'font.weight': style_config.font_weight,
        'grid.alpha': style_config.grid_alpha,
        'grid.linewidth': style_config.grid_line_width,
        'grid.linestyle': style_config.grid_style,
    })


# Configuration validation functions

def validate_visualization_config(config: VisualizationConfig) -> bool:
    """
    Validate visualization configuration.

    Args:
        config: Visualization configuration to validate

    Returns:
        bool: True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate plot style parameters
    if config.plot_style.figure_width <= 0 or config.plot_style.figure_height <= 0:
        raise ValueError("Figure dimensions must be positive")

    if config.plot_style.dpi <= 0:
        raise ValueError("DPI must be positive")

    if not (0 <= config.plot_style.alpha <= 1):
        raise ValueError("Alpha must be between 0 and 1")

    # Validate layout parameters
    if config.layout.n_rows <= 0 or config.layout.n_cols <= 0:
        raise ValueError("Number of rows and columns must be positive")

    if not (0 <= config.layout.horizontal_spacing <= 1):
        raise ValueError("Horizontal spacing must be between 0 and 1")

    if not (0 <= config.layout.vertical_spacing <= 1):
        raise ValueError("Vertical spacing must be between 0 and 1")

    # Validate data processing parameters
    if config.data_processing.time_series_sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive")

    if config.data_processing.outlier_threshold <= 0:
        raise ValueError("Outlier threshold must be positive")

    if not (0 < config.data_processing.confidence_level < 1):
        raise ValueError("Confidence level must be between 0 and 1")

    # Validate animation parameters
    if config.animation.animation_interval <= 0:
        raise ValueError("Animation interval must be positive")

    if config.animation.animation_frames <= 0:
        raise ValueError("Number of animation frames must be positive")

    return True
