"""
Advanced Visualization Tools for Multi-dimensional Analysis

This module provides comprehensive advanced visualization capabilities for MFC systems,
including multi-dimensional plotting, interactive analysis, and statistical visualization.

Classes:
- AdvancedVisualizer: Main visualization framework for complex analyses
- MultiDimensionalPlotter: Specialized plots for high-dimensional data
- InteractiveAnalyzer: Interactive visualization tools
- StatisticalVisualizer: Statistical plots and hypothesis testing visualization
- ComparisonAnalyzer: Model and experiment comparison tools

Features:
- 3D and 4D visualization with projections
- Interactive plots with zoom, pan, and selection
- Statistical hypothesis testing visualization
- Multi-model comparison plots
- Uncertainty visualization with confidence bands
- Real-time data streaming visualization
- Publication-quality figure generation
- Interactive parameter exploration tools

Literature References:
1. Tufte, E. R. (2001). "The Visual Display of Quantitative Information"
2. Cleveland, W. S. (1994). "The Elements of Graphing Data"
3. Wilkinson, L. (2005). "The Grammar of Graphics"
4. Hunter, J. D. (2007). "Matplotlib: A 2D Graphics Environment"
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

# Core visualization dependencies
try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Ellipse, Polygon, Rectangle
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("Matplotlib/Seaborn not available. Visualization features will be limited.")

# Advanced plotting dependencies
try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not available. Interactive visualization will be limited.")

# Statistical analysis
try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_ADVANCED_STATS = True
except ImportError:
    HAS_ADVANCED_STATS = False
    warnings.warn("Advanced statistical packages not available. Some features will be limited.")

# Import configuration classes
from .parameter_optimization import OptimizationResult
from .uncertainty_quantification import UncertaintyResult
from .visualization_config import VisualizationConfig


class PlotType(Enum):
    """Available plot types for advanced visualization."""
    SCATTER_3D = "scatter_3d"
    SURFACE_3D = "surface_3d"
    CONTOUR_FILLED = "contour_filled"
    HEATMAP = "heatmap"
    PARALLEL_COORDINATES = "parallel_coordinates"
    RADAR_CHART = "radar_chart"
    CORRELATION_MATRIX = "correlation_matrix"
    DISTRIBUTION_COMPARISON = "distribution_comparison"
    TIME_SERIES_ENSEMBLE = "time_series_ensemble"
    UNCERTAINTY_BANDS = "uncertainty_bands"
    PARETO_FRONTIER = "pareto_frontier"
    SENSITIVITY_TORNADO = "sensitivity_tornado"
    INTERACTIVE_SCATTER = "interactive_scatter"
    ANIMATED_TIME_SERIES = "animated_time_series"


class InteractionMode(Enum):
    """Interactive visualization modes."""
    STATIC = "static"
    ZOOM_PAN = "zoom_pan"
    SELECTION = "selection"
    HOVER_INFO = "hover_info"
    REAL_TIME = "real_time"
    PARAMETER_SWEEP = "parameter_sweep"


@dataclass
class PlotConfiguration:
    """Configuration for advanced plots."""
    plot_type: PlotType
    title: str
    x_label: str = ""
    y_label: str = ""
    z_label: str = ""
    color_label: str = ""
    size_label: str = ""

    # Styling options
    color_map: str = "viridis"
    alpha: float = 0.7
    marker_size: float = 50.0
    line_width: float = 2.0

    # Layout options
    figure_size: tuple[float, float] = (12.0, 8.0)
    subplot_layout: tuple[int, int] | None = None

    # Interaction options
    interaction_mode: InteractionMode = InteractionMode.STATIC
    enable_tooltips: bool = True
    enable_zoom: bool = True

    # Statistical options
    show_confidence_intervals: bool = False
    confidence_level: float = 0.95
    show_correlation_values: bool = True

    # Export options
    save_format: str = "png"
    dpi: int = 300
    transparent_background: bool = False


@dataclass
class VisualizationResult:
    """Results container for visualization operations."""

    # Plot information
    plot_type: PlotType
    figure_path: str | None = None
    interactive_plot: Any | None = None  # Plotly figure or matplotlib figure

    # Data used in visualization
    data_summary: dict[str, Any] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)

    # Metadata
    creation_time: datetime = field(default_factory=datetime.now)
    configuration: PlotConfiguration | None = None

    # Performance metrics
    rendering_time: float = 0.0
    data_points: int = 0


class AdvancedVisualizer(ABC):
    """Abstract base class for advanced visualization tools."""

    def __init__(self, config: VisualizationConfig):
        """
        Initialize advanced visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Set up matplotlib style
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'library') else 'seaborn')
            plt.rcParams.update({
                'figure.figsize': (config.plot_style.figure_width, config.plot_style.figure_height),
                'figure.dpi': config.plot_style.dpi,
                'lines.linewidth': config.plot_style.line_width,
                'font.size': config.plot_style.font_size,
                'axes.grid': config.plot_style.grid_enabled,
                'grid.alpha': config.plot_style.grid_alpha
            })

    @abstractmethod
    def create_plot(self, data: pd.DataFrame,
                   plot_config: PlotConfiguration,
                   **kwargs) -> VisualizationResult:
        """
        Create visualization plot.

        Args:
            data: Data to visualize
            plot_config: Plot configuration
            **kwargs: Additional plotting parameters

        Returns:
            Visualization result
        """
        pass

    def _prepare_data(self, data: pd.DataFrame,
                     required_columns: list[str]) -> pd.DataFrame:
        """Prepare and validate data for visualization."""
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Remove NaN values
        clean_data = data.dropna(subset=required_columns)

        if len(clean_data) == 0:
            raise ValueError("No valid data remaining after cleaning")

        return clean_data

    def _apply_color_scheme(self, n_colors: int = 10) -> list[str]:
        """Generate colors based on configuration."""
        if self.config.color_scheme_type == "scientific":
            return plt.cm.tab10(np.linspace(0, 1, n_colors)) if HAS_MATPLOTLIB else ['blue', 'red', 'green']
        elif self.config.color_scheme_type == "colorblind_friendly":
            return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:n_colors]
        else:
            return plt.cm.viridis(np.linspace(0, 1, n_colors)) if HAS_MATPLOTLIB else ['blue']


class MultiDimensionalPlotter(AdvancedVisualizer):
    """Specialized plotter for high-dimensional data visualization."""

    def create_plot(self, data: pd.DataFrame,
                   plot_config: PlotConfiguration,
                   **kwargs) -> VisualizationResult:
        """Create multi-dimensional visualization."""
        import time
        start_time = time.time()

        result = VisualizationResult(
            plot_type=plot_config.plot_type,
            configuration=plot_config,
            data_points=len(data)
        )

        if plot_config.plot_type == PlotType.SCATTER_3D:
            figure = self._create_3d_scatter(data, plot_config, **kwargs)
        elif plot_config.plot_type == PlotType.SURFACE_3D:
            figure = self._create_3d_surface(data, plot_config, **kwargs)
        elif plot_config.plot_type == PlotType.PARALLEL_COORDINATES:
            figure = self._create_parallel_coordinates(data, plot_config, **kwargs)
        elif plot_config.plot_type == PlotType.RADAR_CHART:
            figure = self._create_radar_chart(data, plot_config, **kwargs)
        else:
            raise ValueError(f"Unsupported plot type: {plot_config.plot_type}")

        result.interactive_plot = figure
        result.rendering_time = time.time() - start_time

        return result

    def _create_3d_scatter(self, data: pd.DataFrame,
                          config: PlotConfiguration,
                          x_col: str, y_col: str, z_col: str,
                          color_col: str | None = None,
                          size_col: str | None = None) -> Any:
        """Create 3D scatter plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for 3D plotting")

        clean_data = self._prepare_data(data, [x_col, y_col, z_col])

        fig = plt.figure(figsize=config.figure_size)
        ax = fig.add_subplot(111, projection='3d')

        # Prepare plot parameters
        x = clean_data[x_col]
        y = clean_data[y_col]
        z = clean_data[z_col]

        colors = clean_data[color_col] if color_col and color_col in clean_data.columns else 'blue'
        sizes = clean_data[size_col] if size_col and size_col in clean_data.columns else config.marker_size

        # Create scatter plot
        scatter = ax.scatter(x, y, z, c=colors, s=sizes, alpha=config.alpha, cmap=config.color_map)

        # Customize plot
        ax.set_xlabel(config.x_label or x_col)
        ax.set_ylabel(config.y_label or y_col)
        ax.set_zlabel(config.z_label or z_col)
        ax.set_title(config.title)

        # Add colorbar if color mapping is used
        if color_col and color_col in clean_data.columns:
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label(config.color_label or color_col)

        plt.tight_layout()
        return fig

    def _create_3d_surface(self, data: pd.DataFrame,
                          config: PlotConfiguration,
                          x_col: str, y_col: str, z_col: str,
                          grid_resolution: int = 50) -> Any:
        """Create 3D surface plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for 3D plotting")

        clean_data = self._prepare_data(data, [x_col, y_col, z_col])

        # Create interpolated grid
        x = clean_data[x_col]
        y = clean_data[y_col]
        z = clean_data[z_col]

        xi = np.linspace(x.min(), x.max(), grid_resolution)
        yi = np.linspace(y.min(), y.max(), grid_resolution)
        X, Y = np.meshgrid(xi, yi)

        # Interpolate Z values
        from scipy.interpolate import griddata
        Z = griddata((x, y), z, (X, Y), method='cubic')

        # Create surface plot
        fig = plt.figure(figsize=config.figure_size)
        ax = fig.add_subplot(111, projection='3d')

        surface = ax.plot_surface(X, Y, Z, cmap=config.color_map, alpha=config.alpha)

        ax.set_xlabel(config.x_label or x_col)
        ax.set_ylabel(config.y_label or y_col)
        ax.set_zlabel(config.z_label or z_col)
        ax.set_title(config.title)

        # Add colorbar
        cbar = plt.colorbar(surface, ax=ax, shrink=0.6)
        cbar.set_label(config.color_label or z_col)

        plt.tight_layout()
        return fig

    def _create_parallel_coordinates(self, data: pd.DataFrame,
                                   config: PlotConfiguration,
                                   columns: list[str],
                                   class_column: str | None = None) -> Any:
        """Create parallel coordinates plot."""
        clean_data = self._prepare_data(data, columns)

        if HAS_PLOTLY:
            # Use Plotly for interactive parallel coordinates
            dimensions = []
            for col in columns:
                dimensions.append(dict(
                    range=[clean_data[col].min(), clean_data[col].max()],
                    label=col,
                    values=clean_data[col]
                ))

            color = clean_data[class_column] if class_column and class_column in clean_data.columns else None

            fig = go.Figure(data=go.Parcoords(
                line=dict(color=color, colorscale=config.color_map) if color is not None else dict(color='blue'),
                dimensions=dimensions
            ))

            fig.update_layout(
                title=config.title,
                width=config.figure_size[0] * 100,
                height=config.figure_size[1] * 100
            )

            return fig

        else:
            # Fallback to matplotlib
            if not HAS_MATPLOTLIB:
                raise ImportError("Either Plotly or Matplotlib required for parallel coordinates")

            from pandas.plotting import parallel_coordinates

            fig, ax = plt.subplots(figsize=config.figure_size)

            # Add class column if not provided
            if class_column is None:
                clean_data['_class'] = 'all'
                class_column = '_class'

            parallel_coordinates(clean_data[columns + [class_column]],
                               class_column, ax=ax, alpha=config.alpha)

            ax.set_title(config.title)
            plt.xticks(rotation=45)
            plt.tight_layout()

            return fig

    def _create_radar_chart(self, data: pd.DataFrame,
                           config: PlotConfiguration,
                           columns: list[str],
                           group_column: str | None = None) -> Any:
        """Create radar/spider chart."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for radar charts")

        clean_data = self._prepare_data(data, columns)

        # Normalize data to 0-1 scale
        normalized_data = clean_data[columns].copy()
        for col in columns:
            min_val = normalized_data[col].min()
            max_val = normalized_data[col].max()
            if max_val > min_val:
                normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
            else:
                normalized_data[col] = 0.5

        # Set up radar chart
        fig, ax = plt.subplots(figsize=config.figure_size, subplot_kw=dict(projection='polar'))

        # Calculate angles for each axis
        angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot data
        if group_column and group_column in clean_data.columns:
            # Multiple groups
            groups = clean_data[group_column].unique()
            colors = self._apply_color_scheme(len(groups))

            for i, group in enumerate(groups):
                group_data = normalized_data[clean_data[group_column] == group]
                avg_values = group_data.mean().tolist()
                avg_values += avg_values[:1]

                ax.plot(angles, avg_values, 'o-', linewidth=config.line_width,
                       label=str(group), color=colors[i], alpha=config.alpha)
                ax.fill(angles, avg_values, alpha=0.25, color=colors[i])
        else:
            # Single group
            avg_values = normalized_data.mean().tolist()
            avg_values += avg_values[:1]

            ax.plot(angles, avg_values, 'o-', linewidth=config.line_width, alpha=config.alpha)
            ax.fill(angles, avg_values, alpha=0.25)

        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(columns)
        ax.set_ylim(0, 1)
        ax.set_title(config.title, pad=20)

        if group_column:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.tight_layout()
        return fig


class InteractiveAnalyzer(AdvancedVisualizer):
    """Interactive visualization tools with real-time capabilities."""

    def create_plot(self, data: pd.DataFrame,
                   plot_config: PlotConfiguration,
                   **kwargs) -> VisualizationResult:
        """Create interactive visualization."""
        if not HAS_PLOTLY:
            self.logger.warning("Plotly not available. Falling back to static plots.")
            return self._create_static_fallback(data, plot_config, **kwargs)

        import time
        start_time = time.time()

        result = VisualizationResult(
            plot_type=plot_config.plot_type,
            configuration=plot_config,
            data_points=len(data)
        )

        if plot_config.plot_type == PlotType.INTERACTIVE_SCATTER:
            figure = self._create_interactive_scatter(data, plot_config, **kwargs)
        elif plot_config.plot_type == PlotType.ANIMATED_TIME_SERIES:
            figure = self._create_animated_time_series(data, plot_config, **kwargs)
        else:
            raise ValueError(f"Unsupported interactive plot type: {plot_config.plot_type}")

        result.interactive_plot = figure
        result.rendering_time = time.time() - start_time

        return result

    def _create_interactive_scatter(self, data: pd.DataFrame,
                                  config: PlotConfiguration,
                                  x_col: str, y_col: str,
                                  color_col: str | None = None,
                                  size_col: str | None = None,
                                  hover_cols: list[str] | None = None) -> go.Figure:
        """Create interactive scatter plot with hover information."""
        clean_data = self._prepare_data(data, [x_col, y_col])

        # Prepare hover text
        hover_text = []
        hover_columns = hover_cols or [x_col, y_col]
        if color_col and color_col in clean_data.columns:
            hover_columns.append(color_col)
        if size_col and size_col in clean_data.columns:
            hover_columns.append(size_col)

        for _, row in clean_data.iterrows():
            text = "<br>".join([f"{col}: {row[col]:.3f}" if isinstance(row[col], (int, float))
                              else f"{col}: {row[col]}" for col in hover_columns])
            hover_text.append(text)

        # Create figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=clean_data[x_col],
            y=clean_data[y_col],
            mode='markers',
            marker=dict(
                size=clean_data[size_col] if size_col and size_col in clean_data.columns else config.marker_size,
                color=clean_data[color_col] if color_col and color_col in clean_data.columns else 'blue',
                colorscale=config.color_map,
                opacity=config.alpha,
                colorbar=dict(title=config.color_label or color_col) if color_col else None
            ),
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            name='Data Points'
        ))

        # Customize layout
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label or x_col,
            yaxis_title=config.y_label or y_col,
            width=config.figure_size[0] * 100,
            height=config.figure_size[1] * 100,
            hovermode='closest'
        )

        return fig

    def _create_animated_time_series(self, data: pd.DataFrame,
                                   config: PlotConfiguration,
                                   time_col: str,
                                   value_cols: list[str],
                                   group_col: str | None = None) -> go.Figure:
        """Create animated time series plot."""
        clean_data = self._prepare_data(data, [time_col] + value_cols)

        # Sort by time
        clean_data = clean_data.sort_values(time_col)

        # Create figure with animation
        fig = go.Figure()

        if group_col and group_col in clean_data.columns:
            # Multiple groups
            groups = clean_data[group_col].unique()
            colors = px.colors.qualitative.Set1[:len(groups)]

            for i, group in enumerate(groups):
                group_data = clean_data[clean_data[group_col] == group]

                for col in value_cols:
                    fig.add_trace(go.Scatter(
                        x=group_data[time_col],
                        y=group_data[col],
                        mode='lines+markers',
                        name=f"{group}_{col}",
                        line=dict(color=colors[i % len(colors)]),
                        opacity=config.alpha
                    ))
        else:
            # Single group
            colors = px.colors.qualitative.Set1[:len(value_cols)]

            for i, col in enumerate(value_cols):
                fig.add_trace(go.Scatter(
                    x=clean_data[time_col],
                    y=clean_data[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[i % len(colors)]),
                    opacity=config.alpha
                ))

        # Add animation controls
        fig.update_layout(
            title=config.title,
            xaxis_title=config.x_label or time_col,
            yaxis_title=config.y_label or "Value",
            width=config.figure_size[0] * 100,
            height=config.figure_size[1] * 100,
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate", args=[None]),
                    dict(label="Pause", method="animate", args=[[None],
                         {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
                ]
            )]
        )

        return fig

    def _create_static_fallback(self, data: pd.DataFrame,
                              config: PlotConfiguration,
                              **kwargs) -> VisualizationResult:
        """Create static fallback when interactive libraries not available."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Neither Plotly nor Matplotlib available for visualization")

        # Simple matplotlib fallback
        fig, ax = plt.subplots(figsize=config.figure_size)
        ax.text(0.5, 0.5, 'Interactive visualization not available\nFalling back to basic plot',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(config.title)

        return VisualizationResult(
            plot_type=config.plot_type,
            interactive_plot=fig,
            configuration=config
        )


class StatisticalVisualizer(AdvancedVisualizer):
    """Statistical visualization and hypothesis testing plots."""

    def create_plot(self, data: pd.DataFrame,
                   plot_config: PlotConfiguration,
                   **kwargs) -> VisualizationResult:
        """Create statistical visualization."""
        import time
        start_time = time.time()

        result = VisualizationResult(
            plot_type=plot_config.plot_type,
            configuration=plot_config,
            data_points=len(data)
        )

        if plot_config.plot_type == PlotType.CORRELATION_MATRIX:
            figure = self._create_correlation_matrix(data, plot_config, **kwargs)
        elif plot_config.plot_type == PlotType.DISTRIBUTION_COMPARISON:
            figure = self._create_distribution_comparison(data, plot_config, **kwargs)
        elif plot_config.plot_type == PlotType.UNCERTAINTY_BANDS:
            figure = self._create_uncertainty_bands(data, plot_config, **kwargs)
        else:
            raise ValueError(f"Unsupported statistical plot type: {plot_config.plot_type}")

        result.interactive_plot = figure
        result.rendering_time = time.time() - start_time

        return result

    def _create_correlation_matrix(self, data: pd.DataFrame,
                                 config: PlotConfiguration,
                                 columns: list[str] | None = None) -> Any:
        """Create correlation matrix heatmap."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for correlation matrix")

        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        plot_columns = columns if columns else numeric_cols

        clean_data = self._prepare_data(data, plot_columns)

        # Calculate correlation matrix
        corr_matrix = clean_data[plot_columns].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=config.figure_size)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) if config.show_correlation_values else None

        sns.heatmap(corr_matrix,
                             annot=config.show_correlation_values,
                             cmap=config.color_map,
                             center=0,
                             square=True,
                             mask=mask,
                             ax=ax,
                             cbar_kws={'label': 'Correlation Coefficient'})

        ax.set_title(config.title)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        return fig

    def _create_distribution_comparison(self, data: pd.DataFrame,
                                      config: PlotConfiguration,
                                      value_col: str,
                                      group_col: str) -> Any:
        """Create distribution comparison plot."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for distribution plots")

        clean_data = self._prepare_data(data, [value_col, group_col])

        fig, axes = plt.subplots(2, 2, figsize=config.figure_size)
        fig.suptitle(config.title)

        groups = clean_data[group_col].unique()
        colors = self._apply_color_scheme(len(groups))

        # Histogram comparison
        ax1 = axes[0, 0]
        for i, group in enumerate(groups):
            group_data = clean_data[clean_data[group_col] == group][value_col]
            ax1.hist(group_data, alpha=config.alpha, label=str(group),
                    color=colors[i], bins=20)
        ax1.set_title('Histogram Comparison')
        ax1.set_xlabel(value_col)
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Box plot comparison
        ax2 = axes[0, 1]
        box_data = [clean_data[clean_data[group_col] == group][value_col] for group in groups]
        bp = ax2.boxplot(box_data, labels=groups, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(config.alpha)
        ax2.set_title('Box Plot Comparison')
        ax2.set_ylabel(value_col)

        # Q-Q plots
        ax3 = axes[1, 0]
        for i, group in enumerate(groups[:2]):  # Limit to 2 groups for Q-Q plot
            group_data = clean_data[clean_data[group_col] == group][value_col]
            stats.probplot(group_data, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal)')

        # Violin plot
        ax4 = axes[1, 1]
        violin_data = [clean_data[clean_data[group_col] == group][value_col] for group in groups]
        parts = ax4.violinplot(violin_data, positions=range(len(groups)))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(config.alpha)
        ax4.set_xticks(range(len(groups)))
        ax4.set_xticklabels(groups)
        ax4.set_title('Violin Plot')
        ax4.set_ylabel(value_col)

        plt.tight_layout()
        return fig

    def _create_uncertainty_bands(self, data: pd.DataFrame,
                                config: PlotConfiguration,
                                x_col: str, y_col: str,
                                uncertainty_col: str) -> Any:
        """Create plot with uncertainty bands."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for uncertainty bands")

        clean_data = self._prepare_data(data, [x_col, y_col, uncertainty_col])
        clean_data = clean_data.sort_values(x_col)

        fig, ax = plt.subplots(figsize=config.figure_size)

        x = clean_data[x_col]
        y = clean_data[y_col]
        uncertainty = clean_data[uncertainty_col]

        # Main line
        ax.plot(x, y, color='blue', linewidth=config.line_width, label='Mean')

        # Confidence bands
        alpha_factor = stats.norm.ppf(1 - (1 - config.confidence_level) / 2)
        upper_bound = y + alpha_factor * uncertainty
        lower_bound = y - alpha_factor * uncertainty

        ax.fill_between(x, lower_bound, upper_bound,
                       alpha=config.alpha * 0.5, color='blue',
                       label=f'{config.confidence_level:.0%} Confidence Band')

        # ±1σ bands
        ax.fill_between(x, y - uncertainty, y + uncertainty,
                       alpha=config.alpha * 0.3, color='lightblue',
                       label='±1σ Band')

        ax.set_xlabel(config.x_label or x_col)
        ax.set_ylabel(config.y_label or y_col)
        ax.set_title(config.title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class ComparisonAnalyzer(AdvancedVisualizer):
    """Model and experiment comparison visualization tools."""

    def create_optimization_comparison(self,
                                     results: list[OptimizationResult],
                                     config: PlotConfiguration) -> VisualizationResult:
        """Compare multiple optimization results."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for comparison plots")

        fig, axes = plt.subplots(2, 2, figsize=config.figure_size)
        fig.suptitle("Optimization Results Comparison")

        # Convergence comparison
        ax1 = axes[0, 0]
        colors = self._apply_color_scheme(len(results))
        for i, result in enumerate(results):
            if result.convergence_history:
                ax1.plot(result.convergence_history,
                        color=colors[i], label=f'{result.method.value}',
                        linewidth=config.line_width)
        ax1.set_title('Convergence History')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Best objective values comparison
        ax2 = axes[0, 1]
        methods = [r.method.value for r in results]
        best_scores = [r.best_overall_score for r in results if r.best_overall_score is not None]

        ax2.bar(methods[:len(best_scores)], best_scores, color=colors[:len(best_scores)])
        ax2.set_title('Best Objective Values')
        ax2.set_ylabel('Objective Value')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

        # Evaluation efficiency
        ax3 = axes[1, 0]
        evaluations = [r.total_evaluations for r in results]
        opt_times = [r.get_optimization_time() for r in results]

        ax3.scatter(evaluations, opt_times, c=colors[:len(results)],
                            s=config.marker_size, alpha=config.alpha)
        for i, method in enumerate(methods):
            ax3.annotate(method, (evaluations[i], opt_times[i]),
                        xytext=(5, 5), textcoords='offset points')

        ax3.set_xlabel('Total Evaluations')
        ax3.set_ylabel('Optimization Time (s)')
        ax3.set_title('Efficiency Comparison')
        ax3.grid(True, alpha=0.3)

        # Parameter distribution (if available)
        ax4 = axes[1, 1]
        if results and results[0].best_parameters is not None:
            n_params = len(results[0].best_parameters)
            param_data = np.array([r.best_parameters for r in results if r.best_parameters is not None])

            if len(param_data) > 0:
                box_data = [param_data[:, j] for j in range(min(5, n_params))]  # Limit to 5 parameters
                ax4.boxplot(box_data, labels=[f'P{j+1}' for j in range(len(box_data))])
                ax4.set_title('Parameter Distribution')
                ax4.set_ylabel('Parameter Value')

        plt.tight_layout()

        return VisualizationResult(
            plot_type=PlotType.DISTRIBUTION_COMPARISON,
            interactive_plot=fig,
            configuration=config
        )

    def create_uncertainty_comparison(self,
                                    results: list[UncertaintyResult],
                                    config: PlotConfiguration) -> VisualizationResult:
        """Compare uncertainty quantification results."""
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib required for uncertainty comparison")

        fig, axes = plt.subplots(2, 2, figsize=config.figure_size)
        fig.suptitle("Uncertainty Quantification Comparison")

        # Method comparison
        methods = [r.method.value for r in results]
        colors = self._apply_color_scheme(len(results))

        # Output mean comparison
        ax1 = axes[0, 0]
        if results and results[0].output_mean:
            output_names = list(results[0].output_mean.keys())

            x = np.arange(len(output_names))
            width = 0.8 / len(results)

            for i, result in enumerate(results):
                means = [result.output_mean.get(name, 0) for name in output_names]
                ax1.bar(x + i * width, means, width, label=methods[i], color=colors[i])

            ax1.set_title('Output Means')
            ax1.set_xlabel('Outputs')
            ax1.set_ylabel('Mean Value')
            ax1.set_xticks(x + width * (len(results) - 1) / 2)
            ax1.set_xticklabels(output_names)
            ax1.legend()

        # Output standard deviation comparison
        ax2 = axes[0, 1]
        if results and results[0].output_std:
            for i, result in enumerate(results):
                stds = [result.output_std.get(name, 0) for name in output_names]
                ax2.bar(x + i * width, stds, width, label=methods[i], color=colors[i])

            ax2.set_title('Output Standard Deviations')
            ax2.set_xlabel('Outputs')
            ax2.set_ylabel('Standard Deviation')
            ax2.set_xticks(x + width * (len(results) - 1) / 2)
            ax2.set_xticklabels(output_names)

        # Computational efficiency
        ax3 = axes[1, 0]
        comp_times = [r.computation_time for r in results]
        n_samples = [r.n_samples for r in results]

        ax3.scatter(n_samples, comp_times, c=colors,
                            s=config.marker_size, alpha=config.alpha)
        for i, method in enumerate(methods):
            ax3.annotate(method, (n_samples[i], comp_times[i]),
                        xytext=(5, 5), textcoords='offset points')

        ax3.set_xlabel('Number of Samples')
        ax3.set_ylabel('Computation Time (s)')
        ax3.set_title('Computational Efficiency')
        ax3.grid(True, alpha=0.3)

        # Sample distribution comparison (first output)
        ax4 = axes[1, 1]
        if results and results[0].output_samples:
            first_output = list(results[0].output_samples.keys())[0]

            for i, result in enumerate(results):
                if first_output in result.output_samples:
                    samples = result.output_samples[first_output]
                    # Remove NaN values
                    clean_samples = samples[~np.isnan(samples)]
                    if len(clean_samples) > 0:
                        ax4.hist(clean_samples, alpha=config.alpha,
                               label=methods[i], color=colors[i], bins=30)

            ax4.set_title(f'Sample Distribution - {first_output}')
            ax4.set_xlabel('Value')
            ax4.set_ylabel('Frequency')
            ax4.legend()

        plt.tight_layout()

        return VisualizationResult(
            plot_type=PlotType.DISTRIBUTION_COMPARISON,
            interactive_plot=fig,
            configuration=config
        )


# Utility functions for advanced visualization
def create_pareto_frontier_plot(optimization_results: list[OptimizationResult],
                               objective_names: list[str],
                               config: PlotConfiguration) -> VisualizationResult:
    """
    Create Pareto frontier visualization for multi-objective optimization.

    Args:
        optimization_results: List of optimization results
        objective_names: Names of objectives to plot
        config: Plot configuration

    Returns:
        Visualization result with Pareto frontier plot
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for Pareto frontier plots")

    # Extract objective values
    all_objectives = []
    for result in optimization_results:
        if result.all_objectives:
            for obj_dict in result.all_objectives:
                if all(name in obj_dict for name in objective_names):
                    all_objectives.append([obj_dict[name] for name in objective_names])

    if not all_objectives:
        raise ValueError("No valid objective data found")

    objectives_array = np.array(all_objectives)

    # Calculate Pareto frontier
    from .parameter_optimization import calculate_pareto_frontier
    is_pareto = calculate_pareto_frontier(objectives_array)
    pareto_points = objectives_array[is_pareto]

    # Create plot
    fig, ax = plt.subplots(figsize=config.figure_size)

    if len(objective_names) == 2:
        # 2D Pareto frontier
        ax.scatter(objectives_array[:, 0], objectives_array[:, 1],
                  alpha=0.3, color='lightblue', label='All Points')
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1],
                  color='red', s=config.marker_size, label='Pareto Frontier')

        # Connect Pareto points
        sorted_pareto = pareto_points[np.argsort(pareto_points[:, 0])]
        ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1],
               'r--', alpha=0.7, linewidth=config.line_width)

        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])

    else:
        # 3D Pareto frontier (first 3 objectives)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(objectives_array[:, 0], objectives_array[:, 1], objectives_array[:, 2],
                  alpha=0.3, color='lightblue', label='All Points')
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2],
                  color='red', s=config.marker_size, label='Pareto Frontier')

        ax.set_xlabel(objective_names[0])
        ax.set_ylabel(objective_names[1])
        ax.set_zlabel(objective_names[2] if len(objective_names) > 2 else 'Objective 3')

    ax.set_title(config.title or 'Pareto Frontier')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return VisualizationResult(
        plot_type=PlotType.PARETO_FRONTIER,
        interactive_plot=fig,
        configuration=config,
        data_summary={'pareto_points_count': len(pareto_points),
                     'total_points': len(objectives_array)}
    )


def create_sensitivity_tornado_plot(sensitivity_indices: dict[str, float],
                                  config: PlotConfiguration) -> VisualizationResult:
    """
    Create tornado plot for sensitivity analysis.

    Args:
        sensitivity_indices: Dictionary of parameter names to sensitivity values
        config: Plot configuration

    Returns:
        Visualization result with tornado plot
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("Matplotlib required for tornado plots")

    # Sort parameters by sensitivity
    sorted_params = sorted(sensitivity_indices.items(), key=lambda x: abs(x[1]), reverse=True)

    params = [item[0] for item in sorted_params]
    values = [item[1] for item in sorted_params]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=config.figure_size)

    colors = ['red' if v < 0 else 'blue' for v in values]
    bars = ax.barh(params, values, color=colors, alpha=config.alpha)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values, strict=False)):
        ax.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}',
               ha='left' if value >= 0 else 'right', va='center')

    ax.set_xlabel('Sensitivity Index')
    ax.set_title(config.title or 'Parameter Sensitivity Analysis')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    return VisualizationResult(
        plot_type=PlotType.SENSITIVITY_TORNADO,
        interactive_plot=fig,
        configuration=config,
        data_summary={'most_sensitive_param': params[0] if params else None,
                     'n_parameters': len(params)}
    )
