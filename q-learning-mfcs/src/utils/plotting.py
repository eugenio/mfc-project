#!/usr/bin/env python3
"""
Standardized plotting utilities for MFC simulations
Supports arbitrary data length with Latin character subplot labels
"""

import string
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt


class SubplotLabeler:
    """Helper class to generate Latin alphabet labels for subplots"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset to letter 'a' for new plot page"""
        self.current_index = 0

    def next_label(self) -> str:
        """Get next letter in sequence (a, b, c, ..., z, aa, ab, ...)"""
        if self.current_index < 26:
            label = string.ascii_lowercase[self.current_index]
        else:
            # For indices >= 26, generate aa, ab, ac, etc.
            first_letter_idx = (self.current_index - 26) // 26
            second_letter_idx = (self.current_index - 26) % 26
            if first_letter_idx < 26:
                label = string.ascii_lowercase[first_letter_idx] + string.ascii_lowercase[second_letter_idx]
            else:
                # Beyond 'zz', just use numbers
                label = f"subplot_{self.current_index}"

        self.current_index += 1
        return label


def create_labeled_subplots(nrows: int, ncols: int, figsize: Tuple[float, float] = (12, 8),
                           title: Optional[str] = None) -> Tuple[plt.Figure, List[plt.Axes], SubplotLabeler]:
    """
    Create figure with subplots and Latin character labels
    
    Args:
        nrows: Number of subplot rows
        ncols: Number of subplot columns
        figsize: Figure size (width, height)
        title: Optional figure title
    
    Returns:
        fig: Matplotlib figure
        axes: Flattened list of axes
        labeler: SubplotLabeler instance for this figure
    """
    fig, axes_array = plt.subplots(nrows, ncols, figsize=figsize)

    # Flatten axes array for easier iteration
    if nrows * ncols == 1:
        axes = [axes_array]
    else:
        axes = axes_array.flatten() if hasattr(axes_array, 'flatten') else [axes_array]

    # Add figure title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)

    # Create labeler for this figure
    labeler = SubplotLabeler()

    # Add labels to each subplot
    for ax in axes:
        label = labeler.next_label()
        add_subplot_label(ax, label)

    return fig, axes, labeler


def add_subplot_label(ax: plt.Axes, label: str, fontsize: int = 14, fontweight: str = 'bold'):
    """
    Add Latin character label to subplot in upper left corner, outside plot area
    
    Args:
        ax: Matplotlib axes object
        label: Label text (e.g., 'a', 'b', etc.)
        fontsize: Font size for label
        fontweight: Font weight for label
    """
    # Position label outside the plot area in upper left
    # Use transform coordinates: (-0.15, 1.05) places it outside the axes
    ax.text(-0.15, 1.05, f'({label})', transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            verticalalignment='bottom', horizontalalignment='right')


def setup_axis(ax: plt.Axes, xlabel: str, ylabel: str, title: str,
               grid: bool = True, legend: bool = True, legend_loc: str = 'best'):
    """
    Standard axis setup helper
    
    Args:
        ax: Matplotlib axes object
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Subplot title
        grid: Whether to show grid
        legend: Whether to show legend
        legend_loc: Legend location
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if grid:
        ax.grid(True, alpha=0.3)

    if legend and ax.get_legend_handles_labels()[0]:  # Only add legend if there are labeled lines
        ax.legend(loc=legend_loc)


def plot_time_series(ax: plt.Axes, df: Any, time_col: str, y_cols: List[str],
                    labels: Optional[List[str]] = None, colors: Optional[List[str]] = None,
                    linestyles: Optional[List[str]] = None, linewidths: Optional[List[float]] = None):
    """
    Plot multiple time series on the same axes
    
    Args:
        ax: Matplotlib axes object
        df: DataFrame with time series data
        time_col: Name of time column
        y_cols: List of column names to plot
        labels: Optional labels for each series
        colors: Optional colors for each series
        linestyles: Optional line styles for each series
        linewidths: Optional line widths for each series
    """
    if labels is None:
        labels = y_cols

    for i, (col, label) in enumerate(zip(y_cols, labels)):
        plot_kwargs = {'label': label}

        if colors and i < len(colors):
            plot_kwargs['color'] = colors[i]

        if linestyles and i < len(linestyles):
            plot_kwargs['linestyle'] = linestyles[i]

        if linewidths and i < len(linewidths):
            plot_kwargs['linewidth'] = linewidths[i]
        else:
            plot_kwargs['linewidth'] = 2

        ax.plot(df[time_col], df[col], **plot_kwargs)


def add_horizontal_line(ax: plt.Axes, y_value: float, label: str,
                       color: str = 'red', linestyle: str = '--', alpha: float = 0.5):
    """Add a horizontal reference line to axes"""
    ax.axhline(y=y_value, color=color, linestyle=linestyle, alpha=alpha, label=label)


def add_text_annotation(ax: plt.Axes, text: str, x: float = 0.95, y: float = 0.05,
                       ha: str = 'right', va: str = 'bottom', boxstyle: str = 'round',
                       facecolor: str = 'wheat', alpha: float = 0.5):
    """
    Add text annotation box to axes
    
    Args:
        ax: Matplotlib axes object
        text: Text to display
        x, y: Position in axes coordinates (0-1)
        ha: Horizontal alignment
        va: Vertical alignment
        boxstyle: Box style
        facecolor: Box face color
        alpha: Box transparency
    """
    ax.text(x, y, text, transform=ax.transAxes,
            verticalalignment=va, horizontalalignment=ha,
            bbox=dict(boxstyle=boxstyle, facecolor=facecolor, alpha=alpha))


def save_figure(fig: plt.Figure, filename: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """Save figure with standard settings"""
    fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Figure saved to: {filename}")


# Example usage function
def create_standard_mfc_plots(df: Any, output_prefix: str = "mfc_results"):
    """
    Example of creating standardized MFC plots
    
    Args:
        df: DataFrame with MFC simulation data
        output_prefix: Prefix for output filenames
    """
    # Create first figure with 2x2 subplots
    fig1, axes1, labeler1 = create_labeled_subplots(2, 2, figsize=(12, 10),
                                                    title="MFC Simulation Results - Overview")

    # Plot 1a: Substrate concentrations
    ax = axes1[0]
    plot_time_series(ax, df, 'time_hours',
                    ['reservoir_concentration', 'outlet_concentration'],
                    labels=['Reservoir', 'Outlet'])
    add_horizontal_line(ax, 25, 'Target (25 mM)')
    setup_axis(ax, 'Time (hours)', 'Concentration (mmol/L)', 'Substrate Concentrations')

    # Plot 1b: Power output
    ax = axes1[1]
    plot_time_series(ax, df, 'time_hours', ['total_power'])
    setup_axis(ax, 'Time (hours)', 'Power (W)', 'Total Power Output')

    # ... continue with other subplots

    plt.tight_layout()
    save_figure(fig1, f"{output_prefix}_page1.png")

    # Create second figure with different layout
    fig2, axes2, labeler2 = create_labeled_subplots(3, 1, figsize=(10, 12),
                                                    title="MFC Simulation Results - Details")
    # labeler2 automatically starts from 'a' again

    # ... add plots to second figure

    return fig1, fig2
