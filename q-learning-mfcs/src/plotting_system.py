#!/usr/bin/env python3
"""
Standardized plotting system for MFC simulations
Supports arbitrary data length with Latin character subplot labels
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import json
import gzip
from datetime import datetime
import string

# Import path configuration
try:
    from path_config import get_figure_path
except ImportError:
    def get_figure_path(filename): 
        return f"../data/figures/{filename}"


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


def plot_time_series(ax: plt.Axes, df: pd.DataFrame, time_col: str, y_cols: List[str], 
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


def plot_mfc_simulation_results(data_path: Union[str, Path], output_prefix: str = "mfc_results"):
    """
    Create standardized plots for MFC simulation results
    
    Args:
        data_path: Path to CSV file with simulation data
        output_prefix: Prefix for output filenames (can include directory path)
    
    Returns:
        timestamp: Timestamp string used for output files
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Try to load JSON metadata if available
    json_path = Path(data_path).with_suffix('.json')
    metadata = {}
    if json_path.exists():
        with open(json_path, 'r') as f:
            metadata = json.load(f)
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine simulation duration for adaptive plotting
    duration = df['time_hours'].max()
    
    # ===== FIGURE 1: Overview (2x3 layout) =====
    fig1, axes1, labeler1 = create_labeled_subplots(2, 3, figsize=(18, 10),
                                                   title=f'MFC Simulation Results - Overview ({duration:.0f}h)')
    
    # (a) Substrate concentrations over time
    ax = axes1[0]
    plot_time_series(ax, df, 'time_hours', 
                    ['reservoir_concentration', 'outlet_concentration'],
                    labels=['Reservoir', 'Outlet'],
                    colors=['blue', 'orange'])
    
    # Add target lines if they exist in metadata or use defaults
    target_conc = metadata.get('substrate_target_reservoir', 25.0)
    max_threshold = metadata.get('substrate_max_threshold', 30.0)
    add_horizontal_line(ax, target_conc, f'Target ({target_conc} mM)')
    add_horizontal_line(ax, max_threshold, f'Max threshold ({max_threshold} mM)', color='orange', linestyle=':')
    setup_axis(ax, 'Time (hours)', 'Substrate Concentration (mmol/L)', 'Substrate Concentrations')
    
    # (b) Substrate addition rate
    ax = axes1[1]
    plot_time_series(ax, df, 'time_hours', ['substrate_addition_rate'], 
                    colors=['green'], linewidths=[1])
    setup_axis(ax, 'Time (hours)', 'Addition Rate (mmol/h)', 'Substrate Addition Rate', legend=False)
    
    # (c) Substrate concentration sensors
    ax = axes1[2]
    plot_time_series(ax, df, 'time_hours', 
                    ['reservoir_concentration', 'outlet_concentration'], 
                    labels=['Reservoir Sensor', 'Outlet Sensor'],
                    colors=['blue', 'orange'], linewidths=[2])
    add_horizontal_line(ax, target_conc, f'Target ({target_conc} mM)', color='green')
    setup_axis(ax, 'Time (hours)', 'Concentration (mmol/L)', 'Substrate Concentration')
    
    # (d) Q-learning metrics (if available)
    ax = axes1[3]
    if 'q_value' in df.columns and 'epsilon' in df.columns:
        ax.plot(df['time_hours'], df['q_value'], label='Q-value', color='purple', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(df['time_hours'], df['epsilon'], label='Epsilon', color='orange', alpha=0.7)
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Q-value', color='purple')
        ax2.set_ylabel('Epsilon', color='orange')
        ax.set_title('Q-learning Metrics')
        ax.grid(True, alpha=0.3)
    else:
        # Alternative plot if Q-learning data not available
        plot_time_series(ax, df, 'time_hours', ['total_power'], 
                        colors=['darkred'], linewidths=[2])
        setup_axis(ax, 'Time (hours)', 'Power (W)', 'Power Dynamics', legend=False)
    
    # (e) Cumulative substrate addition
    ax = axes1[4]
    time_step = df['time_hours'].iloc[1] - df['time_hours'].iloc[0]
    cumulative_addition = df['substrate_addition_rate'].cumsum() * time_step
    plot_time_series(ax, df, 'time_hours', ['substrate_addition_rate'], 
                    labels=['Rate'], colors=['lightgreen'], linewidths=[1], linestyles=[':'])
    ax2 = ax.twinx()
    ax2.plot(df['time_hours'], cumulative_addition, color='darkgreen', linewidth=2, label='Cumulative')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Rate (mmol/h)', color='green')
    ax2.set_ylabel('Cumulative (mmol)', color='darkgreen')
    ax.set_title('Substrate Addition')
    ax.grid(True, alpha=0.3)
    
    # (f) Summary statistics
    ax = axes1[5]
    ax.axis('off')  # Turn off axis for text display
    
    # Create summary text
    summary_text = f"Final Values (t={duration:.0f}h):\n" + "="*30 + "\n"
    summary_text += f"Reservoir conc.: {df['reservoir_concentration'].iloc[-1]:.1f} mM\n"
    summary_text += f"Outlet conc.: {df['outlet_concentration'].iloc[-1]:.1f} mM\n"
    summary_text += f"Total substrate added: {cumulative_addition.iloc[-1]:.1f} mmol\n"
    summary_text += f"Final power output: {df['total_power'].iloc[-1]:.2f} W\n"
    
    # Add biofilm info if available
    if 'biofilm_thickness_cell_0' in df.columns:
        avg_biofilm = np.mean([df[f'biofilm_thickness_cell_{i}'].iloc[-1] for i in range(5)])
        summary_text += f"Avg biofilm thickness: {avg_biofilm:.2f}\n"
    
    summary_text += "\nControl Performance:\n" + "-"*20 + "\n"
    summary_text += f"Target concentration: {target_conc} mM\n"
    summary_text += f"Final deviation: {df['reservoir_concentration'].iloc[-1] - target_conc:.1f} mM\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig1, get_figure_path(f"{output_prefix}_overview_{timestamp}.png"))
    
    # ===== FIGURE 2: Cell-by-cell analysis (if cell data available) =====
    if 'power_cell_0' in df.columns:
        fig2, axes2, labeler2 = create_labeled_subplots(3, 2, figsize=(15, 12),
                                                       title=f'MFC Simulation Results - Cell Analysis ({duration:.0f}h)')
        
        # (a) Power output per cell
        ax = axes2[0]
        for i in range(5):
            if f'power_cell_{i}' in df.columns:
                plot_time_series(ax, df, 'time_hours', [f'power_cell_{i}'], 
                                labels=[f'Cell {i}'], linewidths=[1.5])
        setup_axis(ax, 'Time (hours)', 'Power (W)', 'Power Output per Cell')
        
        # (b) Biofilm thickness per cell
        ax = axes2[1]
        if 'biofilm_thickness_cell_0' in df.columns:
            for i in range(5):
                if f'biofilm_thickness_cell_{i}' in df.columns:
                    plot_time_series(ax, df, 'time_hours', [f'biofilm_thickness_cell_{i}'], 
                                    labels=[f'Cell {i}'], linewidths=[1.5])
            setup_axis(ax, 'Time (hours)', 'Biofilm Thickness (μm)', 'Biofilm Growth')
        
        # (c) Substrate concentration per cell
        ax = axes2[2]
        if 'substrate_conc_cell_0' in df.columns:
            for i in range(5):
                if f'substrate_conc_cell_{i}' in df.columns:
                    plot_time_series(ax, df, 'time_hours', [f'substrate_conc_cell_{i}'], 
                                    labels=[f'Cell {i}'], linewidths=[1])
            setup_axis(ax, 'Time (hours)', 'Concentration (mmol/L)', 'Cell Substrate Concentrations')
        
        # (d) Q-learning action distribution (if available)
        ax = axes2[3]
        if 'q_action' in df.columns:
            action_counts = df['q_action'].value_counts().sort_index()
            ax.bar(action_counts.index, action_counts.values, alpha=0.7, color='purple')
            ax.set_xlabel('Q-learning Action')
            ax.set_ylabel('Frequency')
            ax.set_title('Action Distribution')
            ax.grid(True, alpha=0.3)
        
        # (e) Substrate control error
        ax = axes2[4]
        control_error = df['reservoir_concentration'] - target_conc
        plot_time_series(ax, df, 'time_hours', ['reservoir_concentration'], 
                        labels=['Actual'], colors=['blue'])
        ax2 = ax.twinx()
        ax2.plot(df['time_hours'], control_error, color='red', alpha=0.7, label='Error')
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Concentration (mM)', color='blue')
        ax2.set_ylabel('Error from target (mM)', color='red')
        ax.set_title('Control Performance')
        ax.grid(True, alpha=0.3)
        
        # (f) Phase portrait: Concentration vs Addition Rate
        ax = axes2[5]
        scatter = ax.scatter(df['reservoir_concentration'], df['substrate_addition_rate'], 
                            c=df['time_hours'], cmap='viridis', alpha=0.5, s=10)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (hours)')
        ax.set_xlabel('Reservoir Concentration (mmol/L)')
        ax.set_ylabel('Addition Rate (mmol/h)')
        ax.set_title('Control Phase Portrait')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(fig2, get_figure_path(f"{output_prefix}_cells_{timestamp}.png"))
    
    # ===== FIGURE 3: Long-term dynamics (for simulations > 100h) =====
    if duration > 100:
        fig3, axes3, labeler3 = create_labeled_subplots(2, 1, figsize=(14, 8),
                                                       title=f'MFC Simulation - Long-term Dynamics ({duration:.0f}h)')
        
        # (a) Substrate concentration with control zones
        ax = axes3[0]
        # Create colored background zones
        ax.axhspan(0, target_conc, alpha=0.1, color='green', label='Target zone')
        ax.axhspan(target_conc, max_threshold, alpha=0.1, color='yellow', label='Warning zone')
        ax.axhspan(max_threshold, df['reservoir_concentration'].max(), alpha=0.1, color='red', label='Excess zone')
        
        plot_time_series(ax, df, 'time_hours', ['reservoir_concentration'], 
                        colors=['blue'], linewidths=[2])
        add_horizontal_line(ax, target_conc, f'Target ({target_conc} mM)', color='darkgreen')
        add_horizontal_line(ax, max_threshold, f'Max threshold ({max_threshold} mM)', color='darkorange')
        setup_axis(ax, 'Time (hours)', 'Concentration (mmol/L)', 'Substrate Control Zones')
        
        # (b) Moving average analysis
        ax = axes3[1]
        window_size = int(duration / 20)  # Adaptive window size
        window_size = max(10, min(window_size, 100))  # Constrain between 10-100 hours
        
        ma_conc = df['reservoir_concentration'].rolling(window=window_size, center=True).mean()
        ma_addition = df['substrate_addition_rate'].rolling(window=window_size, center=True).mean()
        
        ax.plot(df['time_hours'], ma_conc, label=f'{window_size}h MA Concentration', 
                color='blue', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(df['time_hours'], ma_addition, label=f'{window_size}h MA Addition Rate', 
                 color='green', linewidth=2)
        
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('MA Concentration (mmol/L)', color='blue')
        ax2.set_ylabel('MA Addition Rate (mmol/h)', color='green')
        ax.set_title(f'Moving Average Analysis ({window_size}h window)')
        ax.grid(True, alpha=0.3)
        
        # Add trend annotation
        drift_rate = (df['reservoir_concentration'].iloc[-1] - df['reservoir_concentration'].iloc[0])/duration
        trend_text = f"Average drift rate: {drift_rate:.3f} mM/h"
        add_text_annotation(ax, trend_text, x=0.5, y=0.95, ha='center', va='top')
        
        plt.tight_layout()
        save_figure(fig3, get_figure_path(f"{output_prefix}_dynamics_{timestamp}.png"))
    
    # Close all figures to prevent memory leaks
    plt.close('all')
    
    print(f"\nAll plots saved with timestamp: {timestamp}")
    return timestamp


# Convenience function for direct usage
def plot_latest_simulation(pattern: str = "mfc_recirculation_control_*.csv", 
                          data_dir: Union[str, Path] = "../data/simulation_data",
                          output_prefix: str = "mfc_results"):
    """
    Plot the most recent simulation matching the given pattern
    
    Args:
        pattern: Glob pattern for finding simulation files
        data_dir: Directory containing simulation data
        output_prefix: Prefix for output filenames
    
    Returns:
        timestamp: Timestamp used for output files
    """
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob(pattern))
    
    if not csv_files:
        print(f"No simulation files found matching pattern: {pattern}")
        return None
    
    # Get the most recent file
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
    print(f"Plotting data from: {latest_file}")
    
    return plot_mfc_simulation_results(latest_file, output_prefix)


def plot_gpu_simulation_results(data_dir: Union[str, Path], output_prefix: str = "gpu_simulation"):
    """
    Plot GPU-accelerated MFC simulation results with control failure analysis
    
    Args:
        data_dir: Directory containing GPU simulation data files
        output_prefix: Prefix for output filenames
    
    Returns:
        timestamp: Timestamp used for output files
    """
    data_dir = Path(data_dir)
    
    # Find compressed CSV and JSON files
    csv_files = list(data_dir.glob("*.csv.gz"))
    json_files = list(data_dir.glob("*.json"))
    
    if not csv_files:
        print(f"No compressed CSV files found in {data_dir}")
        return None
    
    csv_file = csv_files[0]
    json_file = json_files[0] if json_files else None
    
    print(f"Loading GPU simulation data from: {csv_file}")
    
    # Load compressed CSV data
    with gzip.open(csv_file, 'rt') as f:
        df = pd.read_csv(f)
    
    # Load metadata if available
    metadata = {}
    if json_file:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded metadata from: {json_file}")
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert time to days for better readability
    df['time_days'] = df['time_hours'] / 24.0
    duration_days = df['time_days'].max()
    
    # Extract performance metrics
    performance = metadata.get('performance_metrics', {})
    sim_info = metadata.get('simulation_info', {})
    
    # ===== FIGURE 1: GPU Simulation Overview =====
    fig1, axes1, labeler1 = create_labeled_subplots(2, 3, figsize=(18, 12),
                                                   title=f'GPU-Accelerated MFC Simulation - Control Analysis ({duration_days:.0f} days)')
    
    # (a) Substrate concentration control failure
    ax = axes1[0]
    plot_time_series(ax, df, 'time_days', 
                    ['reservoir_concentration', 'outlet_concentration'],
                    labels=['Reservoir', 'Outlet'],
                    colors=['blue', 'red'], linewidths=[2, 1.5])
    
    # Add target and tolerance zones
    target = 25.0
    ax.axhline(y=target, color='green', linestyle='--', linewidth=2, label='Target (25 mM)')
    ax.fill_between(df['time_days'], 23, 27, alpha=0.2, color='green', label='±2 mM tolerance')
    ax.fill_between(df['time_days'], 20, 30, alpha=0.1, color='yellow', label='±5 mM tolerance')
    
    # Annotate control failure
    final_conc = performance.get('final_reservoir_concentration', df['reservoir_concentration'].iloc[-1])
    ax.annotate(f'CONTROL FAILURE\nFinal: {final_conc:.1f} mM\n(4x target)', 
                xy=(duration_days*0.8, final_conc), 
                xytext=(duration_days*0.5, final_conc*0.7),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                fontsize=10, color='red', weight='bold')
    
    setup_axis(ax, 'Time (days)', 'Substrate Concentration (mM)', 'Substrate Control Performance')
    ax.set_ylim(0, min(df['reservoir_concentration'].max() * 1.1, 150))
    
    # (b) Power output failure
    ax = axes1[1]
    plot_time_series(ax, df, 'time_days', ['total_power'], 
                    colors=['purple'], linewidths=[2])
    mean_power = performance.get('mean_power', df['total_power'].mean()) * 1000  # Convert to mW
    ax.annotate(f'POWER FAILURE\nMean: {mean_power:.3f} mW', 
                xy=(duration_days*0.5, df['total_power'].mean()), 
                xytext=(duration_days*0.3, df['total_power'].max()*0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3),
                fontsize=10, color='red', weight='bold')
    setup_axis(ax, 'Time (days)', 'Power Output (W)', 'Power Generation', legend=False)
    
    # (c) Q-learning actions and control response
    ax = axes1[2]
    ax.plot(df['time_days'], df['q_action'], color='orange', linewidth=1.5, alpha=0.8, label='Q-Action')
    ax2 = ax.twinx()
    ax2.plot(df['time_days'], df['substrate_addition_rate'], color='cyan', linewidth=2, label='Addition Rate')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Q-Learning Action', color='orange')
    ax2.set_ylabel('Substrate Addition (mmol/h)', color='cyan')
    ax.set_title('Q-Learning Control Actions')
    ax.grid(True, alpha=0.3)
    
    # (d) Control error analysis
    ax = axes1[3]
    control_error = df['reservoir_concentration'] - target
    ax.plot(df['time_days'], control_error, 'red', linewidth=2)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Perfect Control')
    ax.fill_between(df['time_days'], -2, 2, alpha=0.2, color='green', label='±2 mM tolerance')
    ax.fill_between(df['time_days'], -5, 5, alpha=0.1, color='yellow', label='±5 mM tolerance')
    setup_axis(ax, 'Time (days)', 'Control Error (mM)', 'Error from Target (25 mM)')
    
    # (e) Q-learning exploration
    ax = axes1[4]
    ax.plot(df['time_days'], df['epsilon'], color='purple', linewidth=2, label='Epsilon')
    ax2 = ax.twinx()
    ax2.plot(df['time_days'], df['reward'], color='brown', linewidth=1, alpha=0.7, label='Reward')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Epsilon (exploration)', color='purple')
    ax2.set_ylabel('Reward', color='brown')
    ax.set_title('Q-Learning Exploration & Rewards')
    ax.grid(True, alpha=0.3)
    
    # (f) Performance summary
    ax = axes1[5]
    ax.axis('off')
    
    # Performance metrics from metadata
    control_eff_2mm = performance.get('control_effectiveness_2mM', 0) * 100
    control_eff_5mm = performance.get('control_effectiveness_5mM', 0) * 100
    runtime_hours = sim_info.get('total_runtime_hours', 0)
    backend = sim_info.get('acceleration_backend', 'Unknown')
    
    summary_text = f"🔥 GPU SIMULATION RESULTS\n{'='*35}\n"
    summary_text += f"Backend: {backend}\n"
    summary_text += f"Runtime: {runtime_hours:.2f} hours\n"
    summary_text += f"Speedup: {duration_days*24/runtime_hours:.0f}x\n\n"
    summary_text += f"❌ CRITICAL CONTROL ISSUES:\n{'-'*25}\n"
    summary_text += f"Final concentration: {final_conc:.1f} mM\n"
    summary_text += f"Target deviation: {final_conc - target:.1f} mM\n"
    summary_text += f"Control effectiveness (±2mM): {control_eff_2mm:.1f}%\n"
    summary_text += f"Control effectiveness (±5mM): {control_eff_5mm:.1f}%\n"
    summary_text += f"Mean power: {mean_power:.3f} mW\n\n"
    summary_text += "⚠️  REQUIRES IMMEDIATE ATTENTION:\n"
    summary_text += "• Q-learning parameters need revision\n"
    summary_text += "• Control algorithm failure\n"
    summary_text += "• System essentially non-functional"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    save_figure(fig1, get_figure_path(f"{output_prefix}_control_failure_analysis_{timestamp}.png"))
    
    # ===== FIGURE 2: Detailed Control Analysis =====
    fig2, axes2, labeler2 = create_labeled_subplots(2, 2, figsize=(16, 10),
                                                   title='GPU Simulation - Detailed Control System Analysis')
    
    # (a) Action distribution
    ax = axes2[0]
    action_counts = df['q_action'].value_counts().sort_index()
    ax.bar(action_counts.index, action_counts.values, alpha=0.7, color='orange')
    setup_axis(ax, 'Q-Learning Action', 'Frequency', 'Action Distribution', legend=False)
    
    # (b) Substrate mass balance
    ax = axes2[1]
    dt = 1.0  # 1 hour timestep
    cumulative_added = df['substrate_addition_rate'].cumsum() * dt
    ax.plot(df['time_days'], cumulative_added, 'blue', linewidth=2, label='Total Added')
    
    # Expected consumption from metadata
    if 'substrate_consumption' in metadata:
        daily_rate = metadata['substrate_consumption']['daily_rate_mmol']
        expected_total = daily_rate * df['time_days']
        ax.plot(df['time_days'], expected_total, 'green', linestyle='--', linewidth=2, label='Expected')
    
    setup_axis(ax, 'Time (days)', 'Cumulative Substrate (mmol)', 'Substrate Mass Balance')
    
    # (c) Control effectiveness over time
    ax = axes2[2]
    window_size = max(240, len(df)//50)  # Adaptive window
    effectiveness_2mm = []
    effectiveness_5mm = []
    window_centers = []
    
    for i in range(0, len(df) - window_size, window_size//4):
        window_data = df.iloc[i:i+window_size]
        error_abs = abs(window_data['reservoir_concentration'] - target)
        
        eff_2mm = (error_abs <= 2).mean()
        eff_5mm = (error_abs <= 5).mean()
        
        effectiveness_2mm.append(eff_2mm)
        effectiveness_5mm.append(eff_5mm)
        window_centers.append(window_data['time_days'].mean())
    
    ax.plot(window_centers, np.array(effectiveness_2mm)*100, 'b-', linewidth=2, 
             marker='o', label='±2 mM', markersize=4)
    ax.plot(window_centers, np.array(effectiveness_5mm)*100, 'g-', linewidth=2, 
             marker='s', label='±5 mM', markersize=4)
    setup_axis(ax, 'Time (days)', 'Control Effectiveness (%)', 'Control Effectiveness Over Time')
    ax.set_ylim(0, 100)
    
    # (d) Phase portrait: Concentration vs Addition Rate
    ax = axes2[3]
    scatter = ax.scatter(df['reservoir_concentration'], df['substrate_addition_rate'], 
                        c=df['time_days'], cmap='viridis', alpha=0.6, s=8)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (days)')
    setup_axis(ax, 'Reservoir Concentration (mM)', 'Addition Rate (mmol/h)', 
               'Control Phase Portrait', legend=False)
    
    plt.tight_layout()
    save_figure(fig2, get_figure_path(f"{output_prefix}_detailed_analysis_{timestamp}.png"))
    
    # Close figures
    plt.close('all')
    
    print(f"\n{'='*70}")
    print("🎯 GPU SIMULATION PLOTTING COMPLETE")
    print(f"{'='*70}")
    print(f"✅ Plots saved with timestamp: {timestamp}")
    print(f"📊 Data duration: {duration_days:.1f} days ({len(df)} data points)")
    print(f"🚀 GPU backend: {backend}")
    print(f"⏱️  Runtime: {runtime_hours:.2f} hours")
    print("❌ CRITICAL: Control system completely failed - needs parameter revision")
    print(f"{'='*70}")
    
    return timestamp


if __name__ == "__main__":
    # Example: Plot the latest 1000h simulation
    # timestamp = plot_latest_simulation("mfc_recirculation_control_25mM_1000h_*.csv", 
    #                                   output_prefix="mfc_1000h")
    
    # Plot GPU simulation results
    gpu_data_dir = "../data/simulation_data/gpu_1year_20250726_160123"
    timestamp = plot_gpu_simulation_results(gpu_data_dir, output_prefix="gpu_1year_analysis")