"""Spatial distribution plots for MFC Streamlit GUI.

This module contains visualization functions for per-cell spatial data.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_spatial_distribution_plots(df, n_cells=5):
    """Create spatial distribution visualization for per-cell parameters.

    Args:
        df: DataFrame or dict with per-cell data.
        n_cells: Number of cells to display.

    Returns:
        Plotly figure with spatial distribution or None if no cell data.

    """
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict("list") if hasattr(df, "to_dict") else df

    cell_data_cols = [
        col
        for col in columns
        if "per_cell" in col
        or "cell_" in col
        or "voltages" in col
        or "densities" in col
    ]
    if not cell_data_cols:
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Cell Voltages Distribution",
            "Current Density Distribution",
            "Temperature Distribution",
            "Biofilm Thickness Distribution",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
    )

    # Cell voltages
    _add_cell_voltages_plot(fig, data_dict, n_cells, row=1, col=1)

    # Current density
    _add_current_density_plot(fig, data_dict, n_cells, row=1, col=2)

    # Temperature distribution
    _add_temperature_plot(fig, data_dict, n_cells, row=2, col=1)

    # Biofilm thickness distribution
    _add_biofilm_distribution_plot(fig, data_dict, n_cells, row=2, col=2)

    fig.update_layout(
        title="Spatial Distribution Analysis",
        height=600,
        showlegend=True,
    )

    return fig


def _get_latest_value(data_list):
    """Get the latest value from a data list."""
    if isinstance(data_list, list) and data_list:
        if isinstance(data_list[-1], list):
            return data_list[-1]
        return data_list[-1]
    return None


def _add_cell_voltages_plot(fig, data_dict, n_cells, row, col):
    """Add cell voltages bar plot."""
    if "cell_voltages" not in data_dict:
        return

    cell_voltages_data = _get_latest_value(data_dict["cell_voltages"])
    if cell_voltages_data:
        if isinstance(cell_voltages_data, list):
            cell_voltages = (
                cell_voltages_data[:n_cells]
                if len(cell_voltages_data) >= n_cells
                else cell_voltages_data + [0.7] * (n_cells - len(cell_voltages_data))
            )
        else:
            cell_voltages = [cell_voltages_data] * n_cells
    else:
        cell_voltages = [0.7] * n_cells

    fig.add_trace(
        go.Bar(
            x=[f"Cell {i + 1}" for i in range(n_cells)],
            y=cell_voltages,
            name="Cell Voltage (V)",
            marker_color="blue",
        ),
        row=row,
        col=col,
    )


def _add_current_density_plot(fig, data_dict, n_cells, row, col):
    """Add current density bar plot."""
    if "current_densities" not in data_dict and "current_density_per_cell" not in data_dict:
        return

    current_key = (
        "current_densities"
        if "current_densities" in data_dict
        else "current_density_per_cell"
    )
    current_densities_data = _get_latest_value(data_dict[current_key])
    if current_densities_data:
        if isinstance(current_densities_data, list):
            current_densities = (
                current_densities_data[:n_cells]
                if len(current_densities_data) >= n_cells
                else current_densities_data + [1.0] * (n_cells - len(current_densities_data))
            )
        else:
            current_densities = [current_densities_data] * n_cells
    else:
        current_densities = [1.0] * n_cells

    fig.add_trace(
        go.Bar(
            x=[f"Cell {i + 1}" for i in range(n_cells)],
            y=current_densities,
            name="Current Density (A/m2)",
            marker_color="red",
        ),
        row=row,
        col=col,
    )


def _add_temperature_plot(fig, data_dict, n_cells, row, col):
    """Add temperature distribution plot."""
    if "temperature_per_cell" not in data_dict:
        return

    temp_data = _get_latest_value(data_dict["temperature_per_cell"])
    if temp_data:
        if isinstance(temp_data, list):
            temperatures = (
                temp_data[:n_cells]
                if len(temp_data) >= n_cells
                else temp_data + [25.0] * (n_cells - len(temp_data))
            )
        else:
            temperatures = [temp_data] * n_cells
    else:
        temperatures = [25.0] * n_cells

    fig.add_trace(
        go.Scatter(
            x=[f"Cell {i + 1}" for i in range(n_cells)],
            y=temperatures,
            mode="markers+lines",
            name="Temperature (C)",
            line={"color": "orange", "width": 2},
            marker={"size": 8},
        ),
        row=row,
        col=col,
    )


def _add_biofilm_distribution_plot(fig, data_dict, n_cells, row, col):
    """Add biofilm thickness distribution plot."""
    if "biofilm_thicknesses" not in data_dict and "biofilm_thickness_per_cell" not in data_dict:
        return

    biofilm_key = (
        "biofilm_thicknesses"
        if "biofilm_thicknesses" in data_dict
        else "biofilm_thickness_per_cell"
    )
    biofilm_data = _get_latest_value(data_dict[biofilm_key])
    if biofilm_data:
        if isinstance(biofilm_data, list):
            biofilm_thickness = (
                biofilm_data[:n_cells]
                if len(biofilm_data) >= n_cells
                else biofilm_data + [10.0] * (n_cells - len(biofilm_data))
            )
        else:
            biofilm_thickness = [biofilm_data] * n_cells
    else:
        biofilm_thickness = [10.0] * n_cells

    fig.add_trace(
        go.Scatter(
            x=[f"Cell {i + 1}" for i in range(n_cells)],
            y=biofilm_thickness,
            mode="markers+lines",
            name="Biofilm Thickness (um)",
            line={"color": "green", "width": 2},
            marker={"size": 8},
        ),
        row=row,
        col=col,
    )
