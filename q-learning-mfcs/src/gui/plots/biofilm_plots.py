"""Biofilm analysis plots for MFC Streamlit GUI.

This module contains visualization functions for biofilm-related data.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_biofilm_analysis_plots(df):
    """Create comprehensive biofilm parameter visualization.

    Args:
        df: DataFrame or dict with biofilm-related data.

    Returns:
        Plotly figure with biofilm analysis or None if no biofilm data.

    """
    # Handle both dict and DataFrame inputs
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict("list") if hasattr(df, "to_dict") else df

    # Check if we have biofilm data
    biofilm_cols = [
        col
        for col in columns
        if "biofilm" in col.lower()
        or "biomass" in col.lower()
        or "attachment" in col.lower()
    ]
    if not biofilm_cols:
        return None

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Biofilm Thickness per Cell",
            "Biomass Density Distribution",
            "Attachment Fraction",
            "Growth vs Detachment Rates",
            "Biofilm Conductivity",
            "Species Composition",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Plot 1: Per-cell biofilm thickness
    _add_biofilm_thickness_plot(fig, data_dict, row=1, col=1)

    # Plot 2: Biomass density heatmap
    _add_biomass_density_plot(fig, data_dict, row=1, col=2)

    # Plot 3: Attachment fraction
    _add_attachment_plot(fig, data_dict, row=1, col=3)

    fig.update_layout(
        title="Biofilm Analysis - Per Cell Parameters",
        showlegend=True,
        height=600,
    )
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="Thickness (um)", row=1, col=1)
    fig.update_yaxes(title_text="Density (g/L)", row=1, col=2)
    fig.update_yaxes(title_text="Attachment", row=1, col=3)

    return fig


def _add_biofilm_thickness_plot(fig, data_dict, row, col):
    """Add biofilm thickness traces to the figure.

    Args:
        fig: Plotly figure object to add traces to.
        data_dict: Data dictionary with biofilm thickness arrays.
        row: Subplot row index.
        col: Subplot column index.

    """
    if "biofilm_thicknesses" not in data_dict and "biofilm_thickness_per_cell" not in data_dict:
        return

    time_data = data_dict.get(
        "time_hours",
        data_dict.get(
            "time",
            list(range(len(data_dict.get("biofilm_thicknesses", [])))),
        ),
    )

    if "biofilm_thicknesses" in data_dict:
        biofilm_data = data_dict["biofilm_thicknesses"]
        if biofilm_data and isinstance(biofilm_data[0], list):
            n_cells = len(biofilm_data[0])
            for i in range(min(n_cells, 5)):
                cell_values = [
                    timepoint[i] if i < len(timepoint) else 0
                    for timepoint in biofilm_data
                ]
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=cell_values,
                        name=f"Cell {i + 1}",
                        line={"width": 2},
                    ),
                    row=row,
                    col=col,
                )
    elif "biofilm_thickness_per_cell" in data_dict:
        for i in range(5):
            col_name = f"biofilm_thickness_cell_{i}"
            if col_name in data_dict:
                fig.add_trace(
                    go.Scatter(
                        x=time_data,
                        y=data_dict[col_name],
                        name=f"Cell {i + 1}",
                        line={"width": 2},
                    ),
                    row=row,
                    col=col,
                )


def _add_biomass_density_plot(fig, data_dict, row, col):
    """Add biomass density heatmap to the figure.

    Args:
        fig: Plotly figure object to add traces to.
        data_dict: Data dictionary with biomass density arrays.
        row: Subplot row index.
        col: Subplot column index.

    """
    if "biomass_density" not in data_dict and "biomass_density_per_cell" not in data_dict:
        return

    time_data = data_dict.get(
        "time_hours",
        data_dict.get(
            "time",
            list(range(len(data_dict.get("biomass_density", [])))),
        ),
    )

    try:
        biomass_data = data_dict.get(
            "biomass_density",
            data_dict.get("biomass_density_per_cell", []),
        )
        if biomass_data and isinstance(biomass_data[0], list):
            transposed_data = list(map(list, zip(*biomass_data, strict=False)))

            fig.add_trace(
                go.Heatmap(
                    z=transposed_data,
                    x=time_data,
                    y=[f"Cell {i + 1}" for i in range(len(transposed_data))],
                    colorscale="Viridis",
                    name="Biomass Density",
                ),
                row=row,
                col=col,
            )
    except Exception:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                name="No biomass data",
                marker={"size": 0},
            ),
            row=row,
            col=col,
        )


def _add_attachment_plot(fig, data_dict, row, col):
    """Add attachment fraction plot to the figure.

    Args:
        fig: Plotly figure object to add traces to.
        data_dict: Data dictionary with attachment fraction data.
        row: Subplot row index.
        col: Subplot column index.

    """
    if "attachment_fraction" in data_dict:
        time_data = data_dict.get(
            "time_hours",
            data_dict.get("time", list(range(len(data_dict["attachment_fraction"])))),
        )
        avg_attachment = data_dict["attachment_fraction"]
    elif "biofilm_thicknesses" in data_dict:
        time_data = data_dict.get(
            "time_hours",
            data_dict.get("time", list(range(len(data_dict["biofilm_thicknesses"])))),
        )
        biofilm_data = data_dict["biofilm_thicknesses"]
        if biofilm_data and isinstance(biofilm_data[0], list):
            avg_attachment = [
                sum(timepoint) / len(timepoint) if timepoint else 0
                for timepoint in biofilm_data
            ]
        else:
            avg_attachment = biofilm_data if isinstance(biofilm_data, list) else [0]
    else:
        time_data = [0]
        avg_attachment = [0]

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=avg_attachment,
            name="Avg Attachment/Thickness",
            line={"color": "green", "width": 2},
        ),
        row=row,
        col=col,
    )
