"""Metabolic analysis plots for MFC Streamlit GUI.

This module contains visualization functions for metabolic pathway data.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_metabolic_analysis_plots(df):
    """Create metabolic pathway visualization.

    Args:
        df: DataFrame or dict with metabolic data.

    Returns:
        Plotly figure with metabolic analysis or None if no metabolic data.

    """
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict("list") if hasattr(df, "to_dict") else df

    metabolic_cols = [
        col
        for col in columns
        if any(term in col.lower() for term in ["nadh", "atp", "electron", "metabolic"])
    ]
    if not metabolic_cols:
        return None

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "NADH/NAD+ Ratios",
            "ATP Levels",
            "Electron Flux",
            "Substrate Uptake Rates",
            "Metabolic Activity",
            "Oxygen Crossover",
        ),
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)],
    )

    time_data = data_dict.get("time_hours", data_dict.get("time", [0]))

    # Plot NADH ratios
    _add_nadh_plot(fig, data_dict, time_data, row=1, col=1)

    # Plot ATP levels
    _add_atp_plot(fig, data_dict, time_data, row=1, col=2)

    # Plot electron flux
    _add_electron_flux_plot(fig, data_dict, time_data, row=1, col=3)

    fig.update_layout(
        title="Metabolic Analysis - Per Cell Parameters",
        showlegend=True,
        height=600,
    )
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="NADH Ratio", row=1, col=1)
    fig.update_yaxes(title_text="ATP (mM)", row=1, col=2)
    fig.update_yaxes(title_text="Electron Flux", row=1, col=3)

    return fig


def _add_nadh_plot(fig, data_dict, time_data, row, col):
    """Add NADH ratio traces to the figure."""
    if "nadh_ratios" not in data_dict and "nadh_ratio" not in data_dict:
        return

    nadh_data = data_dict.get("nadh_ratios", data_dict.get("nadh_ratio", []))
    if nadh_data and isinstance(nadh_data[0], list):
        for i in range(min(len(nadh_data[0]), 5)):
            cell_values = [
                timepoint[i] if i < len(timepoint) else 0.3
                for timepoint in nadh_data
            ]
            fig.add_trace(
                go.Scatter(
                    x=time_data,
                    y=cell_values,
                    name=f"Cell {i + 1} NADH",
                    line={"width": 2},
                ),
                row=row,
                col=col,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=nadh_data if isinstance(nadh_data, list) else [0.3] * len(time_data),
                name="Avg NADH Ratio",
                line={"width": 2},
            ),
            row=row,
            col=col,
        )


def _add_atp_plot(fig, data_dict, time_data, row, col):
    """Add ATP level traces to the figure."""
    if "atp_levels" not in data_dict and "atp_level" not in data_dict:
        return

    atp_data = data_dict.get("atp_levels", data_dict.get("atp_level", []))
    if atp_data and isinstance(atp_data[0], list):
        avg_atp = [
            sum(timepoint) / len(timepoint) if timepoint else 2.0
            for timepoint in atp_data
        ]
    else:
        avg_atp = atp_data if isinstance(atp_data, list) else [2.0] * len(time_data)

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=avg_atp,
            name="Avg ATP",
            line={"color": "red", "width": 2},
        ),
        row=row,
        col=col,
    )


def _add_electron_flux_plot(fig, data_dict, time_data, row, col):
    """Add electron flux traces to the figure."""
    if "electron_flux" in data_dict:
        electron_data = data_dict["electron_flux"]
        if electron_data and isinstance(electron_data[0], list):
            avg_flux = [
                sum(timepoint) / len(timepoint) if timepoint else 0.1
                for timepoint in electron_data
            ]
        else:
            avg_flux = (
                electron_data
                if isinstance(electron_data, list)
                else [0.1] * len(time_data)
            )

        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=avg_flux,
                name="Avg e- Flux",
                line={"color": "blue", "width": 2},
            ),
            row=row,
            col=col,
        )
    elif "total_current" in data_dict:
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=data_dict["total_current"],
                name="Total Current (A)",
                line={"color": "blue", "width": 2},
            ),
            row=row,
            col=col,
        )
