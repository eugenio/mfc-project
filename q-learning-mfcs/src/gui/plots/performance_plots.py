"""Performance analysis plots for MFC Streamlit GUI.

This module contains visualization functions for performance metrics
and correlation analysis.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_performance_analysis_plots(df):
    """Create comprehensive performance analysis visualization.

    Args:
        df: DataFrame or dict with performance data.

    Returns:
        Plotly figure with performance analysis.
    """
    if isinstance(df, dict):
        data_dict = df
    else:
        data_dict = df.to_dict("list") if hasattr(df, "to_dict") else df

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "Energy Efficiency Over Time",
            "Coulombic Efficiency",
            "Power Density",
            "Cumulative Energy Production",
            "Control Performance",
            "Economic Metrics",
        ),
        specs=[[{"secondary_y": True} for _ in range(3)] for _ in range(2)],
    )

    time_data = data_dict.get("time_hours", data_dict.get("time", [0]))
    if not time_data:
        time_data = [0]

    # Energy efficiency
    if "energy_efficiency" in data_dict:
        energy_eff_data = data_dict.get("energy_efficiency", [75] * len(time_data))
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=energy_eff_data if energy_eff_data else [75] * len(time_data),
                name="Energy Efficiency (%)",
                line={"color": "green", "width": 2},
            ),
            row=1,
            col=1,
        )

    # Coulombic efficiency
    _add_coulombic_efficiency_plot(fig, data_dict, time_data, row=1, col=2)

    # Power density
    _add_power_density_plot(fig, data_dict, time_data, row=1, col=3)

    # Cumulative energy
    _add_cumulative_energy_plot(fig, data_dict, time_data, row=2, col=1)

    # Control performance
    _add_control_performance_plot(fig, data_dict, time_data, row=2, col=2)

    # Economic metrics
    _add_economic_metrics_plot(fig, data_dict, time_data, row=2, col=3)

    fig.update_layout(title="Performance Analysis", height=600, showlegend=True)

    return fig


def _add_coulombic_efficiency_plot(fig, data_dict, time_data, row, col):
    """Add coulombic efficiency plot."""
    if "coulombic_efficiency" not in data_dict and "coulombic_efficiency_per_cell" not in data_dict:
        return

    ce_key = (
        "coulombic_efficiency"
        if "coulombic_efficiency" in data_dict
        else "coulombic_efficiency_per_cell"
    )
    ce_data = data_dict.get(ce_key, [85] * len(time_data))
    if isinstance(ce_data, list) and ce_data and isinstance(ce_data[0], list):
        ce_data = [
            sum(timepoint) / len(timepoint) if timepoint else 85
            for timepoint in ce_data
        ]
    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=ce_data if ce_data else [85] * len(time_data),
            name="Coulombic Efficiency (%)",
            line={"color": "blue", "width": 2},
        ),
        row=row,
        col=col,
    )


def _add_power_density_plot(fig, data_dict, time_data, row, col):
    """Add power density plot."""
    if "total_power" not in data_dict and "power_density_per_cell" not in data_dict:
        return

    power_key = "total_power" if "total_power" in data_dict else "power_density_per_cell"
    power_data = data_dict.get(power_key, [2.0] * len(time_data))
    if isinstance(power_data, list) and power_data and isinstance(power_data[0], list):
        if power_key == "total_power":
            power_data = [sum(timepoint) if timepoint else 2.0 for timepoint in power_data]
        else:
            power_data = [
                sum(timepoint) / len(timepoint) if timepoint else 2.0
                for timepoint in power_data
            ]
    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=power_data if power_data else [2.0] * len(time_data),
            name="Power Density (W/m2)",
            line={"color": "red", "width": 2},
        ),
        row=row,
        col=col,
    )


def _add_cumulative_energy_plot(fig, data_dict, time_data, row, col):
    """Add cumulative energy production plot."""
    if "total_energy_produced" not in data_dict and "energy_produced" not in data_dict:
        return

    energy_key = (
        "total_energy_produced"
        if "total_energy_produced" in data_dict
        else "energy_produced"
    )
    energy_data = data_dict.get(energy_key, list(range(len(time_data))))
    if energy_data:
        cumulative_energy = []
        cumsum = 0
        for val in energy_data:
            if isinstance(val, int | float):
                cumsum += val
            elif isinstance(val, list) and val:
                cumsum += sum(val)
            cumulative_energy.append(cumsum)
    else:
        cumulative_energy = list(range(len(time_data)))

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=cumulative_energy,
            name="Cumulative Energy (Wh)",
            line={"color": "purple", "width": 2},
        ),
        row=row,
        col=col,
    )


def _add_control_performance_plot(fig, data_dict, time_data, row, col):
    """Add control performance plot."""
    if "control_error" in data_dict:
        control_error = data_dict.get("control_error", [0] * len(time_data))
    elif "outlet_concentration" in data_dict:
        outlet_conc = data_dict.get("outlet_concentration", [25] * len(time_data))
        control_error = [
            abs(25 - x) if isinstance(x, int | float) else 0 for x in outlet_conc
        ]
    else:
        control_error = [0] * len(time_data)

    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=control_error,
            name="Control Error (mM)",
            line={"color": "orange", "width": 2},
        ),
        row=row,
        col=col,
    )


def _add_economic_metrics_plot(fig, data_dict, time_data, row, col):
    """Add economic metrics plot."""
    if "operating_cost" not in data_dict and "revenue" not in data_dict:
        return

    cost_data = data_dict.get("operating_cost", [1.0] * len(time_data))
    revenue_data = data_dict.get("revenue", [2.0] * len(time_data))
    profit = [
        (r - c) if isinstance(r, int | float) and isinstance(c, int | float) else 1.0
        for r, c in zip(revenue_data, cost_data, strict=False)
    ]
    fig.add_trace(
        go.Scatter(
            x=time_data,
            y=profit,
            name="Profit ($/h)",
            line={"color": "gold", "width": 2},
        ),
        row=row,
        col=col,
    )


def create_parameter_correlation_matrix(df):
    """Create correlation matrix for key parameters.

    Args:
        df: DataFrame or dict with simulation data.

    Returns:
        Plotly figure with correlation heatmap or None if insufficient data.
    """
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict("list") if hasattr(df, "to_dict") else df

    # Select numeric parameters for correlation analysis
    key_params = []
    numeric_data = {}

    for col in columns:
        if col.startswith(("time", "step")):
            continue

        data = data_dict.get(col, [])
        if not data:
            continue

        # Handle different data types
        if isinstance(data, list):
            if data and isinstance(data[0], list):
                try:
                    numeric_values = [
                        sum(timepoint) / len(timepoint) if timepoint else 0
                        for timepoint in data
                    ]
                except (TypeError, ZeroDivisionError):
                    continue
            else:
                try:
                    numeric_values = [
                        float(x) if isinstance(x, int | float) else 0 for x in data
                    ]
                except (ValueError, TypeError):
                    continue
        else:
            continue

        if numeric_values and len(numeric_values) > 1:
            key_params.append(col)
            numeric_data[col] = numeric_values

    if len(key_params) < 2:
        return None

    # Ensure all arrays have the same length
    min_length = min(len(numeric_data[param]) for param in key_params)
    for param in key_params:
        numeric_data[param] = numeric_data[param][:min_length]

    # Calculate correlation matrix
    temp_df = pd.DataFrame(numeric_data)

    try:
        corr_matrix = temp_df.corr()
    except Exception:
        return None

    corr_matrix = corr_matrix.fillna(0)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
        ),
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Parameter Correlation Matrix",
    )

    return fig
