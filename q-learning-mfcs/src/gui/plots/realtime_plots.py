"""Real-time monitoring plots for MFC Streamlit GUI.

This module contains the core real-time visualization functions used
in the MFC simulation monitoring dashboard.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def create_real_time_plots(df):
    """Create real-time monitoring plots.

    Args:
        df: DataFrame with simulation data containing columns like
            time_hours, reservoir_concentration, outlet_concentration,
            total_power, q_action, biofilm_thicknesses, etc.

    Returns:
        Plotly figure with comprehensive monitoring dashboard.
    """
    # Create subplots - 4x3 grid to accommodate all plots
    fig = make_subplots(
        rows=4,
        cols=3,
        subplot_titles=(
            "Substrate Concentration",
            "Power Output",
            "System Voltage",
            "Q-Learning Actions",
            "Biofilm Growth",
            "Flow Control",
            "Individual Cell Powers",
            "Mixing & Control",
            "Q-Values & Rewards",
            "Cumulative Energy",
            "",
            "",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
        ],
    )

    # Substrate concentration plot
    fig.add_trace(
        go.Scatter(
            x=df["time_hours"],
            y=df["reservoir_concentration"],
            name="Reservoir",
            line={"color": "blue", "width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_hours"],
            y=df["outlet_concentration"],
            name="Outlet",
            line={"color": "red", "width": 2},
        ),
        row=1,
        col=1,
    )
    # Target line
    fig.add_hline(
        y=25.0,
        line_dash="dash",
        line_color="green",
        annotation_text="Target (25 mM)",
        row=1,
        col=1,
    )

    # Power output
    fig.add_trace(
        go.Scatter(
            x=df["time_hours"],
            y=df["total_power"],
            name="Power",
            line={"color": "orange", "width": 2},
        ),
        row=1,
        col=2,
    )

    # Q-learning actions
    fig.add_trace(
        go.Scatter(
            x=df["time_hours"],
            y=df["q_action"],
            mode="markers",
            name="Actions",
            marker={"color": "purple", "size": 4},
        ),
        row=2,
        col=1,
    )

    # Biofilm thickness (average)
    if "biofilm_thicknesses" in df.columns:
        biofilm_avg = df["biofilm_thicknesses"].apply(_parse_biofilm_data)

        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=biofilm_avg,
                name="Avg Thickness",
                line={"color": "brown", "width": 2},
            ),
            row=2,
            col=2,
        )

    # Plot 5: Flow Rate & Efficiency (dual y-axis)
    if "flow_rate_ml_h" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["flow_rate_ml_h"],
                name="Flow Rate (mL/h)",
                line={"color": "cyan", "width": 2},
            ),
            row=2,
            col=2,
        )
    if "substrate_efficiency" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["substrate_efficiency"],
                name="Efficiency",
                line={"color": "green", "width": 2},
            ),
            row=2,
            col=2,
            secondary_y=True,
        )

    # Plot 6: System Performance (voltage only)
    if "system_voltage" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["system_voltage"],
                name="Voltage (V)",
                line={"color": "blue", "width": 2},
            ),
            row=2,
            col=3,
        )

    # Plot 7: Individual Cell Powers
    if "individual_cell_powers" in df.columns:
        for i in range(
            min(3, len(df["individual_cell_powers"].iloc[0]) if len(df) > 0 else 0),
        ):
            cell_powers = [
                powers[i] if len(powers) > i else 0
                for powers in df["individual_cell_powers"]
            ]
            fig.add_trace(
                go.Scatter(
                    x=df["time_hours"],
                    y=cell_powers,
                    name=f"Cell {i + 1}",
                    line={"width": 1.5},
                ),
                row=3,
                col=1,
            )

    # Plot 8: Mixing & Control (dual y-axis)
    if "mixing_efficiency" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["mixing_efficiency"],
                name="Mixing Eff",
                line={"color": "purple", "width": 2},
            ),
            row=3,
            col=2,
        )
    if "biofilm_activity_factor" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["biofilm_activity_factor"],
                name="Biofilm Activity",
                line={"color": "orange", "width": 2},
            ),
            row=3,
            col=2,
            secondary_y=True,
        )

    # Plot 9: Q-Values & Rewards (dual y-axis)
    if "q_value" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["q_value"],
                name="Q-Value",
                line={"color": "blue", "width": 2},
            ),
            row=3,
            col=3,
        )
    if "reward" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time_hours"],
                y=df["reward"],
                name="Reward",
                line={"color": "green", "width": 2},
            ),
            row=3,
            col=3,
            secondary_y=True,
        )

    # Plot 10: Cumulative Energy
    if "total_power" in df.columns and "time_hours" in df.columns:
        _add_cumulative_energy_plot(fig, df)

    # Update layout for expanded 4x3 grid
    fig.update_layout(
        height=1200,
        showlegend=True,
        title_text="Comprehensive MFC Simulation Monitoring Dashboard",
    )

    # Update axes labels for all plots
    _update_axes_labels(fig, df)

    return fig


def _parse_biofilm_data(x):
    """Safely parse biofilm thickness data from various formats."""
    try:
        if isinstance(x, list | tuple):
            return sum(x) / len(x) if len(x) > 0 else 1.0
        if isinstance(x, str) and x.strip():
            x_clean = x.strip("[]() ").replace(" ", "")
            if "," in x_clean:
                values = [
                    float(val.strip())
                    for val in x_clean.split(",")
                    if val.strip()
                ]
                return sum(values) / len(values) if len(values) > 0 else 1.0
            return float(x_clean) if x_clean else 1.0
        if isinstance(x, int | float):
            return float(x)
        return 1.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 1.0


def _add_cumulative_energy_plot(fig, df):
    """Add cumulative energy plot to the figure."""
    time_hours = df["time_hours"].values
    power_watts = df["total_power"].values

    # Calculate time differences in hours for integration
    dt_hours = np.diff(time_hours, prepend=0)

    # Energy = Power x Time (Wh = W x h)
    energy_increments = power_watts * dt_hours
    cumulative_energy_wh = np.cumsum(energy_increments)

    # Convert to more appropriate units
    if cumulative_energy_wh[-1] > 1000:
        cumulative_energy_display = cumulative_energy_wh / 1000
        energy_unit = "kWh"
    else:
        cumulative_energy_display = cumulative_energy_wh
        energy_unit = "Wh"

    fig.add_trace(
        go.Scatter(
            x=time_hours,
            y=cumulative_energy_display,
            name=f"Cumulative Energy ({energy_unit})",
            line={"color": "darkgreen", "width": 3},
            fill="tonexty" if len(fig.data) == 0 else "tozeroy",
            fillcolor="rgba(0,128,0,0.1)",
        ),
        row=4,
        col=1,
    )

    # Add energy efficiency indicator
    if len(time_hours) > 1:
        total_time = time_hours[-1] - time_hours[0]
        if total_time > 0:
            avg_power = np.mean(power_watts)
            energy_efficiency = (
                cumulative_energy_display[-1] / total_time if total_time > 0 else 0
            )

            fig.add_annotation(
                x=time_hours[-1] * 0.7,
                y=cumulative_energy_display[-1] * 0.8,
                text=f"Total: {cumulative_energy_display[-1]:.2f} {energy_unit}<br>"
                f"Avg Power: {avg_power:.3f} W<br>"
                f"Rate: {energy_efficiency:.3f} {energy_unit}/h",
                showarrow=True,
                arrowhead=2,
                arrowcolor="darkgreen",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="darkgreen",
                row=4,
                col=1,
            )


def _update_axes_labels(fig, df):
    """Update axes labels for all plots in the figure."""
    # Row 1
    fig.update_yaxes(title_text="Concentration (mM)", row=1, col=1)
    fig.update_yaxes(title_text="Error (mM)", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Current (A)", secondary_y=True, row=1, col=2)
    fig.update_yaxes(title_text="Action ID", row=1, col=3)

    # Row 2
    fig.update_yaxes(title_text="Thickness (um)", row=2, col=1)
    fig.update_yaxes(title_text="Flow Rate (mL/h)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency", secondary_y=True, row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=3)

    # Row 3
    fig.update_yaxes(title_text="Power (W)", row=3, col=1)
    fig.update_yaxes(title_text="Mixing Efficiency", row=3, col=2)
    fig.update_yaxes(title_text="Activity Factor", secondary_y=True, row=3, col=2)
    fig.update_yaxes(title_text="Q-Value", row=3, col=3)
    fig.update_yaxes(title_text="Reward", secondary_y=True, row=3, col=3)

    # Row 4 (Cumulative Energy)
    if "total_power" in df.columns:
        energy_unit = "kWh" if df["total_power"].sum() * len(df) / 1000 > 1 else "Wh"
        fig.update_yaxes(title_text=f"Energy ({energy_unit})", row=4, col=1)

    # Add time labels to bottom row
    fig.update_xaxes(title_text="Time (hours)", row=4, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=4, col=2)
    fig.update_xaxes(title_text="Time (hours)", row=4, col=3)


def create_performance_dashboard(results) -> None:
    """Create performance metrics dashboard.

    Args:
        results: Dictionary containing simulation results with
            'performance_metrics' key.
    """
    metrics = results.get("performance_metrics", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Concentration",
            f"{metrics.get('final_reservoir_concentration', 0):.2f} mM",
            delta=f"{metrics.get('final_reservoir_concentration', 0) - 25:.2f}",
        )

    with col2:
        st.metric(
            "Control Effectiveness (+/-2mM)",
            f"{metrics.get('control_effectiveness_2mM', 0):.1f}%",
        )

    with col3:
        st.metric("Mean Power", f"{metrics.get('mean_power', 0):.3f} W")

    with col4:
        st.metric(
            "Substrate Consumed",
            f"{metrics.get('total_substrate_added', 0):.1f} mmol",
        )
