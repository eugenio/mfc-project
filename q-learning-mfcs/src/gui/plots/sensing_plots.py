"""Sensing analysis plots for MFC Streamlit GUI.

This module contains visualization functions for EIS and QCM sensing data.
"""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_sensing_analysis_plots(df):
    """Create EIS and QCM sensing visualization.

    Args:
        df: DataFrame or dict with sensing data.

    Returns:
        Plotly figure with sensing analysis or None if no sensing data.

    """
    if isinstance(df, dict):
        columns = df.keys()
        data_dict = df
    else:
        columns = df.columns
        data_dict = df.to_dict("list") if hasattr(df, "to_dict") else df

    sensing_cols = [
        col
        for col in columns
        if any(term in col.lower() for term in ["eis", "qcm", "impedance", "frequency"])
    ]
    if not sensing_cols:
        return None

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "EIS Impedance Magnitude",
            "EIS Phase Response",
            "QCM Frequency Shift",
            "Charge Transfer Resistance",
            "QCM Mass Loading",
            "Sensor Calibration",
        ),
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)],
    )

    time_data = data_dict.get("time_hours", data_dict.get("time", [0]))
    if not time_data:
        time_data = [0]

    # EIS Impedance magnitude
    if "eis_impedance_magnitude" in data_dict:
        impedance_data = data_dict.get(
            "eis_impedance_magnitude",
            [1000] * len(time_data),
        )
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=impedance_data if impedance_data else [1000] * len(time_data),
                name="|Z| @ 1kHz",
                line={"color": "purple", "width": 2},
            ),
            row=1,
            col=1,
        )

    # EIS Phase
    if "eis_impedance_phase" in data_dict:
        phase_data = data_dict.get("eis_impedance_phase", [-45] * len(time_data))
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=phase_data if phase_data else [-45] * len(time_data),
                name="Phase @ 1kHz",
                line={"color": "orange", "width": 2},
            ),
            row=1,
            col=2,
        )

    # QCM frequency shift
    if "qcm_frequency_shift" in data_dict:
        freq_data = data_dict.get("qcm_frequency_shift", [-500] * len(time_data))
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=freq_data if freq_data else [-500] * len(time_data),
                name="df (Hz)",
                line={"color": "green", "width": 2},
            ),
            row=1,
            col=3,
        )

    # Charge transfer resistance
    if "charge_transfer_resistance" in data_dict:
        rct_data = data_dict.get("charge_transfer_resistance", [100] * len(time_data))
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=rct_data if rct_data else [100] * len(time_data),
                name="Rct (Ohm)",
                line={"color": "red", "width": 2},
            ),
            row=2,
            col=1,
        )

    # QCM mass loading
    if "qcm_mass_loading" in data_dict:
        mass_data = data_dict.get("qcm_mass_loading", [0.1] * len(time_data))
        fig.add_trace(
            go.Scatter(
                x=time_data,
                y=mass_data if mass_data else [0.1] * len(time_data),
                name="Mass Loading (ug/cm2)",
                line={"color": "brown", "width": 2},
            ),
            row=2,
            col=2,
        )

    fig.update_layout(title="Sensing Analysis", height=600, showlegend=True)

    return fig
