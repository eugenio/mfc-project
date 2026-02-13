"""2D schematic rendering of MFC flow networks.

Produces interactive Plotly HTML and static Matplotlib figures
from a WNTR WaterNetworkModel or directly from CAD flow circuits.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from flow_viz.config import FlowVizConfig
from flow_viz.network_model import get_flow_circuit

if TYPE_CHECKING:
    from cad.cad_config import StackCADConfig
    from cad.hydraulics import FlowCircuit


def plot_schematic_plotly(
    cad_config: StackCADConfig,
    viz_config: FlowVizConfig | None = None,
    circuit: str = "anode",
) -> go.Figure:
    """Create an interactive Plotly schematic of the flow network.

    Parameters
    ----------
    cad_config : StackCADConfig
        Stack geometry configuration.
    viz_config : FlowVizConfig, optional
        Visualization config. Uses defaults if None.
    circuit : str
        ``"anode"`` or ``"cathode"``.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with nodes and pipes.

    """
    if viz_config is None:
        viz_config = FlowVizConfig()

    flow_circuit = get_flow_circuit(cad_config, circuit)

    fig = go.Figure()

    # Pipe traces
    _add_pipe_traces(fig, flow_circuit, viz_config)

    # Node traces
    _add_node_traces(fig, flow_circuit, viz_config)

    # Flow direction arrows
    _add_flow_arrows(fig, flow_circuit, viz_config)

    fig.update_layout(
        title=f"MFC {circuit.title()} Flow Schematic "
        f"({cad_config.num_cells} cells, "
        f"{cad_config.flow_config.value})",
        xaxis_title="Z position (mm)",
        yaxis_title="Y position (mm)",
        showlegend=True,
        template="plotly_dark",
        xaxis={"scaleanchor": "y", "scaleratio": 1},
        height=600,
        width=1200,
    )

    return fig


def _add_pipe_traces(
    fig: go.Figure,
    flow_circuit: FlowCircuit,
    viz_config: FlowVizConfig,
) -> None:
    """Add pipe segment traces to the figure."""
    for seg in flow_circuit.segments:
        x_coords = [seg.start.z]
        y_coords = [seg.start.y]

        for vp in seg.via_points:
            x_coords.append(vp[2])  # z
            y_coords.append(vp[1])  # y

        x_coords.append(seg.end.z)
        y_coords.append(seg.end.y)

        color = "#4fc3f7" if "inlet" in seg.label or "end" in seg.label else "#81c784"
        width = viz_config.pipe_width_scale * 2

        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line={"color": color, "width": width},
            name=seg.label,
            hoverinfo="text",
            text=f"{seg.label}<br>Length: {seg.length_m * 1000:.1f} mm",
            showlegend=False,
        ))


def _add_node_traces(
    fig: go.Figure,
    flow_circuit: FlowCircuit,
    viz_config: FlowVizConfig,
) -> None:
    """Add junction node markers to the figure."""
    ports_seen: dict[str, tuple[float, float, str]] = {}
    for seg in flow_circuit.segments:
        ports_seen[seg.start.label] = (seg.start.z, seg.start.y, seg.start.label)
        ports_seen[seg.end.label] = (seg.end.z, seg.end.y, seg.end.label)

    z_vals = [p[0] for p in ports_seen.values()]
    y_vals = [p[1] for p in ports_seen.values()]
    labels = [p[2] for p in ports_seen.values()]

    colors = []
    for lbl in labels:
        if "inlet" in lbl:
            colors.append("#4fc3f7")
        elif "outlet" in lbl:
            colors.append("#ef5350")
        elif "end_plate" in lbl:
            colors.append("#ffb74d")
        else:
            colors.append("#aaaaaa")

    fig.add_trace(go.Scatter(
        x=z_vals,
        y=y_vals,
        mode="markers+text",
        marker={
            "size": viz_config.node_size,
            "color": colors,
            "line": {"width": 1, "color": "#ffffff"},
        },
        text=[lbl.split("_")[-1] for lbl in labels],
        textposition="top center",
        textfont={"size": 8, "color": "#cccccc"},
        name="Ports",
        hovertext=labels,
    ))


def _add_flow_arrows(
    fig: go.Figure,
    flow_circuit: FlowCircuit,
    viz_config: FlowVizConfig,
) -> None:
    """Add flow direction arrow annotations."""
    for seg in flow_circuit.segments:
        mid_z = (seg.start.z + seg.end.z) / 2
        mid_y = (seg.start.y + seg.end.y) / 2
        dz = seg.end.z - seg.start.z
        dy = seg.end.y - seg.start.y

        scale = viz_config.arrow_scale * 5.0
        fig.add_annotation(
            x=mid_z + dz * 0.1,
            y=mid_y + dy * 0.1,
            ax=mid_z - dz * 0.1,
            ay=mid_y - dy * 0.1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=scale,
            arrowwidth=1.5,
            arrowcolor="#ffeb3b",
        )


def export_schematic_html(
    cad_config: StackCADConfig,
    viz_config: FlowVizConfig | None = None,
    circuit: str = "anode",
    output_path: Path | str | None = None,
) -> Path:
    """Export interactive schematic to an HTML file.

    Returns the path to the generated file.
    """
    if viz_config is None:
        viz_config = FlowVizConfig()

    if output_path is None:
        output_path = viz_config.output_path(f"schematic_{circuit}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_schematic_plotly(cad_config, viz_config, circuit)
    fig.write_html(str(output_path), include_plotlyjs=True)

    return output_path
