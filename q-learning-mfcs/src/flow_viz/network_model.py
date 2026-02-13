"""Bridge between MFC hydraulics and WNTR water network model.

Converts flow paths computed by ``cad.hydraulics`` into a
``wntr.network.WaterNetworkModel`` for hydraulic analysis
and schematic visualization.
"""

from __future__ import annotations

import wntr
from cad.cad_config import FlowConfiguration, StackCADConfig
from cad.hydraulics import (
    FlowCircuit,
    PortPosition,
    compute_parallel_flow_path,
    compute_port_positions,
    compute_series_flow_path,
)
from flow_viz.config import FlowVizConfig


def _port_to_junction_name(port: PortPosition) -> str:
    """Deterministic junction name from a port label."""
    return f"j_{port.label}"


def _port_coords(port: PortPosition) -> tuple[float, float]:
    """Project port 3D position to 2D schematic (Z, Y) plane."""
    return (port.z, port.y)


def build_network(
    cad_config: StackCADConfig,
    viz_config: FlowVizConfig | None = None,
    circuit: str = "anode",
) -> wntr.network.WaterNetworkModel:
    """Build a WNTR network from the MFC hydraulic configuration.

    Parameters
    ----------
    cad_config : StackCADConfig
        Stack geometry configuration.
    viz_config : FlowVizConfig, optional
        Visualization config (fluid properties). Uses defaults if None.
    circuit : str
        ``"anode"`` or ``"cathode"``.

    Returns
    -------
    wntr.network.WaterNetworkModel
        Network with junctions at port positions, pipes along
        tubing segments, a reservoir node, and a pump.

    """
    if viz_config is None:
        viz_config = FlowVizConfig()

    wn = wntr.network.WaterNetworkModel()

    # Compute flow path
    if cad_config.flow_config == FlowConfiguration.SERIES:
        flow_circuit = compute_series_flow_path(cad_config, circuit)
    else:
        flow_circuit = compute_parallel_flow_path(cad_config, circuit)

    # Collect all unique ports from segments
    ports_seen: dict[str, PortPosition] = {}
    for seg in flow_circuit.segments:
        ports_seen[seg.start.label] = seg.start
        ports_seen[seg.end.label] = seg.end

    # Add reservoir node
    wn.add_reservoir(
        "reservoir",
        base_head=cad_config.reservoir.inner_height * 1000,  # mm
        coordinates=(0.0, -100.0),
    )

    # Add junctions for each port
    for port in ports_seen.values():
        jname = _port_to_junction_name(port)
        coords = _port_coords(port)
        wn.add_junction(
            jname,
            base_demand=0.0,
            coordinates=coords,
        )

    # Add pipes for each tubing segment
    tubing_diam_m = cad_config.tubing.inner_diameter
    for _i, seg in enumerate(flow_circuit.segments):
        start_name = _port_to_junction_name(seg.start)
        end_name = _port_to_junction_name(seg.end)
        pipe_name = f"pipe_{seg.label}"

        wn.add_pipe(
            pipe_name,
            start_node_name=start_name,
            end_node_name=end_name,
            length=max(seg.length_m, 0.001),
            diameter=tubing_diam_m,
            roughness=0.0015,  # smooth tubing
        )

    # Connect reservoir to first junction via pump
    first_seg = flow_circuit.segments[0]
    first_junction = _port_to_junction_name(first_seg.start)

    wn.add_pump(
        "pump",
        start_node_name="reservoir",
        end_node_name=first_junction,
        pump_type="POWER",
        pump_parameter=0.001,  # kW — small peristaltic
    )

    return wn


def get_flow_circuit(
    cad_config: StackCADConfig,
    circuit: str = "anode",
) -> FlowCircuit:
    """Get the flow circuit for a given configuration and circuit type."""
    if cad_config.flow_config == FlowConfiguration.SERIES:
        return compute_series_flow_path(cad_config, circuit)
    return compute_parallel_flow_path(cad_config, circuit)


def get_port_positions(
    cad_config: StackCADConfig,
) -> list[PortPosition]:
    """Get all port positions for the given configuration."""
    return list(compute_port_positions(cad_config))
