"""Pure-Python flow path computation for MFC stack hydraulics.

Computes port positions, tubing segment geometry, series/parallel
flow paths, and total circuit lengths + dead volumes.  No CadQuery
dependency — geometry is derived from ``StackCADConfig`` parameters.

All positions are in **mm** (matching CadQuery assembly coordinates).
Lengths returned by flow-path functions are in **metres**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .cad_config import FlowConfiguration, StackCADConfig


def _mm(m: float) -> float:
    """Convert metres to millimetres."""
    return m * 1000.0


def _m(mm: float) -> float:
    """Convert millimetres to metres."""
    return mm / 1000.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PortPosition:
    """3D position + outward normal of a single flow port (in mm)."""

    x: float
    y: float
    z: float
    normal: tuple[float, float, float]
    label: str


@dataclass
class TubingSegment:
    """A single tube segment connecting two ports."""

    start: PortPosition
    end: PortPosition
    via_points: list[tuple[float, float, float]]
    length_m: float
    label: str


@dataclass
class FlowCircuit:
    """Complete flow path for one hydraulic circuit."""

    segments: list[TubingSegment]
    total_length_m: float
    dead_volume_m3: float
    circuit_type: str  # "anode" or "cathode"


# ---------------------------------------------------------------------------
# Port position computation
# ---------------------------------------------------------------------------


def compute_port_positions(config: StackCADConfig) -> list[PortPosition]:
    """Compute 3D positions of all flow ports on the stack.

    Coordinate system (matching assembly.py):
    - Z: compression axis (stack length)
    - X, Y: frame face directions
    - Origin: centre of inlet end plate inner face

    Anode ports: -X wall (inlet), +X wall (outlet)
    Cathode ports: -Y wall (inlet), +Y wall (outlet)

    Returns positions in **mm**.
    """
    ports: list[PortPosition] = []
    ep_thick = _mm(config.end_plate.thickness)
    cell_thick = _mm(config.cell_thickness)
    half_outer = _mm(config.outer_side / 2)
    anode_depth = _mm(config.semi_cell.depth)

    for i in range(config.num_cells):
        # Cell base Z: after end plate + i cells
        cell_base_z = ep_thick + i * cell_thick
        # Anode centre Z is at cell_base + anode_depth / 2
        anode_centre_z = cell_base_z + anode_depth / 2

        # Cathode centre Z is at cell_base + anode_depth + gasket + cathode_depth/2
        gasket = _mm(config.gasket_membrane_thickness)
        cathode_depth = _mm(config.semi_cell.depth)
        cathode_centre_z = cell_base_z + anode_depth + gasket + cathode_depth / 2

        # Anode inlet (-X wall)
        ports.append(PortPosition(
            x=-half_outer, y=0.0, z=anode_centre_z,
            normal=(-1.0, 0.0, 0.0),
            label=f"anode_{i}_inlet",
        ))
        # Anode outlet (+X wall)
        ports.append(PortPosition(
            x=half_outer, y=0.0, z=anode_centre_z,
            normal=(1.0, 0.0, 0.0),
            label=f"anode_{i}_outlet",
        ))
        # Cathode inlet (-Y wall)
        ports.append(PortPosition(
            x=0.0, y=-half_outer, z=cathode_centre_z,
            normal=(0.0, -1.0, 0.0),
            label=f"cathode_{i}_inlet",
        ))
        # Cathode outlet (+Y wall)
        ports.append(PortPosition(
            x=0.0, y=half_outer, z=cathode_centre_z,
            normal=(0.0, 1.0, 0.0),
            label=f"cathode_{i}_outlet",
        ))

    # End plate flow ports (on outer faces)
    ports.append(PortPosition(
        x=0.0, y=0.0, z=0.0,
        normal=(0.0, 0.0, -1.0),
        label="end_plate_inlet",
    ))
    stack_len = _mm(config.stack_length)
    ports.append(PortPosition(
        x=0.0, y=0.0, z=stack_len,
        normal=(0.0, 0.0, 1.0),
        label="end_plate_outlet",
    ))

    return ports


# ---------------------------------------------------------------------------
# Series flow path
# ---------------------------------------------------------------------------


def _utube_length_mm(clearance_mm: float, traverse_mm: float) -> float:
    """Length of one U-tube: exit arm + traverse + re-enter arm."""
    return 2 * clearance_mm + traverse_mm


def compute_series_flow_path(
    config: StackCADConfig,
    circuit: str,
) -> FlowCircuit:
    """Compute a series flow path for the given circuit.

    Parameters
    ----------
    config : StackCADConfig
    circuit : str
        ``"anode"`` or ``"cathode"``.

    Returns
    -------
    FlowCircuit
        Segments include end connections + inter-cell U-tubes.
    """
    ports = compute_port_positions(config)
    n = config.num_cells
    clearance_mm = _mm(config.utube_clearance)

    # Filter ports for this circuit
    if circuit == "anode":
        inlets = sorted(
            [p for p in ports if "anode" in p.label and "inlet" in p.label],
            key=lambda p: p.z,
        )
        outlets = sorted(
            [p for p in ports if "anode" in p.label and "outlet" in p.label],
            key=lambda p: p.z,
        )
        extra_length = config.anode_tubing_extra_length
    else:
        inlets = sorted(
            [p for p in ports if "cathode" in p.label and "inlet" in p.label],
            key=lambda p: p.z,
        )
        outlets = sorted(
            [p for p in ports if "cathode" in p.label and "outlet" in p.label],
            key=lambda p: p.z,
        )
        extra_length = config.cathode_tubing_extra_length

    segments: list[TubingSegment] = []
    total_length_mm = 0.0

    # End connection: inlet end plate to cell 0 inlet
    end_in = [p for p in ports if p.label == "end_plate_inlet"][0]
    inlet_length_mm = _mm(extra_length)
    segments.append(TubingSegment(
        start=end_in,
        end=inlets[0],
        via_points=[],
        length_m=_m(inlet_length_mm),
        label=f"{circuit}_end_inlet",
    ))
    total_length_mm += inlet_length_mm

    # Inter-cell U-tubes: outlet of cell i -> inlet of cell i+1
    cell_thick_mm = _mm(config.cell_thickness)
    for i in range(n - 1):
        out_port = outlets[i]
        in_port = inlets[i + 1]
        traverse_mm = abs(in_port.z - out_port.z)
        utube_len = _utube_length_mm(clearance_mm, traverse_mm)

        # Via points: exit from outlet, traverse along Z, enter next inlet
        nx, ny, nz = out_port.normal
        exit_pt = (
            out_port.x + nx * clearance_mm,
            out_port.y + ny * clearance_mm,
            out_port.z,
        )
        enter_pt = (
            in_port.x + in_port.normal[0] * clearance_mm,
            in_port.y + in_port.normal[1] * clearance_mm,
            in_port.z,
        )

        segments.append(TubingSegment(
            start=out_port,
            end=in_port,
            via_points=[exit_pt, enter_pt],
            length_m=_m(utube_len),
            label=f"{circuit}_utube_{i}",
        ))
        total_length_mm += utube_len

    # End connection: last cell outlet to outlet end plate
    end_out = [p for p in ports if p.label == "end_plate_outlet"][0]
    outlet_length_mm = _mm(extra_length)
    segments.append(TubingSegment(
        start=outlets[-1],
        end=end_out,
        via_points=[],
        length_m=_m(outlet_length_mm),
        label=f"{circuit}_end_outlet",
    ))
    total_length_mm += outlet_length_mm

    total_length_m = _m(total_length_mm)
    dead_volume = config.tubing.dead_volume(total_length_m)

    return FlowCircuit(
        segments=segments,
        total_length_m=total_length_m,
        dead_volume_m3=dead_volume,
        circuit_type=circuit,
    )


# ---------------------------------------------------------------------------
# Parallel flow path
# ---------------------------------------------------------------------------


def compute_parallel_flow_path(
    config: StackCADConfig,
    circuit: str,
) -> FlowCircuit:
    """Compute a parallel flow path with supply/return manifolds.

    Parameters
    ----------
    config : StackCADConfig
    circuit : str
        ``"anode"`` or ``"cathode"``.

    Returns
    -------
    FlowCircuit
        Segments: 1 supply header + N branches + 1 return header.
    """
    ports = compute_port_positions(config)
    n = config.num_cells
    standoff_mm = _mm(config.manifold_standoff)
    stack_len_mm = _mm(config.stack_length)

    if circuit == "anode":
        inlets = sorted(
            [p for p in ports if "anode" in p.label and "inlet" in p.label],
            key=lambda p: p.z,
        )
        outlets = sorted(
            [p for p in ports if "anode" in p.label and "outlet" in p.label],
            key=lambda p: p.z,
        )
    else:
        inlets = sorted(
            [p for p in ports if "cathode" in p.label and "inlet" in p.label],
            key=lambda p: p.z,
        )
        outlets = sorted(
            [p for p in ports if "cathode" in p.label and "outlet" in p.label],
            key=lambda p: p.z,
        )

    segments: list[TubingSegment] = []
    total_length_mm = 0.0

    # Supply header: runs along stack length at standoff distance
    header_len_mm = stack_len_mm
    supply_start = PortPosition(
        x=inlets[0].x + inlets[0].normal[0] * standoff_mm,
        y=inlets[0].y + inlets[0].normal[1] * standoff_mm,
        z=inlets[0].z,
        normal=inlets[0].normal,
        label=f"{circuit}_supply_header_start",
    )
    supply_end = PortPosition(
        x=inlets[-1].x + inlets[-1].normal[0] * standoff_mm,
        y=inlets[-1].y + inlets[-1].normal[1] * standoff_mm,
        z=inlets[-1].z,
        normal=inlets[-1].normal,
        label=f"{circuit}_supply_header_end",
    )
    segments.append(TubingSegment(
        start=supply_start,
        end=supply_end,
        via_points=[],
        length_m=_m(header_len_mm),
        label=f"{circuit}_supply_header",
    ))
    total_length_mm += header_len_mm

    # Branch segments: manifold to each cell inlet port
    for i, inlet in enumerate(inlets):
        branch_len_mm = standoff_mm  # perpendicular from manifold to port
        branch_start = PortPosition(
            x=inlet.x + inlet.normal[0] * standoff_mm,
            y=inlet.y + inlet.normal[1] * standoff_mm,
            z=inlet.z,
            normal=inlet.normal,
            label=f"{circuit}_branch_{i}_start",
        )
        segments.append(TubingSegment(
            start=branch_start,
            end=inlet,
            via_points=[],
            length_m=_m(branch_len_mm),
            label=f"{circuit}_branch_{i}",
        ))
        total_length_mm += branch_len_mm

    # Return header
    return_start = PortPosition(
        x=outlets[0].x + outlets[0].normal[0] * standoff_mm,
        y=outlets[0].y + outlets[0].normal[1] * standoff_mm,
        z=outlets[0].z,
        normal=outlets[0].normal,
        label=f"{circuit}_return_header_start",
    )
    return_end = PortPosition(
        x=outlets[-1].x + outlets[-1].normal[0] * standoff_mm,
        y=outlets[-1].y + outlets[-1].normal[1] * standoff_mm,
        z=outlets[-1].z,
        normal=outlets[-1].normal,
        label=f"{circuit}_return_header_end",
    )
    segments.append(TubingSegment(
        start=return_start,
        end=return_end,
        via_points=[],
        length_m=_m(header_len_mm),
        label=f"{circuit}_return_header",
    ))
    total_length_mm += header_len_mm

    total_length_m = _m(total_length_mm)
    dead_volume = config.tubing.dead_volume(total_length_m)

    return FlowCircuit(
        segments=segments,
        total_length_m=total_length_m,
        dead_volume_m3=dead_volume,
        circuit_type=circuit,
    )


# ---------------------------------------------------------------------------
# Summary functions
# ---------------------------------------------------------------------------


def compute_total_circuit_lengths(
    config: StackCADConfig,
) -> dict[str, float]:
    """Compute total tubing lengths (metres) for anode + cathode circuits.

    Uses ``config.flow_config`` to select series or parallel topology.

    Returns
    -------
    dict
        ``{"anode": m, "cathode": m, "total": m}``
    """
    if config.flow_config == FlowConfiguration.SERIES:
        path_fn = compute_series_flow_path
    else:
        path_fn = compute_parallel_flow_path

    anode = path_fn(config, "anode")
    cathode = path_fn(config, "cathode")

    return {
        "anode": anode.total_length_m,
        "cathode": cathode.total_length_m,
        "total": anode.total_length_m + cathode.total_length_m,
    }


def compute_total_dead_volume(config: StackCADConfig) -> float:
    """Total dead volume (m³) across all tubing circuits.

    Includes reservoir and pump tubing if those lengths are nonzero.
    """
    lengths = compute_total_circuit_lengths(config)
    tube_dv = config.tubing.dead_volume(lengths["total"])

    # Add reservoir + pump tubing dead volumes
    extra_lengths = config.reservoir_tubing_length + config.pump_tubing_length
    extra_dv = config.tubing.dead_volume(extra_lengths)

    return tube_dv + extra_dv
