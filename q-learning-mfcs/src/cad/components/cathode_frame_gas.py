"""Gas-collection cathode frame variant.

Extended depth to provide headspace above the electrode for
H2 or biogas accumulation. Includes a gas outlet port on the
top face and an internal step/ledge to support the electrode.
"""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig
from .oring import compute_face_seal_groove


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the gas-collection cathode frame.

    The total depth = standard cathode depth + headspace_depth.
    """
    sc = config.semi_cell
    gc = config.gas_cathode
    outer = _mm(sc.outer_side)
    inner = _mm(sc.inner_side)
    total_depth = _mm(sc.depth + gc.headspace_depth)

    # --- base plate with extended depth ---
    plate = cq.Workplane("XY").box(outer, outer, total_depth)

    # --- full chamber pocket ---
    plate = (
        plate.faces(">Z")
        .workplane()
        .rect(inner, inner)
        .cutBlind(-_mm(sc.depth + gc.headspace_depth - 0.002))
    )

    # --- internal ledge / step for electrode support ---
    # The electrode rests on a step at the bottom of the chamber.
    # The ledge width is the electrode thickness clearance.
    ledge_depth = _mm(gc.headspace_depth)
    ledge_width = 2.0  # mm — small shelf lip
    plate = (
        plate.faces(">Z")
        .workplane(offset=-ledge_depth)
        .rect(inner + ledge_width, inner + ledge_width)
        .rect(inner, inner)
        .cutBlind(-_mm(0.002))  # 2 mm lip
    )

    # --- O-ring groove on sealing face ---
    face_groove = compute_face_seal_groove(config.face_oring)
    gd = _mm(face_groove.depth)
    groove_offset = _mm(sc.wall_thickness) / 2
    groove_side = inner + groove_offset * 2
    plate = (
        plate.faces(">Z")
        .workplane()
        .rect(groove_side, groove_side)
        .cutBlind(-gd)
    )

    # --- tie-rod clearance holes ---
    tr = config.tie_rod
    for x, y in config.tie_rod_positions:
        plate = (
            plate.faces(">Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(_mm(tr.clearance_hole_diameter), total_depth)
        )

    # --- gas outlet port on +Z (top) face ---
    gas_port_d = _mm(gc.gas_port_diameter)
    plate = (
        plate.faces(">Z")
        .workplane()
        .pushPoints([(0, inner / 2 - _mm(gc.gas_port_offset_top))])
        .hole(gas_port_d, _mm(sc.wall_thickness))
    )

    # --- liquid flow ports (cross-flow: +X / -X) ---
    port_d = _mm(sc.flow_port_diameter)
    plate = (
        plate.faces("<X")
        .workplane()
        .hole(port_d, _mm(sc.wall_thickness))
    )
    plate = (
        plate.faces(">X")
        .workplane()
        .hole(port_d, _mm(sc.wall_thickness))
    )

    # --- current-collector rod passages ---
    cc = config.current_collector
    for x, y in config.collector_positions:
        plate = (
            plate.faces(">Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(_mm(cc.clearance_hole_diameter), total_depth)
        )

    return plate
