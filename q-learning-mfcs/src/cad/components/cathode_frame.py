"""Standard liquid cathode semi-cell frame plate.

Structurally identical to the anode frame but with flow ports
on the top/bottom faces for cross-flow configuration.
"""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig
from .oring import compute_face_seal_groove, compute_rod_seal_groove


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the liquid cathode frame plate.

    Same geometry as anode but flow ports on +X / -X faces
    (cross-flow relative to anode).
    """
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    depth = _mm(sc.depth)
    inner = _mm(sc.inner_side)

    # --- base plate ---
    plate = cq.Workplane("XY").box(outer, outer, depth)

    # --- chamber pocket ---
    plate = (
        plate.faces(">Z")
        .workplane()
        .rect(inner, inner)
        .cutBlind(-_mm(sc.depth - 0.002))
    )

    # --- O-ring groove ---
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
            .hole(_mm(tr.clearance_hole_diameter), depth)
        )

    # --- flow ports (cross-flow: on +X / -X walls) ---
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
            .hole(_mm(cc.clearance_hole_diameter), depth)
        )

    return plate
