"""Back wall plate for sandwich semicell.

Flat polypropylene plate forming the outer structural wall of each
semicell. Features:
- Tie-rod clearance holes at four corners
- O-ring groove on the inner (sealing) face
- Flow distribution pocket on the inner face
- Diagonal flow ports drilled from lateral faces (-X / +X) into the pocket
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig
from cad.components.oring import compute_face_seal_groove


def _mm(m: float) -> float:
    return m * 1000.0


def build(
    config: StackCADConfig,
    is_anode: bool = True,
) -> cq.Workplane:
    """Build a back wall plate.

    Parameters
    ----------
    config : StackCADConfig
        Master parametric configuration.
    is_anode : bool
        ``True`` for anodic back wall, ``False`` for cathodic.
    """
    bw = config.back_wall
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    thickness = _mm(bw.thickness)

    # --- base plate ---
    plate = cq.Workplane("XY").box(outer, outer, thickness)

    # --- O-ring groove on inner face (+Z) ---
    face_groove = compute_face_seal_groove(config.face_oring)
    gd = _mm(face_groove.depth)
    inner = _mm(sc.inner_side)
    groove_offset = _mm(bw.groove_offset)
    groove_side = inner + groove_offset * 2
    gw = _mm(face_groove.width)

    # Cut a rectangular ring groove (outer rect minus inner rect)
    plate = (
        plate.faces(">Z")
        .workplane()
        .rect(groove_side + gw, groove_side + gw)
        .rect(groove_side - gw, groove_side - gw)
        .cutBlind(-gd)
    )

    # --- tie-rod clearance holes ---
    tr = config.tie_rod
    for x, y in config.tie_rod_positions:
        plate = (
            plate.faces(">Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(_mm(tr.clearance_hole_diameter), thickness)
        )

    # --- flow distribution pocket on inner face (+Z) ---
    pocket_d = _mm(bw.pocket_depth)
    if pocket_d > 0:
        plate = (
            plate.faces(">Z")
            .workplane()
            .rect(inner, inner)
            .cutBlind(-pocket_d)
        )

    # --- flow ports (horizontal bores from lateral faces) ---
    port_r = _mm(sc.flow_port_diameter) / 2
    inlet, outlet = config.port_positions(is_anode=is_anode)
    drill_len = _mm(sc.wall_thickness) + 1.0  # through wall into pocket

    # Port at plate centre (z=0), fully embedded, clear of groove
    port_z = 0.0

    half_outer = outer / 2
    half_inner = inner / 2

    for face_label, y_m in (inlet, outlet):
        y_mm = _mm(y_m)
        if face_label == "-X":
            start = cq.Vector(-half_outer, y_mm, port_z)
            direction = cq.Vector(1, 0, 0)
        else:
            start = cq.Vector(half_outer, y_mm, port_z)
            direction = cq.Vector(-1, 0, 0)
        bore = cq.Solid.makeCylinder(
            port_r, drill_len, pnt=start, dir=direction,
        )
        plate = plate.cut(cq.Workplane(bore))

    return plate


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig(), is_anode=True), name="back_wall")  # type: ignore[name-defined]
