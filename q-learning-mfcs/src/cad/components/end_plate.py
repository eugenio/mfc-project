"""End plates (inlet and outlet).

Thick polypropylene plates at each end of the stack.
Features:
- Recessed nut pockets on the outer face for tie-rod nuts/washers
- O-ring groove on the inner (sealing) face
- Flow ports with G1/4 threaded bore
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


def build(config: StackCADConfig, is_inlet: bool = True) -> cq.Workplane:
    """Build an end plate.

    Parameters
    ----------
    config : StackCADConfig
        Master parametric configuration.
    is_inlet : bool
        ``True`` for inlet end plate, ``False`` for outlet.
    """
    ep = config.end_plate
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    thickness = _mm(ep.thickness)

    # --- base plate ---
    plate = cq.Workplane("XY").box(outer, outer, thickness)

    # --- O-ring groove on inner face (+Z) ---
    face_groove = compute_face_seal_groove(config.face_oring)
    gd = _mm(face_groove.depth)
    inner = _mm(sc.inner_side)
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
            .hole(_mm(tr.clearance_hole_diameter), thickness)
        )

    # --- recessed nut pockets on outer face (-Z) ---
    nut_pocket_d = _mm(tr.nut_af) * 1.15  # slight clearance on hex AF
    pocket_depth = _mm(ep.nut_pocket_depth)
    for x, y in config.tie_rod_positions:
        plate = (
            plate.faces("<Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(nut_pocket_d, pocket_depth)
        )

    # --- flow ports (G1/4 bore through plate centre) ---
    port_d = _mm(sc.flow_port_diameter)
    flow_dir = 1 if is_inlet else -1
    plate = (
        plate.faces(">Z")
        .workplane()
        .pushPoints([(0, flow_dir * inner / 4)])
        .hole(port_d, thickness)
    )

    return plate


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig(), is_inlet=True), name="end_plate")  # type: ignore[name-defined]
