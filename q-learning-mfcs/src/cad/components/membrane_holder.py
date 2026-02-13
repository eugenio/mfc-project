"""Membrane holder — wire-grid frame.

Polypropylene frame with a fine wire grid spanning the active
membrane area. Features:
- Wire grid supporting the membrane (thin wires, large openings)
- O-ring groove on the membrane-facing side for sealing
- Tie-rod clearance holes at four corners
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig
from cad.components.oring import compute_rod_seal_groove


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build a membrane holder wire-grid frame."""
    mh = config.membrane_holder
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    thickness = _mm(mh.thickness)
    inner = _mm(sc.inner_side)
    wire = _mm(mh.wire_width)
    n = mh.grid_count

    # --- base plate ---
    plate = cq.Workplane("XY").box(outer, outer, thickness)

    # --- cut wire-grid openings (like electrode_holder but thinner wires) ---
    cell_size = (inner - (n - 1) * wire) / n
    half_inner = inner / 2

    for row in range(n):
        for col in range(n):
            cx = -half_inner + cell_size / 2 + col * (cell_size + wire)
            cy = -half_inner + cell_size / 2 + row * (cell_size + wire)
            plate = (
                plate.faces(">Z")
                .workplane()
                .pushPoints([(cx, cy)])
                .rect(cell_size, cell_size)
                .cutThruAll()
            )

    # --- O-ring groove on membrane-facing side (+Z) ---
    # Use smaller rod-seal O-ring (CS 1.78 mm) to fit thin plate
    rod_groove = compute_rod_seal_groove(config.rod_oring)
    gd = _mm(rod_groove.depth)
    gw = _mm(rod_groove.width)

    # Position groove between grid edge and tie-rod holes
    # Grid extends to ±half_inner; tie-rod nearest edge ~56 mm from centre
    tr_nearest = min(
        abs(x) for x, _y in config.tie_rod_positions
    )
    tr_edge = _mm(tr_nearest) - _mm(config.tie_rod.clearance_hole_diameter) / 2
    groove_centre = (half_inner + tr_edge) / 2
    groove_side = 2 * groove_centre

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

    return plate


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="membrane_holder")  # type: ignore[name-defined]
