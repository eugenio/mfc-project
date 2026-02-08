"""Anode semi-cell frame plate.

A flat polypropylene plate with:
- Central rectangular pocket (chamber)
- O-ring groove around the chamber on the sealing face
- Tie-rod clearance holes at 4 corners
- Flow ports (inlet / outlet) on opposite edges
- Current-collector rod passage holes with seal grooves
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig
from cad.components.oring import compute_face_seal_groove, compute_rod_seal_groove

if TYPE_CHECKING:
    pass


def _mm(m: float) -> float:
    """Convert metres to millimetres (CadQuery default unit)."""
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the anode frame plate solid.

    Parameters
    ----------
    config : StackCADConfig
        Master parametric configuration.

    Returns
    -------
    cq.Workplane
        CadQuery solid of the anode frame.
    """
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    depth = _mm(sc.depth)
    inner = _mm(sc.inner_side)

    # --- base plate ---
    plate = cq.Workplane("XY").box(outer, outer, depth)

    # --- chamber pocket (centered, cut from +Z face) ---
    plate = (
        plate.faces(">Z")
        .workplane()
        .rect(inner, inner)
        .cutBlind(-_mm(sc.depth - 0.002))  # leave 2 mm floor
    )

    # --- O-ring groove on +Z sealing face ---
    face_groove = compute_face_seal_groove(config.face_oring)
    gw = _mm(face_groove.width)
    gd = _mm(face_groove.depth)
    # Groove path: rectangle centred on chamber opening, offset into the wall
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

    # --- flow ports (two on opposite edges, through the wall) ---
    port_d = _mm(sc.flow_port_diameter)
    # Inlet on -X wall
    plate = (
        plate.faces("<X")
        .workplane()
        .pushPoints([(0, 0)])
        .hole(port_d, _mm(sc.wall_thickness))
    )
    # Outlet on +X wall
    plate = (
        plate.faces(">X")
        .workplane()
        .pushPoints([(0, 0)])
        .hole(port_d, _mm(sc.wall_thickness))
    )

    # --- current-collector rod passages ---
    cc = config.current_collector
    rod_groove = compute_rod_seal_groove(cc.seal_oring)
    for x, y in config.anode_collector_positions:
        plate = (
            plate.faces(">Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(_mm(cc.clearance_hole_diameter), depth)
        )

    return plate


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="anode_frame")  # type: ignore[name-defined]
