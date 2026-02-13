"""Membrane gasket / frame.

A thin silicone frame with a central opening matching the
active membrane area. Tie-rod holes at 4 corners.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the membrane gasket frame."""
    mem = config.membrane
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    thickness = _mm(mem.gasket_thickness)
    opening = _mm(mem.active_side)

    # --- flat frame ---
    gasket = cq.Workplane("XY").box(outer, outer, thickness)

    # --- central opening ---
    gasket = (
        gasket.faces(">Z")
        .workplane()
        .rect(opening, opening)
        .cutThruAll()
    )

    # --- tie-rod clearance holes ---
    tr = config.tie_rod
    for x, y in config.tie_rod_positions:
        gasket = (
            gasket.faces(">Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(_mm(tr.clearance_hole_diameter), thickness)
        )

    # --- current-collector rod clearance holes ---
    cc = config.current_collector
    for x, y in config.collector_positions:
        gasket = (
            gasket.faces(">Z")
            .workplane()
            .pushPoints([(_mm(x), _mm(y))])
            .hole(_mm(cc.clearance_hole_diameter), thickness)
        )

    return gasket


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="membrane_gasket")  # type: ignore[name-defined]
