"""Electrode holder — grid frame pressing the electrode flat.

A polypropylene plate with a fine grid of open cells allowing
flow-through, while ribs provide mechanical support. Tie-rod
clearance holes at the four corners.

The *front* variant (between electrode and membrane holder)
includes strip pass-through slots at the +Y edge so Ti current-
collector tabs can exit the cell.
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


def build(config: StackCADConfig, *, is_front: bool = False) -> cq.Workplane:
    """Build an electrode holder grid frame.

    Parameters
    ----------
    config : StackCADConfig
        Master parametric configuration.
    is_front : bool
        If ``True``, cut strip pass-through slots at the +Y edge
        so Ti current-collector tabs can exit the cell.
    """
    eh = config.electrode_holder
    sc = config.semi_cell
    outer = _mm(sc.outer_side)
    thickness = _mm(eh.thickness)
    inner = _mm(sc.inner_side)
    rib = _mm(eh.rib_width)
    n = eh.grid_count

    # --- base plate ---
    plate = cq.Workplane("XY").box(outer, outer, thickness)

    # --- cut grid openings on +Z face ---
    # Cell size = (inner - (n-1)*rib) / n
    cell_size = (inner - (n - 1) * rib) / n
    half_inner = inner / 2

    for row in range(n):
        for col in range(n):
            cx = -half_inner + cell_size / 2 + col * (cell_size + rib)
            cy = -half_inner + cell_size / 2 + row * (cell_size + rib)
            plate = (
                plate.faces(">Z")
                .workplane()
                .pushPoints([(cx, cy)])
                .rect(cell_size, cell_size)
                .cutThruAll()
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

    # --- strip pass-through slots (front holder only) ---
    if is_front:
        strip_w = _mm(config.ti_strip.width)
        strip_t = _mm(config.ti_strip.thickness)
        wall_mm = _mm(sc.wall_thickness)
        slot_cy = half_inner + wall_mm / 2  # centre of +Y wall section

        for sx in [_mm(x) for x in config.ti_strip_x_positions]:
            plate = (
                plate.faces(">Z")
                .workplane()
                .pushPoints([(sx, slot_cy)])
                .rect(strip_w, wall_mm)
                .cutBlind(-strip_t)
            )

    return plate


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="electrode_holder")  # type: ignore[name-defined]
