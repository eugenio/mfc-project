"""Conical frustum bottom section for reservoir vessels.

Frustum from reservoir inner diameter down to drain fitting diameter.
Includes drain boss extending downward from apex with tube fitting bore.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import ReservoirSpec, StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(
    config: StackCADConfig,
    reservoir_spec: ReservoirSpec | None = None,
) -> cq.Workplane:
    """Build the conical bottom section, base at Z=0, opening at Z=cone_height.

    Parameters
    ----------
    config : StackCADConfig
    reservoir_spec : ReservoirSpec, optional
        Override reservoir spec (for different reservoir roles).

    """
    cb = config.conical_bottom
    res = reservoir_spec or config.reservoir
    wall = _mm(res.wall_thickness)

    top_od = _mm(res.inner_diameter)
    bottom_od = _mm(cb.drain_fitting_diameter) + 2 * wall
    cone_h = _mm(cb.cone_height)
    drain_d = _mm(cb.drain_fitting_diameter)
    boss_l = _mm(cb.drain_boss_length)

    # Outer frustum (solid of revolution)
    top_r = top_od / 2 + wall
    bottom_r = bottom_od / 2
    result = (
        cq.Workplane("XZ")
        .moveTo(0, cone_h)
        .lineTo(top_r, cone_h)
        .lineTo(bottom_r, 0)
        .lineTo(0, 0)
        .close()
        .revolve(360, (0, 0, 0), (0, 1, 0))
    )

    # Inner frustum cavity
    inner_top_r = top_od / 2
    inner_bottom_r = drain_d / 2
    inner = (
        cq.Workplane("XZ")
        .moveTo(0, cone_h)
        .lineTo(inner_top_r, cone_h)
        .lineTo(inner_bottom_r, wall)
        .lineTo(0, wall)
        .close()
        .revolve(360, (0, 0, 0), (0, 1, 0))
    )
    result = result.cut(inner)

    # Drain boss extending downward
    boss = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, -boss_l))
        .circle(bottom_r)
        .extrude(boss_l)
    )
    result = result.union(boss)

    # Drain bore through boss and bottom
    bore = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, -boss_l))
        .circle(drain_d / 2)
        .extrude(boss_l + wall + 1)  # through bottom wall
    )
    result = result.cut(bore)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="conical_bottom")  # type: ignore[name-defined]
