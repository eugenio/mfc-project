"""U-cradle support bracket for horizontal stack orientation.

Two feet elevate the stack off the ground plane.
The stack rests horizontally (Z-axis horizontal, resting on -Y face).

Shape: U-bracket = base plate + two vertical walls.
4 mounting holes in base plate.
"""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build a single U-cradle support foot.

    The foot is centred at the origin; assembly.py will position two
    copies at each end of the stack.
    """
    spec = config.support_feet
    w = _mm(spec.foot_width)       # Z-axis extent
    h = _mm(spec.foot_height)      # Y-axis extent (elevation)
    d = _mm(spec.foot_depth)       # X-axis extent (full cradle width)
    t = _mm(spec.wall_thickness)   # wall/base thickness
    hole_d = _mm(spec.mounting_hole_diameter)

    # Start with base plate on XZ plane
    result = (
        cq.Workplane("XZ")
        .box(d, w, t)  # base plate: X=d, Z=w, Y=t
        .translate((0, t / 2, 0))  # shift so bottom face at Y=0
    )

    # Left vertical wall
    left_wall = (
        cq.Workplane("XZ")
        .box(t, w, h)
        .translate((-d / 2 + t / 2, h / 2, 0))
    )

    # Right vertical wall
    right_wall = (
        cq.Workplane("XZ")
        .box(t, w, h)
        .translate((d / 2 - t / 2, h / 2, 0))
    )

    result = result.union(left_wall).union(right_wall)

    # Mounting holes in base plate (4 holes, symmetric pattern)
    hole_inset_x = d / 4
    hole_inset_z = w / 4
    hole_positions = [
        (-hole_inset_x, -hole_inset_z),
        (hole_inset_x, -hole_inset_z),
        (-hole_inset_x, hole_inset_z),
        (hole_inset_x, hole_inset_z),
    ]

    for hx, hz in hole_positions:
        hole = (
            cq.Workplane("XZ")
            .transformed(offset=(hx, t / 2, hz))
            .circle(hole_d / 2)
            .extrude(-t)
        )
        result = result.cut(hole)

    return result
