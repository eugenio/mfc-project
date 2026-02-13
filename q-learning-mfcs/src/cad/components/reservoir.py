"""Cylindrical anolyte reservoir vessel.

Hollowed cylinder with a flat base and open top (lid separate).
3 side ports at different heights:
  - Return (highest)
  - Feed (middle)
  - Drain (lowest)
"""

from __future__ import annotations

import math

import cadquery as cq

from ..cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the reservoir vessel, centred at XY origin, base at Z=0."""
    spec = config.reservoir
    od = _mm(spec.outer_diameter)
    oh = _mm(spec.outer_height)
    id_ = _mm(spec.inner_diameter)
    ih = _mm(spec.inner_height)
    wall = _mm(spec.wall_thickness)
    port_d = _mm(spec.port_diameter)

    # Outer cylinder
    outer = (
        cq.Workplane("XY")
        .circle(od / 2)
        .extrude(oh)
    )

    # Inner cavity (hollowed from top)
    inner = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, wall))
        .circle(id_ / 2)
        .extrude(ih)
    )
    result = outer.cut(inner)

    # Side ports at different heights
    port_heights = [
        wall + ih * 0.2,   # drain (lowest)
        wall + ih * 0.5,   # feed (middle)
        wall + ih * 0.8,   # return (highest)
    ]
    for pz in port_heights:
        port_hole = (
            cq.Workplane("XZ")
            .transformed(offset=(0, pz, od / 2))
            .circle(port_d / 2)
            .extrude(-wall * 2)
        )
        result = result.cut(port_hole)

    return result
