"""Hose barb fitting for tubing connections.

Structure (bottom to top along Z):
  1. Thread cylinder (bore-diameter at OD of port)
  2. Hex section (wrench flats)
  3. Barb cylinder (OD for hose grip)
All bored through with bore_diameter.
"""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build a single barb fitting, oriented along Z."""
    spec = config.barb_fitting
    bore = _mm(spec.bore_diameter)
    thread_l = _mm(spec.thread_length)
    hex_af = _mm(spec.hex_af)
    hex_h = _mm(spec.hex_height)
    barb_od = _mm(spec.barb_od)
    barb_l = _mm(spec.barb_length)

    z = 0.0

    # Thread cylinder (port OD, slightly larger than bore)
    thread_od = bore + 2  # 1 mm wall around bore
    result = (
        cq.Workplane("XY")
        .circle(thread_od / 2)
        .extrude(thread_l)
    )
    z += thread_l

    # Hex section
    hex_solid = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, z))
        .polygon(6, hex_af / 0.8660254)  # circumradius from across-flats
        .extrude(hex_h)
    )
    result = result.union(hex_solid)
    z += hex_h

    # Barb cylinder
    barb_solid = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, z))
        .circle(barb_od / 2)
        .extrude(barb_l)
    )
    result = result.union(barb_solid)

    # Bore through entire length
    total_length = thread_l + hex_h + barb_l
    bore_hole = (
        cq.Workplane("XY")
        .circle(bore / 2)
        .extrude(total_length)
    )
    result = result.cut(bore_hole)

    return result
