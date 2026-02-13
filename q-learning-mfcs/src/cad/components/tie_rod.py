"""Tie rod assembly: threaded rod + hex nuts + flat washers."""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build_rod(config: StackCADConfig) -> cq.Workplane:
    """Build a single tie rod (plain cylinder)."""
    tr = config.tie_rod
    return (
        cq.Workplane("XY")
        .circle(_mm(tr.diameter) / 2)
        .extrude(_mm(config.tie_rod_length))
    )


def build_nut(config: StackCADConfig) -> cq.Workplane:
    """Build a hex nut (inscribed polygon approximation)."""
    tr = config.tie_rod
    af = _mm(tr.nut_af)
    h = _mm(tr.nut_height)
    return (
        cq.Workplane("XY")
        .polygon(6, af)
        .extrude(h)
        .faces(">Z")
        .workplane()
        .hole(_mm(tr.diameter), h)
    )


def build_washer(config: StackCADConfig) -> cq.Workplane:
    """Build a flat washer."""
    tr = config.tie_rod
    od = _mm(tr.washer_od)
    th = _mm(tr.washer_thickness)
    return (
        cq.Workplane("XY")
        .circle(od / 2)
        .extrude(th)
        .faces(">Z")
        .workplane()
        .hole(_mm(tr.clearance_hole_diameter), th)
    )
