"""Triangular support feet for reservoir conical bottom.

3 feet in triangular pattern around the conical bottom base.
Each foot: rectangular pad attached to clamping ring.
Clears the drain fitting at the lowest point.
"""

from __future__ import annotations

import math
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
    """Build feet assembly, centred at XY origin, top at Z=0."""
    feet_spec = config.reservoir_feet
    cb = config.conical_bottom
    res = reservoir_spec or config.reservoir

    foot_h = _mm(feet_spec.foot_height)
    foot_w = _mm(feet_spec.foot_width)
    foot_d = _mm(feet_spec.foot_depth)
    ring_t = _mm(feet_spec.ring_thickness)
    ring_h = _mm(feet_spec.ring_height)

    # Clamping ring around cylinder OD (not cone base)
    cyl_od = _mm(res.outer_diameter)
    ring_id = cyl_od
    ring_od = ring_id + 2 * ring_t
    result = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, -ring_h))
        .circle(ring_od / 2)
        .circle(ring_id / 2)
        .extrude(ring_h)
    )

    # Individual feet radiate outward from ring
    foot_radius = ring_od / 2 + foot_w / 2
    for i in range(feet_spec.foot_count):
        angle = 2 * math.pi * i / feet_spec.foot_count
        fx = foot_radius * math.cos(angle)
        fy = foot_radius * math.sin(angle)
        foot = (
            cq.Workplane("XY")
            .transformed(offset=(fx, fy, -ring_h - foot_h))
            .box(foot_w, foot_d, foot_h)
            .translate((0, 0, foot_h / 2))
        )
        result = result.union(foot)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="reservoir_feet")  # type: ignore[name-defined]
