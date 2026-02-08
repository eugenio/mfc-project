"""Flat platform with feet for pump mounting.

4 cylindrical feet underneath, mounting holes matching pump pattern.
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
    """Build pump support platform, top surface at Z=0."""
    spec = config.pump_support
    pw = _mm(spec.platform_width)
    pd = _mm(spec.platform_depth)
    pt = _mm(spec.platform_thickness)
    fh = _mm(spec.foot_height)
    fd = _mm(spec.foot_diameter)
    mhd = _mm(spec.mounting_hole_diameter)
    mhs = _mm(spec.mounting_hole_spacing)

    # Platform slab
    result = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, -pt))
        .box(pw, pd, pt)
        .translate((0, 0, pt / 2))
    )

    # 4 feet at corners
    fx = pw / 2 - fd
    fy = pd / 2 - fd
    for sx in (-1, 1):
        for sy in (-1, 1):
            foot = (
                cq.Workplane("XY")
                .transformed(offset=(sx * fx, sy * fy, -pt - fh))
                .circle(fd / 2)
                .extrude(fh)
            )
            result = result.union(foot)

    # Mounting holes through platform
    mx = mhs / 2
    my = mhs / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            hole = (
                cq.Workplane("XY")
                .transformed(offset=(sx * mx, sy * my, -pt))
                .circle(mhd / 2)
                .extrude(pt)
            )
            result = result.cut(hole)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="pump_support")  # type: ignore[name-defined]
