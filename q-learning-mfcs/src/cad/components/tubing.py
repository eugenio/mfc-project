"""Parametric tubing segments — straight and U-tube.

Each builder returns ``(cq.Workplane, length_m)`` where *length_m*
is the centreline length in metres.

Approach: use CadQuery extrude along computed directions for solid
tube representation.  The tube OD is used (no hollow bore) for
visual segments to avoid boolean issues at junctions.  Dead volume
is computed analytically in ``hydraulics.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Euclidean distance between two 3D points (mm)."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _solid_cylinder_between(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    radius: float,
) -> cq.Workplane:
    """Create a solid cylinder from p1 to p2 (mm coords)."""
    edge = cq.Edge.makeLine(cq.Vector(*p1), cq.Vector(*p2))
    wire = cq.Wire.assembleEdges([edge])
    return (
        cq.Workplane("XY")
        .transformed(offset=p1)
        .circle(radius)
        .sweep(cq.Workplane().add(wire))
    )


def build_straight(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    config: StackCADConfig,
) -> tuple[cq.Workplane, float]:
    """Build a straight tube segment between two points (mm coordinates).

    The tube is represented as a solid outer cylinder with an inner
    bore subtracted.  For very short segments (<2× OD) only the outer
    cylinder is used.

    Returns
    -------
    tuple[cq.Workplane, float]
        (solid, centreline_length_m)
    """
    spec = config.tubing
    od = _mm(spec.outer_diameter)
    id_ = _mm(spec.inner_diameter)
    length_mm = _dist(start, end)
    length_m = length_mm / 1000.0

    outer = _solid_cylinder_between(start, end, od / 2)

    # Only bore if the segment is long enough for reliable boolean ops
    if length_mm > od * 2:
        try:
            inner = _solid_cylinder_between(start, end, id_ / 2)
            result = outer.cut(inner)
        except (ValueError, Exception):
            result = outer
    else:
        result = outer

    return result, length_m


def build_utube(
    port_a: tuple[float, float, float],
    port_b: tuple[float, float, float],
    clearance_mm: float,
    normal: tuple[float, float, float],
    config: StackCADConfig,
) -> tuple[cq.Workplane, float]:
    """Build a U-shaped tube segment between two ports.

    The tube exits port_a along ``normal`` by ``clearance_mm``,
    traverses to align with port_b, then re-enters.

    Parameters
    ----------
    port_a, port_b : 3-tuple
        Port positions in mm.
    clearance_mm : float
        How far the tube extends from the stack face.
    normal : 3-tuple
        Outward normal direction from the stack face.
    config : StackCADConfig

    Returns
    -------
    tuple[cq.Workplane, float]
        (solid, centreline_length_m)
    """
    spec = config.tubing
    od = _mm(spec.outer_diameter)

    nx, ny, nz = normal
    exit_a = (
        port_a[0] + nx * clearance_mm,
        port_a[1] + ny * clearance_mm,
        port_a[2] + nz * clearance_mm,
    )
    exit_b = (
        port_b[0] + nx * clearance_mm,
        port_b[1] + ny * clearance_mm,
        port_b[2] + nz * clearance_mm,
    )

    # Build 3 solid-OD cylinders and union them.
    # No inner bore on U-tube (visual representation; dead volume
    # is computed analytically in hydraulics.py).
    seg1 = _solid_cylinder_between(port_a, exit_a, od / 2)
    seg2 = _solid_cylinder_between(exit_a, exit_b, od / 2)
    seg3 = _solid_cylinder_between(exit_b, port_b, od / 2)

    result = seg1.union(seg2).union(seg3)

    # Centreline length
    length_mm = (
        _dist(port_a, exit_a)
        + _dist(exit_a, exit_b)
        + _dist(exit_b, port_b)
    )
    length_m = length_mm / 1000.0

    return result, length_m
