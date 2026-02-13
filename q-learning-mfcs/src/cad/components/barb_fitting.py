"""Hose barb fitting for tubing connections.

Structure (bottom to top along Z):
  1. Thread cylinder (bore-diameter at OD of port)
  2. Hex section (wrench flats)
  3. Barb cylinder (OD for hose grip)
All bored through with bore_diameter.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import BarbFittingSpec, StackCADConfig


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
        .polygon(6, hex_af / (math.sqrt(3) / 2))  # circumradius from across-flats
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


def fitting_length(config: StackCADConfig) -> float:
    """Total fitting length in mm."""
    spec = config.barb_fitting
    return _mm(spec.thread_length + spec.hex_height + spec.barb_length)


def fitting_length_from_spec(spec: BarbFittingSpec) -> float:
    """Total fitting length in mm from a BarbFittingSpec."""
    return _mm(spec.thread_length + spec.hex_height + spec.barb_length)


def build_oriented(
    config: StackCADConfig,
    normal: tuple[float, float, float],
) -> cq.Workplane:
    """Build a barb fitting rotated from Z-axis to the given normal.

    Parameters
    ----------
    config : StackCADConfig
    normal : tuple[float, float, float]
        Unit outward normal vector indicating the fitting direction.

    Returns
    -------
    cq.Workplane
        Fitting oriented along the given normal, centred at origin.

    """
    fitting = build(config)

    nx, ny, nz = normal
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-9:
        return fitting

    nx, ny, nz = nx / length, ny / length, nz / length

    # Rotation from Z-axis (0,0,1) to normal
    if abs(nz - 1.0) < 1e-9:
        return fitting  # already along +Z
    if abs(nz + 1.0) < 1e-9:
        # Flip 180 degrees around X
        return fitting.rotate((0, 0, 0), (1, 0, 0), 180)

    # Cross product of Z with normal gives rotation axis
    ax = -ny
    ay = nx
    az = 0.0
    al = math.sqrt(ax * ax + ay * ay + az * az)
    if al < 1e-9:
        return fitting
    ax, ay, az = ax / al, ay / al, az / al

    angle = math.degrees(math.acos(max(-1.0, min(1.0, nz))))
    return fitting.rotate((0, 0, 0), (ax, ay, az), angle)


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="barb_fitting")  # type: ignore[name-defined]
