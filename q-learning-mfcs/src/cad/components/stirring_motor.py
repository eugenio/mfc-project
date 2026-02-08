"""Overhead stirring motor assembly with shaft and impeller.

Motor body (cylinder) sits on top of reservoir lid.
Shaft extends downward through lid into reservoir.
Impeller: flat-blade Rushton-style (reliable CadQuery geometry).
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


def build(
    config: StackCADConfig,
    *,
    shaft_length_override: float | None = None,
) -> cq.Workplane:
    """Build motor + shaft + impeller, motor base at Z=0.

    Parameters
    ----------
    config : StackCADConfig
        Master parametric configuration.
    shaft_length_override : float | None
        If provided, use this shaft length (metres) instead of spec default.

    """
    spec = config.stirring_motor
    motor_d = _mm(spec.motor_diameter)
    motor_h = _mm(spec.motor_height)
    shaft_d = _mm(spec.shaft_diameter)
    actual_shaft = (
        shaft_length_override if shaft_length_override is not None
        else spec.shaft_length
    )
    shaft_l = _mm(actual_shaft)
    imp_d = _mm(spec.impeller_diameter)
    blade_w = _mm(spec.blade_width)
    blade_t = _mm(spec.blade_thickness)

    # Motor body cylinder
    result = cq.Workplane("XY").circle(motor_d / 2).extrude(motor_h)

    # Shaft extending downward from motor base
    shaft = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, -shaft_l))
        .circle(shaft_d / 2)
        .extrude(shaft_l)
    )
    result = result.union(shaft)

    # Impeller: Rushton-style flat blades at bottom of shaft
    # Hub disc
    hub_d = shaft_d * 2.5
    hub_h = blade_t * 2
    hub_z = -shaft_l + blade_w / 2
    hub = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, hub_z - hub_h / 2))
        .circle(hub_d / 2)
        .extrude(hub_h)
    )
    result = result.union(hub)

    # Flat blades (6 radial blades)
    n_blades = 6
    for i in range(n_blades):
        angle = 2 * math.pi * i / n_blades
        cx = (imp_d / 4 + hub_d / 4) * math.cos(angle)
        cy = (imp_d / 4 + hub_d / 4) * math.sin(angle)
        blade_len = imp_d / 2 - hub_d / 2
        blade = (
            cq.Workplane("XY")
            .transformed(offset=(cx, cy, hub_z - blade_w / 2))
            .transformed(rotate=(0, 0, math.degrees(angle)))
            .box(blade_len, blade_t, blade_w)
        )
        result = result.union(blade)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="stirring_motor")  # type: ignore[name-defined]
