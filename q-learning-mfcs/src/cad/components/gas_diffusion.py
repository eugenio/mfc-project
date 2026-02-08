"""Cylindrical porous gas diffusion element.

Mounted horizontally through reservoir wall.
External boss with bore for gas tube fitting.
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
    """Build gas diffusion element, oriented along X-axis, centred at origin."""
    spec = config.gas_diffusion
    elem_d = _mm(spec.element_diameter)
    elem_l = _mm(spec.element_length)
    boss_d = _mm(spec.boss_diameter)
    boss_l = _mm(spec.boss_length)
    bore_d = _mm(spec.bore_diameter)

    # Main cylindrical porous body (along X)
    result = (
        cq.Workplane("YZ")
        .circle(elem_d / 2)
        .extrude(elem_l)
    )

    # External boss for gas tube connection (extending beyond element)
    boss = (
        cq.Workplane("YZ")
        .transformed(offset=(0, 0, elem_l))
        .circle(boss_d / 2)
        .extrude(boss_l)
    )
    result = result.union(boss)

    # Gas bore through boss
    bore = (
        cq.Workplane("YZ")
        .transformed(offset=(0, 0, elem_l))
        .circle(bore_d / 2)
        .extrude(boss_l)
    )
    result = result.cut(bore)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="gas_diffusion")  # type: ignore[name-defined]
