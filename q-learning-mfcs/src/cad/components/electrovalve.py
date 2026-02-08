"""3-way solenoid electrovalve component.

Rectangular body with solenoid cylinder on top.
3 port bosses on different faces (inlet, outlet_a, outlet_b).
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
    """Build electrovalve, centred at origin."""
    spec = config.electrovalve
    bw = _mm(spec.body_width)
    bd = _mm(spec.body_depth)
    bh = _mm(spec.body_height)
    sol_d = _mm(spec.solenoid_diameter)
    sol_h = _mm(spec.solenoid_height)
    port_d = _mm(spec.port_diameter)
    boss_l = _mm(spec.port_boss_length)

    # Valve body
    result = cq.Workplane("XY").box(bw, bd, bh)

    # Solenoid cylinder on top
    solenoid = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, bh / 2))
        .circle(sol_d / 2)
        .extrude(sol_h)
    )
    result = result.union(solenoid)

    # Port bosses: inlet (-X face), outlet_a (+X face), outlet_b (-Y face)
    port_defs = [
        ((-bw / 2 - boss_l, 0, 0), (1, 0, 0)),   # inlet
        ((bw / 2, 0, 0), (1, 0, 0)),               # outlet_a
        ((0, -bd / 2 - boss_l, 0), (0, 1, 0)),     # outlet_b
    ]
    for (px, py, pz), (dx, dy, dz) in port_defs:
        boss = (
            cq.Workplane("XY")
            .transformed(offset=(px, py, pz))
            .circle(port_d / 2 + 2)
            .extrude(boss_l if dx == 0 and dy == 0 else boss_l)
        )
        # Rotate boss to face outward
        if dx != 0:
            boss = (
                cq.Workplane("YZ")
                .transformed(offset=(0, pz, px))
                .circle(port_d / 2 + 2)
                .extrude(boss_l)
            )
        elif dy != 0:
            boss = (
                cq.Workplane("XZ")
                .transformed(offset=(px, pz, py))
                .circle(port_d / 2 + 2)
                .extrude(boss_l)
            )
        result = result.union(boss)

        # Bore through boss
        if dx != 0:
            bore = (
                cq.Workplane("YZ")
                .transformed(offset=(0, pz, px))
                .circle(port_d / 2)
                .extrude(boss_l + bw / 2)
            )
        elif dy != 0:
            bore = (
                cq.Workplane("XZ")
                .transformed(offset=(px, pz, py))
                .circle(port_d / 2)
                .extrude(boss_l + bd / 2)
            )
        else:
            bore = (
                cq.Workplane("XY")
                .transformed(offset=(px, py, pz))
                .circle(port_d / 2)
                .extrude(boss_l + bh / 2)
            )
        result = result.cut(bore)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="electrovalve")  # type: ignore[name-defined]
