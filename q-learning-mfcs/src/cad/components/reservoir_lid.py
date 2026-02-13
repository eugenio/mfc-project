"""Air-tight reservoir lid with gasket groove and port holes.

Flat disc matching reservoir OD with:
- Gasket groove on underside
- Central motor shaft hole with seal boss
- Feed port holes with nipple bosses for barb fittings
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
    """Build reservoir lid, centred at XY origin, bottom face at Z=0."""
    lid = config.reservoir_lid
    res = reservoir_spec or config.reservoir
    od = _mm(res.outer_diameter)
    thick = _mm(lid.thickness)
    groove_d = _mm(lid.gasket_groove_depth)
    groove_w = _mm(lid.gasket_groove_width)
    motor_d = _mm(lid.motor_hole_diameter)
    seal_h = _mm(lid.seal_boss_height)
    feed_d = _mm(lid.feed_port_diameter)
    nipple_h = _mm(lid.nipple_boss_height)

    # Main disc
    result = cq.Workplane("XY").circle(od / 2).extrude(thick)

    # Gasket groove on underside (annular groove at mid-radius)
    groove_radius = _mm(res.inner_diameter) / 2 + groove_w / 2
    groove_ring = (
        cq.Workplane("XY")
        .circle(groove_radius + groove_w / 2)
        .circle(groove_radius - groove_w / 2)
        .extrude(groove_d)
    )
    result = result.cut(groove_ring)

    # Motor shaft hole (through centre)
    motor_hole = cq.Workplane("XY").circle(motor_d / 2).extrude(thick)
    result = result.cut(motor_hole)

    # Seal boss on top around motor hole
    seal_boss = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, thick))
        .circle(motor_d / 2 + _mm(lid.seal_boss_wall))
        .circle(motor_d / 2)
        .extrude(seal_h)
    )
    result = result.union(seal_boss)

    # Feed port holes with nipple bosses
    port_radius = _mm(res.inner_diameter) / 4  # offset from centre
    for i in range(lid.feed_port_count):
        angle = 2 * math.pi * i / lid.feed_port_count + math.pi / 4
        px = port_radius * math.cos(angle)
        py = port_radius * math.sin(angle)

        # Bore through lid
        bore = (
            cq.Workplane("XY")
            .transformed(offset=(px, py, 0))
            .circle(feed_d / 2)
            .extrude(thick)
        )
        result = result.cut(bore)

        # Nipple boss on top
        boss = (
            cq.Workplane("XY")
            .transformed(offset=(px, py, thick))
            .circle(feed_d / 2 + _mm(lid.nipple_boss_wall))
            .extrude(nipple_h)
        )
        result = result.union(boss)

        # Bore through boss
        boss_bore = (
            cq.Workplane("XY")
            .transformed(offset=(px, py, thick))
            .circle(feed_d / 2)
            .extrude(nipple_h)
        )
        result = result.cut(boss_bore)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="reservoir_lid")  # type: ignore[name-defined]
