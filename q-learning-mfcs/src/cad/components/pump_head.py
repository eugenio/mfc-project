"""Peristaltic pump head block (visual representation).

Simple rectangular body with:
- Inlet/outlet ports on opposite faces (along X-axis)
- 4 corner mounting holes on base (-Z face)
No rotor detail — this is a visual placeholder.
"""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the pump head block, centred at origin."""
    spec = config.pump_head
    w = _mm(spec.body_width)
    d = _mm(spec.body_depth)
    h = _mm(spec.body_height)
    port_d = _mm(spec.port_diameter)
    port_sp = _mm(spec.port_spacing)
    mount_d = _mm(spec.mounting_hole_diameter)
    mount_sp = _mm(spec.mounting_hole_spacing)

    # Main body block
    result = cq.Workplane("XY").box(w, d, h)

    # Inlet port (-X face, centred on YZ)
    inlet = (
        cq.Workplane("YZ")
        .transformed(offset=(0, 0, -w / 2))
        .circle(port_d / 2)
        .extrude(w)
    )
    result = result.cut(inlet)

    # Outlet port (+X face, offset by port_spacing along Y)
    outlet = (
        cq.Workplane("YZ")
        .transformed(offset=(0, 0, -w / 2))
        .circle(port_d / 2)
        .extrude(w)
    )
    result = result.cut(outlet)

    # 4 corner mounting holes through base (-Z to +Z)
    mx = mount_sp / 2
    my = mount_sp / 2
    for sx in (-1, 1):
        for sy in (-1, 1):
            hole = (
                cq.Workplane("XY")
                .transformed(offset=(sx * mx, sy * my, -h / 2))
                .circle(mount_d / 2)
                .extrude(h)
            )
            result = result.cut(hole)

    return result
