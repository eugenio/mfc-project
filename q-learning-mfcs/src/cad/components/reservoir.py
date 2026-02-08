"""Cylindrical reservoir vessel with role-based features.

Supports multiple reservoir roles (anolyte, catholyte, nutrient, buffer)
with conical bottom, air-tight lid, side port fittings, and optional
gas diffusion port (catholyte).

All reservoirs are vertical cylinders with:
- Conical bottom section with drain fitting
- Air-tight lid with motor shaft hole and feed ports
- Side ports WITH barb fitting bosses and labels
- Stirring motor mount on lid
- Support feet at base

Catholyte reservoir additionally gets a gas diffusion port.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import ReservoirRole, ReservoirSpec, StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(
    config: StackCADConfig,
    role: ReservoirRole = ReservoirRole.ANOLYTE,
) -> cq.Workplane:
    """Build the reservoir vessel body (cylinder only).

    Centred at XY origin, base at Z=0.
    Use ``build_assembly`` for the full reservoir with lid, cone, feet.
    """
    spec = config.reservoir_spec_for_role(role)
    return _build_cylinder(config, spec, role)


def _build_cylinder(
    config: StackCADConfig,
    spec: ReservoirSpec,
    role: ReservoirRole,
) -> cq.Workplane:
    """Build cylindrical body with side ports and fitting bosses."""
    od = _mm(spec.outer_diameter)
    oh = _mm(spec.outer_height)
    id_ = _mm(spec.inner_diameter)
    ih = _mm(spec.inner_height)
    wall = _mm(spec.wall_thickness)
    port_d = _mm(spec.port_diameter)

    # Outer cylinder
    outer = cq.Workplane("XY").circle(od / 2).extrude(oh)

    # Inner cavity (hollowed from top)
    inner = (
        cq.Workplane("XY")
        .transformed(offset=(0, 0, wall))
        .circle(id_ / 2)
        .extrude(ih)
    )
    result = outer.cut(inner)

    # Side ports with fitting bosses at different heights
    port_heights = [
        wall + ih * 0.2,   # drain (lowest)
        wall + ih * 0.5,   # feed (middle)
        wall + ih * 0.8,   # return (highest)
    ]
    boss_od = port_d + 4  # 2mm wall around bore
    boss_len = _mm(config.barb_fitting.thread_length)

    for i, pz in enumerate(port_heights):
        # Distribute ports around circumference (120 degrees apart)
        angle = 2 * math.pi * i / len(port_heights)

        # Port hole through wall (radially outward)
        port_hole = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, pz))
            .transformed(rotate=(0, 0, math.degrees(angle)))
            .transformed(offset=(id_ / 2, 0, 0))
            .transformed(rotate=(0, 90, 0))
            .circle(port_d / 2)
            .extrude(wall + boss_len)
        )
        result = result.cut(port_hole)

        # Fitting boss (nipple) protruding radially outward
        boss = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, pz))
            .transformed(rotate=(0, 0, math.degrees(angle)))
            .transformed(offset=(od / 2, 0, 0))
            .transformed(rotate=(0, 90, 0))
            .circle(boss_od / 2)
            .extrude(boss_len)
        )
        result = result.union(boss)

        # Bore through boss (radially outward)
        bore = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, pz))
            .transformed(rotate=(0, 0, math.degrees(angle)))
            .transformed(offset=(od / 2, 0, 0))
            .transformed(rotate=(0, 90, 0))
            .circle(port_d / 2)
            .extrude(boss_len)
        )
        result = result.cut(bore)

    # Gas diffusion port for catholyte only
    if role == ReservoirRole.CATHOLYTE:
        gas = config.gas_diffusion
        gas_z = wall + _mm(gas.port_height_from_bottom)
        gas_angle = math.pi / 3  # 60 degrees, between existing ports
        gas_boss_d = _mm(gas.boss_diameter)
        gas_boss_l = _mm(gas.boss_length)
        gas_bore_d = _mm(gas.bore_diameter)

        # Gas port hole (radially outward)
        gas_hole = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, gas_z))
            .transformed(rotate=(0, 0, math.degrees(gas_angle)))
            .transformed(offset=(id_ / 2, 0, 0))
            .transformed(rotate=(0, 90, 0))
            .circle(gas_bore_d / 2)
            .extrude(wall + gas_boss_l)
        )
        result = result.cut(gas_hole)

        # Gas fitting boss (radially outward)
        gas_boss = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, gas_z))
            .transformed(rotate=(0, 0, math.degrees(gas_angle)))
            .transformed(offset=(od / 2, 0, 0))
            .transformed(rotate=(0, 90, 0))
            .circle(gas_boss_d / 2)
            .extrude(gas_boss_l)
        )
        result = result.union(gas_boss)

        # Bore through gas boss (radially outward)
        gas_bore = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, gas_z))
            .transformed(rotate=(0, 0, math.degrees(gas_angle)))
            .transformed(offset=(od / 2, 0, 0))
            .transformed(rotate=(0, 90, 0))
            .circle(gas_bore_d / 2)
            .extrude(gas_boss_l)
        )
        result = result.cut(gas_bore)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="reservoir")  # type: ignore[name-defined]
