"""Parallel-flow header pipe + tee junction.

``build_header`` creates a bored cylinder running the full stack length.
``build_tee`` creates a T-junction body with a perpendicular branch.
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


def build_header(config: StackCADConfig) -> cq.Workplane:
    """Build the manifold header pipe, centred at XY origin, along Z.

    The pipe runs from Z=0 to Z=stack_length (mm).
    """
    spec = config.manifold
    od = _mm(spec.header_od)
    id_ = _mm(spec.header_id)
    length = _mm(config.stack_length)

    outer = (
        cq.Workplane("XY")
        .circle(od / 2)
        .extrude(length)
    )
    inner = (
        cq.Workplane("XY")
        .circle(id_ / 2)
        .extrude(length)
    )
    return outer.cut(inner)


def build_tee(config: StackCADConfig) -> cq.Workplane:
    """Build a single T-junction body.

    Main bore along Z, perpendicular branch along +Y.
    Centred at origin; the branch extends in +Y direction.
    """
    spec = config.manifold
    hod = _mm(spec.header_od)
    hid = _mm(spec.header_id)
    bod = _mm(spec.branch_od)
    bid = _mm(spec.branch_id)

    # Main pipe section (short segment, ~2x header OD)
    main_len = hod * 2
    main_outer = (
        cq.Workplane("XY")
        .circle(hod / 2)
        .extrude(main_len)
        .translate((0, 0, -main_len / 2))
    )
    main_inner = (
        cq.Workplane("XY")
        .circle(hid / 2)
        .extrude(main_len)
        .translate((0, 0, -main_len / 2))
    )
    result = main_outer.cut(main_inner)

    # Branch perpendicular along +Y
    branch_len = hod * 1.5
    branch_outer = (
        cq.Workplane("XZ")
        .circle(bod / 2)
        .extrude(branch_len)
    )
    branch_inner = (
        cq.Workplane("XZ")
        .circle(bid / 2)
        .extrude(branch_len)
    )
    branch = branch_outer.cut(branch_inner)

    result = result.union(branch)

    # Bore through the intersection to connect main and branch
    bore = (
        cq.Workplane("XZ")
        .circle(bid / 2)
        .extrude(hod / 2 + branch_len)
        .translate((0, -hod / 2, 0))
    )
    result = result.cut(bore)

    return result


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    _cfg = StackCADConfig()
    show_object(build_header(_cfg), name="manifold_header")  # type: ignore[name-defined]
    show_object(build_tee(_cfg), name="manifold_tee")  # type: ignore[name-defined]
