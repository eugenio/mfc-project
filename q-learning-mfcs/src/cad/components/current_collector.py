"""Titanium current-collector rod assembly.

A simple rod that passes through the porous electrode and
frame walls, sealed with small O-rings.
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
    """Build a single Ti current-collector rod."""
    cc = config.current_collector
    d = _mm(cc.diameter)
    length = _mm(cc.rod_length)

    rod = cq.Workplane("XY").circle(d / 2).extrude(length)

    # --- O-ring groove near each end (for wall sealing) ---
    groove = cc.seal_oring
    gw = _mm(groove.groove_width)
    gd = _mm(groove.groove_depth)
    groove_pos = _mm(0.005)  # 5 mm from each end

    for offset in [groove_pos, length - groove_pos - gw]:
        rod = (
            rod.faces(">Z")
            .workplane(offset=-(length - offset))
            .circle(d / 2 + 0.01)  # just outside rod surface
            .circle(d / 2 - gd)
            .cutBlind(-gw)
        )

    return rod


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="current_collector")  # type: ignore[name-defined]
