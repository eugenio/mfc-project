"""Titanium strip current collector.

Thin Ti Grade 2 strip running along the Y-axis, sandwiched between
the electrode and the front electrode holder. The strip protrudes
from the +Y side of the frame for external electrical connection.
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
    """Build a single Ti current-collector strip.

    Strip oriented along Y-axis: width (X) × length (Y) × thickness (Z).
    Length = electrode_side + wall_thickness + protrusion.
    """
    ts = config.ti_strip
    length = _mm(config.ti_strip_length)
    width = _mm(ts.width)
    thickness = _mm(ts.thickness)
    return cq.Workplane("XY").box(width, length, thickness)


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="ti_strip")  # type: ignore[name-defined]
