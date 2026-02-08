"""Visual placeholder for the porous electrode (carbon felt).

A simple coloured box representing the electrode volume inside
the semi-cell chamber. Not intended for fabrication — purely
for visualisation in the assembly.
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
    """Build an electrode placeholder block."""
    e = config.electrode
    side = _mm(e.side_length)
    thick = _mm(e.thickness)
    return cq.Workplane("XY").box(side, side, thick)


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="electrode")  # type: ignore[name-defined]
