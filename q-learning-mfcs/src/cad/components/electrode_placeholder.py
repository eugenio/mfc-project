"""Visual placeholder for the porous electrode (carbon felt).

A simple coloured box representing the electrode volume inside
the semi-cell chamber. Not intended for fabrication — purely
for visualisation in the assembly.
"""

from __future__ import annotations

import cadquery as cq

from ..cad_config import StackCADConfig


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build an electrode placeholder block."""
    e = config.electrode
    side = _mm(e.side_length)
    thick = _mm(e.thickness)
    return cq.Workplane("XY").box(side, side, thick)
