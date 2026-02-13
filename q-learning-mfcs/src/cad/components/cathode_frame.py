"""Standard liquid cathode semi-cell frame plate.

Structurally identical to the anode frame but with flow ports
on the top/bottom faces for cross-flow configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cadquery as cq

from cad.cad_config import StackCADConfig
from cad.components.anode_frame import _build_frame


def _mm(m: float) -> float:
    return m * 1000.0


def build(config: StackCADConfig) -> cq.Workplane:
    """Build the liquid cathode frame plate.

    Same geometry as anode but flow ports on +Y / -Y faces
    (cross-flow relative to anode).
    """
    return _build_frame(
        config,
        port_faces=("<Y", ">Y"),
        collector_positions=config.cathode_collector_positions,
    )


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build(StackCADConfig()), name="cathode_frame")  # type: ignore[name-defined]
