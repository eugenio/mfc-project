"""3D text label plates for port identification.

Each label is a thin rectangular plate with embossed text.
Falls back to a plain plate if CadQuery text rendering fails.
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


def build_label(text: str, config: StackCADConfig) -> cq.Workplane:
    """Build a label plate with embossed text.

    Parameters
    ----------
    text : str
        Label text (e.g. "AN IN", "CA OUT").
    config : StackCADConfig
        Configuration for label dimensions.

    Returns
    -------
    cq.Workplane
        A thin plate, optionally with 3D text on top.

    """
    spec = config.port_label
    font_size = _mm(spec.font_size)
    text_depth = _mm(spec.text_depth)
    plate_thick = _mm(spec.plate_thickness)

    # Plate dimensions scale with text length
    char_count = max(len(text), 1)
    plate_width = font_size * char_count * 0.8 + font_size
    plate_height = font_size * 1.8

    # Base plate
    plate = (
        cq.Workplane("XY")
        .box(plate_width, plate_height, plate_thick)
        .translate((0, 0, plate_thick / 2))
    )

    if not text:
        return plate

    # Try to add 3D text
    try:
        text_solid = (
            cq.Workplane("XY")
            .transformed(offset=(0, 0, plate_thick))
            .text(text, font_size, text_depth)
        )
        plate = plate.union(text_solid)
    except Exception:
        # Font unavailable or text rendering fails — return plain plate
        pass

    return plate


def plate_dimensions(
    text: str,
    config: StackCADConfig,
) -> tuple[float, float]:
    """Return (plate_width_mm, plate_height_mm) for the given label text."""
    spec = config.port_label
    font_size = _mm(spec.font_size)
    char_count = max(len(text), 1)
    plate_width = font_size * char_count * 0.8 + font_size
    plate_height = font_size * 1.8
    return plate_width, plate_height


# -- CQ-Editor live preview ------------------------------------------------
if "show_object" in dir():
    show_object(build_label("AN IN", StackCADConfig()), name="port_label")  # type: ignore[name-defined]
