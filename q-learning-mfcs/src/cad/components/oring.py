"""ISO 3601 O-ring groove geometry helpers.

Provides functions that compute groove cross-section dimensions and
CadQuery groove-cutting operations for rectangular face seals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..cad_config import ORingSpec, RodSealORingSpec


@dataclass(frozen=True)
class GrooveCrossSection:
    """Computed groove cross-section for machining / CAD modelling."""

    width: float  # m — groove width (radial or along seal face)
    depth: float  # m — groove depth into the sealing face
    corner_radius: float  # m — bottom-corner fillet radius
    volume_fill_ratio: float  # fraction of groove occupied by O-ring


def compute_face_seal_groove(spec: ORingSpec) -> GrooveCrossSection:
    """Compute groove cross-section for a rectangular face seal.

    ISO 3601-2 guidelines for static face seal, ~25 % compression.
    Volume fill ratio should be 70-85 % to allow thermal expansion.
    """
    cs = spec.cross_section_diameter
    depth = spec.groove_depth
    width = spec.groove_width
    corner_r = 0.2e-3  # 0.2 mm standard fillet

    oring_cross_area = math.pi * (cs / 2) ** 2
    groove_cross_area = width * depth
    fill = oring_cross_area / groove_cross_area if groove_cross_area > 0 else 0.0

    return GrooveCrossSection(
        width=width,
        depth=depth,
        corner_radius=corner_r,
        volume_fill_ratio=fill,
    )


def compute_rod_seal_groove(spec: RodSealORingSpec) -> GrooveCrossSection:
    """Compute groove for sealing a rod passing through a plate."""
    cs = spec.cross_section_diameter
    depth = spec.groove_depth
    width = spec.groove_width
    corner_r = 0.1e-3  # 0.1 mm fillet for smaller groove

    oring_cross_area = math.pi * (cs / 2) ** 2
    groove_cross_area = width * depth
    fill = oring_cross_area / groove_cross_area if groove_cross_area > 0 else 0.0

    return GrooveCrossSection(
        width=width,
        depth=depth,
        corner_radius=corner_r,
        volume_fill_ratio=fill,
    )


def rectangular_groove_path_length(inner_side: float, wall_offset: float) -> float:
    """Perimeter of a rectangular O-ring groove path.

    The groove runs around the chamber opening at *wall_offset* from
    the inner edge of the frame, with rounded corners (radius = wall_offset).

    Parameters
    ----------
    inner_side : float
        Chamber opening side length [m].
    wall_offset : float
        Distance from chamber edge to groove centre line [m].

    Returns
    -------
    float
        Groove centre-line perimeter [m].
    """
    straight = inner_side + 2 * wall_offset
    # Four quarter-circle corners → one full circle
    corner_circumference = 2 * math.pi * wall_offset
    return 4 * straight + corner_circumference - 8 * wall_offset
    # simplification: 4*(inner_side + 2*offset) - 8*offset + 2*pi*offset
    # = 4*inner_side + 2*pi*offset


def oring_count_per_cell(num_collector_rods: int) -> dict[str, int]:
    """Count O-rings needed for a single cell.

    Each cell has:
    - 2 face-seal O-rings (anode face + cathode face)
    - *num_collector_rods* × 2 rod-seal O-rings (one per rod per frame)

    Returns
    -------
    dict
        Keys: ``"face_seal"``, ``"rod_seal"``; values: counts per cell.
    """
    return {
        "face_seal": 2,
        "rod_seal": num_collector_rods * 2,
    }


def total_oring_count(
    num_cells: int,
    num_collector_rods: int,
    num_end_plates: int = 2,
) -> dict[str, int]:
    """Total O-ring count for the full stack.

    End plates each have one face-seal O-ring on the inner face.
    """
    per_cell = oring_count_per_cell(num_collector_rods)
    return {
        "face_seal": per_cell["face_seal"] * num_cells + num_end_plates,
        "rod_seal": per_cell["rod_seal"] * num_cells,
    }
