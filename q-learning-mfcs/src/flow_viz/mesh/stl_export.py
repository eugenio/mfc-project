"""CadQuery solid -> STL export for internal flow volumes.

This module runs in the **cad-dev** environment (Python 3.10-3.11,
CadQuery available). STL files are the serialization format between
the CadQuery and gmsh environments.
"""

from __future__ import annotations

from pathlib import Path


def export_reservoir_stl(
    output_path: Path | str,
    volume_liters: float = 10.0,
    wall_thickness: float = 0.003,  # noqa: ARG001
    aspect_ratio: float = 2.0,
) -> Path:
    """Export reservoir inner cavity as STL.

    Creates a cylinder representing the inner volume
    of the anolyte reservoir.
    """
    import math  # noqa: PLC0415

    import cadquery as cq  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vol_m3 = volume_liters * 1e-3
    d_cubed = vol_m3 / (math.pi / 4 * aspect_ratio)
    inner_d = d_cubed ** (1.0 / 3.0)
    inner_h = aspect_ratio * inner_d

    # Inner cavity: simple cylinder (negative space)
    inner_d_mm = inner_d * 1000
    inner_h_mm = inner_h * 1000

    cavity = (
        cq.Workplane("XY")
        .circle(inner_d_mm / 2)
        .extrude(inner_h_mm)
    )

    cq.exporters.export(cavity, str(output_path), exportType="STL")
    return output_path


def export_chamber_stl(
    output_path: Path | str,
    inner_side: float = 0.10,
    depth: float = 0.025,
    chamber_type: str = "anode",  # noqa: ARG001
) -> Path:
    """Export a semi-cell chamber inner volume as STL.

    Creates a rectangular box representing the inner
    chamber volume (negative space).
    """
    import cadquery as cq  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    side_mm = inner_side * 1000
    depth_mm = depth * 1000

    chamber = (
        cq.Workplane("XY")
        .rect(side_mm, side_mm)
        .extrude(depth_mm)
    )

    cq.exporters.export(chamber, str(output_path), exportType="STL")
    return output_path


def export_tubing_stl(
    output_path: Path | str,
    inner_diameter: float = 0.008,
    length: float = 0.1,
) -> Path:
    """Export tubing inner bore as STL cylinder.

    Creates a cylinder representing the inner bore
    of a straight tubing segment.
    """
    import cadquery as cq  # noqa: PLC0415

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r_mm = (inner_diameter / 2) * 1000
    l_mm = length * 1000

    bore = (
        cq.Workplane("XY")
        .circle(r_mm)
        .extrude(l_mm)
    )

    cq.exporters.export(bore, str(output_path), exportType="STL")
    return output_path
