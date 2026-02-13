"""Pressure/velocity contour plots on cross-sections."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_contour_slice(
    field_data: dict[str, Any],
    normal: tuple[float, float, float] = (0, 0, 1),
    origin: tuple[float, float, float] | None = None,
    scalar_key: str = "magnitude",
) -> dict[str, Any] | None:
    """Compute a contour slice through a scalar field.

    Parameters
    ----------
    field_data : dict
        Field data with x, y, z and scalar arrays.
    normal : tuple
        Slice plane normal.
    origin : tuple, optional
        Slice plane origin. Defaults to field centroid.
    scalar_key : str
        Name of the scalar field to contour.

    Returns
    -------
    dict or None
        Slice data with 'points' and 'scalars' arrays.

    """
    import pyvista as pv  # noqa: PLC0415

    pv.OFF_SCREEN = True

    x = np.asarray(field_data["x"])
    if x.size == 0:
        return None
    y = np.asarray(field_data["y"])
    z = np.asarray(field_data["z"])
    scalars = np.asarray(field_data[scalar_key])

    points = np.column_stack([x, y, z])
    mesh = pv.PolyData(points)
    mesh[scalar_key] = scalars

    if origin is None:
        origin = tuple(points.mean(axis=0))

    try:
        sliced = mesh.slice(
            normal=normal, origin=origin,
        )
    except Exception:  # noqa: BLE001
        return None

    if sliced is None or sliced.n_points == 0:
        return None

    return {
        "points": sliced.points,
        "scalars": sliced[scalar_key],
    }
