"""Generate streamlines from velocity field data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import types


def compute_streamlines(
    velocity_data: dict[str, Any],
    n_points: int = 50,
    max_length: float = 500.0,
) -> list[dict[str, Any]]:
    """Compute streamlines from a velocity field.

    Parameters
    ----------
    velocity_data : dict
        Velocity field with x, y, z, Ux, Uy, Uz keys.
    n_points : int
        Number of seed points.
    max_length : float
        Maximum streamline length (mm).

    Returns
    -------
    list[dict]
        List of streamlines, each with 'points' (Nx3 array)
        and 'velocities' (N array of magnitudes).

    """
    pv = _import_pyvista()

    x = np.asarray(velocity_data["x"])
    if x.size == 0:
        return []
    y = np.asarray(velocity_data["y"])
    z = np.asarray(velocity_data["z"])
    ux = np.asarray(velocity_data["Ux"])
    uy = np.asarray(velocity_data["Uy"])
    uz = np.asarray(velocity_data["Uz"])

    points = np.column_stack([x, y, z])
    vectors = np.column_stack([ux, uy, uz])

    mesh = pv.PolyData(points)
    mesh["velocity"] = vectors
    mesh.set_active_vectors("velocity")

    # Create streamlines from evenly spaced seed points
    x_range = [x.min(), x.max()]
    y_range = [y.min(), y.max()]
    z_min = z.min()

    # Seed at inlet face
    seed_pts = np.column_stack([
        np.linspace(x_range[0] * 0.5, x_range[1] * 0.5, n_points),
        np.linspace(y_range[0] * 0.5, y_range[1] * 0.5, n_points),
        np.full(n_points, z_min),
    ])

    seed = pv.PolyData(seed_pts)

    try:
        streamlines = mesh.streamlines_from_source(
            seed,
            vectors="velocity",
            max_steps=2000,
            max_length=max_length,
            integration_direction="forward",
        )
    except Exception:  # noqa: BLE001
        return []

    # Extract individual lines
    result = []
    if streamlines.n_points > 0:
        pts = streamlines.points
        mag = np.linalg.norm(streamlines["velocity"], axis=1)
        result.append({
            "points": pts,
            "velocities": mag,
        })

    return result


def _import_pyvista() -> types.ModuleType:
    """Import pyvista with offscreen rendering."""
    import pyvista as pv  # noqa: PLC0415

    pv.OFF_SCREEN = True
    return pv  # type: ignore[no-any-return]
