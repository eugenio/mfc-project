"""Read OpenFOAM results using fluidfoam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_velocity_field(
    case_dir: Path | str,
    time_step: str = "latestTime",
) -> dict[str, Any]:
    """Read velocity field from OpenFOAM results.

    Parameters
    ----------
    case_dir : Path
        OpenFOAM case directory.
    time_step : str
        Time step to read. "latestTime" reads the last available.

    Returns
    -------
    dict
        Keys: x, y, z (coordinates), Ux, Uy, Uz (velocity components),
        magnitude (velocity magnitude).

    """
    from fluidfoam import readfield, readmesh  # noqa: PLC0415

    case_dir = Path(case_dir)
    sol = str(case_dir)

    if time_step == "latestTime":
        time_step = _find_latest_time(case_dir)

    x, y, z = readmesh(sol)
    u = readfield(sol, time_step, "U")

    mag = np.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    return {
        "x": x, "y": y, "z": z,
        "Ux": u[0], "Uy": u[1], "Uz": u[2],
        "magnitude": mag,
    }


def read_pressure_field(
    case_dir: Path | str,
    time_step: str = "latestTime",
) -> dict[str, Any]:
    """Read pressure field from OpenFOAM results."""
    from fluidfoam import readfield, readmesh  # noqa: PLC0415

    case_dir = Path(case_dir)
    sol = str(case_dir)

    if time_step == "latestTime":
        time_step = _find_latest_time(case_dir)

    x, y, z = readmesh(sol)
    p = readfield(sol, time_step, "p")

    return {
        "x": x, "y": y, "z": z,
        "p": p,
    }


def _find_latest_time(case_dir: Path) -> str:
    """Find the latest numerical time directory."""
    time_dirs = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append((t, d.name))
            except ValueError:
                continue

    if not time_dirs:
        return "0"

    time_dirs.sort(key=lambda x: x[0])
    return time_dirs[-1][1]
