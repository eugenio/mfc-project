"""CAD <-> simulator parameter bridge.

Maps between the parametric CAD configuration (``StackCADConfig``)
and the electrochemical model parameters used by ``odes.mojo``
(``MFCModel`` struct fields).

The simulator models a *single cell*; the CAD config describes
the full *N-cell stack*.  The bridge extracts per-cell geometric
quantities from the CAD config for use in the ODE model.
"""

from __future__ import annotations

import math
from typing import Any

from .cad_config import (
    ElectrodeDimensions,
    MembraneDimensions,
    SemiCellDimensions,
    StackCADConfig,
)

# Simulator default values (from odes.mojo __init__)
_SIM_DEFAULTS: dict[str, float] = {
    "V_a": 5.5e-5,  # m³  — anode volume (per cell)
    "V_c": 5.5e-5,  # m³  — cathode volume
    "A_m": 5.0e-4,  # m²  — membrane / electrode area
    "d_m": 1.778e-4,  # m   — membrane thickness
    "d_cell": 2.2e-2,  # m   — cell thickness (electrode spacing)
}

# Simulator default geometry (5 cm × 5 cm, 5.5 mL per chamber)
_SIM_ELECTRODE_SIDE = 0.05  # m  — derived: sqrt(A_m × 2) ≈ 0.032, but actual is 5 cm
_SIM_NUM_CELLS = 5


def cad_to_simulator(config: StackCADConfig) -> dict[str, float]:
    """Extract per-cell simulator parameters from a CAD config.

    Returns
    -------
    dict
        Keys match ``MFCModel`` struct field names:
        ``V_a``, ``V_c``, ``A_m``, ``d_m``, ``d_cell``, ``n_cells``.
    """
    return {
        "V_a": config.semi_cell.chamber_volume,
        "V_c": config.semi_cell.chamber_volume,
        "A_m": config.active_membrane_area,
        "d_m": config.membrane.thickness,
        "d_cell": config.cell_thickness,
        "n_cells": float(config.num_cells),
    }


def simulator_to_cad(
    params: dict[str, float],
    num_cells: int | None = None,
) -> StackCADConfig:
    """Build a ``StackCADConfig`` from simulator parameters.

    This is an *approximate* inverse: the simulator only stores
    a subset of the CAD geometry, so reasonable defaults are used
    for wall thickness, O-ring specs, etc.

    Parameters
    ----------
    params : dict
        Must include at least ``V_a`` (or ``V_c``), ``A_m``, ``d_m``.
        Optionally ``d_cell`` and ``n_cells``.
    num_cells : int, optional
        Override number of cells (default: from params or 10).
    """
    n = num_cells or int(params.get("n_cells", 10))
    A_m = params.get("A_m", _SIM_DEFAULTS["A_m"])
    V_a = params.get("V_a", _SIM_DEFAULTS["V_a"])
    d_m = params.get("d_m", _SIM_DEFAULTS["d_m"])

    # Derive electrode side length from membrane area (square)
    side = math.sqrt(A_m)

    # Derive chamber depth from volume and area
    depth = V_a / (side**2)

    electrode = ElectrodeDimensions(side_length=side, thickness=0.005)
    semi_cell = SemiCellDimensions(
        inner_side=side,
        depth=depth,
        wall_thickness=0.015,
    )
    membrane = MembraneDimensions(
        thickness=d_m,
        gasket_thickness=max(d_m * 2, 0.002),  # at least 2 mm
        active_side=side,
    )

    return StackCADConfig(
        num_cells=n,
        electrode=electrode,
        semi_cell=semi_cell,
        membrane=membrane,
    )


def round_trip_check(config: StackCADConfig) -> dict[str, tuple[float, float]]:
    """Verify round-trip consistency: CAD -> sim -> CAD.

    Returns a dict mapping parameter names to
    ``(original, round_tripped)`` pairs.
    """
    sim_params = cad_to_simulator(config)
    reconstructed = simulator_to_cad(sim_params, num_cells=config.num_cells)
    sim_back = cad_to_simulator(reconstructed)

    result: dict[str, tuple[float, float]] = {}
    for key in ("V_a", "V_c", "A_m", "d_m"):
        result[key] = (sim_params[key], sim_back[key])
    return result
