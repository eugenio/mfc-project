"""OpenFOAM solver configuration presets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SolverType(Enum):
    """Available OpenFOAM solvers for MFC flow."""

    SIMPLE_FOAM = "simpleFoam"  # Steady-state
    ICO_FOAM = "icoFoam"  # Transient laminar


@dataclass(frozen=True)
class SolverConfig:
    """OpenFOAM solver settings for MFC simulation.

    MFC flows are very low Re (< 100), laminar regime.
    """

    solver: SolverType = SolverType.SIMPLE_FOAM
    end_time: float = 1000.0  # iterations (steady) or seconds (transient)
    write_interval: int = 100
    delta_t: float = 0.001  # seconds (transient only)
    num_processors: int = 1  # serial by default

    # Convergence
    residual_target: float = 1e-4
    relaxation_p: float = 0.3
    relaxation_u: float = 0.7

    # Discretization schemes
    div_scheme: str = "bounded Gauss linearUpwind grad(U)"
    grad_scheme: str = "Gauss linear"
    laplacian_scheme: str = "Gauss linear corrected"
