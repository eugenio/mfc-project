"""OpenFOAM boundary condition generation for MFC flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from flow_viz.config import FlowVizConfig

if TYPE_CHECKING:
    from cad.cad_config import StackCADConfig


@dataclass(frozen=True)
class BoundaryConditions:
    """Boundary conditions for an MFC flow simulation.

    Computed from the CAD and flow visualization configs.
    """

    inlet_velocity: float  # m/s
    outlet_pressure: float  # Pa (gauge)
    kinematic_viscosity: float  # m^2/s
    wall_type: str = "noSlip"


def compute_boundary_conditions(
    cad_config: StackCADConfig,
    viz_config: FlowVizConfig | None = None,
) -> BoundaryConditions:
    """Compute boundary conditions from MFC configuration.

    Inlet velocity = max_flow_rate / tubing cross-section area.
    """
    if viz_config is None:
        viz_config = FlowVizConfig()

    flow_rate = cad_config.pump_head.max_flow_rate  # m^3/s
    area = cad_config.tubing.cross_section_area  # m^2
    inlet_v = flow_rate / area

    kin_visc = viz_config.fluid_viscosity / viz_config.fluid_density

    return BoundaryConditions(
        inlet_velocity=inlet_v,
        outlet_pressure=0.0,
        kinematic_viscosity=kin_visc,
    )


def compute_reynolds_number(
    cad_config: StackCADConfig,
    viz_config: FlowVizConfig | None = None,
) -> float:
    """Compute Reynolds number for the MFC tubing flow."""
    bc = compute_boundary_conditions(cad_config, viz_config)
    d_h: float = cad_config.tubing.inner_diameter  # hydraulic diameter
    return bc.inlet_velocity * d_h / bc.kinematic_viscosity
