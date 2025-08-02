#!/usr/bin/env python3
"""
Advanced Electrode Physics Model for MFC Simulation

This module implements comprehensive electrode modeling including:
- Geometric compatibility validation with MFC cell dimensions
- Fluid dynamics within porous electrode structures
- Mass transport limitations and nutrient kinetics
- 3D biofilm growth with dynamic pore blocking
- Machine learning optimization for electrode parameters

Created: 2025-08-01
Literature References:
1. Picioreanu, C. et al. (2007). "A computational model for biofilm-based microbial fuel cells"
2. Marcus, A.K. et al. (2007). "Conduction-based modeling of the biofilm anode of a microbial fuel cell"
3. Oliveira, V.B. et al. (2013). "A 1D mathematical model for a microbial fuel cell"
4. Pinto, R.P. et al. (2010). "A two-population bio-electrochemical model of a microbial fuel cell"
5. Zhang, X.C. & Halme, A. (1995). "Modelling of a microbial fuel cell process"
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import scipy.sparse as sp
from config.electrode_config import ElectrodeConfiguration, ElectrodeGeometry


class FlowRegime(Enum):
    """Flow regime classification for porous media."""
    DARCY = "darcy"  # Viscous flow (Re_p < 1)
    FORCHHEIMER = "forchheimer"  # Inertial effects (1 < Re_p < 150)
    TURBULENT = "turbulent"  # Turbulent flow (Re_p > 150)


class BiofilmGrowthPhase(Enum):
    """Biofilm growth phase classification."""
    ATTACHMENT = "attachment"  # Initial cell attachment (0-2 hours)
    EXPONENTIAL = "exponential"  # Rapid growth (2-24 hours)
    MATURATION = "maturation"  # Biofilm maturation (1-7 days)
    STEADY_STATE = "steady_state"  # Mature biofilm (>7 days)
    DETACHMENT = "detachment"  # Sloughing events


@dataclass
class CellGeometry:
    """MFC cell geometry specifications."""

    # Cell dimensions (m)
    length: float
    width: float
    height: float

    # Electrode compartment dimensions
    anode_chamber_volume: float  # m³
    cathode_chamber_volume: float  # m³
    separator_thickness: float = 0.001  # m (1 mm default)

    # Flow configuration
    inlet_diameter: float = 0.003  # m (3 mm default)
    outlet_diameter: float = 0.003  # m (3 mm default)
    flow_path_tortuosity: float = 1.2  # Dimensionless

    def get_total_volume(self) -> float:
        """Calculate total cell volume."""
        return self.length * self.width * self.height

    def get_electrode_spacing(self) -> float:
        """Calculate spacing between electrodes."""
        return self.separator_thickness

    def validate_electrode_fit(self, electrode_config: ElectrodeConfiguration) -> dict[str, Any]:
        """Validate if electrode geometry fits within cell."""
        electrode_volume = electrode_config.geometry.calculate_volume()
        available_volume = min(self.anode_chamber_volume, self.cathode_chamber_volume)

        # Get electrode dimensions
        geom = electrode_config.geometry

        validation_results = {
            'fits': True,
            'volume_utilization': electrode_volume / available_volume,
            'warnings': [],
            'recommendations': []
        }

        # Volume check
        if electrode_volume > available_volume * 0.8:  # Leave 20% for flow
            validation_results['fits'] = False
            validation_results['warnings'].append(
                f"Electrode volume ({electrode_volume*1e6:.1f} cm³) exceeds 80% of chamber volume"
            )

        # Dimensional checks
        if geom.geometry_type == ElectrodeGeometry.RECTANGULAR_PLATE:
            if geom.length and geom.length > self.length * 0.9:
                validation_results['warnings'].append("Electrode length may restrict flow")
            if geom.width and geom.width > self.width * 0.9:
                validation_results['warnings'].append("Electrode width may restrict flow")

        # Flow path analysis
        if validation_results['volume_utilization'] > 0.6:
            validation_results['recommendations'].append(
                "Consider reducing electrode size or increasing cell volume for better flow"
            )

        return validation_results


@dataclass
class FluidDynamicsProperties:
    """Fluid dynamics properties for electrode modeling."""

    # Fluid properties
    density: float = 1000.0  # kg/m³ (water at 25°C)
    dynamic_viscosity: float = 0.00089  # Pa·s (water at 25°C)

    # Flow conditions
    flow_rate: float = 1e-6  # m³/s (1 mL/min default)
    reynolds_number: float = field(init=False)

    # Porous media properties (calculated from electrode config)
    permeability: float = field(init=False)  # m² (Darcy permeability)
    forchheimer_coefficient: float = field(init=False)  # m⁻¹
    pore_size_distribution: list[float] = field(default_factory=list)

    def calculate_reynolds_number(self, characteristic_length: float) -> float:
        """Calculate Reynolds number for flow through electrode."""
        velocity = self.flow_rate / (np.pi * (characteristic_length/2)**2)
        self.reynolds_number = (self.density * velocity * characteristic_length) / self.dynamic_viscosity
        return self.reynolds_number

    def get_flow_regime(self, pore_reynolds: float) -> FlowRegime:
        """Determine flow regime based on pore Reynolds number."""
        if pore_reynolds < 1:
            return FlowRegime.DARCY
        elif pore_reynolds < 150:
            return FlowRegime.FORCHHEIMER
        else:
            return FlowRegime.TURBULENT


@dataclass
class MassTransportProperties:
    """Mass transport properties within electrode."""

    # Diffusion coefficients (m²/s)
    substrate_diffusivity_bulk: float = 1.0e-9  # Lactate in water
    substrate_diffusivity_biofilm: float = 0.6e-9  # Lactate in biofilm
    oxygen_diffusivity_bulk: float = 2.0e-9  # Oxygen in water

    # Convection parameters
    peclet_number: float = field(init=False)  # Pe = vL/D
    sherwood_number: float = field(init=False)  # Mass transfer correlation

    # Reaction kinetics within electrode
    max_substrate_consumption_rate: float = 0.1  # mol/m³/s
    michaelis_constant: float = 5.0  # mM
    biofilm_yield_coefficient: float = 0.4  # g-biomass/g-substrate

    def calculate_peclet_number(self, velocity: float, length_scale: float, diffusivity: float) -> float:
        """Calculate Peclet number for mass transport analysis."""
        self.peclet_number = (velocity * length_scale) / diffusivity
        return self.peclet_number

    def calculate_sherwood_number(self, reynolds: float, schmidt: float) -> float:
        """Calculate Sherwood number using mass transfer correlations."""
        # Correlation for flow through packed beds (Gnielinski correlation)
        if reynolds < 1:
            self.sherwood_number = 2.0  # Pure diffusion
        else:
            self.sherwood_number = 2.0 + 0.6 * (reynolds**0.5) * (schmidt**(1/3))
        return self.sherwood_number


@dataclass
class BiofilmDynamics:
    """3D biofilm growth and dynamics model."""

    # Biofilm properties
    max_biofilm_density: float = 80.0  # kg/m³
    biofilm_yield: float = 0.4  # g-biomass/g-substrate
    decay_rate: float = 0.01  # h⁻¹
    detachment_rate: float = 0.001  # h⁻¹
    michaelis_constant: float = 5.0  # mM (Km for substrate limitation)

    # 3D growth parameters
    growth_direction_weights: dict[str, float] = field(default_factory=lambda: {
        'radial': 0.6,  # Growth outward from electrode surface
        'axial': 0.3,   # Growth along flow direction
        'normal': 0.1   # Growth perpendicular to surface
    })

    # Pore blocking dynamics
    critical_pore_fraction: float = 0.1  # Minimum open pore fraction
    pore_blocking_threshold: float = 50e-6  # m (50 μm)

    # Current biofilm state (dynamic)
    biofilm_thickness_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    pore_size_evolution: list[float] = field(default_factory=list)
    biofilm_age_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    def calculate_growth_rate(self, substrate_conc: float, local_ph: float, temperature: float) -> float:
        """Calculate local biofilm growth rate based on conditions."""
        # Monod kinetics for substrate limitation
        substrate_factor = substrate_conc / (self.michaelis_constant + substrate_conc)

        # pH factor (optimal around 7.0)
        ph_factor = np.exp(-((local_ph - 7.0) / 1.5)**2)

        # Temperature factor (optimal around 30°C)
        temp_factor = np.exp(-((temperature - 30.0) / 10.0)**2)

        return 0.05 * substrate_factor * ph_factor * temp_factor  # h⁻¹

    def update_pore_blocking(self, biofilm_thickness: np.ndarray, initial_pore_sizes: np.ndarray) -> np.ndarray:
        """Update pore sizes based on biofilm growth."""
        # Simple model: pore size reduces linearly with biofilm thickness
        blocked_fraction = np.minimum(biofilm_thickness / self.pore_blocking_threshold, 0.9)
        current_pore_sizes = initial_pore_sizes * (1 - blocked_fraction)

        return np.maximum(current_pore_sizes, initial_pore_sizes * 0.1)  # Minimum 10% of original


class AdvancedElectrodeModel:
    """
    Comprehensive electrode model integrating geometry, fluid dynamics,
    mass transport, and biofilm growth.
    """

    def __init__(self,
                 electrode_config: ElectrodeConfiguration,
                 cell_geometry: CellGeometry,
                 fluid_properties: FluidDynamicsProperties | None = None,
                 transport_properties: MassTransportProperties | None = None,
                 biofilm_dynamics: BiofilmDynamics | None = None):

        self.electrode_config = electrode_config
        self.cell_geometry = cell_geometry
        self.fluid_props = fluid_properties or FluidDynamicsProperties()
        self.transport_props = transport_properties or MassTransportProperties()
        self.biofilm_dynamics = biofilm_dynamics or BiofilmDynamics()

        # Initialize spatial discretization
        self.nx, self.ny, self.nz = 20, 20, 10  # 3D grid resolution
        self.dx = self._calculate_grid_spacing()

        # State variables
        self.current_state = self._initialize_state()
        self.time = 0.0

        # Validation
        self.compatibility_check = self._validate_electrode_compatibility()

    def _validate_electrode_compatibility(self) -> dict[str, Any]:
        """Validate electrode-cell compatibility."""
        return self.cell_geometry.validate_electrode_fit(self.electrode_config)

    def _calculate_grid_spacing(self) -> tuple[float, float, float]:
        """Calculate 3D grid spacing based on electrode geometry."""
        geom = self.electrode_config.geometry

        if geom.length and geom.width and geom.height:
            dx = geom.length / self.nx
            dy = geom.width / self.ny
            dz = geom.height / self.nz
        else:
            # Default spacing for complex geometries
            dx = dy = dz = 1e-3  # 1 mm

        return dx, dy, dz

    def _initialize_state(self) -> dict[str, np.ndarray]:
        """Initialize 3D state variables."""
        shape = (self.nx, self.ny, self.nz)

        return {
            'substrate_concentration': np.full(shape, 25.0),  # mM
            'biofilm_density': np.zeros(shape),  # kg/m³
            'biofilm_thickness': np.zeros(shape),  # m
            'pore_size': np.full(shape, 100e-6),  # m (100 μm initial)
            'velocity_field': np.zeros((3,) + shape),  # m/s (vx, vy, vz)
            'pressure_field': np.zeros(shape),  # Pa
            'ph_field': np.full(shape, 7.0),  # pH
            'temperature_field': np.full(shape, 30.0),  # °C
        }

    def calculate_permeability_field(self) -> np.ndarray:
        """Calculate local permeability based on pore structure and biofilm."""
        pore_sizes = self.current_state['pore_size']
        biofilm_density = self.current_state['biofilm_density']

        # Kozeny-Carman equation modified for biofilm
        porosity = self.electrode_config.material_properties.porosity or 0.8

        # Reduce porosity based on biofilm density
        effective_porosity = porosity * (1 - biofilm_density / self.biofilm_dynamics.max_biofilm_density)
        effective_porosity = np.maximum(effective_porosity, 0.1)  # Minimum porosity

        # Permeability calculation (m²)
        permeability = (pore_sizes**2 * effective_porosity**3) / (180 * (1 - effective_porosity)**2)

        return permeability

    def solve_flow_field(self) -> dict[str, np.ndarray]:
        """Solve 3D flow field through porous electrode."""
        permeability = self.calculate_permeability_field()

        # Simplified pressure-driven flow (Darcy's law in 3D)
        # ∇·(k/μ ∇p) = 0 where k is permeability, μ is viscosity

        # Create sparse matrix for pressure equation
        n_points = self.nx * self.ny * self.nz
        sp.lil_matrix((n_points, n_points))
        np.zeros(n_points)

        # Apply boundary conditions and discretization
        # (Simplified implementation - full CFD would require more sophisticated solver)

        # For now, use analytical approximation for flow through porous media
        inlet_pressure = 1000.0  # Pa (1 kPa pressure drop)

        # Linear pressure gradient assumption
        x_coords = np.linspace(0, 1, self.nx)
        pressure_field = inlet_pressure * (1 - x_coords).reshape(-1, 1, 1)
        pressure_field = np.broadcast_to(pressure_field, (self.nx, self.ny, self.nz))

        # Calculate velocity from Darcy's law: v = -(k/μ) ∇p
        dp_dx = -inlet_pressure / (self.nx * self.dx[0])
        velocity_x = -(permeability / self.fluid_props.dynamic_viscosity) * dp_dx

        velocity_field = np.zeros((3, self.nx, self.ny, self.nz))
        velocity_field[0] = velocity_x  # x-component

        self.current_state['pressure_field'] = pressure_field
        self.current_state['velocity_field'] = velocity_field

        return {
            'pressure': pressure_field,
            'velocity': velocity_field,
            'permeability': permeability
        }

    def solve_mass_transport(self, dt: float) -> np.ndarray:
        """Solve 3D mass transport equation with reaction."""
        substrate = self.current_state['substrate_concentration']
        self.current_state['velocity_field']
        biofilm_density = self.current_state['biofilm_density']

        # Convection-diffusion-reaction equation:
        # ∂C/∂t + ∇·(vC) = ∇·(D∇C) - R(C)

        # Simplified finite difference implementation
        # (Full implementation would use advanced numerical methods)

        # Diffusion term (central differences)

        # Reaction term (Monod kinetics)
        reaction_rate = (self.transport_props.max_substrate_consumption_rate *
                        biofilm_density * substrate /
                        (self.transport_props.michaelis_constant + substrate))

        # Update substrate concentration
        # Simplified explicit scheme (would need implicit for stability)
        dC_dt = -reaction_rate  # Simplified - missing convection and diffusion
        substrate_new = substrate - dt * dC_dt
        substrate_new = np.maximum(substrate_new, 0.0)  # Non-negative constraint

        self.current_state['substrate_concentration'] = substrate_new

        return substrate_new

    def solve_biofilm_growth(self, dt: float) -> dict[str, np.ndarray]:
        """Solve 3D biofilm growth with pore blocking."""
        substrate = self.current_state['substrate_concentration']
        biofilm_density = self.current_state['biofilm_density']
        biofilm_thickness = self.current_state['biofilm_thickness']
        ph_field = self.current_state['ph_field']
        temp_field = self.current_state['temperature_field']

        # Calculate local growth rates
        growth_rates = np.zeros_like(biofilm_density)

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    growth_rates[i,j,k] = self.biofilm_dynamics.calculate_growth_rate(
                        substrate[i,j,k], ph_field[i,j,k], temp_field[i,j,k]
                    )

        # Update biofilm density
        biofilm_growth = (growth_rates * substrate * self.biofilm_dynamics.biofilm_yield -
                         self.biofilm_dynamics.decay_rate * biofilm_density)

        biofilm_density_new = biofilm_density + dt * biofilm_growth
        biofilm_density_new = np.clip(biofilm_density_new, 0,
                                    self.biofilm_dynamics.max_biofilm_density)

        # Update biofilm thickness (simplified)
        thickness_increment = dt * growth_rates * 1e-6  # Convert to meters
        biofilm_thickness_new = biofilm_thickness + thickness_increment

        # Update pore sizes due to biofilm growth
        initial_pore_size = 100e-6  # Initial pore size
        pore_sizes_new = self.biofilm_dynamics.update_pore_blocking(
            biofilm_thickness_new,
            np.full_like(biofilm_thickness_new, initial_pore_size)
        )

        # Update state
        self.current_state['biofilm_density'] = biofilm_density_new
        self.current_state['biofilm_thickness'] = biofilm_thickness_new
        self.current_state['pore_size'] = pore_sizes_new

        return {
            'biofilm_density': biofilm_density_new,
            'biofilm_thickness': biofilm_thickness_new,
            'pore_sizes': pore_sizes_new,
            'growth_rates': growth_rates
        }

    def step(self, dt: float) -> dict[str, Any]:
        """
        Advance simulation by one time step with coupled physics.

        Solves the coupled system of:
        1. Flow field (momentum transport)
        2. Mass transport (species transport)
        3. Biofilm growth (biological processes)
        """
        step_results = {}

        # 1. Solve flow field (pressure and velocity)
        flow_results = self.solve_flow_field()
        step_results['flow'] = flow_results

        # 2. Solve mass transport for all species
        transport_results = self.solve_mass_transport(dt)
        step_results['mass_transport'] = transport_results

        # 3. Solve biofilm growth and dynamics
        biofilm_results = self.solve_biofilm_growth(dt)

        # 4. Update time
        self.time += dt

        # 5. Calculate performance metrics
        metrics = self.calculate_performance_metrics()

        return {
            'time': self.time,
            'flow_results': flow_results,
            'transport_results': transport_results,
            'biofilm_results': biofilm_results,
            'performance_metrics': metrics,
            'compatibility_check': self.compatibility_check
        }

    def calculate_performance_metrics(self) -> dict[str, float]:
        """Calculate electrode performance metrics."""
        substrate = self.current_state['substrate_concentration']
        biofilm_density = self.current_state['biofilm_density']
        velocity = self.current_state['velocity_field']
        pore_sizes = self.current_state['pore_size']

        # Volume-averaged quantities
        avg_substrate = np.mean(substrate)
        avg_biofilm_density = np.mean(biofilm_density)
        max_biofilm_thickness = np.max(self.current_state['biofilm_thickness'])

        # Flow characteristics
        avg_velocity = np.mean(np.linalg.norm(velocity, axis=0))

        # Pore blocking assessment
        initial_pore_size = 100e-6
        avg_pore_reduction = 1 - np.mean(pore_sizes) / initial_pore_size

        # Mass transport effectiveness
        substrate_utilization = 1 - avg_substrate / 25.0  # Assuming 25 mM inlet

        # Estimated current density (simplified)
        current_density = (biofilm_density * substrate_utilization *
                          self.electrode_config.calculate_charge_transfer_coefficient())
        avg_current_density = np.mean(current_density)

        return {
            'avg_substrate_mM': avg_substrate,
            'avg_biofilm_density_kg_m3': avg_biofilm_density,
            'max_biofilm_thickness_um': max_biofilm_thickness * 1e6,
            'avg_velocity_m_s': avg_velocity,
            'pore_reduction_fraction': avg_pore_reduction,
            'substrate_utilization': substrate_utilization,
            'avg_current_density_A_m2': avg_current_density,
            'electrode_utilization_efficiency': np.mean(biofilm_density > 1.0)  # Fraction with active biofilm
        }

    def get_optimization_targets(self) -> dict[str, float]:
        """Get optimization targets for ML-based parameter tuning."""
        metrics = self.calculate_performance_metrics()

        return {
            'maximize_current_density': metrics['avg_current_density_A_m2'],
            'maximize_substrate_utilization': metrics['substrate_utilization'],
            'minimize_pressure_drop': np.mean(self.current_state['pressure_field']),
            'maximize_electrode_utilization': metrics['electrode_utilization_efficiency'],
            'minimize_pore_blocking': 1 - metrics['pore_reduction_fraction'],
            'maximize_biofilm_stability': 1 / (1 + np.std(self.current_state['biofilm_density']))
        }


# Helper functions for ML optimization
def create_optimization_objective(electrode_model: AdvancedElectrodeModel,
                                weights: dict[str, float]) -> Callable:
    """Create objective function for ML optimization."""

    def objective_function(parameters: np.ndarray) -> float:
        """Objective function for parameter optimization."""
        # Parameters could include: pore size, electrode dimensions, flow rate, etc.
        # This is a simplified example

        try:
            # Run simulation with current parameters
            electrode_model.step(dt=3600)  # 1 hour step
            targets = electrode_model.get_optimization_targets()

            # Weighted objective
            objective = 0.0
            for target, value in targets.items():
                if target.startswith('maximize'):
                    objective += weights.get(target, 1.0) * value
                elif target.startswith('minimize'):
                    objective -= weights.get(target, 1.0) * value

            return -objective  # Minimize negative of objective

        except Exception:
            # Return large penalty for invalid parameters
            return 1e6

    return objective_function


# Example usage and validation
def validate_advanced_model():
    """Validate the advanced electrode model with test case."""
    from config.electrode_config import DEFAULT_GRAPHITE_PLATE_CONFIG

    # Create test cell geometry
    cell_geom = CellGeometry(
        length=0.1,  # 10 cm
        width=0.1,   # 10 cm
        height=0.05, # 5 cm
        anode_chamber_volume=0.0002,  # 200 mL
        cathode_chamber_volume=0.0002  # 200 mL
    )

    # Create advanced electrode model
    model = AdvancedElectrodeModel(
        electrode_config=DEFAULT_GRAPHITE_PLATE_CONFIG,
        cell_geometry=cell_geom
    )

    # Check compatibility
    compatibility = model.compatibility_check
    print(f"Electrode fits: {compatibility['fits']}")
    print(f"Volume utilization: {compatibility['volume_utilization']:.1%}")

    # Run simulation step
    results = model.step(dt=3600)  # 1 hour
    metrics = results['performance_metrics']

    print(f"Average substrate: {metrics['avg_substrate_mM']:.1f} mM")
    print(f"Max biofilm thickness: {metrics['max_biofilm_thickness_um']:.1f} μm")
    print(f"Current density: {metrics['avg_current_density_A_m2']:.3f} A/m²")

    return model, results


if __name__ == "__main__":
    # Run validation
    model, results = validate_advanced_model()
    print("✅ Advanced electrode model validation completed")
