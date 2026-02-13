#!/usr/bin/env python3
"""Proton Exchange Membrane (PEM) models for MFC simulations.

Implements various PEM types including Nafion, SPEEK, and composite membranes
with detailed proton transport, water management, and degradation mechanisms.

Literature sources:
- Nafion properties: Mauritz & Moore, Chem. Rev. 2004
- SPEEK membranes: Hickner et al., Chem. Rev. 2004
- Transport mechanisms: Kreuer, J. Membr. Sci. 2001

Created: 2025-07-27
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .base_membrane import (
    BaseMembraneModel,
    IonTransportDatabase,
    IonTransportMechanisms,
    IonType,
    MembraneParameters,
)


@dataclass
class PEMParameters(MembraneParameters):
    """Parameters specific to proton exchange membranes."""

    # Membrane type and grade
    membrane_type: str = "Nafion"  # Nafion, SPEEK, PFSA, etc.
    membrane_grade: str = "117"  # 117, 115, 212, etc.

    # Proton conductivity parameters
    conductivity_ref: float = 0.1  # S/cm at reference conditions
    conductivity_activation: float = 15000  # J/mol - activation energy

    # Water management
    water_uptake_max: float = 22.0  # mol H2O/mol SO3H - maximum
    water_diffusion: float = 2.5e-10  # m²/s - water diffusion
    electro_osmotic_drag: float = 2.5  # H2O/H+ - drag coefficient
    hydraulic_permeability: float = 1e-18  # m² - for pressure-driven flow

    # Mechanical properties
    youngs_modulus: float = 250e6  # Pa - elastic modulus (wet)
    tensile_strength: float = 25e6  # Pa - tensile strength
    max_swelling: float = 0.25  # - maximum dimensional change

    # Chemical stability
    peroxide_rate_constant: float = 1e-8  # s⁻¹ - degradation by H2O2
    fluoride_release_rate: float = 1e-10  # mol/m²/s - for Nafion

    # Methanol crossover (for DMFC applications)
    methanol_permeability: float = 2e-6  # cm²/s

    # Cost parameters
    material_cost_per_m2: float = 800.0  # $/m² - Nafion cost


class ProtonExchangeMembrane(BaseMembraneModel):
    """Proton exchange membrane model with comprehensive transport mechanisms.

    Features:
    - Proton transport via Grotthuss and vehicle mechanisms
    - Water management and electro-osmotic drag
    - Temperature and humidity dependencies
    - Chemical and mechanical degradation
    - Methanol crossover for DMFC
    """

    def __init__(self, parameters: PEMParameters) -> None:
        self.pem_params = parameters
        super().__init__(parameters)

        # Initialize PEM-specific state
        self.water_content_history = []
        self.degradation_history = []
        self.cumulative_fluoride_loss = 0.0

    def _setup_ion_transport(self) -> None:
        """Setup proton transport mechanisms for PEM."""
        self.ion_transport = {}

        # Primary: Proton transport
        proton_transport = IonTransportDatabase.get_proton_transport()
        # Adjust for PEM environment
        proton_transport.partition_coefficient = 1.2  # Favors protons
        proton_transport.diffusion_coefficient *= 2.0  # Fast in PEM
        self.ion_transport[IonType.PROTON] = proton_transport

        # Secondary ions (typically excluded but can permeate)
        sodium_transport = IonTransportDatabase.get_sodium_transport()
        sodium_transport.partition_coefficient = 0.01  # Strongly excluded
        self.ion_transport[IonType.SODIUM] = sodium_transport

        # Add other cations that might be present
        potassium_transport = IonTransportMechanisms(
            ion_type=IonType.POTASSIUM,
            diffusion_coefficient=2.0e-9,
            mobility=7.6e-8,
            partition_coefficient=0.01,  # Excluded like Na+
            hydration_number=3.0,
            charge=1,
            stokes_radius=3.3e-10,
        )
        self.ion_transport[IonType.POTASSIUM] = potassium_transport

    def _calculate_membrane_properties(self) -> None:
        """Calculate PEM-specific properties."""
        # Water content based on activity
        self.calculate_water_content()

        # Update conductivity based on water content
        self.base_conductivity = self.calculate_ionic_conductivity()

    def calculate_water_content(self, water_activity: float = 1.0) -> float:
        """Calculate membrane water content using sorption isotherm.

        For Nafion: λ = 0.043 + 17.81*a - 39.85*a² + 36.0*a³ (for a < 1)
                    λ = 14 + 1.4*(a - 1) (for a > 1, liquid water)

        Args:
            water_activity: Water activity (0-1 for vapor, >1 for liquid)

        Returns:
            Water content (mol H2O/mol SO3H)

        """
        if self.pem_params.membrane_type == "Nafion":
            if water_activity <= 1.0:
                # Vapor equilibrated
                lambda_water = (
                    0.043
                    + 17.81 * water_activity
                    - 39.85 * water_activity**2
                    + 36.0 * water_activity**3
                )
            else:
                # Liquid equilibrated (Schroeder's paradox)
                lambda_water = 14.0 + 1.4 * (water_activity - 1.0)

        elif self.pem_params.membrane_type == "SPEEK":
            # SPEEK has different water uptake
            if water_activity <= 1.0:
                lambda_water = 8.0 * water_activity
            else:
                lambda_water = 8.0 + 2.0 * (water_activity - 1.0)

        else:
            # Generic PEM
            lambda_water = self.pem_params.water_uptake_max * water_activity

        # Limit to maximum
        lambda_water = min(lambda_water, self.pem_params.water_uptake_max)

        # Update parameter
        self.params.water_content = lambda_water

        return float(lambda_water)

    def calculate_ionic_conductivity(
        self,
        temperature: float | None = None,
        water_content: float | None = None,
    ) -> float:
        """Calculate proton conductivity with temperature and water dependencies.

        σ = σ_ref * exp(Ea/R * (1/T_ref - 1/T)) * f(λ)

        Args:
            temperature: Temperature (K)
            water_content: Water content (mol H2O/mol SO3H)

        Returns:
            Ionic conductivity (S/m)

        """
        if temperature is None:
            temperature = self.temperature
        if water_content is None:
            water_content = self.params.water_content

        # Temperature dependence (Arrhenius)
        T_ref = 303.0  # 30°C reference
        R = self.params.gas_constant
        Ea = self.pem_params.conductivity_activation

        temp_factor = jnp.exp(-Ea / R * (1.0 / temperature - 1.0 / T_ref))

        # Water content dependence
        if self.pem_params.membrane_type == "Nafion":
            # Empirical relation for Nafion
            if water_content >= 1.0:
                water_factor = (0.005193 * water_content - 0.00326) * jnp.exp(
                    1268 * (1.0 / 303.0 - 1.0 / temperature),
                )
            else:
                water_factor = 0.001  # Very low conductivity when dry
        else:
            # Generic relation
            water_factor = (water_content / 14.0) ** 1.5

        # Base conductivity in S/cm
        conductivity_S_cm = (
            self.pem_params.conductivity_ref * temp_factor * water_factor
        )

        # Account for degradation
        degradation_factor = jnp.exp(
            -self.params.degradation_rate * self.operating_hours,
        )
        conductivity_S_cm *= degradation_factor

        # Convert to S/m
        conductivity = conductivity_S_cm * 100.0

        return float(conductivity)

    def calculate_electro_osmotic_drag(
        self,
        water_content: float | None = None,
    ) -> float:
        """Calculate electro-osmotic drag coefficient.

        n_drag = 2.5 * λ/22 for Nafion

        Args:
            water_content: Water content (mol H2O/mol SO3H)

        Returns:
            Drag coefficient (H2O/H+)

        """
        if water_content is None:
            water_content = self.params.water_content

        if self.pem_params.membrane_type == "Nafion":
            # Empirical relation for Nafion
            n_drag = 2.5 * water_content / 22.0
        else:
            # Linear approximation for others
            n_drag = self.pem_params.electro_osmotic_drag * water_content / 14.0

        return float(n_drag)

    def calculate_water_flux(
        self,
        current_density: float,
        water_activity_anode: float,
        water_activity_cathode: float,
        pressure_difference: float = 0.0,
    ) -> dict[str, float]:
        """Calculate water flux including all transport mechanisms.

        Args:
            current_density: Current density (A/m²)
            water_activity_anode: Water activity at anode
            water_activity_cathode: Water activity at cathode
            pressure_difference: Pressure difference (Pa)

        Returns:
            Dictionary with water fluxes (mol/m²/s)

        """
        # Electro-osmotic drag
        n_drag = self.calculate_electro_osmotic_drag()
        proton_flux = current_density / self.params.faraday_constant
        flux_electro_osmotic = n_drag * proton_flux

        # Back diffusion
        lambda_anode = self.calculate_water_content(water_activity_anode)
        lambda_cathode = self.calculate_water_content(water_activity_cathode)

        # Water concentration in membrane (mol/m³)
        EW = 1100  # g/mol SO3H for Nafion
        rho_dry = 2000  # kg/m³ dry membrane density
        C_SO3H = rho_dry / EW * 1000  # mol/m³

        C_water_anode = lambda_anode * C_SO3H
        C_water_cathode = lambda_cathode * C_SO3H

        # Diffusion flux
        D_water = self.pem_params.water_diffusion
        flux_diffusion = -D_water * (C_water_cathode - C_water_anode) / self.thickness

        # Hydraulic permeation (if pressure difference exists)
        if abs(pressure_difference) > 0:
            mu_water = 1e-3  # Pa·s water viscosity
            flux_hydraulic = (
                self.pem_params.hydraulic_permeability
                * pressure_difference
                / (mu_water * self.thickness)
            )
        else:
            flux_hydraulic = 0.0

        # Net water flux (positive = anode to cathode)
        net_flux = flux_electro_osmotic + flux_diffusion + flux_hydraulic

        return {
            "electro_osmotic_flux": float(flux_electro_osmotic),
            "diffusion_flux": float(flux_diffusion),
            "hydraulic_flux": float(flux_hydraulic),
            "net_water_flux": float(net_flux),
            "drag_coefficient": float(n_drag),
        }

    def calculate_methanol_crossover(
        self,
        methanol_conc_anode: float,
        methanol_conc_cathode: float,
        current_density: float,
    ) -> float:
        """Calculate methanol crossover for DMFC applications.

        Args:
            methanol_conc_anode: Methanol concentration at anode (mol/m³)
            methanol_conc_cathode: Methanol concentration at cathode (mol/m³)
            current_density: Current density (A/m²)

        Returns:
            Methanol flux (mol/m²/s)

        """
        # Diffusion component
        D_methanol = self.pem_params.methanol_permeability * 1e-4  # Convert to m²/s
        flux_diffusion = (
            D_methanol * (methanol_conc_anode - methanol_conc_cathode) / self.thickness
        )

        # Electro-osmotic drag of methanol
        # Methanol drag coefficient is typically 1-2 CH3OH/H+
        n_drag_methanol = 1.5 * self.params.water_content / 14.0
        proton_flux = current_density / self.params.faraday_constant
        flux_electro_osmotic = n_drag_methanol * proton_flux

        total_flux = flux_diffusion + flux_electro_osmotic

        return float(total_flux)

    def calculate_degradation_rate(
        self,
        operating_conditions: dict[str, float],
    ) -> float:
        """Calculate membrane degradation rate based on operating conditions.

        Args:
            operating_conditions: Dict with temperature, RH, potential, etc.

        Returns:
            Degradation rate (h⁻¹)

        """
        T = operating_conditions.get("temperature", self.temperature)
        RH = operating_conditions.get("relative_humidity", 100.0)
        potential = operating_conditions.get("cathode_potential", 0.7)

        # Chemical degradation (peroxide attack)
        # Higher at low RH and high temperature
        RH_factor = jnp.exp(-RH / 50.0)  # Worse at low RH
        temp_factor = jnp.exp((T - 333) / 20.0)  # Accelerates above 60°C

        # Potential accelerates peroxide formation
        potential_factor = jnp.exp((potential - 0.6) * 2.0)

        # Base degradation rate
        chemical_degradation = (
            self.pem_params.peroxide_rate_constant
            * RH_factor
            * temp_factor
            * potential_factor
        )

        # Mechanical degradation (RH cycling)
        # Track if this is implemented in real system
        mechanical_degradation = 0.0

        total_degradation = chemical_degradation + mechanical_degradation

        return float(total_degradation)

    def calculate_gas_crossover(
        self,
        gas_type: str,
        partial_pressure_anode: float,
        partial_pressure_cathode: float,
    ) -> float:
        """Calculate gas crossover with water content effects.

        Args:
            gas_type: Type of gas
            partial_pressure_anode: Anode side partial pressure (Pa)
            partial_pressure_cathode: Cathode side partial pressure (Pa)

        Returns:
            Gas flux (mol/m²/s)

        """
        # Base permeability values at λ=14 (mol·m/(m²·s·Pa))
        permeability_wet = {"O2": 1.8e-15, "H2": 3.5e-15, "CO2": 5.2e-15, "N2": 0.9e-15}

        if gas_type not in permeability_wet:
            return 0.0

        # Water content effect (permeability increases with hydration)
        lambda_norm = self.params.water_content / 14.0
        water_factor = 0.5 + 0.5 * lambda_norm  # 50% at dry, 100% at λ=14

        P_gas = permeability_wet[gas_type] * water_factor

        # Temperature correction
        T_ref = 303.0
        activation_energy = 20000  # J/mol
        temp_factor = jnp.exp(
            -activation_energy
            / self.params.gas_constant
            * (1.0 / self.temperature - 1.0 / T_ref),
        )

        P_gas *= temp_factor

        # Pressure difference
        delta_P = partial_pressure_cathode - partial_pressure_anode

        # Gas flux
        gas_flux = P_gas * delta_P / self.thickness

        return float(gas_flux)

    def simulate_humidity_cycling(
        self,
        n_cycles: int,
        RH_high: float = 100.0,
        RH_low: float = 30.0,
        cycle_time: float = 1.0,
    ) -> dict[str, Any]:
        """Simulate effects of humidity cycling on membrane.

        Args:
            n_cycles: Number of RH cycles
            RH_high: High RH value (%)
            RH_low: Low RH value (%)
            cycle_time: Time per cycle (hours)

        Returns:
            Degradation metrics

        """
        initial_conductivity = self.calculate_ionic_conductivity()

        # Mechanical stress from swelling/shrinking
        lambda_high = self.calculate_water_content(RH_high / 100.0)
        lambda_low = self.calculate_water_content(RH_low / 100.0)

        # Dimensional change
        swelling_strain = (
            (lambda_high - lambda_low) / 22.0 * self.pem_params.max_swelling
        )

        # Fatigue damage accumulation (Miner's rule)
        # N_f = C * (Δε)^(-b) where b≈2 for polymers
        cycles_to_failure = 1e6 * (swelling_strain / 0.1) ** (-2)
        damage_fraction = n_cycles / cycles_to_failure

        # Conductivity loss
        conductivity_loss = damage_fraction * 0.5  # 50% loss at failure
        final_conductivity = initial_conductivity * (1 - conductivity_loss)

        # Update degradation
        self.operating_hours += n_cycles * cycle_time
        additional_degradation = -jnp.log(1 - conductivity_loss) / self.operating_hours
        self.params.degradation_rate += additional_degradation

        return {
            "n_cycles": n_cycles,
            "swelling_strain": float(swelling_strain),
            "cycles_to_failure": float(cycles_to_failure),
            "damage_fraction": float(damage_fraction),
            "conductivity_loss_percent": float(conductivity_loss * 100),
            "initial_conductivity_S_cm": float(initial_conductivity / 100),
            "final_conductivity_S_cm": float(final_conductivity / 100),
        }

    def get_cost_analysis(self) -> dict[str, float]:
        """Calculate membrane cost analysis."""
        area_m2 = self.area

        # Material cost
        material_cost = self.pem_params.material_cost_per_m2 * area_m2

        # Performance metrics at standard conditions

        # Cost per unit power (at 0.6V cell voltage)
        current_density = 10000  # A/m² (1 A/cm²)
        cell_power = 0.6 * current_density * area_m2  # W
        cost_per_kW = material_cost / (cell_power / 1000)

        # Lifetime cost (assuming 5000 hour lifetime)
        lifetime_hours = 5000
        lifetime_energy = cell_power * lifetime_hours / 1000  # kWh
        cost_per_kWh = material_cost / lifetime_energy

        return {
            "material_cost_USD": float(material_cost),
            "cost_per_m2_USD": self.pem_params.material_cost_per_m2,
            "cost_per_kW_USD": float(cost_per_kW),
            "cost_per_kWh_USD": float(cost_per_kWh),
            "membrane_type": self.pem_params.membrane_type,
            "membrane_grade": self.pem_params.membrane_grade,
            "thickness_um": self.thickness * 1e6,
        }

    def get_pem_properties(self) -> dict[str, Any]:
        """Get comprehensive PEM properties."""
        base_properties = self.get_transport_properties()

        pem_specific = {
            "membrane_type": self.pem_params.membrane_type,
            "membrane_grade": self.pem_params.membrane_grade,
            "water_content": float(self.params.water_content),
            "conductivity_S_cm": float(self.calculate_ionic_conductivity() / 100),
            "drag_coefficient": float(self.calculate_electro_osmotic_drag()),
            "methanol_permeability_cm2_s": self.pem_params.methanol_permeability,
            "max_operating_temp_C": (
                80.0 if self.pem_params.membrane_type == "Nafion" else 160.0
            ),
            "ion_exchange_capacity_meq_g": self.params.ion_exchange_capacity * 1000,
            "cost_per_m2_USD": self.pem_params.material_cost_per_m2,
        }

        return {**base_properties, **pem_specific}


def create_nafion_membrane(
    thickness_um: float = 183.0,  # Nafion 117
    area_cm2: float = 1.0,
    temperature_C: float = 30.0,
) -> ProtonExchangeMembrane:
    """Create a Nafion membrane with standard properties.

    Args:
        thickness_um: Membrane thickness in micrometers
        area_cm2: Membrane area in cm²
        temperature_C: Operating temperature in °C

    Returns:
        Configured Nafion membrane

    """
    # Determine Nafion grade from thickness
    if thickness_um > 200:
        grade = "N117"
    elif thickness_um > 150:
        grade = "N115"
    elif thickness_um > 100:
        grade = "N1110"
    elif thickness_um > 70:
        grade = "N112"
    else:
        grade = "N212"

    params = PEMParameters(
        membrane_type="Nafion",
        membrane_grade=grade,
        thickness=thickness_um * 1e-6,
        area=area_cm2 * 1e-4,
        temperature=temperature_C + 273.15,
        ion_exchange_capacity=0.91,  # mol/kg for Nafion
        conductivity_ref=0.1,  # S/cm at 80°C, 100% RH
        water_uptake_max=22.0,
        electro_osmotic_drag=2.5,
        material_cost_per_m2=800.0,
    )

    return ProtonExchangeMembrane(params)


def create_speek_membrane(
    degree_sulfonation: float = 0.7,
    thickness_um: float = 100.0,
    area_cm2: float = 1.0,
    temperature_C: float = 30.0,
) -> ProtonExchangeMembrane:
    """Create a SPEEK membrane with specified degree of sulfonation.

    Args:
        degree_sulfonation: DS value (0-1)
        thickness_um: Membrane thickness in micrometers
        area_cm2: Membrane area in cm²
        temperature_C: Operating temperature in °C

    Returns:
        Configured SPEEK membrane

    """
    # SPEEK properties depend on degree of sulfonation
    IEC = 1.75 * degree_sulfonation  # mol/kg
    conductivity = 0.001 * jnp.exp(5.0 * degree_sulfonation)  # S/cm empirical

    params = PEMParameters(
        membrane_type="SPEEK",
        membrane_grade=f"DS{int(degree_sulfonation * 100)}",
        thickness=thickness_um * 1e-6,
        area=area_cm2 * 1e-4,
        temperature=temperature_C + 273.15,
        ion_exchange_capacity=IEC,
        conductivity_ref=conductivity,
        water_uptake_max=12.0 * degree_sulfonation,
        electro_osmotic_drag=1.5,
        material_cost_per_m2=200.0,  # Cheaper than Nafion
    )

    return ProtonExchangeMembrane(params)
