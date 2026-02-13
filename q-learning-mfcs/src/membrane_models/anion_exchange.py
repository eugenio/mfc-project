#!/usr/bin/env python3
"""Anion Exchange Membrane (AEM) models for MFC simulations.

Implements AEM with hydroxide transport, bicarbonate/carbonate interference,
and unique degradation mechanisms relevant to alkaline fuel cells.

Literature sources:
- Varcoe et al., Energy Environ. Sci. 2014
- Dekel, J. Power Sources 2018
- Gottesfeld et al., J. Power Sources 2018

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
class AEMParameters(MembraneParameters):
    """Parameters specific to anion exchange membranes."""

    # Membrane type
    membrane_type: str = "Quaternary Ammonium"  # QA, Imidazolium, Phosphonium
    polymer_backbone: str = "PPO"  # PPO, PEEK, Fluorinated

    # Ionic conductivity
    hydroxide_conductivity_ref: float = 0.04  # S/cm at reference (lower than PEM)
    conductivity_activation: float = 20000  # J/mol - higher than PEM

    # Ion transport selectivity
    hydroxide_selectivity: float = 0.95  # OH- vs HCO3-/CO3--
    bicarbonate_permeability: float = 0.8  # Relative to OH-

    # Water management
    water_uptake_max: float = 15.0  # mol H2O/mol functional group
    water_diffusion: float = 1.5e-10  # m²/s - water diffusion
    electro_osmotic_drag: float = 4.0  # H2O/OH- - higher than PEM

    # Chemical stability
    alkaline_stability_factor: float = 0.8  # Stability vs PEM
    degradation_activation: float = 50000  # J/mol - for Hofmann elimination
    max_operating_ph: float = 14.0  # Maximum stable pH

    # Carbonation effects
    co2_absorption_rate: float = 1e-8  # mol/m²/s/Pa CO2
    carbonate_conductivity_ratio: float = 0.3  # σ(CO3--)/σ(OH-)

    # Mechanical properties
    youngs_modulus: float = 200e6  # Pa - typically lower than PEM
    tensile_strength: float = 20e6  # Pa
    max_swelling: float = 0.30  # Higher swelling than PEM

    # Cost parameters
    material_cost_per_m2: float = 400.0  # $/m² - target cost


class AnionExchangeMembrane(BaseMembraneModel):
    """Anion exchange membrane model with comprehensive transport mechanisms.

    Features:
    - Hydroxide ion transport as primary charge carrier
    - Bicarbonate/carbonate competition and carbonation
    - Unique degradation mechanisms (Hofmann elimination)
    - pH gradient effects
    - CO2 absorption and mitigation strategies
    """

    def __init__(self, parameters: AEMParameters) -> None:
        self.aem_params = parameters

        # AEM-specific state (initialize before super().__init__)
        self.carbonate_fraction = 0.0  # Fraction of sites with CO3--
        self.cumulative_degradation = 0.0
        self.ph_gradient_history = []

        super().__init__(parameters)

    def _setup_ion_transport(self) -> None:
        """Setup anion transport mechanisms for AEM."""
        self.ion_transport = {}

        # Primary: Hydroxide transport
        hydroxide_transport = IonTransportDatabase.get_hydroxide_transport()
        hydroxide_transport.partition_coefficient = 1.0  # Reference
        hydroxide_transport.diffusion_coefficient *= 0.8  # Slower in polymer
        hydroxide_transport.hydration_number = 4.0  # More water drag
        self.ion_transport[IonType.HYDROXIDE] = hydroxide_transport

        # Bicarbonate (major competitor)
        bicarbonate_transport = IonTransportMechanisms(
            ion_type=IonType.BICARBONATE,
            diffusion_coefficient=1.2e-9,
            mobility=4.6e-8,
            partition_coefficient=self.aem_params.bicarbonate_permeability,
            hydration_number=2.0,
            charge=-1,
            stokes_radius=3.4e-10,
        )
        self.ion_transport[IonType.BICARBONATE] = bicarbonate_transport

        # Carbonate
        carbonate_transport = IonTransportMechanisms(
            ion_type=IonType.CARBONATE,
            diffusion_coefficient=0.9e-9,
            mobility=7.0e-8,
            partition_coefficient=0.7,
            hydration_number=3.0,
            charge=-2,
            stokes_radius=3.8e-10,
        )
        self.ion_transport[IonType.CARBONATE] = carbonate_transport

        # Chloride (common impurity)
        chloride_transport = IonTransportMechanisms(
            ion_type=IonType.CHLORIDE,
            diffusion_coefficient=2.0e-9,
            mobility=7.9e-8,
            partition_coefficient=0.5,
            hydration_number=1.0,
            charge=-1,
            stokes_radius=3.3e-10,
        )
        self.ion_transport[IonType.CHLORIDE] = chloride_transport

    def _calculate_membrane_properties(self) -> None:
        """Calculate AEM-specific properties."""
        # Water content
        self.calculate_water_content()

        # Base conductivity
        self.base_conductivity = self.calculate_ionic_conductivity()

        # Carbonate fraction from CO2 exposure
        self.update_carbonation()

    def calculate_water_content(self, water_activity: float = 1.0) -> float:
        """Calculate AEM water content.

        AEMs typically have lower water uptake than PEMs.

        Args:
            water_activity: Water activity (0-1)

        Returns:
            Water content (mol H2O/mol functional group)

        """
        # Generic isotherm for AEM
        if water_activity <= 1.0:
            lambda_water = self.aem_params.water_uptake_max * (
                water_activity / (1 + 0.5 * (1 - water_activity))
            )
        else:
            # Liquid water
            lambda_water = self.aem_params.water_uptake_max

        self.params.water_content = lambda_water
        return float(lambda_water)

    def calculate_ionic_conductivity(
        self,
        temperature: float | None = None,
        water_content: float | None = None,
    ) -> float:
        """Calculate ionic conductivity accounting for carbonation.

        Args:
            temperature: Temperature (K)
            water_content: Water content

        Returns:
            Ionic conductivity (S/m)

        """
        if temperature is None:
            temperature = self.temperature
        if water_content is None:
            water_content = self.params.water_content

        # Temperature dependence
        T_ref = 298.15
        R = self.params.gas_constant
        Ea = self.aem_params.conductivity_activation

        temp_factor = jnp.exp(-Ea / R * (1.0 / temperature - 1.0 / T_ref))

        # Water content dependence (stronger than PEM)
        water_factor = (water_content / self.aem_params.water_uptake_max) ** 2.0

        # Base hydroxide conductivity
        conductivity_OH = (
            self.aem_params.hydroxide_conductivity_ref * temp_factor * water_factor
        )

        # Account for carbonation
        # Mixed conductivity: σ = (1-x)σ_OH + x·σ_CO3
        carbonate_conductivity = (
            conductivity_OH * self.aem_params.carbonate_conductivity_ratio
        )

        effective_conductivity = (
            1 - self.carbonate_fraction
        ) * conductivity_OH + self.carbonate_fraction * carbonate_conductivity

        # Degradation
        degradation_factor = 1.0 - self.cumulative_degradation
        effective_conductivity *= degradation_factor

        # Convert to S/m
        return float(effective_conductivity * 100.0)

    def update_carbonation(
        self,
        co2_partial_pressure: float = 400e-6 * 101325,
        exposure_time: float = 0.0,
    ) -> None:
        """Update membrane carbonation from CO2 exposure.

        Args:
            co2_partial_pressure: CO2 partial pressure (Pa)
            exposure_time: Exposure time (hours)

        """
        if exposure_time <= 0:
            return

        # CO2 absorption rate
        co2_flux = self.aem_params.co2_absorption_rate * co2_partial_pressure

        # Moles of CO2 absorbed
        co2_absorbed = co2_flux * self.area * exposure_time * 3600  # mol

        # Ion exchange sites
        IEC = self.params.ion_exchange_capacity  # mol/kg
        dry_mass = self.thickness * self.area * 2000  # kg (assume 2000 kg/m³)
        total_sites = IEC * dry_mass

        # Fraction converted to carbonate/bicarbonate
        # 2OH- + CO2 → CO3-- + H2O
        new_carbonate = co2_absorbed / total_sites

        # Update carbonate fraction (max 1.0)
        self.carbonate_fraction = min(1.0, self.carbonate_fraction + new_carbonate)

    def calculate_ph_gradient_effect(
        self,
        ph_anode: float,
        ph_cathode: float,
    ) -> dict[str, float]:
        """Calculate effects of pH gradient across membrane.

        Args:
            ph_anode: Anode pH
            ph_cathode: Cathode pH

        Returns:
            pH gradient effects

        """
        # OH- concentrations
        pOH_anode = 14.0 - ph_anode
        pOH_cathode = 14.0 - ph_cathode

        C_OH_anode = 10 ** (-pOH_anode) * 1000  # mol/m³
        C_OH_cathode = 10 ** (-pOH_cathode) * 1000

        # Concentration gradient driving force
        gradient = (C_OH_cathode - C_OH_anode) / self.thickness

        # Diffusion potential (Nernst)
        R = self.params.gas_constant
        T = self.temperature
        F = self.params.faraday_constant

        if C_OH_anode > 0 and C_OH_cathode > 0:
            diffusion_potential = (R * T / F) * jnp.log(C_OH_cathode / C_OH_anode)
        else:
            diffusion_potential = 0.0

        # pH gradient stress factor (affects degradation)
        ph_stress = abs(ph_cathode - ph_anode) / 14.0

        return {
            "ph_gradient": float(ph_cathode - ph_anode),
            "OH_concentration_gradient": float(gradient),
            "diffusion_potential_V": float(diffusion_potential),
            "ph_stress_factor": float(ph_stress),
        }

    def calculate_degradation_rate(
        self,
        temperature: float,
        ph: float,
        current_density: float,
    ) -> float:
        """Calculate AEM degradation rate.

        Main mechanisms:
        - Hofmann elimination (high T, high pH)
        - Nucleophilic substitution
        - Oxidative degradation

        Args:
            temperature: Temperature (K)
            ph: Local pH
            current_density: Current density (A/m²)

        Returns:
            Degradation rate (h⁻¹)

        """
        # Hofmann elimination rate
        # R4N+ + OH- → R3N + R-OH (β-elimination)
        R = self.params.gas_constant
        Ea = self.aem_params.degradation_activation

        # Base rate at pH 14, 80°C
        k_base = 1e-5  # h⁻¹

        # Temperature effect (Arrhenius)
        T_ref = 353.15  # 80°C
        temp_factor = jnp.exp(-Ea / R * (1.0 / temperature - 1.0 / T_ref))

        # pH effect (higher pH accelerates)
        OH_conc = 10 ** (ph - 14) * 1000  # mol/m³
        OH_ref = 1000  # 1 M OH-
        ph_factor = OH_conc / OH_ref

        # Current density effect (local heating, radical formation)
        current_factor = 1.0 + current_density / 10000  # Normalized to 1 A/cm²

        # Stability factor for different chemistries
        stability = self.aem_params.alkaline_stability_factor

        degradation_rate = k_base * temp_factor * ph_factor * current_factor / stability

        return float(degradation_rate)

    def simulate_co2_mitigation(
        self,
        operating_conditions: dict[str, Any],
    ) -> dict[str, float]:
        """Simulate CO2 mitigation strategies.

        Strategies:
        1. CO2 removal from inlet air
        2. High pH operation
        3. Pulsed operation

        Args:
            operating_conditions: Operating parameters

        Returns:
            Mitigation effectiveness metrics

        """
        co2_ppm = operating_conditions.get("co2_ppm", 400)  # Air CO2
        ph_cathode = operating_conditions.get("ph_cathode", 13.0)
        use_scrubber = operating_conditions.get("use_co2_scrubber", False)
        pulse_frequency = operating_conditions.get("pulse_frequency_hz", 0)

        # CO2 scrubber effectiveness
        if use_scrubber:
            co2_removal_efficiency = 0.95  # 95% removal
            effective_co2_ppm = co2_ppm * (1 - co2_removal_efficiency)
        else:
            effective_co2_ppm = co2_ppm

        # pH effect on carbonation
        # Higher pH converts CO2 to CO3-- faster but also promotes it
        if ph_cathode > 12:
            carbonation_rate_factor = 0.5  # Reduced due to CO3-- formation
        else:
            carbonation_rate_factor = 1.5  # Enhanced in mild alkaline

        # Pulsed operation effect
        # Periodic high current can drive out CO2
        if pulse_frequency > 0:
            pulse_mitigation = 0.3 * jnp.tanh(pulse_frequency / 10.0)
        else:
            pulse_mitigation = 0.0

        # Calculate carbonation rate
        co2_pressure = effective_co2_ppm * 1e-6 * 101325  # Pa
        base_carbonation_rate = self.aem_params.co2_absorption_rate * co2_pressure

        mitigated_rate = (
            base_carbonation_rate * carbonation_rate_factor * (1 - pulse_mitigation)
        )

        # Performance retention (fraction of initial performance after time constant)
        carbonation_time_constant = 100  # hours to 63% carbonation
        # Normalize by reference absorption rate to get dimensionless exponent
        ref_absorption_rate = 1e-8  # mol/m²/s/Pa - reference rate
        normalized_rate = mitigated_rate / ref_absorption_rate
        performance_retention = jnp.exp(-normalized_rate * 0.01)  # Small decay factor

        return {
            "effective_co2_ppm": float(effective_co2_ppm),
            "carbonation_rate_factor": float(carbonation_rate_factor),
            "pulse_mitigation_factor": float(pulse_mitigation),
            "mitigated_carbonation_rate": float(mitigated_rate),
            "performance_retention_percent": float(performance_retention * 100),
            "carbonation_time_constant_hours": float(carbonation_time_constant),
        }

    def calculate_water_balance(
        self,
        current_density: float,
        rh_anode: float,
        rh_cathode: float,
    ) -> dict[str, float]:
        """Calculate water balance in AEM.

        AEMs have higher water drag than PEMs.

        Args:
            current_density: Current density (A/m²)
            rh_anode: Anode relative humidity (%)
            rh_cathode: Cathode relative humidity (%)

        Returns:
            Water transport fluxes

        """
        # Water content at each side
        lambda_anode = self.calculate_water_content(rh_anode / 100.0)
        lambda_cathode = self.calculate_water_content(rh_cathode / 100.0)

        # Electro-osmotic drag (OH- carries more water)
        hydroxide_flux = current_density / self.params.faraday_constant
        drag_coefficient = self.aem_params.electro_osmotic_drag

        # Account for carbonate (carries less water)
        effective_drag = (
            drag_coefficient * (1 - self.carbonate_fraction)
            + 2.0 * self.carbonate_fraction
        )  # CO3-- drags less

        flux_electro_osmotic = effective_drag * hydroxide_flux

        # Back diffusion
        C_fixed = self.params.fixed_charge_density
        C_water_anode = lambda_anode * C_fixed
        C_water_cathode = lambda_cathode * C_fixed

        D_water = self.aem_params.water_diffusion
        flux_diffusion = D_water * (C_water_cathode - C_water_anode) / self.thickness

        # Net flux (positive = anode to cathode)
        net_flux = flux_electro_osmotic - flux_diffusion

        # Water production at cathode (ORR: O2 + 2H2O + 4e- → 4OH-)
        # Consumes water!
        orr_consumption = 0.5 * hydroxide_flux  # 2H2O per 4e-

        return {
            "electro_osmotic_flux_mol_m2_s": float(flux_electro_osmotic),
            "diffusion_flux_mol_m2_s": float(flux_diffusion),
            "net_water_flux_mol_m2_s": float(net_flux),
            "orr_water_consumption_mol_m2_s": float(orr_consumption),
            "effective_drag_coefficient": float(effective_drag),
            "water_content_anode": float(lambda_anode),
            "water_content_cathode": float(lambda_cathode),
        }

    def get_stability_assessment(
        self,
        operating_hours: float,
        average_temperature: float,
        average_ph: float,
    ) -> dict[str, float]:
        """Assess AEM stability and remaining lifetime.

        Args:
            operating_hours: Hours of operation
            average_temperature: Average temperature (K)
            average_ph: Average pH

        Returns:
            Stability metrics

        """
        # Calculate degradation
        deg_rate = self.calculate_degradation_rate(
            average_temperature,
            average_ph,
            5000,
        )
        self.cumulative_degradation = 1 - jnp.exp(-deg_rate * operating_hours)

        # Conductivity retention (degradation should reduce conductivity)
        initial_cond = self.aem_params.hydroxide_conductivity_ref * 100  # S/m
        current_cond = self.calculate_ionic_conductivity()
        # Apply degradation effect to ensure retention is <= 1.0
        degradation_factor = 1.0 - self.cumulative_degradation
        conductivity_retention = min(
            1.0,
            (current_cond / initial_cond) * degradation_factor,
        )

        # Estimate remaining lifetime (to 50% conductivity)
        if deg_rate > 0:
            lifetime_to_50 = -jnp.log(0.5) / deg_rate
            remaining_hours = max(0, lifetime_to_50 - operating_hours)
        else:
            lifetime_to_50 = float("inf")
            remaining_hours = float("inf")

        # Chemical stability indicators
        if self.aem_params.membrane_type == "Quaternary Ammonium":
            max_stable_temp = 333.15  # 60°C
        elif self.aem_params.membrane_type == "Imidazolium":
            max_stable_temp = 353.15  # 80°C
        else:
            max_stable_temp = 343.15  # 70°C

        temp_margin = max_stable_temp - average_temperature

        return {
            "cumulative_degradation_fraction": float(self.cumulative_degradation),
            "conductivity_retention_percent": float(conductivity_retention * 100),
            "estimated_lifetime_hours": float(lifetime_to_50),
            "remaining_lifetime_hours": float(remaining_hours),
            "temperature_margin_K": float(temp_margin),
            "carbonate_fraction": float(self.carbonate_fraction),
            "degradation_rate_per_hour": float(deg_rate),
            "membrane_type": self.aem_params.membrane_type,
        }

    def get_aem_properties(self) -> dict[str, Any]:
        """Get comprehensive AEM properties."""
        base_properties = self.get_transport_properties()

        aem_specific = {
            "membrane_type": self.aem_params.membrane_type,
            "polymer_backbone": self.aem_params.polymer_backbone,
            "hydroxide_conductivity_S_cm": float(
                self.calculate_ionic_conductivity() / 100,
            ),
            "carbonate_fraction": float(self.carbonate_fraction),
            "water_uptake_max": self.aem_params.water_uptake_max,
            "electro_osmotic_drag": self.aem_params.electro_osmotic_drag,
            "alkaline_stability_factor": self.aem_params.alkaline_stability_factor,
            "max_operating_ph": self.aem_params.max_operating_ph,
            "cost_per_m2_USD": self.aem_params.material_cost_per_m2,
            "carbonation_sensitivity": (
                "High" if self.carbonate_fraction > 0.1 else "Low"
            ),
        }

        return {**base_properties, **aem_specific}


def create_aem_membrane(
    membrane_type: str = "Quaternary Ammonium",
    thickness_um: float = 100.0,
    area_cm2: float = 1.0,
    temperature_C: float = 30.0,
    ion_exchange_capacity: float = 2.0,
) -> AnionExchangeMembrane:
    """Create an anion exchange membrane.

    Args:
        membrane_type: Type of AEM chemistry
        thickness_um: Membrane thickness in micrometers
        area_cm2: Membrane area in cm²
        temperature_C: Operating temperature in °C
        ion_exchange_capacity: IEC in mol/kg

    Returns:
        Configured AEM

    """
    # Set properties based on membrane type
    if membrane_type == "Quaternary Ammonium":
        conductivity = 0.04
        stability = 0.7
        backbone = "PPO"
    elif membrane_type == "Imidazolium":
        conductivity = 0.05
        stability = 0.85
        backbone = "PEEK"
    elif membrane_type == "Phosphonium":
        conductivity = 0.03
        stability = 0.9
        backbone = "Fluorinated"
    else:
        conductivity = 0.03
        stability = 0.8
        backbone = "Polymer"

    params = AEMParameters(
        membrane_type=membrane_type,
        polymer_backbone=backbone,
        thickness=thickness_um * 1e-6,
        area=area_cm2 * 1e-4,
        temperature=temperature_C + 273.15,
        ion_exchange_capacity=ion_exchange_capacity,
        hydroxide_conductivity_ref=conductivity,
        alkaline_stability_factor=stability,
        water_uptake_max=15.0,
        electro_osmotic_drag=4.0,
        material_cost_per_m2=400.0,
    )

    return AnionExchangeMembrane(params)
