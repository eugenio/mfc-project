#!/usr/bin/env python3
"""
Biological cathode (biocathode) model for MFC simulations

Implements combined Butler-Volmer and Monod kinetics for electroactive bacteria
performing oxygen reduction with biofilm growth dynamics.

Literature-derived parameters:
- Microbial communities: Sphingobacterium, Acinetobacter, Acidovorax sp.
- Biofilm conductivity: 5 × 10⁻⁵ S/cm (Geobacter-like)
- Maximum growth rate: 0.5 h⁻¹ (typical for electroactive bacteria)
- Half-saturation constants from environmental biotechnology literature

Created: 2025-07-26
"""

import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base_cathode import BaseCathodeModel, CathodeParameters, ButlerVolmerKinetics


@dataclass
class BiologicalParameters:
    """Parametrizable biological cathode parameters with literature defaults"""
    
    # Microbial kinetic parameters
    max_growth_rate: float = 0.5  # h⁻¹ - maximum specific growth rate
    decay_rate: float = 0.05  # h⁻¹ - microbial decay rate
    detachment_rate: float = 0.01  # h⁻¹ - biofilm detachment rate
    yield_coefficient: float = 0.4  # g_biomass/g_substrate - microbial yield
    
    # Monod kinetics parameters
    ks_oxygen: float = 0.5e-3  # mol/L - half-saturation constant for O₂
    ks_potential: float = 0.1  # V - half-saturation constant for electrode potential
    ki_potential: float = 1.0  # V - inhibition constant for high potential
    
    # Optimal operating conditions
    optimal_potential: float = 0.4  # V vs SHE - optimal cathode potential
    optimal_ph: float = 7.0  # Optimal pH for microbial activity
    optimal_temperature: float = 303.15  # K (30°C) - optimal temperature
    
    # Biofilm properties
    biofilm_density: float = 80.0  # kg/m³ - wet biofilm density
    biofilm_conductivity: float = 5e-7  # S/m (5×10⁻⁵ S/cm) - biofilm conductivity
    biofilm_porosity: float = 0.8  # Void fraction in biofilm
    biofilm_tortuosity: float = 3.0  # Diffusion path tortuosity
    max_biofilm_thickness: float = 500e-6  # m - maximum stable thickness (500 μm)
    
    # Initial conditions
    initial_biofilm_thickness: float = 1e-6  # m - initial biofilm thickness (1 μm)
    initial_biomass_density: float = 1.0  # kg/m³ - initial biomass concentration
    
    # Electrochemical parameters
    exchange_current_density_base: float = 1e-4  # A/m² - base exchange current
    biomass_activity_factor: float = 1e3  # m²/kg - specific electroactive area
    transfer_coefficient: float = 0.5  # Electron transfer coefficient
    
    # Environmental tolerance ranges
    ph_tolerance_range: float = 2.0  # pH units - range around optimal pH
    temperature_tolerance_range: float = 15.0  # K - temperature tolerance
    
    # Oxygen mass transport in biofilm
    oxygen_diffusion_biofilm: float = 8e-10  # m²/s - O₂ diffusion in biofilm
    oxygen_consumption_rate_max: float = 0.1  # mol/(m³·s) - max O₂ consumption rate
    
    # Substrate utilization
    substrate_affinity: float = 1e-3  # mol/L - affinity for organic substrates
    substrate_inhibition: float = 10e-3  # mol/L - substrate inhibition threshold
    
    # Performance benchmarks (literature)
    reference_current_density: float = 500.0  # A/m² - typical biocathode performance
    reference_biofilm_thickness: float = 100e-6  # m - reference thickness for benchmarks
    
    # Economic parameters
    inoculum_cost_per_m2: float = 10.0  # $/m² - cost of microbial inoculation
    maintenance_cost_factor: float = 0.1  # Annual maintenance as fraction of initial cost
    operational_lifetime_years: float = 5.0  # Expected operational lifetime


class BiologicalCathodeModel(BaseCathodeModel):
    """
    Biological cathode model with parametrizable biofilm dynamics.
    
    Features:
    - Monod kinetics for microbial growth
    - Dynamic biofilm thickness evolution
    - pH and temperature effects
    - Biofilm conductivity and mass transport
    - Substrate utilization modeling
    - Long-term performance prediction
    """
    
    def __init__(self, parameters: CathodeParameters, bio_params: Optional[BiologicalParameters] = None):
        self.bio_params = bio_params or BiologicalParameters()
        super().__init__(parameters)
        
        # Initialize biofilm state variables
        self.biofilm_thickness = self.bio_params.initial_biofilm_thickness
        self.biomass_density = self.bio_params.initial_biomass_density
        self.biofilm_age_hours = 0.0
        
        self._setup_biofilm_parameters()
    
    def _setup_kinetic_parameters(self):
        """Setup temperature and pH dependent kinetic parameters."""
        # Temperature effect on growth rate (Arrhenius-like)
        T = self.temperature_K
        T_opt = self.bio_params.optimal_temperature
        T_range = self.bio_params.temperature_tolerance_range
        
        # Gaussian-like temperature response
        temp_factor = jnp.exp(-((T - T_opt) / T_range) ** 2)
        
        # pH effect on growth rate
        ph = self.params.ph
        ph_opt = self.bio_params.optimal_ph
        ph_range = self.bio_params.ph_tolerance_range
        
        # Gaussian pH response
        ph_factor = jnp.exp(-((ph - ph_opt) / ph_range) ** 2)
        
        # Combined environmental factor
        self.environmental_factor = temp_factor * ph_factor
        
        # Effective growth parameters
        self.effective_max_growth_rate = self.bio_params.max_growth_rate * self.environmental_factor
        self.effective_decay_rate = self.bio_params.decay_rate * (2.0 - self.environmental_factor)  # Higher decay in stress
        
    def _setup_biofilm_parameters(self):
        """Setup biofilm mass transport and electrochemical parameters."""
        # Effective diffusion coefficient in biofilm
        self.effective_oxygen_diffusion = (self.bio_params.oxygen_diffusion_biofilm * 
                                         self.bio_params.biofilm_porosity / 
                                         self.bio_params.biofilm_tortuosity)
        
        # Current biofilm-dependent exchange current density
        biomass_per_area = self.biomass_density * self.biofilm_thickness  # kg/m²
        self.exchange_current_density = (self.bio_params.exchange_current_density_base * 
                                       (1.0 + self.bio_params.biomass_activity_factor * biomass_per_area))
        
        # Biofilm resistance
        self.biofilm_resistance = self.biofilm_thickness / self.bio_params.biofilm_conductivity  # Ω·m²
    
    def calculate_monod_growth_rate(self, oxygen_conc: float, electrode_potential: float) -> float:
        """
        Calculate microbial growth rate using Monod kinetics.
        
        Args:
            oxygen_conc: Oxygen concentration in mol/L
            electrode_potential: Electrode potential in V
        
        Returns:
            Growth rate in h⁻¹
        """
        # Oxygen limitation
        oxygen_factor = oxygen_conc / (self.bio_params.ks_oxygen + oxygen_conc)
        
        # Electrode potential effect (electrons as energy source)
        potential_factor = electrode_potential / (self.bio_params.ks_potential + electrode_potential)
        
        # Potential inhibition at very high potentials
        inhibition_factor = self.bio_params.ki_potential / (self.bio_params.ki_potential + electrode_potential)
        
        # Combined growth rate
        growth_rate = (self.effective_max_growth_rate * 
                      oxygen_factor * potential_factor * inhibition_factor)
        
        return growth_rate
    
    def update_biofilm_dynamics(self, dt_hours: float, oxygen_conc: float, electrode_potential: float):
        """
        Update biofilm thickness and biomass density over time.
        
        Args:
            dt_hours: Time step in hours
            oxygen_conc: Oxygen concentration in mol/L
            electrode_potential: Electrode potential in V
        """
        # Calculate growth rate
        growth_rate = self.calculate_monod_growth_rate(oxygen_conc, electrode_potential)
        
        # Biomass density change
        growth_term = growth_rate * self.biomass_density
        decay_term = self.effective_decay_rate * self.biomass_density
        detachment_term = self.bio_params.detachment_rate * self.biomass_density
        
        dX_dt = growth_term - decay_term - detachment_term
        self.biomass_density = float(jnp.maximum(self.biomass_density + dX_dt * dt_hours, 0.1))
        
        # Biofilm thickness change (growth increases thickness)
        thickness_growth_rate = growth_rate * self.biofilm_thickness
        thickness_detachment = self.bio_params.detachment_rate * self.biofilm_thickness
        
        dL_dt = thickness_growth_rate - thickness_detachment
        
        # Update thickness with maximum limit
        new_thickness = self.biofilm_thickness + dL_dt * dt_hours
        self.biofilm_thickness = float(jnp.minimum(jnp.maximum(new_thickness, 1e-6), 
                                                 self.bio_params.max_biofilm_thickness))
        
        # Update age
        self.biofilm_age_hours += dt_hours
        
        # Recalculate biofilm-dependent parameters
        self._setup_biofilm_parameters()
    
    def calculate_current_density(self, overpotential: float, 
                                oxygen_conc: Optional[float] = None) -> float:
        """
        Calculate current density with biofilm-mediated electron transfer.
        
        Args:
            overpotential: Cathode overpotential in V
            oxygen_conc: Oxygen concentration in mol/L
        
        Returns:
            Current density in A/m²
        """
        if oxygen_conc is None:
            oxygen_conc = self.params.oxygen_concentration
        
        if overpotential <= 0:
            return 0.0
        
        # Concentration effect including biofilm mass transport
        # Effective oxygen concentration in biofilm considering consumption
        consumption_factor = 1.0 / (1.0 + self.bio_params.oxygen_consumption_rate_max * 
                                   self.biofilm_thickness / 
                                   (self.effective_oxygen_diffusion * oxygen_conc))
        
        effective_oxygen_conc = oxygen_conc * consumption_factor
        concentration_ratio = effective_oxygen_conc / self.params.oxygen_reference_conc
        
        # Butler-Volmer kinetics with biofilm modification
        current_density_kinetic = ButlerVolmerKinetics.calculate_current_density(
            exchange_current_density=self.exchange_current_density,
            transfer_coefficient=self.bio_params.transfer_coefficient,
            overpotential=overpotential,
            temperature_K=self.temperature_K,
            concentration_ratio=concentration_ratio
        )
        
        # Apply biofilm resistance (ohmic loss)
        # Effective overpotential reduced by biofilm resistance
        biofilm_voltage_drop = current_density_kinetic * self.biofilm_resistance
        effective_overpotential = jnp.maximum(overpotential - biofilm_voltage_drop, 0.0)
        
        # Recalculate with effective overpotential
        if effective_overpotential > 0:
            current_density = ButlerVolmerKinetics.calculate_current_density(
                exchange_current_density=self.exchange_current_density,
                transfer_coefficient=self.bio_params.transfer_coefficient,
                overpotential=float(effective_overpotential),
                temperature_K=self.temperature_K,
                concentration_ratio=concentration_ratio
            )
        else:
            current_density = 0.0
        
        return current_density
    
    def calculate_performance_metrics(self, operating_overpotential: float,
                                    oxygen_conc: Optional[float] = None) -> Dict[str, float]:
        """Calculate standard performance metrics (compatibility with base class)."""
        current_density = self.calculate_current_density(operating_overpotential, oxygen_conc)
        current = current_density * self.area_m2
        power_loss = self.calculate_power_loss(operating_overpotential, oxygen_conc)
        
        # Standard metrics
        power_density_mW_m2 = power_loss * 1000 / self.area_m2
        
        # Efficiency calculations
        equilibrium_potential = self.calculate_equilibrium_potential(oxygen_conc)
        voltage_efficiency = (equilibrium_potential - operating_overpotential) / equilibrium_potential * 100
        
        # Oxygen utilization
        o2_consumption_rate = self.calculate_oxygen_consumption_rate(operating_overpotential, oxygen_conc)
        
        return {
            'current_density_A_m2': float(current_density),
            'current_A': float(current),
            'power_loss_W': float(power_loss),
            'power_density_mW_m2': float(power_density_mW_m2),
            'voltage_efficiency_percent': float(voltage_efficiency),
            'overpotential_V': float(operating_overpotential),
            'equilibrium_potential_V': float(equilibrium_potential),
            'o2_consumption_rate_mol_s': float(o2_consumption_rate),
            'exchange_current_density_A_m2': float(self.exchange_current_density)
        }
    
    def calculate_biofilm_performance_metrics(self, operating_overpotential: float,
                                            oxygen_conc: Optional[float] = None) -> Dict[str, float]:
        """Calculate biofilm-specific performance metrics."""
        
        current_density = self.calculate_current_density(operating_overpotential, oxygen_conc)
        
        # Biofilm metrics
        total_biomass_per_area = self.biomass_density * self.biofilm_thickness  # kg/m²
        
        # Performance indicators
        current_per_biomass = current_density / jnp.maximum(total_biomass_per_area, 1e-6)  # A/kg
        power_per_biomass = (current_density * operating_overpotential / 
                           jnp.maximum(total_biomass_per_area, 1e-6))  # W/kg
        
        # Biofilm stability metrics
        growth_rate = self.calculate_monod_growth_rate(
            oxygen_conc or self.params.oxygen_concentration, 
            self.calculate_equilibrium_potential(oxygen_conc) - operating_overpotential
        )
        
        # Efficiency metrics
        biofilm_resistance_loss = current_density * self.biofilm_resistance  # V
        resistance_efficiency = (operating_overpotential - biofilm_resistance_loss) / operating_overpotential * 100
        
        return {
            'biofilm_thickness_um': float(self.biofilm_thickness * 1e6),
            'biomass_density_kg_m3': float(self.biomass_density),
            'total_biomass_per_area_kg_m2': float(total_biomass_per_area),
            'biofilm_age_hours': float(self.biofilm_age_hours),
            'current_per_biomass_A_kg': float(current_per_biomass),
            'power_per_biomass_W_kg': float(power_per_biomass),
            'growth_rate_h_inv': float(growth_rate),
            'biofilm_resistance_ohm_m2': float(self.biofilm_resistance),
            'biofilm_resistance_loss_V': float(biofilm_resistance_loss),
            'resistance_efficiency_percent': float(resistance_efficiency),
            'environmental_factor': float(self.environmental_factor),
            'exchange_current_density_A_m2': float(self.exchange_current_density)
        }
    
    def predict_long_term_performance(self, simulation_days: int = 30,
                                    oxygen_conc: Optional[float] = None,
                                    electrode_potential: float = 0.4) -> Dict[str, Any]:
        """
        Predict long-term biofilm development and performance.
        
        Args:
            simulation_days: Number of days to simulate
            oxygen_conc: Oxygen concentration in mol/L
            electrode_potential: Electrode potential in V
        
        Returns:
            Dictionary with time-series predictions
        """
        if oxygen_conc is None:
            oxygen_conc = self.params.oxygen_concentration
        
        # Time arrays
        dt_hours = 1.0  # 1 hour time steps
        n_steps = int(simulation_days * 24 / dt_hours)
        time_hours = np.linspace(0, simulation_days * 24, n_steps)
        
        # Storage arrays
        thickness_history = np.zeros(n_steps)
        biomass_history = np.zeros(n_steps)
        current_density_history = np.zeros(n_steps)
        growth_rate_history = np.zeros(n_steps)
        
        # Store initial state
        initial_thickness = self.biofilm_thickness
        initial_biomass = self.biomass_density
        
        for i in range(n_steps):
            # Update biofilm
            self.update_biofilm_dynamics(dt_hours, oxygen_conc, electrode_potential)
            
            # Calculate performance
            overpotential = self.calculate_equilibrium_potential(oxygen_conc) - electrode_potential
            current_density = self.calculate_current_density(overpotential, oxygen_conc)
            growth_rate = self.calculate_monod_growth_rate(oxygen_conc, electrode_potential)
            
            # Store results
            thickness_history[i] = self.biofilm_thickness * 1e6  # Convert to μm
            biomass_history[i] = self.biomass_density
            current_density_history[i] = current_density
            growth_rate_history[i] = growth_rate
        
        # Restore initial state
        self.biofilm_thickness = initial_thickness
        self.biomass_density = initial_biomass
        self.biofilm_age_hours = 0.0
        self._setup_biofilm_parameters()
        
        return {
            'time_hours': time_hours.tolist(),
            'biofilm_thickness_um': thickness_history.tolist(),
            'biomass_density_kg_m3': biomass_history.tolist(),
            'current_density_A_m2': current_density_history.tolist(),
            'growth_rate_h_inv': growth_rate_history.tolist(),
            'final_thickness_um': float(thickness_history[-1]),
            'final_current_density_A_m2': float(current_density_history[-1]),
            'average_current_density_A_m2': float(np.mean(current_density_history)),
            'time_to_steady_state_days': float(simulation_days * 0.8)  # Approximate
        }
    
    def estimate_economic_analysis(self) -> Dict[str, float]:
        """Economic analysis for biological cathode."""
        
        # Installation costs
        inoculation_cost = self.bio_params.inoculum_cost_per_m2 * self.area_m2
        
        # Operating costs (maintenance, monitoring)
        annual_maintenance = inoculation_cost * self.bio_params.maintenance_cost_factor
        
        # Lifetime costs
        total_lifetime_cost = inoculation_cost + (annual_maintenance * 
                                                self.bio_params.operational_lifetime_years)
        
        # Performance at reference conditions
        ref_metrics = self.calculate_performance_metrics(0.2)  # 200 mV overpotential
        power_density_mW_m2 = ref_metrics.get('power_density_mW_m2', 100)
        
        # Cost per unit power
        cost_per_kW = (total_lifetime_cost / self.area_m2) / (power_density_mW_m2 * 1e-6)
        
        return {
            'inoculation_cost_per_m2': float(inoculation_cost / self.area_m2),
            'annual_maintenance_cost_per_m2': float(annual_maintenance / self.area_m2),
            'total_lifetime_cost_per_m2': float(total_lifetime_cost / self.area_m2),
            'cost_per_kW': float(cost_per_kW),
            'operational_lifetime_years': self.bio_params.operational_lifetime_years,
            'maintenance_cost_factor': self.bio_params.maintenance_cost_factor
        }
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all configurable biological cathode parameters."""
        base_params = super().get_model_info()
        
        bio_specific = {
            'microbial_kinetics': {
                'max_growth_rate_h_inv': self.bio_params.max_growth_rate,
                'decay_rate_h_inv': self.bio_params.decay_rate,
                'detachment_rate_h_inv': self.bio_params.detachment_rate,
                'yield_coefficient': self.bio_params.yield_coefficient,
                'effective_max_growth_rate_h_inv': float(self.effective_max_growth_rate),
                'environmental_factor': float(self.environmental_factor)
            },
            'monod_parameters': {
                'ks_oxygen_mol_L': self.bio_params.ks_oxygen,
                'ks_potential_V': self.bio_params.ks_potential,
                'ki_potential_V': self.bio_params.ki_potential,
                'optimal_potential_V': self.bio_params.optimal_potential,
                'optimal_ph': self.bio_params.optimal_ph,
                'optimal_temperature_C': self.bio_params.optimal_temperature - 273.15
            },
            'biofilm_properties': {
                'current_thickness_um': float(self.biofilm_thickness * 1e6),
                'current_biomass_density_kg_m3': float(self.biomass_density),
                'biofilm_age_hours': float(self.biofilm_age_hours),
                'biofilm_density_kg_m3': self.bio_params.biofilm_density,
                'biofilm_conductivity_S_m': self.bio_params.biofilm_conductivity,
                'biofilm_porosity': self.bio_params.biofilm_porosity,
                'max_thickness_um': self.bio_params.max_biofilm_thickness * 1e6,
                'current_resistance_ohm_m2': float(self.biofilm_resistance)
            },
            'mass_transport': {
                'oxygen_diffusion_biofilm_m2_s': self.bio_params.oxygen_diffusion_biofilm,
                'effective_oxygen_diffusion_m2_s': float(self.effective_oxygen_diffusion),
                'oxygen_consumption_rate_max_mol_m3_s': self.bio_params.oxygen_consumption_rate_max
            },
            'electrochemical': {
                'exchange_current_density_base_A_m2': self.bio_params.exchange_current_density_base,
                'current_exchange_current_density_A_m2': float(self.exchange_current_density),
                'biomass_activity_factor_m2_kg': self.bio_params.biomass_activity_factor,
                'transfer_coefficient': self.bio_params.transfer_coefficient
            }
        }
        
        return {**base_params, **bio_specific}


def create_biological_cathode(area_cm2: float = 1.0,
                            temperature_C: float = 30.0,
                            ph: float = 7.0,
                            oxygen_mg_L: float = 8.0,
                            initial_thickness_um: float = 1.0,
                            custom_bio_params: Optional[Dict[str, float]] = None) -> BiologicalCathodeModel:
    """
    Create biological cathode with configurable parameters.
    
    Args:
        area_cm2: Cathode area in cm²
        temperature_C: Operating temperature in °C
        ph: Solution pH
        oxygen_mg_L: Dissolved oxygen in mg/L
        initial_thickness_um: Initial biofilm thickness in μm
        custom_bio_params: Custom biological parameters
    
    Returns:
        Configured BiologicalCathodeModel
    """
    # Convert units
    area_m2 = area_cm2 * 1e-4
    temperature_K = temperature_C + 273.15
    oxygen_mol_L = oxygen_mg_L / (32.0 * 1000)
    
    # Create base parameters
    cathode_params = CathodeParameters(
        area_m2=area_m2,
        temperature_K=temperature_K,
        oxygen_concentration=oxygen_mol_L,
        ph=ph
    )
    
    # Create biological parameters
    bio_params = BiologicalParameters(
        initial_biofilm_thickness=initial_thickness_um * 1e-6
    )
    
    if custom_bio_params:
        for key, value in custom_bio_params.items():
            if hasattr(bio_params, key):
                setattr(bio_params, key, value)
    
    return BiologicalCathodeModel(cathode_params, bio_params)