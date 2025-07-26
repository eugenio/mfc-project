#!/usr/bin/env python3
"""
Platinum cathode model for MFC simulations

Implements Butler-Volmer kinetics for platinum-based oxygen reduction reaction (ORR)
with literature-based parameters and temperature dependency.

Literature-derived default parameters:
- Exchange current density: 3.0 × 10⁻⁹ A/cm² (Khan et al., 2024)
- Tafel slope: 60 mV/decade (low η), 120 mV/decade (high η)
- Platinum loading: 0.5 mg/cm² (typical commercial)
- Activation energy: 20 kJ/mol (ORR on Pt)

Created: 2025-07-26
"""

import jax.numpy as jnp
from typing import Optional, Dict, Any
from dataclasses import dataclass

from .base_cathode import BaseCathodeModel, CathodeParameters, ButlerVolmerKinetics


@dataclass
class PlatinumParameters:
    """Parametrizable platinum cathode parameters with literature-derived defaults"""
    
    # Kinetic parameters (literature-derived)
    exchange_current_density_ref: float = 3.0e-5  # A/m² (3.0e-9 A/cm²) - Khan et al. 2024
    transfer_coefficient: float = 0.5  # Dimensionless - typical for ORR
    
    # Tafel slopes (literature values)
    tafel_slope_low: float = 0.060   # V/decade - low overpotential regime
    tafel_slope_high: float = 0.120  # V/decade - high overpotential regime  
    overpotential_transition: float = 0.100  # V - transition between regimes
    
    # Catalyst properties
    platinum_loading: float = 0.5  # mg/cm² - typical commercial loading
    platinum_surface_area: float = 50.0  # m²/g - specific surface area
    platinum_utilization: float = 0.7  # Fraction of Pt electrochemically active
    
    # Temperature dependency (Arrhenius parameters)
    activation_energy: float = 20000.0  # J/mol - ORR activation energy on Pt
    reference_temperature: float = 298.15  # K - reference temperature
    
    # Physical constants (literature values)
    oxygen_diffusion_coeff: float = 2.1e-9  # m²/s - O₂ in water at 25°C
    oxygen_solubility: float = 1.3e-3  # mol/L - O₂ solubility in water at 1 atm
    
    # Performance benchmarks (literature)
    reference_power_density: float = 400.0  # mW/m² - typical Pt/C performance
    reference_current_density_at_0_7V: float = 1000.0  # A/m² - benchmark at 0.7V vs SHE
    
    # Economic parameters
    platinum_price_per_kg: float = 30000.0  # $/kg - approximate market price
    manufacturing_cost_factor: float = 2.5  # Multiplier for total cost vs material
    
    # Mass transport parameters
    boundary_layer_thickness: float = 100e-6  # m - diffusion boundary layer
    oxygen_mass_transfer_coeff: float = 1e-5  # m/s - typical for stirred conditions
    
    # Electrode structure parameters
    porosity: float = 0.4  # Void fraction in catalyst layer
    tortuosity: float = 2.5  # Diffusion path tortuosity
    catalyst_layer_thickness: float = 10e-6  # m - typical thickness


class PlatinumCathodeModel(BaseCathodeModel):
    """
    Platinum cathode model with parametrizable literature-based constants.
    
    Features:
    - All parameters configurable with literature defaults
    - Temperature-dependent kinetics (Arrhenius)
    - Mass transport limitations
    - Dual Tafel slope regions
    - Economic cost estimation
    - Performance benchmarking
    """
    
    def __init__(self, parameters: CathodeParameters, platinum_params: Optional[PlatinumParameters] = None):
        self.pt_params = platinum_params or PlatinumParameters()
        super().__init__(parameters)
        self._setup_mass_transport_parameters()
    
    def _setup_kinetic_parameters(self):
        """Setup temperature-dependent kinetic parameters."""
        # Temperature-corrected exchange current density (Arrhenius equation)
        R = self.params.gas_constant
        T = self.temperature_K
        T_ref = self.pt_params.reference_temperature
        Ea = self.pt_params.activation_energy
        
        arrhenius_factor = jnp.exp(-Ea / R * (1.0 / T - 1.0 / T_ref))
        
        # Base exchange current density
        i0_base = self.pt_params.exchange_current_density_ref * arrhenius_factor
        
        # Account for platinum utilization and surface area
        utilization_factor = self.pt_params.platinum_utilization
        surface_area_factor = self.pt_params.platinum_surface_area / 50.0  # Normalized to 50 m²/g
        loading_factor = self.pt_params.platinum_loading / 0.5  # Normalized to 0.5 mg/cm²
        
        self.exchange_current_density = i0_base * utilization_factor * jnp.sqrt(surface_area_factor * loading_factor)
        
        # Temperature-corrected diffusion coefficient
        # D(T) = D_ref * (T/T_ref)^1.5 * (μ_ref/μ(T))
        temp_ratio = T / T_ref
        self.oxygen_diffusion_coeff = self.pt_params.oxygen_diffusion_coeff * (temp_ratio ** 1.5)
        
        # Store other parameters
        self.transfer_coefficient = self.pt_params.transfer_coefficient
        self.tafel_slope_low = self.pt_params.tafel_slope_low
        self.tafel_slope_high = self.pt_params.tafel_slope_high
        self.overpotential_transition = self.pt_params.overpotential_transition
    
    def _setup_mass_transport_parameters(self):
        """Setup mass transport parameters."""
        # Calculate limiting current density based on mass transport
        n_electrons = 4.0  # O₂ + 4H⁺ + 4e⁻ → 2H₂O
        F = self.params.faraday_constant
        
        # Limiting current from diffusion through boundary layer
        self.limiting_current_density = (n_electrons * F * 
                                       self.pt_params.oxygen_mass_transfer_coeff * 
                                       self.params.oxygen_concentration)
        
        # Account for porosity and tortuosity in catalyst layer
        effective_diffusivity = (self.oxygen_diffusion_coeff * 
                                self.pt_params.porosity / 
                                self.pt_params.tortuosity)
        
        # Additional limiting current from catalyst layer diffusion
        catalyst_layer_limiting = (n_electrons * F * effective_diffusivity * 
                                 self.params.oxygen_concentration / 
                                 self.pt_params.catalyst_layer_thickness)
        
        # Overall limiting current (harmonic mean of resistances)
        self.limiting_current_density = (1.0 / (1.0/self.limiting_current_density + 
                                               1.0/catalyst_layer_limiting)) ** -1
    
    def calculate_current_density(self, overpotential: float, 
                                oxygen_conc: Optional[float] = None) -> float:
        """
        Calculate current density with mass transport limitations.
        
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
        
        # Concentration ratio effect
        concentration_ratio = jnp.maximum(oxygen_conc / self.params.oxygen_reference_conc, 0.01)
        
        # Calculate kinetic current density
        if overpotential <= self.overpotential_transition:
            # Low overpotential: Butler-Volmer
            i_kinetic = ButlerVolmerKinetics.calculate_current_density(
                exchange_current_density=self.exchange_current_density,
                transfer_coefficient=self.transfer_coefficient,
                overpotential=overpotential,
                temperature_K=self.temperature_K,
                concentration_ratio=float(concentration_ratio)
            )
        else:
            # High overpotential: Mixed Butler-Volmer and Tafel
            i_transition = ButlerVolmerKinetics.calculate_current_density(
                exchange_current_density=self.exchange_current_density,
                transfer_coefficient=self.transfer_coefficient,
                overpotential=self.overpotential_transition,
                temperature_K=self.temperature_K,
                concentration_ratio=float(concentration_ratio)
            )
            
            extra_overpotential = overpotential - self.overpotential_transition
            i_kinetic = ButlerVolmerKinetics.calculate_tafel_current(
                exchange_current_density=i_transition,
                tafel_slope=self.tafel_slope_high,
                overpotential=extra_overpotential,
                concentration_ratio=1.0
            )
        
        # Apply mass transport limitation only if significant
        i_limiting = self.limiting_current_density * concentration_ratio
        
        # Only apply mass transport limitation if kinetic current approaches limiting current
        if i_kinetic > 0.1 * i_limiting:
            # Combined kinetic and mass transport (using resistor analogy)
            # 1/i_total = 1/i_kinetic + 1/i_limiting
            current_density = (i_kinetic * i_limiting) / (i_kinetic + i_limiting)
        else:
            # At low currents, kinetics dominates
            current_density = i_kinetic
        
        return current_density
    
    def calculate_performance_metrics(self, operating_overpotential: float,
                                    oxygen_conc: Optional[float] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        current_density = self.calculate_current_density(operating_overpotential, oxygen_conc)
        current = current_density * self.area_m2
        power_loss = self.calculate_power_loss(operating_overpotential, oxygen_conc)
        
        # Performance metrics
        power_density_mW_m2 = power_loss * 1000 / self.area_m2
        
        # Efficiency calculations
        equilibrium_potential = self.calculate_equilibrium_potential(oxygen_conc)
        voltage_efficiency = (equilibrium_potential - operating_overpotential) / equilibrium_potential * 100
        
        # Mass transport analysis
        kinetic_limited = current_density < (0.8 * self.limiting_current_density)
        mass_transport_utilization = current_density / self.limiting_current_density * 100
        
        # Oxygen utilization
        o2_consumption_rate = self.calculate_oxygen_consumption_rate(operating_overpotential, oxygen_conc)
        o2_flux_mol_m2_s = o2_consumption_rate / self.area_m2
        
        # Platinum utilization metrics
        theoretical_max_current = self.exchange_current_density * 1000  # Rough estimate
        pt_utilization_percent = current_density / theoretical_max_current * 100
        
        return {
            'current_density_A_m2': float(current_density),
            'current_A': float(current),
            'power_loss_W': float(power_loss),
            'power_density_mW_m2': float(power_density_mW_m2),
            'voltage_efficiency_percent': float(voltage_efficiency),
            'overpotential_V': float(operating_overpotential),
            'equilibrium_potential_V': float(equilibrium_potential),
            'kinetic_limited': bool(kinetic_limited),
            'mass_transport_utilization_percent': float(mass_transport_utilization),
            'limiting_current_density_A_m2': float(self.limiting_current_density),
            'o2_consumption_rate_mol_s': float(o2_consumption_rate),
            'o2_flux_mol_m2_s': float(o2_flux_mol_m2_s),
            'platinum_utilization_percent': float(pt_utilization_percent),
            'exchange_current_density_A_m2': float(self.exchange_current_density)
        }
    
    def estimate_cost_per_area(self) -> float:
        """
        Estimate platinum cathode cost per unit area.
        
        Returns:
            Cost in $/m²
        """
        # Material costs
        pt_loading_kg_m2 = self.pt_params.platinum_loading * 1e-6 * 1e4  # mg/cm² to kg/m²
        material_cost_per_m2 = pt_loading_kg_m2 * self.pt_params.platinum_price_per_kg
        
        # Total cost including manufacturing
        total_cost_per_m2 = material_cost_per_m2 * self.pt_params.manufacturing_cost_factor
        
        return total_cost_per_m2

    def estimate_cost_analysis(self) -> Dict[str, float]:
        """Comprehensive cost analysis."""
        # Material costs
        pt_loading_kg_m2 = self.pt_params.platinum_loading * 1e-6 * 1e4  # mg/cm² to kg/m²
        material_cost_per_m2 = pt_loading_kg_m2 * self.pt_params.platinum_price_per_kg
        
        # Total cost including manufacturing
        total_cost_per_m2 = material_cost_per_m2 * self.pt_params.manufacturing_cost_factor
        
        # Cost per unit power at reference conditions
        ref_metrics = self.calculate_performance_metrics(0.2)  # 200 mV overpotential
        cost_per_kW = total_cost_per_m2 / (ref_metrics['power_density_mW_m2'] * 1e-6)  # $/kW
        
        return {
            'material_cost_per_m2': float(material_cost_per_m2),
            'total_cost_per_m2': float(total_cost_per_m2),
            'cost_per_kW': float(cost_per_kW),
            'platinum_loading_kg_m2': float(pt_loading_kg_m2),
            'platinum_price_per_kg': self.pt_params.platinum_price_per_kg,
            'manufacturing_factor': self.pt_params.manufacturing_cost_factor
        }
    
    def compare_to_benchmark(self) -> Dict[str, Any]:
        """
        Compare current parameters to literature benchmarks.
        
        Returns:
            Dictionary with comparison metrics
        """
        # Use standard conditions for comparison
        standard_overpotential = 0.2  # V (typical operating point)
        
        metrics = self.calculate_performance_metrics(standard_overpotential)
        
        # Literature benchmarks
        benchmark_power_density = self.pt_params.reference_power_density  # mW/m²
        benchmark_exchange_current = self.pt_params.exchange_current_density_ref  # A/m²
        
        return {
            'performance_vs_benchmark': {
                'power_density_ratio': metrics['power_density_mW_m2'] / benchmark_power_density,
                'exchange_current_ratio': metrics['exchange_current_density_A_m2'] / benchmark_exchange_current,
                'benchmark_power_density_mW_m2': benchmark_power_density,
                'actual_power_density_mW_m2': metrics['power_density_mW_m2'],
                'benchmark_i0_A_m2': benchmark_exchange_current,
                'actual_i0_A_m2': metrics['exchange_current_density_A_m2']
            },
            'operating_conditions': {
                'temperature_C': self.temperature_K - 273.15,
                'oxygen_mg_L': self.params.oxygen_concentration * 32.0 * 1000,
                'cathode_area_cm2': self.area_m2 * 1e4
            },
            'cost_analysis': {
                'estimated_cost_per_m2': self.estimate_cost_per_area(),
                'platinum_loading_mg_cm2': self.pt_params.platinum_loading
            }
        }
    
    def get_all_parameters(self) -> Dict[str, Any]:
        """Get all configurable parameters with current values."""
        return {
            'cathode_parameters': {
                'area_m2': self.area_m2,
                'area_cm2': self.area_m2 * 1e4,
                'temperature_K': self.temperature_K,
                'temperature_C': self.temperature_K - 273.15,
                'oxygen_concentration_mol_L': self.params.oxygen_concentration,
                'oxygen_concentration_mg_L': self.params.oxygen_concentration * 32.0 * 1000,
                'ph': self.params.ph
            },
            'platinum_kinetic_parameters': {
                'exchange_current_density_ref_A_m2': self.pt_params.exchange_current_density_ref,
                'exchange_current_density_actual_A_m2': float(self.exchange_current_density),
                'transfer_coefficient': self.pt_params.transfer_coefficient,
                'tafel_slope_low_mV_decade': self.pt_params.tafel_slope_low * 1000,
                'tafel_slope_high_mV_decade': self.pt_params.tafel_slope_high * 1000,
                'overpotential_transition_mV': self.pt_params.overpotential_transition * 1000,
                'activation_energy_kJ_mol': self.pt_params.activation_energy / 1000
            },
            'catalyst_parameters': {
                'platinum_loading_mg_cm2': self.pt_params.platinum_loading,
                'platinum_surface_area_m2_g': self.pt_params.platinum_surface_area,
                'platinum_utilization': self.pt_params.platinum_utilization,
                'porosity': self.pt_params.porosity,
                'tortuosity': self.pt_params.tortuosity,
                'catalyst_layer_thickness_um': self.pt_params.catalyst_layer_thickness * 1e6
            },
            'mass_transport_parameters': {
                'oxygen_diffusion_coeff_m2_s': float(self.oxygen_diffusion_coeff),
                'boundary_layer_thickness_um': self.pt_params.boundary_layer_thickness * 1e6,
                'mass_transfer_coeff_m_s': self.pt_params.oxygen_mass_transfer_coeff,
                'limiting_current_density_A_m2': float(self.limiting_current_density)
            },
            'economic_parameters': {
                'platinum_price_per_kg': self.pt_params.platinum_price_per_kg,
                'manufacturing_cost_factor': self.pt_params.manufacturing_cost_factor
            },
            'benchmark_parameters': {
                'reference_power_density_mW_m2': self.pt_params.reference_power_density,
                'reference_current_density_at_0_7V_A_m2': self.pt_params.reference_current_density_at_0_7V
            }
        }


def create_platinum_cathode(area_cm2: float = 1.0, 
                          temperature_C: float = 25.0,
                          oxygen_mg_L: float = 8.0,
                          platinum_loading_mg_cm2: float = 0.5,
                          custom_pt_params: Optional[Dict[str, float]] = None) -> PlatinumCathodeModel:
    """
    Create platinum cathode with configurable parameters.
    
    Args:
        area_cm2: Cathode area in cm²
        temperature_C: Operating temperature in °C  
        oxygen_mg_L: Dissolved oxygen in mg/L
        platinum_loading_mg_cm2: Pt loading in mg/cm²
        custom_pt_params: Custom platinum parameters to override defaults
    
    Returns:
        Configured PlatinumCathodeModel
    """
    # Convert units
    area_m2 = area_cm2 * 1e-4
    temperature_K = temperature_C + 273.15
    oxygen_mol_L = oxygen_mg_L / (32.0 * 1000)
    
    # Create base parameters
    cathode_params = CathodeParameters(
        area_m2=area_m2,
        temperature_K=temperature_K,
        oxygen_concentration=oxygen_mol_L
    )
    
    # Create platinum parameters with custom overrides
    pt_params = PlatinumParameters(platinum_loading=platinum_loading_mg_cm2)
    
    if custom_pt_params:
        for key, value in custom_pt_params.items():
            if hasattr(pt_params, key):
                setattr(pt_params, key, value)
    
    return PlatinumCathodeModel(cathode_params, pt_params)