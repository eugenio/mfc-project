#!/usr/bin/env python3
"""
Membrane fouling and degradation models

Comprehensive fouling mechanisms for MFC membranes including:
- Biological fouling (biofilm formation)
- Chemical fouling (precipitation, organic adsorption)
- Physical fouling (particle deposition)
- Degradation (chemical, mechanical, thermal)

Created: 2025-07-27
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class FoulingType(Enum):
    """Types of membrane fouling."""
    BIOLOGICAL = "biological"
    CHEMICAL = "chemical"
    PHYSICAL = "physical"
    DEGRADATION = "degradation"


@dataclass
class FoulingParameters:
    """Parameters for membrane fouling models."""
    
    # Biological fouling
    biofilm_growth_rate: float = 0.1        # h⁻¹ - biofilm growth rate
    biofilm_detachment_rate: float = 0.01   # h⁻¹ - detachment rate
    biofilm_conductivity: float = 1e-7      # S/m - biofilm conductivity
    biofilm_thickness_max: float = 100e-6   # m - maximum thickness
    substrate_concentration: float = 0.01    # mol/L - for biofilm growth
    
    # Chemical fouling
    precipitation_rate: float = 1e-8         # mol/m²/s - salt precipitation
    organic_adsorption_rate: float = 1e-9    # mol/m²/s - organic matter
    scaling_tendency: float = 0.5            # Langelier saturation index
    fouling_layer_porosity: float = 0.3     # Porosity of fouling layer
    
    # Physical fouling
    particle_deposition_rate: float = 1e-10  # kg/m²/s - particle flux
    particle_size: float = 1e-6              # m - average particle size
    particle_density: float = 2000           # kg/m³ - particle density
    shear_rate: float = 100                  # s⁻¹ - wall shear rate
    
    # Degradation parameters
    thermal_degradation_rate: float = 1e-6   # h⁻¹ at reference temperature
    chemical_degradation_rate: float = 1e-7  # h⁻¹ - chemical attack
    mechanical_fatigue_rate: float = 1e-8    # cycle⁻¹ - mechanical stress
    radiation_damage_rate: float = 1e-10     # h⁻¹ - UV/gamma radiation
    
    # Environmental factors
    temperature: float = 298.15              # K - operating temperature
    ph: float = 7.0                         # Solution pH
    ionic_strength: float = 0.1              # mol/L - solution ionic strength
    dissolved_oxygen: float = 8e-3           # mol/L - DO concentration


class FoulingModel:
    """
    Comprehensive membrane fouling model.
    
    Models multiple fouling mechanisms and their interactions:
    - Biological fouling with biofilm dynamics
    - Chemical precipitation and scaling
    - Physical particle deposition
    - Various degradation mechanisms
    """
    
    def __init__(self, parameters: FoulingParameters):
        self.params = parameters
        
        # Fouling state variables
        self.biofilm_thickness = 0.0         # m
        self.chemical_layer_thickness = 0.0  # m
        self.particle_layer_thickness = 0.0  # m
        self.degradation_fraction = 0.0      # Fractional property loss
        
        # Historical tracking
        self.fouling_history = []
        self.resistance_history = []
        self.operating_time = 0.0            # hours
        
        # Initialize fouling mechanisms
        self._setup_fouling_mechanisms()
    
    def _setup_fouling_mechanisms(self):
        """Initialize fouling mechanism parameters."""
        # Biofilm properties
        self.biofilm_density = 1200  # kg/m³ wet biofilm
        self.biofilm_permeability = 1e-15  # m² - hydraulic permeability
        
        # Chemical fouling
        self.precipitation_density = 2500  # kg/m³ - salt deposits
        
        # Particle fouling
        self.cake_porosity = 0.4  # Particle cake porosity
    
    def calculate_biofilm_growth(self, dt_hours: float, nutrient_conc: float,
                                current_density: float = 0.0) -> float:
        """
        Calculate biofilm growth using Monod kinetics.
        
        Args:
            dt_hours: Time step (hours)
            nutrient_conc: Nutrient concentration (mol/L)
            current_density: Current density (A/m²)
        
        Returns:
            Biofilm thickness change (m)
        """
        # Monod growth kinetics
        K_s = 0.001  # mol/L - half-saturation constant
        growth_rate = self.params.biofilm_growth_rate * nutrient_conc / (K_s + nutrient_conc)
        
        # Current density effect (can be positive or negative)
        if current_density > 0:
            # Cathodic conditions might promote certain microbes
            current_factor = 1.0 + 0.1 * jnp.tanh(current_density / 1000)
        else:
            # Anodic conditions might inhibit
            current_factor = 1.0 - 0.05 * jnp.tanh(abs(current_density) / 1000)
        
        # Growth term
        growth_term = growth_rate * current_factor * self.biofilm_thickness
        
        # Detachment term
        detachment_term = self.params.biofilm_detachment_rate * self.biofilm_thickness
        
        # Net growth
        dL_dt = growth_term - detachment_term
        
        # Update thickness
        thickness_change = dL_dt * dt_hours
        new_thickness = self.biofilm_thickness + thickness_change
        
        # Limit to maximum
        self.biofilm_thickness = min(new_thickness, self.params.biofilm_thickness_max)
        
        return thickness_change
    
    def calculate_chemical_fouling(self, dt_hours: float, ion_concentrations: Dict[str, float],
                                 temperature: float) -> float:
        """
        Calculate chemical fouling (precipitation, scaling).
        
        Args:
            dt_hours: Time step (hours)
            ion_concentrations: Ion concentrations (mol/L)
            temperature: Temperature (K)
        
        Returns:
            Chemical layer thickness change (m)
        """
        # Calculate precipitation tendency
        Ca_conc = ion_concentrations.get('Ca2+', 0.001)  # mol/L
        CO3_conc = ion_concentrations.get('CO3--', 0.001)
        
        # Simplified CaCO3 solubility product
        Ksp = 3.3e-9  # mol²/L² at 25°C
        
        # Temperature effect on solubility
        T_ref = 298.15
        dH_sol = -12000  # J/mol - dissolution enthalpy
        R = 8.314
        
        Ksp_T = Ksp * jnp.exp(-dH_sol / R * (1/temperature - 1/T_ref))
        
        # Supersaturation ratio
        Q = Ca_conc * CO3_conc
        S = Q / Ksp_T
        
        # Precipitation rate (only if supersaturated)
        if S > 1.0:
            precip_rate = self.params.precipitation_rate * (S - 1.0) ** 2
        else:
            precip_rate = 0.0
        
        # Convert to thickness (assuming density)
        molar_volume = 0.037  # L/mol for CaCO3
        thickness_change = precip_rate * dt_hours * 3600 * molar_volume / 1000  # m
        
        self.chemical_layer_thickness += thickness_change
        
        return thickness_change
    
    def calculate_particle_fouling(self, dt_hours: float, particle_concentration: float,
                                 flow_velocity: float) -> float:
        """
        Calculate particle deposition fouling.
        
        Args:
            dt_hours: Time step (hours)
            particle_concentration: Particle concentration (kg/m³)
            flow_velocity: Flow velocity (m/s)
        
        Returns:
            Particle layer thickness change (m)
        """
        # Deposition velocity (Stokes settling + convective transport)
        particle_radius = self.params.particle_size / 2
        mu = 1e-3  # Pa·s water viscosity
        g = 9.81  # m/s²
        
        # Settling velocity
        v_settling = (2 * particle_radius**2 * self.params.particle_density * g) / (9 * mu)
        
        # Convective mass transfer
        Re = self.params.particle_density * flow_velocity * 2 * particle_radius / mu
        Sc = mu / (self.params.particle_density * 1e-9)  # Assuming D~1e-9 m²/s
        
        if Re > 0:
            Sh = 0.664 * Re**0.5 * Sc**(1/3)  # Sherwood number
            k_conv = Sh * 1e-9 / (2 * particle_radius)  # Mass transfer coefficient
        else:
            k_conv = 0.0
        
        # Total deposition velocity
        v_dep = v_settling + k_conv
        
        # Deposition flux
        deposition_flux = v_dep * particle_concentration  # kg/m²/s
        
        # Convert to thickness
        thickness_change = (deposition_flux * dt_hours * 3600 / 
                          (self.params.particle_density * (1 - self.cake_porosity)))
        
        self.particle_layer_thickness += thickness_change
        
        return thickness_change
    
    def calculate_thermal_degradation(self, dt_hours: float, temperature: float) -> float:
        """
        Calculate thermal degradation.
        
        Args:
            dt_hours: Time step (hours)
            temperature: Temperature (K)
        
        Returns:
            Degradation increment
        """
        # Arrhenius temperature dependence
        T_ref = 298.15
        Ea = 80000  # J/mol - activation energy for polymer degradation
        R = 8.314
        
        rate_factor = jnp.exp(-Ea / R * (1/temperature - 1/T_ref))
        
        degradation_rate = self.params.thermal_degradation_rate * rate_factor
        degradation_increment = degradation_rate * dt_hours
        
        return degradation_increment
    
    def calculate_chemical_degradation(self, dt_hours: float, ph: float,
                                     oxidizing_species: float = 0.0) -> float:
        """
        Calculate chemical degradation.
        
        Args:
            dt_hours: Time step (hours)
            ph: Solution pH
            oxidizing_species: Concentration of oxidizing species (mol/L)
        
        Returns:
            Degradation increment
        """
        # pH effect (extreme pH accelerates degradation)
        ph_factor = jnp.exp(abs(ph - 7.0) / 3.0)
        
        # Oxidizing species effect
        oxidation_factor = 1.0 + 10.0 * oxidizing_species
        
        degradation_rate = (self.params.chemical_degradation_rate * 
                          ph_factor * oxidation_factor)
        
        degradation_increment = degradation_rate * dt_hours
        
        return degradation_increment
    
    def calculate_total_resistance(self, membrane_resistance: float) -> Dict[str, float]:
        """
        Calculate total membrane resistance including fouling.
        
        Args:
            membrane_resistance: Clean membrane resistance (Ω·m²)
        
        Returns:
            Resistance breakdown
        """
        # Biofilm resistance
        if self.biofilm_thickness > 0:
            R_biofilm = self.biofilm_thickness / self.params.biofilm_conductivity
        else:
            R_biofilm = 0.0
        
        # Chemical fouling resistance
        if self.chemical_layer_thickness > 0:
            # Assume low conductivity scaling layer
            scaling_conductivity = 1e-6  # S/m
            R_chemical = self.chemical_layer_thickness / scaling_conductivity
        else:
            R_chemical = 0.0
        
        # Particle layer resistance
        if self.particle_layer_thickness > 0:
            # Resistive particle cake
            cake_conductivity = 1e-5  # S/m
            R_particle = self.particle_layer_thickness / cake_conductivity  
        else:
            R_particle = 0.0
        
        # Degradation effect (reduces membrane conductivity)
        degradation_factor = 1.0 + 5.0 * self.degradation_fraction
        R_membrane_degraded = membrane_resistance * degradation_factor
        
        # Total resistance (series)
        R_total = R_membrane_degraded + R_biofilm + R_chemical + R_particle
        
        return {
            'clean_membrane_resistance': float(membrane_resistance),
            'degraded_membrane_resistance': float(R_membrane_degraded),
            'biofilm_resistance': float(R_biofilm),
            'chemical_fouling_resistance': float(R_chemical),
            'particle_fouling_resistance': float(R_particle),
            'total_resistance': float(R_total),
            'fouling_resistance_fraction': float((R_total - membrane_resistance) / membrane_resistance)
        }
    
    def update_fouling(self, dt_hours: float, operating_conditions: Dict[str, Any]):
        """
        Update all fouling mechanisms.
        
        Args:
            dt_hours: Time step (hours)
            operating_conditions: Operating conditions dictionary
        """
        # Extract conditions
        temperature = operating_conditions.get('temperature', self.params.temperature)
        ph = operating_conditions.get('ph', self.params.ph)
        nutrient_conc = operating_conditions.get('nutrient_concentration', 
                                               self.params.substrate_concentration)
        current_density = operating_conditions.get('current_density', 0.0)
        ion_concentrations = operating_conditions.get('ion_concentrations', {})
        particle_conc = operating_conditions.get('particle_concentration', 0.001)
        flow_velocity = operating_conditions.get('flow_velocity', 0.1)
        oxidizing_species = operating_conditions.get('oxidizing_species', 0.0)
        
        # Update fouling mechanisms
        biofilm_change = self.calculate_biofilm_growth(dt_hours, nutrient_conc, current_density)
        chemical_change = self.calculate_chemical_fouling(dt_hours, ion_concentrations, temperature)
        particle_change = self.calculate_particle_fouling(dt_hours, particle_conc, flow_velocity)
        
        # Update degradation
        thermal_deg = self.calculate_thermal_degradation(dt_hours, temperature)
        chemical_deg = self.calculate_chemical_degradation(dt_hours, ph, oxidizing_species)
        
        self.degradation_fraction += thermal_deg + chemical_deg
        self.degradation_fraction = min(self.degradation_fraction, 0.9)  # Max 90% degradation
        
        # Update operating time
        self.operating_time += dt_hours
        
        # Store history
        fouling_state = {
            'time_hours': self.operating_time,
            'biofilm_thickness_um': self.biofilm_thickness * 1e6,
            'chemical_layer_thickness_um': self.chemical_layer_thickness * 1e6,
            'particle_layer_thickness_um': self.particle_layer_thickness * 1e6,
            'degradation_fraction': self.degradation_fraction,
            'biofilm_change_um': biofilm_change * 1e6,
            'chemical_change_um': chemical_change * 1e6,
            'particle_change_um': particle_change * 1e6
        }
        
        self.fouling_history.append(fouling_state)
    
    def predict_fouling_trajectory(self, simulation_hours: float,
                                 operating_conditions: Dict[str, Any],
                                 time_step: float = 1.0) -> Dict[str, Any]:
        """
        Predict fouling development over time.
        
        Args:
            simulation_hours: Total simulation time (hours)
            operating_conditions: Operating conditions
            time_step: Time step for simulation (hours)
        
        Returns:
            Fouling trajectory data
        """
        # Store initial state
        initial_biofilm = self.biofilm_thickness
        initial_chemical = self.chemical_layer_thickness
        initial_particle = self.particle_layer_thickness
        initial_degradation = self.degradation_fraction
        initial_time = self.operating_time
        
        # Simulation arrays
        n_steps = int(simulation_hours / time_step)
        time_array = np.linspace(0, simulation_hours, n_steps)
        biofilm_array = np.zeros(n_steps)
        chemical_array = np.zeros(n_steps)
        particle_array = np.zeros(n_steps)
        degradation_array = np.zeros(n_steps)
        resistance_array = np.zeros(n_steps)
        
        # Reference membrane resistance
        base_resistance = 0.1  # Ω·m² (typical)
        
        # Run simulation
        for i, t in enumerate(time_array):
            if i > 0:
                self.update_fouling(time_step, operating_conditions)
            
            # Store state
            biofilm_array[i] = self.biofilm_thickness * 1e6  # μm
            chemical_array[i] = self.chemical_layer_thickness * 1e6
            particle_array[i] = self.particle_layer_thickness * 1e6
            degradation_array[i] = self.degradation_fraction
            
            # Calculate resistance
            resistance_data = self.calculate_total_resistance(base_resistance)
            resistance_array[i] = resistance_data['total_resistance']
        
        # Restore initial state
        self.biofilm_thickness = initial_biofilm
        self.chemical_layer_thickness = initial_chemical
        self.particle_layer_thickness = initial_particle
        self.degradation_fraction = initial_degradation
        self.operating_time = initial_time
        
        return {
            'time_hours': time_array.tolist(),
            'biofilm_thickness_um': biofilm_array.tolist(),
            'chemical_layer_thickness_um': chemical_array.tolist(),
            'particle_layer_thickness_um': particle_array.tolist(),
            'degradation_fraction': degradation_array.tolist(),
            'total_resistance_ohm_m2': resistance_array.tolist(),
            'final_fouling_thickness_um': float((biofilm_array[-1] + 
                                               chemical_array[-1] + 
                                               particle_array[-1])),
            'resistance_increase_factor': float(resistance_array[-1] / base_resistance)
        }
    
    def get_cleaning_effectiveness(self, cleaning_method: str) -> Dict[str, float]:
        """
        Calculate cleaning effectiveness for different methods.
        
        Args:
            cleaning_method: Cleaning method type
        
        Returns:
            Cleaning effectiveness metrics
        """
        cleaning_methods = {
            'chemical_cleaning': {
                'biofilm_removal': 0.9,
                'chemical_removal': 0.8,
                'particle_removal': 0.6,
                'cost_per_m2': 10.0,
                'downtime_hours': 4.0
            },
            'backwash': {
                'biofilm_removal': 0.3,
                'chemical_removal': 0.1,
                'particle_removal': 0.8,
                'cost_per_m2': 2.0,
                'downtime_hours': 0.5
            },
            'ultrasonic_cleaning': {
                'biofilm_removal': 0.8,
                'chemical_removal': 0.5,
                'particle_removal': 0.9,
                'cost_per_m2': 15.0,
                'downtime_hours': 2.0
            },
            'electrochemical_cleaning': {
                'biofilm_removal': 0.95,
                'chemical_removal': 0.7,
                'particle_removal': 0.4,
                'cost_per_m2': 8.0,
                'downtime_hours': 1.0
            }
        }
        
        if cleaning_method not in cleaning_methods:
            return {'error': f"Unknown cleaning method: {cleaning_method}"}
        
        effectiveness = cleaning_methods[cleaning_method]
        
        # Calculate cleaning benefit
        current_resistance = self.calculate_total_resistance(0.1)['total_resistance']
        
        # Simulate cleaning
        biofilm_removed = self.biofilm_thickness * effectiveness['biofilm_removal']
        chemical_removed = self.chemical_layer_thickness * effectiveness['chemical_removal']
        particle_removed = self.particle_layer_thickness * effectiveness['particle_removal']
        
        # Resistance after cleaning
        temp_biofilm = self.biofilm_thickness
        temp_chemical = self.chemical_layer_thickness
        temp_particle = self.particle_layer_thickness
        
        self.biofilm_thickness -= biofilm_removed
        self.chemical_layer_thickness -= chemical_removed
        self.particle_layer_thickness -= particle_removed
        
        cleaned_resistance = self.calculate_total_resistance(0.1)['total_resistance']
        
        # Restore state
        self.biofilm_thickness = temp_biofilm
        self.chemical_layer_thickness = temp_chemical
        self.particle_layer_thickness = temp_particle
        
        return {
            'cleaning_method': cleaning_method,
            'biofilm_removal_percent': effectiveness['biofilm_removal'] * 100,
            'chemical_removal_percent': effectiveness['chemical_removal'] * 100,
            'particle_removal_percent': effectiveness['particle_removal'] * 100,
            'resistance_before_ohm_m2': float(current_resistance),
            'resistance_after_ohm_m2': float(cleaned_resistance),
            'resistance_reduction_percent': float((current_resistance - cleaned_resistance) / 
                                                current_resistance * 100),
            'cleaning_cost_per_m2': effectiveness['cost_per_m2'],
            'downtime_hours': effectiveness['downtime_hours']
        }
    
    def get_fouling_status(self) -> Dict[str, Any]:
        """Get current fouling status."""
        total_fouling_thickness = (self.biofilm_thickness + 
                                 self.chemical_layer_thickness + 
                                 self.particle_layer_thickness)
        
        # Fouling severity assessment
        if total_fouling_thickness < 10e-6:  # < 10 μm
            severity = "Low"
        elif total_fouling_thickness < 50e-6:  # < 50 μm
            severity = "Moderate"
        elif total_fouling_thickness < 100e-6:  # < 100 μm
            severity = "High"
        else:
            severity = "Severe"
        
        return {
            'biofilm_thickness_um': float(self.biofilm_thickness * 1e6),
            'chemical_layer_thickness_um': float(self.chemical_layer_thickness * 1e6),
            'particle_layer_thickness_um': float(self.particle_layer_thickness * 1e6),
            'total_fouling_thickness_um': float(total_fouling_thickness * 1e6),
            'degradation_fraction': float(self.degradation_fraction),
            'operating_time_hours': float(self.operating_time),
            'fouling_severity': severity,
            'dominant_fouling_type': self._get_dominant_fouling_type(),
            'cleaning_recommended': total_fouling_thickness > 50e-6
        }
    
    def _get_dominant_fouling_type(self) -> str:
        """Determine the dominant fouling mechanism."""
        thicknesses = {
            'biofilm': self.biofilm_thickness,
            'chemical': self.chemical_layer_thickness,
            'particle': self.particle_layer_thickness
        }
        
        dominant = max(thicknesses, key=thicknesses.get)
        return dominant


def calculate_fouling_resistance(fouling_thickness: float,
                               fouling_conductivity: float) -> float:
    """
    Calculate resistance of a fouling layer.
    
    Args:
        fouling_thickness: Thickness of fouling layer (m)
        fouling_conductivity: Conductivity of fouling layer (S/m)
    
    Returns:
        Fouling resistance (Ω·m²)
    """
    if fouling_conductivity <= 0:
        return float('inf')
    
    return fouling_thickness / fouling_conductivity