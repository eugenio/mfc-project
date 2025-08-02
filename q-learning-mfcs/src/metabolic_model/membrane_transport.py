"""
Membrane transport model for Nafion membranes in MFC systems.

This module implements oxygen crossover calculations, proton conductivity,
and mass transport through Nafion membranes based on literature properties
and experimental data.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np

# Add path for GPU acceleration
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gpu_acceleration import get_gpu_accelerator


@dataclass
class MembraneProperties:
    """Container for Nafion membrane properties."""

    # Physical properties
    thickness: float          # μm - membrane thickness
    density: float           # g/cm³ - dry membrane density
    equivalent_weight: float  # g/mol - equivalent weight

    # Transport properties
    proton_conductivity: float     # S/cm - proton conductivity
    oxygen_permeability: float     # mol/m/s/Pa - O2 permeability
    water_uptake: float           # H2O/SO3H - water uptake ratio
    drag_coefficient: float       # H2O/H+ - electro-osmotic drag

    # Temperature dependencies
    conductivity_temp_coeff: float  # S/cm/K - temperature coefficient
    permeability_temp_coeff: float  # activation energy factor

    # Humidity dependencies
    conductivity_humidity_exp: float  # humidity exponent
    water_uptake_humidity_slope: float


class MembraneTransport:
    """
    Comprehensive membrane transport model for MFC applications.
    
    Features:
    - Oxygen crossover calculations with concentration gradients
    - Proton conductivity with temperature and humidity effects
    - Water transport and electro-osmotic drag
    - Membrane resistance calculations
    - GPU acceleration support
    """

    def __init__(self, membrane_grade: str = "Nafion-117", use_gpu: bool = True):
        """
        Initialize membrane transport model.
        
        Args:
            membrane_grade: Nafion membrane grade ("Nafion-117", "Nafion-115", "Nafion-212")
            use_gpu: Enable GPU acceleration
        """
        self.membrane_grade = membrane_grade
        self.use_gpu = use_gpu

        # GPU acceleration setup
        self.gpu_acc = get_gpu_accelerator() if use_gpu else None
        self.gpu_available = self.gpu_acc.is_gpu_available() if self.gpu_acc else False

        # Load membrane properties
        self.properties = self._load_membrane_properties()

        # Physical constants
        self.R = 8.314          # J/(mol·K) - Gas constant
        self.F = 96485          # C/mol - Faraday constant
        self.T_ref = 298.15     # K - Reference temperature
        self.RH_ref = 100.0     # % - Reference relative humidity

    def _load_membrane_properties(self) -> MembraneProperties:
        """Load properties for specified membrane grade."""

        # Nafion membrane database with literature values
        membrane_db = {
            "Nafion-117": MembraneProperties(
                thickness=183.0,              # μm (7.2 mil)
                density=1.98,                 # g/cm³
                equivalent_weight=1100.0,     # g/mol
                proton_conductivity=0.1,      # S/cm at 80°C, 100% RH
                oxygen_permeability=1.8e-15,  # mol/m/s/Pa
                water_uptake=14.0,            # H2O/SO3H at 80°C
                drag_coefficient=2.5,         # H2O/H+
                conductivity_temp_coeff=0.0015,  # S/cm/K
                permeability_temp_coeff=0.3,     # Activation factor
                conductivity_humidity_exp=1.5,   # RH exponent
                water_uptake_humidity_slope=0.12
            ),

            "Nafion-115": MembraneProperties(
                thickness=127.0,              # μm (5.0 mil)
                density=1.98,                 # g/cm³
                equivalent_weight=1100.0,     # g/mol
                proton_conductivity=0.1,      # S/cm at 80°C, 100% RH
                oxygen_permeability=1.8e-15,  # mol/m/s/Pa
                water_uptake=14.0,            # H2O/SO3H at 80°C
                drag_coefficient=2.5,         # H2O/H+
                conductivity_temp_coeff=0.0015,
                permeability_temp_coeff=0.3,
                conductivity_humidity_exp=1.5,
                water_uptake_humidity_slope=0.12
            ),

            "Nafion-212": MembraneProperties(
                thickness=50.8,               # μm (2.0 mil)
                density=1.98,                 # g/cm³
                equivalent_weight=1100.0,     # g/mol
                proton_conductivity=0.1,      # S/cm at 80°C, 100% RH
                oxygen_permeability=1.8e-15,  # mol/m/s/Pa
                water_uptake=14.0,            # H2O/SO3H at 80°C
                drag_coefficient=2.5,         # H2O/H+
                conductivity_temp_coeff=0.0015,
                permeability_temp_coeff=0.3,
                conductivity_humidity_exp=1.5,
                water_uptake_humidity_slope=0.12
            )
        }

        if self.membrane_grade not in membrane_db:
            available = list(membrane_db.keys())
            raise ValueError(f"Membrane grade '{self.membrane_grade}' not available. "
                           f"Available: {available}")

        return membrane_db[self.membrane_grade]

    def calculate_oxygen_crossover(self, anode_o2_conc: float, cathode_o2_conc: float,
                                 temperature: float = 303.0, pressure: float = 101325.0) -> float:
        """
        Calculate oxygen crossover flux through membrane.
        
        Args:
            anode_o2_conc: Oxygen concentration at anode (mol/m³)
            cathode_o2_conc: Oxygen concentration at cathode (mol/m³)
            temperature: Temperature (K)
            pressure: Pressure (Pa)
            
        Returns:
            Oxygen flux (mol/m²/s)
        """
        # Temperature correction for permeability
        temp_factor = np.exp(self.properties.permeability_temp_coeff *
                           (temperature - self.T_ref) / self.T_ref)

        permeability = self.properties.oxygen_permeability * temp_factor

        # Convert thickness to meters
        thickness_m = self.properties.thickness * 1e-6

        # Concentration gradient (driving force)
        if self.gpu_available:
            anode_gpu = self.gpu_acc.array([anode_o2_conc])
            cathode_gpu = self.gpu_acc.array([cathode_o2_conc])

            conc_gradient_gpu = (cathode_gpu - anode_gpu) / thickness_m
            conc_gradient = self.gpu_acc.to_cpu(conc_gradient_gpu)[0]
        else:
            conc_gradient = (cathode_o2_conc - anode_o2_conc) / thickness_m

        # Fick's first law: J = -D * (dC/dx)
        # For membrane: J = P * (C_cathode - C_anode) / thickness
        oxygen_flux = permeability * conc_gradient

        return oxygen_flux

    def calculate_proton_conductivity(self, temperature: float = 303.0,
                                    relative_humidity: float = 100.0) -> float:
        """
        Calculate proton conductivity with temperature and humidity corrections.
        
        Args:
            temperature: Temperature (K)
            relative_humidity: Relative humidity (%)
            
        Returns:
            Proton conductivity (S/cm)
        """
        # Temperature correction
        temp_correction = 1 + self.properties.conductivity_temp_coeff * (temperature - self.T_ref)

        # Humidity correction (empirical relationship)
        humidity_factor = (relative_humidity / self.RH_ref) ** self.properties.conductivity_humidity_exp

        # Base conductivity with corrections
        conductivity = (self.properties.proton_conductivity *
                       temp_correction *
                       humidity_factor)

        return max(0.001, conductivity)  # Minimum conductivity

    def calculate_membrane_resistance(self, area: float, temperature: float = 303.0,
                                    relative_humidity: float = 100.0) -> float:
        """
        Calculate membrane resistance.
        
        Args:
            area: Membrane area (m²)
            temperature: Temperature (K)
            relative_humidity: Relative humidity (%)
            
        Returns:
            Membrane resistance (Ω)
        """
        conductivity = self.calculate_proton_conductivity(temperature, relative_humidity)

        # Convert conductivity from S/cm to S/m
        conductivity_si = conductivity * 100  # S/m

        # Convert thickness to meters
        thickness_m = self.properties.thickness * 1e-6

        # Resistance = ρ * L / A = L / (σ * A)
        resistance = thickness_m / (conductivity_si * area)

        return resistance

    def calculate_water_transport(self, current_density: float, temperature: float = 303.0,
                                relative_humidity: float = 100.0) -> float:
        """
        Calculate water transport rate through electro-osmotic drag.
        
        Args:
            current_density: Current density (A/m²)
            temperature: Temperature (K)
            relative_humidity: Relative humidity (%)
            
        Returns:
            Water flux (mol/m²/s)
        """
        # Water uptake correction for humidity
        water_uptake = (self.properties.water_uptake *
                       (1 + self.properties.water_uptake_humidity_slope *
                        (relative_humidity - self.RH_ref) / 100))

        # Drag coefficient may depend on water uptake
        drag_coefficient = self.properties.drag_coefficient * (water_uptake / 14.0)

        # Proton flux = current_density / F
        proton_flux = current_density / self.F

        # Water flux = drag_coefficient * proton_flux
        water_flux = drag_coefficient * proton_flux

        return water_flux

    def calculate_potential_drop(self, current_density: float, area: float,
                               temperature: float = 303.0,
                               relative_humidity: float = 100.0) -> float:
        """
        Calculate potential drop across membrane.
        
        Args:
            current_density: Current density (A/m²)
            area: Membrane area (m²)
            temperature: Temperature (K)
            relative_humidity: Relative humidity (%)
            
        Returns:
            Potential drop (V)
        """
        resistance = self.calculate_membrane_resistance(area, temperature, relative_humidity)
        current = current_density * area

        # Ohm's law: V = I * R
        potential_drop = current * resistance

        return potential_drop

    def calculate_oxygen_consumption_loss(self, oxygen_flux: float, area: float,
                                        substrate_flux: float) -> float:
        """
        Calculate efficiency loss due to oxygen crossover.
        
        Args:
            oxygen_flux: Oxygen crossover flux (mol/m²/s)
            area: Membrane area (m²)
            substrate_flux: Substrate consumption flux (mol/m²/s)
            
        Returns:
            Efficiency loss fraction (0-1)
        """
        # Total oxygen crossover
        total_o2_crossover = oxygen_flux * area

        # Oxygen consumption reduces available electrons
        # O2 + 4H+ + 4e- → 2H2O
        electrons_lost = 4 * total_o2_crossover  # mol e-/s

        # Total available electrons from substrate (depends on substrate)
        # Assume 4 electrons per substrate (average)
        total_electrons_available = 4 * substrate_flux * area

        # Efficiency loss
        if total_electrons_available > 0:
            efficiency_loss = electrons_lost / total_electrons_available
        else:
            efficiency_loss = 0.0

        return min(1.0, max(0.0, efficiency_loss))

    def get_mass_transport_equations(self) -> Dict[str, str]:
        """Get mass transport equations for documentation."""
        return {
            "oxygen_flux": r"J_{O_2} = P_{O_2} \cdot \frac{C_{cathode} - C_{anode}}{L}",
            "proton_conductivity": r"\sigma_H = \sigma_0 \cdot (1 + \alpha \Delta T) \cdot (RH/100)^n",
            "membrane_resistance": r"R_{mem} = \frac{L}{\sigma_H \cdot A}",
            "water_transport": r"J_{H_2O} = n_d \cdot \frac{j}{F}",
            "potential_drop": r"\Delta V = j \cdot A \cdot R_{mem}"
        }

    def get_membrane_properties(self) -> Dict[str, float]:
        """Get current membrane properties for inspection."""
        return {
            "grade": self.membrane_grade,
            "thickness_um": self.properties.thickness,
            "density": self.properties.density,
            "equivalent_weight": self.properties.equivalent_weight,
            "base_conductivity": self.properties.proton_conductivity,
            "oxygen_permeability": self.properties.oxygen_permeability,
            "water_uptake": self.properties.water_uptake,
            "drag_coefficient": self.properties.drag_coefficient,
            "gpu_available": self.gpu_available
        }

    def optimize_membrane_thickness(self, target_resistance: float, area: float,
                                  temperature: float = 303.0,
                                  relative_humidity: float = 100.0) -> float:
        """
        Calculate optimal membrane thickness for target resistance.
        
        Args:
            target_resistance: Target membrane resistance (Ω)
            area: Membrane area (m²)
            temperature: Temperature (K)
            relative_humidity: Relative humidity (%)
            
        Returns:
            Optimal thickness (μm)
        """
        conductivity = self.calculate_proton_conductivity(temperature, relative_humidity)
        conductivity_si = conductivity * 100  # S/m

        # From R = L / (σ * A), solve for L
        optimal_thickness_m = target_resistance * conductivity_si * area
        optimal_thickness_um = optimal_thickness_m * 1e6

        return optimal_thickness_um

    def calculate_crossover_current_loss(self, oxygen_crossover: float, area: float) -> float:
        """
        Calculate current loss due to oxygen crossover.
        
        Args:
            oxygen_crossover: Oxygen flux (mol/m²/s)
            area: Membrane area (m²)
            
        Returns:
            Current loss (A)
        """
        # Oxygen reduction: O2 + 4H+ + 4e- → 2H2O
        electrons_consumed = 4 * oxygen_crossover * area  # mol e-/s
        current_loss = electrons_consumed * self.F  # A

        return current_loss

    @staticmethod
    def get_available_membranes() -> list:
        """Get list of available membrane grades."""
        return ["Nafion-117", "Nafion-115", "Nafion-212"]
