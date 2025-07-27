#!/usr/bin/env python3
"""
Base cathode model for MFC simulations

Implements Butler-Volmer kinetics for oxygen reduction reaction (ORR)
at the cathode with temperature dependency and oxygen concentration effects.

Created: 2025-07-26
"""

import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CathodeParameters:
    """Parameters for cathode models"""
    area_m2: float = 1e-4  # Cathode area in m² (default: 1 cm²)
    temperature_K: float = 298.15  # Temperature in Kelvin (25°C)
    oxygen_concentration: float = 8.0e-3  # Dissolved oxygen in mol/L (air-saturated water)
    oxygen_reference_conc: float = 8.0e-3  # Reference oxygen concentration
    
    # Physical constants
    faraday_constant: float = 96485.0  # C/mol
    gas_constant: float = 8.314  # J/(mol*K)
    
    # Default pH for calculations
    ph: float = 7.0


class BaseCathodeModel(ABC):
    """
    Abstract base class for cathode models in MFC simulations.
    
    Provides common functionality for all cathode types including:
    - Butler-Volmer kinetics framework
    - Temperature dependency
    - Oxygen concentration effects
    - Power consumption calculations
    """
    
    def __init__(self, parameters: CathodeParameters):
        self.params = parameters
        self.area_m2 = parameters.area_m2
        self.temperature_K = parameters.temperature_K
        
        # Initialize model-specific parameters
        self._setup_kinetic_parameters()
    
    @abstractmethod
    def _setup_kinetic_parameters(self):
        """Setup cathode-specific kinetic parameters. Must be implemented by subclasses."""
        pass
    
    def calculate_equilibrium_potential(self, oxygen_conc: Optional[float] = None, ph: Optional[float] = None) -> float:
        """
        Calculate equilibrium potential for O2/H2O reaction using Nernst equation.
        
        O2 + 4H+ + 4e- -> 2H2O
        
        Args:
            oxygen_conc: Oxygen concentration in mol/L
            ph: Solution pH
        
        Returns:
            Equilibrium potential in V vs SHE
        """
        if oxygen_conc is None:
            oxygen_conc = self.params.oxygen_concentration
        if ph is None:
            ph = self.params.ph
        
        # Standard potential for O2/H2O at pH=0 and 25°C
        E0_O2_H2O_298K = 1.229  # V vs SHE at 298.15 K
        
        # Temperature dependency of standard potential (literature: ~-0.8 mV/K)
        # ΔE0/ΔT ≈ -8.5e-4 V/K for O2/H2O reaction
        dE0_dT = -8.5e-4  # V/K
        temp_correction = dE0_dT * (self.temperature_K - 298.15)
        E0_O2_H2O = E0_O2_H2O_298K + temp_correction
        
        # Nernst equation: E = E0(T) + (RT/nF) * ln(C_O2 * [H+]^4)
        # At pH != 0: E = E0(T) - (RT/F)*ln(10)*pH + (RT/4F)*ln(C_O2/C_O2_ref)
        
        RT_over_4F = self.params.gas_constant * self.temperature_K / (4.0 * self.params.faraday_constant)
        RT_over_F = self.params.gas_constant * self.temperature_K / self.params.faraday_constant
        ph_correction = -RT_over_F * jnp.log(10.0) * ph  # Temperature-dependent pH correction
        concentration_term = RT_over_4F * jnp.log(oxygen_conc / self.params.oxygen_reference_conc)
        
        equilibrium_potential = E0_O2_H2O + ph_correction + concentration_term
        
        return float(equilibrium_potential)
    
    def calculate_overpotential(self, cathode_potential: float, 
                              oxygen_conc: Optional[float] = None,
                              ph: Optional[float] = None) -> float:
        """
        Calculate cathode overpotential.
        
        Args:
            cathode_potential: Applied cathode potential in V
            oxygen_conc: Oxygen concentration in mol/L
            ph: Solution pH
        
        Returns:
            Overpotential in V (positive = overpotential for reduction)
        """
        E_eq = self.calculate_equilibrium_potential(oxygen_conc, ph)
        overpotential = E_eq - cathode_potential
        return overpotential
    
    @abstractmethod
    def calculate_current_density(self, overpotential: float, 
                                oxygen_conc: Optional[float] = None) -> float:
        """
        Calculate current density using Butler-Volmer kinetics.
        Must be implemented by specific cathode models.
        
        Args:
            overpotential: Cathode overpotential in V
            oxygen_conc: Oxygen concentration in mol/L
        
        Returns:
            Current density in A/m²
        """
        pass
    
    def calculate_current(self, overpotential: float, 
                         oxygen_conc: Optional[float] = None) -> float:
        """
        Calculate total cathode current.
        
        Args:
            overpotential: Cathode overpotential in V
            oxygen_conc: Oxygen concentration in mol/L
        
        Returns:
            Current in A
        """
        current_density = self.calculate_current_density(overpotential, oxygen_conc)
        return current_density * self.area_m2
    
    def calculate_power_loss(self, overpotential: float, 
                           oxygen_conc: Optional[float] = None) -> float:
        """
        Calculate power loss at the cathode.
        
        Args:
            overpotential: Cathode overpotential in V
            oxygen_conc: Oxygen concentration in mol/L
        
        Returns:
            Power loss in W (always positive)
        """
        current = self.calculate_current(overpotential, oxygen_conc)
        # Power loss = overpotential * current (both positive for cathodic overpotential)
        power_loss = abs(overpotential * current)
        return power_loss
    
    def calculate_oxygen_consumption_rate(self, overpotential: float,
                                        oxygen_conc: Optional[float] = None) -> float:
        """
        Calculate oxygen consumption rate based on current.
        
        Args:
            overpotential: Cathode overpotential in V
            oxygen_conc: Oxygen concentration in mol/L
        
        Returns:
            Oxygen consumption rate in mol/s
        """
        current = self.calculate_current(overpotential, oxygen_conc)
        # O2 + 4H+ + 4e- -> 2H2O, so 4 electrons per O2 molecule
        n_electrons = 4.0
        consumption_rate = current / (n_electrons * self.params.faraday_constant)
        return consumption_rate
    
    def update_temperature(self, new_temperature_K: float):
        """Update operating temperature and recalculate temperature-dependent parameters."""
        self.temperature_K = new_temperature_K
        self.params.temperature_K = new_temperature_K
        self._setup_kinetic_parameters()
    
    def update_area(self, new_area_m2: float):
        """Update cathode area."""
        self.area_m2 = new_area_m2
        self.params.area_m2 = new_area_m2
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and current parameters.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_type': self.__class__.__name__,
            'area_m2': self.area_m2,
            'area_cm2': self.area_m2 * 1e4,
            'temperature_K': self.temperature_K,
            'temperature_C': self.temperature_K - 273.15,
            'oxygen_concentration_mol_L': self.params.oxygen_concentration,
            'oxygen_concentration_mg_L': self.params.oxygen_concentration * 32.0 * 1000,  # Convert to mg/L
            'ph': self.params.ph,
            'equilibrium_potential_V': self.calculate_equilibrium_potential()
        }
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (f"{info['model_type']}("
                f"area={info['area_cm2']:.2f} cm², "
                f"T={info['temperature_C']:.1f}°C, "
                f"O2={info['oxygen_concentration_mg_L']:.1f} mg/L)")


class ButlerVolmerKinetics:
    """
    Utility class for Butler-Volmer kinetics calculations.
    Can be used by specific cathode models.
    """
    
    @staticmethod
    def calculate_current_density(exchange_current_density: float,
                                transfer_coefficient: float,
                                overpotential: float,
                                temperature_K: float,
                                concentration_ratio: float = 1.0) -> float:
        """
        Calculate current density using Butler-Volmer equation.
        
        Args:
            exchange_current_density: Exchange current density in A/m²
            transfer_coefficient: Charge transfer coefficient (typically 0.5)
            overpotential: Overpotential in V (positive for cathodic overpotential)
            temperature_K: Temperature in K
            concentration_ratio: C_O2/C_O2_ref
        
        Returns:
            Current density in A/m² (positive for cathodic current)
        """
        F = 96485.0  # Faraday constant
        R = 8.314    # Gas constant
        
        # Butler-Volmer equation for ORR (cathodic reaction)
        # For cathodic reaction: i = i0 * [C_O2/C_O2_ref] * [exp(alpha*F*eta/RT) - exp(-(1-alpha)*F*eta/RT)]
        # Sign convention: positive overpotential and positive current for cathodic reaction
        
        alpha_F_eta_RT = transfer_coefficient * F * overpotential / (R * temperature_K)
        one_minus_alpha_F_eta_RT = (1.0 - transfer_coefficient) * F * overpotential / (R * temperature_K)
        
        # For cathodic overpotential (eta > 0), we want positive current
        # Corrected signs for cathodic reaction
        current_density = (exchange_current_density * concentration_ratio * 
                         (jnp.exp(alpha_F_eta_RT) - jnp.exp(-one_minus_alpha_F_eta_RT)))
        
        return float(current_density)
    
    @staticmethod
    def calculate_tafel_current(exchange_current_density: float,
                              tafel_slope: float,
                              overpotential: float,
                              concentration_ratio: float = 1.0) -> float:
        """
        Calculate current density using Tafel equation (high overpotential approximation).
        
        Args:
            exchange_current_density: Exchange current density in A/m²
            tafel_slope: Tafel slope in V/decade
            overpotential: Overpotential in V
            concentration_ratio: C_O2/C_O2_ref
        
        Returns:
            Current density in A/m²
        """
        # Tafel equation: eta = a + b*log(i), so i = i0*10^(eta/b)
        # For cathodic reaction: i = i0 * C_ratio * 10^(eta/b)
        
        if overpotential <= 0:
            return 0.0
        
        decade_factor = overpotential / tafel_slope
        current_density = exchange_current_density * concentration_ratio * (10.0 ** decade_factor)
        
        return current_density